#include <metal_stdlib>

using namespace metal;

#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

inline ulong chroma_index(long index, ulong extent, bool negative, device atomic_uint* error) {
    if (negative && index < 0) {
        index += long(extent);
    }
    if (index < 0 || ulong(index) >= extent) {
        atomic_store_explicit(error, 1u, memory_order_relaxed);
        return 0;
    }
    return ulong(index);
}

struct ChromaLinearArgs {
    uint rows, columns, inner, bias;
};

kernel void chroma_f32_linear_tiled(constant ChromaLinearArgs& args [[buffer(0)]], device const float* x [[buffer(1)]],
                                    device const float* weight [[buffer(2)]], device const float* bias [[buffer(3)]],
                                    device float* output [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float a[8 * 16];
    threadgroup float b[32 * 16];
    const uint row = group.y * 8 + tid / 32;
    const uint column = group.x * 32 + tid % 32;
    float sum = 0.0f;
    for (uint offset = 0; offset < args.inner; offset += 16) {
        for (uint i = tid; i < 8 * 16; i += 256) {
            const uint r = group.y * 8 + i / 16, k = offset + i % 16;
            a[i] = r < args.rows && k < args.inner ? x[ulong(r) * args.inner + k] : 0.0f;
        }
        for (uint i = tid; i < 32 * 16; i += 256) {
            const uint c = group.x * 32 + i / 16, k = offset + i % 16;
            b[i] = c < args.columns && k < args.inner ? weight[ulong(c) * args.inner + k] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < 16; ++k) {
            sum += a[tid / 32 * 16 + k] * b[tid % 32 * 16 + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < args.rows && column < args.columns) {
        output[ulong(row) * args.columns + column] = sum + (args.bias ? bias[column] : 0.0f);
    }
}

// chroma_f32_bf16w_matmul and chroma_f32_bf16w_rmsnorm are adapted from gpt-oss
// (https://github.com/openai/gpt-oss, Apache-2.0).

struct ChromaMatmulArgs {
    uint num_column_vecs;
    uint num_rows;
    uint has_bias;
};

kernel void chroma_f32_bf16w_matmul(constant ChromaMatmulArgs& args [[buffer(0)]],
                                    const device float4* input [[buffer(1)]],
                                    const device bfloat4* weight [[buffer(2)]], const device bfloat* bias [[buffer(3)]],
                                    device float* output [[buffer(4)]], const device uint* error [[buffer(5)]],
                                    uint2 gid [[threadgroup_position_in_grid]],
                                    uint simdgroup_tid [[thread_index_in_simdgroup]],
                                    uint simdgroup_idx [[simdgroup_index_in_threadgroup]],
                                    uint num_simdgroups [[simdgroups_per_threadgroup]]) {
    const uint simdgroup_size = 32;
    if (*error != 0) {
        return;
    }

    const uint num_column_vecs = args.num_column_vecs;
    const uint row = gid.x * num_simdgroups + simdgroup_idx;

    input += gid.y * num_column_vecs + simdgroup_tid;
    weight += num_column_vecs * row + simdgroup_tid;
    bias += row;
    output += gid.y * args.num_rows + row;

    uint num_iter = (num_column_vecs - simdgroup_tid + (simdgroup_size - 1)) / simdgroup_size;

    float4 sum4 = 0.0f;
    do {
        const bfloat4 w = *weight;
        const float4 i = *input;
        sum4 = metal::fma(static_cast<float4>(w), i, sum4);

        weight += simdgroup_size;
        input += simdgroup_size;
    } while (--num_iter != 0);
    const float2 sum2 = sum4.xy + sum4.zw;
    float sum = sum2.x + sum2.y;
    sum = metal::simd_sum(sum);
    if (metal::simd_is_first()) {
        *output = sum + (args.has_bias ? static_cast<float>(*bias) : 0.0f);
    }
}

struct ChromaRMSNormArgs {
    uint num_vecs;
    float num_channels;
    float epsilon;
};

[[max_total_threads_per_threadgroup(1024)]]
kernel void chroma_f32_bf16w_rmsnorm(constant ChromaRMSNormArgs& args [[buffer(0)]],
                                     const device float4* input [[buffer(1)]],
                                     const device bfloat4* weights [[buffer(2)]], device float4* output [[buffer(3)]],
                                     const device uint* error [[buffer(4)]], uint gid [[threadgroup_position_in_grid]],
                                     uint tid [[thread_position_in_threadgroup]],
                                     uint threadgroup_size [[threads_per_threadgroup]]) {
    const uint simdgroup_size = 32;
    threadgroup float threadgroup_buffer[32];
    if (*error != 0) {
        return;
    }

    input += gid * args.num_vecs;
    output += gid * args.num_vecs;

    float4 sumsq4 = 0.0f;
    for (uint i = tid; i < args.num_vecs; i += threadgroup_size) {
        const float4 val = input[i];
        sumsq4 = metal::fma(val, val, sumsq4);
    }

    const float2 sumsq2 = sumsq4.xy + sumsq4.zw;
    float sumsq = sumsq2.x + sumsq2.y;
    sumsq = metal::simd_sum(sumsq);
    if (metal::simd_is_first()) {
        const uint simdgroup_idx = tid / simdgroup_size;
        threadgroup_buffer[simdgroup_idx] = sumsq;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const uint simdgroup_tid = tid % simdgroup_size;
    sumsq = threadgroup_buffer[simdgroup_tid];
    sumsq = metal::simd_sum(sumsq);

    const float avgsq = sumsq / args.num_channels;
    const float scale = metal::precise::rsqrt(avgsq + args.epsilon);
    for (uint i = tid; i < args.num_vecs; i += threadgroup_size) {
        const float4 val = input[i] * scale;
        const float4 weight_val = static_cast<float4>(weights[i]);
        output[i] = val * weight_val;
    }
}
