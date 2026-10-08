from chroma.models.gpt_oss import ModelConfig

dev = ModelConfig(
    num_hidden_layers=12,
    num_experts=12,
    experts_per_token=4,
    vocab_size=201088,
    hidden_size=128,
    intermediate_size=128,
    swiglu_limit=7.0,
    head_dim=16,
    num_attention_heads=8,
    num_key_value_heads=1,
    sliding_window=0,
    initial_context_length=128,
    rope_theta=150000.0,
    rope_scaling_factor=1.0,
    rope_ntk_alpha=1.0,
    rope_ntk_beta=32.0,
)
