from dataclasses import dataclass


@dataclass
class Allocation:
    size: int
    start: int
    end: int
    offset: int = 0


def plan_memory(allocations: dict[str, Allocation]) -> int:
    active = []
    free = []
    total = 0

    for allocation in sorted(allocations.values(), key=lambda a: a.start):
        retained = []

        for old in active:
            if old.end < allocation.start:
                free.append((old.offset, old.size))
            else:
                retained.append(old)

        active = retained
        merged = []

        for offset, size in sorted(free):
            if merged and sum(merged[-1]) == offset:
                previous, length = merged.pop()
                merged.append((previous, length + size))
            else:
                merged.append((offset, size))

        free = merged
        candidates = [(size, i) for i, (_, size) in enumerate(free) if size >= allocation.size]

        if candidates:
            _, index = min(candidates)
            offset, size = free.pop(index)
            allocation.offset = offset
            if size > allocation.size:
                free.append((offset + allocation.size, size - allocation.size))
        else:
            allocation.offset = total
            total += allocation.size
        active.append(allocation)

    return total
