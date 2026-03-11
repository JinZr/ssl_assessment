from __future__ import annotations

import os

import torch.distributed as dist


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def init_distributed() -> tuple[int, int, int]:
    if not is_distributed():
        return 0, 1, 0
    if not dist.is_initialized():
        dist.init_process_group(backend=os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    return rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0

