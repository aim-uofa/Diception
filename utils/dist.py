import os
import torch
import torch.distributed as dist
import subprocess as sp
from datetime import timedelta

import random
import numpy as np
import json

from .logs import LOGGER

def get_world_size():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE'])
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE', '1'))


def get_rank():
    if 'RANK' in os.environ:
        return int(os.environ['RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_RANK', '0'))


def get_local_rank():
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', '0'))


def get_local_size():
    if 'LOCAL_SIZE' in os.environ:
        return int(os.environ['LOCAL_SIZE'])
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_SIZE', '1'))

def is_main_process():
    if get_rank()==0:
        return True
        # try:
        #     if torch.distributed.get_rank()==0:
        #         return True
        #     else:
        #         return False
        # except RuntimeError:
        #     return True
    else:
        return False
    
# Training-specific deepspeed functions removed for inference-only repo


def fp32_to_fp16(batch):
    # deepspeed does not auto cast inputs.
    if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
        return batch.to(dtype=torch.half)
    elif isinstance(batch, list):
        new_batch = [fp32_to_fp16(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(fp32_to_fp16(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: fp32_to_fp16(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch

def fp32_to_bf16(batch):
    # deepspeed does not auto cast inputs.
    if isinstance(batch, torch.Tensor) and batch.dtype == torch.float32:
        return batch.to(dtype=torch.bfloat16)
    elif isinstance(batch, list):
        new_batch = [fp32_to_bf16(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(fp32_to_bf16(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: fp32_to_bf16(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch

def move_to_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda(non_blocking=True)
    elif isinstance(batch, list):
        new_batch = [move_to_cuda(t) for t in batch]
    elif isinstance(batch, tuple):
        new_batch = tuple(move_to_cuda(t) for t in batch)
    elif isinstance(batch, dict):
        new_batch = {n: move_to_cuda(t) for n, t in batch.items()}
    else:
        return batch
    return new_batch


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = get_world_size()
    if world_size == 1:
        return
    t = torch.randn((), device='cuda')
    dist.all_reduce(t)
    torch.cuda.synchronize()
    return
    # dist.barrier()

# Training-specific distributed initialization and seed setting functions removed for inference-only repo