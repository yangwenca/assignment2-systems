import os
import time
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


"""
distributed_communication_single_node
========== All-Reduce Benchmark ==========
Backend     : gloo
World Size  : 4
Tensor Size : 4.00 MB
DType       : torch.float32
Avg Time    : 3.087 ms
"""


def setup(rank, world_size, backend):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    if backend == "nccl":
        assert torch.cuda.is_available(), "NCCL backend requires CUDA"
        torch.cuda.set_device(rank)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )


def cleanup():
    dist.destroy_process_group()


def benchmark(rank, world_size, backend, tensor_size, iters, warmup, dtype):
    setup(rank, world_size, backend)

    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
    else:  # gloo
        device = torch.device("cpu")

    tensor = torch.ones(tensor_size, device=device, dtype=dtype)

    # Warm-up
    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if backend == "nccl":
        torch.cuda.synchronize()

    # Benchmark
    start = time.time()
    for _ in range(iters):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    if backend == "nccl":
        torch.cuda.synchronize()

    end = time.time()
    avg_time_ms = (end - start) * 1000 / iters

    if rank == 0:
        size_mb = tensor.numel() * tensor.element_size() / (1024 ** 2)
        print("========== All-Reduce Benchmark ==========")
        print(f"Backend     : {backend}")
        print(f"World Size  : {world_size}")
        print(f"Tensor Size : {size_mb:.2f} MB")
        print(f"DType       : {dtype}")
        print(f"Avg Time    : {avg_time_ms:.3f} ms")

    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["nccl", "gloo"], default="nccl")
    parser.add_argument("--world-size", type=int, default=None)
    parser.add_argument("--tensor-size", type=int, default=1024 * 1024)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--dtype", type=str, default="float32")
    args = parser.parse_args()

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if args.backend == "nccl":
        assert torch.cuda.is_available(), "CUDA not available"
        max_procs = torch.cuda.device_count()
    else:
        max_procs = os.cpu_count()

    world_size = args.world_size or max_procs
    world_size = min(world_size, max_procs)

    mp.spawn(
        benchmark,
        args=(
            world_size,
            args.backend,
            args.tensor_size,
            args.iters,
            args.warmup,
            dtype_map[args.dtype],
        ),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
