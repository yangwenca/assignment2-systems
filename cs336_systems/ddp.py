from copy import deepcopy
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

naive_ddp_benchmarking
========== DDP Benchmark ==========
total time: 375us
gradient overhead: 163us

ddp_flat_benchmarking
graident overhead reduces from 163us to 120us
"""


class Module(torch.nn.Module):
    def __init__(self):
        super(Module, self).__init__()
        self.linear = torch.nn.Linear(10, 5, bias=False)
        self.linear.weight.data = torch.arange(50, dtype=torch.float32).view(5, 10) / 50
        self.linear2 = torch.nn.Linear(5, 5, bias=False)
        self.linear2.weight.data = torch.arange(25, dtype=torch.float32).view(5, 5) / 25
    
    def forward(self, x):
        return self.linear2(self.linear(x))


ToyModel = Module()


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


def manual_allreduce_gradients(model, world_size):
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world_size
    synced = torch._utils._unflatten_dense_tensors(flat, grads)
    for g, s in zip(grads, synced):
        g.copy_(s)


def benchmark(rank, world_size, backend, check_weight, benchmark, gradient):
    setup(rank, world_size, backend)

    if backend == "nccl":
        device = torch.device(f"cuda:{rank}")
    else:  # gloo
        device = torch.device("cpu")

    ddp_model = deepcopy(ToyModel).to(device)
    loss_fn = torch.nn.MSELoss()

    ddp_optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.1)
    if benchmark:
        result = torch.zeros((3)).to(device)
        gather_list = [torch.zeros_like(result).to(device) for _ in range(world_size)]
    for i in range(3):
        all_x = torch.randn(20, 10).to(device)
        all_y = torch.rand(20, 5).to(device)
        if check_weight:
            for group in ddp_model.parameters():
                print(f"rank {rank} iter {i} before param: {group}")
        ddp_optimizer.zero_grad()
        if benchmark:
            torch.cuda.synchronize() if backend == "nccl" else None
            start = time.time()
        outputs = ddp_model(all_x)
        loss = loss_fn(outputs, all_y)
        loss.backward()
        if gradient:
            manual_allreduce_gradients(ddp_model, world_size)
        else:
            for p in ddp_model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, async_op=False)
                    p.grad /= world_size
        ddp_optimizer.step()
        if check_weight:
            for group in ddp_model.parameters():
                print(f"rank {rank} iter {i} after param: {group}")
        if benchmark:
            torch.cuda.synchronize() if backend == "nccl" else None
            end = time.time()
            result[i] = end - start
            dist.all_gather(gather_list, result)
            result = torch.mean(torch.stack(gather_list), dim=0)
            print(f"Rank {rank} Iteration {i} Time: {(end - start) * 1000:.3f} ms, {result}")
    cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, choices=["nccl", "gloo"], default="gloo")
    parser.add_argument("--world_size", type=int, default=None)
    parser.add_argument("--check_weight", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--gradient", action="store_true")
    args = parser.parse_args()

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
            args.check_weight,
            args.benchmark,
            args.gradient,
        ),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    main()
