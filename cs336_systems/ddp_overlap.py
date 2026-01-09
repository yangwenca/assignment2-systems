import torch
import torch.distributed as dist


"""
ddp_overlap_individual_parameters_benchmarking
(a)
in the tory model, the graident overhead reduces from 163us to 120us to 100us
this means overlapping communication with computation helps further reduce the graident overhead
(b)
nsys profiler output shows that the all-reduce operations are overlapped with the backward computation
"""


class DDPOverlap(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super(DDPOverlap, self).__init__()
        self.module = module

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.transform_grad)

        # Store async handles for gradient all-reduce (if needed)
        self.handles = []


    def transform_grad(self, param):
        with torch.no_grad():
            param.grad.data /= dist.get_world_size()

        self.handles.append(dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True))


    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()


"""
ddp_bucketed_benchmark
(a)
the bucket size is too small, lots of communication
the bucket size is too large, communcation can't overlap with computation

(b)
bucket size: b = s/n_b
communication time per bucket: b/w + o
communication for n_b-1 bucket is overlapped with computation
last bucket do not overlap
per bucket overhead is not overlapped.
total cost is b/w + n_b * o = s/(n_b * w) + n_b * o
find the minimal of the above equation
n_b = sqrt(s / (w * o))
"""


class DDPOverlapBucketed(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.target = bucket_size_mb * 1024 * 1024

        self.buckets = []
        self.param_to_bucket = {} 
        self.bucket_handles = {}
        self.bucket_ready_params = {}

        current_bucket = []
        current_bucket_size = 0.0
        bucket_idx = 0

        for param in list(self.module.parameters())[::-1]:
            # broadcast parameters
            dist.broadcast(param.data, src=0)
            if not param.requires_grad:
                continue
            
            # register hooks for gradient
            param.register_post_accumulate_grad_hook(self.transform_grad)

            param_size_mb = param.numel() * param.element_size()
            if current_bucket and current_bucket_size + param_size_mb > self.target:
                self.buckets.append(current_bucket)
                self.bucket_ready_params[bucket_idx] = set()
                bucket_idx += 1
                current_bucket = []
                current_bucket_size = 0.0

            current_bucket.append(param)
            current_bucket_size += param_size_mb
            self.param_to_bucket[param] = bucket_idx

        if current_bucket:
            self.buckets.append(current_bucket)
            self.bucket_ready_params[bucket_idx] = set()


    def transform_grad(self, param: torch.Tensor):
        if param.grad is None:
            return

        bucket_idx = self.param_to_bucket[param]
        self.bucket_ready_params[bucket_idx].add(param)

        if len(self.bucket_ready_params[bucket_idx]) == len(self.buckets[bucket_idx]) \
            and bucket_idx not in self.bucket_handles:
            bucket_grads = [param.grad for param in self.buckets[bucket_idx] if param.grad is not None]

            if bucket_grads:
                flattened_grad = torch._utils._flatten_dense_tensors(bucket_grads)
                handle = dist.all_reduce(flattened_grad, async_op=True)
                self.bucket_handles[bucket_idx] = (handle, flattened_grad, bucket_grads)


    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()

        for handle, flattened_grad, original_grads in self.bucket_handles.values():
            handle.wait()
            flattened_grad /= world_size
            synced = torch._utils._unflatten_dense_tensors(flattened_grad, original_grads)
            for g, s in zip(original_grads, synced):
                g.copy_(s)

        self.bucket_handles.clear()
        for bucket_idx in self.bucket_ready_params:
            self.bucket_ready_params[bucket_idx].clear()


"""
communication accounting
(a)
weight, accumulated gradient, optimizer state(first momentum + second momentum):
2 * num_blocks * d_model * d_ff
= 2 * 126 * 16384 * 53248
= 220B
In Memory
220B * 4 * 4 = 3.52TB

backward memory saving for bf16 (gradient is bf16):
220B * 2 = 440GB save for backward
= 44.05GB (precision is bf16) save for half

3.52TB / 80 GB = 44 H100 80G GPUs

(b)
Parameter = 220B
fp32 master weights, gradient, adam (m, v) = 4 * 4 = 16bytes
backward bf 2 bytes per parameter

sharded
fp 32 state: fully shared 16P / N_fsdp
half of bf16 activation: 1/2 * 2P / N_fsdp = P / N_fsdp

unshard activation
num_blocks * (d_model + d_ff) = 8.7e6
8.7e6 * 2 bytes = 17.4MB
half of them is sharded
8.7MB

total memory
16P / N_fsdp + 8.7MB/N_fdsp + 8.7MB
16 * 200GB / N + 8.7MB <= 95GB. It is impossible
N = 34
(c)
flop per token
2 * (2 * d_model * d_ff) = 3.49e9 FLOPs per token
126 blocks, total 4.4e11 FLOPs per token
Y = 4, F = 4.4e11 /4 = 1.1e11 FLOPs
per device batch size
b * 1.1e11 /4.6e14 = b * 2.39e-4

bytes to receive for one device
(1 - 1/X)P * 2 = 15/16 * 2.2e11 * 2 = 4.1e11
communication time
4.1e11 / 1.8e11 = 2.28s

b * 2.39e-4 >= 2.28
b >= 9.5e3

XY = 64

global = 64 * 9.5e3 = 6e5
(d)
pipeline parallel, tensor parallel, activation checkpoints
"""
