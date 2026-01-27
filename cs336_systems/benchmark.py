import argparse
import timeit

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


"""
benchmark script
(a) this is the script
(b)
forward mean is 0.019560, std is 0.000172
backward mean is 0.020665, std is 0.000319
low variability
(c)
without warmup
forward mean is 0.044841, std is 0.103478
backward mean is 0.027186, std is 0.021972
high variability
at the beginning, need to initialize the machine

nsys_profile
(a)
0.03s, slower than python standard library
(b)
same kernel for backward and forward
forward
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
16.5%	10.969 ms	3504	3.130 μs	3.168 μs	2.688 μs	3.680 μs	136 ns	void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
forward + backward:
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
10.8%	21.622 ms	7032	3.074 μs	2.976 μs	2.432 μs	36.032 μs	1.884 μs	void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3)
(c)
elementwise_kernel, reduce_kernel, index_elementwise_kernel
(d)
matrix multiplication fraction reduces
more time on elementwise_kernel
(e)
Difference in runtimes between softmax and matrix multiplication is
much slower than difference in FLOPs
"""

"""
memory profiling
(a)
yes, can see stage
only forward peak is even larger, because next iteration starts, previous iteration does not release memory
due to preparation for back propagation
(b)
full training: 900MB
only forward: 1.5GB
(c)
yes, mixed precision reduces memory usage
(d)
2 * parameters * d_model
(e)
largest allocation is around 100MB
sdpa, softmax
"""

"""
pytorch attention
(a)
never run out of memory
backward memory for attention grows quadratically with sequence length
eliminate this cost by not storing the attention matrix and recomputing it in backward

torch compile
(a)
reduce time and memory usage
(b) forward, backward, optimizer reduces after using torch.compile
"""

def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main(args):
    device = torch.device(args.device)

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.theta,
    ).to(device)
    
    optimizer = AdamW(model.parameters())
    if args.speed:
        forward_time = torch.zeros((args.execute_iter), dtype=torch.float32, device=device)
        backward_time = torch.zeros((args.execute_iter), dtype=torch.float32, device=device)
    if args.memory:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    for i in range(args.warmup_iter + args.execute_iter):
        cur = i - args.warmup_iter
        inputs = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(args.batch_size, args.context_length),
            device=device,
            dtype=torch.int64,
        )
        target = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(args.batch_size, args.context_length),
            device=device,
            dtype=torch.int64,
        )
        optimizer.zero_grad(set_to_none=True)
        if i >= args.warmup_iter and args.speed:
            sync(device)
            start = timeit.default_timer()
        outputs = model(inputs)
        loss = cross_entropy(outputs, target)
        if i >= args.warmup_iter and args.speed:
            sync(device)
            diff = timeit.default_timer() - start
            forward_time[cur] = diff
        loss.backward()
        if i >= args.warmup_iter and args.speed:
            sync(device)
            diff = timeit.default_timer() - start - forward_time[cur]
            backward_time[cur] = diff
            print(
                f"iter is {(i - args.warmup_iter):4d} |"
                f"forward time is {forward_time[cur]:.4f} |"
                f"backward time is {backward_time[cur]:.4f}"
            )
        optimizer.step()
        if i >= args.warmup_iter and args.speed:
            sync(device)
    if args.speed:
        print(f"forward mean is {forward_time.mean():4f}, std is {forward_time.std():6f}")
        print(f"backward mean is {backward_time.mean():4f}, std is {backward_time.std():6f}")
    if args.memory:
        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda")

    # Model
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--theta", type=float, default=10000)

    # iteration
    parser.add_argument("--warmup_iter", type=int, default=4)
    parser.add_argument("--execute_iter", type=int, default=20)

    parser.add_argument("--memory", action="store_true", default=False)
    parser.add_argument("--speed", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
