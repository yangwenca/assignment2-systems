import torch
import torch.distributed as dist

from typing import Any, Callable, Dict, Optional, Type


"""
optimizer_state_sharding_accounting
(a)
peak memory usage after model initialization is the same
before optimizer step reduces
after optimizer step reduces
model parameters and activations are not sharded. optimizer states are sharded
(b)
it slows down the training a little bit
introduces an extra broadcast of parameters after each optimizer step
(c)
Unlike ZeRO stage-1, which keeps parameters synchronized implicitly via replicated gradients and deterministic updates,
your approach explicitly broadcasts updated parameters each step,
increasing communication volume without reducing memory further.

ZeRO-1 step:
- Gradients are already all-reduced (DDP)
- Every rank updates ALL parameters
- Optimizer state (m, v) exists only on owner ranks
All ranks have identical gradients (via DDP all-reduce)

Owner rank:
Updates optimizer state (m, v)
Computes update Δp
Owner rank broadcasts Δp
All ranks apply p += Δp
"""


class Shard(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: Type[torch.optim.Optimizer],
        **kwargs: Any,
    ):
        self.world_size = dist.get_world_size()

        # Will be created after param groups are known
        self._param_to_rank: Dict[torch.nn.Parameter, int] = {}

        super().__init__(params, kwargs)

        # Create local optimizer using only local params
        local_param_groups = []
        for group in self.param_groups:
            local_params = [
                p for p in group["params"]
                if p.requires_grad and (self._param_to_rank[p] == dist.get_rank())
            ]
            if local_params:
                g = dict(group)
                g["params"] = local_params
                local_param_groups.append(g)

        self.local_optimizer = optimizer_cls(
            local_param_groups, **kwargs
        )


    def add_param_group(self, param_group: Dict[str, Any]):
        # Assign parameters to ranks (round-robin)
        for p in param_group["params"]:
            if (not p.requires_grad) or (p in self._param_to_rank):
                continue
            idx = len(self._param_to_rank)
            self._param_to_rank[p] = idx % self.world_size

        super().add_param_group(param_group)


    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        loss = None if closure is None else closure()

        # Step only local shard
        self.local_optimizer.step(closure, **kwargs)

        # Synchronize parameters across ranks
        for p, owner in self._param_to_rank.items():
            dist.broadcast(p.data, src=owner)
        return loss
