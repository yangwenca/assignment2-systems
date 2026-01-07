from einops import einsum
import torch
import triton
import triton.language as tl


"""
flash benchmarking
(a)
triton forward is faster than regular pytorch
triton backward is slower than regular pytorch
"""


class FlashPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        assert K.shape == V.shape
        assert K.shape[-1] == Q.shape[-1]
        v_shape = V.shape
        hidden = v_shape[-1]
        v_batch = v_shape.numel() // hidden
        q_shape = Q.shape
        output_l = torch.empty(q_shape[:-1], device=Q.device)
        output_o = torch.empty(q_shape, device=Q.device)
        q_batch = output_l.numel()
        batch_size = V.shape[0]
        q_tile = 16
        kv_tile = 32
        q_cnt = q_batch // q_tile // batch_size
        kv_cnt = v_batch // kv_tile // batch_size
        for batch_idx in range(batch_size):
            for i in range(q_cnt):
                q_i = Q[batch_idx, i*q_tile:(i+1)*q_tile]
                output_l[batch_idx, i*q_tile:(i+1)*q_tile] = 0
                output_o[batch_idx, i*q_tile:(i+1)*q_tile] = 0
                tmp_max = output_l[batch_idx, i*q_tile:(i+1)*q_tile] + float('-inf')
                for j in range(kv_cnt):
                    k_j = K[batch_idx, j*kv_tile:(j+1)*kv_tile]
                    v_j = V[batch_idx, j*kv_tile:(j+1)*kv_tile]
                    softmax = q_i @ k_j.T / (hidden ** 0.5)
                    if is_causal:
                        softmax = torch.where(
                            torch.arange(i*q_tile, (i+1)*q_tile)[:, None] >= \
                                torch.arange(j*kv_tile,(j+1)*kv_tile)[None, :],
                            softmax,
                            -float('inf'),
                        )
                    pre_tmp_max = tmp_max.clone()
                    tmp_max = torch.max(tmp_max, torch.amax(softmax, dim=-1))
                    p = torch.exp(softmax - tmp_max[:, None])
                    tmp = torch.exp(pre_tmp_max - tmp_max)
                    output_l[batch_idx, i*q_tile:(i+1)*q_tile] = tmp * output_l[batch_idx, i*q_tile:(i+1)*q_tile] \
                        + torch.sum(p, dim=-1)
                    output_o[batch_idx, i*q_tile:(i+1)*q_tile] = \
                        torch.diag(tmp) @ output_o[batch_idx, i*q_tile:(i+1)*q_tile] + p @ v_j
                output_o[batch_idx, i*q_tile:(i+1)*q_tile] = \
                    torch.diag(1.0 / output_l[batch_idx, i*q_tile:(i+1)*q_tile]) @ output_o[batch_idx, i*q_tile:(i+1)*q_tile]
                output_l[batch_idx, i*q_tile:(i+1)*q_tile] = \
                    torch.log(output_l[batch_idx, i*q_tile:(i+1)*q_tile]) + tmp_max
        ctx.save_for_backward(output_l, Q, K, V, output_o)
        ctx.is_causal = is_causal
        return output_o


    @staticmethod
    def backward(ctx, grad_out):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        hidden = Q.size(-1)
        scale = 1 / (hidden ** 0.5)
        S = einsum(Q, K, '... q d, ... k d -> ... q k') * scale
        if is_causal:
            S = torch.where(
                torch.arange(0, Q.size(-2))[None, :, None] >= torch.arange(0, K.size(-2))[None, None, :],
                S,
                -float('inf'),
            )
        P = torch.exp(S - L[..., None])
        dV = einsum(P, grad_out, '... q k, ... q d -> ... k d')
        dP = einsum(grad_out, V, '... q d, ... k d -> ... q k')
        D = torch.sum(O * grad_out, dim=-1, keepdims=True)
        dS = P * (dP - D)
        dQ = einsum(dS, K, '... q k, ... k d -> ... q d') * scale
        dK = einsum(dS, Q, '... q k, ... q d -> ... k d') * scale
        return dQ, dK, dV, None


@triton.jit
def flash_attn_fwd(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    pre_max_tmp = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    o_tmp = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    l_tmp = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero").to(tl.float32)
    q_tmp = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    off_m = tl.arange(0, Q_TILE_SIZE)
    off_n = tl.arange(0, Q_TILE_SIZE)
    is_diag = (off_m[:, None] == off_n[None, :])
    for i in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        k_tmp = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tmp = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        softmax = tl.dot(q_tmp, tl.trans(k_tmp, 1, 0)) * scale
        if is_causal:
            softmax = tl.where(
                (query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE))[:, None] >= \
                    (i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE))[None, :],
                softmax,
                -float("inf"),
            )
        max_tmp = tl.maximum(pre_max_tmp, tl.max(softmax, axis=-1))
        p = tl.exp(softmax - max_tmp[:, None]).to(V_block_ptr.type.element_ty)
        tmp = tl.exp(pre_max_tmp - max_tmp)
        l_tmp = tmp * l_tmp + tl.sum(p, axis=-1)
        diagonal = tl.where(is_diag, tmp, 0.0)
        o_tmp = tl.dot(diagonal, o_tmp) + tl.dot(p, v_tmp)
        pre_max_tmp = max_tmp

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    diagonal = tl.where(is_diag, 1 / l_tmp, 0.0)
    o_tmp = tl.dot(diagonal, o_tmp).to(O_block_ptr.type.element_ty)
    tl.store(O_block_ptr, o_tmp, boundary_check=(0, 1))
    l_tmp = (tl.log(l_tmp) + pre_max_tmp).to(L_block_ptr.type.element_ty)
    tl.store(L_block_ptr, l_tmp, boundary_check=(0,))


@triton.jit
def flash_attn_bwd(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr, D_ptr,
    dQ_ptr, dK_ptr, dV_ptr, dO_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    stride_db, stride_dq,
    stride_dqb, stride_dqq, stride_dqd,
    stride_dkb, stride_dkk, stride_dkd,
    stride_dvb, stride_dvk, stride_dvd,
    stride_dob, stride_doq, stride_dod,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # Program indices
    key_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(key_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    k_tmp = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
    v_tmp = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dk_tmp = tl.load(dK_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dv_tmp = tl.load(dV_block_ptr, boundary_check=(0, 1), padding_option="zero")

    offset_m = tl.arange(0, Q_TILE_SIZE)[:, None]
    offset_n = tl.arange(0, D)[None, :]
    for i in range(tl.cdiv(N_QUERIES, Q_TILE_SIZE)):
        q_tmp = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        o_tmp = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
        do_tmp = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        l_tmp = tl.load(L_block_ptr, boundary_check=(0,), padding_option="zero")
        d_tmp = tl.load(D_block_ptr, boundary_check=(0,), padding_option="zero")

        softmax = tl.dot(q_tmp, tl.trans(k_tmp, 1, 0)) * scale
        if is_causal:
            softmax = tl.where(
                offset_m >= (key_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE))[None, :],
                softmax,
                -float("inf"),
            )
        p = tl.exp(softmax - l_tmp[:, None])
        dv_tmp += tl.dot(tl.trans(p, 1, 0), do_tmp)
        dp_tmp = tl.dot(do_tmp, tl.trans(v_tmp, 1, 0))
        ds = p * (dp_tmp - d_tmp[:, None]) * scale
        mask = (offset_m < N_QUERIES) & (offset_n < D)
        ds_k_tmp = tl.dot(ds, k_tmp)
        tl.atomic_add(dQ_ptr + batch_index * stride_qb + offset_m * stride_qq + offset_n * stride_qd, \
            ds_k_tmp, mask=mask)
        offset_m += Q_TILE_SIZE
        dk_tmp += tl.dot(tl.trans(ds, 1, 0), q_tmp)

        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
        D_block_ptr = D_block_ptr.advance((Q_TILE_SIZE,))

    tl.store(dK_block_ptr, dk_tmp, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dv_tmp, boundary_check=(0,))


@triton.jit
def preprocess(
    O_ptr, dO_ptr, D_ptr,
    stride_ob, stride_oq, stride_od,
    stride_dob, stride_doq, stride_dod,
    stride_db, stride_dq,
    N_QUERIES,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
):
    # Program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # Offset each pointer with the corresponding batch index
    # multiplied with the batch stride for each tensor
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    D_block_ptr = tl.make_block_ptr(
        D_ptr + batch_index * stride_db,
        shape=(N_QUERIES,),
        strides=(stride_dq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    o_tmp = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    do_tmp = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    ans = tl.sum(o_tmp * do_tmp, axis=-1)
    tl.store(D_block_ptr, ans, boundary_check=(0,))


class FlashTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        assert Q.is_cuda and K.is_cuda and V.is_cuda
        assert Q.is_contiguous() and K.is_contiguous() and V.is_contiguous()
        assert K.shape == V.shape
        assert K.shape[-1] == Q.shape[-1]
        assert Q.dtype == K.dtype == V.dtype
        dtype = Q.dtype
        q_shape = Q.shape
        hidden = K.shape[-1]
        batch_size = V.shape[0]
        output_l = torch.zeros(q_shape[:-1], device=Q.device, dtype=dtype)
        output_o = torch.zeros(q_shape, device=Q.device, dtype=dtype)
        N_keys = K.numel() // hidden // batch_size
        N_queries = Q.numel() // hidden // batch_size
        ctx.Q_TILE_SIZE = 16
        ctx.K_TILE_SIZE = 32
        ctx.is_causal = is_causal
        flash_attn_fwd[(triton.cdiv(N_queries, ctx.Q_TILE_SIZE), batch_size)](
            Q, K, V, output_o, output_l,
            Q.stride(0), hidden, Q.stride(-1),
            K.stride(0), hidden, K.stride(-1),
            V.stride(0), hidden, V.stride(-1),
            Q.stride(0), hidden, Q.stride(-1),
            output_l.stride(0), output_l.stride(-1),
            N_QUERIES=N_queries, N_KEYS=N_keys,
            scale=1 / (hidden ** 0.5),
            D=hidden,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=is_causal,
        )
        ctx.save_for_backward(output_l, Q, K, V, output_o)
        return output_o


    @staticmethod
    def backward(ctx, grad_out):
        L, Q, K, V, O = ctx.saved_tensors
        is_causal = ctx.is_causal
        D = torch.zeros_like(O[:,:,0])
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        dQ = torch.zeros_like(Q)
        hidden = Q.size(-1)
        scale = 1 / (hidden ** 0.5)
        batch_size = Q.size(0)
        N_keys = K.numel() // hidden // batch_size
        N_queries = Q.numel() // hidden // batch_size
        preprocess[(triton.cdiv(N_queries, ctx.Q_TILE_SIZE), batch_size)](
            O, grad_out, D,
            Q.stride(0), hidden, Q.stride(-1),
            Q.stride(0), hidden, Q.stride(-1),
            D.stride(0), D.stride(-1),
            N_QUERIES=N_queries,
            D=hidden,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
        )

        flash_attn_bwd[(triton.cdiv(N_keys, ctx.K_TILE_SIZE), batch_size)](
            Q, K, V, O, L, D, dQ, dK, dV, grad_out,
            Q.stride(0), hidden, Q.stride(-1),
            K.stride(0), hidden, K.stride(-1),
            V.stride(0), hidden, V.stride(-1),
            Q.stride(0), hidden, Q.stride(-1),
            L.stride(0), L.stride(-1),
            D.stride(0), D.stride(-1),
            Q.stride(0), hidden, Q.stride(-1),
            K.stride(0), hidden, K.stride(-1),
            V.stride(0), hidden, V.stride(-1),
            Q.stride(0), hidden, Q.stride(-1),
            N_QUERIES=N_queries, N_KEYS=N_keys,
            scale=1 / (hidden ** 0.5),
            D=hidden,
            Q_TILE_SIZE=ctx.Q_TILE_SIZE,
            K_TILE_SIZE=ctx.K_TILE_SIZE,
            is_causal=is_causal,
        )
        return dQ, dK, dV, None
