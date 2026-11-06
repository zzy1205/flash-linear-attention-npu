def npu_fused_recurrent_rwkv8(
    q,
    w,
    k,
    v,
    z,
    b,
    *,
    scale=1.0,
    initial_state=None,
    reverse=False,
    output_chunk_state=False,
    output_sa=False,
    chunk_len=16,
):
    """RWKV-v8（WKV7）fused recurrent 前向，逐 token 递推。

    布局为 BHTC：q/w/k/z/b 是 (B,H,T,K)，v 是 (B,H,T,V)（K 与 V 独立，
    K==V 为特例）。与 fla 上游的 (B,T,H,C) 口径不同，对接时需
    transpose(1,2)——transpose 只改视图，建议随后 .contiguous()，否则
    aclnn 层会对每个非连续输入静默补一次 GM 拷贝。initial_state 朝向
    (B,H,K,V)（= 内核 Sᵀ 原样，与 s 快照一致），fp32，缺省零初态。

    返回 (o, s, sa)：o 为 (B,H,T,V)、dtype 跟随 q；s 为 chunk 快照
    (B,H,T//chunk_len,K,V) fp32（output_chunk_state=True 时产出，
    否则为 None）；sa 为每 token 更新前的 state@z，(B,H,T,V) fp32
    （output_sa=True 时产出，否则为 None）。
    """
    import torch

    allowed_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    tensors = {"q": q, "w": w, "k": k, "v": v, "z": z, "b": b}
    shapes = {name: _shape(tensor) for name, tensor in tensors.items()}
    for name, tensor in tensors.items():
        if len(shapes[name]) != 4:
            raise RuntimeError(f"npu_fused_recurrent_rwkv8: {name} must be rank-4 (B,H,T,C).")
        if tensor.dtype not in allowed_dtypes:
            raise RuntimeError(f"npu_fused_recurrent_rwkv8: {name} dtype must be float16/bfloat16/float32.")
        if tensor.dtype != q.dtype:
            raise RuntimeError("npu_fused_recurrent_rwkv8: q/w/k/v/z/b must share the same dtype.")
    q_shape = shapes["q"]
    for name in ("w", "k", "z", "b"):
        if shapes[name] != q_shape:
            raise RuntimeError(f"npu_fused_recurrent_rwkv8: {name} shape must equal q shape (B,H,T,K).")
    batch, heads, seqlen, k_dim = q_shape
    v_shape = shapes["v"]
    if v_shape[:3] != (batch, heads, seqlen):
        raise RuntimeError("npu_fused_recurrent_rwkv8: v must be (B,H,T,V) with the same B/H/T as q.")
    v_dim = v_shape[3]
    if k_dim % 8 or k_dim > 128 or v_dim % 8 or v_dim > 128:
        raise RuntimeError("npu_fused_recurrent_rwkv8: K/V must be multiples of 8 and <= 128.")

    reverse = _optional_bool(reverse, False)
    output_chunk_state = _optional_bool(output_chunk_state, False)
    output_sa = _optional_bool(output_sa, False)
    chunk_len = int(chunk_len)
    if chunk_len < 1:
        raise RuntimeError("npu_fused_recurrent_rwkv8: chunk_len must be >= 1.")

    if initial_state is not None:
        if _shape(initial_state) != (batch, heads, k_dim, v_dim):
            raise RuntimeError("npu_fused_recurrent_rwkv8: initial_state must be (B,H,K,V).")
        if initial_state.dtype != torch.float32:
            raise RuntimeError("npu_fused_recurrent_rwkv8: initial_state must be float32.")

    out = _empty((batch, heads, seqlen, v_dim), q)
    # T < chunk_len 时快照数 NT=0：aclnn 要求 sOut 真实 shape (B,H,0,K,V)，
    # 占位只能落在 storage 层——给底层多分配 1 行（避免 0 字节分配的空指针），
    # 传 aclnn 与返回的都是 [:, :, :NT] 视图（手法同 C++ example 的 buffer 占位）。
    num_chunks = seqlen // chunk_len
    s = None
    if output_chunk_state:
        s = _empty((batch, heads, max(num_chunks, 1), k_dim, v_dim), q, dtype=torch.float32)[:, :, :num_chunks]
    sa = _empty((batch, heads, seqlen, v_dim), q, dtype=torch.float32) if output_sa else None

    outputs = tuple(tensor for tensor in (out, s, sa) if tensor is not None)
    _call_aclnn(
        "aclnnFusedRecurrentRwkv8",
        lambda ctx: [
            ctx.tensor(q, "q"),
            ctx.tensor(w, "w"),
            ctx.tensor(k, "k"),
            ctx.tensor(v, "v"),
            ctx.tensor(z, "z"),
            ctx.tensor(b, "b"),
            ctx.tensor(initial_state, "initial_state"),
            ctypes.c_float(float(scale)),
            ctypes.c_bool(reverse),
            ctypes.c_bool(output_chunk_state),
            ctypes.c_bool(output_sa),
            ctypes.c_int64(chunk_len),
            ctx.tensor(out, "out"),
            ctx.tensor(s, "s"),
            ctx.tensor(sa, "sa"),
        ],
        outputs,
    )
    return out, s, sa
