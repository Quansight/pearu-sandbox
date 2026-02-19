# Got confirmation from Luca Wehrstedt that this is okay to share
# Luca Wehrstedt: we don't have any issues with that code being shared or becoming public (as long as it's just that file)

import ctypes
from functools import cache

import torch

# import torch.distributed._functional_collectives as funcol
# from torch.distributed.distributed_c10d import _resolve_process_group
# from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.utils.flop_counter import register_flop_formula

WEIGHT_BLOCK = 4096


@cache
def get_cublas_library() -> ctypes.CDLL:
    return ctypes.CDLL("libcublas.so.12")


CUBLAS_OP_N = ctypes.c_int(0)
CUBLAS_OP_T = ctypes.c_int(1)
CUBLAS_COMPUTE_32F = ctypes.c_int(68)
CUDA_R_16BF = ctypes.c_int(14)
CUDA_R_32F = ctypes.c_int(0)
CUBLAS_GEMM_DEFAULT = ctypes.c_int(-1)

float_dtype = torch.bfloat16
float_dtype = torch.float32
float32_dtype = torch.float32


def addmm_fp32(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Provide (add)mm with bf16 inputs but fp32 output

    PyTorch doesn't support out-dtypes being different than in-dtypes, even
    though the GPU's TensorCores always accumulate in fp32 for bf16 inputs.
    We need this because we'll do a manual split-k matmul (where each split is a
    separate kernel) and we must accumulate in fp32 to preserve accuracy.
    If we were to use a regular matmul and cast its inputs to fp32 this would
    end up being 2x as slow because the TensorCores would operate in tf32.

    """
    # PyTorch 2.8.0+ (or nightlies since April 18, 2025) provides this operator.
    # See https://github.com/pytorch/pytorch/pull/150812
    if pt_op := getattr(torch.ops.aten.addmm, "dtype_out", None):
        return pt_op(out, a, b, out_dtype=float32_dtype, out=out)

    torch._check(a.layout == torch.strided)
    torch._check(a.ndim == 2)
    torch._check(a.stride(0) == 1 or a.stride(1) == 1)
    torch._check(a.dtype == float_dtype)
    torch._check(a.is_cuda)

    torch._check(b.layout == torch.strided)
    torch._check(b.ndim == 2)
    torch._check(b.stride(0) == 1 or b.stride(1) == 1)
    torch._check(b.dtype == float_dtype)
    torch._check(b.is_cuda)

    torch._check(out.layout == torch.strided)
    torch._check(out.ndim == 2)
    torch._check(out.stride(0) == 1 or out.stride(1) == 1)
    torch._check(out.dtype == float32_dtype)
    torch._check(out.is_cuda)

    torch._check(a.device == b.device)
    torch._check(a.device == out.device)

    torch._check(a.size(1) == b.size(0))
    torch._check(out.size(0) == a.size(0))
    torch._check(out.size(1) == b.size(1))

    k = a.size(1)

    if out.stride(0) != 1:
        out = out.t()
        a, b = b.t(), a.t()

    transpose_a = False
    if a.stride(0) != 1:
        a = a.t()
        transpose_a = True

    transpose_b = False
    if b.stride(0) != 1:
        b = b.t()
        transpose_b = True

    handle = ctypes.c_void_p(torch.cuda.current_blas_handle())
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(1.0)

    res = get_cublas_library().cublasGemmEx(
        handle,
        CUBLAS_OP_T if transpose_a else CUBLAS_OP_N,
        CUBLAS_OP_T if transpose_b else CUBLAS_OP_N,
        ctypes.c_int(out.size(0)),
        ctypes.c_int(out.size(1)),
        ctypes.c_int(k),
        ctypes.byref(alpha),
        ctypes.c_void_p(a.data_ptr()),
        CUDA_R_16BF,
        ctypes.c_int(a.stride(1)),
        ctypes.c_void_p(b.data_ptr()),
        CUDA_R_16BF,
        ctypes.c_int(b.stride(1)),
        ctypes.byref(beta),
        ctypes.c_void_p(out.data_ptr()),
        CUDA_R_32F,
        ctypes.c_int(out.stride(1)),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT,
    )

    torch._check(res == 0)

    return out


@torch.compile(fullgraph=True)
def fwd_epilogue(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight_offset: int,
    max_logits: torch.Tensor,
    logsumexps: torch.Tensor,
    positive_logits: torch.Tensor,
) -> torch.Tensor:
    logits = logits.float()

    new_max_logits = logits.amax(dim=-1)
    new_max_logits = torch.maximum(new_max_logits, max_logits)

    shifted_logits = logits - new_max_logits[:, None]
    new_logsumexps = torch.log(
        torch.exp(shifted_logits).sum(dim=-1)
        + torch.exp(logsumexps + (max_logits - new_max_logits))
    )

    shifted_targets = targets - weight_offset
    targets_mask = torch.logical_and(
        0 <= shifted_targets, shifted_targets < shifted_logits.shape[1]
    )
    shifted_targets = shifted_targets.clamp(min=0, max=shifted_logits.shape[1] - 1)
    new_positive_logits = torch.where(
        targets_mask,
        shifted_logits.gather(1, shifted_targets.unsqueeze(1)).squeeze(1),
        positive_logits + (max_logits - new_max_logits),
    )

    max_logits.copy_(new_max_logits)
    logsumexps.copy_(new_logsumexps)
    positive_logits.copy_(new_positive_logits)


@torch.compile(fullgraph=True)
def bwd_epilogue(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight_offset: int,
    max_logits: torch.Tensor,
    logsumexps: torch.Tensor,
    grad_loss: torch.Tensor,
) -> torch.Tensor:
    orig_logits = logits
    logits = logits.float()

    shifted_logits = logits - max_logits[:, None]
    log_softmaxes = shifted_logits - logsumexps[:, None]
    softmaxes = torch.exp(log_softmaxes)

    shifted_targets = targets - weight_offset
    arange = torch.arange(softmaxes.shape[1], device=softmaxes.device)
    positives_mask = torch.where(shifted_targets[:, None] == arange, 1, 0)
    grad_logits = (softmaxes - positives_mask) * grad_loss[:, None]

    # Cheat to reuse the input memory and avoid an extra allocation
    orig_logits.copy_(grad_logits.to(float_dtype))
    return orig_logits


def fused_matmul_cross_entropy_fwd_checks(
    residual: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    tp_size: int,
) -> None:
    torch._check(residual.layout == torch.strided)
    torch._check(residual.ndim == 2)
    torch._check(residual.dtype == float_dtype)

    torch._check(weight.layout == torch.strided)
    torch._check(weight.ndim == 2)
    torch._check(weight.dtype == float_dtype)
    torch._check(weight.device == residual.device)

    torch._check(targets.layout == torch.strided)
    torch._check(targets.ndim == 1)
    torch._check(targets.dtype == torch.int64)
    torch._check(targets.device == residual.device)

    torch._check(residual.size(1) == weight.size(1))
    torch._check(residual.size(0) * tp_size == targets.size(0))


@torch.library.custom_op(
    "amaia::fused_matmul_cross_entropy_fwd",
    mutates_args=(),
    device_types="cuda",
)
def fused_matmul_cross_entropy_fwd(
    residual: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    pg_name: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tp_size = 1
    # pg: torch.distributed.ProcessGroup | None = None
    pg = None
    if pg_name is not None:
        pg = _resolve_process_group(pg_name)
        tp_size = pg.size()

    fused_matmul_cross_entropy_fwd_checks(residual, weight, targets, tp_size)
    torch._check(residual.is_cuda)

    weight_offset = 0
    if pg is not None:
        weight_offset = weight.shape[0] * pg.rank()
        residual = funcol.all_gather_tensor(residual, gather_dim=0, group=pg)

    max_logits = residual.new_full((targets.shape[0],), float("-inf"))
    logsumexps = residual.new_full(
        (targets.shape[0],), float("-inf"), dtype=float32_dtype
    )
    # We need to track missing positives when merging partial results during TP,
    # we encode them as NaNs so that they're unaffected by renormalizations.
    positive_logits = residual.new_full(
        (targets.shape[0],), float("nan"), dtype=float32_dtype
    )

    weight_chunks = weight.split(WEIGHT_BLOCK, dim=0)

    for weight_chunk in weight_chunks:
        logits = torch.matmul(residual, weight_chunk.t())
        fwd_epilogue(
            logits, targets, weight_offset, max_logits, logsumexps, positive_logits
        )
        weight_offset += WEIGHT_BLOCK
        del logits

    if pg is not None:
        old_max_logits = max_logits
        max_logits = funcol.all_reduce(max_logits, "max", group=pg).wait()
        logsumexps += old_max_logits - max_logits
        positive_logits += old_max_logits - max_logits
        logsumexps = funcol.all_reduce(logsumexps.exp(), "sum", group=pg).wait().log()
        positive_logits = funcol.all_reduce(
            positive_logits.nan_to_num(0.0), "sum", group=pg
        ).wait()

    loss = -(positive_logits - logsumexps)

    return loss, max_logits, logsumexps


@torch.library.register_fake("amaia::fused_matmul_cross_entropy_fwd")
def fused_matmul_cross_entropy_fwd_fake(
    residual: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    pg_name: str | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tp_size = 1 if pg_name is None else _resolve_process_group(pg_name).size()
    fused_matmul_cross_entropy_fwd_checks(residual, weight, targets, tp_size)

    loss = residual.new_empty((targets.shape[0],), dtype=float32_dtype)
    max_logits = residual.new_empty((targets.shape[0],))
    logsumexps = residual.new_empty((targets.shape[0],), dtype=float32_dtype)
    return loss, max_logits, logsumexps


@register_flop_formula(torch.ops.amaia.fused_matmul_cross_entropy_fwd)
def fused_matmul_cross_entropy_fwd_flops(
    residual_shape: torch.Size,
    weight_shape: torch.Size,
    targets_shape: torch.Size,
    pg_name: str | None,
    out_shape: torch.Size,
) -> int:
    return residual_shape[0] * residual_shape[1] * weight_shape[0] * 2


def fused_matmul_cross_entropy_bwd_checks(
    residual: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    grad_loss: torch.Tensor,
    max_logits: torch.Tensor,
    logsumexps: torch.Tensor,
    tp_size: int,
) -> None:
    fused_matmul_cross_entropy_fwd_checks(residual, weight, targets, tp_size)

    torch._check(grad_loss.layout == torch.strided)
    torch._check(grad_loss.ndim == 1)
    torch._check(grad_loss.dtype == float32_dtype)
    torch._check(grad_loss.device == residual.device)

    torch._check(max_logits.layout == torch.strided)
    torch._check(max_logits.ndim == 1)
    torch._check(max_logits.dtype == float_dtype)
    torch._check(max_logits.device == residual.device)

    torch._check(logsumexps.layout == torch.strided)
    torch._check(logsumexps.ndim == 1)
    torch._check(logsumexps.dtype == float32_dtype)
    torch._check(logsumexps.device == residual.device)

    torch._check(grad_loss.size(0) == targets.size(0))
    torch._check(max_logits.size(0) == targets.size(0))
    torch._check(logsumexps.size(0) == targets.size(0))


@torch.library.custom_op(
    "amaia::fused_matmul_cross_entropy_bwd",
    mutates_args=(),
    device_types="cuda",
)
def fused_matmul_cross_entropy_bwd(
    residual: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    grad_loss: torch.Tensor,
    max_logits: torch.Tensor,
    logsumexps: torch.Tensor,
    pg_name: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tp_size = 1
    # pg: torch.distributed.ProcessGroup | None = None
    pg = None
    if pg_name is not None:
        pg = _resolve_process_group(pg_name)
        tp_size = pg.size()

    fused_matmul_cross_entropy_bwd_checks(
        residual, weight, targets, grad_loss, max_logits, logsumexps, tp_size
    )
    torch._check(residual.is_cuda)

    weight_offset = 0
    if pg is not None:
        weight_offset = weight.shape[0] * pg.rank()
        residual = funcol.all_gather_tensor(residual, gather_dim=0, group=pg)

    grad_residual = torch.zeros_like(residual, dtype=float32_dtype)
    grad_weight = torch.empty_like(weight, dtype=float32_dtype)

    weight_chunks = weight.split(WEIGHT_BLOCK, dim=0)
    grad_weight_chunks = grad_weight.split(WEIGHT_BLOCK, dim=0)

    for weight_chunk, grad_weight_chunk in zip(
        weight_chunks, grad_weight_chunks, strict=True
    ):
        logits = torch.matmul(residual, weight_chunk.t())
        grad_logits = bwd_epilogue(
            logits, targets, weight_offset, max_logits, logsumexps, grad_loss
        )
        weight_offset += WEIGHT_BLOCK
        addmm_fp32(grad_logits, weight_chunk, out=grad_residual)
        torch.matmul(grad_logits.t(), residual, out=grad_weight_chunk)
        del logits
        del grad_logits

    if pg is not None:
        # This collective happens in fp32 to match non-TP numerics.
        grad_residual = funcol.reduce_scatter_tensor(
            grad_residual, "sum", scatter_dim=0, group=pg
        ).wait()

    return grad_residual.to(float_dtype), grad_weight


@torch.library.register_fake("amaia::fused_matmul_cross_entropy_bwd")
def fused_matmul_cross_entropy_bwd_fake(
    residual: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    grad_loss: torch.Tensor,
    max_logits: torch.Tensor,
    logsumexps: torch.Tensor,
    pg_name: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tp_size = 1 if pg_name is None else _resolve_process_group(pg_name).size()
    fused_matmul_cross_entropy_bwd_checks(
        residual, weight, targets, grad_loss, max_logits, logsumexps, tp_size
    )

    grad_residual = torch.empty_like(residual, dtype=float_dtype)
    grad_weight = torch.empty_like(weight, dtype=float_dtype)
    return grad_residual, grad_weight


@register_flop_formula(torch.ops.amaia.fused_matmul_cross_entropy_bwd)
def fused_matmul_cross_entropy_bwd_flops(
    residual_shape: torch.Size,
    weight_shape: torch.Size,
    targets_shape: torch.Size,
    grad_loss_shape: torch.Size,
    max_logits_shape: torch.Size,
    logsumexps_shape: torch.Size,
    pg_name: str | None,
    out_shape: torch.Size,
) -> int:
    return 2 * residual_shape[0] * residual_shape[1] * weight_shape[0] * 2


def fused_matmul_cross_entropy_setup_context(
    ctx: torch.autograd.function.FunctionCtx,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, str | None],
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    residual, weight, targets, pg_name = inputs
    _, max_logits, logsumexps = output
    torch._check(residual.requires_grad)
    torch._check(weight.requires_grad)
    torch._check(not targets.requires_grad)
    ctx.mark_non_differentiable(max_logits, logsumexps)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(residual, weight, targets, max_logits, logsumexps)
    ctx.pg_name = pg_name


def fused_matmul_cross_entropy_bwd_bridge(
    ctx: torch.autograd.function.FunctionCtx,
    grad_loss: torch.Tensor,
    grad_max_logits: None,
    grad_logsumexps: None,
) -> tuple[torch.Tensor, torch.Tensor, None, None]:
    assert grad_loss is not None
    assert grad_max_logits is None
    assert grad_logsumexps is None
    residual: torch.Tensor
    weight: torch.Tensor
    targets: torch.Tensor
    max_logits: torch.Tensor
    logsumexps: torch.Tensor
    residual, weight, targets, max_logits, logsumexps = ctx.saved_tensors
    grad_residual, grad_weight = fused_matmul_cross_entropy_bwd(
        residual, weight, targets, grad_loss, max_logits, logsumexps, ctx.pg_name
    )
    return grad_residual, grad_weight, None, None


torch.library.register_autograd(
    "amaia::fused_matmul_cross_entropy_fwd",
    fused_matmul_cross_entropy_bwd_bridge,
    setup_context=fused_matmul_cross_entropy_setup_context,
)


def fused_matmul_cross_entropy(
    residual: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    leading_dims = residual.shape[:-1]
    residual = residual.flatten(end_dim=-2)
    targets = targets.flatten(end_dim=-1)

    pg_name: str | None = None
    if isinstance(residual, DTensor):
        device_mesh = residual.device_mesh
        torch._assert(isinstance(weight, DTensor), f"{type(weight)} != DTensor")
        # We "tolerate" targets being a regular Tensor because that's how it is
        # in the reference code, due to use_local_tensor=True for the out proj.
        if not isinstance(targets, DTensor):
            targets = DTensor.from_local(targets, device_mesh, (Replicate(),))
        torch._assert(device_mesh.ndim == 1, f"{device_mesh.ndim=} != 1")
        torch._assert(
            weight.device_mesh == device_mesh,
            f"{weight.device_mesh=} != {device_mesh=}",
        )
        torch._assert(
            targets.device_mesh == device_mesh,
            f"{targets.device_mesh=} != {device_mesh=}",
        )
        torch._assert(
            residual.placements == (Shard(0),), f"{residual.placements=} != (Shard(0),)"
        )
        torch._assert(
            weight.placements == (Shard(0),),
            f"{weight.placements=} != (Shard(0),) (HINT: vocab_parallel should be enabled)",
        )
        torch._assert(
            targets.shape[0] % device_mesh.size() == 0,
            f"{targets.shape=}[0] % {device_mesh.size()=} != 0",
        )
        targets = targets.redistribute(placements=(Replicate(),))

        pg_name = device_mesh.get_group().group_name
        residual = residual.to_local()
        weight = weight.to_local()
        targets = targets.to_local()

    loss, _, _ = fused_matmul_cross_entropy_fwd(residual, weight, targets, pg_name)

    if pg_name is not None:
        loss = DTensor.from_local(loss, device_mesh, (Replicate(),))

    return loss.unflatten(0, leading_dims)
