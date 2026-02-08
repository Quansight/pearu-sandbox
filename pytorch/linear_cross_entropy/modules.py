"""
How to derive a backward formula for a tensor operation?
========================================================

See https://github.com/pearu/pearu.github.io/blob/main/torch_autograd_backward.md

"""

import random
import torch, torch.nn as nn
from torch.nn.parameter import Parameter
import math
import itertools


class AxpyFunction(torch.autograd.Function):
    """output = alpha * x + y"""

    @staticmethod
    def samples(device=None, dtype=None):
        for shape in [(3,), (3, 2)]:
            for alpha in [-1.5, 2.5]:
                x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                y = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                yield (alpha, x, y), alpha * x + y

    @staticmethod
    def forward(ctx, alpha: float, x: torch.Tensor, y: torch.Tensor):
        ctx.alpha = alpha
        return alpha * x + y

    @staticmethod
    def backward(ctx, grad_output):
        result = [None] * 3
        if ctx.needs_input_grad[1]:
            result[1] = grad_output * ctx.alpha
        if ctx.needs_input_grad[2]:
            result[2] = grad_output
        return tuple(result)


class MatmulFunction(torch.autograd.Function):
    """output = A @ B

    d(L) / d(A) = d(L) / d(A @ B) @ (d(A @ B) / d(A)) = d(L) / d(A @ B) @ B^T
    d(L) / d(B) = d(L) / d(BT @ AT) @ (d(A @ B) / d(A)) = d(L) / d(A @ B) @ B^T
    """

    @staticmethod
    def samples(device=None, dtype=None):
        for N, M, K in [(3, 2, 4)]:
            x = torch.randn((N, M), device=device, dtype=dtype, requires_grad=True)
            y = torch.randn((M, K), device=device, dtype=dtype, requires_grad=True)
            yield (x, y), x @ y

    @staticmethod
    def forward(ctx, x: torch.Tensor, y: torch.Tensor):
        saved = []
        indices = []
        if x.requires_grad:
            indices.append(len(saved))
            saved.append(y)
        if y.requires_grad:
            indices.append(len(saved))
            saved.append(x)
        ctx.save_for_backward(*saved)
        ctx.indices = indices
        return x @ y

    @staticmethod
    def backward(ctx, grad_output):
        result = [None] * 2
        if ctx.needs_input_grad[0]:
            result[0] = grad_output @ ctx.saved_tensors[ctx.indices[0]].T
        if ctx.needs_input_grad[1]:
            result[1] = ctx.saved_tensors[ctx.indices[1]].T @ grad_output
            # result[1] = (grad_output.T @ ctx.saved_tensors[ctx.indices[1]]).T
        return tuple(result)


class TransposeFunction(torch.autograd.Function):
    """output = A.T

    l = L(a=op(A))

    dl/dA = sum(dl/da_ij * dop(A)_ij/dA, i, j)

    d(L) / d(A) = d(L) / d(A.T) @ d(A.T) / d(A)

    """

    @staticmethod
    def samples(device=None, dtype=None):
        for N, M in [(3, 2)]:
            x = torch.randn((N, M), device=device, dtype=dtype, requires_grad=True)
            yield (x,), x.T

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return x.T

    @staticmethod
    def backward(ctx, grad_output):
        result = [None] * 1
        if ctx.needs_input_grad[0]:
            result[0] = grad_output.T
        return tuple(result)


def _get_saved_indices(args, save):
    saved = []
    indices = []
    for a, s in zip(args, save):
        if isinstance(a, torch.Tensor) and a.requires_grad:
            indices.append(len(saved))
            saved.append(s)
    return saved, indices


class LinearFunction(torch.autograd.Function):
    """output = A @ B.T + C"""

    @staticmethod
    def samples(device=None, dtype=None):
        for N, M, K in [(3, 2, 4)]:
            x = torch.randn((N, M), device=device, dtype=dtype, requires_grad=True)
            y = torch.randn((K, M), device=device, dtype=dtype, requires_grad=True)
            z = torch.randn((N, K), device=device, dtype=dtype, requires_grad=True)
            yield (x, y, z), x @ y.T + z

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        saved, indices = _get_saved_indices((x, y, z), (y, x, torch.ones_like(z)))
        if x.requires_grad and 0:
            indices.append(len(saved))
            saved.append(y)
        if y.requires_grad and 0:
            indices.append(len(saved))
            saved.append(x)
        if z.requires_grad and 0:
            indices.append(len(saved))
            saved.append(torch.ones_like(z))
        ctx.save_for_backward(*saved)
        ctx.indices = indices
        return x @ y.T + z

    @staticmethod
    def backward(ctx, grad_output):
        result = [None] * 3
        if ctx.needs_input_grad[0]:
            result[0] = grad_output @ ctx.saved_tensors[ctx.indices[0]]
        if ctx.needs_input_grad[1]:
            result[1] = grad_output.T @ ctx.saved_tensors[ctx.indices[1]]
        if ctx.needs_input_grad[2]:
            result[2] = grad_output
        return tuple(result)


def call_forward(ctx, cls, *args):
    if not hasattr(ctx, "context_stack"):
        ctx.context_stack = []
    saved = ctx.saved_tensors
    ctx1 = torch.autograd.function.FunctionCtx()
    ctx1.needs_input_grad = [
        isinstance(a, torch.Tensor) and a.requires_grad for a in args
    ]
    output = cls.forward(ctx1, *args)
    if hasattr(ctx1, "to_save"):
        ctx1.saved_tensors = ctx1.to_save
        ctx.save_for_backward(*saved, *ctx1.to_save)
    ctx.context_stack.insert(0, (ctx1, cls))
    return output


class LinearAsCompositionFunction(torch.autograd.Function):
    """output = A @ B.T + C"""

    @staticmethod
    def samples(device=None, dtype=None):
        for N, M, K in [(3, 2, 4)]:
            x = torch.randn((N, M), device=device, dtype=dtype, requires_grad=True)
            y = torch.randn((K, M), device=device, dtype=dtype, requires_grad=True)
            z = torch.randn((N, K), device=device, dtype=dtype, requires_grad=True)
            yield (x, y, z), x @ y.T + z

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        ctx.save_for_backward(x, y, z)
        return x @ y.T + z

    @staticmethod
    def backward(ctx, grad_output):
        x, y, z = ctx.saved_tensors
        results = [None] * 3
        if ctx.needs_input_grad[0]:
            results[0] = grad_output @ y
        if ctx.needs_input_grad[1]:
            results[1] = grad_output.T @ x
        if ctx.needs_input_grad[2]:
            results[2] = grad_output
        return tuple(results)


class LogFunction(torch.autograd.Function):
    """output = log(x)"""

    @staticmethod
    def samples(device=None, dtype=None):
        for shape in [(3,), (3, 2)]:
            x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
            yield (torch.exp(x),), x

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        if x.requires_grad:
            ctx.save_for_backward(1 / x)
        return torch.log(x)

    @staticmethod
    def backward(ctx, grad_output):
        result = [None]
        if ctx.needs_input_grad[0]:
            result[0] = grad_output * ctx.saved_tensors[0]
        return tuple(result)


class SumFunction(torch.autograd.Function):
    """output = sum(x, dim=, keepdim=)"""

    @staticmethod
    def samples(device=None, dtype=None):
        for shape in [(3,), (3, 2), (2, 4, 3)]:
            for dim in set([0, len(shape) - 1, max(0, len(shape) - 2)]):
                for keepdim in [False, True]:
                    x = torch.randn(
                        shape, device=device, dtype=dtype, requires_grad=True
                    )
                    yield (x, dim, keepdim), x.sum(dim=dim, keepdim=keepdim)

    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int, keepdim: bool):
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.shape = x.shape
        return x.sum(dim=dim, keepdim=keepdim)

    @staticmethod
    def backward(ctx, grad_output):
        result = [None] * 3
        if ctx.needs_input_grad[0]:
            if ctx.keepdim:
                result[0] = grad_output.expand(ctx.shape)
            else:
                result[0] = grad_output.unsqueeze(ctx.dim).expand(ctx.shape)
        return tuple(result)


class MaxFunction(torch.autograd.Function):
    """output = max(x, dim=, keepdim=)"""

    @staticmethod
    def samples(device=None, dtype=None):
        for shape in [(3,), (3, 2), (2, 4, 3), (3, 3)]:
            for dim in set([0, len(shape) - 1, max(0, len(shape) - 2)]):
                for keepdim in [False, True]:
                    x = torch.randn(
                        shape, device=device, dtype=dtype, requires_grad=True
                    )
                    yield (x, dim, keepdim), x.max(dim=dim, keepdim=keepdim)[0]

    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int, keepdim: bool):
        ctx.dim = dim
        ctx.keepdim = keepdim
        ctx.shape = x.shape
        output, indices = x.max(dim=dim, keepdim=keepdim)
        mask = torch.zeros_like(x)
        if keepdim:
            grad_input = output.expand(x.shape) == x
        else:
            grad_input = output.unsqueeze(dim).expand(x.shape) == x
            indices = indices.unsqueeze(dim)
        mask.scatter_(ctx.dim, indices, 1)
        ctx.save_for_backward(mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        result = [None] * 3
        if ctx.needs_input_grad[0]:
            (mask,) = ctx.saved_tensors
            if ctx.keepdim:
                result[0] = grad_output * mask
            else:
                result[0] = grad_output.unsqueeze(ctx.dim) * mask
        return tuple(result)


class SoftmaxFunction(torch.autograd.Function):
    """output = softmax(x, dim=)"""

    @staticmethod
    def samples(device=None, dtype=None):
        for shape in [(3,), (3, 2), (2, 4, 3), (3, 3)]:
            for dim in set([0, len(shape) - 1, max(0, len(shape) - 2)]):
                x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                yield (x, dim), x.softmax(dim=dim)

    @staticmethod
    def forward(ctx, x: torch.Tensor, dim: int):
        ctx.dim = dim
        output = x.softmax(dim=dim)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        result = [None] * 2
        if ctx.needs_input_grad[0]:
            (output,) = ctx.saved_tensors
            result[0] = (
                grad_output - (grad_output * output).sum(dim=ctx.dim, keepdim=True)
            ) * output
        return tuple(result)


class NNLLossFunction(torch.autograd.Function):
    """output = nnl_loss(x, target, weight)"""

    @staticmethod
    def samples(device=None, dtype=None):
        for reduction in ["sum", "mean"]:
            for shape in [(4, 5)]:
                for ii in set([-100, 0, len(shape) - 1, max(0, len(shape) - 2)]):
                    x = torch.randn(
                        shape, device=device, dtype=dtype, requires_grad=True
                    )
                    t = torch.randint(
                        0,
                        shape[1],
                        shape[:-1],
                        device=device,
                        dtype=torch.int64,
                        requires_grad=False,
                    )
                    w = torch.exp(
                        torch.randn(
                            shape[1], device=device, dtype=dtype, requires_grad=False
                        )
                    )
                    yield (x, t, w, reduction, ii), torch.nn.functional.nll_loss(
                        x, t, weight=w, ignore_index=ii, reduction=reduction
                    )

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        t: torch.Tensor,
        weight: torch.Tensor,
        reduction: str,
        ignore_index: int,
    ):
        ctx.reduction = reduction
        output = torch.nn.functional.nll_loss(
            x, t, weight=weight, reduction=reduction, ignore_index=ignore_index
        )
        if x.requires_grad:
            if ignore_index >= 0:
                weight = weight.clone()
                weight[ignore_index] = 0
            wmask = torch.zeros_like(x).scatter_(
                1, t.unsqueeze(1), weight.index_select(0, t).unsqueeze(1)
            )
            if ignore_index >= 0 and 0:
                # might not be correct!!
                # this is equivalent to setting weight[ii] = 0, see above
                wmask.select(1, ignore_index).zero_()
            if reduction == "sum":
                pass
            elif reduction == "mean":
                wmask /= weight.index_select(0, t).sum()
            elif reduction == "none":
                assert 0  # not impl
            else:
                assert 0, reduction  # not supported
            ctx.save_for_backward(wmask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        result = [None] * 5
        if ctx.needs_input_grad[0]:
            (wmask,) = ctx.saved_tensors
            result[0] = -wmask * grad_output
        return tuple(result)


class LinearCrossEntropyFunctionBase(torch.autograd.Function):
    @staticmethod
    def samples(device=None, dtype=None):
        for num_chunks in [1, 2, 4]:
            for reduction in ["sum", "mean"]:
                for num_batches in [8, 4]:
                    for in_features in [5, 16]:
                        for num_classes in [4, 6]:
                            for ii in set(
                                [-100, 0, min(1, num_classes - 1), num_classes - 1][:1]
                            ):
                                x = torch.randn(
                                    (num_batches, in_features),
                                    device=device,
                                    dtype=dtype,
                                    requires_grad=True,
                                )
                                L = torch.randn(
                                    (num_classes, in_features),
                                    device=device,
                                    dtype=dtype,
                                    requires_grad=True,
                                )
                                t = torch.randint(
                                    0,
                                    num_classes,
                                    (num_batches,),
                                    device=device,
                                    dtype=torch.int64,
                                    requires_grad=False,
                                )
                                if ii >= 0 and torch.all(t == ii):
                                    t[0] = random.sample(
                                        sorted(set(range(num_classes)) - {ii}), 1
                                    )[0]
                                w = torch.exp(
                                    torch.randn(
                                        num_classes,
                                        device=device,
                                        dtype=dtype,
                                        requires_grad=False,
                                    )
                                )
                                yield (
                                    x,
                                    L,
                                    t,
                                    w,
                                    reduction,
                                    ii,
                                    num_chunks,
                                ), torch.nn.functional.linear_cross_entropy(
                                    x,
                                    L,
                                    t,
                                    weight=w,
                                    ignore_index=ii,
                                    reduction=reduction,
                                )
                                w = None
                                yield (
                                    x,
                                    L,
                                    t,
                                    w,
                                    reduction,
                                    ii,
                                    num_chunks,
                                ), torch.nn.functional.linear_cross_entropy(
                                    x,
                                    L,
                                    t,
                                    weight=w,
                                    ignore_index=ii,
                                    reduction=reduction,
                                )


import vocab_chunking


class VoLinearCrossEntropyFunction(LinearCrossEntropyFunctionBase):

    @staticmethod
    def samples(device=None, dtype=None):
        for (
            x,
            L,
            t,
            w,
            reduction,
            ii,
            splits,
        ), ref in LinearCrossEntropyFunctionBase.samples(device=device, dtype=dtype):
            if reduction == "sum" and ii < 0 and w is None:
                yield (x, L, t), ref

    @staticmethod
    def forward(ctx, input: torch.Tensor, L: torch.Tensor, target: torch.Tensor):
        loss, max_logits, logsumexps = vocab_chunking.fused_matmul_cross_entropy_fwd(
            input, L, target, None
        )
        ctx.save_for_backward(input, L, target, max_logits, logsumexps)
        return loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        input, L, target, max_logits, logsumexps = ctx.saved_tensors
        grad_residual, grad_weight = vocab_chunking.fused_matmul_cross_entropy_bwd(
            input, L, target, grad_output, max_logits, logsumexps, None
        )
        return grad_residual, grad_weight, None


def _validate_params(params):
    for k in params:
        if k not in [
            "grad_in_forward",
            "grad_inplace",
            "chunks_features",
            "chunks_batches",
            "chunks_classes",
            "chunk_size_batches",
            "chunk_size_features",
            "chunk_size_classes",
            "chunk_size",
        ]:
            print(f"Warning: unknown parameter {k}")


class MyLinearCrossEntropyFunction(LinearCrossEntropyFunctionBase):

    @staticmethod
    def samples(device=None, dtype=None):
        for (
            reduction,
            num_batches,
            in_features,
            num_classes,
            params,
        ) in itertools.product(
            ["sum", "mean"],
            [8, 3],
            [5, 16],
            [4, 7],
            [
                dict(),
                dict(chunks_batches=2, chunks_features=2),
                dict(grad_in_forward=True),
                dict(grad_in_forward=True, chunks_batches=2, chunks_features=2),
                dict(grad_in_forward=True, grad_inplace=True),
                dict(
                    grad_in_forward=True,
                    grad_inplace=True,
                    chunks_batches=2,
                    chunks_features=2,
                ),
                dict(grad_in_forward=True, chunks_classes=2),
                dict(grad_in_forward=True, chunks_batches=2, chunks_classes=2),
                dict(grad_in_forward=True, chunks_features=2, chunks_classes=2),
                dict(
                    grad_in_forward=True,
                    chunks_features=2,
                    chunks_batches=2,
                    chunks_classes=2,
                ),
            ],
        ):
            for ii in set([-100, 0, min(1, num_classes - 1), num_classes - 1]):
                x = torch.randn(
                    (num_batches, in_features),
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                L = torch.randn(
                    (num_classes, in_features),
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
                t = torch.randint(
                    0,
                    num_classes,
                    (num_batches,),
                    device=device,
                    dtype=torch.int64,
                    requires_grad=False,
                )
                if ii >= 0 and torch.all(t == ii):
                    t[0] = random.sample(sorted(set(range(num_classes)) - {ii}), 1)[0]
                w1 = torch.ones(
                    num_classes, device=device, dtype=dtype, requires_grad=False
                )
                w2 = torch.exp(
                    torch.randn(
                        num_classes, device=device, dtype=dtype, requires_grad=False
                    )
                )
                for w in [None, w1, w2]:
                    unnormalized_weight = (
                        MyLinearCrossEntropyFunction.get_unnormalized_weight(
                            w, num_classes, reduction, ii, device, dtype
                        )
                    )
                    yield (
                        x,
                        L,
                        t,
                        unnormalized_weight,
                        reduction,
                        params,
                    ), torch.nn.functional.linear_cross_entropy(
                        x, L, t, weight=w, ignore_index=ii, reduction=reduction
                    )

    @staticmethod
    def get_unnormalized_weight(
        weight: torch.Tensor,
        num_classes: int,
        reduction: str,
        ignore_index: int,
        device,
        dtype,
    ):
        # resolves unspecified weight and specified ignore_index
        if weight is None:
            weight = torch.ones(
                (num_classes,), device=device, dtype=dtype, requires_grad=False
            )

        if ignore_index >= 0:
            weight = weight.clone()
            weight.narrow(0, ignore_index, 1).zero_()
        return weight

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        L: torch.Tensor,
        target: torch.Tensor,
        unnormalized_weight: torch.Tensor,
        reduction: str,
        params: dict,
    ):
        # input: (num_batches, in_features)
        # L: (num_classes, in_features)
        # target: (num_batches,)
        # unnormalized_weight: (num_classes,)
        # reduction: "mean" or "sum", "none" is not implemented yet
        # params: dict(
        #   grad_in_forward: bool,  # when True, pre-compute gradients data in forward, default is False
        #   grad_inplace: bool,     # when True, use inplace mul in computing gradients, gradcheck will complain, default is False
        #   chunks_batches: int,    # number of chunks in num_batches dimension, default is 1
        #   chunks_features: int,   # number of chunks in in_features dimension, default is 1
        #   chunks_classes: int,    # number of chunks in num_classes dimension, default is 1
        # )
        device = input.device
        dtype = input.dtype
        num_batches, in_features = input.shape
        num_classes, _ = L.shape
        _validate_params(params)

        def chunk_iter(size, chunks):
            """Defined by equivalence of

              [torch.narrow(tensor, dim, start, length) for start, length in chunk_iter(tensor.shape[dim], chunks)]

            and

              torch.chunk(tensor, chunks, dim)

            Useful when chunking several tensors with the same chunking strategy.
            """
            if chunks == 1:
                yield 0, size
                return
            length = max(1, size // chunks)
            start = 0
            size_ = 0
            while start < size:
                size_ += length
                if size_ > size:
                    length -= size_ - size
                yield start, length
                start += length

        def make_zeros(*shape):
            return torch.zeros(shape, device=device, dtype=dtype, requires_grad=False)

        def make_empty(*shape):
            return torch.empty(shape, device=device, dtype=dtype, requires_grad=False)

        def ensure_size(input, dim, size):
            if input.shape[dim] != size:
                return input.narrow(dim, 0, size)
            return input

        ctx.params = params
        if not params.get("grad_in_forward", False):
            ctx.chunk_iter = chunk_iter

        output = make_zeros()
        X = None
        if input.requires_grad or L.requires_grad:
            if params.get("grad_in_forward", False):
                # A chunk buffer used to hold logits, softmax of logits:
                X = make_empty(
                    max(1, num_batches // params.get("chunks_batches", 1)),
                    max(1, num_classes // params.get("chunks_classes", 1)),
                )
                # grad_input and grad_L contain initial gradients
                # (grad_output==1) to be passed over to backward.
                if input.requires_grad:
                    grad_input = make_empty(*input.shape)
                if L.requires_grad:
                    grad_L = make_zeros(*L.shape)
                    # A chunk buffer used in grad_L computation:
                    G = make_empty(
                        max(1, num_classes // params.get("chunks_classes", 1)),
                        max(1, in_features // params.get("chunks_features", 1)),
                    )
            else:
                # X will contain pre-computed gradient data to be
                # passed over to backward together with inputs.
                X = make_empty(num_batches, num_classes)

        if reduction == "mean":
            weight = unnormalized_weight.clone()
            weight_target = weight.index_select(0, target)
            d = weight_target.sum()
            weight.div_(d)
            weight_target.div_(d)
        else:
            weight = unnormalized_weight
            weight_target = weight.index_select(0, target)

        for bchunk_start, bchunk_size in chunk_iter(
            num_batches, params.get("chunks_batches", 1)
        ):
            x = input.narrow(0, bchunk_start, bchunk_size)
            t = target.narrow(0, bchunk_start, bchunk_size)
            weight_t = weight_target.narrow(0, bchunk_start, bchunk_size)
            if params.get("grad_in_forward", False):
                X_ = ensure_size(X, 0, bchunk_size)
            else:
                X_ = X.narrow(0, bchunk_start, bchunk_size)

            for cchunk_start, cchunk_size in chunk_iter(
                num_classes, params.get("chunks_classes", 1)
            ):
                L_ = L.narrow(0, cchunk_start, cchunk_size)
                X__ = ensure_size(X_, 1, cchunk_size)

                torch.mm(x, L_.T, out=X__)  # projection

                if cchunk_start == 0:
                    Xmax = X__.max(dim=1, keepdim=True)[0]
                else:
                    corrXmax = Xmax
                    Xmax = X__.max(dim=1, keepdim=True)[0].max(corrXmax)
                    corrXmax.sub_(Xmax)

                X__.sub_(Xmax)

                if cchunk_start > 0:
                    # correct under-estimated Xmax
                    total_mask = t < cchunk_start
                    output.sub_(
                        weight_t[total_mask].dot(corrXmax[total_mask].squeeze(1))
                    )

                if cchunk_size == num_classes:
                    output.sub_(weight_t.dot(X__.gather(1, t.unsqueeze(1)).squeeze(1)))
                else:
                    mask = (cchunk_start <= t) & (t < cchunk_start + cchunk_size)
                    t_ = t.masked_select(mask) - cchunk_start
                    weight_t_ = weight_t.masked_select(mask)
                    output.sub_(
                        weight_t_.dot(X__[mask].gather(1, t_.unsqueeze(1)).squeeze(1))
                    )

                X__.exp_()

                if cchunk_start == 0:
                    expXsum = X__.sum(dim=1)
                else:
                    # correct under-estimated Xmax
                    expXsum.add_(corrXmax.squeeze(1))
                    expXsum.exp_()
                    expXsum.add_(X__.sum(dim=1))

                if cchunk_size == num_classes and (
                    input.requires_grad or L.requires_grad
                ):
                    # X__ will be used in the for-loop below or in
                    # backward if grad_in_forward is False
                    X__.mul_(-(weight_t / expXsum).unsqueeze(1))

                expXsum.log_()

            output.add_(weight_t.dot(expXsum))

            if (input.requires_grad or L.requires_grad) and params.get(
                "grad_in_forward", False
            ):
                if params.get("chunks_classes", 1) > 1:
                    # required only for recomputing X__ below
                    expXsum.exp_()
                if input.requires_grad:
                    grad_x = grad_input.narrow(0, bchunk_start, bchunk_size)
                    torch.index_select(L, 0, t, out=grad_x)
                    grad_x.mul_(-weight_t.unsqueeze(1))

                for cchunk_start, cchunk_size in chunk_iter(
                    num_classes, params.get("chunks_classes", 1)
                ):
                    if cchunk_size == num_classes:
                        t_ = t
                        L_ = L
                        weight_ = weight
                        # X__ is computed in the previous for-loop
                    else:
                        # recompute X__, however, we can re-use Xmax
                        # and expXsum computed from the previous
                        # for-loop
                        mask = (cchunk_start <= t) & (t < cchunk_start + cchunk_size)
                        t_ = t.masked_select(mask) - cchunk_start
                        L_ = L.narrow(0, cchunk_start, cchunk_size)
                        weight_ = weight.narrow(0, cchunk_start, cchunk_size)
                        X__ = ensure_size(X_, 1, cchunk_size)
                        torch.addmm(Xmax, x, L_.T, beta=-1, out=X__)
                        X__.exp_()
                        X__.mul_(-(weight_t / expXsum).unsqueeze(1))  # X is `S * w`

                    if input.requires_grad:
                        grad_x.addmm_(X__, L_, alpha=-1)

                    if L.requires_grad:
                        G_ = ensure_size(G, 0, cchunk_size)
                        grad_L_ = grad_L.narrow(0, cchunk_start, cchunk_size)
                        for fchunk_start, fchunk_size in chunk_iter(
                            in_features, params.get("chunks_features", 1)
                        ):
                            x_ = x.narrow(1, fchunk_start, fchunk_size)
                            G__ = ensure_size(G_, 1, fchunk_size)
                            G__.zero_()
                            if cchunk_size == num_classes:
                                G__.index_add_(0, t, x_)
                            else:
                                G__.index_add_(0, t_, x_[mask])
                            G__.mul_(weight_.unsqueeze(1))
                            G__.addmm_(X__.T, x_, alpha=-1, beta=-1)
                            grad_L_.narrow(1, fchunk_start, fchunk_size).add_(G__)

        if params.get("grad_in_forward", False):
            save_indices = [None, None]
            saved = []
            if input.requires_grad:
                save_indices[0] = len(saved)
                saved.append(grad_input)
            if L.requires_grad:
                save_indices[1] = len(saved)
                saved.append(grad_L)
            if saved:
                ctx.save_indices = save_indices
                ctx.save_for_backward(*saved)
        else:
            if input.requires_grad or L.requires_grad:
                ctx.save_for_backward(input, X, L, target, weight, weight_target)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        result = [None] * 6
        if ctx.params.get("grad_in_forward", False):
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
                saved = ctx.saved_tensors
                if ctx.params.get("grad_inplace", False):
                    if ctx.needs_input_grad[0]:
                        grad_input = saved[ctx.save_indices[0]]
                        grad_input.mul_(grad_output)
                        result[0] = grad_input
                    if ctx.needs_input_grad[1]:
                        grad_L = saved[ctx.save_indices[1]]
                        grad_L.mul_(grad_output)
                        result[1] = grad_L
                else:
                    if ctx.needs_input_grad[0]:
                        grad_input = saved[ctx.save_indices[0]]
                        # creates a new tensor that increases memory usage peak
                        result[0] = grad_input * grad_output
                    if ctx.needs_input_grad[1]:
                        grad_L = saved[ctx.save_indices[1]]
                        # creates a new tensor that increases memory usage peak
                        result[1] = grad_L * grad_output
        elif ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            input, X, L, target, weight, weight_target = ctx.saved_tensors
            num_batches, in_features = input.shape
            num_classes, _ = L.shape
            assert (
                ctx.params.get("chunks_classes", 1) == 1
            )  # chunking along num_classes dimension not implemented

            if ctx.needs_input_grad[0]:
                grad_input = torch.empty_like(input, requires_grad=False)
            if ctx.needs_input_grad[1]:
                grad_L = torch.zeros_like(L, requires_grad=False)
                G = torch.zeros(
                    (
                        num_classes,
                        max(1, in_features // ctx.params.get("chunks_features", 1)),
                    ),
                    device=L.device,
                    dtype=L.dtype,
                    requires_grad=False,
                )

            for bchunk_start, bchunk_size in ctx.chunk_iter(
                num_batches, ctx.params.get("chunks_batches", 1)
            ):
                x = input.narrow(0, bchunk_start, bchunk_size)
                t = target.narrow(0, bchunk_start, bchunk_size)
                X_ = X.narrow(0, bchunk_start, bchunk_size)
                weight_t = weight_target.narrow(0, bchunk_start, bchunk_size)

                if ctx.needs_input_grad[0]:
                    grad_x = grad_input.narrow(0, bchunk_start, bchunk_size)
                    torch.index_select(L, 0, t, out=grad_x)
                    grad_x.mul_(weight_t.unsqueeze(1))
                    grad_x.addmm_(X_, L, alpha=-grad_output, beta=-grad_output)

                if ctx.needs_input_grad[1]:
                    G_ = G
                    for fchunk_start, fchunk_size in ctx.chunk_iter(
                        in_features, ctx.params.get("chunks_features", 1)
                    ):
                        x1 = x.narrow(1, fchunk_start, fchunk_size)
                        if G_.shape[1] != fchunk_size:
                            G_ = G.narrow(1, 0, fchunk_size)
                        else:
                            G_ = G
                        G_.zero_()
                        G_.index_add_(0, t, x1)
                        G_.mul_(weight.unsqueeze(1))
                        G_.addmm_(X_.T, x1, alpha=-grad_output, beta=-grad_output)
                        grad_L.narrow(1, fchunk_start, fchunk_size).add_(G_)

            if ctx.needs_input_grad[0]:
                result[0] = grad_input
            if ctx.needs_input_grad[1]:
                result[1] = grad_L
        return tuple(result)


class LinearCrossEntropyLossBase(nn.Module):

    @staticmethod
    def samples(device="cuda", dtype=torch.float32):
        for reduction in ["sum", "mean"][:1]:
            for num_batches in [8]:
                for in_features in [5, 3]:
                    for num_classes in [4, 6]:
                        for ii in set(
                            [-100, 0, min(1, num_classes - 1), num_classes - 1]
                        ):
                            input = torch.randn(
                                (num_batches, in_features),
                                device=device,
                                dtype=dtype,
                                requires_grad=True,
                            )
                            target = torch.randint(
                                0,
                                num_classes,
                                (num_batches,),
                                device=device,
                                dtype=torch.int64,
                                requires_grad=False,
                            )
                            module_args, module_kwargs, forward_args = (
                                (in_features, num_classes),
                                dict(
                                    ignore_index=ii,
                                    reduction=reduction,
                                    label_smoothing=0.0,
                                    bias=False,
                                    device=device,
                                    dtype=dtype,
                                ),
                                (input, target),
                            )
                            yield module_args, module_kwargs, forward_args


class MyLinearCrossEntropyLoss(LinearCrossEntropyLossBase):

    @staticmethod
    def samples(device=None, dtype=None):
        for (
            reduction,
            num_batches,
            in_features,
            num_classes,
            params,
        ) in itertools.product(
            ["sum", "mean"],
            [8, 3],
            [5, 16],
            [4, 7],
            [
                dict(),
                dict(chunks_batches=2, chunks_features=2),
                dict(grad_in_forward=True),
                dict(grad_in_forward=True, chunks_batches=2, chunks_features=2),
                dict(grad_in_forward=True, grad_inplace=True),
                dict(
                    grad_in_forward=True,
                    grad_inplace=True,
                    chunks_batches=2,
                    chunks_features=2,
                ),
                dict(grad_in_forward=True, chunks_classes=2),
                dict(grad_in_forward=True, chunks_batches=2, chunks_classes=2),
                dict(grad_in_forward=True, chunks_features=2, chunks_classes=2),
                dict(
                    grad_in_forward=True,
                    chunks_features=2,
                    chunks_batches=2,
                    chunks_classes=2,
                ),
            ],
        ):
            for ii in set([-100, 0, min(1, num_classes - 1), num_classes - 1]):
                t = torch.randint(
                    0,
                    num_classes,
                    (num_batches,),
                    device=device,
                    dtype=torch.int64,
                    requires_grad=False,
                )
                if ii >= 0 and torch.all(t == ii):
                    t[0] = random.sample(sorted(set(range(num_classes)) - {ii}), 1)[0]
                w1 = torch.ones(
                    num_classes, device=device, dtype=dtype, requires_grad=False
                )
                w2 = torch.exp(
                    torch.randn(
                        num_classes, device=device, dtype=dtype, requires_grad=False
                    )
                )
                for w in [None, w1, w2]:
                    x = torch.randn(
                        (num_batches, in_features),
                        device=device,
                        dtype=dtype,
                        requires_grad=True,
                    )
                    L = torch.randn(
                        (num_classes, in_features),
                        device=device,
                        dtype=dtype,
                        requires_grad=True,
                    )
                    unnormalized_weight = (
                        MyLinearCrossEntropyFunction.get_unnormalized_weight(
                            w, num_classes, reduction, ii, device, dtype
                        )
                    )
                    module_args, module_kwargs, forward_args = (
                        (in_features, num_classes),
                        dict(
                            ignore_index=ii,
                            reduction=reduction,
                            label_smoothing=0.0,
                            bias=False,
                            params=params,
                            device=device,
                            dtype=dtype,
                        ),
                        (x, t),
                    )
                    yield module_args, module_kwargs, forward_args

    weight: torch.Tensor | None

    def __init__(
        self,
        dim: int,
        n_classes: int,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        bias: bool = False,
        weight: torch.Tensor | None = None,
        device=None,
        dtype=None,
        params: dict | None = None,
    ):
        super().__init__()
        assert reduction in {"sum", "mean"}, reduction
        assert label_smoothing == 0.0
        assert not bias
        _validate_params(params)
        # params: dict(
        #   grad_in_forward: bool,   # when True, pre-compute gradients data in forward, default is False
        #   grad_inplace: bool,      # when True, use inplace mul in computing gradients, gradcheck will complain, default is False
        #   chunks_batches: int,     # number of chunks in num_batches dimension, default is 1
        #   chunks_features: int,    # number of chunks in in_features dimension, default is 1
        #   chunk_size_batches: int  # used to initialize chunks_batches
        #   chunk_size_features: int # used to initialize chunks_features
        #   chunk_size: int          # used to initialize chunks_batches and chunks_features
        # )
        self.params = params or dict()
        self.projection = Parameter(
            torch.empty((n_classes, dim), device=device, dtype=dtype)
        )

        unnormalized_weight = MyLinearCrossEntropyFunction.get_unnormalized_weight(
            weight, n_classes, reduction, ignore_index, device, dtype
        )
        self.register_buffer("weight", unnormalized_weight)

        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

        nn.init.kaiming_uniform_(self.projection, a=math.sqrt(5))

    def forward(self, x, targ):
        assert targ.dtype == torch.int64
        params = self.params
        if params is not None:
            if "chunks_batches" not in params:
                size = params.get("chunk_size_batches", params.get("chunk_size"))
                if size is not None:
                    params["chunks_batches"] = max(1, targ.shape[0] // size)
            if "chunks_features" not in params:
                size = params.get("chunk_size_features", params.get("chunk_size"))
                if size is not None:
                    params["chunks_features"] = max(1, x.shape[0] // size)
            if "chunks_classes" not in params:
                size = params.get("chunk_size_classes", params.get("chunk_size"))
                if size is not None:
                    params["chunks_classes"] = max(1, self.projection.shape[0] // size)
            if "grad_in_forward" not in params and params.get("grad_inplace", False):
                params["grad_in_forward"] = True
        print(f'{params=}')
        return MyLinearCrossEntropyFunction.apply(
            x, self.projection, targ, self.weight, self.reduction, params
        )


class VoLinearCrossEntropyLoss(nn.Module):

    @staticmethod
    def samples(device="cuda", dtype=torch.float32):
        for reduction in ["sum", "mean"]:
            for num_batches in [8]:
                for in_features in [5, 3]:
                    for num_classes in [4, 6]:
                        for ii in set(
                            [-100, 0, min(1, num_classes - 1), num_classes - 1][:1]
                        ):
                            input = torch.randn(
                                (num_batches, in_features),
                                device=device,
                                dtype=dtype,
                                requires_grad=True,
                            )
                            target = torch.randint(
                                0,
                                num_classes,
                                (num_batches,),
                                device=device,
                                dtype=torch.int64,
                                requires_grad=False,
                            )
                            module_args, module_kwargs, forward_args = (
                                (in_features, num_classes),
                                dict(
                                    ignore_index=ii,
                                    reduction=reduction,
                                    label_smoothing=0.0,
                                    bias=False,
                                    device=device,
                                    dtype=dtype,
                                ),
                                (input, target),
                            )
                            yield module_args, module_kwargs, forward_args

    def __init__(
        self,
        dim: int,
        n_classes: int,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        bias: bool = False,
        weight: torch.Tensor | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert reduction in {"sum", "mean"}, reduction
        assert label_smoothing == 0.0
        assert not bias
        assert ignore_index == -100
        assert weight is None
        self.projection = Parameter(
            torch.empty((n_classes, dim), device=device, dtype=dtype)
        )

        unnormalized_weight = MyLinearCrossEntropyFunction.get_unnormalized_weight(
            None, n_classes, reduction, ignore_index, device, dtype
        )
        self.register_buffer("weight", unnormalized_weight)

        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        nn.init.kaiming_uniform_(self.projection, a=math.sqrt(5))

    def forward(self, x, targ):
        assert targ.dtype == torch.int64
        l = VoLinearCrossEntropyFunction.apply(x, self.projection, targ)
        if self.reduction == "sum":
            return l.sum()
        if self.reduction == "mean":
            return l.mean()
        return l


from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


class LiLinearCrossEntropyLoss(LinearCrossEntropyLossBase):

    def __init__(
        self,
        dim: int,
        n_classes: int,
        ignore_index: int = -100,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        bias: bool = False,
        weight: torch.Tensor | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert not bias, bias
        self.module = LigerFusedLinearCrossEntropyLoss(
            ce_weight=weight,
            ignore_index=ignore_index,
            lse_square_scale=0.0,
            label_smoothing=label_smoothing,
            reduction=reduction,
            softcap=None,
            return_z_loss=False,
            accum_dtype=dtype,
            use_token_scaling=False,
            return_token_accuracy=False,
        ).to(dtype)
        self.projection = Parameter(
            torch.empty((n_classes, dim), device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.projection, a=math.sqrt(5))

    def forward(self, x, targ):
        l = self.module(self.projection, x, targ)
        return l


def test_function(cls, device="cpu", dtype=torch.float64):
    torch.manual_seed(5431)
    print(f"{cls.__name__}")
    f = cls.apply
    for args, expected in cls.samples(device=device, dtype=dtype):
        print(".", end="", flush=True)
        l = f(*args)
        torch.testing.assert_close(l, expected)
        if isinstance(args[-1], dict) and args[-1].get("grad_inplace"):
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            continue
        if dtype == torch.float64:
            torch.autograd.gradcheck(f, args)

    print()


def test_module(cls, ref_cls, device="cuda", dtype=torch.float64):
    print(f"{cls.__name__} {ref_cls.__name__}")

    for module_args, module_kwargs, forward_args in cls.samples(
        device=device, dtype=dtype
    ):
        print(".", end="", flush=True)
        torch.manual_seed(5431)
        m = cls(*module_args, **module_kwargs)
        params = module_kwargs.pop("params", None)

        torch.manual_seed(5431)
        ref_m = ref_cls(*module_args, **module_kwargs)

        ref_forward_args = [
            a.detach().clone().requires_grad_(a.requires_grad) for a in forward_args
        ]
        l = m(*forward_args)
        ref_l = ref_m(*ref_forward_args)
        torch.testing.assert_close(l, ref_l)
        l.sum().backward()
        ref_l.sum().backward()
        torch.testing.assert_close(forward_args[0].grad, ref_forward_args[0].grad)

        if issubclass(cls, VoLinearCrossEntropyLoss) or issubclass(
            cls, LiLinearCrossEntropyLoss
        ):
            print(f"skip gradcheck")
            continue

        if params is not None and params.get("grad_inplace"):
            continue

        torch.autograd.gradcheck(m, forward_args)
        torch.autograd.gradcheck(ref_m, ref_forward_args)
    print()


if __name__ == "__main__":
    for cls in [
        AxpyFunction,
        LogFunction,
        SumFunction,
        MaxFunction,
        MatmulFunction,
        LinearFunction,
        TransposeFunction,
        LinearAsCompositionFunction,
        SoftmaxFunction,
        NNLLossFunction,
        MyLinearCrossEntropyFunction,
    ]:
        test_function(cls)

    for cls, ref_cls, dtype, device in [
        (MyLinearCrossEntropyLoss, nn.LinearCrossEntropyLoss, torch.float64, 'cpu'),
        (VoLinearCrossEntropyLoss, nn.LinearCrossEntropyLoss, torch.float32, 'cuda'),
        (LiLinearCrossEntropyLoss, nn.LinearCrossEntropyLoss, torch.float32, 'cuda'),
    ]:
        test_module(cls, ref_cls, device=device, dtype=dtype)
