"""
How to derive a backward formula for a tensor operation?
========================================================

See https://github.com/pearu/pearu.github.io/blob/main/torch_autograd_backward.md

"""

import random
import torch, torch.nn as nn
from torch.nn.parameter import Parameter
import math

class AxpyFunction(torch.autograd.Function):
    """output = alpha * x + y 
    """

    @staticmethod
    def samples(device=None, dtype=None):
        for shape in [(3,), (3, 2)]:
            for alpha in [-1.5, 2.5]:
                x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                y = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                yield (alpha, x, y), alpha * x + y
        
    @staticmethod
    def forward(ctx,
                alpha: float,
                x: torch.Tensor,
                y: torch.Tensor):
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
    def forward(ctx,
                x: torch.Tensor,
                y: torch.Tensor):
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
            #result[1] = (grad_output.T @ ctx.saved_tensors[ctx.indices[1]]).T
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
    """output = A @ B.T + C
    """
    @staticmethod
    def samples(device=None, dtype=None):
        for N, M, K in [(3, 2, 4)]:
            x = torch.randn((N, M), device=device, dtype=dtype, requires_grad=True)
            y = torch.randn((K, M), device=device, dtype=dtype, requires_grad=True)
            z = torch.randn((N, K), device=device, dtype=dtype, requires_grad=True)
            yield (x, y, z), x @ y.T + z

    @staticmethod
    def forward(ctx,
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
    ctx1.needs_input_grad = [isinstance(a, torch.Tensor) and a.requires_grad for a in args]
    output = cls.forward(ctx1, *args)
    if hasattr(ctx1, "to_save"):
        ctx1.saved_tensors = ctx1.to_save
        ctx.save_for_backward(*saved, *ctx1.to_save)
    ctx.context_stack.insert(0, (ctx1, cls))
    return output
    
class LinearAsCompositionFunction(torch.autograd.Function):
    """output = A @ B.T + C
    """
    @staticmethod
    def samples(device=None, dtype=None):
        for N, M, K in [(3, 2, 4)]:
            x = torch.randn((N, M), device=device, dtype=dtype, requires_grad=True)
            y = torch.randn((K, M), device=device, dtype=dtype, requires_grad=True)
            z = torch.randn((N, K), device=device, dtype=dtype, requires_grad=True)
            yield (x, y, z), x @ y.T + z

    @staticmethod
    def forward(ctx,
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
    """output = log(x) 
    """
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
    """output = sum(x, dim=, keepdim=)
    """
    @staticmethod
    def samples(device=None, dtype=None):
        for shape in [(3,), (3, 2), (2, 4, 3)]:
            for dim in set([0, len(shape)-1, max(0, len(shape)-2)]):
                for keepdim in [False, True]:
                    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
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
    """output = max(x, dim=, keepdim=)
    """
    @staticmethod
    def samples(device=None, dtype=None):
        for shape in [(3,), (3, 2), (2, 4, 3), (3, 3)]:
            for dim in set([0, len(shape)-1, max(0, len(shape)-2)]):
                for keepdim in [False, True]:
                    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
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
            mask, = ctx.saved_tensors
            if ctx.keepdim:
                result[0] = grad_output * mask
            else:
                result[0] = grad_output.unsqueeze(ctx.dim) * mask
        return tuple(result)

class SoftmaxFunction(torch.autograd.Function):
    """output = softmax(x, dim=)
    """
    @staticmethod
    def samples(device=None, dtype=None):
        for shape in [(3,), (3, 2), (2, 4, 3), (3, 3)]:
            for dim in set([0, len(shape)-1, max(0, len(shape)-2)]):
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
            output, = ctx.saved_tensors
            result[0] = (grad_output - (grad_output * output).sum(dim=ctx.dim, keepdim=True)) * output
        return tuple(result)

class NNLLossFunction(torch.autograd.Function):
    """output = nnl_loss(x, target, weight)
    """
    @staticmethod
    def samples(device=None, dtype=None):
        for reduction in ["sum", "mean"]:
            for shape in [(4, 5)]:
                for ii in set([-100, 0, len(shape)-1, max(0, len(shape)-2)]):
                    x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
                    t = torch.randint(0, shape[1], shape[:-1], device=device, dtype=torch.int64, requires_grad=False)
                    w = torch.exp(torch.randn(shape[1], device=device, dtype=dtype, requires_grad=False))
                    yield (x, t, w, reduction, ii), torch.nn.functional.nll_loss(x, t, weight=w, ignore_index=ii, reduction=reduction)

    @staticmethod
    def forward(ctx, x: torch.Tensor, t: torch.Tensor, weight: torch.Tensor, reduction: str, ignore_index: int):
        ctx.reduction = reduction
        output = torch.nn.functional.nll_loss(x, t, weight=weight, reduction=reduction, ignore_index=ignore_index)
        if x.requires_grad:
            if ignore_index >= 0:
                weight = weight.clone()
                weight[ignore_index] = 0
            wmask = (torch.zeros_like(x)
                     .scatter_(1, t.unsqueeze(1), weight.index_select(0, t).unsqueeze(1)))
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
            wmask, = ctx.saved_tensors
            result[0] = -wmask * grad_output
        return tuple(result)


class LinearCrossEntropyFunction(torch.autograd.Function):
    """output = linear_cross_entropy(x, L, target, weight, bias)
    """
    @staticmethod
    def samples(device=None, dtype=None):
        for reduction in ["sum", "mean"]:
            for num_batches in [8, 2]:
                for in_features in [5, 3]:
                    for num_classes in [4, 6]:
                        for ii in set([-100, 0, min(1, num_classes-1), num_classes-1]):
                            x = torch.randn((num_batches, in_features), device=device, dtype=dtype, requires_grad=True)
                            L = torch.randn((num_classes, in_features), device=device, dtype=dtype, requires_grad=True)
                            t = torch.randint(0, num_classes, (num_batches,), device=device, dtype=torch.int64, requires_grad=False)
                            if ii >= 0 and torch.all(t == ii):
                                t[0] = random.sample(sorted(set(range(num_classes)) - {ii}), 1)[0]
                            w = torch.exp(torch.randn(num_classes, device=device, dtype=dtype, requires_grad=False))
                            yield (x, L, t, w, reduction, ii), torch.nn.functional.linear_cross_entropy(x, L, t, weight=w, ignore_index=ii, reduction=reduction)

    @staticmethod
    def forward(ctx, x: torch.Tensor, L: torch.Tensor, t: torch.Tensor, weight: torch.Tensor, reduction: str, ignore_index: int):
        num_classes = L.shape[0]
        
        # Prepare weight tensors:
        if weight is None:
            weight = torch.ones((num_classes,), device=x.device, dtype=x.dtype, requires_grad=False)
        elif reduction == "mean" or ignore_index >= 0:
            # we'll change weight inplace
            weight = weight.clone()

        if ignore_index >= 0:
            weight[ignore_index] = 0

        weight_t = weight.index_select(0, t)
        if reduction == "mean":
            d = weight_t.sum()
            weight_t.div_(d)
            weight.div_(d)

        # Compute projection that will be transformed to scaled softmax of the projection:
        X = torch.mm(x, L.T)  # projection
        X.sub_(X.max(dim=1, keepdim=True)[0])  # ensure stable softmax

        output = -weight_t.dot(X.gather(1, t.unsqueeze(1)).squeeze(1))  # the first part of the output

        X.exp_()  # switch to S computation
        expXsum = X.sum(dim=1, keepdim=False)
        if x.requires_grad or L.requires_grad:
            X.mul_((weight_t / expXsum).unsqueeze(1))  # X is `S * w`

        expXsum.log_()
        output.add_(weight_t.dot(expXsum))  # add the remaining part of the output

        if x.requires_grad or L.requires_grad:
            ctx.save_for_backward(X, x, L, t, weight, weight_t)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        result = [None] * 6
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            X, x, L, t, weight, weight_t = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_x = L.index_select(0, t)
            grad_x.mul_(weight_t.unsqueeze(1))
            torch.addmm(grad_x, X, L, alpha=-1, out=grad_x)
            grad_x.mul_(-grad_output)
            result[0] = grad_x

        if ctx.needs_input_grad[1]:
            grad_L = torch.zeros_like(L).scatter_reduce_(
                0,
                t.unsqueeze(1).expand(x.shape),
                x,
                'sum',
                include_self=False)
            grad_L.mul_(weight.unsqueeze(1))
            torch.addmm(grad_L, X.T, x, alpha=-1, out=grad_L)
            grad_L.mul_(-grad_output)
            result[1] = grad_L
        return tuple(result)


class MyLinearCrossEntropyLoss(nn.Module):
    def __init__(self, dim: int, n_classes: int,
                 ignore_index: int = -100,
                 reduction: str = "mean",
                 label_smoothing: float = 0.0,
                 bias: bool = False,
                 device=None,
                 dtype=None,
                 num_chunks=1):
        super().__init__()
        self.num_chunks = num_chunks
        assert reduction in {"sum", "mean"}, reduction
        assert label_smoothing == 0.0
        assert not bias
        self.proj = nn.Linear(dim, n_classes, bias=False, device=device, dtype=dtype)
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)

    def forward(self, x, targ):
        assert targ.dtype == torch.int64
        return LinearCrossEntropyFunction.apply(x, self.proj.weight, targ, self.loss.weight, self.loss.reduction, self.loss.ignore_index)

    @staticmethod
    def samples():
        device = "cuda"
        dtype = torch.float64
        for reduction in ["sum", "mean"]:
            for num_batches in [8]:
                for in_features in [5, 3]:
                    for num_classes in [4, 6]:
                        for ii in set([-100, 0, min(1, num_classes-1), num_classes-1]):
                            input = torch.randn((num_batches, in_features), device=device, dtype=dtype, requires_grad=True)
                            target = torch.randint(0, num_classes, (num_batches,), device=device, dtype=torch.int64, requires_grad=False)
                            module_args, module_kwargs, forward_args = (in_features, num_classes), dict(ignore_index=ii, reduction=reduction, label_smoothing=0.0, bias=False, device=device, dtype=dtype), (input, target)
                            yield module_args, module_kwargs, forward_args

def test_function(cls):
    torch.manual_seed(5431)
    print(f'{cls.__name__}')
    f = cls.apply
    for args, expected in cls.samples(device='cuda', dtype=torch.float64):
        torch.testing.assert_close(f(*args), expected)
        torch.autograd.gradcheck(f, args)


def test_module(cls, ref_cls):
    print(f'{cls.__name__} {ref_cls.__name__}')

    for module_args, module_kwargs, forward_args in cls.samples():
        torch.manual_seed(5431)
        m = cls(*module_args, **module_kwargs)
        torch.manual_seed(5431)
        ref_m = ref_cls(*module_args, **module_kwargs)

        ref_forward_args = [a.detach().clone().requires_grad_(a.requires_grad) for a in forward_args]
        
        l = m(*forward_args)
        ref_l = ref_m(*ref_forward_args)
        torch.testing.assert_close(l, ref_l)

        l.sum().backward()
        ref_l.sum().backward()
        torch.testing.assert_close(forward_args[0].grad, ref_forward_args[0].grad)

        torch.autograd.gradcheck(m, forward_args)
        torch.autograd.gradcheck(ref_m, ref_forward_args)
        
if __name__ == "__main__":
    for cls in [AxpyFunction, LogFunction, SumFunction, MaxFunction, MatmulFunction, LinearFunction, TransposeFunction,
                LinearAsCompositionFunction, SoftmaxFunction, NNLLossFunction, LinearCrossEntropyFunction][-1:]:
        test_function(cls)

    for (cls, ref_cls) in [(MyLinearCrossEntropyLoss, nn.LinearCrossEntropyLoss)]:
        test_module(cls, ref_cls)
