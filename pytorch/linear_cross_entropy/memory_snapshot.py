import torch
import math
from torch import nn


from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss


class LigerLinearCrossEntropyLoss(nn.Module):

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            *,
            out_features: tuple[int, ...] = (),
            device=None,
            dtype=None,
            reduction: str = "mean",
            weight: Tensor | None = None,
            ignore_index: int = -100,
            label_smoothing: float = 0.0,
    ):
        super().__init__()
        assert out_features == (), out_features
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
        self.projection = nn.Parameter(
            torch.empty((num_classes, in_features), device=device, dtype=dtype)
        )
        nn.init.kaiming_uniform_(self.projection, a=math.sqrt(5))

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.module(self.projection, input, target)


def make_snapshot(cls,
                  num_batches,
                  in_features,
                  num_classes,
                  reduction: str = "mean",
                  ignore_index: int = -100,
                  **kwargs
                  ):
    use_profile = kwargs.pop("use_profile", False)
    label = f'{cls.__name__}_{num_batches}x{in_features}x{num_classes}_{reduction}'
    if ignore_index >= 0:
        label += f'_ii{ignore_index}'
    extra_label_fmt = kwargs.pop("extra_label_fmt", None)
    if extra_label_fmt is not None:
        label += '_' + eval('f'+repr(extra_label_fmt), {}, kwargs)
    print(label)

    torch.manual_seed(5431)
    device = torch.device("cuda")
    dtype = torch.float32
    target_dtype = torch.int32

    target = torch.randint(
        0,
        num_classes,
        (num_batches,),
        device=device,
        dtype=torch.int64,
        requires_grad=False,
    )
    if ignore_index >= 0 and torch.all(target == ignore_index):
        target[0] = random.sample(sorted(set(range(num_classes)) - {ignore_index}), 1)[0]

    weight = torch.exp(torch.randn(in_features, device=device, dtype=dtype, requires_grad=False))

    module = cls(
        in_features, num_classes,
        device=device, dtype=dtype,
        reduction=reduction, weight=weight,
        ignore_index=ignore_index,
        **kwargs
    )
    # warm up
    for _ in range(10):
        input = torch.randn(
            (num_batches, in_features),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        module(input, target).backward()

    input = torch.randn(
        (num_batches, in_features),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    torch.cuda.memory._record_memory_history(device=device, enabled="all")
    l1 = module(input, target)
    l1.backward()
    snapshot = torch.cuda.memory._snapshot(device=device)
    torch.cuda.memory._record_memory_history(enabled=None)
    del l1

    if use_profile:
        from torch.profiler import profile, ProfilerActivity, record_function

        input = torch.randn(
            (num_batches, in_features),
            device=device,
            dtype=dtype,
            requires_grad=True,
        )

        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True,
                acc_events=True,
        ) as prof:
            l2 = module(input, target)
            l2.backward()
        prof.export_chrome_trace(f"{label}_trace.json")

    return label, snapshot
    
if __name__ == "__main__":
    import torch.cuda._memory_viz as memory_viz
    import pickle
    import json
    import base64
    
    args = dict(
        num_batches=8192 * 4,
        in_features=8192,
        num_classes=8192,
        ignore_index=-100,
    )
    plot_segments = False
    filter_freed = False

    local_files = []
    for cls, kwargs in [
            (nn.LinearCrossEntropyLoss, dict(options=None, extra_label_fmt="native", **args)),
            (nn.LinearCrossEntropyLoss, dict(options=nn.functional.LinearCrossEntropyOptions(grad_inplace=True),
                                             extra_label_fmt='{"gradinplace" if options.grad_inplace else ""}_batch_csz{options.batch_chunk_size}',
                                             use_profile=not True,
                                             **args)),
            *((nn.LinearCrossEntropyLoss,
               dict(options=nn.functional.LinearCrossEntropyOptions(grad_inplace=True, batch_chunk_size=1024 * k),
                    extra_label_fmt='{"gradinplace" if options.grad_inplace else ""}_batch_csz{options.batch_chunk_size}',
                    use_profile=not True,
                    **args)) for k in [1, 2]),
            (nn.LinearCrossEntropyLoss,
             dict(options=nn.functional.LinearCrossEntropyOptions(grad_inplace=False, batch_chunk_size=1024 * 1),
                  extra_label_fmt='{"gradinplace" if options.grad_inplace else ""}_batch_csz{options.batch_chunk_size}', **args)),
            (nn.LinearCrossEntropyLoss,
             dict(options=nn.functional.LinearCrossEntropyOptions(grad_inplace=False, chunking_method="liger"),
                  use_profile=not True,
                  extra_label_fmt='{"gradinplace" if options.grad_inplace else ""}_batch_csz{options.batch_chunk_size or options.chunking_method}', **args)),
            (nn.LinearCrossEntropyLoss,
             dict(options=nn.functional.LinearCrossEntropyOptions(grad_inplace=True, chunking_method="liger"),
                  use_profile=True,
                  extra_label_fmt='{"gradinplace" if options.grad_inplace else ""}_batch_csz{options.batch_chunk_size or options.chunking_method}', **args)),
            (LigerLinearCrossEntropyLoss, dict(use_profile=True, **args)),
    ][-3:]:
        label, snapshot = make_snapshot(cls, **kwargs)
        if filter_freed:
            snapshot = memory_viz.filter_alloc_free_pairs(snapshot)
        buffer = pickle.dumps(snapshot, protocol=4)
        buffer += b"\x00" * (3 - len(buffer) % 3)
        encoded_buffer = base64.b64encode(buffer).decode("utf-8")
        local_files.append(dict(name=label, base64=encoded_buffer))

    json_format = json.dumps(local_files)
    viz_kind = "Active Memory Timeline" if not plot_segments else "Active Cached Memory Timeline"
    html = memory_viz._memory_viz_template.replace("$VIZ_KIND", repr(viz_kind)).replace(
        "$SNAPSHOT", json_format
    )

    with open(f'memory_snapshot.html', 'w') as f:
        f.write(html)

    print(f'Created memory_snapshot.html')
