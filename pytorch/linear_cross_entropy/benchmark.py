from copy import copy
from itertools import product
from pathlib import Path
import time
import os
import resource
import multiprocessing as mp
import torch
from torch import nn


import efficient_cross_entropy_modules


class FusedProjectionPlusCrossEntropyLoss(nn.Module):
    # Wraps the class from https://github.com/mgmalek/efficient_cross_entropy/blob/main/modules.py
    def __init__(self, dim: int, n_classes: int, **kwargs):
        super().__init__()
        dtype = kwargs.pop('dtype')
        device = kwargs.pop('device')
        assert kwargs.pop('bias') == False
        assert kwargs.pop('label_smoothing') == 0.0
        module = efficient_cross_entropy_modules.FusedProjectionPlusCrossEntropyLoss(dim, n_classes, **kwargs)
        if device == "cuda":
            module = module.cuda()
        if dtype is not None:
            module = module.to(dtype)
        self.module = module

    def forward(self, x, targ):
        assert targ.dtype == torch.int64, targ.dtype
        return self.module(x, targ)

        
class PyTorchProjectionPlusCrossEntropyLoss(nn.Module):
    # Same as https://github.com/mgmalek/efficient_cross_entropy/blob/main/modules.py
    # Used as a reference to LinearCrossEntropyLoss benchmarks.
    # Not implemented features: loss_dims, weight

    def __init__(self, dim: int, n_classes: int,
                 ignore_index: int = -100,
                 reduction: str = "mean",
                 label_smoothing: float = 0.0,
                 bias: bool = True,
                 device=None,
                 dtype=None,):
        super().__init__()
        self.proj = nn.Linear(dim, n_classes, bias=bias, device=device, dtype=dtype)
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, label_smoothing=label_smoothing)

    def forward(self, x, targ):
        logits = self.proj(x)
        return self.loss(logits, targ)

config_defaults = dict(num_classes=[32768, 8192][1], num_tokens=8192, in_features=2048)

def configs():
    params = dict(
        dtype = [torch.float32],
        token_dtype = [torch.long, torch.float32][1:],
        device = ["cuda", "cpu"][:1],
        in_features = [1024, 2048, 4096, 8192],
        num_classes = [2048, 4096, 8192, 16384, 32768, 65536],
        num_tokens = [1024, 2048, 4096, 8192, 16384],
        bias = [False],
        reduction = ["mean", "sum"],
        ignore_index = [-100],
        label_smoothing = [0.0],
        module_extra_kwargs = [
            (PyTorchProjectionPlusCrossEntropyLoss, {}),
            (nn.LinearCrossEntropyLoss, {}),
            (FusedProjectionPlusCrossEntropyLoss, dict(n_loop_iters=1)),
            (FusedProjectionPlusCrossEntropyLoss, dict(n_loop_iters=2)),
            (FusedProjectionPlusCrossEntropyLoss, dict(n_loop_iters=4)),
            (FusedProjectionPlusCrossEntropyLoss, dict(n_loop_iters=8)),
        ],
        )

    for field in config_defaults:
        config_ = dict((k, (params[k] if field==k else config_defaults[k])) for k in config_defaults)
        names, values_lst = zip(*[(name, value) for name, value in params.items() if name not in config_])
        for values in product(*values_lst):
            config = copy(config_)
            for i, name in enumerate(names):
                config[name] = values[i]
            key = [k for k, v in config.items() if isinstance(v, list)][0]
            for value in config[key]:
                kwargs = copy(config)
                cls, extra_kwargs = kwargs.pop("module_extra_kwargs")
                num_tokens = kwargs.pop("num_tokens")
                token_dtype = kwargs.pop("token_dtype")
                if key == "num_tokens":
                    num_tokens = value
                elif key == "token_dtype":
                    token_dtype = value
                else:
                    kwargs[key] = value
                label = ' '.join([f'{k}={v}' for k, v in extra_kwargs.items()])
                label = f"{cls.__name__}[{label}]" if label else cls.__name__
                if issubclass(cls, FusedProjectionPlusCrossEntropyLoss) and token_dtype != torch.int64:
                    continue
                yield label, cls, kwargs, extra_kwargs, num_tokens, token_dtype


def measure(queue, cls, args, kwargs, num_tokens, token_dtype):

    torch.manual_seed(321)

    device = torch.device(kwargs["device"])
    cpu_initial_peak_mem_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1e3
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        initial_peak_mem_bytes = torch.cuda.max_memory_allocated(device=device)
    elif device.type == "cpu":
        initial_peak_mem_bytes = 0
    else:
        assert 0, device.type

    in_features, num_classes = args

    message = ''
    try:
        features = torch.randn((num_tokens, in_features), device=device, dtype=kwargs["dtype"], requires_grad=True)
        if token_dtype.is_floating_point:
            tokens = torch.randn((num_tokens, num_classes), device=device, dtype=token_dtype).softmax(dim=1)
        else:
            tokens = torch.randint(0, num_classes, (num_tokens,), device=device, dtype=token_dtype)

        module = cls(*args, **kwargs)
        loss = module(features, tokens).mean()
        _ = loss.backward()
        loss = loss.detach().item()
    except torch.OutOfMemoryError as msg:
        message = str(msg)
        loss = None
        print(f'{message=}')
        
    if device.type == "cuda":
        final_peak_mem_bytes = torch.cuda.max_memory_allocated(device=device)
    elif device.type == "cpu":
        final_peak_mem_bytes = 0
    else:
        assert 0, device.type

    cpu_final_peak_mem_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1e3
    repeat = 10
    
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        if not message:
            lst = []  # avoid gc until cuda sync is done
            for _ in range(repeat):
                try:
                    lst.append(module(features, tokens).mean().backward())
                except torch.OutOfMemoryError as msg:
                    repeat = max(1, len(lst))
                    break

        end.record()
        torch.cuda.synchronize()
        time_ms = start.elapsed_time(end) / repeat   
    elif device.type == "cpu":
        start = time.process_time()
        if not message:
            for _ in range(repeat):
                _ = module(features, tokens).mean().backward()
        end = time.process_time()
        time_ms = (end - start) * 1e3 / repeat
    else:
        assert 0, device.type

    queue.put(dict(
        message=message,
        cpu_mem_gb = (cpu_final_peak_mem_bytes - cpu_initial_peak_mem_bytes) / 1e9,
        cuda_mem_gb = (final_peak_mem_bytes - initial_peak_mem_bytes) / 1e9,
        time_ms = time_ms,
        loss = loss))

measurement_fields = ["cpu_mem_gb", "cuda_mem_gb", "time_ms", "loss", "message"]
string_fields = ["gpu_name", "label", "message", "device", "reduction", "device_name"]
int_fields = ["in_features", "num_classes", "ignore_index", ""]
float_fields = ["label_smoothing", "cpu_mem_gb", "cuda_mem_gb", "time_ms", "loss"]
bool_fields = ["bias"]

def unserialize(fieldname, rawvalue):
    if fieldname in string_fields:
        return rawvalue
    if isinstance(rawvalue, str) and rawvalue == "None":
        return None
    if fieldname in int_fields:
        return int(rawvalue)
    if fieldname in float_fields:
        return float(rawvalue)
    if fieldname in bool_fields:
        return {"False": False, "True": True}[rawvalue]
    if ":" not in rawvalue:
        return rawvalue
    typename, value = rawvalue.split(":", 1)
    if typename == "float":
        return float(value)
    elif typename == "int":
        return int(value)
    elif typename == "bool":
        return {"False": False, "True": True}[value]
    if typename == "dtype":
        assert value.startswith("torch."), value
        return getattr(torch, value.split(".", 1)[1])
    print(f'TODO: unserialize({fieldname}, {value})')
    return value


def serialize(fieldname, value):
    if fieldname in string_fields:
        assert isinstance(value, str), (type(value), value)
        return value
    if fieldname in int_fields + float_fields + bool_fields:
        return str(value)
    return f"{type(value).__name__}:{value}"


def load_measurements(path):
    data = []
    if path.exists():
        import csv
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                row = dict([(k, unserialize(k, v)) for k, v in row.items()])
                data.append(row)
    return data


def get_fieldnames(data):
    fieldnames = None
    for row in data:
        if fieldnames is None:
            fieldnames = list(row)
        else:
            for name in row:
                if name not in fieldnames:
                    fieldnames.append(name)    
    return fieldnames

def save_measurements(path, data):
    import csv
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=get_fieldnames(data), delimiter=';')
        writer.writeheader()
        for row in data:
            writer.writerow(dict([(k, serialize(k, v)) for k, v in row.items()]))


def filter_measurements(data, match):
    result = []
    for row in data:
        for k, v in match.items():
            if row.get(k) != v:
                break
        else:
            result.append(row)
    return result


def select_measurements(data, match):
    result = []
    for row in data:
        for k, v in match.items():
            if row.get(k) != v:
                break
        else:
            result.append(row)
    return result


def compute(data):
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties()
        gpu_name = f"{p.name}-CC{p.major}.{p.minor}-{p.total_memory // 2**20}MB"
    else:
        gpu_name = 'N/A'
    p = open('/proc/cpuinfo').read()
    cpu_name = p[p.index("model name"):].split("\n", 1)[0].split(":")[1].strip()
    print(f'{gpu_name=} {cpu_name=}')
    try:
        for label, cls, kwargs, extra_kwargs, num_tokens, token_dtype in configs():
            device_name = dict(cpu=cpu_name, cuda=gpu_name)[kwargs["device"]]
            match_dct = dict(device_name=device_name, label=label, num_tokens=num_tokens, token_dtype=token_dtype, **kwargs)
            row = filter_measurements(data, match_dct)
            if row:
                assert len(row) == 1
                continue

            args = (kwargs.pop("in_features"), kwargs.pop("num_classes"))
            kwargs = dict(kwargs, **extra_kwargs)

            q = mp.Queue()
            # executing measure in mp.Process enables correct CPU memory
            # usage measurement
            p = mp.Process(target=measure, args=(q, cls, args, kwargs, num_tokens, token_dtype))
            p.start()
            r = q.get()
            p.join()
            cpu_mem_gb, cuda_mem_gb, time_ms, message = r['cpu_mem_gb'], r['cuda_mem_gb'], r['time_ms'], r['message']

            print(f'{cls.__name__}[{kwargs["device"]}], {args=} {num_tokens=} {cpu_mem_gb, cuda_mem_gb, time_ms=} loss={r["loss"]}')

            if message:
                print(message)
            data.append(dict(match_dct, **r))
    except KeyboardInterrupt:
        pass
    return data


def get_values(data, field):
    return list(set([row[field] for row in data if field in row]))


def get_series(data, field, ignore=['device_name', 'device']):
    fieldnames = get_fieldnames(data)
    field_values = dict()
    for f in fieldnames:
        if f == field or f in measurement_fields:
            continue
        if f not in field_values:
            field_values[f] = set()
        field_values[f].update(get_values(data, f))

    result = []
    
    for p in product(*[tuple((k, v) for v in s) for k, s in field_values.items()]):
        m = filter_measurements(data, dict(p))
        if len(m) > 1:
            x, m = list(zip(*sorted([[row[field], row] for row in m])))
            x = list(x)
            values = dict()
            params = dict()
            for k in measurement_fields:
                values[k] = [r[k] for r in m]
            for k in fieldnames:
                if not (k == field or k in measurement_fields):
                    v = get_values(m, k)
                    assert len(v) == 1, v
                    params[k] = v[0]
            result.append((params, (field, x), values))

    series_lst = []
    series_params = dict()
    for p, x, y in result:
        for k, v in p.items():
            if k not in series_params:
                series_params[k] = set()
            series_params[k].add(v)
        series_lst.append((p, x, y))

    common_params = dict([(k, list(v)[0]) for k, v in series_params.items() if len(v) == 1])
    series = dict()
    for p, x, y in series_lst:
        label = tolabel(p, ignore=common_params)
        series[label] = (x, y)
    return common_params, series


def tolabel(data, ignore=[]):
    defaults = dict(ignore_index=-100, label_smoothing=0.0)
    names = ['label', 'dtype', 'token_dtype', 'features_in', 'num_classes', 'num_tokens', 'bias', 'reduction', 'ignore_index', 'label_smoothing']
    lst = []

    def tostr(n, v):
        if n in ['dtype', 'token_dtype']:
            return str(v).split('.', 1)[-1]
        elif n == "label":
            return v
        else:
            return f'{n}={v}'

    for n in names:
        if n not in data or n in ignore:
            continue
        elif n in defaults and data[n] == defaults[n]:
            continue
        else:
            lst.append(tostr(n, data[n]))
    for n in data:
        if n in names or n in ignore:
            continue
        elif n in defaults and data[n] == defaults[n]:
            continue
        else:
            lst.append(tostr(n, data[n]))

    return ' '.join(lst)


def toxlabel(field):
    if field in ["cuda_mem_gb", "cpu_mem_gb"]:
        return "Peak Memory Usage (GB)"
    if field == "time_ms":
        return "Average Timing (ms)"
    return field
    
def plot(data, plot_params):
    import matplotlib.pyplot as plt

    paths = []
    root_path = Path('images')
    root_path.mkdir(exist_ok=True, parents=True)
    
    for device_name in get_values(data, "device_name"):
        print(f'{device_name=}')
        device_data = filter_measurements(data, dict(device_name=device_name, **plot_params))
        if not device_data:
            continue
        device = get_values(device_data, "device")
        assert len(device) == 1, device
        device = device[0]

        xfields = ["in_features", "num_classes", "num_tokens"]
        xfields = list(config_defaults)
        xfields_complementary = dict((k, dict((k1, config_defaults[k1]) for k1 in xfields if k1 != k)) for k in xfields)
        yfields = [dict(cuda="cuda_mem_gb", cpu="cpu_mem_gb")[device], "time_ms"]

        all_series = []
        common_params = dict()
        existing_xfields = []
        for xfield in xfields:
            params, series = get_series(filter_measurements(device_data, xfields_complementary[xfield]), xfield)
            for k, v in params.items():
                if k not in common_params:
                    common_params[k] = set()
                common_params[k].add(v)

            if series:
                all_series.append((params, series))
                existing_xfields.append(xfield)
        xfields = existing_xfields

        common_params = dict([(k, list(v)[0]) for k, v in common_params.items() if len(v) == 1 and k not in xfields])
        suptitle = tolabel(common_params, ignore=['device_name', 'device', 'token_dtype'])

        fig, axes = plt.subplots(len(yfields), len(xfields), figsize=(24, 12), dpi=300)
        
        for xi, (params, series) in enumerate(all_series):
            title = tolabel(params, ignore=common_params)
            for yi, yfield in enumerate(yfields):
                ax = axes[yi][xi]
                for label, ((xfield, x), ydata) in sorted(series.items()):
                    y = ydata[yfield]
                    x, y = zip(*[(x_, y_) for x_, y_, l in zip(x, y, ydata["loss"]) if l is not None])
                    ax.plot(x, y, "--o", label=label)
                ax.set_xlabel(toxlabel(xfield))
                ax.set_ylabel(toxlabel(yfield))
                ax.set_title(title)
                ax.legend(loc="upper left")

        suffix = ['']
        if 'reduction' in common_params:
            suffix.append(common_params['reduction'])
        if 'token_dtype' in common_params:
            if common_params['token_dtype'].is_floating_point:
                suffix.append('probabilities')
                suptitle = f'{suptitle}\nprobability targets'
            else:
                suffix.append('indices')
                suptitle = f'{suptitle}\nclass indices targets'
        suffix = '-'.join(suffix)
        
        filename = device_name.replace(' ', '_') + f'{suffix}.png'
        path = root_path / filename
        paths.append(path)

        fig.suptitle(f'{device_name}\n{suptitle}')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    return paths

def main():
    measurements_path = Path('measurements.csv')
    data = load_measurements(measurements_path)
    if 1:
        data = compute(data)
        save_measurements(measurements_path, data)
    plot(data, dict(reduction="mean", token_dtype=torch.float32))
    plot(data, dict(reduction="mean", token_dtype=torch.int64))


if __name__ == "__main__":
    main()
