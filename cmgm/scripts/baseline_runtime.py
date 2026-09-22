"""Forward-only wall-clock benchmark, separate from selection and evaluation."""
import platform
import time
import numpy as np
import torch

RUNTIME_PROTOCOL = dict(
    batch_size=64, warmup_batches=3, repeats=10, split='full TEST origins, including final partial batch',
    mode='eval + inference_mode, restored best checkpoint',
    includes='model forward including its input adapter and CPU dispatch',
    excludes='data preprocessing/loading, host-to-device transfer, checkpoint loading, metrics and file I/O',
    synchronization='CUDA synchronize before and after each complete pass',
    latency='amortized milliseconds per origin at batch64; NOT single-sample latency',
)


def environment(device):
    device = torch.device(device)
    return dict(device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor() or platform.machine(),
                host=platform.node(), platform=platform.platform(), torch_version=str(torch.__version__),
                cuda_version=torch.version.cuda, threads=torch.get_num_threads(),
                interop_threads=torch.get_num_interop_threads(), dtype='float32',
                cudnn_version=torch.backends.cudnn.version(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)


def measure_inference(model, loader, device):
    """Never consumes labels, computes metrics, or changes learned parameters."""
    device = torch.device(device)
    modes = [(module, module.training) for module in model.modules()]
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    try:
        # Includes the loader generator: timing must not advance training RNG state.
        generator = getattr(loader, 'generator', None)
        generator_state = generator.get_state() if generator is not None else None
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            model.eval()
            batches = [batch[0].to(device=device, dtype=torch.float32) for batch in loader]
            if not batches or sum(len(x) for x in batches) != len(loader.dataset):
                raise ValueError('Runtime benchmark requires every origin, drop_last=False')
            if loader.batch_size != RUNTIME_PROTOCOL['batch_size']:
                raise ValueError('Runtime benchmark requires batch_size=64')
            for i in range(RUNTIME_PROTOCOL['warmup_batches']):
                pred = model(batches[i % len(batches)])
                if not torch.isfinite(pred).all():
                    raise ValueError('Nonfinite runtime warmup prediction')
            seconds = []
            for _ in range(RUNTIME_PROTOCOL['repeats']):
                sync()
                start = time.perf_counter()
                for x in batches:
                    pred = model(x)
                sync()
                seconds.append(time.perf_counter() - start)
            count = sum(len(x) for x in batches)
            mean = float(np.mean(seconds))
            return dict(protocol=RUNTIME_PROTOCOL, environment=environment(device), origins=count,
                        batch_sizes=[len(x) for x in batches], pass_seconds=seconds,
                        test_forward_seconds_mean=mean, test_forward_seconds_std=float(np.std(seconds)),
                        test_forward_seconds_median=float(np.median(seconds)),
                        inference_ms_per_origin=1000 * mean / count,
                        origins_per_second=count / mean)
    finally:
        if 'generator_state' in locals() and generator_state is not None:
            generator.set_state(generator_state)
        for module, training in modes:
            module.training = training
