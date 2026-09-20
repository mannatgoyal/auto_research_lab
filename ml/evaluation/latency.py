"""Statistically rigorous inference latency benchmarking."""

from __future__ import annotations

import gc
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List
import numpy as np
import torch

from ml.config import LatencyConfig
from ml.models.base import BaseModel


@dataclass(frozen=True)
class LatencyBenchmark:
    """Rigorous inference latency profile for a specific batch size."""
    batch_size: int
    mean_latency_ms: float
    median_latency_ms: float  # p50
    p90_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_samples_per_sec: float  # Samples processed per second (batch_size / (mean_latency_ms / 1000))
    throughput_qps: float  # Alias for backward compatibility
    iterations_evaluated: int
    warmup_iterations: int
    device: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def benchmark_inference_latency(
    model: BaseModel,
    sample_features: np.ndarray,
    config: LatencyConfig,
) -> List[LatencyBenchmark]:
    """Measures inference latency across specified batch sizes using repeated microbenchmarking.

    Methodology:
    1. Replicates sample rows to create batches of exact requested sizes.
    2. Runs warmup_iterations to prime instruction cache and CPU/GPU buffers.
    3. Pauses Python GC to prevent memory collection pauses during measurement loops.
    4. Runs benchmark_iterations, capturing nanosecond timestamps with time.perf_counter_ns().
    5. Synchronizes CUDA devices before and after execution when running on GPU.
    6. Computes true statistical percentiles (p50, p90, p95, p99) and mean.
    """
    benchmarks: List[LatencyBenchmark] = []
    num_available_rows = sample_features.shape[0]
    is_cuda = config.device == "cuda" and torch.cuda.is_available()

    for batch_size in config.batch_sizes:
        # Prepare input batch of exact size
        if num_available_rows >= batch_size:
            batch = sample_features[:batch_size].copy()
        else:
            repeats = int(np.ceil(batch_size / num_available_rows))
            batch = np.tile(sample_features, (repeats, 1))[:batch_size].copy()

        # Step 1: Warmup passes (not recorded)
        for _ in range(config.warmup_iterations):
            if is_cuda:
                torch.cuda.synchronize()
            _ = model.predict(batch)
            if is_cuda:
                torch.cuda.synchronize()

        # Step 2: Timed execution passes with GC disabled to isolate compute latency
        latencies_ms: List[float] = []
        gc.collect()
        gc.disable()
        try:
            for _ in range(config.benchmark_iterations):
                if is_cuda:
                    torch.cuda.synchronize()
                t_start = time.perf_counter_ns()
                _ = model.predict(batch)
                if is_cuda:
                    torch.cuda.synchronize()
                t_end = time.perf_counter_ns()
                duration_ms = (t_end - t_start) / 1_000_000.0
                latencies_ms.append(duration_ms)
        finally:
            gc.enable()

        latencies_arr = np.array(latencies_ms, dtype=np.float64)

        mean_ms = float(np.mean(latencies_arr))
        p50_ms = float(np.percentile(latencies_arr, 50))
        p90_ms = float(np.percentile(latencies_arr, 90))
        p95_ms = float(np.percentile(latencies_arr, 95))
        p99_ms = float(np.percentile(latencies_arr, 99))
        min_ms = float(np.min(latencies_arr))
        max_ms = float(np.max(latencies_arr))

        mean_sec = mean_ms / 1000.0
        samples_per_sec = float(batch_size / mean_sec) if mean_sec > 0 else 0.0

        benchmarks.append(
            LatencyBenchmark(
                batch_size=batch_size,
                mean_latency_ms=round(mean_ms, 4),
                median_latency_ms=round(p50_ms, 4),
                p90_latency_ms=round(p90_ms, 4),
                p95_latency_ms=round(p95_ms, 4),
                p99_latency_ms=round(p99_ms, 4),
                min_latency_ms=round(min_ms, 4),
                max_latency_ms=round(max_ms, 4),
                throughput_samples_per_sec=round(samples_per_sec, 2),
                throughput_qps=round(samples_per_sec, 2),
                iterations_evaluated=config.benchmark_iterations,
                warmup_iterations=config.warmup_iterations,
                device=config.device,
            )
        )

    return benchmarks
