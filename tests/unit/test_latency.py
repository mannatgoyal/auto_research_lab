"""Unit tests for inference latency benchmarking."""

import numpy as np
from ml.config import LatencyConfig, ModelConfig
from ml.evaluation.latency import benchmark_inference_latency
from ml.models.base import create_model


def test_latency_benchmark_percentiles():
    np.random.seed(42)
    X = np.random.randn(50, 8).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    model = create_model(ModelConfig(name="logistic_regression")).fit(X, y)
    config = LatencyConfig(
        batch_sizes=[1, 8],
        warmup_iterations=10,
        benchmark_iterations=50,
        device="cpu",
    )

    benchmarks = benchmark_inference_latency(model, X, config)

    assert len(benchmarks) == 2

    for b in benchmarks:
        assert b.iterations_evaluated == 50
        assert b.warmup_iterations == 10
        assert b.throughput_qps > 0.0

        # Percentile ordering invariants
        assert b.min_latency_ms <= b.median_latency_ms
        assert b.median_latency_ms <= b.p90_latency_ms
        assert b.p90_latency_ms <= b.p95_latency_ms
        assert b.p95_latency_ms <= b.p99_latency_ms
        assert b.p99_latency_ms <= b.max_latency_ms


def test_batch_latency_scaling():
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)

    model = create_model(ModelConfig(name="logistic_regression")).fit(X, y)
    config = LatencyConfig(
        batch_sizes=[1, 32],
        warmup_iterations=5,
        benchmark_iterations=30,
        device="cpu",
    )

    benchmarks = benchmark_inference_latency(model, X, config)
    b1 = benchmarks[0]
    b32 = benchmarks[1]

    assert b1.batch_size == 1
    assert b32.batch_size == 32
    # Batch throughput should be substantially higher than single-item throughput
    assert b32.throughput_qps > b1.throughput_qps
