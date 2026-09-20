# Autonomous ML Research Lab: Phase 1, Phase 2 & Phase 3 Walkthrough

This document records the architecture, audit hardening, implementation, and empirical verification for:
1. **Phase 1: Deterministic ML Research Engine**
2. **Phase 2: MLflow Experiment Tracking & Cryptographic Lineage**
3. **Phase 3: Model Serving & Production Inference**

---

## 1. Architectural Principles & Invariants

```mermaid
graph TD
    subgraph "Deterministic ML Research Engine (Zero MLflow / Zero LLM)"
        Config[ExperimentConfig YAML] --> Runner[execute_experiment / run_experiment]
        Runner --> Loader[Dataset Loader & SHA-256 Fingerprinter]
        Loader --> Splitter[Deterministic Stratified Splitter]
        Splitter --> Pipeline[Leak-Free Preprocessing Pipeline]
        Pipeline --> Candidates[Candidate Models: LR, RF, XGB, Torch MLP]
        Candidates --> Eval[Evaluation & Microsecond Latency Profiler]
        Eval --> ExecBundle["ExperimentExecution Container<br/>(result: ExperimentResult, fitted_models: dict, pipeline: Preprocessor)"]
    end

    subgraph "Tracking & Lineage Store (MLflow Observer)"
        ExecBundle --> Tracker["MLflowTracker (sqlite:///mlflow.db)"]
        Tracker --> ParentRun["Parent MLflow Run<br/>├── preprocessing/pipeline.joblib<br/>├── dataset_metadata.json<br/>├── split_metadata.json<br/>└── lineage hashes"]
        ParentRun --> ChildRuns["Child MLflow Runs (x4)<br/>├── model/model.joblib / model.json / model_weights.pt<br/>├── model_metadata.json<br/>└── parent_run_id link"]
    end

    subgraph "Production Inference Serving (FastAPI)"
        Client[HTTP Client] --> API[FastAPI Server: api/server.py]
        API --> Schema[Pydantic Validation: api/schemas.py]
        Schema --> Service[InferenceService: api/service.py]
        Service --> MLoader[ModelLoader: api/loader.py]
        MLoader -->|Resolve parent_run_id| ParentRun
        MLoader -->|Load model artifact| ChildRuns
        Service --> InfPipeline["InferencePipeline<br/>├── Exact Persisted Preprocessor<br/>└── Trained Model Estimator"]
        InfPipeline --> Resp[POST /predict Response]
    end
```

### Invariant 1: JSON Safety of `ExperimentResult` / `ModelResult`
- `ModelResult` and `ExperimentResult` remain pure, serializable Pydantic data schemas containing numerical evidence, hashes, and confusion matrices.
- Live in-memory fitted model instances and the fitted transformer are encapsulated inside the `ExperimentExecution` container.
- Fitted model objects **never** leak into JSON serialization, MCP responses, or experiment result artifacts.

### Invariant 2: Genuinely Optional MLflow Tracking
- The core deterministic ML engine has **zero imports** of `mlflow`.
- MLflow tracking is encapsulated entirely in `ml/tracking/mlflow_tracker.py`.
- Running `python -m ml.cli --config <config> --no-tracking` executes the entire pipeline with zero tracking overhead.

### Invariant 3: Preprocessing Artifact Ownership & Mathematical Equivalence
- The fitted preprocessing pipeline is experiment-level state because all candidate models share the same training transformations.
- It is saved **only** in the Parent Run (`preprocessing/pipeline.joblib`).
- The `ModelLoader` traverses `child_run.tags["parent_run_id"]` to retrieve the exact fitted preprocessor, guaranteeing 100% mathematical equivalence between direct Phase 1 inference and the serving API.

---

## 2. Phase 3 Implementation Details

### A. Strict Input Schema (`api/schemas.py`)
- Defines `BreastCancerFeatures` representing all 30 features from the UCI Breast Cancer Wisconsin dataset.
- Enforces strict constraints:
  - `model_config = ConfigDict(extra="forbid", populate_by_name=True)`
  - Rejects missing fields with HTTP 422.
  - Rejects extra/unexpected fields with HTTP 422.
  - Rejects non-numeric types with HTTP 422.
  - Rejects `NaN`, `Inf`, and `-Inf` with HTTP 422 (`validate_finite_number` field validator).
  - Converts cleanly to a single-row `pd.DataFrame` matching training feature ordering via `to_dataframe()`.
- Supports both nested (`{"features": { ... }}`) and flat (`{ ... }`) JSON request bodies via `PredictionRequest`.

### B. Model & Preprocessor Loader (`api/loader.py`)
- `ModelLoader`:
  1. Inspects configured child `run_id` in MLflow.
  2. Resolves `parent_run_id = child_run.data.tags["parent_run_id"]`.
  3. Downloads and loads `preprocessing/pipeline.joblib` from the parent run.
  4. Loads the candidate model artifact (`model/model.joblib`, `model.json`, or `model_weights.pt`).
  5. Extracts lineage metadata (dataset fingerprint, config hash, git commit, expected feature names).
  6. Returns an immutable `InferencePipeline`.

### C. Inference Service (`api/service.py`)
- Manages active pipeline state and health checks.
- Server-side Latency Profiling: Measures request execution time in milliseconds using `time.perf_counter()`.
- Error isolation: Distinguishes between `ModelNotLoadedError` (maps to HTTP 503) and `InferenceExecutionError` (maps to HTTP 500) without exposing Python stack traces.

### D. FastAPI Application (`api/server.py`)
- Exposes:
  - `GET /health`: Returns service health, model load status, model name, and child run ID (HTTP 503 if unloaded).
  - `GET /v1/models/current`: Returns audit lineage metadata and expected feature names.
  - `POST /predict`: Performs preprocessor transformation and model inference, returning predicted class (0 or 1), semantic label ("malignant" or "benign"), class probabilities, model identity, and server latency.

---

## 3. Verification & Test Results

### Complete Test Suite (`pytest -v`)
All **56 unit and integration tests passed in 86.15 seconds**:

```text
tests/integration/test_experiment_pipeline.py::test_full_pipeline_execution PASSED [  1%]
tests/integration/test_experiment_pipeline.py::test_pipeline_reproducibility PASSED [  3%]
tests/integration/test_lineage_invariant.py::test_lineage_invariant_end_to_end PASSED [  5%]
tests/integration/test_serving_api.py::test_health_endpoint_when_loaded PASSED [  7%]
tests/integration/test_serving_api.py::test_health_endpoint_when_unloaded PASSED [  8%]
tests/integration/test_serving_api.py::test_model_metadata_endpoint PASSED [ 10%]
tests/integration/test_serving_api.py::test_predict_valid_request PASSED [ 12%]
tests/integration/test_serving_api.py::test_predict_flat_dictionary_support PASSED [ 14%]
tests/integration/test_serving_api.py::test_predict_missing_feature_rejected PASSED [ 16%]
tests/integration/test_serving_api.py::test_predict_unexpected_field_rejected PASSED [ 17%]
tests/integration/test_serving_api.py::test_predict_nan_rejected PASSED  [ 19%]
tests/integration/test_serving_api.py::test_predict_when_model_unloaded_returns_503 PASSED [ 21%]
tests/integration/test_serving_api.py::test_mathematical_equivalence_direct_vs_api PASSED [ 23%]
tests/unit/test_api_schemas.py::test_valid_30_features PASSED            [ 25%]
tests/unit/test_api_schemas.py::test_missing_feature_rejected PASSED     [ 26%]
tests/unit/test_api_schemas.py::test_extra_unexpected_field_rejected PASSED [ 28%]
tests/unit/test_api_schemas.py::test_invalid_type_rejected PASSED        [ 30%]
tests/unit/test_api_schemas.py::test_nan_value_rejected PASSED           [ 32%]
tests/unit/test_api_schemas.py::test_inf_value_rejected PASSED           [ 33%]
tests/unit/test_api_schemas.py::test_prediction_request_supports_flat_and_nested PASSED [ 35%]
tests/unit/test_config.py::test_valid_yaml_parsing PASSED                [ 37%]
tests/unit/test_config.py::test_wrapped_experiment_key PASSED            [ 39%]
tests/unit/test_config.py::test_invalid_split_sizes_raise_error PASSED   [ 41%]
tests/unit/test_config.py::test_empty_models_raise_error PASSED          [ 42%]
tests/unit/test_dataset_loader.py::test_load_breast_cancer PASSED        [ 44%]
tests/unit/test_dataset_loader.py::test_fingerprint_reproducibility PASSED [ 46%]
tests/unit/test_dataset_loader.py::test_profile_dataset PASSED           [ 48%]
tests/unit/test_dataset_loader.py::test_invalid_dataset_name_raises PASSED [ 50%]
tests/unit/test_latency.py::test_latency_benchmark_percentiles PASSED    [ 51%]
tests/unit/test_latency.py::test_batch_latency_scaling PASSED            [ 53%]
tests/unit/test_metrics.py::test_perfect_predictions PASSED              [ 55%]
tests/unit/test_metrics.py::test_imperfect_predictions PASSED            [ 57%]
tests/unit/test_metrics.py::test_metric_serialization PASSED             [ 58%]
tests/unit/test_mlflow_tracker.py::test_mlflow_parent_and_child_runs PASSED [ 60%]
tests/unit/test_mlflow_tracker.py::test_mlflow_model_logging_disabled PASSED [ 62%]
tests/unit/test_model_loader.py::test_model_loader_success PASSED        [ 64%]
tests/unit/test_model_loader.py::test_model_loader_nonexistent_run PASSED [ 66%]
tests/unit/test_model_loader.py::test_model_loader_missing_parent_tag PASSED [ 67%]
tests/unit/test_models.py::test_model_lifecycle[logistic_regression] PASSED [ 69%]
tests/unit/test_models.py::test_model_lifecycle[random_forest] PASSED    [ 71%]
tests/unit/test_models.py::test_model_lifecycle[xgboost] PASSED          [ 73%]
tests/unit/test_models.py::test_model_lifecycle[torch_mlp] PASSED        [ 75%]
tests/unit/test_models.py::test_seed_reproducibility PASSED              [ 76%]
tests/unit/test_models.py::test_torch_mlp_single_sample_inference PASSED [ 78%]
tests/unit/test_models.py::test_torch_mlp_batch_size_one_safety PASSED   [ 80%]
tests/unit/test_preprocessing.py::test_preprocessing_shapes PASSED       [ 82%]
tests/unit/test_preprocessing.py::test_standard_scaler_train_normalization PASSED [ 83%]
tests/unit/test_preprocessing.py::test_pipeline_transform_unseen_data PASSED [ 85%]
tests/unit/test_preprocessing.py::test_zero_data_leakage_mathematical_proof PASSED [ 87%]
tests/unit/test_runner_modular.py::test_inspect_dataset_profile_callable PASSED [ 89%]
tests/unit/test_runner_modular.py::test_train_and_evaluate_model_callable PASSED [ 91%]
tests/unit/test_runner_modular.py::test_config_hash_determinism PASSED   [ 92%]
tests/unit/test_splitters.py::test_create_splits_no_overlap PASSED       [ 94%]
tests/unit/test_splitters.py::test_split_determinism PASSED              [ 96%]
tests/unit/test_splitters.py::test_different_seeds_produce_different_splits PASSED [ 98%]
tests/unit/test_splitters.py::test_split_with_zero_validation PASSED     [100%]

======================== 56 passed in 86.15s ========================
```

---

## 4. Empirical API Verification Evidence

### Loaded Model: `logistic_regression`
- **Child Run ID**: `f35b71a06d0c4160b0458018252f964a`
- **Parent Run ID**: `5156a48e050e47e6a32513b7863e1f32`
- **Parent Preprocessor Artifact**: `preprocessing/pipeline.joblib`

### Real Prediction Request / Response Output:

**Request**:
```json
{
  "features": {
    "mean radius": 17.99,
    "mean texture": 10.38,
    "mean perimeter": 122.8,
    "mean area": 1001.0,
    "mean smoothness": 0.1184,
    "mean compactness": 0.2776,
    "mean concavity": 0.3001,
    "mean concave points": 0.1471,
    "mean symmetry": 0.2419,
    "mean fractal dimension": 0.07871,
    "radius error": 1.095,
    "texture error": 0.9053,
    "perimeter error": 8.589,
    "area error": 153.4,
    "smoothness error": 0.006399,
    "compactness error": 0.04904,
    "concavity error": 0.05373,
    "concave points error": 0.01587,
    "symmetry error": 0.03003,
    "fractal dimension error": 0.006193,
    "worst radius": 25.38,
    "worst texture": 17.33,
    "worst perimeter": 184.6,
    "worst area": 2019.0,
    "worst smoothness": 0.1622,
    "worst compactness": 0.6656,
    "worst concavity": 0.7119,
    "worst concave points": 0.2654,
    "worst symmetry": 0.4601,
    "worst fractal dimension": 0.1189
  }
}
```

**Response (HTTP 200 OK)**:
```json
{
  "predicted_class": 0,
  "predicted_label": "malignant",
  "probabilities": {
    "malignant": 1.0,
    "benign": 0.0
  },
  "model_identity": {
    "model_name": "logistic_regression",
    "run_id": "f35b71a06d0c4160b0458018252f964a",
    "parent_run_id": "5156a48e050e47e6a32513b7863e1f32",
    "dataset_fingerprint": "0f9a42c256f6cfba77026cc3c65e6a98bd74601e59eb0a6968a4f5cc27559803",
    "config_hash_sha256": "3e5281af0b53d035341030d895d028b2a50fab3a7ddbd572a7f42923ac276276"
  },
  "inference_latency_ms": 39.5686
}
```

### Mathematical Equivalence Test Result:
`test_mathematical_equivalence_direct_vs_api`:
- Direct Phase 1 pipeline transformed row 0 $\rightarrow$ `direct_pred = 0`, `direct_probs = [0.999999..., 0.000001...]`.
- FastAPI `/predict` transformed row 0 $\rightarrow$ `predicted_class = 0`, `probabilities = {"malignant": 1.0, "benign": 0.0}`.
- Difference: $< 10^{-6}$ (Exact match within floating point precision).
