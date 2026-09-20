# Autonomous ML Research Lab: Design Decisions & Architectural Rationale

This document explains the key engineering decisions, trade-offs, and invariants enforced across Phases 1 through 5.

---

## 1. Why Model Context Protocol (MCP) is the Interface Boundary

### Problem:
In many agentic AI systems, LLM agents directly invoke raw Python functions, manipulate internal data structures, or execute arbitrary shell commands. This creates critical failure modes:
- **Execution Arbitrariness**: Agents can inject arbitrary code, access unvetted file paths, or execute unsafe operations.
- **Tight Coupling**: Swapping the reasoning agent requires rewriting the underlying ML code.
- **Lack of Protocol Auditability**: Direct Python calls lack structured tool schemas, standard error models, and uniform audit logging.

### Decision:
We use the **official Python Model Context Protocol (MCP) SDK** over local `stdio` as the sole boundary between agents and the ML engine.
- Tools have typed Pydantic input and output schemas.
- Tool invocations are validated against a three-tier permission model (`READ_ONLY`, `SAFE_WRITE`, `HIGH_RISK`).
- Every invocation is audited with timestamp, arguments, duration, and caller authorization in `logs/mcp_audit.jsonl`.
- The orchestrator interacts strictly through `await mcp_server.call_tool(tool_name, arguments)`, making the underlying engine agent-agnostic.

---

## 2. Why Deterministic ML Computation is Separated from Agent Reasoning

### Problem:
LLMs frequently hallucinate numerical results, generate inaccurate statistics, or evaluate models using flawed calculations. In quantitative and scientific environments, hallucinated metrics invalidate research integrity.

### Decision:
We enforce a strict separation of concerns:
> **"Agents decide what to investigate. Deterministic systems decide what the numbers are."**

- The LLM orchestrator decides hypotheses, models to compare, and experiment sequence.
- Deterministic Python modules (`ml/datasets/`, `ml/models/`, `ml/evaluation/`) perform the actual computations: data profiling, stratified splitting, model fitting, metric calculation, and microsecond latency measurement.
- The agent is never permitted to calculate, fabricate, or manipulate experimental numbers.

---

## 3. Why Fitted Model Objects Are Kept Outside JSON-Safe Results

### Problem:
Embedding live, in-memory model instances (`sklearn.base.BaseEstimator`, `torch.nn.Module`, `xgb.XGBClassifier`) inside result containers causes major architectural bugs:
- **Serialization Failures**: Python model instances cannot be serialized cleanly to JSON for tool calling, MCP responses, or file persistence.
- **Memory Leaks**: Preserving references to large models in result objects inflates memory usage.
- **Lineage Corruption**: Comparing experiment outputs becomes impossible when comparing in-memory pointers rather than deterministic numerical evidence.

### Decision:
We introduced the `ExperimentExecution` runtime container to cleanly decouple data from runtime artifacts:
```text
ExperimentExecution
├── result: ExperimentResult (JSON-safe metrics, hashes, percentiles)
├── fitted_models: Dict[str, BaseModel] (Runtime memory objects)
└── pipeline: PreprocessorPipeline (Fitted scikit-learn transformer)
```
- `ExperimentResult` and `ModelResult` remain pure, serializable Pydantic data schemas.
- Live model objects are consumed exclusively by the MLflow tracker or local disk saver, and are **never** exposed through MCP tool returns or agent state.

---

## 4. Why MLflow Parent/Child Lineage Exists (Preprocessing Artifact Ownership)

### Problem:
In tabular machine learning experimentation, multiple candidate models (Logistic Regression, Random Forest, XGBoost, Neural Networks) are evaluated on the exact same dataset split.
- Duplicating the fitted preprocessor inside every child model artifact wastes storage and risks pipeline skew.
- Failing to persist the preprocessor prevents production serving from transforming raw feature payloads accurately.

### Decision:
We established an experiment-level **Parent-to-Child Run Hierarchy**:
```text
Parent MLflow Run (Experiment-Level State)
├── preprocessing/pipeline.joblib  (Exact fitted scikit-learn transformer)
├── dataset_metadata.json          (SHA-256 fingerprint, source)
├── split_metadata.json            (Partition sizes, train/val/test checksums)
└── lineage tags                   (config_hash_sha256, git_commit)
    │
    ├── Child Run 1 (logistic_regression) ──► parent_run_id link
    ├── Child Run 2 (random_forest)       ──► parent_run_id link
    ├── Child Run 3 (xgboost)             ──► parent_run_id link
    └── Child Run 4 (torch_mlp)           ──► parent_run_id link
```
At serving time, `ModelLoader` loads the model weights from the child run, resolves `parent_run_id`, and loads the exact persisted preprocessing transformer from the parent run. This mathematically guarantees identical transformations between training and inference.

---

## 5. Why Execution Budgets and Guardrails Exist

### Problem:
Autonomous agents given loop execution capabilities can fall into infinite retry loops, repeatedly call failed tools, or consume excessive computational resources when hypotheses are ambiguous or tools fail.

### Decision:
Every orchestration run is bounded by an immutable `BudgetConfig`:
- **`max_tool_calls` (15)**: Caps total tool dispatches.
- **`max_iterations` (10)**: Binds planning and execution loops.
- **`max_failed_tool_calls` (3)**: Immediately aborts runaway failure cascades.
- **`timeout_seconds` (300.0)**: Guarantees wall-clock execution limits.

If any budget is exhausted, `ResearchOrchestrator` halts execution gracefully, sets status to `BUDGET_EXHAUSTED`, notes the limitation, and returns partial evidence without crashing.

---

## 6. Why the Orchestrator is Intentionally Bounded (No Bloated Frameworks)

### Problem:
Many multi-agent frameworks (LangChain, CrewAI, AutoGen) introduce heavy abstractions, opaque prompting layers, unpredictable retry loops, and hundreds of transitive dependencies. This makes debugging difficult and obscures the underlying systems engineering.

### Decision:
Phase 5 implements a **lightweight, explicit Python orchestrator**:
- Self-contained control flow: `plan -> validate -> execute -> compare -> synthesize`.
- Clear, auditable state machine (`ResearchState`).
- Standard Pydantic schemas throughout.
- Strict provenance tracking linking every metric back to an audited `ToolCallRecord`.
- Zero black-box dependencies.
