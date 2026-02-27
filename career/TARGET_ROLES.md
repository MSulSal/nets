# Target Roles

These roles are selected based on the current and planned work in this repository.

## 1) ML Research Engineer (Foundations)

Why this repo fits:
- from-scratch tensor and autograd implementation
- correctness-oriented checks and gradient validation path
- explicit experiment design and ablation framing

Current evidence:
- Tensor core and primitive ops scaffolded
- module-based correctness checks for tensor init and forward ops

Evidence to add next:
- complete backward coverage and numerical gradcheck
- minimal MLP training with reproducible runs

## 2) ML Systems Engineer (Training Infrastructure)

Why this repo fits:
- low-level control over tensor operations and graph mechanics
- explicit software architecture for NN training components
- clear direction toward CPU/GPU kernel-level work

Current evidence:
- operator-level API and computation graph metadata
- staged check workflow for reliability

Evidence to add next:
- performance benchmarking harness
- first custom GPU kernel and host/device memory utilities

## 3) Applied AI Engineer (Agentic Systems Path)

Why this repo fits:
- strong model internals foundation for reliable applied AI work
- experiment-card and failure-analysis discipline
- roadmap includes agent runtime, tools, memory, and evaluation

Current evidence:
- structured project plan and track separation (chronological + agentic)
- reproducibility-focused workflow docs

Evidence to add next:
- end-to-end task benchmarks and tool-use loops
- reliability metrics for agent trajectories
