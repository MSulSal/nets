# Your Name

ML/AI Engineer building neural network systems from first principles

- Target Role: ML Systems Engineer (Training Infrastructure)
- Location: City, State
- Email: your.email@example.com
- GitHub: https://github.com/MSulSal

## Summary

Systems-focused ML engineer building core tensor and training infrastructure with explicit control over computation graphs, numerical behavior, and testing discipline.

## Role Focus

- Core tensor abstractions and operator design
- Deterministic checks and reliability-first engineering
- Path toward custom GPU kernels and performance profiling

## Selected Project Experience

### Neural Cave Lab (From-Scratch Neural Network Systems)

- Implemented a custom Tensor core in Python/NumPy with explicit graph metadata (`_prev`, `_backward`), `requires_grad`, and deterministic float64 tensor state.
- Implemented primitive tensor operators and graph-node construction for forward execution, including scalar/tensor mixed-operation support.
- Prepared autograd validation harness and staged checks for reverse-mode autodiff implementation.
- Built module-based correctness checks (`check_backward_basic, check_ops_basic, check_tensor_init`) to verify behavior incrementally before model scaling.
- Maintained fine-grained implementation history with 5 focused commits and one technical idea per commit.

## Technical Skills

Python, NumPy, Git, Neural Network Fundamentals, Autograd, Experiment Design, Computational Graphs, Autodiff Design, Numerical Correctness Checks, Reproducible Experimentation

## Education

- Your School | Your Program | YYYY - YYYY
- Relevant coursework or research focus

## Evidence Snapshot (Auto)

- Tensor methods detected: __add__, __init__, __mul__, __neg__, __radd__, __repr__, __rmul__, __rsub__, __sub__, _ensure_tensor, dtype, mean, pow, shape, sum
- Check modules detected: check_backward_basic, check_ops_basic, check_tensor_init
- Stage plan present: no
- Total commits in repo: 5

## Recent Commits

- 415577c career: add live role-specific resume generation pipeline
- bd8c14d core(autograd): add backward pass for primitive ops
- c5441d1 checks(core): add forward-ops correctness module check
- 2dba960 checks(core): add tensor init correctness module check
- 183aee5 core(tensor): add minimal Tensor container and graph metadata

---
Generated automatically by `career/generate_resumes.py` on 2026-02-27 19:43 UTC
