from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RepoSignals:
    commit_count: int
    recent_commits: list[str]
    tensor_methods: list[str]
    check_modules: list[str]
    has_backward_method: bool
    has_experiment_card: bool
    has_stage_plan: bool


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def run_git(args: list[str], cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def collect_repo_signals(repo_root: Path) -> RepoSignals:
    tensor_path = repo_root / "core" / "tensor" / "tensor.py"
    tensor_text = tensor_path.read_text(encoding="utf-8") if tensor_path.exists() else ""

    tensor_methods = sorted(set(re.findall(r"^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", tensor_text, re.MULTILINE)))
    has_backward_method = "backward" in tensor_methods

    check_dir = repo_root / "checks" / "core"
    check_modules = sorted([p.stem for p in check_dir.glob("check_*.py")]) if check_dir.exists() else []

    count_text = run_git(["rev-list", "--count", "HEAD"], repo_root)
    commit_count = int(count_text) if count_text.isdigit() else 0

    log_text = run_git(["log", "--oneline", "-n", "5"], repo_root)
    recent_commits = [line.strip() for line in log_text.splitlines() if line.strip()]

    return RepoSignals(
        commit_count=commit_count,
        recent_commits=recent_commits,
        tensor_methods=tensor_methods,
        check_modules=check_modules,
        has_backward_method=has_backward_method,
        has_experiment_card=(repo_root / "docs" / "EXPERIMENT_CARD.md").exists(),
        has_stage_plan=(repo_root / "docs" / "PROJECT_PLAN.md").exists(),
    )


def build_project_bullets(signals: RepoSignals) -> list[str]:
    bullets: list[str] = []

    if "__init__" in signals.tensor_methods:
        bullets.append(
            "Implemented a custom Tensor core in Python/NumPy with explicit graph metadata (`_prev`, `_backward`), `requires_grad`, and deterministic float64 tensor state."
        )

    op_methods = [m for m in ["__add__", "__sub__", "__mul__", "__neg__", "pow", "sum", "mean"] if m in signals.tensor_methods]
    if op_methods:
        bullets.append(
            "Implemented primitive tensor operators and graph-node construction for forward execution, including scalar/tensor mixed-operation support."
        )

    if signals.has_backward_method:
        bullets.append(
            "Implemented reverse-mode autodiff traversal and local gradient propagation for primitive operations."
        )
    else:
        bullets.append(
            "Prepared autograd validation harness and staged checks for reverse-mode autodiff implementation."
        )

    if signals.check_modules:
        bullets.append(
            f"Built module-based correctness checks (`{', '.join(signals.check_modules)}`) to verify behavior incrementally before model scaling."
        )

    if signals.has_experiment_card:
        bullets.append(
            "Added experiment-card workflow to enforce hypothesis-first experimentation, baseline comparisons, and explicit scale-up gates."
        )

    if signals.commit_count:
        bullets.append(
            f"Maintained fine-grained implementation history with {signals.commit_count} focused commits and one technical idea per commit."
        )

    return bullets


def merge_skills(profile: dict[str, Any], signals: RepoSignals) -> list[str]:
    skills = list(profile.get("base_skills", []))

    inferred = [
        "Computational Graphs",
        "Reverse-Mode Autodiff" if signals.has_backward_method else "Autodiff Design",
        "Numerical Correctness Checks",
        "Reproducible Experimentation",
    ]

    merged = []
    seen = set()
    for item in [*skills, *inferred]:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    return merged


def render_resume(
    profile: dict[str, Any],
    role: dict[str, Any],
    signals: RepoSignals,
    project_bullets: list[str],
) -> str:
    links = []
    if profile.get("github"):
        links.append(f"GitHub: {profile['github']}")
    if profile.get("website"):
        links.append(f"Website: {profile['website']}")
    if profile.get("linkedin"):
        links.append(f"LinkedIn: {profile['linkedin']}")

    skills = merge_skills(profile, signals)

    edu_lines = []
    for edu in profile.get("education", []):
        edu_lines.append(f"- {edu.get('school', '')} | {edu.get('program', '')} | {edu.get('dates', '')}")
        notes = edu.get("notes", "").strip()
        if notes:
            edu_lines.append(f"- {notes}")

    focus_lines = "\n".join(f"- {item}" for item in role.get("focus", []))
    project_lines = "\n".join(f"- {item}" for item in project_bullets)
    skill_line = ", ".join(skills)
    commit_lines = "\n".join(f"- {c}" for c in signals.recent_commits) or "- No commit history available"

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""# {profile.get('name', 'Your Name')}

{profile.get('headline', '')}

- Target Role: {role.get('title', '')}
- Location: {profile.get('location', '')}
- Email: {profile.get('email', '')}
{chr(10).join(f"- {line}" for line in links) if links else ""}

## Summary

{role.get('summary', '')}

## Role Focus

{focus_lines}

## Selected Project Experience

### Neural Cave Lab (From-Scratch Neural Network Systems)

{project_lines}

## Technical Skills

{skill_line}

## Education

{chr(10).join(edu_lines) if edu_lines else "- Add education details in `career/profile.json`."}

## Evidence Snapshot (Auto)

- Tensor methods detected: {', '.join(signals.tensor_methods) if signals.tensor_methods else 'none'}
- Check modules detected: {', '.join(signals.check_modules) if signals.check_modules else 'none'}
- Stage plan present: {'yes' if signals.has_stage_plan else 'no'}
- Total commits in repo: {signals.commit_count}

## Recent Commits

{commit_lines}

---
Generated automatically by `career/generate_resumes.py` on {generated_at}
""".strip() + "\n"


def write_outputs(repo_root: Path, profile: dict[str, Any], roles: list[dict[str, Any]], signals: RepoSignals) -> None:
    output_dir = repo_root / "career" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    project_bullets = build_project_bullets(signals)

    index_lines = [
        "# Generated Resumes",
        "",
        "Generated from project state and role targets.",
        "",
        "## Files",
    ]

    for role in roles:
        content = render_resume(profile, role, signals, project_bullets)
        out_path = output_dir / f"{role['id']}.md"
        out_path.write_text(content, encoding="utf-8")
        index_lines.append(f"- `{out_path.name}`")

    index_lines.extend(
        [
            "",
            "## Refresh Command",
            "",
            "```bash",
            "python career/generate_resumes.py",
            "```",
        ]
    )

    (output_dir / "README.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    profile = read_json(repo_root / "career" / "profile.json")
    roles = read_json(repo_root / "career" / "roles.json")
    signals = collect_repo_signals(repo_root)
    write_outputs(repo_root, profile, roles, signals)
    print("Resumes generated in career/output")


if __name__ == "__main__":
    main()
