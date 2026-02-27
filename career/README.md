# Live Resume Pipeline

This folder generates role-specific resumes from project state.

## Files

- `profile.json`: editable identity/contact/education and base skills
- `roles.json`: target role definitions
- `generate_resumes.py`: builds role-specific resumes into `career/output/`
- `TARGET_ROLES.md`: rationale for selected role targets

## Local Usage

From repo root:

```bash
python career/generate_resumes.py
```

Generated outputs:

- `career/output/ml-research-engineer-foundations.md`
- `career/output/ml-systems-engineer.md`
- `career/output/applied-ai-engineer-agentic.md`
- `career/output/README.md`

## Update Rule

Re-run generator after meaningful project updates (new commits, checks, features).

## Notes

- This generator avoids overstating current progress.
- It auto-derives evidence bullets from repo files and commit history.
