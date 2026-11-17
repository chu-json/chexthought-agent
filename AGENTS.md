# Notes for agents working in this repo

Read this first. It applies to every agent and session (Claude Code, Codex, others).

## Push results to `main` often
The owner is often away from this machine and checks progress on GitHub. Work that sits only on a local
branch is invisible to them.
- Commit small and often. After every milestone (a run finishes, a result is scored, a doc is updated),
  **merge into `main` and `git push origin main`**. Feature branches are fine for work in progress, but
  don't let `main` fall behind by more than a few hours of work.
- Keep `docs/progress_log.md` up to date (newest first) and push it with each milestone. It is the one place
  the owner reads to catch up.
- Put aggregate results that the owner needs (tables of metrics, CIs) in `docs/` so they reach GitHub;
  `outputs/` is gitignored.

## Never commit or upload protected data
Images, reports, labels, image/patient ids and generated reports are under a research use agreement.
`outputs/`, `data/`, images and CSVs are gitignored; keep it that way. Committed files hold aggregate numbers only.

## Other house rules
- No Claude/Anthropic API spend (e.g. Sonnet labeling) without asking the owner first; use local models
  (CheXbert, vLLM) where possible.
- The GPU is shared with BEHAVIOR-1K jobs: never kill processes you didn't start without asking.
- Python envs: `.venv` (main), `.venv-cxas`, `.venv-chexone` (vLLM, CheXbert, scoring), `.venv-maira`.
