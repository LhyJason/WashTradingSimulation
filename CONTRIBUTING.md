# Contributing

This is a two-person research project. These rules keep the code reproducible and keep merge
conflicts rare. They are short on purpose.

## Workflow

`main` should always run: the smoke tests pass and the result notebooks execute top to bottom.
Do not push work in progress to `main`; use a short-lived branch and a Pull Request (PR).

```bash
git switch main && git pull
git switch -c <your-name>/<topic>        # e.g. alex/wash-gap-scan
# ...edit, commit...
git push -u origin <your-name>/<topic>   # then open a PR on GitHub
```

- The other person reviews and merges the PR (prefer **Squash and merge**: one commit per PR).
- Before opening a PR, bring your branch up to date: `git fetch && git merge origin/main`.
- Keep PRs small and about one thing. A PR that touches many files is hard to review.
- Say in advance which directory or notebook you are working on, so you do not edit the same file at the same time.

## Before you open a PR

1. Run the smoke tests for the part you changed (each ends with `RESULT: ALL PASS`):
   ```bash
   python "capstone codes/washtrade/v3/run_smoke.py"
   python "capstone codes/detection/v1/run_smoke.py"
   ```
2. If you changed the simulator or the dataset generator, regenerate the dataset and re-run the
   affected notebooks (commands in `README.md`).
3. Check `git status`: no `data/` folders, no `.venv`, no editor files.

## Project rules

- **Existing versions are frozen by default; new work goes into a new version.** `washtrade/v1`, `v2`,
  `v3` and `detection/v1` keep their behaviour, so earlier results stay reproducible. A new mechanism or
  detector goes into the next directory (`washtrade/v4`, `detection/v2`, ...). If it should reproduce an
  earlier version under some setting, add a smoke-test check that proves it. Edit an existing version
  only to fix a clear bug: say so in the PR and re-run the results that depend on it.
- **Do not edit `marketsim/`.** The capstone code only imports and monkey-patches it, so the upstream
  simulator stays comparable. If you need different behaviour, patch it from the calling code.
- **Everything is seeded.** Keep runs deterministic for a given seed. With the pinned `numpy` and `torch`
  versions the dataset is reproduced bit for bit; keep it that way.
- **The dataset schema is versioned.** If you change the files or columns written by
  `washtrade/v3/export.py` (or its successor in a newer version), bump `SCHEMA_VERSION` so old and new
  runs can be told apart.
- **New dependency?** Add it to `requirements.txt` with a version.
- **Unsettled choices** are marked `TODO:` in the code or notebook where the choice is made, with one line
  on what is undecided.

## Notebooks

- Commit the `results_*.ipynb` notebooks **with outputs**; they are the record of the results.
  Run them top to bottom on a clean kernel before committing.
- Do not let absolute paths or your username end up in outputs (print paths relative to the repo root).
- Do not commit a notebook whose only change is a re-run with different timings.
- Only one person edits a given notebook at a time; notebooks with outputs merge badly.

## Style

- Code, comments, notebooks, commit messages and PR descriptions are in English.
- Match the surrounding code: module docstring at the top saying what the file is for, short comments
  that explain why, not what.
- Commit messages: a short imperative subject (`Add ...`, `Fix ...`), then a few lines on why if it is not obvious.

## Never commit

Generated data, virtual environments, credentials or tokens, personal notes and slides, and anything
containing local file paths.
