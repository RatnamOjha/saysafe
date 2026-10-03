# Releasing

Releases are cut by pushing a tag. `.github/workflows/release.yml` then builds once, publishes to TestPyPI, installs it from there on Python 3.10 and runs a smoke test, publishes to PyPI, and creates the GitHub release with the notes from `CHANGELOG.md`. The docs site is published from the same tag by `docs.yml`.

## One-time setup (repository owner)

1. **Make the repository public.** Settings > General > Danger Zone. Check first that no secrets or recordings are in the git history: `git log -p | grep -iE "api_key|secret|ntfy"` should only show variable names.
2. **Private vulnerability reporting:** Settings > Code security > Private vulnerability reporting > Enable. SECURITY.md points people there.
3. **GitHub environments:** Settings > Environments, create `testpypi` and `pypi`. On `pypi`, add yourself as a required reviewer, so every release waits for one click after the TestPyPI smoke test.
4. **Trusted publishers** (no API tokens anywhere): on [pypi.org](https://pypi.org/manage/account/publishing/) and [test.pypi.org](https://test.pypi.org/manage/account/publishing/), add a *pending publisher*:
   - project name `saysafe`
   - owner `RatnamOjha`, repository `saysafe`
   - workflow `release.yml`
   - environment `pypi` (on PyPI) or `testpypi` (on TestPyPI)
5. **GitHub Pages:** Settings > Pages > Source: GitHub Actions.

## Each release

1. `main` is green on CI.
2. `make bench` has run on the commit being released, and its results are committed. The private-reply numbers need the reviewed labels (`uv run python eval/review.py`, then copy `eval/data/replies.yaml` to `bench/data/replies.yaml`).
3. In `CHANGELOG.md`, rename `## Unreleased` to `## [X.Y.Z] - YYYY-MM-DD` and add a fresh empty `## Unreleased` above it.
4. Set `__version__ = "X.Y.Z"` in `src/saysafe/__init__.py`. The build job refuses a tag that doesn't match it, or a changelog without that section.
5. Commit, then tag and push:

    ```bash
    git tag -a vX.Y.Z -m "saysafe X.Y.Z"
    git push origin vX.Y.Z
    ```

6. Watch the release workflow. After the smoke test passes, approve the `pypi` environment.
7. Check `pip install saysafe==X.Y.Z` in a fresh environment, and the docs site.

If something is wrong after publishing, don't delete the release: PyPI never allows a version number to be reused. Fix it and release X.Y.Z+1.
