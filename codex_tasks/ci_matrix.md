# Task: Expand CI to Python matrix (3.10/3.11) + cache

Goal
- Update `.github/workflows/ci.yml` to test on multiple Python versions and cache pip.

Steps
1) Branch `codex/ci_matrix`.
2) Edit CI:
   - Use strategy matrix for `python-version: ["3.10", "3.11"]`.
   - Add actions/cache for pip based on `hashFiles('**/requirements*.txt')`.
3) Commit: "CI: add Python matrix + pip cache".
4) PR: "CI matrix (3.10/3.11) + cache".
