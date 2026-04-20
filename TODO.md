# TODO.md Updated: Progress on MLflow CI Fix (Kriteria 3)

## Breakdown of Approved Plan (from BLACKBOXAI analysis):
1. [x] **Step 1:** Create/update TODO.md with detailed steps tracking.
2. [x] **Step 2:** Edit .github/workflows/ci-mlflow-training.yml to fix env activation (miniconda + bash shell & conda activate).
3. [ ] **Step 3:** Commit changes and push to trigger GitHub Actions test.
4. [ ] **Step 4:** Verify run logs with gh CLI (no mlflow command not found error).
5. [ ] **Step 5:** Update TODO.md with test results and mark complete.
6. [ ] **Step 6:** Open PR if all good.

**Changes Applied:**
- Setup Miniconda: Clean miniconda setup (no invalid params).
- Run MLflow Project: `shell: bash -l {0}`, explicit `conda activate mlflow-env`, added `which mlflow` debug.

**Original TODO Context:** Fix MLflow Run Error (Exit Code 1).

**Next:** User commit/push (git add . && git commit -m "fix: ci mlflow env activation for Kriteria 3" && git push), then check Actions.

