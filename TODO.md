# TODO: Fix MLflow Run Error in GitHub Actions (Exit Code 1)

## Plan Steps:
- [x] **Step 1:** Add miniconda installation step to workflow after setup-python.
- [x] **Step 2:** Adjust/remove pip install dependencies step (use conda instead).
- [x] **Step 3:** Update "Run MLflow Project" step to `--env-manager=conda`.
- [x] **Step 4:** Edit and save .github/workflows/ci-mlflow-training.yml with changes.
- [ ] **Step 5:** Test workflow by committing/pushing and checking GitHub Actions logs.

**Current Progress:** Steps 1-4 complete. Workflow updated with micromamba for conda.yaml env setup and `mlflow run ... --env-manager=conda`. Ready for testing. The fix addresses the exit code 1 error by properly using conda environment as specified in MLproject/conda.yaml.

To test: Commit changes to a branch and push to trigger GitHub Actions run. Check logs for success.

