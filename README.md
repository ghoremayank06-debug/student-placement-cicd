# Student Placement Prediction — GitHub Actions CI/CD

## What this project does
- Validates the CSV schema, missing values, data types, and feature ranges.
- Trains a Decision Tree model and fails when test accuracy is below 80%.
- Saves `model.pkl` and checks that the artifact exists.
- Provides a FastAPI prediction API (`/health`, `/model-info`, `/predict`).
- Runs automated unit and API tests with pytest.
- Runs a code-quality check with Ruff in GitHub Actions.
- Runs CI on pushes and pull requests to `main`; the final job is a deployment gate. A real external deployment target is not configured in this assignment.

## Run locally on Windows
Open this folder in VS Code. Use a Python 3.11 installation.

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
pytest -q
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000/docs. In Swagger, expand `POST /predict`, click **Try it out**, and submit:

```json
{
  "cgpa": 8.5,
  "attendance": 90,
  "coding_score": 85,
  "projects": 3,
  "internship": 1
}
```

Health check: http://127.0.0.1:8000/health

## Run CI/CD with GitHub Actions
1. Create a new GitHub repository (for example, `student-placement-cicd`).
2. Open a terminal in this project directory and run:

```bat
git init
git branch -M main
git add .
git commit -m "Implement student placement CI/CD pipeline"
git remote add origin https://github.com/YOUR-USERNAME/student-placement-cicd.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and use the URL of the repository you created. If Git says a remote already exists, inspect it with `git remote -v` and update it with `git remote set-url origin YOUR-REPOSITORY-URL`.

3. On GitHub, open the repository's **Actions** tab. The `Student Placement CI-CD` workflow should start on push. Open the workflow run to inspect each step.
4. To demonstrate CI, change a source file, commit and push again; a new workflow run should start. A failed test or an accuracy below 80% makes the workflow fail and prevents the dependent deployment-gate job from running.

## Notes
- This dataset contains only 12 rows, so metrics are unstable and 100% accuracy is not evidence of real-world performance.
- The `deploy` job is a gate/demo message, not a live deployment. To deploy publicly, configure a hosting provider and secrets, then add the provider-specific deployment commands.
