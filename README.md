# Resume Job Match Predictor

A college-level resume analysis project using PDF text extraction, direct job-keyword matching, and transparent skill-gap analysis.

## Run locally

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Deploy on Render

The browser version is a static site in `index.html`; it needs no server-side Python packages.

1. Push this folder to a GitHub repository.
2. In [Render](https://render.com), select **New +** → **Blueprint** and connect the repository.
3. Render reads `render.yaml`, publishes the project root, and provides a public `onrender.com` URL.

Alternatively, select **New +** → **Static Site**, choose the repository, leave the build command blank, and set **Publish Directory** to `.`.

## How scoring works

- Resume PDF text is extracted with `pypdf`.
- Both inputs are normalized before matching.
- A curated technical-skills dictionary identifies matched and missing job skills, including common aliases such as `PowerBI` for `Power BI`.
- When the job description contains recognized skills, the final score equals required-skill coverage. A title-only `Software Developer` input uses the same five-group profile in the Python and static browser apps: programming languages, web development, databases, version control, and build/delivery tools. A group counts as covered when the resume names at least one listed technology in it. Other unrecognized descriptions are compared using meaningful keywords.

The category is intentionally rule-based and explainable. To train a Logistic Regression or Random Forest classifier, add a labelled historical dataset of resume/job-description pairs and replace the threshold classifier with the trained model.
