# Resume Job Match Predictor

A college-level NLP resume analysis project using PDF text extraction, TF-IDF, cosine similarity, and transparent skill-gap analysis.

## Run locally

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## How scoring works

- Resume PDF text is extracted with `pypdf`.
- Both inputs are normalized before TF-IDF vectorization.
- Cosine similarity measures overall text relevance.
- A curated technical-skills dictionary identifies matched and missing job skills.
- The final score combines 45% text similarity and 55% required-skill coverage, then maps to Low, Medium, or High Match.

The category is intentionally rule-based and explainable. To train a Logistic Regression or Random Forest classifier, add a labelled historical dataset of resume/job-description pairs and replace the threshold classifier with the trained model.
