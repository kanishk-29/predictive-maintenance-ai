# Predictive Maintenance System using AI

Predicts machine failure from sensor readings, with a FastAPI backend
and a browser dashboard.

## Results (AI4I 2020 dataset, Random Forest)
- Recall on failures: about 90% at threshold 0.10
- Precision: about 55%
- Accuracy is not the headline metric, because 96.6% of rows are "no failure"

## Run it
pip install -r requirements.txt
python -m uvicorn app:app --reload
Then open index.html in your browser.

## Limitations
Trained on a simulated dataset. Single-reading classifier, no time-series.
Real deployment needs site-specific sensor data and retraining.
