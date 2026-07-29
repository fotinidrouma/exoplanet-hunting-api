# Exoplanet Hunting — End-to-End ML Deployment

Predicts whether a star hosts a transiting exoplanet from its raw brightness
(flux) measurements over time, served as a REST API.

This project's goal isn't just "train a model" — it's demonstrating a full
path from raw data to a deployed, servable prediction service, with
deliberate handling of the awkward realities of this specific dataset
(severe class imbalance, tiny positive class, high-dimensional raw input).

---

## Dataset

[Kepler labelled time series data](https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data)
— flux measurements for ~5,600 stars, labeled as confirmed exoplanet host
(`LABEL=2`) or not (`LABEL=1`).

| | Rows | Positives | Positive rate |
|---|---|---|---|
| Train | 5,087 | 37 | 0.73% |
| Test | 570 | 5 | 0.88% |

**This imbalance is the central challenge of the whole project** and shapes
almost every downstream decision: feature engineering, model choice,
evaluation strategy, and even how much to trust the test set numbers.

---

## What I found in Exoplanet Data Analysis
## What I found in Exoplanet Data Analysis

- **Raw flux is not remotely normally distributed** — it's a noisy signal
  around a large per-star baseline, occasionally dipping. Per-row
  normalization and differencing get closer to workable distributions.
- **The official test set is too small to be a reliable sole benchmark**
  (5 positives). Stratified cross-validation on the training set is the
  more trustworthy signal throughout this project; test set results are
  reported as a sanity check, not a precise measurement.
- **No missing values** in the flux columns; a small number of rows show
  signs of instrument artifacts (flat/constant sections, extreme single-point
  outliers) rather than genuine astrophysical signal.
- Full EDA script: [`data_analysis/analyze_data.py`](./data_analysis/analyze_data.py)
- Full EDA script: [`data_analysis/analyze_data.py`](./data_analysis/analyze_data.py)

---

## Feature engineering

Raw input is 3,197 flux columns per star — high-dimensional relative to
only 37 positive training examples, so dimensionality reduction via
engineered features (rather than feeding the model raw flux directly) was
the deliberate choice here.

**Time-domain features:** mean, std, variance, skewness, kurtosis, range —
capture the general "shape" of the noise/signal in each row.

**Frequency-domain (FFT) features:** since a planetary transit is a
periodic dip, it should show up as a concentrated spike in the frequency
domain, potentially easier for a model to exploit than a dip buried in
time-domain noise. Extracted: dominant frequency, max magnitude, spectral
energy, and the top-5 magnitude peaks (to allow for multi-planet systems
with more than one periodic signal).

All variance/magnitude/energy-based features are log-transformed
(`log1p`), since they span several orders of magnitude.

Full implementation: [`app/preprocessing.py`](./app/preprocessing.py)

---

## Model selection

Compared Random Forest and XGBoost under identical conditions: same
stratified 5-fold CV splits, `class_weight="balanced"` (Random Forest) /
`scale_pos_weight` (XGBoost) to address the imbalance, evaluated on
precision/recall/F1/ROC-AUC rather than accuracy (accuracy is meaningless
at 0.7% positive rate — a model that always predicts "no planet" would
score ~99%).

> **TODO — paste your actual train_model.py output here**, e.g.:
>
> | Model | Planet precision | Planet recall | Planet F1 | ROC-AUC |
> |---|---|---|---|---|
> | Random Forest | 0.00 | 0.00 | 0.00 | 0.7772 |
> | XGBoost | 0.04 | 0.22 | 0.07 | 0.7735 |
>
> Best model by CV ROC-AUC: random_forest
> ROC-AUC-based pick in favor of better recall.
> (Missing a real exoplanet is arguably costlier than a false alarm, since
> a false positive just gets ruled out on follow-up — a missed detection
> is never looked at again.)

Full script: [`train_model.py`](./train_model.py)

---

## Architecture

```
project/
├── app/                          # the API package
│   ├── preprocessing.py              # extract_features() — shared by
│   │                                 # training AND the live API, so both
│   │                                 # apply the exact same transformation
│   ├── schemas.py                    # request/response validation (Pydantic)
│   └── main.py                       # FastAPI app: /health, /predict
├── models/
│   └── exoplanet_model.joblib        # trained model + expected feature order
├── preprocess_data.py             # CLI: raw flux -> engineered features
├── train_model.py                 # CLI: trains + compares models, saves winner
└── requirements.txt
```

**Key design decision:** `extract_features()` lives in exactly one place
(`app/preprocessing.py`) and is imported by both the offline training
pipeline and the live API. This guarantees the exact same transformation is
applied at training time and prediction time — a common, hard-to-debug
source of production ML bugs is these two silently drifting apart.

---

## Setup & reproduction

```bash
# 1. Get the data
# Download exoTrain.csv and exoTest.csv from the Kaggle link above,
# place them in data/

# 2. Set up environment
python -m venv exoplanet_venv
exoplanet_venv\Scripts\activate        # backslashes needed
pip install -r requirements.txt

# 3. Preprocess raw data into engineered features
python preprocess_data.py --data_dir data --train_fn exoTrain.csv --test_fn exoTest.csv --output_prefix processed

# 4. Train and select a model
python train_model.py
# move the resulting exoplanet_model.joblib into models/

# 5. Run the API
uvicorn app.main:app --reload
```

Then visit `http://127.0.0.1:8000/docs` for interactive API documentation
and a built-in way to test `/predict` without writing a client.

---

## API usage

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"flux": [<3197 comma-separated flux values>]}'
```

Response:
```json
{
  "prediction": "planet",
  "probability": 0.87
}
```

---

## Known limitations

- The official test set (5 positives) is too small for precise performance
  claims — cross-validated training metrics are the more trustworthy
  numbers throughout.
- FFT peak detection doesn't distinguish a genuine transit signal from
  other periodic astrophysical noise (stellar variability, instrument
  drift) — this is a *feature* the model can exploit, not a definitive
  detector, and shouldn't be read as one.
- Not yet load-tested or benchmarked for prediction latency.

---

## Dockerization

```bash
# Build the image (the "." means "use the Dockerfile in this directory")
docker build -t exoplanet-api .

# Run it, mapping container port 8000 to your machine's port 8000
docker run -p 8000:8000 exoplanet-api
```
Open [127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
Open [127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Automated Tests

```bash
python -m pytest

```

---

## Deployment (Render)

Open [https://exoplanet-hunting-api.onrender.com/docs](https://exoplanet-hunting-api.onrender.com/docs)

---

## Roadmap
- [x] EDA
- [x] Feature engineering (time-domain + FFT)
- [x] Model comparison (Random Forest vs XGBoost) with proper imbalance handling
- [x] FastAPI service with input validation
- [x] Dockerize
- [x] Automated tests
- [x] Deploy to a live URL (Render)
- [x] Logging + basic monitoring
- [x] CI/CD (GitHub Actions)
