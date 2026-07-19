"""
Tests for the Exoplanet Hunting API.

Run from the project root:
    pytest

Uses synthetic (random) flux data rather than a real row from the dataset,
deliberately — the data/ folder isn't committed to the repo (see README),
so tests must not depend on it existing. These tests check that the API
behaves correctly given *any* correctly-shaped input, not that the model's
predictions are astronomically meaningful (that's what the CV metrics in
train_model.py / ablation_study.py are for).
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import EXPECTED_FLUX_LENGTH

client = TestClient(app)


@pytest.fixture
def valid_flux():
    """A random but correctly-shaped flux array — enough to exercise the
    full extract_features() -> model.predict_proba() path without needing
    real astrophysical data."""
    rng = np.random.default_rng(seed=42)
    # Roughly flux-shaped: centered around a baseline, with noise -
    # doesn't need to look like a real star, just be numeric and full-length.
    return (1000 + rng.normal(0, 50, size=EXPECTED_FLUX_LENGTH)).tolist()


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    # If this fails with "model_not_loaded", check that models/exoplanet_model.joblib
    # exists relative to wherever you're running pytest from.
    assert body["status"] == "ok"


def test_predict_valid_input(valid_flux):
    response = client.post("/predict", json={"flux": valid_flux})
    assert response.status_code == 200

    body = response.json()
    assert "prediction" in body
    assert "probability" in body
    assert body["prediction"] in ("planet", "no_planet")
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_rejects_wrong_length_input():
    too_short = [0.0] * 100  # nowhere near the required 3197
    response = client.post("/predict", json={"flux": too_short})
    # Pydantic's field_validator should reject this before it ever reaches
    # the model - proves the validation in schemas.py is actually wired up,
    # not just decorative.
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_rejects_non_numeric_values():
    bad_flux = ["not_a_number"] * EXPECTED_FLUX_LENGTH
    response = client.post("/predict", json={"flux": bad_flux})
    assert response.status_code == 422