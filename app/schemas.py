"""
Request/response schemas for the exoplanet prediction API.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List

EXPECTED_FLUX_LENGTH = 3197  # matches the Kepler dataset's FLUX.1 ... FLUX.3197

class FluxInput(BaseModel):
    flux: List[float] = Field(
        ...,
        description=f"Raw flux time series, must contain exactly {EXPECTED_FLUX_LENGTH} values",
    )

    @field_validator("flux")
    @classmethod
    def check_length(cls, v):
        if len(v) != EXPECTED_FLUX_LENGTH:
            raise ValueError(
                f"Expected {EXPECTED_FLUX_LENGTH} flux values, got {len(v)}"
            )
        return v


class PredictionOutput(BaseModel):
    prediction: str        # "planet" or "no_planet"
    probability: float     # model's predicted probability of "planet"