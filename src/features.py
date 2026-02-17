from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass(frozen=True)
class FeatureConfig:
    epsilon: float
    minutes_watched_weight: float
    days_on_platform_weight: float
    courses_started_weight: float


def add_engineered_features(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Add deterministic engineered features without altering existing columns."""
    out = df.copy()

    out["engagement_score"] = (
        out["minutes_watched"] * cfg.minutes_watched_weight
        + out["days_on_platform"] * cfg.days_on_platform_weight
        + out["courses_started"] * cfg.courses_started_weight
    )

    out["exam_success_rate"] = np.where(
        out["practice_exams_started"] > 0,
        out["practice_exams_passed"] / (out["practice_exams_started"] + cfg.epsilon),
        0.0,
    )

    out["learning_consistency"] = out["minutes_watched"] / np.maximum(
        out["days_on_platform"], 1
    )

    return out


class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip numeric values to IQR bounds learned on train only."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        q1 = X_df.quantile(0.25)
        q3 = X_df.quantile(0.75)
        iqr = q3 - q1

        self.lower_bounds_ = (q1 - self.factor * iqr).to_numpy(dtype=float)
        self.upper_bounds_ = (q3 + self.factor * iqr).to_numpy(dtype=float)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([f"feature_{i}" for i in range(len(self.lower_bounds_))], dtype=object)
        return np.asarray(input_features, dtype=object)
