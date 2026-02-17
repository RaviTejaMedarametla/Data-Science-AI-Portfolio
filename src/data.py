from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.features import FeatureConfig, add_engineered_features


def load_config(path: str | Path = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(config: dict) -> pd.DataFrame:
    data_path = config["data"]["path"]
    df = pd.read_csv(data_path)

    fcfg = FeatureConfig(
        epsilon=float(config["features"]["epsilon"]),
        minutes_watched_weight=float(config["features"]["engagement"]["minutes_watched_weight"]),
        days_on_platform_weight=float(config["features"]["engagement"]["days_on_platform_weight"]),
        courses_started_weight=float(config["features"]["engagement"]["courses_started_weight"]),
    )
    return add_engineered_features(df, fcfg)


def split_data(df: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    target = config["data"]["target"]
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=float(config["data"]["test_size"]),
        random_state=int(config["seed"]),
        stratify=y,
    )
