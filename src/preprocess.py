"""Data loading, synthetic generation, scaling, and stratified splitting.

This module handles:
1. Loading the Kaggle ULB Credit Card Fraud dataset or generating a synthetic fallback.
2. Robust scaling of 'Time' and 'Amount' features (resistant to extreme outliers).
3. Maintaining PCA features V1-V28 intact.
4. Stratified 80/20 train-test splitting to preserve minority class proportion.
5. Imbalanced data resampling (SMOTE / Random Undersampling) strictly on training data.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Feature column ordering expected throughout the system
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def generate_synthetic_creditcard_data(
    n_samples: int = 20000,
    fraud_ratio: float = 0.002,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic imbalanced credit card transaction dataset.

    Fallback used when data/creditcard.csv is not present, allowing the pipeline
    to execute end-to-end out-of-the-box.

    Args:
        n_samples: Total number of transactions (default: 20,000).
        fraud_ratio: Fraction of fraudulent transactions (default: 0.002 = 0.2%).
        random_state: Seed for reproducible generation.

    Returns:
        pd.DataFrame: DataFrame containing Time, V1-V28, Amount, and Class columns.
    """
    rng = np.random.RandomState(random_state)
    weights = [1.0 - fraud_ratio, fraud_ratio]

    # Generate 28 PCA-like numerical features
    X_pca, y = make_classification(
        n_samples=n_samples,
        n_features=28,
        n_informative=24,
        n_redundant=4,
        n_clusters_per_class=2,
        weights=weights,
        flip_y=0.0005,
        random_state=random_state,
    )

    # Generate synthetic Time (elapsed seconds over 48 hours: 0 to 172,800)
    time_feature = np.sort(rng.uniform(low=0.0, high=172800.0, size=n_samples))

    # Generate synthetic Amount with heavy right tail (log-normal distribution)
    # Legitimate transactions average ~$88; fraudulent transactions often have specific distributions
    normal_amounts = rng.lognormal(mean=3.5, sigma=1.2, size=n_samples)
    amounts = np.round(np.clip(normal_amounts, 0.5, 25000.0), 2)

    # Assemble complete DataFrame with exact Kaggle column schema
    df_data = {"Time": time_feature}
    for i in range(28):
        df_data[f"V{i+1}"] = X_pca[:, i]
    df_data["Amount"] = amounts
    df_data["Class"] = y

    df = pd.DataFrame(df_data)
    fraud_count = int(df["Class"].sum())
    logger.warning(
        "[WARNING] 'data/creditcard.csv' was not found! "
        "Generated synthetic fallback dataset: %d samples, %d frauds (%.2f%% fraud rate).",
        len(df),
        fraud_count,
        (fraud_count / len(df)) * 100,
    )
    return df


def load_data(
    data_path: Optional[Union[str, Path]] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, bool]:
    """Load credit card dataset from CSV, or fall back to synthetic data.

    Args:
        data_path: Path to creditcard.csv. If None, defaults to project data/ directory.
        random_state: Seed for synthetic fallback generation.

    Returns:
        Tuple[pd.DataFrame, bool]: (DataFrame, is_synthetic_flag)
    """
    if data_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        data_path = base_dir / "data" / "creditcard.csv"
    else:
        data_path = Path(data_path)

    if data_path.exists():
        logger.info("Loading genuine dataset from: %s", data_path)
        df = pd.read_csv(data_path)
        logger.info(
            "Dataset loaded successfully: %d rows, %d fraud cases (%.4f%%).",
            len(df),
            df["Class"].sum(),
            (df["Class"].sum() / len(df)) * 100,
        )
        return df, False
    else:
        logger.warning("Dataset file not found at: %s", data_path)
        print("\n" + "=" * 80)
        print(" [WARNING] 'data/creditcard.csv' not found.")
        print(" Falling back to synthetic imbalanced dataset (20,000 rows, 0.2% fraud)")
        print(" generated with sklearn.datasets.make_classification for end-to-end execution.")
        print(" Please place the Kaggle 'creditcard.csv' into 'data/' for real-world benchmarking.")
        print("=" * 80 + "\n")
        df_synthetic = generate_synthetic_creditcard_data(
            n_samples=20000,
            fraud_ratio=0.002,
            random_state=random_state,
        )
        return df_synthetic, True


def scale_features(
    df: pd.DataFrame,
    scaler: Optional[RobustScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[pd.DataFrame, RobustScaler]:
    """Scale 'Time' and 'Amount' using RobustScaler, keeping V1-V28 untouched.

    Why RobustScaler?
    'Amount' exhibits severe positive skewness with high outliers ($0.01 to $25,000+).
    'Time' spans over two full days. Standard scaling (mean/variance) is sensitive to
    extreme outliers. RobustScaler removes the median and scales according to the
    Interquartile Range (IQR = 75th - 25th percentile), making it ideal for financial
    outlier-heavy distributions.

    Args:
        df: Input DataFrame containing Time, V1-V28, Amount, and optionally Class.
        scaler: Pre-fitted RobustScaler instance. If None and fit_scaler=True, a new one is created.
        fit_scaler: Whether to fit the scaler on the provided data.

    Returns:
        Tuple[pd.DataFrame, RobustScaler]: (Scaled DataFrame, Fitted RobustScaler)
    """
    df_scaled = df.copy()
    columns_to_scale = ["Time", "Amount"]

    if fit_scaler:
        scaler = RobustScaler()
        df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])
    else:
        if scaler is None:
            raise ValueError("Must provide a pre-fitted RobustScaler when fit_scaler=False.")
        df_scaled[columns_to_scale] = scaler.transform(df_scaled[columns_to_scale])

    return df_scaled, scaler


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train-test split.

    Preserves the rare fraud class ratio in both training and testing partitions.
    Resampling is strictly NOT applied here to avoid data leakage onto the test set.

    Args:
        df: Input DataFrame.
        test_size: Fraction of data for testing (default: 0.20 = 20%).
        random_state: Seed for reproducible split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            (X_train, X_test, y_train, y_test)
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    logger.info(
        "Stratified Split: Train set = %d (Frauds: %d, %.3f%%) | Test set = %d (Frauds: %d, %.3f%%)",
        len(X_train),
        y_train.sum(),
        (y_train.sum() / len(y_train)) * 100,
        len(X_test),
        y_test.sum(),
        (y_test.sum() / len(y_test)) * 100,
    )

    return X_train, X_test, y_train, y_test


def apply_resampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "class_weight",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply resampling strategies to the TRAINING set only.

    CRITICAL RULE:
    Resampling is NEVER applied to the test set. Doing so would artificially alter
    the fraud base rate and cause severe data leakage, resulting in inflated, unrealistic
    validation metrics.

    Args:
        X_train: Training features.
        y_train: Training labels.
        strategy: One of 'class_weight' (identity), 'smote', or 'undersample'.
        random_state: Random seed for sampling reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Resampled (X_train, y_train).
    """
    strategy = strategy.lower().strip()

    if strategy in ("class_weight", "none", "baseline"):
        logger.info("Resampling Strategy: 'class_weight' - No artificial resampling applied.")
        return X_train.copy(), y_train.copy()

    elif strategy == "smote":
        logger.info("Resampling Strategy: 'SMOTE' - Synthetically oversampling minority class...")
        # Check minority sample count to set k_neighbors safely
        minority_count = int((y_train == 1).sum())
        k_neighbors = min(5, max(1, minority_count - 1))
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        logger.info(
            "After SMOTE: Total = %d (Legit: %d, Fraud: %d)",
            len(X_resampled),
            (y_resampled == 0).sum(),
            (y_resampled == 1).sum(),
        )
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name=y_train.name)

    elif strategy in ("undersample", "random_undersample"):
        logger.info("Resampling Strategy: 'Random Undersampling' - Downsampling majority class...")
        rus = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        logger.info(
            "After Undersampling: Total = %d (Legit: %d, Fraud: %d)",
            len(X_resampled),
            (y_resampled == 0).sum(),
            (y_resampled == 1).sum(),
        )
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled, name=y_train.name)

    else:
        raise ValueError(
            f"Unknown resampling strategy: '{strategy}'. Supported: 'class_weight', 'smote', 'undersample'"
        )
