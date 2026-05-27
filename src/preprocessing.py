import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


# 28-feature order — both the scaler and model expect exactly this
FEATURE_ORDER = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges',
    'tenure_group', 'total_services', 'has_any_security', 'has_any_backup',
    'is_long_term', 'avg_monthly_charges', 'charges_per_service',
    'paperless_electronic', 'tenure_monthly_interaction',
]

# columns that get label-encoded
CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'tenure_group',
]

# used when computing total_services
SERVICE_COLS = [
    'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
]


def load_data(filepath: str) -> pd.DataFrame:
    """Read raw CSV and drop customerID."""
    df = pd.read_csv(filepath)
    df.drop(columns=['customerID'], errors='ignore', inplace=True)
    print(f"[load]  {df.shape[0]:,} rows x {df.shape[1]} columns loaded.")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix TotalCharges (whitespace → 0) and encode Churn as 0/1.
    SeniorCitizen is already numeric in the source.
    """
    df = df.copy()

    df['TotalCharges'] = (
        df['TotalCharges']
        .replace(r'^\s*$', '0', regex=True)
        .astype(float)
    )

    # handles object, bool, or already-int Churn safely
    df['Churn'] = (
        df['Churn'].astype(str).str.strip().str.lower() == 'yes'
    ).astype(int)

    n_null = df.isnull().sum().sum()
    print(f"[clean] Nulls remaining: {n_null}  |  Churn rate: {df['Churn'].mean():.2%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 9 derived features. No fitted state — safe to call on a single row
    at inference time with the same results as during training.
    """
    df = df.copy()

    df['tenure_group'] = pd.cut(
        df['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0-12', '13-24', '25-48', '49-72'],
        include_lowest=True,
    ).astype(str)

    df['total_services'] = (
        sum((df[c] == 'Yes').astype(int) for c in SERVICE_COLS)
        + (df['InternetService'] != 'No').astype(int)
    )

    df['has_any_security'] = (
        (df['OnlineSecurity'] == 'Yes') | (df['TechSupport'] == 'Yes')
    ).astype(int)

    df['has_any_backup'] = (
        (df['OnlineBackup'] == 'Yes') | (df['DeviceProtection'] == 'Yes')
    ).astype(int)

    df['is_long_term'] = df['Contract'].isin(['One year', 'Two year']).astype(int)

    df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['charges_per_service'] = df['MonthlyCharges'] / (df['total_services'] + 1)

    # customers on paperless billing who also pay by e-check churn noticeably more
    df['paperless_electronic'] = (
        (df['PaperlessBilling'] == 'Yes')
        & (df['PaymentMethod'] == 'Electronic check')
    ).astype(int)

    df['tenure_monthly_interaction'] = df['tenure'] * df['MonthlyCharges']

    return df


def fit_encoders_and_scaler(df: pd.DataFrame):
    """
    Fit LabelEncoders and StandardScaler on the full dataset before splitting.
    Fitting on all data ensures every label class is seen by the encoders.

    The scaler is fitted on all 28 encoded columns, not just the numeric ones,
    because that's how the original pkl was created and the two must match.

    Returns encoders dict and fitted scaler.
    """
    df = df.copy()
    if 'Churn' in df.columns:
        df.drop(columns=['Churn'], inplace=True)

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        le.fit(df[col].astype(str))
        encoders[col] = le
        df[col] = le.transform(df[col].astype(str))

    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_ORDER]

    scaler = StandardScaler()
    scaler.fit(df)

    return encoders, scaler


def apply_encoders_and_scaler(
    df: pd.DataFrame,
    encoders: dict,
    scaler,
    has_target: bool = True,
):
    """
    Encode and scale a DataFrame using pre-fitted objects.
    Pass has_target=False when calling from inference (no Churn column).
    Returns (X, y) where y is None when has_target is False.
    """
    df = df.copy()
    y = df.pop('Churn') if has_target else None

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        le = encoders[col]
        known = set(le.classes_)
        # map unseen labels to the first known class rather than raising
        df[col] = (
            df[col].astype(str)
            .apply(lambda x: x if x in known else le.classes_[0])
        )
        df[col] = le.transform(df[col])

    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0
    df = df[FEATURE_ORDER]

    X = pd.DataFrame(
        scaler.transform(df),
        columns=FEATURE_ORDER,
        index=df.index,
    )

    return X, y


def prepare_input(raw: dict, encoders: dict, scaler) -> pd.DataFrame:
    """
    Convert a single customer dict from the app into a model-ready row.
    This is the only preprocessing function called at inference time.
    """
    df = pd.DataFrame([raw])
    df = engineer_features(df)
    X, _ = apply_encoders_and_scaler(df, encoders, scaler, has_target=False)
    return X
