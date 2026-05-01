import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file):
    try:
        return pd.read_csv(file)
    except Exception as e:
        return f"Error loading file: {e}"


def preprocess(df, target_col, scale=False):
    if df.empty:
        return "Dataframe is empty"

    df = df.copy()  # avoid mutating the original

    cols_to_drop = []
    num_col = []
    non_num_col = []

    for col in df.columns:
        if col == target_col:
            continue
        elif pd.api.types.is_numeric_dtype(df[col]):  # replaces dtype != 'object'
            num_col.append(col)
        else:
            unique_ratio = df[col].nunique() / len(df)
            avg_char_len = df[col].astype(str).str.len().mean()
            if unique_ratio > 0.9 or avg_char_len > 50:
                cols_to_drop.append(col)
            else:
                non_num_col.append(col)

    df.drop(columns=cols_to_drop, inplace=True)

    # Fill numeric columns
    for col in num_col:
        df[col].fillna(df[col].mean(), inplace=True)

    # Encode categorical columns BEFORE any numeric conversion
    for col in non_num_col:
        unique_count = df[col].nunique()

        if unique_count < 4:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    df = df.astype('float64')
    for col in df.columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if not scale:
        return X_train, X_test, y_train, y_test

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)
    return X_train_sc, X_test_sc, y_train, y_test