import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    DATASET,
    TEST_SIZE,
    RANDOM_STATE
)

def load_data():
    """
    Load the customer churn dataset.
    """

    df = pd.read_csv(DATASET)

    return df

def inspect_data(df):
    """
    Display basic information about the dataset.
    """

    print("\n===== Dataset Shape =====")
    print(df.shape)

    print("\n===== First 5 Rows =====")
    print(df.head())

    print("\n===== Column Names =====")
    print(df.columns)

    print("\n===== Data Types =====")
    print(df.dtypes)

    print("\n===== Missing Values =====")
    print(df.isnull().sum())

    print("\n===== Duplicate Rows =====")
    print(df.duplicated().sum())

def check_data_quality(df):
    """
    Check the dataset for common data quality issues.
    """

    print("\n========== DATA QUALITY REPORT ==========\n")

    # Dataset dimensions
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    print("\n-------------------------------")

    # Missing values
    print("\nMissing Values\n")
    print(df.isna().sum())

    print("\n-------------------------------")

    # Duplicate rows
    print("\nDuplicate Rows")
    print(df.duplicated().sum())

    print("\n-------------------------------")

    # Unique values
    print("\nUnique Values Per Column\n")

    for column in df.columns:
        print(f"{column:20} {df[column].nunique()}")

def clean_data(df):
    """
    Clean the dataset.
    """

    print("\n========== CLEANING DATA ==========\n")

    # Make a copy
    df = df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    print("\nMissing values after conversion:\n")
    print(df.isna().sum())

    print("\nRemoving missing rows...\n")

    df = df.dropna()

    print(f"Remaining rows: {len(df)}")

    return df