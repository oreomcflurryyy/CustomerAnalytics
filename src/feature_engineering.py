import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def prepare_features(df):
    """
    Prepare the dataset for machine learning.
    """

    df = df.copy()

    # customerID is only an identifier
    df = df.drop(columns=["customerID"])

    # Convert target variable to binary
    df["Churn"] = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return df

def split_features_target(df):
    """
    Separate the feature matrix (X) and target vector (y).
    """

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y

def train_test_data(X, y):
    """
    Split the data into training and testing sets.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def build_preprocessor(X):
    """
    Build the preprocessing pipeline.
    """

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    print("\nCategorical Columns")
    print(categorical_columns)

    print("\nNumerical Columns")
    print(numerical_columns)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_columns
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            )
        ]
    )

    return preprocessor

def preprocess_data(preprocessor, X_train, X_test):
    """
    Fit the preprocessor on the training data and transform both
    training and testing datasets.
    """

    X_train_processed = preprocessor.fit_transform(X_train)

    X_test_processed = preprocessor.transform(X_test)

    print("\nPreprocessing Complete!")

    print(f"Training Shape : {X_train_processed.shape}")
    print(f"Testing Shape  : {X_test_processed.shape}")

    return X_train_processed, X_test_processed