from src.preprocessing import (
    load_data,
    inspect_data,
    check_data_quality,
    clean_data
)
from src.eda import (
    churn_distribution,
    analyze_all_categorical_features
)
from src.feature_engineering import (
    prepare_features,
    split_features_target,
    train_test_data,
    build_preprocessor,
    preprocess_data
)
from src.modeling import (
    train_logistic_regression,
    train_random_forest
)

from src.evaluation import (
    evaluate_model,
    save_results
)
from src.utils import save_model

df = load_data()

inspect_data(df)

check_data_quality(df)

df = clean_data(df)

# ---------- EDA ----------
churn_distribution(df)
analyze_all_categorical_features(df)

# ---------- ML ----------
df = prepare_features(df)

X, y = split_features_target(df)

X_train, X_test, y_train, y_test = train_test_data(X, y)

preprocessor = build_preprocessor(X_train)

X_train_processed, X_test_processed = preprocess_data(
    preprocessor,
    X_train,
    X_test
)

logistic_model = train_logistic_regression(
    X_train_processed,
    y_train
)

save_model(
    logistic_model,
    "logistic_regression.pkl"
)

rf_model = train_random_forest(
    X_train_processed,
    y_train
)

save_model(
    rf_model,
    "random_forest.pkl"
)