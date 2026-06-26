from src.model_plots import save_evaluation_plots

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

from src.tuning import tune_xgboost

from src.evaluation import (
    evaluate_model,
    save_results
)

from src.visualization import plot_feature_importance

from src.explainability import (
    shap_summary_plot,
    shap_bar_plot
)

from src.utils import save_model

from src.pipeline_builder import build_pipeline


def run_pipeline():

    df = load_data()

    inspect_data(df)

    check_data_quality(df)

    df = clean_data(df)

    churn_distribution(df)

    analyze_all_categorical_features(df)

    df = prepare_features(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_data(X, y)

    preprocessor = build_preprocessor(X_train)

    X_train_processed, X_test_processed, feature_names = preprocess_data(
        preprocessor,
        X_train,
        X_test
    )

    logistic_model = train_logistic_regression(
        X_train_processed,
        y_train
    )

    logistic_results = evaluate_model(
        logistic_model,
        X_test_processed,
        y_test,
        "Logistic Regression"
    )

    save_evaluation_plots(
    logistic_model,
    X_test_processed,
    y_test,
    "logistic"
    )

    save_model(
        logistic_model,
        "logistic_regression.pkl"
    )

    rf_model = train_random_forest(
        X_train_processed,
        y_train
    )

    rf_results = evaluate_model(
        rf_model,
        X_test_processed,
        y_test,
        "Random Forest"
    )

    rf_results = evaluate_model(
    rf_model,
    X_test_processed,
    y_test,
    "Random Forest"
    )

    save_model(
        rf_model,
        "random_forest.pkl"
    )

    xgb_model = tune_xgboost(
        X_train_processed,
        y_train
    )

    pipeline = build_pipeline(
        preprocessor,
        xgb_model
    )

    pipeline.fit(
        X_train,
        y_train
    )

    save_model(
        pipeline,
        "customer_churn_pipeline.pkl"
    )

    xgb_results = evaluate_model(
        xgb_model,
        X_test_processed,
        y_test,
        "XGBoost"
    )

    save_evaluation_plots(
    xgb_model,
    X_test_processed,
    y_test,
    "xgboost"
    )

    save_results([
        logistic_results,
        rf_results,
        xgb_results
    ])

    plot_feature_importance(
        xgb_model,
        feature_names
    )

    shap_summary_plot(
        xgb_model,
        X_train_processed,
        feature_names
    )

    shap_bar_plot(
        xgb_model,
        X_train_processed,
        feature_names
    )