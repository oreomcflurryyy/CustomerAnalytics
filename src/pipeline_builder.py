from sklearn.pipeline import Pipeline


def build_pipeline(preprocessor, model):
    """
    Combine preprocessing and model into one pipeline.
    """

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline