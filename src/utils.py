import joblib

from src.config import MODEL_DIR


def save_model(model, filename):
    """
    Save a trained model to the models directory.
    """

    filepath = MODEL_DIR / filename

    joblib.dump(model, filepath)

    print(f"✓ Saved model: {filepath}")