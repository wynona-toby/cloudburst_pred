
import ast
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


def clean_model_input(X):
    """Convert scalar or single-element array strings to numeric values."""
    X = X.copy()

    def to_number(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value).reshape(-1)
            if len(arr) != 1:
                raise ValueError(
                    f"Expected one value per feature, received {value!r}"
                )
            value = arr[0]

        if isinstance(value, str):
            value = value.strip()

            # Handle strings such as '[3.1726265E-1]'
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, (list, tuple)):
                        if len(parsed) != 1:
                            raise ValueError(
                                f"Feature contains multiple values: {value}"
                            )
                        value = parsed[0]
                    else:
                        value = parsed
                except (ValueError, SyntaxError):
                    value = value[1:-1].strip()

        return float(value)

    for col in X.columns:
        X[col] = X[col].map(to_number)

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Model inputs contain NaN or infinite values.")

    return X.astype(float)



def explain_xgboost(model, X_latest):
    """Explain an XGBoost regression prediction using native contributions."""
    import xgboost as xgb
    import shap
    import matplotlib.pyplot as plt

    X_numeric = clean_model_input(X_latest)

    booster = model.get_booster()

    dmatrix = xgb.DMatrix(
        X_numeric,
        feature_names=X_numeric.columns.tolist()
    )

    # Last column contains the baseline/bias contribution.
    raw = booster.predict(dmatrix, pred_contribs=True)

    shap_values = raw[0, :-1]
    base_value = float(raw[0, -1])

    contributions = pd.DataFrame({
        "Feature": X_numeric.columns,
        "Current Value": X_numeric.iloc[0].to_numpy(),
        "SHAP Contribution": shap_values
    })

    contributions["Absolute Impact"] = (
        contributions["SHAP Contribution"].abs()
    )

    contributions = contributions.sort_values(
        "Absolute Impact", ascending=False
    )

    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_value,
        data=X_numeric.iloc[0].to_numpy(),
        feature_names=X_numeric.columns.tolist()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(
        explanation,
        max_display=12,
        show=False
    )
    fig.tight_layout()

    return contributions, fig, base_value
