import numpy as np
from pymatgen.core import Structure


def predict_from_structure_file(path,
                                model,
                                feature_engine,
                                scaler_y=None,
                                unit_scale=1.0):
    """Load CIF/POSCAR file and make property prediction with uncertainty"""
    try:
        structure = Structure.from_file(path)
    except Exception as e:
        return {"error": f"Failed to load structure: {e}"}

    X_latent = feature_engine.transform([structure])
    t = np.zeros(1)

    z_pred, z_std = model.predict(X_latent, t)

    if scaler_y is not None:
        log_pred = scaler_y.inverse_transform(z_pred.reshape(-1, 1))[0, 0]
        log_upper = scaler_y.inverse_transform((z_pred + z_std).reshape(-1, 1))[0, 0]
        log_lower = scaler_y.inverse_transform((z_pred - z_std).reshape(-1, 1))[0, 0]

        val_pred = 10 ** log_pred
        uncertainty = (10 ** log_upper - 10 ** log_lower) / 2
    else:
        val_pred = z_pred[0]
        uncertainty = z_std[0]

    return {
        "prediction": val_pred * unit_scale,
        "uncertainty": uncertainty * unit_scale,
        "confidence": (
            "Low" if z_std[0] > 1.2 else
            "Medium" if z_std[0] > 0.8 else
            "High"
        )
    }