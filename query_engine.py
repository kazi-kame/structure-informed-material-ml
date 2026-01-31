"""Query Engine Module - Search and rank materials based on constraints"""

import pandas as pd
import numpy as np
import operator
from typing import Dict, Tuple, Callable, Optional, Any


OPS = {
    '<': operator.lt,
    '<=': operator.le,
    '>': operator.gt,
    '>=': operator.ge,
    '==': operator.eq,
    '!=': operator.ne
}


def query_materials(df: pd.DataFrame,
                   model: Any,
                   feature_engine: Any,
                   theory_fn: Optional[Callable] = None,
                   target: str = 'G',
                   constraints: Dict[str, Tuple[str, float]] = None,
                   feature_cols: list = None,
                   top_k: int = 5,
                   sort_by: str = 'predicted',
                   include_uncertainty: bool = True) -> pd.DataFrame:
    """Query materials database and rank by predictions"""
    if constraints is None:
        constraints = {}

    if feature_cols is None:
        raise ValueError("feature_cols must be specified")

    missing_cols = [col for col in constraints.keys() if col not in df.columns]
    if missing_cols:
        print(f"[Query Engine] Warning: Constraint columns not found: {missing_cols}")
        for col in missing_cols:
            del constraints[col]

    mask = np.ones(len(df), dtype=bool)

    for col, (op_str, threshold) in constraints.items():
        if op_str not in OPS:
            print(f"[Query Engine] Warning: Invalid operator '{op_str}', skipping constraint")
            continue

        try:
            mask = mask & OPS[op_str](df[col].values, threshold)
        except Exception as e:
            print(f"[Query Engine] Warning: Failed to apply constraint on {col}: {e}")
            continue

    candidates = df[mask].copy()

    if candidates.empty:
        print("[Query Engine] No materials satisfy the constraints")
        return pd.DataFrame()

    print(f"[Query Engine] {len(candidates)}/{len(df)} materials satisfy constraints")

    try:
        X_latent = feature_engine.transform(candidates[feature_cols].values)
    except Exception as e:
        print(f"[Query Engine] Error transforming features: {e}")
        return pd.DataFrame()

    if theory_fn is not None:
        try:
            t_vals = candidates.apply(theory_fn, axis=1).values
        except Exception as e:
            print(f"[Query Engine] Warning: Theory function failed, using zeros: {e}")
            t_vals = np.zeros(len(candidates))
    else:
        t_vals = np.zeros(len(candidates))

    try:
        pred_mean, pred_std = model.predict(X_latent, t_vals, return_std=True)
    except Exception as e:
        print(f"[Query Engine] Error making predictions: {e}")
        return pd.DataFrame()

    candidates = candidates.copy()
    candidates[f'Predicted_{target}'] = pred_mean

    if include_uncertainty:
        candidates['Uncertainty'] = pred_std

        if pred_std.max() > pred_std.min():
            uncertainty_norm = (pred_std - pred_std.min()) / (pred_std.max() - pred_std.min())
            candidates['Confidence'] = 1.0 - uncertainty_norm
        else:
            candidates['Confidence'] = 1.0

    if target in candidates.columns:
        candidates['Residual'] = np.abs(candidates[target] - candidates[f'Predicted_{target}'])
        candidates['Relative_Error'] = candidates['Residual'] / np.maximum(candidates[target], 1e-10)

    if sort_by == 'predicted':
        candidates = candidates.sort_values(by=f'Predicted_{target}', ascending=False)
    elif sort_by == 'uncertainty':
        candidates = candidates.sort_values(by='Uncertainty', ascending=True)
    elif sort_by == 'actual' and target in candidates.columns:
        candidates = candidates.sort_values(by=target, ascending=False)
    elif sort_by == 'confidence' and include_uncertainty:
        candidates = candidates.sort_values(by='Confidence', ascending=False)
    else:
        candidates = candidates.sort_values(by=f'Predicted_{target}', ascending=False)

    return candidates.head(top_k).reset_index(drop=True)


def compare_predictions_to_actual(results: pd.DataFrame,
                                 target: str = 'G') -> Dict[str, float]:
    """Calculate error metrics for predictions vs actual values"""
    if target not in results.columns:
        return {}

    pred_col = f'Predicted_{target}'
    if pred_col not in results.columns:
        return {}

    y_true = results[target].values
    y_pred = results[pred_col].values

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2,
        'n_samples': len(results)
    }


def find_pareto_frontier(results: pd.DataFrame,
                        objectives: list,
                        maximize: list = None) -> pd.DataFrame:
    """Find Pareto-optimal materials (non-dominated solutions)"""
    if maximize is None:
        maximize = [True] * len(objectives)

    if len(maximize) != len(objectives):
        raise ValueError("Length of maximize must match objectives")

    missing = [obj for obj in objectives if obj not in results.columns]
    if missing:
        raise ValueError(f"Objectives not found in results: {missing}")

    obj_values = results[objectives].values.copy()

    for i, is_max in enumerate(maximize):
        if not is_max:
            obj_values[:, i] = -obj_values[:, i]

    is_pareto = np.ones(len(obj_values), dtype=bool)

    for i, point in enumerate(obj_values):
        if is_pareto[i]:
            is_pareto[is_pareto] = np.any(
                obj_values[is_pareto] > point, axis=1
            )
            is_pareto[i] = True

    return results[is_pareto].copy()