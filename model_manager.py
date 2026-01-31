"""Model Manager Module - Gaussian Process regression for material property prediction"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, WhiteKernel, ConstantKernel, RBF
)
from typing import Tuple, Optional
import warnings


class TaMLModel:
    """
    Theory-augmented Machine Learning (TaML) model using Gaussian Process
    to learn residuals between observed and physics-based predictions.
    """

    def __init__(self,
                 length_scale: float = 1.0,
                 noise_level: float = 0.01,
                 kernel_type: str = 'matern'):
        self.length_scale = length_scale
        self.noise_level = noise_level
        self.kernel_type = kernel_type
        self._is_fitted = False

        self.kernel = self._build_kernel()

        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
            alpha=1e-10,
            random_state=42
        )

    def _build_kernel(self):
        """Build GP kernel based on configuration"""
        k_const = ConstantKernel(
            constant_value=1.0,
            constant_value_bounds=(1e-3, 1e3)
        )

        if self.kernel_type == 'matern':
            k_main = Matern(
                length_scale=self.length_scale,
                length_scale_bounds=(1e-2, 1e4),
                nu=2.5
            )
        elif self.kernel_type == 'rbf':
            k_main = RBF(
                length_scale=self.length_scale,
                length_scale_bounds=(1e-2, 1e4)
            )
        else:
            k_main = (
                Matern(length_scale=self.length_scale, nu=2.5) +
                RBF(length_scale=self.length_scale * 0.5)
            )

        k_noise = WhiteKernel(
            noise_level=self.noise_level,
            noise_level_bounds=(1e-10, 1e0)
        )

        return k_const * k_main + k_noise

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> 'TaMLModel':
        """Fit GP model to training data. y = observed, t = theory predictions"""
        if len(X) != len(y) or len(y) != len(t):
            raise ValueError(f"Inconsistent shapes: X={X.shape}, y={y.shape}, t={t.shape}")

        if len(X) < 3:
            raise ValueError(f"Need at least 3 samples to fit GP, got {len(X)}")

        residuals = y - t

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(X, residuals)

        self._is_fitted = True

        print(f"[Model Manager] GP fitted with {len(X)} samples")
        print(f"[Model Manager] Optimized kernel: {self.gp.kernel_}")

        return self

    def predict(self,
                X: np.ndarray,
                t: np.ndarray,
                return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with uncertainty. Output = theory + learned residuals"""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction. Call fit() first.")

        if len(X) != len(t):
            raise ValueError(f"Inconsistent shapes: X={X.shape}, t={t.shape}")

        if return_std:
            delta_mean, delta_std = self.gp.predict(X, return_std=True)
            y_pred = t + delta_mean
            return y_pred, delta_std
        else:
            delta_mean = self.gp.predict(X, return_std=False)
            y_pred = t + delta_mean
            return y_pred, None

    def get_kernel_params(self) -> dict:
        """Get optimized kernel hyperparameters"""
        if not self._is_fitted:
            return {}

        return {
            'kernel': str(self.gp.kernel_),
            'log_marginal_likelihood': self.gp.log_marginal_likelihood_value_
        }

    def score(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
        """Calculate R² score on test data"""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before scoring")

        y_pred, _ = self.predict(X, t)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return r2


class EnsembleTaMLModel:
    """Ensemble of multiple TaML models for robust predictions"""

    def __init__(self, n_models: int = 3, **model_kwargs):
        self.n_models = n_models
        self.models = [TaMLModel(**model_kwargs) for _ in range(n_models)]
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, t: np.ndarray) -> 'EnsembleTaMLModel':
        """Fit all models in ensemble"""
        for i, model in enumerate(self.models):
            model.gp.random_state = 42 + i
            model.fit(X, y, t)

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using ensemble average with total uncertainty"""
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted first")

        predictions = []
        uncertainties = []

        for model in self.models:
            pred, std = model.predict(X, t)
            predictions.append(pred)
            uncertainties.append(std)

        ensemble_mean = np.mean(predictions, axis=0)

        model_unc = np.mean(uncertainties, axis=0)
        ensemble_var = np.std(predictions, axis=0)
        total_std = np.sqrt(model_unc**2 + ensemble_var**2)

        return ensemble_mean, total_std