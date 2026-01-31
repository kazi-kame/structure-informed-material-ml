"""Structure-aware feature engine for elastic property prediction"""

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pymatgen.core import Structure
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from typing import List


class StructureFeatureEngine:
    """
    Extract fixed-length numerical features from pymatgen Structure objects.
    Uses global, local bonding, and chemical environment descriptors.
    """

    def __init__(self, n_components: int = 10):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.n_components = n_components
        self._is_fitted = False
        self.voronoi = VoronoiNN()

    def _global_features(self, structure: Structure) -> List[float]:
        """Global structure-level descriptors"""
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        return [
            structure.volume,
            len(structure),
            structure.density,
            sga.get_space_group_number()
        ]

    def _local_bond_features(self, structure: Structure) -> List[float]:
        """Local bonding descriptors aggregated over all sites"""
        bond_lengths = []
        coordination_numbers = []

        for i, site in enumerate(structure):
            try:
                neighbors = self.voronoi.get_nn_info(structure, i)
            except Exception:
                continue

            coordination_numbers.append(len(neighbors))

            for nn in neighbors:
                bond_lengths.append(nn["site"].distance(site))

        if len(bond_lengths) == 0:
            return [0.0] * 6

        return [
            np.mean(coordination_numbers),
            np.std(coordination_numbers),
            np.mean(bond_lengths),
            np.std(bond_lengths),
            np.min(bond_lengths),
            np.max(bond_lengths),
        ]

    def _chemical_environment_features(self, structure: Structure) -> List[float]:
        """Chemical composition and atomic properties"""
        Z = [site.specie.Z for site in structure]
        atomic_weights = [site.specie.atomic_mass for site in structure]

        return [
            np.mean(Z),
            np.std(Z),
            np.mean(atomic_weights),
        ]

    def _featurize_structure(self, structure: Structure) -> np.ndarray:
        """Combine all feature types"""
        features = []
        features.extend(self._global_features(structure))
        features.extend(self._local_bond_features(structure))
        features.extend(self._chemical_environment_features(structure))
        return np.array(features, dtype=float)

    def fit(self, structures: List[Structure]) -> "StructureFeatureEngine":
        print("[StructureFeatureEngine] Extracting structure features...")

        X = np.array([self._featurize_structure(s) for s in structures])

        if len(X) < 5:
            raise ValueError("Too few structures to fit feature engine")

        X_scaled = self.scaler.fit_transform(X)

        max_components = min(X_scaled.shape[0], X_scaled.shape[1])
        if self.n_components > max_components:
            self.n_components = max_components
            self.pca = PCA(n_components=self.n_components)

        self.pca.fit(X_scaled)

        explained = np.sum(self.pca.explained_variance_ratio_)
        print(f"[StructureFeatureEngine] PCA({self.n_components}) explains {explained*100:.1f}% variance")

        self._is_fitted = True
        return self

    def transform(self, structures: List[Structure]) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Feature engine not fitted")

        X = np.array([self._featurize_structure(s) for s in structures])
        X_scaled = self.scaler.transform(X)
        return self.pca.transform(X_scaled)

    def fit_transform(self, structures: List[Structure]) -> np.ndarray:
        self.fit(structures)
        return self.transform(structures)