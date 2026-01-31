# Structure-Informed Material ML

This repository implements an undergraduate-level **structure-informed machine learning framework** for predicting material properties directly from crystal structure data. The workflow combines explicit geometric feature construction with probabilistic regression using **Gaussian Process Regression (GPR)**.

The project is designed as an **educational prototype**, emphasizing transparency of the mathematical and algorithmic steps rather than research-grade predictive accuracy.

---

## 1. Mathematical Framework

### 1.1 Problem Setup
Each material is represented by a crystal structure $S$, which is deterministically mapped to a numerical feature vector:

$$
\Phi: S \rightarrow \mathbf{x} \in \mathbb{R}^d
$$

The goal is to learn a function $f(\mathbf{x})$ that predicts a scalar target property $y$ (e.g. a phonon-related or elastic quantity).

### 1.2 Gaussian Process Regression (GPR)

The observed target value is modeled as:

$$
y = f(\mathbf{x}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_{n}^2)
$$

A Gaussian Process prior is placed on the latent function:

$$
f(\mathbf{x}) \sim \mathcal{GP}(0, k(\mathbf{x}, \mathbf{x}'))
$$

where $k(\mathbf{x}, \mathbf{x}')$ is a kernel function encoding similarity between materials in feature space.

### 1.3 Kernel Choice

This implementation uses a **Matérn kernel** with smoothness parameter $\nu = 2.5$, combined with a white-noise kernel to model observation noise:

$$
k_{\nu}(\mathbf{x}, \mathbf{x}') = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \sqrt{2\nu} \frac{d(\mathbf{x}, \mathbf{x}')}{\ell} \right)^\nu K_{\nu}\left( \sqrt{2\nu} \frac{d(\mathbf{x}, \mathbf{x}')}{\ell} \right)
$$

where:
- $d(\mathbf{x}, \mathbf{x}')$ is the Euclidean distance in feature space
- $\ell$ is the characteristic length scale
- $K_{\nu}$ is the modified Bessel function of the second kind

### 1.4 Predictive Distribution

Given training data $(\mathbf{X}, \mathbf{y})$ and a new input $\mathbf{x}_{\star}$, the predictive posterior is Gaussian:

$$
p(f_{\star} \mid \mathbf{x}_{\star}, \mathbf{X}, \mathbf{y}) = \mathcal{N}(\bar{f}_{\star}, \mathbb{V}[f_{\star}])
$$

with:

$$
\bar{f}_{\star} = \mathbf{k}_{\star}^T (\mathbf{K} + \sigma_{n}^2 \mathbf{I})^{-1} \mathbf{y}
$$

$$
\mathbb{V}[f_{\star}] = k(\mathbf{x}_{\star}, \mathbf{x}_{\star}) - \mathbf{k}_{\star}^T (\mathbf{K} + \sigma_{n}^2 \mathbf{I})^{-1} \mathbf{k}_{\star}
$$

This provides both a mean prediction and a model-based uncertainty estimate.

---

## 2. Feature Engineering

Each crystal structure is converted into numerical descriptors using explicit geometric and statistical formulas prior to machine learning.

### 2.1 Global Structural Descriptors
- Unit-cell volume $V$
- Mass density:
$$
\rho = \frac{\sum_{i} m_{i}}{V}
$$
- Number of atoms per unit cell

### 2.2 Local Bonding Descriptors (Voronoi Analysis)

For each atom $i$, the coordination number is defined as the count of Voronoi neighbors:

$$
\mathrm{CN}_{i} = \text{count}\{\text{Voronoi neighbors of atom } i\}
$$

Statistical summaries are then computed:

$$
\mu_{\mathrm{CN}} = \frac{1}{N} \sum_{i=1}^{N} \mathrm{CN}_{i}
$$

$$
\sigma_{\mathrm{CN}} = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} (\mathrm{CN}_{i} - \mu_{\mathrm{CN}})^2 }
$$

Interatomic bond lengths are calculated using:

$$
d_{ij} = \lVert \mathbf{r}_{i} - \mathbf{r}_{j} \rVert
$$

Mean, standard deviation, minimum, and maximum bond lengths are used as features.

### 2.3 Chemical Descriptors

Composition-averaged atomic properties (e.g. atomic number $Z$, atomic mass) are computed as:

$$
\langle P \rangle = \sum_{\alpha} x_{\alpha} P_{\alpha}
$$

where $x_{\alpha}$ is the fractional abundance of element $\alpha$.

### 2.4 Feature Scaling and Dimensionality Reduction

All features are standardized:

$$
x' = \frac{x - \mu}{\sigma}
$$

Principal Component Analysis (PCA) is then applied:

$$
\mathbf{z} = W^T \mathbf{x}'
$$

The reduced feature vector $\mathbf{z}$ is used as input to the Gaussian Process model.

---

## 3. Scope and Limitations

- All physics enters **indirectly** through structure-derived descriptors.
- No explicit elasticity or lattice-dynamical theory is embedded.
- Uncertainty represents **model confidence**, not experimental error.
- Alloy predictions are heuristic and illustrative only.

This framework is intended for **educational and exploratory use**.
