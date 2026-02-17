import numpy as np
from pydantic import BaseModel
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Config(BaseModel, frozen=True):
    n_dim: int = 128
    num_few_shots: int = 20
    normalize_embeddings: bool = True
    lr_C: float = 1.0
    lr_solver: str = 'saga'
    lr_max_iter: int = 1000
    lr_class_weight: str = 'balanced'
    oof_folds: int = 5
    knn_k: int = 20                    # Neighborhood size for consistency
    density_k: int = 7                 # For local density
    random_state: int = 42
    alpha_ce: float = 0.6
    alpha_ent: float = 0.4
    beta_knn: float = 0.8
    beta_density: float = 0.2
    prune_fraction: float = 0.1
    importance_weights: tuple[float, float, float] = (0.4, 0.5, 0.1)
    importance_per_class: bool = False  # Whether to sample num_few_shots based on importance within each class


# -----------------------------
# Helpers
# -----------------------------


def standardize_per_class(values, y):
    """Z-score within each class to reduce class-imbalance bias."""
    values = np.asarray(values)
    z = np.zeros_like(values, dtype=float)
    labels = np.unique(y)
    for c in labels:
        idx = (y == c)
        v = values[idx]
        mu = v.mean() if v.size > 0 else 0.0
        sd = v.std(ddof=1) if v.size > 1 else 1.0
        z[idx] = (v - mu) / (sd + 1e-8)
    return z


def entropy_per_sample(proba):
    eps = 1e-12
    p = np.clip(proba, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def fit_lr_proba_oof(X, y, config: Config):
    svd = TruncatedSVD(n_components=config.n_dim, random_state=config.random_state)
    X = svd.fit_transform(X)

    skf = StratifiedKFold(n_splits=config.oof_folds, shuffle=True, random_state=config.random_state)

    proba_train = np.zeros((X.shape[0], len(np.unique(y))), dtype=float)

    for train_idx, hold_idx in skf.split(X, y):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_hold = X[hold_idx]
        pipe = make_pipeline(
            StandardScaler(with_mean=True, with_std=True) if config.normalize_embeddings else 'passthrough',
            LogisticRegression(
                C=config.lr_C,
                solver=config.lr_solver,
                max_iter=config.lr_max_iter,
                class_weight=config.lr_class_weight,
                random_state=config.random_state,
            )
        )
        pipe.fit(X_tr, y_tr)
        proba_hold = pipe.predict_proba(X_hold)
        proba_train[hold_idx] = proba_hold

    return proba_train


def knn_consistency_and_density(X, y, k_consistency=16, k_density=8):
    nbrs_c = NearestNeighbors(n_neighbors=min(k_consistency+1, len(X)), algorithm='auto', metric='cosine').fit(X)
    dists_c, idxs_c = nbrs_c.kneighbors(X, return_distance=True)

    # Exclude self neighbor at position 0
    idxs_c = idxs_c[:, 1:1+k_consistency]

    labels = np.asarray(y)
    same = (labels[idxs_c] == labels[:, None])
    consistency = same.mean(axis=1)

    # Density via mean distance to first k_density neighbors (excluding self)
    nbrs_d = NearestNeighbors(n_neighbors=min(k_density+1, len(X)), algorithm='auto', metric='cosine').fit(X)
    dists_d, idxs_d = nbrs_d.kneighbors(X, return_distance=True)
    d_k = dists_d[:, 1:1+k_density]

    # Negative mean distance => higher value means denser region
    density = -d_k.mean(axis=1)

    return consistency, density


def get_importance_scores(X, y, config: Config | None = None):
    config = config or Config()
    proba_train = fit_lr_proba_oof(X, y, config)
    ent_train = entropy_per_sample(proba_train)
    knn_cons_train, density_train = knn_consistency_and_density(
        X, y,
        k_consistency=config.knn_k,
        k_density=config.density_k
    )
    z_ent = standardize_per_class(ent_train, y)
    z_knn = standardize_per_class(knn_cons_train, y)
    z_dens_inv = standardize_per_class(-density_train, y)  # lower density => higher importance
    return np.vstack((z_ent, z_knn, z_dens_inv)).T
