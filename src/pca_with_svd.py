import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def gram_schmidt_qr(A):
    n, m = A.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for j in range(m):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], v)
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j] if R[j, j] > 1e-12 else 0
    return Q, R

def qr_algorithm(A, iters=100):
    A = A.copy()
    n = A.shape[0]
    Q_total = np.eye(n)
    
    for _ in range(iters):
        Q, R = gram_schmidt_qr(A)
        A = R @ Q
        Q_total = Q_total @ Q
    return np.diag(A), Q_total

def svd(A):
    AtA = A.T @ A
    eigen_vals, V = qr_algorithm(AtA)
    S = np.sqrt(np.maximum(eigen_vals, 0))
    
    S_inv = np.zeros_like(S)
    mask = S > 1e-12
    S_inv[mask] = 1 / S[mask]
    U = A @ V @ np.diag(S_inv)

    return U, S, V.T

def pca(X, n_components=2, feature_names=None, plot=False, compare_with_sklearn = False):
    X_mean = np.mean(X, axis=0)
    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = svd(X_centered)

    components = Vt[:n_components]
    X_proj = X_centered @ components.T
    X_reconstructed = X_proj @ components + X_mean

    if plot:
        var_exp = (S ** 2) / np.sum(S ** 2)

        # Scree plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(S)+1), var_exp, alpha=0.7, label='Individual')
        plt.plot(np.cumsum(var_exp), 'r-o', label='Cumulative')
        plt.axhline(y=0.95, linestyle='--', color='g', label='95% Threshold')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Scree Plot')
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Loadings heatmap
        if feature_names is None:
            feature_names = [f'Feature {i+1}' for i in range(X.shape[1])]
        plt.figure(figsize=(12, 6))
        sns.heatmap(components, annot=True, cmap='coolwarm',
                    xticklabels=feature_names,
                    yticklabels=[f'PC{i+1}' for i in range(n_components)])
        plt.title('Principal Component Loadings')
        plt.tight_layout()
        plt.show()

    if compare_with_sklearn:
        from sklearn.decomposition import PCA as SkPCA

        sklearn_pca = SkPCA(n_components=n_components)
        X_sklearn_proj = sklearn_pca.fit_transform(X)
        X_sklearn_recon = sklearn_pca.inverse_transform(X_sklearn_proj)

        error = np.mean(np.abs(X_reconstructed - X_sklearn_recon))
        print(f"\n🔍 Mean Absolute Difference (Our PCA vs sklearn.PCA): {error:.6f}")

    return X_proj, components, S[:n_components], X_reconstructed
