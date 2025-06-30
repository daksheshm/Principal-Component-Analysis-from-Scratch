import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from pca_with_svd import pca 

data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_components = 2
X_proj, pcs, svals, X_recon = pca(
    X_scaled,
    n_components=n_components,
    feature_names=feature_names,
    plot=True,
    compare_with_sklearn=True
)

print("Original shape:", X_scaled.shape)
print("Projected shape:", X_proj.shape)
print(f"\nFirst principal component:\n{pcs[0]}")
print(f"\nSecond principal component:\n{pcs[1]}")

explained_variance = (svals ** 2) / np.sum(svals ** 2)
print(f"\nExplained variance (first {n_components} PCs):")
for i in range(n_components):
    print(f"  PC{i+1}: {explained_variance[i]*100:.2f}%")
print(f"Total: {np.sum(explained_variance[:n_components])*100:.2f}%")

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, cmap='viridis', edgecolor='k', s=70)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Iris Dataset - PCA Projection (2D)')
plt.colorbar(scatter, label='Species')
plt.tight_layout()
plt.show()
