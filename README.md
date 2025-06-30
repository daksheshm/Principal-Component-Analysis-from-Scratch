# Principal-Component-Analysis-from-Scratch
    An educational implementation of Principal Component Analysis (PCA) in Python from first principles, exploring SVD, and the underlying QR algorithm for eigensolving.

![Python](https://img.shields.io/badge/python-3.x-blue.svg) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white) ![Matplotlib](https://img.shields.io/badge/matplotlib-%23ffffff.svg?style=flat&logo=matplotlib&logoColor=black) ![Seaborn](https://img.shields.io/badge/seaborn-%23007ACC.svg?style=flat&logo=seaborn&logoColor=white)

An educational, from-scratch implementation of the Principal Component Analysis (PCA) algorithm in Python. This project eschews high-level library functions for core computations, instead building the entire algorithmic stack from the ground up: **Gram-Schmidt -> QR Decomposition -> QR Algorithm (Eigensolver) -> SVD -> PCA**.

The primary goal is to demonstrate a deep, practical understanding of the numerical linear algebra that powers modern machine learning.

## Project Overview

This repository provides a complete, self-contained PCA pipeline. Unlike typical implementations that rely on `np.linalg.svd` or `sklearn.decomposition.PCA` for the heavy lifting, this project implements these transformations from first principles.

The algorithmic journey is as follows:
1.  A **QR Decomposition** function is built using the Gram-Schmidt process.
2.  This decomposition is used to create a **QR Algorithm**, an iterative method for finding the eigenvalues and eigenvectors of a matrix.
3.  The custom eigensolver is then used to implement **Singular Value Decomposition (SVD)** by leveraging its mathematical relationship with the covariance matrix (`A^T * A`).
4.  Finally, the custom **SVD** is used as the engine for the main **PCA** function.

The entire implementation is validated against Scikit-learn's industry-standard PCA to ensure correctness, and includes built-in functions for visualizing and interpreting the results.

## Core Features

-   **Custom Eigensolver:** A from-scratch implementation of the QR algorithm to find eigenvalues and eigenvectors of a symmetric matrix.
-   **Custom SVD:** A full implementation of Singular Value Decomposition powered by the custom eigensolver.
-   **Complete PCA Pipeline:** Performs data centering, projection onto principal components, and data reconstruction.
-   **Built-in Visualization:** Automatically generates key diagnostic plots:
    -   **Scree Plot:** To determine the amount of variance captured by each component.
    -   **Component Loadings Heatmap:** To interpret the relationship between principal components and original features.
-   **Validation Framework:** Includes a direct comparison to `scikit-learn`'s PCA, quantifying the mean absolute difference in reconstruction to verify accuracy.

## Technical Concepts Explored

-   **Linear Algebra:** Eigendecomposition, Singular Value Decomposition, Orthogonality, Gram-Schmidt Process, Change of Basis, Vector Projections.
-   **Numerical Methods:** The QR Algorithm, Iterative Methods for Matrix Factorization.
-   **Machine Learning:** Dimensionality Reduction, Data Preprocessing (Centering), Model Interpretation, Data Reconstruction.

## Visualizations & Results

The implementation was validated on the Iris dataset. The plots below are generated automatically by the main script.

### Scree Plot
This plot clearly shows that the first two principal components capture over 95% of the total variance in the original 4-dimensional dataset, justifying the reduction to 2D for visualization.

### Component Loadings
This heatmap reveals the composition of each principal component. We can interpret that PC1 is primarily a measure of the flower's overall size (high correlation with petal length/width and sepal length), while PC2 contrasts the sepal dimensions.


### 2D Projection
The final projection of the data onto the first two principal components shows excellent separation between the three Iris species.


## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the main script:**
    ```bash
    python main.py
    ```
    The script will print the validation results to the console and display the generated plots.

## Code Structure
```
.
├── main.py                # Main script to load data, run PCA, and show results
├── pca_with_svd.py        # Contains the core library: pca(), svd(), qr_algorithm(), etc.
├── requirements.txt
└── README.md
```
