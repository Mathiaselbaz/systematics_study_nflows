import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from utils.dataset_class import SystematicDataset

# Ensure output directory exists
output_dir = "img"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# =============================================================================
#  Compute and Plot Pearson Correlation Matrix
# =============================================================================
def compute_pearson_correlation_matrix(dataset, save_full_matrix=True, top_correlated=20):
    """
    Computes the Pearson correlation matrix for all dimensions in the dataset
    and plots it efficiently for large dimensions.

    Args:
        dataset (SystematicDataset): The dataset instance.
        save_full_matrix (bool): Whether to save the full 711x711 correlation matrix.
        top_correlated (int): Number of most correlated dimensions to plot in detail.

    Returns:
        np.ndarray: The Pearson correlation matrix.
    """
    print("Computing Pearson correlation matrix...")

    # Convert dataset to NumPy array for fast computation
    data_np = dataset.data.numpy()

    # Compute Pearson correlation matrix
    corr_matrix = np.corrcoef(data_np, rowvar=False)

    # Save full matrix as a numpy file for later analysis
    np.save(os.path.join(output_dir, "pearson_correlation_matrix.npy"), corr_matrix)

    if save_full_matrix:
        print("Saving full correlation matrix heatmap...")

        # Large figure size to accommodate 711x711 dimensions
        fig, ax = plt.subplots(figsize=(16, 14))

        # Use a lower resolution color map to handle large matrix better
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    cbar_kws={'label': 'Pearson Correlation'}, xticklabels=False, yticklabels=False)

        plt.title("Pearson Correlation Matrix (All 711 Dimensions)")
        plt.savefig(os.path.join(output_dir, "pearson_correlation_matrix_large.png"), dpi=300, bbox_inches="tight")
        plt.close()

    # Extract top correlated pairs
    print("Extracting most correlated dimensions...")
    upper_triangle = np.triu(np.abs(corr_matrix), k=1)  # Ignore diagonal
    top_indices = np.unravel_index(np.argsort(upper_triangle, axis=None)[-top_correlated:], upper_triangle.shape)
    
    # List the top correlated pairs
    top_pairs = [(i, j, corr_matrix[i, j]) for i, j in zip(top_indices[0], top_indices[1])]
    top_pairs.sort(key=lambda x: -abs(x[2]))  # Sort by absolute correlation
    
    # Print the most correlated dimensions
    print("\nTop Most Correlated Dimensions:")
    for i, (dim1, dim2, corr) in enumerate(top_pairs):
        print(f"{i+1}. Dimension {dim1} ↔ Dimension {dim2} | Correlation: {corr:.4f}")

    # Plot a focused heatmap of the most correlated pairs
    print("Saving focused heatmap of most correlated dimensions...")

    top_dim_indices = list(set([dim for pair in top_pairs for dim in pair[:2]]))  # Unique top correlated dimensions
    top_dim_indices.sort()  # Sort for better readability

    if len(top_dim_indices) > 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix[np.ix_(top_dim_indices, top_dim_indices)], 
                    cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                    xticklabels=top_dim_indices, yticklabels=top_dim_indices, 
                    cbar_kws={'label': 'Pearson Correlation'})

        plt.title(f"Pearson Correlation of Top {top_correlated} Most Correlated Dimensions")
        plt.savefig(os.path.join(output_dir, "pearson_correlation_top.png"), dpi=300, bbox_inches="tight")
        plt.close()

    return corr_matrix

# =============================================================================
#  Rank and Plot Least Normal Dimensions
# =============================================================================
def rank_and_plot_least_normal_dimensions(dataset, n_dimensions=5):
    """
    Identifies dimensions that deviate the most from a standard normal distribution N(0,1)
    and plots their distributions along with an overlaid Gaussian.

    Args:
        dataset: An instance of SystematicDataset.
        n_dimensions: Number of least normal dimensions to plot.
    """
    print("Ranking dimensions based on normality...")

    # Compute deviation scores
    scores = []
    for dim in range(dataset.ndim):
        dim_data = dataset.data[:, dim].numpy()

        # Compute mean squared error (MSE) from a standard normal distribution
        empirical_pdf, bins = np.histogram(dim_data, bins=100, density=True)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        normal_pdf = norm.pdf(bin_centers)
        mse = np.mean((empirical_pdf - normal_pdf) ** 2)
        scores.append((dim, mse))

    # Rank dimensions by deviation from normality
    ranked_dimensions = sorted(scores, key=lambda x: x[1])
    least_normal_dims = ranked_dimensions[:n_dimensions]
    print("Least normal dimensions:", least_normal_dims)

    # Plot distributions of least normal dimensions
    print("Saving least normal dimension histograms...")

    fig, axes = plt.subplots(nrows=n_dimensions, figsize=(10, 4 * n_dimensions))
    if n_dimensions == 1:
        axes = [axes]  # Ensure iterable behavior if only 1 dimension

    for i, (dim, score) in enumerate(least_normal_dims):
        dim_data = dataset.data[:, dim].numpy()
        mean, std = np.mean(dim_data), np.std(dim_data)
        x_vals = np.linspace(dim_data.min(), dim_data.max(), 1000)
        gaussian_pdf = norm.pdf(x_vals, loc=mean, scale=std)

        # Plot histogram and Gaussian
        axes[i].hist(dim_data, bins=100, density=True, alpha=0.6, label="Empirical")
        axes[i].plot(x_vals, gaussian_pdf, label=f"Gaussian N({mean:.2f}, {std:.2f})", color='red')
        axes[i].set_title(f"Dimension {dim} (MSE from N(0,1): {score:.4e})")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Density")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "least_normal_dimensions.png"), dpi=300, bbox_inches="tight")
    plt.close()

# =============================================================================
#  Run Analysis
# =============================================================================
if __name__ == "__main__":
    dataset = SystematicDataset('Dataset/pickle_files/configMarginalise_Fit_configOa2021_With_localMC_Asimov_merged_1M.pkl', range(711))

    # Compute and plot correlation matrix
    correlation_matrix = compute_pearson_correlation_matrix(dataset)

    # Rank and plot least normal dimensions
    rank_and_plot_least_normal_dimensions(dataset, n_dimensions=15)
