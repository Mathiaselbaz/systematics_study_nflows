import random

import ROOT
import matplotlib.pyplot as plt
import sys

import uproot
import pickle
import numpy as np
from datetime import datetime

if (len(sys.argv) != 3):
    print("Usage: python TTreeConverter.py <input_file> <output_file>")
    sys.exit(1)

# open the ROOT file
input_file_name = sys.argv[1]
# example: "Datasets/12params/configMarginalise_Fit_configOa2021_With_localMC_toy_9Pars_Asimov_nToys_10000.root"
input_file = uproot.open(input_file_name)
print("ROOT file contents")
print(input_file.keys())
tree_name = "margThrow"
tree = input_file[tree_name]
print("TTree branches:")
print(tree.keys())

# read TTree branches
parameters_array = tree["Parameters"].array()
dimension = len(parameters_array[0])
chisquares_array = tree["weightsChiSquare"].array()
NLL = tree["LLH"].array()
NLG = tree["gLLH"].array()
filtered_parameters = [
    params for params, nll, chisquares in zip(parameters_array, NLL, chisquares_array) if
    (len(chisquares) == dimension and not np.isinf(nll))
]
filtered_NLL = [
    nll for nll, chisquares in zip(NLL, chisquares_array) if (len(chisquares) == dimension and not np.isinf(nll))
]

if np.any(np.isinf(NLL)):
    print("There are infinite values in the NLL")
    # Count the number of infinite values
    print("Number of infinite values in NLL:")
    print(np.count_nonzero(np.isinf(NLL)))
    print(f"{np.count_nonzero(np.isinf(NLL)) / len(NLL) * 100}% of the values are infinite. Trashing them.")

filtered_NLG = [
    nlg for nlg, nll, chisquares in zip(NLG, NLL, chisquares_array) if
    (len(chisquares) == dimension and not np.isinf(nll))
]
# overwrite the chisquare array AT THE END
chisquares_array = [
    chisquares for chisquares, nll in zip(chisquares_array, NLL) if (len(chisquares) == dimension and not np.isinf(nll))
]
chisquares_array = np.array(chisquares_array)
# read the parameter names
tnamed_name = "parameterFullTitles"
tnamed = input_file[tnamed_name]
tnamed_titles = list(tnamed)
print("List of Strings:")
print(type(tnamed_titles))
for i in range(len(tnamed_titles)):
    print(f"Parameter {i}: {tnamed_titles[i]}")

# read covariance matrix
covariance_TH2D = input_file["postFitInfo/postFitCovarianceOriginal_TH2D"]
cov_values, cov_x_edges, cov_y_edges = covariance_TH2D.to_numpy()
covariance_matrix = np.array(cov_values)
print("Covariance matrix extracted from TH2D:")
print(covariance_matrix)

# read vector of mean values
f = ROOT.TFile.Open(input_file_name)
tvector = f.Get("bestFitParameters_TVectorD")
print(type(tvector))
print(dir(tvector))
means_vector = np.array(tvector)
print("Means vector extracted from TVectorD:")
print(means_vector)

# print the shapes of the arrays (debug)
print(np.shape(filtered_parameters))
filtered_parameters = np.array(filtered_parameters)
print(filtered_parameters)
print(type(filtered_parameters))

# convert back to the eigen space using inverse cholesky decomposition: x = L^-1 * (y - mu)
shifted_parameters = filtered_parameters - means_vector
cholesky = np.linalg.cholesky(covariance_matrix)
inv_cholesky = np.linalg.inv(cholesky)
#check if shifted_parameters has infinite values
print("Are there infinite values in the shifted parameters?")
print(np.any(np.isinf(shifted_parameters)))
# print(f"LogDeterminant: {np.log(np.linalg.det(cholesky))/12}")
eigen_space = np.array([inv_cholesky @ vector for vector in shifted_parameters])
#check if eigen_space has infinite values
print("Are there infinite values in the eigen space?")
print(np.any(np.isinf(eigen_space)))
print("Are there infinite values in the NLG?")
print(np.any(np.isinf(filtered_NLG)))
print("Are there infinite values in the NLL?")
print(np.any(np.isinf(filtered_NLL)))

# Dictionary combining all elements
data_dict = {
    "data": eigen_space,
    "log_p": filtered_NLL,
    "cov": covariance_matrix,
    "mean": means_vector,
    "par_names": tnamed_titles
}

# save pickle file with the dictionary
output_file_name = sys.argv[2]
with open(output_file_name, "wb") as f:
    pickle.dump(data_dict, f)

dim_to_plot = 12
# Create a grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))  # 3x4 grid for 12 variables
axes = axes.flatten()  # Flatten the axes array for easy indexing

random.seed(datetime.now().timestamp())
# Plot each variable
for i in range(dim_to_plot):
    # Extract a random variable across all arrays
    var_to_plot = random.randrange(dimension)
    variable_data = eigen_space[:, var_to_plot]

    # Plot histogram of the variable in its respective subplot
    usedbins = 20
    axes[i].hist(variable_data, bins=usedbins, color='blue', alpha=0.7, edgecolor='black')
    axes[i].set_title(f"{tnamed_titles[var_to_plot]}")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Frequency")
    # Fit a gaussian on each subplot
    mu = np.mean(variable_data)
    sigma = np.std(variable_data)
    # print out the mean and standard deviation (only 3 decimals)
    print(f"--------------{tnamed_titles[var_to_plot]}--------------")
    print(f"Mean: {mu:.3f}")
    print(f"Stde: {sigma:.3f}")
    # overlap a normal distribution
    xmin, xmax = axes[i].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-0.5 * (x ** 2)) / (np.sqrt(2 * np.pi))
    axes[i].plot(x, p * len(variable_data) * (xmax - xmin) / usedbins, 'k', linewidth=2)
# Adjust layout for better spacing
plt.tight_layout()
plt.savefig("eigen_space_histograms.png")

# Create another grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))  # 3x4 grid for 12 variables
axes = axes.flatten()  # Flatten the axes array for easy indexing
for i in range(dim_to_plot):
    # Extract a random variable across all arrays
    var_to_plot = random.randrange(dimension)
    # Extract the ith variable across all arrays
    variable_data = chisquares_array[:, var_to_plot] - 0.5 * np.square(eigen_space[:, var_to_plot])
    # Plot histogram of the variable in its respective subplot
    axes[i].hist(variable_data, bins=200, color='blue', alpha=0.7, edgecolor='black')
    print("Diff between chisquare and x^2/2")
    print(f"mu: {np.mean(variable_data):.3f}")
    print(f"sigma: {np.std(variable_data):.3f}")
plt.tight_layout()
plt.savefig("chisquare_vs_x2.png")

# draw covariance and correlation matrix as 2d histograms
# Calculate correlation matrix
std_dev = np.sqrt(np.diag(covariance_matrix))  # Standard deviations
correlation_matrix = covariance_matrix / np.outer(std_dev, std_dev)  # Normalize by standard deviations
correlation_matrix[covariance_matrix == 0] = 0  # Handle NaNs from division by zero
# Create a single canvas with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
vmin = np.percentile(covariance_matrix, 0.1)
vmax = np.percentile(covariance_matrix, 99.9)
clipped_matrix = np.clip(covariance_matrix, vmin, vmax)
# Plot covariance matrix
cax1 = axes[0].imshow(clipped_matrix, cmap='seismic', interpolation='nearest')
axes[0].set_title("Covariance Matrix (clipped for visualization)")
fig.colorbar(cax1, ax=axes[0])
# Plot correlation matrix
cax2 = axes[1].imshow(correlation_matrix, cmap='seismic', interpolation='nearest', vmin=-1, vmax=1)
axes[1].set_title("Correlation Matrix")
fig.colorbar(cax2, ax=axes[1])
for ax in axes:
    ax.set_xlabel("Variables")
    ax.set_ylabel("Variables")
plt.tight_layout()
plt.savefig("covariance_and_correlation.png")

# check 2: the NLG and the NLL should be reasonably close
fig, ax = plt.subplots()
ax.hist(filtered_NLL, bins=200, color='red', alpha=0.7, edgecolor='red', label="NLL")
ax.hist(filtered_NLG, bins=200, color='blue', alpha=0.7, edgecolor='blue', label="NLG")
ax.set_title("negative-log-probability for LH and its gaussian approximation")
ax.set_xlabel("negative log probability")
ax.legend()
# save the plots
plt.savefig("NLg_vs_NLL.png")
# plt.show()

# check 3: plot NLg and NLL in 2D histogram
fig, ax = plt.subplots()
h, x_edges, y_edges = np.histogram2d(filtered_NLG, filtered_NLL, bins=200)
# Mask bins with no entries
masked_hist = np.ma.masked_where(h == 0, h)
cmap = plt.cm.get_cmap("viridis").copy()
cmap.set_bad(color='white')
mesh = ax.pcolormesh(x_edges, y_edges, masked_hist.T, cmap=cmap, shading='auto')
# Add a colorbar
cbar = plt.colorbar(mesh, ax=ax)
cbar.set_label("Counts")
# Draw the line at x = y
x_min, x_max = ax.get_xlim()  # Get the current x-axis limits
y_min, y_max = ax.get_ylim()  # Get the current y-axis limits
line_min = max(x_min, y_min)
line_max = min(x_max, y_max)
ax.plot([line_min, line_max], [line_min, line_max], color='red', linestyle='--', linewidth=0.8)
ax.set_xlabel("NLG")
ax.set_ylabel("NLL")
plt.savefig("NLg_vs_NLL_2D.png")
# plt.show()
plt.close()
