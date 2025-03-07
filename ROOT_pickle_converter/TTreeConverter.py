import random

import ROOT
import matplotlib.pyplot as plt
import sys

import os
os.system('pip install uproot')
import uproot
import pickle
import numpy as np
from datetime import datetime
import os


if (len(sys.argv) != 3):
    print("Usage: python TTreeConverter.py <input_file> <output_file>")
    sys.exit(1)

trans = 1
# There are three types of "transformation" to be applied on the beta samples.
# trans = 0: we apply the inversee cholesky decomposition, so that we transform the beta vector into the eigenspace
# trans = 1: ("r1") we just scale each parameter accoding to its sigma. We do not mix the beta parameters between them.
# trans = 2: ("r2") this is an inversse cholesky decomposition but we set to 0 the correlations betwen gaussian and non gaussian parameters: 
#            BE CAREFUL, the indeces of gaussian and non gaussian parameters is set manually. 



# open the ROOT file
input_file_name = sys.argv[1]
# example: "Datasets/12params/configMarginalise_Fit_configOa2021_With_localMC_toy_9Pars_Asimov_nToys_10000.root"
input_file = uproot.open(input_file_name)
print("---> ROOT file contents")
print(input_file.keys())
tree_name = "margThrow"
tree = input_file[tree_name]
print("---> TTree branches:")
print(tree.keys())
# Check how many times "gundam/version_TNamed" appears in the keys
print("---> Number of times 'gundam/version_TNamed' appears in the keys: ",end='')
scaling_factor_covariance = sum("gundam/build/root/version_TNamed" in key for key in input_file.keys())
print(scaling_factor_covariance)
print("This will be applied as a scaling factor to the covariance matrix")

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
    print("---> There are infinite values in the NLL")
    # Count the number of infinite values
    print("---> Number of infinite values in NLL:")
    print(np.count_nonzero(np.isinf(NLL)))
    print(f"{(np.count_nonzero(np.isinf(NLL)) / len(NLL) * 100):.3f}% of the values are infinite. Trashing them.")

filtered_NLG = [
    nlg for nlg, nll, chisquares in zip(NLG, NLL, chisquares_array) if
    (len(chisquares) == dimension and not np.isinf(nll))
]
# overwrite the chisquare array AT THE END
chisquares_array = [
    chisquares for chisquares, nll in zip(chisquares_array, NLL) if (len(chisquares) == dimension and not np.isinf(nll))
]
chisquares_array = np.array(chisquares_array)

print("---> Number of valid entries:")
print(len(filtered_parameters))
# read the parameter names
tnamed_name = "parameterFullTitles"
tnamed = input_file[tnamed_name]
tnamed_titles = list(tnamed)
print("---> List of Parameter names:")
for i in range(len(tnamed_titles)):
    print(f"Parameter {i}: {tnamed_titles[i]}")

# read covariance matrix
covariance_TH2D = input_file["postFitInfo/postFitCovarianceOriginal_TH2D"]
cov_values, cov_x_edges, cov_y_edges = covariance_TH2D.to_numpy()
covariance_matrix = np.array(cov_values/scaling_factor_covariance)
print("---> Covariance matrix extracted from TH2D:")
print(covariance_matrix)

# read vector of mean values
f = ROOT.TFile.Open(input_file_name)
tvector = f.Get("bestFitParameters_TVectorD")
print(type(tvector))
print(dir(tvector))
means_vector = np.array(tvector)
print("---> Means vector extracted from TVectorD:")
print(means_vector)

# print the shapes of the arrays (debug)
print(np.shape(filtered_parameters))
filtered_parameters = np.array(filtered_parameters)
print(filtered_parameters)
print(type(filtered_parameters))

if(trans==0):
    # convert back to the eigen space using inverse cholesky decomposition: x = L^-1 * (y - mu)
    shifted_parameters = filtered_parameters - means_vector
    transformation_matrix = covariance_matrix
    cholesky = np.linalg.cholesky(covariance_matrix)
    inv_cholesky = np.linalg.inv(cholesky)
    #check if shifted_parameters has infinite values
    eigen_space = np.array([inv_cholesky @ vector for vector in shifted_parameters])
    print("---> Are there infinite values in the transformed parameters? ", end='')
    print(np.any(np.isinf(eigen_space)))
elif(trans==1):
    shifted_parameters = filtered_parameters - means_vector
    sigma_vector = np.sqrt(np.diag(covariance_matrix))
    i, j = np.indices(covariance_matrix.shape)
    mask = (i!=j)
    transformation_matrix = covariance_matrix
    transformation_matrix[mask] = 0
    cholesky = np.linalg.cholesky(transformation_matrix)
    inv_cholesky = np.linalg.inv(cholesky)
    # just use the vector of sigmas to compute the transformed systematics vector
    eigen_space = np.array([ vector/sigma_vector for vector in shifted_parameters])
    print("---> Are there infinite values in the transformed parameters? ", end='')
    print(np.any(np.isinf(eigen_space)))
elif(trans==2):
    shifted_parameters = filtered_parameters - means_vector
    # set to zero the elements of the covariance matrix that mix gaussian and non-gaussian
    gaussian_dimensions = 671
    i, j = np.indices(covariance_matrix.shape)
    mask = ((i >= gaussian_dimensions) & (j < gaussian_dimensions)) | ((i < gaussian_dimensions) & (j >= gaussian_dimensions))
    # Set the selected elements to zero
    transformation_matrix = covariance_matrix
    transformation_matrix[mask] = 0
    cholesky = np.linalg.cholesky(transformation_matrix)
    inv_cholesky = np.linalg.inv(cholesky)
    #check if shifted_parameters has infinite values
    eigen_space = np.array([inv_cholesky @ vector for vector in shifted_parameters])
    print("---> Are there infinite values in the transformed parameters? ", end='')
    print(np.any(np.isinf(eigen_space)))

print(shifted_parameters.shape)
print(cholesky.shape)
for vector in shifted_parameters:
    print(vector.shape)
    print(inv_cholesky.shape)
    print((inv_cholesky @ vector).shape)
    break
print(eigen_space.shape)
print(1/np.std(eigen_space)**2)
print(1/np.std(shifted_parameters)**2)
#check if eigen_space has infinite values
print("---> Are there infinite values in the eigen space?  ", end='')
print(np.any(np.isinf(eigen_space)))
print("---> Are there infinite values in the NLG? ", end='')
print(np.any(np.isinf(filtered_NLG)))
print("---> Are there infinite values in the NLL? ", end='')
print(np.any(np.isinf(filtered_NLL)))

# Dictionary combining all elements
data_dict = {
    "data": eigen_space,
    "log_p": filtered_NLL,
    "cov": covariance_matrix,
    "trans": transformation_matrix,
    "mean": means_vector,
    "par_names": tnamed_titles
}

# # save pickle file with the dictionary
# output_file_name = sys.argv[2]
# with open(output_file_name, "wb") as f:
#     pickle.dump(data_dict, f)

# save npz file with the dictionary
output_file_name = sys.argv[2]
output_file_name = output_file_name + "_r" + str(trans)
np.savez(output_file_name, **data_dict)

dim_to_plot = 12
# Create a grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))  # 3x4 grid for 12 variables
axes = axes.flatten()  # Flatten the axes array for easy indexing

# Create a folder to put the plots, named after the output file name
# Get the current working directory
workdir = os.getcwd()
folder_name = os.path.splitext(os.path.basename(output_file_name))[0]
folder_name = folder_name + "_conversion_plots"
folder_path = os.path.join(workdir, folder_name)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"Folder created: {folder_path}")
else:
    print(f"Folder already exists: {folder_path}")


random.seed(datetime.now().timestamp())
# Plot each variable
for i in range(dim_to_plot):
    # Extract a random variable across all arrays
    var_to_plot = random.randrange(dimension)
    variable_data = eigen_space[:, var_to_plot]

    # Plot histogram of the variable in its respective subplot
    usedbins = 100
    axes[i].hist(variable_data, bins=usedbins, color='white', alpha=0.7, edgecolor='black')
    axes[i].set_title(f"{tnamed_titles[var_to_plot]}")
    axes[i].set_xlabel("Value")
    # Fit a gaussian on each subplot
    mu = np.mean(variable_data)
    sigma = np.std(variable_data)
    # print out the mean and standard deviation (only 3 decimals)
    print(f"--------------{tnamed_titles[var_to_plot]}--------------")
    print(f"Mean: {mu:.3f}")
    print(f"Stde: {sigma:.3f}")
    # overlap a normal distribution
    xmin, xmax = axes[i].get_xlim()
    x = np.linspace(xmin, xmax, usedbins)
    p = np.exp(-0.5 * (x ** 2)) / (np.sqrt(2 * np.pi))
    axes[i].plot(x, p * len(variable_data) * (xmax - xmin) / usedbins, 'k', linewidth=2)
# Adjust layout for better spacing
plt.tight_layout()
plt.savefig(f"{folder_path}/eigen_space_histograms.png")

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
plt.savefig(f"{folder_path}/chisquare_vs_x2.png")

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
plt.savefig(f"{folder_path}/covariance_and_correlation.png")

# check 2: the NLG and the NLL should be reasonably close
fig, ax = plt.subplots()
ax.hist(filtered_NLL, bins=200, color='red', alpha=0.7, edgecolor='red', label="NLL")
ax.hist(filtered_NLG, bins=200, color='blue', alpha=0.7, edgecolor='blue', label="NLG")
ax.set_title("negative-log-probability for LH and its gaussian approximation")
ax.set_xlabel("negative log probability")
ax.legend()
# save the plots
plt.savefig(f"{folder_path}/NLg_vs_NLL.png")
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
plt.savefig(f"{folder_path}/NLg_vs_NLL_2D.png")
# plt.show()
plt.close()
