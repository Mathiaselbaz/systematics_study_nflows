import ROOT
import matplotlib.pyplot as plt
import sys

import uproot
import pickle
import numpy as np

if(len(sys.argv) != 3):
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
    params for params, chisquares in zip(parameters_array, chisquares_array) if len(chisquares) == dimension
]
filtered_NLL = [
    nll for nll, chisquares in zip(NLL, chisquares_array) if len(chisquares) == dimension
]
filtered_NLG = [
    nlg for nlg, chisquares in zip(NLG, chisquares_array) if len(chisquares) == dimension
]
# overwrite the chisquare array AT THE END
chisquares_array = [x for x in chisquares_array if len(x) == dimension]
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
print(f"LogDeterminant: {np.log(np.linalg.det(cholesky))/12}")
eigen_space = np.array([inv_cholesky @ vector for vector in shifted_parameters])


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




if(dimension <= 12):


    # Create a grid of subplots
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))  # 3x4 grid for 12 variables
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    # Plot each variable
    for i in range(dimension):
        # Extract the ith variable across all arrays
        variable_data = eigen_space[:, i] 
        
        # Plot histogram of the variable in its respective subplot
        axes[i].hist(variable_data, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axes[i].set_title(f"{tnamed_titles[i]}")
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        # Fit a gaussian on each subplot
        mu = np.mean(variable_data)
        sigma = np.std(variable_data)
        # print out the mean and standard deviation (only 3 decimals)
        print(f"{tnamed_titles[i]}")
        print(f"Mean: {mu:.3f}")
        print(f"Stde: {sigma:.3f}")

    # Hide any unused subplots (if the grid size exceeds 12)
    for j in range(dimension, len(axes)):
        axes[j].axis("off")

    # Adjust layout for better spacing
    plt.tight_layout()
    # plt.show()

    # draw covariance matrix as a 2d histogram
    fig, ax = plt.subplots()
    cax = ax.matshow(covariance_matrix, cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(tnamed_titles)))
    ax.set_yticks(np.arange(len(tnamed_titles)))
    ax.set_xticklabels(tnamed_titles)
    ax.set_yticklabels(tnamed_titles)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # plt.show()

    # check 1: the eigen space squared, divided by 2 should be equivalent to the chisquare vector
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))  # 3x4 grid for 12 variables
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    for i in range(dimension):
        # Extract the ith variable across all arrays
        variable_data = chisquares_array[:, i] - 0.5 * np.square(eigen_space[:, i])
        # Plot histogram of the variable in its respective subplot
        axes[i].hist(variable_data, bins=200, color='blue', alpha=0.7, edgecolor='black')
        print(f"mu: {np.mean(variable_data)}")
        print(f"sigma: {np.std(variable_data)}")

    plt.tight_layout()
    # plt.show()

# check 2: the NLG and the NLL should be reasonably close
fig, ax = plt.subplots()
ax.hist(NLG, bins=200, color='blue', alpha=0.7, edgecolor='black', label="NLG")
ax.hist(NLL, bins=200, color='red', alpha=0.7, edgecolor='black', label="NLL")
ax.set_title("NLg vs. NLL")
ax.set_xlabel("negative log probability")
ax.legend()
print(f"Log of sqrt(2pi = {np.log((np.sqrt(2*np.pi)))}")



plt.show()

