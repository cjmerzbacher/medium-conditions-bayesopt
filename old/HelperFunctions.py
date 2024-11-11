# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Helper Functions for BayesOpt

# +
# for running in colab
#pip install cobra

# +
# for running in colab
#pip install botorch

# +
# imports
from cobra.io import load_model

import torch
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
# sampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import sample_simplex

# Acquisition function
from botorch.utils.transforms import unnormalize, normalize # for normalising media components
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

import random # for initial data
import numpy as np

# +
# to enable GPU processing
if torch.cuda.is_available():
    #print(f"CUDA is available. Number of devices: {torch.cuda.device_count()}")
    # If you have multiple GPUs, specify the desired device ordinal:
    device = torch.device(f"cuda:0")  # Use GPU 0
else:
    #print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")  

tkwargs = {'device': device, 'dtype': torch.double}
# output 'dtype': torch.float64 bc. in PyTorch double & float64 are equivalent
#print(tkwargs)
# -

# ## Function Definitions

def calc_cost_tot(costs, medium):
    """
    calculates the total cost for a given medium composition
    
    PARAMETERS
    * costs - dictionary - cost for each medium component
    * medium - dictionary - medium composition (component and amount)
    RETURNS
    * a tensor containing the total cost of the medium
    """
    
    cost_tot = sum(concentration * costs[key] for key, concentration in medium.items())
    cost_tot_tensor = torch.tensor([cost_tot], dtype=torch.double).to(**tkwargs) # ensure it is on the device previously decided
    
    return cost_tot_tensor


def generate_initial_data(MetModel, medium, bounds, costs, n_samples = 5):
    """
    Creates initial data points needed to start Bayesian Optimisation
    * randomly creates media compositions within the concentration boundaries.
    * for each medium composition calculates total cost
    * for each performs FBA finding the optimal growth rate
    * stores all in lists

    PARAMETERS
    * MetModel - COBRApy model - the metabolic model to be evaluated
    * medium - dictionary - the medium composition of that model; if not provided defaults to default medium provided by CobraPy
    * bounds - dictionary - upper and lower bounds for the values the medium components are allowed to take,
    determines the search space;
    * costs - dictionary - the (monetary) cost of each component
    * n_samples - integer - how many random media compositions are to be created
    
    RETURNS
    * initial_para - list of dictionaries - random medium compositions
    * initial_cost - tensor - corresponding medium costs
    * initial_grwoth - tensor - corresponding growth rates
    """
    
    # initalise empty lists
    initial_para = []
    initial_cost = []
    initial_growth = []

    for i in range(n_samples):
        # generate random medium parameters within bounds
        random_medium = {} # empty dictionary
        for key in medium.keys():
            lower_bound, upper_bound = bounds[key]
            # Randomly choose a concentration within the provided bounds
            random_medium[key] = random.uniform(lower_bound, upper_bound)
        # Update the model's medium with the randomly generated medium
        MetModel.medium = random_medium
        medium = MetModel.medium

        # caclulate total cost
        cost_tot = calc_cost_tot(costs, medium)
        cost_tot = -cost_tot # BOtorch assumes maximisation, so we maximise the negative of the costs.
        # perform FBA
        growth = MetModel.slim_optimize()
        if np.isnan(growth):
            growth = 0

        # Store the parameters (random medium), total cost, and growth in respective lists
        initial_para.append(random_medium)
        initial_cost.append(cost_tot)
        initial_growth.append(growth)
    
    return initial_para, torch.tensor(initial_cost, dtype=torch.double).to(**tkwargs), torch.tensor(initial_growth, dtype=torch.double).to(**tkwargs)


def initialise_model(medium_tensors_stacked, cost_tensor, growth_tensor):
    """
    Initialises the BO Model using all tried medium compositions and using cost and growth as objectives;
    Will estimate the cost function f

    https://botorch.org/tutorials/constrained_multi_objective_bo
    " We use a multi-output SingleTaskGP to model the two objectives with
    a homoskedastic Gaussian likelihood with an inferred noise level"


    PARAMETERS
    * medium_tensors_stacked - tensor - all previously evaluated medium compositions
    * cost_tensor - tensor - corresponding medium costs
    * growth_tensor - tensor - corresponding growth rates

    RETURNS
    * mll - SumMarginalLikelihoo of the model 
    * model - list of botorch models - List of SingleTaskGP models
    """

    # combine growth and cost tensors into a single tensor
    objective_data = torch.cat((growth_tensor.view(-1,1), cost_tensor.view(-1,1)), dim = -1)

    models = [] # initialise empty list
    for i in range(objective_data.shape[-1]): # in range(2), two "columns" - so for each column
        train_objective = objective_data[:, i] # the column - each being one objective (growth and cost)

        # train a model for the chosen objective and append it to the models list
        models.append(
            SingleTaskGP(medium_tensors_stacked, train_objective.unsqueeze(-1)).to(**tkwargs)
        )

    model = ModelListGP(*models)
    # likelihood of the GP
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    # returns SumMarginalLogLikelihood and model
    return mll, model


def convert_to_dict(candidate_tensor, keys):
    """
    Converts the tensor representation of a medium back to a dictionary

    PARAMETERS
    * candidate_tensor - tensor - values of the medium composition stored in a tensor
    * keys - list - keys corresponding to all possible medium components

    RETURNS
    * candidate_dict - dictionary - a dictionary containing medium components as keys and their amount as values
    """
    
    # Squeeze the tensor to remove extra dimensions if necessary
    candidate_values = candidate_tensor.squeeze().tolist()
    
    # Create a dictionary by pairing the keys with the corresponding values from the tensor
    candidate_dict = {key: value for key, value in zip(keys, candidate_values)}
    
    return candidate_dict
