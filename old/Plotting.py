# ---
# jupyter:
#   jupytext:
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

# # Plotting functions for the various BayesOpt Implementations

# imports
from matplotlib.cm import ScalarMappable
import torch
import numpy as np
from matplotlib import pyplot as plt
from botorch.utils.multi_objective.pareto import is_non_dominated

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

def plot_pareto(results, figname = "figure.png"):
    """
    Plots growth rate (x-axis) against medium cost (y-axis) for each candidate medium
    Each dot is colour-coded according to the iteration it resulted from.
    Plots Pareto front deduced from data.
    Saves the figure (as png file)

    PARAMETERS
    * results - dictionary - output of media_BayesOpt
    * figname - string - name under which to save the figure
    
    RETURNS
    - 
    """
    
    # extract data from results (growth rate and medium costs)
    growth_rates = results['growth rate tensors'].cpu().numpy() # are positive
    medium_costs = results['cost tensor'].cpu().numpy() # are positive
    
    # Stack the two objectives (growth rate and medium cost) into a single 2Darray
    # rows: candidates
    # columns: grwoth rate, medium costs
    y = np.column_stack([growth_rates, medium_costs])
    
    # Define batch numbers (iterations)
    N_ITER = len(growth_rates) # number of candidate mediums = length of growth_rate array
    iterations = np.arange(1, N_ITER + 1)  # Create iteration numbers for each sample
    
    # Create the plot with given size
    fig, axes = plt.subplots(1, 1, figsize = (10, 7))
    
    # get the colormap
    cm = plt.colormaps.get_cmap('viridis')
    
    # Scatter plot of all points, color-coded by iteration (c = iterations)
    # apply colour moa (cmap = cm) and transparence (alpha = 0.8)
    sc = axes.scatter(y[:, 0], y[:, 1], c = iterations, cmap = cm, alpha = 0.8)
    
    """
    Pareto Front
    """
    # negate growth rate because pareto front assumes that minimisation is the goal
    y[:, 0] = -y[:, 0]
    # Sort points by the first objective (growth rate) 
    # -> allows to plot front in order of increasing growth rate
    y_sorted = y[np.argsort(y[:, 0])]
    
    # Compute non-dominated (Pareto front) points; i.e. optimal trade.offs
    is_pareto = is_non_dominated(torch.tensor(-y_sorted).to(**tkwargs))

    # Plot the Pareto front
    axes.plot(
        [-y[0] for pareto, y in zip(is_pareto, y_sorted) if pareto], # negate again so it's back o orig. value
        [y[1] for pareto, y in zip(is_pareto, y_sorted) if pareto],
        label="Pareto Front",
        color="r",
        linewidth=2,
    )
    
    # Add labels and titles
    axes.set_xlabel("Growth Rate [1/h]")
    axes.set_ylabel("Medium Cost")
    axes.set_title("Growth Rate vs Medium Cost with Pareto Front")
    
    # Normalize the color bar according to iteration
    norm = plt.Normalize(iterations.min(), iterations.max())
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    
    # Add the color bar
    cbar = fig.colorbar(sm, ax=axes)
    cbar.ax.set_title("Iteration")
    
    # Display the legend
    axes.legend()
    
    # Show the plot
    #plt.show()

    # Save the figure
    figname = figname
    fig.savefig(figname, dpi=fig.dpi)


def plot_growth_per_cost(results, figname = "figure.png"):
    """
    Plots growth rate per cost (x-axis) against iteration for each candidate medium
    Each dot is colour-coded according to the iteration it resulted from.
    Saves the figure (as png file)

    PARAMETERS
    * results - dictionary - output of media_BayesOpt
    * figname - string - name under which to save the figure
    
    RETURNS
    - 
    """
    
    # extract data from results (growth rate and medium costs)
    growth_rates = results['growth rate tensors'].cpu().numpy() # are positive
    medium_costs = results['cost tensor'].cpu().numpy() # are positive
    
    #growth_costs = calc_growth_cost(growth_rates, medium_costs)
    growth_costs = np.divide(growth_rates, medium_costs, out=np.zeros_like(growth_rates), where=medium_costs!=0)
    
    
    # Define batch numbers (iterations)
    N_ITER = len(growth_costs) # number of candidate mediums = length of growth_rate array
    iterations = np.arange(1, N_ITER + 1)  # Create iteration numbers for each sample
    
    # Create the plot with given size
    fig, axes = plt.subplots(1, 1, figsize = (10, 7))
    
    # get the colormap
    cm = plt.colormaps.get_cmap('viridis')
    
    # Scatter plot of all points, color-coded by iteration (c = iterations)
    # apply colour moa (cmap = cm) and transparence (alpha = 0.8)
    sc = axes.scatter(x = iterations, y = growth_costs, c = iterations, cmap = cm, alpha = 0.8)
    
    # Add labels and titles
    axes.set_xlabel("Iteration")
    axes.set_ylabel("Growth [1/h] per Cost")
    axes.set_title("Growth [1/h] per Cost for each tested medium composition")
    
    # Normalize the color bar according to iteration
    norm = plt.Normalize(iterations.min(), iterations.max())
    sm = ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    
    # Add the color bar
    cbar = fig.colorbar(sm, ax=axes)
    cbar.ax.set_title("Iteration")
    
    # Show the plot
    #plt.show()

    # Save the figure
    figname = figname
    fig.savefig(figname, dpi=fig.dpi)
