### Import Key Libraries

import numpy as np

%matplotlib widget
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from scipy.special import gamma, factorial

### Create your Plotting Grid
# taken from the Dirichlet_Distribution_Visualization lab

## ===============================================================
## This function creates the mu simplex that we will plot over
## ===============================================================
## This function takes in N_xy, which is the number of data points
## along one axis of the N_xy by N_xy grid. 
def compute_K3_dirichlet_grid(N_matrix_dim):
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ## Define the x and y vectors as a function of the N_data_points
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    x = np.linspace(0,1,N_matrix_dim)
    y = np.linspace(0,1,N_matrix_dim)
    mu_x_temp,mu_y_temp = np.meshgrid(x,y)
    
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ## Define the values of z implicitly
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ## We are subject to the constriaint mu_x+mu_y+mu_z = 1.
    ## To deal with some very small differences created, we round
    ## and then replace any elements of -0 with 0 in the array.
    mu_z_temp = np.round(1-mu_x_temp-mu_y_temp,10)
    mu_z_temp = np.where(mu_z_temp == -0, 0, mu_z_temp)
    
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ## Create a triangular matrix
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #Constrain the matricies:
    mu_x = np.where(mu_z_temp < 0, np.nan, mu_x_temp)
    mu_y = np.where(mu_z_temp < 0, np.nan, mu_y_temp)
    mu_z = np.where(mu_z_temp < 0, np.nan, mu_z_temp)
    
    return mu_x, mu_y, mu_z

### Define a Function that Implements the Bayesian Update
# based on compute_K3_dirichlet_dist from Dirichlet_Distribution_Visualization lab

def bayesian_update(posterior, mu_x, mu_y, mu_z, alphas, new_data, i):
    # update the alphas with the new data
    alphas = alphas + new_data
    
    # compute dirichlet distribution for K=3
    numerator = gamma(np.sum(alphas[i]))
    denominator = gamma(alphas[0]) * gamma(alphas[1]) * gamma(alphas[2])
    mus_raised = (mu_x ** (alphas[0] - 1)) * (mu_y ** (alphas[1] - 1)) * (mu_z ** (alphas[2] - 1))
    dirichlet_dist = (numerator / denominator) * mus_raised
    
    # clean up for visualization
    # we want to make sure that the function is only defined in the same region as mu_x, mu_y, and mu_z:
    dirichlet_dist = np.where(np.isnan(mu_z), np.NaN, dirichlet_dist)
    
    # when any mu = 0, the dirichlet distribution evaluates to inf. This is not plottable in MATPLOTLIB.
    # instead, we will find the largest value of the distribution that is not a NaN or inf. 
    # NumPy provides a function to do just that:
    max_val = np.nanmax(dirichlet_dist[dirichlet_dist != np.inf])
    
    # and replace any values of inf with 3x the max value found:
    dirichlet_dist = np.where(dirichlet_dist == float("inf"), 3 * max_val, dirichlet_dist)

    # store it - FIX ME, THIS ISN'T ACCESSIBLE BY THE PLOTTER
    posterior = np.vstack((posterior, dirichlet_dist))
    
    # move the iterator forward
    i += 1
    
    return posterior, alphas, i

### Define a Plotting Function - unchanged from given
## ===============================================================
## This function plots the Dirichlet Distribution:
## ===============================================================
## mu_x and mu_y are N_matrix_dim x N_matrix_dim numpy arrays
## That implement a triangular plotting region, and implicitly
## define mu_z

## dirichlet_dist is a N_matrix_dim x N_matrix_dim x 3 matrix that 
## holds an N_matrix_dim x N_matrix_dim frame for each iteration,
## starting from iteration 0, which sets the prior. This will let us
## create multiple subplots with this function.

def plot_K3_Dirichlet_dist(mu_x,mu_y,dirichlet_dist):
     ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ## Find the min and max of the distribution for the color bar
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    dist_min = np.nanmin(dirichlet_dist)
    dist_max = np.nanmax(dirichlet_dist)
    
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ## Create a new 3D figure:
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    fig = plt.figure(figsize=(10,4))

    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ## Plot Iteration 0
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')  
    surf1 = ax1.plot_surface(mu_x, mu_y, dirichlet_dist[:,:,0], cmap=cm.cividis, linewidth=0, alpha = 0.5, vmin = dist_min, vmax = dist_max)    
    ax1.set_title(r'Iteration 0', pad=20)
    ax1.set_xlabel(r'$\mu_{1}$')
    ax1.set_ylabel(r'$\mu_{2}$')
    ax1.set_zlabel(r'$Dir(\vec{\mu}|\vec{\alpha})$',rotation='horizontal')
    ax1.set_ylim(1,0)
    ax1.set_xlim(0,1)
    
    
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ## Plot Iteration 1 
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ax2 = fig.add_subplot(1, 3, 2, projection='3d') 
    surf2 = ax2.plot_surface(mu_x, mu_y, dirichlet_dist[:,:,1], cmap=cm.cividis, linewidth=0, alpha = 0.5, vmin = dist_min, vmax = dist_max)    
    ax2.set_title(r'Iteration 1', pad=20)
    ax2.set_xlabel(r'$\mu_{1}$')
    ax2.set_ylabel(r'$\mu_{2}$')
    ax2.set_zlabel(r'$Dir(\vec{\mu}|\vec{\alpha})$',rotation='horizontal')
    ax2.set_ylim(1,0)
    ax2.set_xlim(0,1)
    
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ## Plot Iteration 2
    ## +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')  
    surf3 = ax3.plot_surface(mu_x, mu_y, dirichlet_dist[:,:,2], cmap=cm.cividis, linewidth=0, alpha = 0.5, vmin = dist_min, vmax = dist_max)    
    ax3.set_title(r'Iteration 2', pad=20)
    ax3.set_xlabel(r'$\mu_{1}$')
    ax3.set_ylabel(r'$\mu_{2}$')
    ax3.set_zlabel(r'$Dir(\vec{\mu}|\vec{\alpha})$',rotation='horizontal')
    ax3.set_ylim(1,0)
    ax3.set_xlim(0,1)

    return

### Implement your Bayesian Update
# set matrix dimensions for mu
N_matrix_dim = 11 # for testing, 128 for the real run
# generate the meshgrid for mu_x, mu_y, and mu_z
mu_x, mu_y, mu_z = compute_K3_dirichlet_grid(N_matrix_dim)
# generate alphas to start with
alphas = np.zeros(3)
# create data structure for the posterior - FIXME, this won't work
posterior = np.zeros([1, len(mu_x), len(mu_x)])
# store the data of the various iterations
data = np.array([[3, 3, 15], [4, 30, 2], [55, 3, 5]])

# we're on iteration 0
i = 0

# initialize with the prior
posterior, alphas, i = bayesian_update(None, mu_x, mu_y, mu_z, alphas, data[i], i)
# and run for iterations 1 and 2
posterior, alphas, i = bayesian_update(posterior, mu_x, mu_y, mu_z, alphas, data[i], i)
posterior, alphas, i = bayesian_update(posterior, mu_x, mu_y, mu_z, alphas, data[i], i)
# yes, this could have been a loop, but then tracking i like this would have felt silly
# and i am not sure how much it matters that the guidelines specify we should track i like this
# plot it
plot_K3_Dirichlet_dist(mu_x,mu_y,posterior)
