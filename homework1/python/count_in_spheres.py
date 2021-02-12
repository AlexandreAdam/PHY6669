"""
=============================================
Title: 2 point correletion of SDSS sample

Author(s): Alexandre Adam

Last modified: February 11th, 2021

Description: We compute an estimate of the two 
    point correlation function, assuming isotropy of the Universe. 
    This measure will uncover the scale at which the Universe 
    becomes homogeneous.
=============================================
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
import os
from tqdm import tqdm
from sklearn.neighbors import KDTree
from numba import jit
from scipy.stats import gaussian_kde

h = 0.72
Omega_m = 0.3
data_path = "../../data/Skyserver_SQL1_20_2021 12 59 33 AM.csv"

def cone_synthetic_catalogue(kde, theta, phi, N):
    """
    Generate a synthetic catalogue with N object from a uniform density field
    """
    r_rand = kde.resample(N)[0]
    # r_rand = np.random.uniform(r.min(), r.max(), N)
    theta_rand = np.random.uniform(theta.min(), theta.max(), N)
    phi_rand = np.random.uniform(phi.min(), phi.max(), N)
    return r_rand, theta_rand, phi_rand


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def kde_weights(r):
    print("fitting kde")
    kde = gaussian_kde(r)
    w = 1/kde.evaluate(r)
    return w, kde

@jit(nopython=True)
def brute_force(positions):
    N = positions.shape[0]
    size = N * (N - 1) //2
    pair_distance = np.zeros(size)
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            d = positions[i] - positions[j]
            pair_distance[k] = np.sqrt(np.dot(d, d)) 
            k += 1
    return pair_distance

@jit(nopython=True)
def pair_weights(weights):
    N = weights.shape[0]
    w = np.zeros(N * (N - 1)//2)
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            w[k] = weights[i]
            k += 1
    return w

@jit(nopython=True)
def brute_force_cross(positions, positions_rand):
    N = positions.shape[0]
    M = positions_rand.shape[0]
    size = N * M
    pair_distance = np.zeros(size)
    k = 0
    for i in range(N):
        for j in range(M):
            d = positions[i] - positions_rand[j]
            pair_distance[k] = np.sqrt(np.dot(d, d))
            k += 1
    return pair_distance

def estimate_density(positions):
    theta = np.arctan2(np.hypot(positions[:, 0], positions[:, 1]), positions[:, 2])
    phi = np.arctan2(positions[:, 1], positions[:, 0])
    r = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2 + positions[:, 2]**2)
    _r = r.max() - r.min()
    _theta = theta.max() - theta.min()
    _phi = phi.max() - phi.min()
    opening = max(_theta, _phi)
    N = positions.shape[0]
    V = np.pi * _r**3 * np.tan(opening)**2 / 12 # extimate of the volume of the survey
    return N / V # Mpc-3


def two_point_correlation_estimator(positions, positions_rand, weights, bins, method="brute_force", estimator="count_in_sphere"):
    """
    r, theta and sigma are 
    positions, in real space, of each galaxies in the catalogue
    """
    
    if method=="brute_force":
        # print("using brute_force method to count pairs")
        pair_distance = brute_force(positions)
        pair_distance_rand = brute_force(positions_rand)
        w = pair_weights(weights)
        count, _ = np.histogram(pair_distance, bins=bins, weights=w, density=True)
        count_rand, _ = np.histogram(pair_distance_rand, bins=bins, weights=None, density=True)
        if estimator == "count-in-sphere":
            # transform count into cumulative sum
            cum_count = np.cumsum(count)
            cum_count_rand = np.cumsum(count_rand)
            # get bin centers
            return cum_count / cum_count_rand

        elif estimator == "Peebles-Davis":
            xi = count / count_rand - 1
            return xi

        elif estimator == "Hamilton":
            pairs_cross = brute_force_cross(positions, positions_rand)
            count_cross, _ = np.histogram(pairs_cross, bins=bins, weights=None, density=True)
            xi = count * count_rand / count_cross**2 - 1
            return xi

        elif estimator == "Landy-Szalay":
            pairs_cross = brute_force_cross(positions, positions_rand)
            count_cross, _ = np.histogram(pairs_cross, bins=bins, weights=None, density=True)
            n_D = estimate_density(positions) # global estimaste
            n_R = estimate_density(positions_rand)
            return (count * (n_R / n_D)**2 - 2 * count_cross * (n_R / n_D) + count_rand) / count_rand


    elif method=="kdtree":
        if method != "count_in_sphere":
            print("KDTree can only do count in sphere estimate")
        # print("using KDTree method to query radius")
        kd = KDTree(positions, leaf_size=40)
        kd_r = KDTree(positions_rand, leaf_size=40)

        # take as center only a portion of the cone

        N_cal = np.zeros_like(bins)
        N_cal_rand = np.zeros_like(bins)
        for i, d in enumerate(tqdm(bins)):
            N_cal[i] = (kd.query_radius(positions, d, count_only=True) * weights).sum()/weights.sum()
            N_cal_rand[i] = kd_r.query_radius(positions_rand, d, count_only=True).mean()

        return N_cal / N_cal_rand

    else:
        raise ValueError(f"method {method} is not in available options")


def main(args):
    cosmo = FlatLambdaCDM(100 * h, Omega_m)
    data = pd.read_csv(data_path, skiprows=1)
    # preprocessing 
    data = data[(data["Spec_redshift"] > 1e-4) & (data["Spec_redshift"] < 1)]
    M = int(args.fraction * len(data)) # sample size
    if args.fraction < 1:
        indexes = np.random.choice(range(len(data)), size=M, replace=False)
        data = data.iloc[indexes, :]

    # weights = redshift_weigth(data["Spec_redshift"])
    r = cosmo.comoving_distance(data["Spec_redshift"].to_numpy()).value # remove unit to accelerate compute speed
    theta = np.deg2rad(data["dec"]).to_numpy()
    phi = np.deg2rad(data["ra"]).to_numpy()
    bins = np.linspace(20, 500, args.bins) # Mpc
    bin_center = (bins[1:] + bins[:-1])/2

    weights, kde = kde_weights(r)
    x, y, z = spherical_to_cartesian(r, theta, phi)
    positions = np.column_stack([x, y, z])

    N = r.size # number of galaxies in the catalogue
    r_rand, theta_rand, phi_rand = cone_synthetic_catalogue(kde, theta, phi, N)
    positions_rand = np.column_stack(spherical_to_cartesian(r, theta_rand, phi_rand))

    N_cal = two_point_correlation_estimator(positions, positions_rand, weights, bins=bins, method=args.method, estimator=args.estimator)
    plt.plot(bin_center, N_cal)
    np.savetxt(f"Ncal_{M}_{args.method}.txt", N_cal)
    plt.xlabel(fr"$r$ [Mpc]")
    plt.ylabel(r"$\mathcal{N}(<r)$")
    plt.title(f"Pour {M} objets selectiés du catalogue")
    plt.savefig(f"../tex/figures/count_in_sphere_ratio_{M}_{args.method}.png")
    plt.show()



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-f", "--fraction", default=0.1, type=float, required=False, help="Fraction of the dataset to perform analysis on")
    parser.add_argument("-b", "--bins", default=250, type=int, required=False, help="Number of bins for the distance histogram")
    parser.add_argument("-m", "--method", default="brute_force", type=str, required=False, help="Method to compute count, can be brute_force or kdtree")
    parser.add_argument("-e", "--estimator", default="count-in-sphere", type=str, required=False, help="Estimator used")
    args = parser.parse_args()
    main(args)

