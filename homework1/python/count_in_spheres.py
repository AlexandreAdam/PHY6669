"""
=============================================
Title: 2 point correletion of SDSS sample

Author(s): Alexandre Adam

Last modified: February 10th, 2021

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

h = 0.72
Omega_m = 0.3
data_path = "../../data/Skyserver_SQL1_20_2021 12 59 33 AM.csv"

def cone_synthetic_catalogue(r, theta, phi, N):
    """
    Generate a synthetic catalogue with N object from a uniform density field
    """
    r_rand = np.random.uniform(r.min(), r.max(), N)
    theta_rand = np.random.uniform(theta.min(), theta.max(), N)
    phi_rand = np.random.uniform(phi.min(), phi.max(), N)
    return r_rand, theta_rand, phi_rand


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def redshift_weigth(z, bins=100):
    _bins = pd.cut(z, bins=np.linspace(z.min(), z.max() + 1, bins))
    w = z.map(1/z.groupby(_bins).sum()) 
    return w.to_numpy()


def center_selection(r, theta, phi, fraction):
    theta_center = (theta.max() + theta.min())/2
    phi_center = (phi.max() + phi.min())/2
    out = theta < (theta.max() + theta_center) * fraction
    out *= theta > (theta_center + theta.min()) * fraction
    out *= phi < (phi.max() + phi_center) * fraction
    out *= phi > (phi_center + phi.min()) * fraction
    return out


@jit
def brute_force(positions, positions_centers, weights):
    N = positions.shape[0]
    M = positions_centers.shape[0]
    size = N*M
    pair_distance = np.zeros(size)
    
    w = np.zeros_like(pair_distance)
    # print(f"There is {N} object in catalog, start counting pair distances")
    k = 0
    for i in range(M):
        for j in range(N):
            d = positions_centers[i] - positions[j]
            pair_distance[k] = np.sqrt(np.dot(d, d))
            w[k] = weights[i]
            k += 1

    w /= w.sum()
    return pair_distance, w


def count_in_sphere_ratio(r, theta, phi, weights=None, bins=30, method="brute_force"):
    """
    r, theta and sigma are 
    positions, in real space, of each galaxies in the catalogue
    """
    x, y, z = spherical_to_cartesian(r, theta, phi)
    positions = np.column_stack([x, y, z])

    N = r.size # number of galaxies in the catalogue
    r_rand, theta_rand, phi_rand = cone_synthetic_catalogue(r, theta, phi, N)
    positions_rand = np.column_stack(spherical_to_cartesian(r_rand, theta_rand, phi_rand))

    selection = center_selection(r, theta, phi, fraction=0.5)
    positions_centers = positions[selection]
    positions_centers_rand = positions_rand[selection]
    weights = weights[selection]
    
    if method=="brute_force":
        print("using brute_force method")
        pair_distance, w = brute_force(positions, positions_centers, weights)
        pair_distance_rand, _ = brute_force(positions_rand, positions_centers_rand, weights)
        count, distance = np.histogram(pair_distance, bins=bins, weights=w, density=True)
        count_rand, distance_rand = np.histogram(pair_distance_rand, bins=distance, density=True)
        # transform count into cumulative sum
        cum_count = np.cumsum(count)
        cum_count_rand = np.cumsum(count_rand)
        # get bin centers
        distance = (distance[1:] + distance[:1])/2
        return cum_count / cum_count_rand, distance

    elif method=="kdtree":
        kd = KDTree(positions, leaf_size=40)
        kd_r = KDTree(positions_rand, leaf_size=40)

        # take as center only a portion of the cone

        distances = np.linspace(10, 500, 200)
        N_cal = np.zeros_like(distances)
        N_cal_rand = np.zeros_like(distances)
        for i, d in enumerate(tqdm(distances)):
            N_cal[i] = (kd.query_radius(positions_centers, d, count_only=True) * weights).sum()/weights.sum()
            N_cal_rand[i] = kd_r.query_radius(positions_centers_rand, d, count_only=True).sum()


        return N_cal / N_cal_rand, distances

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

    weights = redshift_weigth(data["Spec_redshift"])
    r = cosmo.comoving_distance(data["Spec_redshift"].to_numpy()).value # remove unit to accelerate compute speed
    theta = np.deg2rad(data["dec"]).to_numpy()
    phi = np.deg2rad(data["ra"]).to_numpy()
    N_cal, distance = count_in_sphere_ratio(r, theta, phi, weights=weights, bins=args.bins, method=args.method)
    plt.plot(distance, N_cal)
    np.savetxt(f"Ncal_{M}.txt", N_cal)
    np.savetxt(f"Distance_{M}.txt", distance, comments="in Mpc")
    plt.xlabel(fr"$r$ [Mpc]")
    plt.ylabel(r"$\mathcal{N}(<r)$")
    plt.title(f"Pour {M} objets selectionnés du catalogue")
    plt.savefig(f"../tex/figures/count_in_sphere_ratio_{M}.png")
    plt.show()



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-f", "--fraction", default=0.1, type=float, required=False, help="Fraction of the dataset to perform analysis on")
    parser.add_argument("-b", "--bins", default=50, type=int, required=False, help="Number of bins for the distance histogram")
    parser.add_argument("-m", "--method", default="brute_force", type=str, required=False, help="Method to compute count, can be brute_force or kdtree")
    args = parser.parse_args()
    main(args)

