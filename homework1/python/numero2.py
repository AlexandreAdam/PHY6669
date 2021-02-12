"""
=============================================
Title: Two point correlation of BOSS sample (2)

Author(s): Alexandre Adam

Last modified: Feb. 11th 2021

Description: We make a collection of random 
    catalogue and compute error bars for different
    estimators fof the correlation function
=============================================
""" 
from count_in_spheres import *
import astropy.units as u
from glob import glob
import matplotlib.pylab as pylab
plt.style.use("science")
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'
}
pylab.rcParams.update(params)


SIZE = 8000

def create_mocks_population(kde, theta, phi, N, population):
    """
    kde: kde instance fitted on radial distance in the survey
    N: Number of samples in each mock catalogue
    population: Number of catalogue to produce
    """
    for i in tqdm(range(population)):
        r, _theta, _phi = cone_synthetic_catalogue(kde, theta, phi, N)
        x, y, z = spherical_to_cartesian(r, _theta, _phi)
        positions = np.column_stack([x, y, z]) 
        np.savetxt(f"mocks/mock_sample_{i:03d}.txt", positions)

def correlation_plot(xi, bins, args, color="b"):
    bin_center = (bins[1:] + bins[:-1])/2
    xi_mean = xi.mean(axis=0)
    xi_std = xi.std(axis=0)
    plt.figure()
    plt.plot(bin_center, xi.mean(axis=0), color=color, lw=3)
    plt.fill_between(bin_center, xi_mean + xi_std, xi_mean - xi_std, color=color, alpha=0.5)
    plt.xlabel(r"$r$ [Mpc]")
    if args.estimator == "count-in-sphere":
        plt.ylabel(r"$\mathcal{N}(<r)$")
        plt.axhline(1, color="k", ls="--")
        plt.title("Compte des voisins")
    else:
        plt.ylabel(r"$\xi(r)$")
        plt.axhline(0, color="k", ls="--")
        plt.title(args.estimator)
    plt.savefig(f"../tex/figures/coorelation_{args.estimator}_8000.png")
    plt.show()



def main(args):

    bins = np.linspace(20, 400, args.bins)
    if args.plot_results:
        xi = np.loadtxt(f"correlation_{args.estimator}_8000.txt")
        correlation_plot(xi, bins, args)
        return

    h = args.hubble
    cosmo = FlatLambdaCDM(100 * h, Omega_m)
    data = pd.read_csv(data_path, skiprows=1)
    # preprocessing 
    data = data[(data["Spec_redshift"] > 1e-4) & (data["Spec_redshift"] < 2)]
    r = cosmo.comoving_distance(data["Spec_redshift"].to_numpy()).value # Mpc
    theta = np.deg2rad(data["dec"]).to_numpy()
    phi = np.deg2rad(data["ra"]).to_numpy()
    weights, kde = kde_weights(r)
    if args.create_mocks:
        create_mocks_population(kde, theta, phi, SIZE, args.population)
        return 


    positions = np.column_stack(spherical_to_cartesian(r, theta, phi)).astype(np.float64)
    xi = []
    for f in tqdm(glob("mocks/mock_sample_*.txt")):
        positions_rand = np.loadtxt(f).astype(np.float64)
        indexes = np.random.choice(range(len(data)), size=SIZE, replace=False)
        _xi = two_point_correlation_estimator(positions[indexes], positions_rand, weights[indexes], bins, estimator=args.estimator)
        xi.append(_xi)

    xi = np.row_stack(xi)
    np.savetxt(f"correlation_{args.estimator}_8000.txt", xi)



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--hubble", required=False, default=0.6774, type=float, help="Hubble parameter")
    parser.add_argument("--omega_m", required=False, default=0.3089, help="Matter density parameter")
    parser.add_argument("--create_mocks", action="store_true", required=False, help="Creat mocks, required for first use")
    parser.add_argument("--population", required=False, default=100, help="Number of mocks to create")
    parser.add_argument("--bins", required=False, default=200, help="Number of bins for the distance")
    parser.add_argument("--estimator", required=False, default="count-in-sphere", help="Estimator, can be count_in_sphere, Peebles-Davis, Hamilton, Landy-Szalay")
    parser.add_argument("--plot_results", required=False, action="store_true", help="Plot results after simulation")
    args = parser.parse_args()
    main(args)

