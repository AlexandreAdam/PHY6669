import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)
plt.style.use("science")

mu = 2
sigma_true = 1
D = np.random.normal(mu, sigma_true, 1000)

def std_gamma(x, n):
    d = D[:n]
    alpha = np.sqrt(0.5 * np.sum((d - mu)**2))
    beta = n - 1
    return 2 * alpha**(beta) / gamma(beta/2) * (1/x)**(beta + 1) * np.exp(-(alpha/x)**2)

def var_gamma(x, n):
    d = D[:n]
    alpha = 0.5 * np.sum((d - mu)**2)
    beta = n/2 - 1
    return 2 * alpha**(beta) / gamma(beta) * (1/x)**(beta + 1) * np.exp(-(alpha/x))

def main(args):
    x = np.linspace(1e-1, 4, 1000)
    N = np.arange(5, 100, 10)
    cmap = mpl.cm.get_cmap("jet")
    norm = mpl.colors.Normalize(vmin=N.min(), vmax=N.max())
    mappable = mpl.cm.ScalarMappable(norm, cmap)
    plt.figure(figsize=(8, 6))
    for n in N:
        plt.plot(x, std_gamma(x, n), "-", color=cmap(norm(n)))

    plt.title(r"$d_i \sim_{\text{i.i.d}} \mathcal{N}(\mu, \sigma = 1)$")
    plt.axvline(1, ls="--", color="k")
    plt.xlabel(r"$\sigma$")
    plt.ylabel(r"$\text{GIG}(\sigma \mid \alpha, \beta, 2)$")
    plt.colorbar(mappable, label=r"$|\mathcal{D}|$")
    plt.savefig("../tex/figures/gig_2.png")

    plt.figure(figsize=(8, 6))
    for n in N:
        plt.plot(x, var_gamma(x, n), "-", color=cmap(norm(n)))

    plt.title(r"$d_i \sim_{\text{i.i.d}} \mathcal{N}(\mu, \sigma^2 = 1)$")
    plt.xlabel(r"$\sigma^2$")
    plt.axvline(1, ls="--", color="k")
    plt.ylabel(r"$\text{GIG}(\sigma^2 \mid \alpha, \beta, 1)$")
    plt.colorbar(mappable, label=r"$|\mathcal{D}|$")
    plt.savefig("../tex/figures/gig_1.png")

    plt.show()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
