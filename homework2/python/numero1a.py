import numpy as np
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib as mpl
import astropy.cosmology as cosmo
import astropy.units as u
from astropy.constants import G, c, M_sun, R_sun, hbar
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          # 'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.style.use("science")


deg2as = 3600

def main(args):
    print("Les distances en cosmologie")
    h = 1
    H_0 = h * 100 * u.km / u.s / u.Mpc
    a_0 = 1
    Omega_0 = np.linspace(0.5, 1.5, 1000)[np.newaxis, :]
    ell = 10 * u.kpc
    z = np.linspace(0.9, 2, 1000)[:, np.newaxis]
    def xi(z, Omega_0):
        amp = 2 * c / H_0 / a_0
        numerator = Omega_0 * z + (Omega_0 - 2) * (np.sqrt(Omega_0 * z + 1) - 1)
        denominator = Omega_0**2 * (1 + z)
        return amp * numerator / denominator

    _theta = (ell * (1 + z) / xi(z, Omega_0)).decompose()
    theta = np.rad2deg(_theta.value) * deg2as
    norm = mpl.colors.Normalize(vmin=Omega_0.min(), vmax=Omega_0.max())
    cmap = mpl.cm.get_cmap("jet")
    cmappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig, ax = plt.subplots()
    ax_divider = make_axes_locatable(ax)
    cax = ax_divider.append_axes("right", size="7%", pad="2%")
    for i in range(Omega_0.size):
        ax.plot(z[:, 0], theta[:, i], color=cmappable.to_rgba(Omega_0[0, i]))
    fig.colorbar(cmappable, cax=cax)
    cax.set_ylabel(r"$\Omega_0$")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\theta$ [$h$ arcsec]")
    ax.set_title(r"$\ell = 10$ kpc")
    # ax.set_xscale("log")
    plt.savefig("../tex/figures/theta_z.png")


    plt.figure()
    z_min = []
    for i in range(Omega_0.size):
        z_min.append(z[np.argmin(theta[:, i]), 0])

    plt.plot(Omega_0[0, :], z_min, "k-")
    plt.axvline(1, color="k", ls="--", lw=1)
    plt.axhline(1.25, color="k", ls="--", lw=1)
    plt.xlabel(r"$\Omega_0$")
    plt.ylabel(r"$z$")
    plt.title(r"argmin $\theta(z)$")
    plt.savefig("../tex/figures/zmin.png")



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
