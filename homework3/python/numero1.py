import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import sigma_sb, k_B, c, h, hbar, m_e
from scipy.special import zeta
import astropy.units as u
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

def n_e_relativistic(T):
    T = T.to(u.eV, equivalencies=u.temperature_energy()) # T can be in Kelvin or eV
    return (3 * zeta(3) / np.pi**2 * (T / hbar / c)**3).to(1/u.cm**3)

def n_e(T): # non-relativistic approximation
    T = T.to(u.eV, equivalencies=u.temperature_energy())
    return (2 / np.pi**2 *  (m_e * T / hbar**2)**(3/2) * np.exp(-m_e * c**2 / T)).to(1/u.cm**3)

def n_gamma(T):
    T = T.to(u.eV, equivalencies=u.temperature_energy())
    return (2 * zeta(3) / np.pi**2 * (T / hbar / c)**3).to(1/u.cm**3)

@np.vectorize
def n_e_general(T):
    me = (m_e * c**2).to(u.MeV).value
    def integrand(x): # trick for numerical stability
        if np.sqrt(x**2 + me**2) > 5 * T:
            return x**2 * np.exp(-np.sqrt(x**2 + me**2) / T)
        return x**2 / (np.exp(np.sqrt(x**2 + me**2) / T) + 1)
    factor =  2 / np.pi**2 * ((u.MeV)**3 / (hbar)**3 / c**3).to(1/u.cm**3).value 
    integral = quad(integrand, 0, np.inf)[0]
    return factor * integral

#TODO clean up prefactor and should be good

def main(args):
    # print((1 * u.MeV).to(u.K, equivalencies=u.temperature_energy()))
    # print(n_e(4000 * u.K))
    # print(n_e_relativistic(4000 * u.K) / 1501**3 / 1.6)


    T = np.logspace(-2, 0, 400)
    fig = plt.figure(figsize=(8, 6))
    plt.style.use("science")

    frame1 = fig.add_axes((.1, .3, .8, .6))
    ne = n_e_general(T)
    ne1 = n_e(T * u.MeV)
    ne2 = n_e_relativistic(T * u.MeV)
    plt.plot(T, ne, "k-", label="Vrai")
    plt.plot(T, ne1, "k--", label="Non-relativistique")
    plt.plot(T, ne2, "k-.", label="Relativistique")
    plt.legend()
    plt.gca().invert_xaxis()
    plt.ylabel(r"$n_{e^{\pm}}$ [cm$^{-3}$]")
    plt.yscale("log")
    plt.xscale("log")

    fram2 = fig.add_axes((.1, .1, .8, .2))
    plt.plot(T, (ne1.value - ne)/ne * 100, "k--")
    plt.plot(T, (ne2.value - ne)/ne * 100, "k-.")
    plt.xlabel(r"$k_B$T [MeV]")
    plt.ylabel(r"Erreur relative ($\%$)")
    plt.ylim(-100, 100)
    plt.gca().invert_xaxis()
    plt.savefig("../tex/figures/nucleosynthesis_electron_density.png")


    # main result
    n_gam = n_gamma(T * u.MeV)
    f = interp1d(ne/n_gam, T)
    T_star = f(0.01)
    plt.figure(figsize=(8, 6))
    plt.plot(T, ne/n_gam, "k")
    plt.ylabel(r"$n_{e^{\pm}} / n_\gamma $")
    plt.xlabel(r"$k_B$T [MeV]")
    plt.gca().invert_xaxis()
    plt.axhline(0.01, color="k", ls="--" )
    plt.axvline(T_star, color="k", ls="--")
    plt.annotate(fr"$k_B T^\star={T_star:.2f}$ MeV", (T_star+0.05, 0.5), rotation=90, fontsize=15)
    plt.annotate(r"$n_{e^{\pm}} / n_\gamma = 0.01$", (1, 0.04), fontsize=15)
    plt.savefig("../tex/figures/resultat_ratio_ne_ngamma.png")
    plt.show()


    # T = np.logspace(-6, -2, 400)
    T_des = f(6e-10)
    print(T_des)




if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
