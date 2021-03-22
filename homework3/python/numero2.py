import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import m_p, m_e, k_B, c, G, sigma_T, hbar, Ryd as Ry
from scipy.special import zeta
from scipy.optimize import bisect
import matplotlib.pylab as pylab
import matplotlib as mpl
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

H_0 = 100  * u.km / u.s / u.Mpc
Omega_r = 2.47e-5
Omega_m = 0.31
Omega_b = 0.048
Omega_L = 0.679
h = 0.68
X_e = 1e-2
Omega_b_hsq = 0.022383 # Planck2018
eta_b = 5.5e-10 * (Omega_b_hsq/0.02)
print(f"eta_b = {eta_b:.3e}")
TCMB = ((k_B * 2.725 * u.K)/hbar/c).to(1/u.cm)
z_rec_true = 1090
Ry = (2 * np.pi * Ry * (hbar * c)).to(u.eV)

def Saha_solution(a, eta):
    factor = 2 * zeta(3) / np.pi**2 * (2 * np.pi)**(3/2)
    T = (k_B * 2.725 * u.K).to(u.eV) / a
    S = factor * eta_b * (T/m_e/c**2)**(3/2) * np.exp(Ry / T)
    S = S.decompose()
    X_e = (-1 + np.sqrt(1 + 4 * S)) / (2*S)
    return X_e 

def a_dec_saha_solution():
    factor = (3 * h * H_0 / 8 / np.pi / G / m_p * sigma_T * c * Omega_b).decompose()
    # left = lambda a: np.sqrt(Omega_r * a**2 + Omega_m * a**3)
    left = lambda a: np.sqrt(Omega_m * a**3)
    func = lambda a: left(a) - factor*Saha_solution(a, eta_b)
    a_dec, res = bisect(func, 1e-4, 1e-2, full_output=True)
    # print(res)
    print(f"X_e(a_dec) = {Saha_solution(a_dec, eta_b):.2e}")
    return 1/a_dec - 1

def main(args):
    rho_crit = (3 * H_0**2 / 8 / np.pi / G * c / hbar).to(1/u.cm**4)
    rho_crit_phys = (3 * H_0**2 / 8 / np.pi / G).to(u.g/u.cm**3)

    print(f"rho_crit = {rho_crit:.4e}")
    # energy_conversion_factor = (1/(hbar * c / u.eV).to(u.cm))**4
    # print(energy_conversion_factor)
    print(f"T_CMB = {TCMB}")
    R_factor = 5 * rho_crit / 2 / TCMB**4 #/ zeta(2)
    print(f"R_factor = {R_factor:.2f}")
    factor1 = 3 * H_0**2 / 8 / np.pi / G / m_p * sigma_T * c
    print(f"Gamma factor = {factor1.to(1/u.s):.4e}")

    R_dec_true = R_factor / (1 + z_rec_true) / zeta(2) * Omega_b_hsq
    print(f"R_dec_true = {R_dec_true}")
    
    # print(Omega_r * (1 + 1090)**4)
    # print(Omega_m * (1 + 1090)**3)
    # print(Omega_r / Omega_m * (1091))

    factor2 = factor1 / H_0
    print(f"Gamma_T/H factor = {factor2.decompose():.4e}")
    a_rec_factor = factor2.decompose()**(2/3)
    print(f"a_rec_factor = {a_rec_factor:.4e}")
    z_rec = 1/(a_rec_factor*(Omega_b_hsq)**(1/3))/X_e**(2/3)
    print(f"z_rec = {z_rec}")
    a_rec = (a_rec_factor * X_e**(2/3) * (Omega_b_hsq)**(1/3)).decompose().value
    print(f"a_rec = {a_rec}")

    R_factor2_rec = R_factor * a_rec_factor * X_e**(2/3) / zeta(2) * Omega_b_hsq**(4/3)
    print(f"R factor final {R_factor2_rec:.2f}")

    z_dec_saha = a_dec_saha_solution()
    print(f"z_dec_saha = {z_dec_saha}")

    T_dec = (k_B * 2.725 * u.K).to(u.eV) *(1 + z_dec_saha)
    print(f"T_dec = {T_dec}")


    R_factor_final2 = R_factor /(1 + z_dec_saha) * Omega_b_hsq
    print(f"R_factor_final_saha = {R_factor_final2}")
    

    if args.plots:
        # R = lambda x: R_factor2_rec / zeta(2) * x**(4/3)
        R = lambda a: R_factor_final2 * x / Omega_b_hsq
        c_s = lambda x: np.sqrt(1/3/(1 + R(x)))
        plt.figure()
        plt.style.use("science")
        x = np.logspace(-3, -1, 500)
        plt.plot(x, c_s(x), "k-")
        plt.xscale("log")
        plt.ylabel(r"$c_s$")
        plt.xlabel(r"$\Omega_b h^2$")
        plt.savefig("../tex/figures/speed_of_sound.png")

        a = np.logspace(np.log10(a_rec/100), np.log10(a_rec*1.2), 400)[np.newaxis, :]
        x = x[:, np.newaxis]
        rho_b = lambda a, omega_b_h: rho_crit_phys * omega_b_h * a**(-3)
        R = lambda a, omega_b_h: R_factor / zeta(2) * omega_b_h * a
        c_s = lambda a, omega_b_h: c**2/(3 * (1 + R(a, omega_b_h)))
        k_J = np.sqrt(4 * np.pi * G * rho_b(a, x) * a**2 / c_s(a, x)).to(1/u.Mpc)
        horizon = (c * a / (2 * H_0 * Omega_r**(1/2))).to(u.Mpc)

        plt.figure(figsize=(8, 6))
        plt.style.use("science")
        cmap = mpl.cm.get_cmap("turbo")
        norm = mpl.colors.Normalize(vmin=x.min(), vmax=x.max())
        mappable = mpl.cm.ScalarMappable(norm, cmap)
        for i, _x in enumerate(x):
            plt.plot(a[0], k_J[i, :], color=cmap(norm(_x))[0])

        plt.xscale("log")
        plt.colorbar(mappable, label=r"$\Omega_b h^2$")
        plt.xlabel("a")
        plt.ylabel(r"$k_J$ [$h$ Mpc$^{-1}$]")


        plt.figure(figsize=(8, 6))
        for i, _x in enumerate(x):
            plt.plot(a[0], horizon[0]* k_J[i, :], color=cmap(norm(_x))[0])

        plt.xscale("log")
        plt.colorbar(mappable, label=r"$\Omega_b h^2$")
        plt.xlabel("a")
        plt.ylabel(r"$\eta k_J$")
        plt.yscale("log")
        plt.axhline(1, ls="--", color="k")
        plt.axvline(1/(1 + z_rec_true), ls="--", color="k")
        plt.annotate(r"$a_{\text{dec}}$", (0.75/(1 + z_rec_true), 0.05), rotation=90, fontsize=15)
        plt.xlim(a.min(), a.max())
        plt.savefig("../tex/figures/eta_kj.png")

        plt.show()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--plots", action="store_false")
    args = parser.parse_args()
    main(args)
