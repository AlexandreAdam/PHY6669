import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import m_p, m_e, k_B, c, G, sigma_T, hbar
from scipy.special import zeta

H_0 = 100  * u.km / u.s / u.Mpc
Omega_r = 2.47e-5/(0.7)**2
Omega_m = 0.31
Omega_b = 0.048
h = 0.68
X_e = 1e-2
Omega_b_hsq = 0.022
TCMB = ((k_B * 2.725 * u.K)/hbar/c).to(1/u.cm)

def main(args):
    rho_crit = (3 * H_0**2 / 8 / np.pi / G * c / hbar).to(1/u.cm**4)
    print(f"rho_crit = {rho_crit:.4e}")
    # energy_conversion_factor = (1/(hbar * c / u.eV).to(u.cm))**4
    # print(energy_conversion_factor)
    print(f"T_CMB = {TCMB}")
    R_factor = 5 * rho_crit / 2 / TCMB**4
    print(f"R_factor = {R_factor:.2f}")
    factor1 = 3 * H_0**2 / 8 / np.pi / G / m_p * sigma_T * c
    print(f"Gamma factor = {factor1.to(1/u.s):.4e}")
    
    # print(Omega_r * (1 + 1090)**4)
    # print(Omega_m * (1 + 1090)**3)
    # print(Omega_r / Omega_m * (1091))

    factor2 = factor1 / H_0
    print(f"Gamma_T/H factor = {factor2.decompose():.4e}")
    a_rec_factor = factor2.decompose()**(2/3)
    print(f"a_rec_factor = {a_rec_factor:.4e}")
    z_rec = 1/(a_rec_factor*(Omega_b_hsq)**(1/3))/X_e**(2/3)
    print(f"z_rec = {z_rec}")
    a_rec = a_rec_factor * X_e**(2/3) * (Omega_b_hsq)**(1/3)
    print(a_rec)

    R_factor2_rec = R_factor * a_rec_factor
    print(R_factor2_rec)


    return

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
