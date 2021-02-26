import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import m_p, G
from astropy import units as u
from astropy.cosmology import Planck15

# tau = .066
# dtau = 0.012
Tcmb = 2.725
h_50 = 0.6774/2
# H = h_50 * 50 * u.km / u.s / u.Mpc
Sigma_b = 0.0486
del_Sigma_b = 0.0010
T21 = -0.5 # dip
X = 0.7 # hydrogen ratio

def main(args):
    tau = lambda z: 0.00485 * ((0.31 * (1 + z)**3 + 0.69)**(1/2) - 1)
    print(tau(7.8))
    T_gam = lambda z: Tcmb * (1 + z)
    T_gaz = lambda z: Tcmb/201 * (1 + z)**2 
    print(f"T_gaz(z=17) = {T_gaz(17)}")
    print(f"T_gaz(z=20) = {T_gaz(20)}")
    print(f"T_gam(z=20)  = {T_gam(20)}")
    # T_21 = lambda z, tau: (1 + z)**(-1) * (T_gaz(z) - T_gam(z)) * (1 - np.exp(-tau(z)))
    T_21 = lambda z: 0.011 / h_50 * (Sigma_b * h_50**2 / 0.05) * np.sqrt(1 + z)/3 \
            * (T_gaz(z) - T_gam(z))/T_gaz(z)
    print(f"T_21(z = 17): {T_21(17):.3f}")
    print(f"T_21(z = 20): {T_21(20):.3f}")

    # T_H = lambda z, tau, T21: T21 * (1 + z) / (1 - np.exp(-tau(z))) + T_gam(z)
    T_H = lambda z, t21: -T_gam(z) / (t21/0.011 * h_50 * 0.05 / Sigma_b / h_50**2 
            * 3 / np.sqrt(1 + z) - 1)
    print(f"T_H(z=20, t21=-500mK) = {T_H(20, T21):.3f}")
    cosmo = Planck15
    rho_crit = 3 * cosmo.H(20)**2 / 8 / np.pi / G
    n_h = (X * cosmo.Ob(20) * rho_crit / m_p).to(1/u.cm**3)
    print(f"{n_h:.3e}")
    n_chi = n_h * (T_gaz(20) / T_H(20, T21) - 1)
    print(T_gaz(20) / T_H(20, T21))
    print(n_chi)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
