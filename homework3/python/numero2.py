import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.constants import m_p, m_e, k_B, c, G, sigma_T

H_0 = 100  * u.km / u.s / u.Mpc
Omega_r = 2.47e-5/(0.7)**2
Omega_m = 0.3

def main(args):
    factor1 = 3 * H_0**2 / 8 / np.pi / G / m_p * sigma_T * c
    print(factor1.to(1/u.s))
    
    print(Omega_r * (1 + 1090)**4)
    print(Omega_m * (1 + 1090)**3)
    print(Omega_r / Omega_m * (1091))

    factor2 = factor1 * 1e-2 / H_0
    print(f"{factor2.decompose():.4e}")
    return

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
