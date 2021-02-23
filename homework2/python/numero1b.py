import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import c
import astropy.units as u
from scipy.integrate import quad

A = -1.08
da = 0.2
B = -0.26
db = 0.06

Omega_m = 0.30
Omega_L = 0.70
H0 = 70 * u.km / u.s / u.Mpc
DIST = (c/H0).decompose().to(u.Mpc).value
TIME = (1/H0).to(u.Gyr).value

def logphi_T(t):
    return A * np.log10(t) + B

def phiT(t):
    return 10**(logphi_T(t)) #/ u.Mpc**3

def H(z):
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_L)

def t(z):
    integrand = lambda z: 1 / (1 + z) / H(z)
    out = quad(integrand, z, np.inf)
    out = out[0]
    return out * TIME

def DA(z):
    integrand = lambda z: 1/H(z)
    out = quad(integrand, 0, z)
    return 1/(1 + z) * out[0] * DIST

def Ntot(a, b):
    const = (4 * np.pi * c / H0) / u.Mpc
    integrand = lambda z: (1 + z)**2 * DA(z)**2/H(z) * phiT(t(z))
    out = quad(integrand, 0, 6)
    print(out)
    return (const * out[0]).decompose()



def main(args):
    print(f"{Ntot(A, B):.3e}")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
