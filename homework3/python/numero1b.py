import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.constants import hbar, c, k_B, m_n, m_p, m_e
from scipy.integrate import quad


Q = ((m_n - m_p)*c**2).to(u.MeV)
G_F = (1.16639e-5 * (1/u.GeV**2) * (hbar * c)**3).to(u.MeV * u.cm**3)
G_F_planck = (1.16639e-5 * (1/u.GeV**2))
m_e_planck = (m_e * c**2).to(u.MeV)
tau_n = 879.5 * u.s

_Q = Q.value
_G_F = G_F.value
_m_e = (m_e * c**2).to(u.MeV).value # 0.5 MeV
def _lambda_0(x): # equation3.62 Dodelson
    return x * (x - _Q/_m_e)**2 * np.sqrt(x**2 - 1)

lambda_0 = quad(_lambda_0, 1, _Q/_m_e)[0]


def main(args):
    print(Q)
    print(lambda_0)
    print(m_e_planck)

    factor = ((c * tau_n)**(-1) * (G_F_planck**2 * m_e_planck**5 * lambda_0 / 2 * np.pi**3 / (hbar * c) )**(-1)).decompose()
    print(factor)

    factor2 = (4 *   Q**5  /  m_e_planck**5 /  lambda_0 * m_p / m_n).decompose()
    print(factor2)
    return

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
