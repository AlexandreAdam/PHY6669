import astropy.units as u
from astropy.constants import k_B, c, hbar, sigma_sb, L_sun
from numpy import pi
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from numba import jit, njit
from tqdm import tqdm


plt.style.use("science")
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (6, 6),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'
}
pylab.rcParams.update(params)


h = 0.72 # hubble parameter
H_0 = 100 * u.km / u.s / u.Mpc * h
T_0 = 6000 * u.K
lambda_0 = 5100 * u.angstrom
epsilon_0 = 2.5e8 * h * L_sun / u.Mpc**3

alpha = pi * hbar * c**3 / sigma_sb
beta = epsilon_0 / (T_0**4 * lambda_0**5 * H_0)
gamma = (2 * pi * hbar * c / k_B / T_0 / lambda_0).decompose().value # dimensionless

# print((alpha * beta).decompose().to(u.erg / u.s / u.cm**3))
# print(gamma)



M = 300
lam = np.logspace(3, 7, M) * u.angstrom

def safe_log(x):
    return np.log10(x + 1e-10)

# @jit(nopython=True)
def intensity(z_f, sigma_0, q_0, gamma):
    def integrand(z):
        num = (1 + z)**2
        term1 = 2 * sigma_0 * (1 + z)**3
        term2 = (1 + q_0 - 3 * sigma_0) * (1 + z)**2
        factor1 = term1 + term2 + sigma_0 - q_0
        if gamma * (1 + z) > 100:
            factor2 = np.exp(-gamma * (1 + z)) 
        else:
            factor2 = 1/(np.exp(gamma * (1 + z)) - 1)
        return num* factor2 / factor1**(1/2)
    out = quad(integrand, 0, z_f)
    return out[0]

# @jit(nopython=True)
def intensity_LCDM(z_f, omega_m, omega_L, omega_r, gamma):
    def integrand(z):
        num = (1 + z)**2
        term1 = omega_m * (1 + z)
        term2 = omega_r * (1 + z)**2
        term3 = omega_L * (1 + z)**(-2)
        term4 = 1- omega_m - omega_L - omega_r
        factor1 = np.exp(gamma * (1 + z)) - 1
        factor2 = term1 + term2 + term3 + term4
        return num / factor1 / factor2**(1/2)
    out = quad(integrand, 0, z_f)
    return out[0]

# in construction, abandonned
def bolometric_LCDM(z_f, omega_m, omega_L, omega_r):
    gamma = lambda z, lam: (2 * pi * hbar * c / k_B / T(z) / lam * u.angstrom).decompose().value
    beta = lambda lam: (epsilon_0 / (T_0**4 * (lam * u.angstrom**5 * H_0))).value

    def integrand(z):
        num = (1 + z)**2
        term1 = omega_m * (1 + z)
        term2 = omega_r * (1 + z)**2
        term3 = omega_L * (1 + z)**(-2)
        term4 = 1- omega_m - omega_L - omega_r
        def integrand2(lam): 
            factor1 = np.exp(gamma(z, lam) * (1 + z)) - 1
            factor2 = term1 + term2 + term3 + term4
            return beta(lam) * num / factor1 / factor2**(1/2)
        return quad(integrand2, 0, np.inf)
    return quad(integrand, 0, zf)



# adaptive step method
# @jit(nopython=True, parallel=True)
def z_median(lambda_0, sigma_0, q_0, T=lambda z: T_0):
    beta = epsilon_0 / (T_0**4 * lambda_0**5 * H_0)
    gamma = lambda z: (2 * pi * hbar * c / k_B / T(z) / lambda_0).decompose().value 

    I = []
    z = [1e-3]
    step = 0.01 # in logspace
    epsilon = 1e-5 # convergence criterion

    delta = np.inf
    i = 0
    # adaptive step method
    while delta > epsilon and i < 1e5:
        I.append(intensity(z[-1], sigma_0, q_0, gamma(z[-1])))
        z.append(10**(np.log10(z[i]) + step))
        if i != 0:
            delta = np.abs((I[-1] - I[-2])/(I[-2] + 1e-16))

        if i > 300:
            step = 0.05
        if i > 1000:
            step = 0.1
        i += 1
    if i >= 1e5:
        print("Did not converge")

    Imax = I[-1]
    I = np.array(I)
    z = np.array(z[:-1])
    x_med = UnivariateSpline(z, (I / Imax) - 0.5, s=0).roots()
    return x_med, (I * alpha * beta).to(u.erg / u.angstrom / u.s /u.cm**2), z

# @jit(nopython=True, parallel=True)
def z_median_LCDM(lambda_0, omega_m, omega_L, omega_r, T=lambda z: T_0, h=0.6766):
    H_0 = 100 * u.km / u.s / u.Mpc * h
    beta = epsilon_0 / (T_0**4 * lambda_0**5 * H_0)
    gamma = lambda z: (2 * pi * hbar * c / k_B / T(z) / lambda_0).decompose().value 

    I = []
    z = [1e-3]
    step = 0.01 # in logspace
    epsilon = 1e-5 # convergence criterion

    delta = np.inf
    i = 0
    while delta > epsilon and i < 1e5:
        I.append(intensity_LCDM(z[-1], omega_m, omega_L, omega_r, gamma(z[-1])))
        z.append(10**(np.log10(z[i]) + step))
        if i != 0:
            delta = np.abs((I[-1] - I[-2])/(I[-2] + 1e-16))
        i += 1
    if i >= 1e5:
        print("Did not converge")

    Imax = I[-1]
    I = np.array(I)
    z = np.array(z[:-1])
    x_med = UnivariateSpline(z, (I / Imax) - 0.5, s=0).roots()
    return x_med, (I * alpha * beta).to(u.erg / u.angstrom / u.s /u.cm**2), z

# @jit(nopython=True, parallel=True)
def z_median_all(n, M=M):
    T = lambda z: T_0 * (1 + z)**(n/4)
    omega_m = 0.3089
    omega_L = 0.679
    omega_r = 0
    z_milne = np.zeros(M) 
    z_einstein = np.zeros(M)
    z_LCDM = np.zeros(M)
    for i, l in enumerate(tqdm(lam)):
        z_milne[i], _, _ = z_median(l, 0, 0, T) 
        z_einstein[i], _, _ = z_median(l, 0.5, 0.5, T)
        z_LCDM[i], _, _ = z_median_LCDM(l, omega_m, omega_L, omega_r, T)

    return z_milne, z_einstein, z_LCDM

def z_median_bolometrique(n, M=M):
    T = lambda z: T_0 * (1 + z)**(n/4)
    omega_m = 0.3089
    omega_L = 0.679
    omega_r = 0
    # z_milne = np.zeros(M) 
    # z_einstein = np.zeros(M)
    z_LCDM = np.zeros(M)



# Enstein-de Sitter universe
sigma_0 = 0.5
q_0 = 0.5
z_einstein, I_einstein, z_f_e = z_median(lambda_0, sigma_0, q_0)


# Milne universe
sigma_0 = 0
q_0 = 0
z_milne, I_milne, z_f_m = z_median(lambda_0, sigma_0, q_0)


# LambdaCDM
omega_m = 0.3089
omega_L = 0.679
omega_r = 0
z_LCDM, I_LCDM, z_f_L = z_median_LCDM(lambda_0, omega_m, omega_L, omega_r)

plt.plot(z_f_e, I_einstein, "r-", label="Einstein-de Sitter")
plt.plot(z_f_m, I_milne, "-k", label="Milne")
plt.plot(z_f_L, I_LCDM, "-g", label=r"$\Lambda$CDM")
plt.ylabel(r"$\partial I_{\lambda_0} / \partial \Omega$ [erg $\AA^{-1}$ cm$^{-2}$ s$^{-1}$ sr$^{-1}$]")
plt.xlabel(r"$z$")
plt.title(r"$\lambda_0 = 5100 \AA$")
plt.axvline(z_einstein, color="r")
plt.axvline(z_milne, color="k")
plt.axvline(z_LCDM, color="g")
plt.legend()
plt.savefig("../tex/figures/flux_univers.png")

plt.show()


z_milne, z_einstein, z_LCDM = z_median_all(0)

plt.title(r"$T_0$ = 6000 K")
plt.plot(lam, z_milne, "-k", label="Milne")
plt.plot(lam, z_einstein, "-r", label="Einstein-de Sitter")
plt.plot(lam, z_LCDM, "-g", label=r"$\Lambda$CDM")
plt.ylabel(r"$z_{\text{median}}$")
plt.xlabel(r"$\lambda_0$ [$\AA$]")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.savefig("../tex/figures/z_median_univers.png")

