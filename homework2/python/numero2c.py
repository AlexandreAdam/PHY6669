import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (10, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)
plt.style.use("science")


def main(args):
    omega_0 = np.logspace(-2, 0, 1000)
    eta_0 = np.arccosh((2 - omega_0)/omega_0)
    a = 0.5 * omega_0 / (1 - omega_0) * (np.cosh(0.5 * eta_0) - 1)
    z = 1/a - 1
    plt.style.use("science")
    plt.plot(omega_0, z, "k")
    # plt.xscale("log")
    plt.ylabel(r"$z$")
    plt.xlabel(r"$\Omega_0$")
    plt.axhline(3, ls="--", color="k")
    plt.annotate(r"$z=3$", (0.1, 3.1), fontsize=15)
    plt.title(r"$(1 + z)^{-1} = a(0.5 \eta_0)$")
    plt.savefig("../tex/figures/numero2c_sol.png")
    plt.show()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
