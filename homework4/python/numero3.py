import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use("science")
mpl.rcParams['figure.figsize']=(12.0,9.0)    
mpl.rcParams['font.size']=20               
mpl.rcParams['savefig.dpi']= 200          
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams["font.family"] = "Times New Roman"
Omega_0 = 0.3
Omega_r = 9e-5
Omega_m = 0.3
TCMB = 2.729
TGUT = 1e28
Omega_GUT = 0
a_GUT = TCMB / TGUT
N = 60

def Omega(a):
    return (Omega_m * a + Omega_r) / (Omega_m * a + Omega_r + (1 - Omega_0) * a**2)

def y(a):
    return (1 - Omega_0) * a**2 / (Omega_m * a + Omega_r + (1 - Omega_0) * a**2)

def y_i(a, N):
    return np.exp(-2 * ((a /a_GUT)**2 - 1))

def main(args):
    plt.figure()
    a = np.logspace(-33, 0, 1000)
    # _y = np.abs(1 - Omega(a))
    _y = y(a)
    plt.loglog(a, _y, "-k")
    plt.xlabel("a")
    plt.ylabel("$|1 - \Omega(t)|$")
    plt.axhline(1, color="k", ls="--", alpha=0.3)
    plt.annotate("$\Omega_0 = 0.3$", (1e-30, 0.7))
    plt.axvline(1/(1 + 1090), color="k", ls="--", alpha=0.3)
    plt.annotate("Découplage", (1/(1 + 1090), 1e-30), rotation=-90)
    plt.axvline(3.6e-9, color="k", ls="--", alpha=0.3)
    plt.annotate("Nucléosynthèse", (3.6e-9, 1e-30), rotation=-90)
    plt.axvline(1e-32, color="k", ls="--", alpha=0.3)
    plt.annotate("Temps de Planck: $a\sim 10^{-32}$", (2e-32, 1e-33), rotation=-90)
    plt.axhline(y(1e-32), color="r", ls="--", alpha=0.3)
    plt.annotate(r"$\boxed{|1 - \Omega(t_P)| = %.2e}$" % (y(1e-32)), (1e-29, y(5e-32)), color="r")
    plt.savefig("../tex/figures/omega.png")

    print(f"{TCMB/TGUT: .2e}")
    ai = np.logspace(-33, np.log10(a_GUT * np.sqrt(N + 1)), 5000)
    yi = y_i(ai, N)
    ai = ai[yi > y(a_GUT * np.sqrt(N + 1))]
    yi = yi[yi > y(a_GUT * np.sqrt(N + 1))]
    yi = np.concatenate([yi, _y[a >= a_GUT * np.sqrt(N + 1)]])
    ai = np.concatenate([ai, a[a >= a_GUT * np.sqrt(N + 1)]])
    plt.figure()

    plt.loglog(a, _y, "-k", label="Sans inflation")
    plt.loglog(ai, yi, "-b", label="Avec inflation")
    plt.xlabel("a")
    plt.ylabel("$|1 - \Omega(t)|$")
    plt.axhline(1, color="k", ls="--", alpha=0.3)
    plt.annotate("$\Omega_0 = 0.3$", (1e-15, 0.7))
    plt.axvline(1/(1 + 1090), color="k", ls="--", alpha=0.3)
    plt.annotate("Découplage", (1/(1 + 1090), 1e-30), rotation=-90)
    plt.axvline(3.6e-9, color="k", ls="--", alpha=0.3)
    plt.annotate("Nucléosynthèse", (3.6e-9, 1e-30), rotation=-90)
    plt.axvline(a_GUT, color="k", ls="--", alpha=0.3)
    plt.annotate("GUT", (a_GUT, 1e-60), rotation=-90)
    plt.fill_between(ai[ai > a_GUT], yi[ai > a_GUT], y(ai[ai > a_GUT]), color="b", alpha=0.3)
    plt.annotate("Inflation", (a_GUT*3.8, 1e-30), rotation=-88.5)
    plt.legend()
    plt.savefig("../tex/figures/omega_inflation.png")
    plt.show()

    print(f"{y_i(a_GUT, N): .2e}")

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    args = parser.parse_args()
    main(args)
