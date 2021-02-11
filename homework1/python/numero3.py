from astropy.constants import c, k_B, hbar, G, M_sun, m_p
from numpy import pi
import astropy.units as u
from wtt import * # github/AlexandreAdam/write_to_tex
from pprint import pprint

results = {}
H0 = 100 * u.km / u.s / u.Mpc

results.update({"lightspeed": f"{c.to(u.cm / u.s).value:.0f}"})

results.update({"hbarc": format_tex((hbar * c / u.eV).to(u.cm).value)})

results.update({"hbarc/c": format_tex((hbar / c / u.g).to(u.cm).value)})

ai = (hbar * H0).to(u.eV)
print(f"a i) {ai:.3e}")
results.update({"ai": format_tex(ai.value)})

aii = (H0 / c).to(1/u.Mpc)
print(f"a ii) {aii:.3e}")
results.update({"aii": format_tex(aii.value)})

aiii = (H0).to(1/u.Gyr)
print(f"a iii) {aiii:.3e}")
results.update({"aiii": format_tex(aiii.value)})

aiv = (H0 * c).to(u.m / u.s**2)
print(f"a iv) {aiv:.3e}")
results.update({"aiv": format_tex(aiv.value)})

M = 10**15 * M_sun 

# b = ((1 / H0**2 * G * M)**(1/3)).to(u.Mpc)
b = ((G * M / H0 / c)**(1/2)).to(u.Mpc)
print(f"b {b:.3e}")
results.update({"b": format_tex(b.value)})


rho_crit = 3 * H0**2 / 8 / pi / G 

ci = rho_crit.to(u.g / u.cm**3)
print(f"c i) {ci:.3e}")
results.update({"ci": format_tex(ci.value)})

cii = (c / hbar * hbar**4 * c**4 * rho_crit ).to(u.GeV**4)
print(f"c ii) {cii:.3e}")
results.update({"cii": format_tex(cii.value)})


ciii = (c**2 * rho_crit).to(u.eV / u.cm**3)
print(f"c iii) {ciii:3e}")


civ = (rho_crit / m_p).to(1/u.cm**3)
print(f"c iv) {civ:.3e}")

cv = (rho_crit / M_sun).to(1 / u.Mpc**3)
print(f"c v) {cv:.3e}")

pprint(results)

# write_to_tex("../tex/hwk1_alexandre_adam_H21.tex", results)
