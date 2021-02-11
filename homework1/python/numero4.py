import astropy.units as u
from astropy.constants import c, G, M_sun
from numpy import pi

H0 = 100 * u.km / u.s / u.Mpc

hubble_radius = (c / H0).to(u.Mpc)

print(f"RH = {hubble_radius:.3e}")

rho_crit = 3 * H0**2 / 8 / pi / G 

Mu = rho_crit * 4 * pi * hubble_radius**3 / 3

print(f"Mu = {Mu.to(u.g):.3e}")

print(f"Mu = {(Mu/M_sun).decompose():.3e}")

rs = 2 * G * Mu / c**2

print(f"rs = {rs.to(u.lyr)/0.72 * 0.25:.3e}")
