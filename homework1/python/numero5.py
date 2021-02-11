from numpy import pi
from astropy.constants import G, c, M_sun, R_sun, R_earth, M_earth
import astropy.units as u

l1 = 69 * u.km
l2 = 85 * u.km
l3 = 107 * u.km

s = 0.5 * (l1 + l2 + l3)
# Heron formula
A_triangle = ((s * (s - l1) * (s - l2) * (s - l3))**(1/2)).decompose()

print(A_triangle.to(u.km**2))

geometrical_term = (A_triangle / R_sun**2).decompose()
geometrical_term2 = (A_triangle / R_earth**2).decompose()
print(f"Geometrical factor Sun: {geometrical_term:.3e}")
print(f"Geometrical factor Earth: {geometrical_term2:.3e}")

gr_term = (G * M_sun / c**2 / R_sun).decompose()
gr_term_earth = (G * M_earth / c**2 / R_earth).decompose()
print(f"GR factor Sun: {gr_term:.3e}")
print(f"GR factor Earth: {gr_term_earth:.3e}")

deviation = geometrical_term * gr_term
deviation2 = geometrical_term2 * gr_term_earth

print(f"Deviation from flat sum Sun: {deviation:.3e} radians")
print(f"Deviation from flat sum Earth: {deviation2} radians")
