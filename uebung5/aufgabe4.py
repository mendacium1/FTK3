import si
from sympy import symbols, Eq, solve
from sympy.abc import x, y
import math

a, b = 28, 42
p = 89
ec = si.EC(a, b, p)
order = 103
G = si.Point(ec, (2, 27))
P = si.Point(ec, (47, 28))

# Baby Steps
m = math.ceil(math.sqrt(order + 1))
baby_steps = [ G.mult(i) for i in range(m) ]

# Giant Steps
inv_mG = G.mult(-m)
current = P
for j in range(m):
    if current in baby_steps:
        # Gefunden: x = jm + i
        i = baby_steps.index(current)
        x = j * m + i
        break
    current = current.add(inv_mG) #elliptic_add(current, inv_mG, a, p)

print(x)


