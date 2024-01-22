import si

# ECDH Schlüsselaustausch
# Gegeben:
# Kurve E: y^2 = x^3 + 13x + 13
# Primzahl p: 23
# Punkt G = (1, 2)
# Ordnung w = 29
# Alice:
#   alpha = 8
#   beta = 18

p = 23
curve = si.EC(13, 13, p)
G = si.Point(curve, (1, 2))
w = 29
alpha = 8
beta = 18

print(f"Input:\np = {p}\nKurve = {curve.__str__()}\nG = {G.__str__()}\nOrdnung = {w}\nalpha = {alpha}\nbeta = {beta}")

print("\n-------------------------\n")

A = G.mult(alpha)
print(f"A = alpha * G = {alpha} * {G} = {A}")

B = G.mult(beta)
print(f"B = beta * G = {beta} * {G} = {B}")

print("\n-------------------------\n")

KA = B.mult(alpha)
print(f"Key-A = alpha * B = {alpha} * {B} = {KA}")

KB = A.mult(beta)
print(f"Key-B = beta * a = {beta} * {A} = {KB}")
