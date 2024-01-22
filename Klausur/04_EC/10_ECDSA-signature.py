import si

# ECDSA-Signatur
# Gegeben:
# Kurve E: y^2 = x^3 + 5x + 200
# Primzahl p: 601
# Punkt G = (3, 38)
# Ordnung w = 577
# Alice:
#   alpha = 281

p = 601
curve = si.EC(5, 200, p)
G = si.Point(curve, (3, 38))
w = 577
alpha = 281

print(f"Input:\np = {p}\nKurve = {curve.__str__()}\nG = {G.__str__()}\nOrdnung = {w}\nalpha = {alpha}")

print("\n-------------------------\n")

A = G.mult(alpha)
print(f"A = alpha * G = {alpha} * {G} = {A}")

print("\n-------------------------\n")

k = 3
h = 333
print(f"Signatur mit:\n\tk = {k}\n\th(m)={h}")
r = pow(G.mult(k).x, 1, w)
print(f"r = (k * G)_x-coord mod w = ({k} * {G.__str__()})_x-coord mod {w}")
print(f"r = {r}")

k_1 = pow(k, -1)
s = pow(pow(k,-1,w) * (h + alpha * r), 1, w)
print(f"s = k^-1 * (H(m) + alpha * r) mod w = {pow(k,-1,w)} * ({h} + {alpha} * {r}) mod {w}")
print(f"s = {s}")

print(f"Signatur (r,s): ({r}, {s})")

print("\n-------------------------\n")

print("Prüfen der Signatur:")
print(f"Check: 1<=r<w && 1<=s<w")
print(f"\t1<={r}<{w} && 1<={s}<{w}\n")
if (1 <= r < w) and (1 <= s < w):
    print("Calc x & y")
    x = pow(pow(s,-1,w) * h, 1, w)
    print(f"x = s^-1 * H(m) mod w = {pow(s,-1,w)} * {h} mod {w} = {x}")
    y = pow(pow(s,-1,w) * r, 1, w)
    print(f"y = s^-1 * r mod w = {pow(s,-1,w)} * {r} mod {w} = {y}")
    print()
    print(f"Check: r = (x * G + y * A)_x-coord mod w")
    print(f"\t{r} = ({x} * {G.__str__()} + {y} * {A.__str__()})_x-coord mod {w}\n")
    if r == pow((G.mult(x) + A.mult(y)).x, 1, w):
        print(f"Signature gute!")
