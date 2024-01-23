import si

# Gegeben (Klausurbeispiel):
# Problem: Schlüssel (k) wiederverwendet
# Gesucht priv_key

p = 29
w = 37
curve = si.EC(4, 20, p)
G = si.Point(curve, (1, 5))

h_m1 = 25
h_m2 = 17

# Signaturen:
r_1, s_1 = 24, 36
r_2, s_2 = 24, 19

print(f"Input:\np = {p}\nKurve = {curve.__str__()}\nG = {G.__str__()}\nOrdnung = {w}")

print(f"Signaturen:\n(r_1, s_1): {r_1, s_1}\n(r_2, s_2): {r_2, s_2}")

# 2.6.2 Attacken auf DSA-Signaturen:
k = pow(pow((s_1 - s_2), -1, w) * (h_m1 - h_m2), 1, w)

# r_1 = r_2
alpha = pow(pow(r_1, -1, w) * (k * s_1 - h_m1), 1, w)

print("\nAngriff:")
print(f"alpha: {alpha}")


print("\n Test:")
h = h_m1
r = r_1

print(f"Signatur mit:\n\tk = {k}\n\th(m1)={h}")
r = pow(G.mult(k).x, 1, w)
print(f"r = (k * G)_x-coord mod w = ({k} * {G.__str__()})_x-coord mod {w}")
print(f"r = {r}")

k_1 = pow(k, -1)
s = pow(pow(k,-1,w) * (h + alpha * r), 1, w)
print(f"s = k^-1 * (H(m) + alpha * r) mod w = {pow(k,-1,w)} * ({h} + {alpha} * {r}) mod {w}")
print(f"s = {s}")

h = h_m2
r = r_2

print(f"Signatur mit:\n\tk = {k}\n\th(m)={h}")
r = pow(G.mult(k).x, 1, w)
print(f"r = (k * G)_x-coord mod w = ({k} * {G.__str__()})_x-coord mod {w}")
print(f"r = {r}")

k_1 = pow(k, -1)
s = pow(pow(k,-1,w) * (h + alpha * r), 1, w)
print(f"s = k^-1 * (H(m) + alpha * r) mod w = {pow(k,-1,w)} * ({h} + {alpha} * {r}) mod {w}")
print(f"s = {s}")


