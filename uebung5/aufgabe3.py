from math import gcd

# Berechnung der Ordnung eines Elementes g^alpha mit Hilfe von Satz 2.13
# Ist g ein Element der Ordnung omega und alpha ein Element der natürlichen Zahlen,
# dann ist die Ordnung von g^alpha gleich omega / gcd(omega, alpha).

# Elliptische Kurve: y^2 = x^3 + ax + b
a = 3
b = 6
p = 11  # Z_11

def add_points(p1, p2, a, p):
    if p1 == p2:  # Punktverdopplung
        if p1[1] == 0:  # Punkt im Unendlichen
            return None
        # Steigung (Lambda) berechnen
        lam = (3 * p1[0] * p1[0] + a) * pow(2 * p1[1], p - 2, p)
    else:  # Addition unterschiedlicher Punkte
        if p1[0] == p2[0]:  # vertikale Linie
            return None
        # Steigung (Lambda) berechnen
        lam = (p2[1] - p1[1]) * pow(p2[0] - p1[0], p - 2, p)

    # Neuen Punkt berechnen
    x3 = (lam * lam - p1[0] - p2[0]) % p
    y3 = (lam * (p1[0] - x3) - p1[1]) % p
    return (x3, y3)



# Erzeugendes Element (Punkt mir Ordnung 15 aka Ordnung der Kurve)
omega = 15
generating_point = (4, 4)

# all: omega / gcd(omega, alpha) = 5
alpha_candidates = [alpha for alpha in range(1, omega) if omega // gcd(omega, alpha) == 5]

elements_of_order_5 = {}

for alpha in alpha_candidates:
    order = omega // gcd(omega, alpha)
    current_point = generating_point
    for _ in range(1, alpha):
        current_point = add_points(current_point, generating_point, a, p)
    elements_of_order_5[alpha] = (current_point, order)

print(f"Erzeugendes Element (mit Ordnung {omega}): \t{generating_point}")
for alpha, (point, order) in elements_of_order_5.items():
        print(f"Alpha: {alpha}, Punkt: {point}, Ordnung: {order}")

