import math

# Fermat Methode -> p & q zu Nahe beieinander
# Gegeben pub_key (e, n) z.B.: (3718548079, 65537)
# Gesucht priv_key

n = 3718548079
e = 65537

print(f"Input:\nn = {n}\ne = {e}")

print("\n-------------------------\n")

def fermat_factor(n):
    assert n % 2 != 0  

    print("Immer größere a probieren, damit b möglichst klein, beginn mit a = sqrt(n):")
    print(f"a = sqrt(n)\n-> a = sqrt{n}")
    a = math.isqrt(n) + 1
    print(f"a: {a}")
    b2 = a * a - n
    print(f"b2: {b2}")

    while not is_perfect_square(b2):  # Wir verwenden eine benutzerdefinierte Funktion statt sympy
        a += 1
        b2 = a * a - n
        print(f"\na: {a}")
        print(f"b2: {b2}")

    b = int(math.sqrt(b2))  # Wieder die Verwendung von math anstelle von sympy
    print(f"Ermitteltes b: {b}")
    p = a - b
    q = a + b

    return int(p), int(q)

def is_perfect_square(n):
    # Eine einfache Funktion, um zu überprüfen, ob eine Zahl ein perfektes Quadrat ist.
    h = n & 0xF  # Letzte hexadezimale Ziffer, "F" bedeutet "15" (dezimal), also 1111 in binär.

    # Wenn es kein perfektes Quadrat ist, gibt es nichts weiter zu tun
    if h > 9:
        return False

    # Einige schnelle Entscheidungen für die Überprüfung
    if (h != 2 and h != 3 and h != 5 and h != 6 and h != 7 and h != 8):
        t = int(math.sqrt(n))  # Wir verwenden int(), um sicherzustellen, dass wir Ganzzahlen erhalten.
        return t*t == n

    return False

def compute_private_key(e, phi):
    d = pow(e, -1, phi)
    return d

# Faktor n.
p, q = fermat_factor(n)
print(f"Daraus ergibt sich:\np: {p}\nq: {q}")
print(f"n = p * q = {p} * {q} = {p*q}")

print("\n-------------------------\n")

print("priv_key über e & phi (d = e^-1 mod phi(n)):")
phi = (p - 1) * (q - 1)  # Berechnung von Euler's Totient.
print(f"phi = (p - 1) * (q - 1) = {phi}")
d = compute_private_key(e, phi)
print(f"priv_key (n, d): ({n}, {d})")
