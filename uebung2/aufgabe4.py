import math

def fermat_factor(n):
    assert n % 2 != 0  

    a = math.isqrt(n) + 1
    b2 = a * a - n

    while not is_perfect_square(b2):  # Wir verwenden eine benutzerdefinierte Funktion statt sympy
        a += 1
        b2 = a * a - n

    b = int(math.sqrt(b2))  # Wieder die Verwendung von math anstelle von sympy
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

def main():
    # Teil 1: Faktorisieren Sie die Zahl mit der Fermat-Methode.
    n = 3718548079  # Öffentlicher Schlüssel.
    e = 65537  # Öffentlicher Exponent.

    # Faktor n.
    p, q = fermat_factor(n)
    print(f"Faktoren von n: p = {p}, q = {q}")

    # Teil 2: Generieren Sie den privaten Schlüssel.
    phi = (p - 1) * (q - 1)  # Berechnung von Euler's Totient.
    d = compute_private_key(e, phi)

    print(f"Privater Schlüssel (n, d): ({n}, {d})")

if __name__ == "__main__":
    main()

