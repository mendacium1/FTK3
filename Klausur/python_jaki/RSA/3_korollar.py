import si, math

# Korrolar bietet eine schnellere Methode um Potenzen in Gruppe zu rechnen

# Gegeben z.B.:
# 12345^1234 mod 131
# z^a mod n
n = 131
z = 12345
b = 1234

print(f"Input:\n{z}^{b} mod {n}\n-> mit z^b mod n\n")

phi = si.euler_phi(n)
print(f"phi(n) = {phi}\n")
a = pow(b, 1, phi)
print(f"a = b mod phi = {a}\n")

# 1. Ist a = b (mod phi(n)), dann ist z^a = z^b (mod n).
if pow(z,a,n) == pow(z,b,n):
    print("Korrolar möglich da pow(z,a,n) == pow(z,b,n)\n-> mit a = pow(b, 1, phi)")
else:
    print("Korrolar nicht möglich")
    exit()

# Basis und Exponent können zuerst reduiziert werden.
# z^a = z^(a mod phi(n)) (mod n)
z = pow(z, 1, n)

result = pow(z, a, n)

print(result)


