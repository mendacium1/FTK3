import si

# Entschlüsseln einer Nachricht unter verwendung RSA-CRS

# Chiffre
c = 1969
# private key
p = 47
q = 53
d = 2153

print(f"Input:\nc = {c}\np = {p}\nq = {q}\nd = {d}")

print("\n-------------------------\n")

print("d_p = d mod p-1")
print(f"-> d_p = {d} mod {p-1}")
d_p = pow(d, 1, p - 1)
print("d_q = d mod q-1")
print(f"-> d_q = {d} mod {q-1}")
d_q = pow(d, 1, q - 1)
print(f"d_p: {d_p}")
print(f"d_q: {d_q}")

print("\n-------------------------\n")

print(f"Erweiterter Euklid von p & q ({p} & {q}):")
_, x, y = si.extended_gcd(p, q, verbose=1)
print(f"q*y + p*x = 1\n-> ({q} * {y}) + ({p} * {x}) = 1")
print(f"y: {y}")

print("\n-------------------------\n")

print("m_p := c^(d_p) mod p")
print(f"-> m_p = {c}^{d_p} mod {p}")
m_p = pow(c, d_p, p)
print(f"m_p: {m_p}")

print("m_q := c^(d_q) mod q")
print(f"-> m_q = {c}^{d_q} mod {q}")
m_q = pow(c, d_q, q)
print(f"m_q: {m_q}")

print("\n-------------------------\n")

print("h = (m_p - m_q) * y mod p")
print(f"-> h = ({m_p} - {m_q}) * {y} mod {p}")
h = pow((m_p - m_q) * y, 1, p)
print(f"h: {h}")

print("\n-------------------------\n")

print("m := m_q + q*h mod p*q")
print(f"-> m := {m_q} + {q}*{h} mod {p}*{q}")
m = pow(m_q + q*h, 1, p*q)
print(f"m: {m}")
