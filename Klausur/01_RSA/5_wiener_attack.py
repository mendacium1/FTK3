import si, itertools, math

# Wiener Attacke - zu kleiner privater Exponent d
# Gegeben pub_key z.B.: (n, e) = (308911, 87943)
# Gesucht: priv_key

n = 308911
e = 87943

print(f"Input:\nn = {n}\ne = {e}")

print("\n-------------------------\n")

print("Siehe Kettenbruchentwicklung rechts:")

si.extended_gcd(n, e, verbose=1)

fractions = list(si.cf_approx(e,n))

print(f"Brüche:\n{fractions}")

fractions = itertools.islice(fractions, 1, None, 2)
p = 0
q = 0

# Implementation Jürgen
DPhis = ((D, (e * D - 1) // K) for (K, D) in fractions if D % 2 == 1 and (e * D - 1) % K == 0)
pqs = ((-(n - Phi + 1), n) for (D, Phi) in DPhis)

potential_factors = []

for (p, q) in pqs:
    x = (-p + math.isqrt(p**2 - 4*q)) // 2
    if n % x == 0:
        potential_factors.append((x, n // x))

        # Once we have the correct (p, q) pair, break out of the loop
        break

# Extracting and printing the factors and 'd'
if potential_factors:
    p, q = potential_factors[0]
    print(f"p: {p}")
    print(f"q: {q}")

    # Calculating 'd'
    phi = (p-1) * (q-1)
    d = pow(e, -1, phi)
    print(f"d: {d}")
else:
    print("No factors found. Wiener's attack was not successful with the given input.")


#Check if solution of n from p and q is correct
print("p*q=n?: " + str(n==p*q))

m = 420
print(f"Test solution with message {m}")
c = pow(m,e,n)
m_entsch = pow(c,d,n)
print("message = message entschlüsselt?: " + str(m==m_entsch))


