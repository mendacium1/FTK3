from sympy.ntheory import factorint
from sympy.ntheory.modular import solve_congruence

# Pohlig-Hellman-Algorithmus (zu kleiner primfaktor des public keys -> zur Verhinderung möglichst große prime Ordnung)
# Gegeben Zahl target (37), Basis b (6) und Gruppe (Z^*_131)

base = 6
target = 37
n = 131

def baby_step_giant_step(base, target, n):
    m = int(n**0.5) + 1
    baby_steps = {pow(base, j, n): j for j in range(m)}
    base_inv = pow(base, -m, n)

    for i in range(m):
        y = (target * pow(base_inv, i, n)) % n
        if y in baby_steps:
            return i * m + baby_steps[y]
    return None

#find factors of n
#solve discrete logarithm in subgroups
#combine results with chinese remainder
def pohlig_hellman(base, target, n):
    factors = factorint(n - 1)
    congruences = []

    for p, e in factors.items():
        pe = p ** e
        base_pe = pow(base, (n - 1) // pe, n)
        target_pe = pow(target, (n - 1) // pe, n)
        x_pe = baby_step_giant_step(base_pe, target_pe, n)
        congruences.append((x_pe, pe))

    return solve_congruence(*congruences)[0]



print(f"Input:\nbase = {base}\ntarget = {target}\nn = {n}")

print("\n-------------------------\n")

discrete_log = pohlig_hellman(base, target, n)

print(f"Diskreter Logarithmus: {discrete_log}")
