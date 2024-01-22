import math

# Baby-Step Giant-Step zur Berechnung des diskreten Logarithmus
# Gegeben Zahl target (37), base (6) und Gruppe (Z^*_131)
# a = Ergebnis
# b = pub_key
# n = aus Gruppe
# a = b^(x) mod prim

base = 6
target = 37
n = 131

def baby_step_giant_step(base, target, n):
    m = math.ceil(math.sqrt(n))
    
    baby_steps = {pow(base, j, n): j for j in range(m)}

    c = pow(base, -m, n)
    value = target
    for i in range(m):
        if value in baby_steps:
            return i * m + baby_steps[value]
        value = (value * c) % n

    return "No solution found"

print(f"Input:\nbase = {base}\ntarget = {target}\nn = {n}")

print("\n-------------------------\n")

print(f"Diskreter Logarithmus: {baby_step_giant_step(base, target, n)}")

