import time
from sympy.ntheory import factorint
from sympy.ntheory.modular import solve_congruence

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

input("Aufgabe 7")
base = 6
target = 37
n = 131

start_time = time.time()  # Start timing
discrete_log = pohlig_hellman(base, target, n)
end_time = time.time()  # End timing

print(discrete_log)
print(f"Time taken (square): {end_time - start_time} seconds")

input("Aufgabe 8")
base = 3116701003
target = 1059878588
n = 3696837919
start_time = time.time()  # Start timing
discrete_log = pohlig_hellman(base, target, n)
end_time = time.time()  # End timing

print(discrete_log)
print(f"Time taken (square): {end_time - start_time} seconds")
