import math
import time

def baby_step_giant_step(base, target, n):
    baby_steps = [pow(base, j, n) for j in range(10)]

    giant_step_multiplier = pow(base, 10, n)
    giant_steps = [1]

    for i in range(1, n // 10 + 1):
        giant_steps.append((giant_steps[-1] * giant_step_multiplier) % n)

    for i, giant in enumerate(giant_steps):
        for j, baby in enumerate(baby_steps):
            if (giant * baby) % n == target:
                return 10 * i + j

    return "No solution found"

def baby_step_giant_step_square(base, target, n):
    m = math.ceil(math.sqrt(n))
    
    baby_steps = {pow(base, j, n): j for j in range(m)}

    c = pow(base, -m, n)
    value = target
    for i in range(m):
        if value in baby_steps:
            return i * m + baby_steps[value]
        value = (value * c) % n

    return "No solution found"

input("Aufgabe 5")
base = 6
target = 37
n = 131

print(baby_step_giant_step_square(base, target, n))

#input("Aufgabe 6")
base = 3116701003
target = 1059878588
n = 3696837919
"""
start_time = time.time()  # Start timing

result = baby_step_giant_step(base, target, n)  # Function call

end_time = time.time()  # End timing

print(result)
print(f"Time taken: {end_time - start_time} seconds")
"""

input("Aufgabe 6 - square")
start_time = time.time()  # Start timing

result = baby_step_giant_step_square(base, target, n)  # Function call

end_time = time.time()  # End timing

print(result)
print(f"Time taken (square): {end_time - start_time} seconds")

