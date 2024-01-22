import random
import time

base = 3116701003
target = 1059878588
n = 3696837919
attempts = 1000000

start_time = time.time()

found = False
tried_exponents = set()

while len(tried_exponents) < attempts:
    x = random.randint(0, n-1)  # Random number between 0 and n-1
    if x in tried_exponents:
        continue  # Skip if this exponent has already been tried

    tried_exponents.add(x)

    if pow(base, x, n) == target:
        found = True
        print(f"Found x: {x}")
        break

end_time = time.time()
time_taken = end_time - start_time

if not found:
    print("No solution found with the given number of attempts.")

print(f"Time taken for {attempts} attempts: {time_taken} seconds")

# Estimate total time for brute-forcing all numbers
total_time_estimate = (time_taken / attempts) * (n - 1)
print(f"Estimated time to brute-force all numbers: {total_time_estimate} seconds\n /2 for average time: {total_time_estimate/2} seconds")
