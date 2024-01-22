import galois
import numpy as np

GF = galois.GF(2**8, irreducible_poly="x^8+x^4+x^3+x+1")

print("Input:")
print(GF.properties)

inverse = np.reciprocal(GF("a^7 + a^3"))

inverse_coeffs = [int(d) for d in str(bin(inverse))[2:]][::-1]

vector = np.array(inverse_coeffs)
print(f"Vector to transform:\n{vector}")

matrix = np.array([
    [1, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1]])
print(f"Matrix:\n{matrix}")

b = np.array([1, 1, 0, 0, 0, 1, 1, 0])
print(f"Vector b:\n{b}")

# matrix * vector + b
transformed_vector = (np.dot(matrix, vector) % 2 + b % 2) % 2

print(transformed_vector)
