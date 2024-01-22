import galois
import numpy as np

# irreducible_poly = galois.Poly([1, 0, 0, 0, 1, 1, 0, 1, 1], field=galois.GF2)

GF = galois.GF(2**8, irreducible_poly="x^8+x^4+x^3+x+1")
print(GF.irreducible_poly)

poly = GF("a^7 + a^3")
print(poly)
print(f"Inverse with numpy: {np.reciprocal(poly)}")
print(f"Inverse without numpy: {poly ** -1}")

inverse = np.reciprocal(poly)
print(bin(inverse))
inverse_coeffs = [int(d) for d in str(bin(inverse))[2:]]

vector = np.array(inverse_coeffs)
vector = [ 1, 1, 0, 1, 1, 0, 0, 1]

matrix = np.array([
    [1, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1]])

b = np.array([1, 1, 0, 0, 0, 1, 1, 0])

# matrix * vector + b
transformed_vector = (np.dot(matrix, vector) % 2 + b % 2) % 2

print(transformed_vector)
