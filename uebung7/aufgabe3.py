import galois

k = galois.GF( 7**3, irreducible_poly="x^3+x^2+1", repr="poly")

a = k("5b^2 + 2b")
print(a**4)
