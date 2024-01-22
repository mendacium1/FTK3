import galois

k = galois.GF( 2**6, irreducible_poly="x^6+x+1", repr="poly")

a_str = "x^5+x+1"
b_str = "x^3+x^2"

a = k(a_str)
b = k(b_str)

print(f"\n[{a_str}]_p + [{b_str}]_p")
print(a+b)
print(f"\n[{a_str}]_p - [{b_str}]_p")
print(a-b) # - wird zu +, da mod 2
print(f"\n[{a_str}]_p * [{b_str}]_p")
print(a*b)
print(f"\n[{a_str}]_p / [{b_str}]_p")
print(a/b)
