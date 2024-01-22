import si

# Ordnung einer Gruppe = Anzahl der Elemente in Gruppe
# Ordnung einer Zahl a (in einer Gruppe) = kleinste positive ganze Zahl m, so dass a^m = neutrales Element (m = order)
# Gegeben z.B.: Ordnung einer Zahl (a = 3116701003) in Gruppe Z^*_3696837919
# Gesucht Ordnung der Zahl a

a = 3116701003
p = 3696837919

print(f"Input:\na = {a}\np = {p}")

order = si.multiplicative_order(3116701003, 3696837919)

print(f"Ordnung: {order}")

print("\nBeweis:")
print(f"a^m mod p = neutrales Element:\n{a}^{order} mod {p} = {pow(a,order,p)}")
