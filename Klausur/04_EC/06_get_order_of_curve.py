import si

# Ordnung einer Kurve
# Gegeben sei Kurve
# Gesucht sei Ordnung
# Ordnung = Anzahl der rationalen Punkte auf der Kurve (+ Unendlichkeitspunkt) aka Größe der Gruppe

# y^2 = x^3 + 3x + 6
# über Z_11
a = 28
b = 42
p = 89

curve = si.EC(a, b, p)
print(curve.__str__())
curve_order = curve.order()
print(f"Order of EC ({curve.__str__()}):\n{curve_order}")

# Test
points = curve.list_of_points()

print(f"Primefactors of {curve_order}: {si.prime_factors(curve_order)}")
for point in points:
    print(f"Point: {point}")
    print(f"\torder: {point.order()}")
