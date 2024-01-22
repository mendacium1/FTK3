import si

# Ordnung einer Kurve
# Gegeben sei Kurve
# Gesucht sei Ordnung
# Ordnung = Anzahl der rationalen Punkte auf der Kurve (+ Unendlichkeitspunkt) aka Größe der Gruppe

# y^2 = x^3 + 3x + 6
# über Z_11

curve = si.EC(3, 6, 11)
print(curve.__str__())
print(f"Order of EC ({curve.__str__()}):\n{curve.order()}")
