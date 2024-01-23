import si

# Hasse: Ordnung liegt zwischen Hasse-grenzen
# Wenn Kurve mit prim, dann liegt Ordnung der Kurve zw. Hasse-grenzen der Primzahl
p = 23

curve = si.EC(13, 13, p)
G = si.Point(curve, (1, 2))

print(f"hasse: {si.hasse_bounds(p)}")
print(f"order: {curve.order()}")
