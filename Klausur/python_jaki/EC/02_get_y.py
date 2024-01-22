import si

# Berechnen der y-Koordinate zu einer x-Koordinate auf einer elliptischen Kurve
# Gegeben elliptische Kurve (E: y^2 = x^3 - 4x + 4), x-Koordinate (8 oder -8)
# Gesucht y-Koordinate

x1 = 8
x2 = -8

curve = si.EC(4, 4)
y1 = curve.points_with_xcoord(x1)
y2 = curve.points_with_xcoord(x2)

print(f"y1 mit (x1 = {x1}): {y1}")
print(f"y2 mit (x2 = {x2}): {y2}")
