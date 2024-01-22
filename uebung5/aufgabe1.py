import si
import itertools

# SI module
ec = si.EC(3, 6, 11)
print("Ordnung:\t", ec.order())



# Eigenimplementierung
# Elliptische Kurve: y^2 = x^3 + ax + b
a = 3
b = 6
p = 11  # Z_11

# Punkte auf der elliptischen Kurve finden
def is_point_on_curve(x, y, a, b, p):
    return (y * y) % p == (x * x * x + a * x + b) % p

# Alle möglichen Punkte generieren und prüfen, ob sie auf der Kurve liegen
points = [(x, y) for x in range(p) for y in range(p) if is_point_on_curve(x, y, a, b, p)]

# Die Ordnung der Kurve ist die Anzahl der Punkte plus der unendliche Punkt
order_of_curve = len(points) + 1  # +1 für den unendlichen Punkt

print("Punkte:\n", points)
print("Ordnung:\t", order_of_curve)

