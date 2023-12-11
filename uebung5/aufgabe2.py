import si
# SI module
ec = si.EC(3, 6, 11)
a = si.Point(ec, (9,5))
b = si.Point(ec, (4,4))
c = si.Point(ec, (2,3))

print("Ordnung a:", a.order())
print("Ordnung b:", b.order())
print("Ordnung c:", c.order())



# Eigenimplementierung
# Elliptische Kurve: y^2 = x^3 + ax + b
a = 3
b = 6
p = 11  # Z_11

# Punkte auf der elliptischen Kurve finden
def is_point_on_curve(x, y, a, b, p):
    return (y * y) % p == (x * x * x + a * x + b) % p


# Funktion zur Addition von zwei Punkten auf einer elliptischen Kurve
def add_points(p1, p2, a, p):
    if p1 == p2:  # Punktverdopplung
        if p1[1] == 0:  # Punkt im Unendlichen
            return None
        # Steigung (Lambda) berechnen
        lam = (3 * p1[0] * p1[0] + a) * pow(2 * p1[1], p - 2, p)
    else:  # Addition unterschiedlicher Punkte
        if p1[0] == p2[0]:  # vertikale Linie
            return None
        # Steigung (Lambda) berechnen
        lam = (p2[1] - p1[1]) * pow(p2[0] - p1[0], p - 2, p)

    # Neuen Punkt berechnen
    x3 = (lam * lam - p1[0] - p2[0]) % p
    y3 = (lam * (p1[0] - x3) - p1[1]) % p
    return (x3, y3)

# Punktordnungen ermitteln
def find_order_of_point(point, a, b, p):
    current_point = point
    order = 1
    while current_point is not None:
        current_point = add_points(current_point, point, a, p)
        order += 1
        if current_point == point:
            break
    return order

# Punkte aus der vorherigen Anfrage
points_to_check = [(9, 5), (4, 4), (2, 3)]

# Ordnungen der Punkte bestimmen
point_orders = {point: find_order_of_point(point, a, b, p) for point in points_to_check}
print(point_orders)

