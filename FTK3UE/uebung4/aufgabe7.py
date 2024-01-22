def elliptic_curve_y_squared(x):
    return x**3 + x + 2

def add_points(x1, y1, x2, y2):
    if x1 == x2 and y1 == -y2:
        return None, None

    if x1 == x2 and y1 == y2:
        if y1 == 0:
            return None, None
        m = (3 * x1**2 + 1) / (2 * y1)
    else:
        if x1 == x2:
            return None, None
        m = (y2 - y1) / (x2 - x1)

    x3 = m**2 - x1 - x2
    y3 = m * (x1 - x3) - y1
    return x3, y3

P = (1, 2)
current_point = P
order = 1

while True:
    current_point = add_points(*current_point, *P)
    order += 1
    if current_point[0] is None:
        break

print("Die endliche Ordnung von P =", P, "auf der Kurve ist", order)
