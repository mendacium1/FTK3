import sympy as sp

def elliptic_curve_y_squared(x):
    return x**3 + 2*x + 4

def is_on_curve(x, y):
    return y**2 == elliptic_curve_y_squared(x)

# ! neutrales element könnte dazu addiert werden
def add_points(x1, y1, x2, y2):
    # Inverse Punkte
    if x1 == x2 and y1 == -y2:
        return None, None  # "Unendlicher Punkt" auf der Kurve

    # Berechnen der Steigung (m) und des resultierenden Punktes (x3, y3)
    if x1 == x2 and y1 == y2:  # Punktverdopplung
        if y1 == 0:  # Tangente ist vertikal
            return None, None
        m = (3 * x1**2 + 2) / (2 * y1)
    else:  # Verschiedene Punkte addieren
        m = (y2 - y1) / (x2 - x1)

    x3 = m**2 - x1 - x2
    y3 = m * (x1 - x3) - y1
    return x3, y3

# Gegebene Punkte
points = [((-1, 1), (2, 4)),   # a) P = (-1, 1), Q = (2, 4)
          ((-1, 1), (-1, 1)),  # b) P = (-1, 1), Q = (-1, 1) (Punktverdopplung)
          ((-1, 1), (-1, -1))] # c) P = (-1, 1), Q = (-1, -1) (inverse Punkte)

# Berechnen von P + Q für jedes Paar und Überprüfen, ob das Ergebnis auf der Kurve liegt
for (x1, y1), (x2, y2) in points:
    x3, y3 = add_points(x1, y1, x2, y2)
    on_curve = is_on_curve(x3, y3) if x3 is not None else "Unendlicher Punkt"
    print(f"P + Q für {(x1, y1)} und {(x2, y2)}: {(x3, y3)}, auf der Kurve: {on_curve}")
