def elliptic_curve_y_squared(x):
    return x**3 - 8*x + 8

def add_points(x1, y1, x2, y2):
    if x1 == x2 and y1 == -y2:
        return None, None

    if x1 == x2 and y1 == y2:
        if y1 == 0:
            return None, None
        m = (3 * x1**2 - 8) / (2 * y1)
    else:
        m = (y2 - y1) / (x2 - x1)

    x3 = m**2 - x1 - x2
    y3 = m * (x1 - x3) - y1
    return x3, y3

P = (1, 1)
Q = (-2, -4)
R = (34/9, -152/27)

# (P + Q) + R
P_plus_Q = add_points(*P, *Q)
result_a = add_points(*P_plus_Q, *R) if P_plus_Q[0] is not None else (None, None)

# P + (Q + R)
Q_plus_R = add_points(*Q, *R)
result_b = add_points(*P, *Q_plus_R) if Q_plus_R[0] is not None else (None, None)

# Ergebnisse ausgeben
print("a)\t(P + Q) + R:", result_a)
print("b)\tP + (Q + R):", result_b)
