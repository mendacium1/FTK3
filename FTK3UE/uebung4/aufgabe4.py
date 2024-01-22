import sympy as sp

# function:
def elliptic_curve_y_squared(x):
    return x**3 - 4*x + 4

# check:
def is_on_curve(x, y):
    return y**2 == elliptic_curve_y_squared(x)

# a)
point1 = (-2, 2)
point2 = (-1, 7)
point1_on_curve = is_on_curve(*point1)
point2_on_curve = is_on_curve(*point2)

print(f"Punkt {point1} auf der Kurve: {point1_on_curve}")
print(f"Punkt {point2} auf der Kurve: {point2_on_curve}")

# b)
y = sp.symbols('y')
y_values_8 = sp.solve(y**2 - elliptic_curve_y_squared(8), y)
y_values_minus_8 = sp.solve(y**2 - elliptic_curve_y_squared(-8), y)

print(f"y-Koordinaten für (8, y): {y_values_8}")
print(f"y-Koordinaten für (-8, y): {y_values_minus_8}")
