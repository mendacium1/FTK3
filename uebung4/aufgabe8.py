import numpy as np
import matplotlib.pyplot as plt

def plot_elliptic_curve(a, b, title, subplot):
    def elliptic_curve(x):
        y2 = x**3 + a*x + b
        with np.errstate(all='ignore'):
            y = np.sqrt(y2)
        return y

    x_values = np.linspace(-10, 10, 400)

    y_values = elliptic_curve(x_values)

    plt.subplot(subplot)
    plt.plot(x_values, y_values, 'b', label="y^2 = x^3 + {}x + {}".format(a, b))
    plt.plot(x_values, -y_values, 'b')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()

plt.figure(figsize=(12, 10))

curves = [
    (-4, 4, "y^2 = x^3 − 4x + 4", 221),
    (2, 4, "y^2 = x^3 + 2x + 4", 222),
    (-8, 8, "y^2 = x^3 − 8x + 8", 223),
    (1, 2, "y^2 = x^3 + x + 2", 224)
]

for a, b, title, subplot in curves:
    plot_elliptic_curve(a, b, title, subplot)

plt.tight_layout()
plt.show()

