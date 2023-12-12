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
    (3, 6, "y^2 = x^3 + 3x + 6", 221),
    (0, 0, "y^2 = x^3 + 0x + 0", 222),
    (0, 0, "y^2 = x^3 + 0x + 0", 223),
    (0, 0, "y^2 = x^3 + 0x + 0", 224)
]

for a, b, title, subplot in curves:
    plot_elliptic_curve(a, b, title, subplot)

plt.tight_layout()
plt.show()

