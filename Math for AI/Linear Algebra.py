# Plot y = sqrt(x)
import numpy as np
import matplotlib.pyplot as plt

# Define x values (x >= 0 for sqrt)
x_values = np.linspace(0, 100, 100_000)
y_values = np.sqrt(x_values)

# Plot
graph = plt.plot(x_values, y_values, label="y = sqrt(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("y = sqrt(x)")
plt.legend()
plt.grid(True)
plt.show()