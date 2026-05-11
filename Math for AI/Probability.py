# Binomial Distribution

from scipy.stats import binom
n = 5
k = 3
p = 0.5

prob = binom.pmf(k,n,p)
print(prob)

# Uniform Distribution

import numpy as np
import matplotlib.pyplot as plt

values = np.random.uniform(0, 10, 1_000_000)  # Values from 0 to 10

plt.hist(values, bins=100, density=True, alpha=0.3)

plt.title("Uniform Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.grid(True)

plt.show()

# Normal Distribution

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

""" Parameters """

mu = 70 # Mean
sigma = 10 # std.dev

# x-axis value
x = np.linspace(30,110,1000)

# Normal PDF
y = norm.pdf(x,mu,sigma)

# Plot the bell Curve
plt.plot(x,y,color="black",linewidth = 2,label = "Normal Distribution")

# Labels and title
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()