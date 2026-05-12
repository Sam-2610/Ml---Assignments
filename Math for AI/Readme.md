# Linear Algebra & Probability in Python
 
A collection of Python programs covering foundational **Linear Algebra** and **Probability & Statistics** concepts used in Data Science, Machine Learning, and AI.
 
---
 
## Libraries Used
 
| Library | Purpose |
|---------|---------|
| `numpy` | Numerical computations, array operations, random sampling |
| `matplotlib` | Plotting and data visualization |
| `scipy` | Statistical distributions and mathematical functions |
 
### Install Dependencies
 
```bash
pip install numpy matplotlib scipy
```
 
---
 
## Files Included
 
| # | File | Concepts Covered |
|---|------|-----------------|
| 1 | `Linear_Algebra.py` | Plotting `y = sqrt(x)` using NumPy and Matplotlib |
| 2 | `Probability.py` | Binomial Distribution, Uniform Distribution, Normal Distribution |
 
---
 
## Linear Algebra
 
### File: `Linear_Algebra.py`
 
#### Plot y = √x
 
Plots the square root function over a range of values using NumPy and Matplotlib.
 
```python
import numpy as np
import matplotlib.pyplot as plt
 
x_values = np.linspace(0, 100, 100_000)   # 100,000 points from 0 to 100
y_values = np.sqrt(x_values)              # compute sqrt for each x
 
plt.plot(x_values, y_values, label="y = sqrt(x)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("y = sqrt(x)")
plt.legend()
plt.grid(True)
plt.show()
```
 
**Line by line:**
 
| Code | Description |
|------|-------------|
| `np.linspace(0, 100, 100_000)` | Creates 100,000 evenly spaced points between 0 and 100 |
| `np.sqrt(x_values)` | Applies square root to every value in the array at once |
| `plt.plot()` | Draws the line graph |
| `plt.xlabel / ylabel` | Labels for x and y axes |
| `plt.legend()` | Shows the label "y = sqrt(x)" on the chart |
| `plt.grid(True)` | Adds a background grid for readability |
| `plt.show()` | Renders and displays the plot |
 
**Graph shape:** Starts steep near 0, then gradually flattens — a classic concave curve.
 
```
y
|*
| *
|  *
|    *
|       *
|            *
|___________________________ x
0                          100
```
 
---
 
## Probability & Statistics
 
### File: `Probability.py`
 
---
 
### 1. Binomial Distribution
 
Models the probability of getting exactly `k` successes in `n` trials, where each trial has probability `p` of success.
 
```python
from scipy.stats import binom
 
n = 5     # number of trials
k = 3     # desired number of successes
p = 0.5   # probability of success per trial
 
prob = binom.pmf(k, n, p)
print(prob)   # Output: 0.3125
```
 
**Real-world analogy:** What is the probability of getting exactly 3 heads in 5 coin flips?
 
- `binom.pmf(k, n, p)` — PMF stands for **Probability Mass Function**, gives the exact probability for a discrete outcome
- Formula: `P(X=k) = C(n,k) * p^k * (1-p)^(n-k)`
- Result: `0.3125` → there's a **31.25%** chance of getting exactly 3 heads in 5 flips
---
 
### 2. Uniform Distribution
 
Every value in a given range has an **equal probability** of occurring.
 
```python
import numpy as np
import matplotlib.pyplot as plt
 
values = np.random.uniform(0, 10, 1_000_000)   # 1 million random values between 0 and 10
 
plt.hist(values, bins=100, density=True, alpha=0.3)
plt.title("Uniform Distribution")
plt.xlabel("Value")
plt.ylabel("Density")
plt.grid(True)
plt.show()
```
 
**Line by line:**
 
| Code | Description |
|------|-------------|
| `np.random.uniform(0, 10, 1_000_000)` | Generates 1 million random numbers uniformly between 0 and 10 |
| `plt.hist(..., bins=100)` | Divides data into 100 buckets and plots frequency |
| `density=True` | Normalizes so the total area = 1 (converts to probability density) |
| `alpha=0.3` | Makes the bars semi-transparent (0 = invisible, 1 = solid) |
 
**Graph shape:** A flat, rectangular bar chart — all values appear roughly equally often.
 
**Real-world analogy:** Rolling a fair die — every number (1–6) has equal probability.
 
---
 
### 3. Normal Distribution (Bell Curve)
 
The most important distribution in statistics — data clusters symmetrically around a mean, forming a bell shape.
 
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
 
mu = 70      # Mean (center of the curve)
sigma = 10   # Standard deviation (spread/width)
 
x = np.linspace(30, 110, 1000)    # x-axis range
y = norm.pdf(x, mu, sigma)        # compute probability density
 
plt.plot(x, y, color="black", linewidth=2, label="Normal Distribution")
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
```
 
**Line by line:**
 
| Code | Description |
|------|-------------|
| `mu = 70` | Mean — the peak of the bell curve is at 70 |
| `sigma = 10` | Std. deviation — controls how wide/narrow the bell is |
| `np.linspace(30, 110, 1000)` | 1000 evenly spaced x values from 30 to 110 |
| `norm.pdf(x, mu, sigma)` | PDF = Probability Density Function — height of curve at each x |
 
**The 68-95-99.7 Rule:**
 
```
        68% of data
     ←─────────────→
          95% of data
     ←───────────────────→
              99.7% of data
     ←───────────────────────────→
              |
    ──────────┬──────────
           μ = 70
    (σ=10: range roughly 40 to 100)
```
 
| Range | % of Data |
|-------|-----------|
| μ ± 1σ (60 to 80) | ~68% |
| μ ± 2σ (50 to 90) | ~95% |
| μ ± 3σ (40 to 100) | ~99.7% |
 
**Real-world analogy:** Heights, exam scores, and measurement errors all follow a normal distribution.
 
---
 
## Comparison of Distributions
 
| Distribution | Shape | Use Case |
|---|---|---|
| Binomial | Discrete spikes | Coin flips, pass/fail outcomes |
| Uniform | Flat rectangle | Random number generation, fair dice |
| Normal | Bell curve | Heights, test scores, natural phenomena |
 
---
 
## How to Run
 
```bash
python Linear_Algebra.py
python Probability.py
```
 
> Each script will open a matplotlib window displaying the plot. Close the window to continue to the next plot.
 
---
 
## Author
 
**Satyam Sagar**
📧 satyamsagar827@gmail.com