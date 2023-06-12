import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from tabulate import tabulate

theta0 = np.deg2rad(90)
sigma = np.deg2rad(1)

def odf(theta):
    return np.exp((-0.5 * (theta - theta0)**2) / sigma)

x = np.array([np.deg2rad(theta) for theta in range(0, 180, 1)])
y = odf(x)
# print(tabulate(np.array([x, y]).T))
plt.plot(x, y)
plt.show()