import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
As = [40, 70, 30, -50, 50, -40]
NR = 15
peak_positions = [2960, 3000, 3030, 3050, 3100, 3110]
gamma = 10


def add_noise_to_array(arr, percentage):
    arr_max = np.max(arr)
    arr_min = np.min(arr)
    max_noise = abs((arr_max - arr_min) * percentage / 100)
    noise = np.abs(np.random.normal(0, max_noise, arr.shape))
    return arr + noise

def func(omega):
    summation = complex(NR, 0)
    for i in range(len(As)):
        summation += (As[i] / complex((omega - peak_positions[i]), gamma))
    return abs(summation)**2


x = np.arange(2800, 3300, 5)
y = np.vectorize(func)(x)
y2 = add_noise_to_array(y, 5)
print(tabulate(np.array([x, y, y2]).T))
fig, ax = plt.subplots()
ax.plot(x, y2)
extra_index = 0
normal_modes = peak_positions[extra_index:]
extra_modes = peak_positions[:extra_index]
for peak in normal_modes:
    pass
    ax.axvline(peak*1, color="black", ls="dotted")

for peak in extra_modes:
    pass
    ax.axvline(peak, color="red", ls="dotted")

fig.savefig("10 non res {} {}.png".format(As, peak_positions), dpi=600)