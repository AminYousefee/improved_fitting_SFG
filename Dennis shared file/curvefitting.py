from numpy import *
from scipy.optimize import fmin_tnc
import pylab

def model(x, xstart, smax, offset, k):
    return (smax - offset)*exp(-k*(x-xstart)) + offset

def error(p):
    residuals = signal2 - model(time2, 120., p[0], p[1], p[2])
    return sum(residuals**2)

# setup
pylab.rc('font', size=8)
fig = pylab.figure(figsize=(3.25, 2.5))
ax = fig.add_subplot(111)

# grab the data
time, signal = loadtxt('data.dat', unpack=True)

# create a subset of the data (for fitting)
time2 = time[12:]
signal2 = signal[12:]

# inspect an initial guess at the solution
timeFine = linspace(120, 400, 1000)
ax.plot(timeFine, model(timeFine, 120., 47., 15., 0.05), label='guess 1')
ax.plot(timeFine, model(timeFine, 120., 46.5, 17., 0.05), label='guess 2')

# search for optimum solution
p0 = [47., 15., 0.05]
pbounds = [(30,100), (5,20), (0.001, 0.1)]
opt = fmin_tnc(error, p0, approx_grad=True, bounds=pbounds, maxfun=500)[0]
print(opt)
ax.plot(timeFine, model(timeFine, 120., opt[0], opt[1], opt[2]), label='best fit')

# plot the data
ax.plot(time, signal, marker='o', ms=4, lw=0, mec='k', mfc='w', mew=1, label='data')
ax.set_xlabel('time / s')
ax.set_ylabel('signal / mV')
ax.set_xlim(0, 340)
ax.set_ylim(10, 50)
ax.legend(loc=0, frameon=False, fontsize=8)
fig.set_tight_layout(True)
pylab.savefig('fit.png', dpi=600)

