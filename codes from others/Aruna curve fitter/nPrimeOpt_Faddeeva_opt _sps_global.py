#!/usr/bin/env python
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fmin_tnc
import pylab
import pandas as pd
from numba import njit

pylab.rc('font', size=8)
fig = pylab.figure(figsize=(12., 4.25))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)


def sampleSize(step):
    listval = []
    for facAir in np.arange(0,1.+step,step):
        for facWater in np.arange(0,(1.-facAir+step) , step):
            if facAir+facWater < 1.+step:
                faccyano = 1. - facAir-facWater
                listval.append(facAir)
    return print(f"Number of data points:{len(listval)}") 


def R(Axxz, Axzx, Azzz):
    Rval = (Axxz - Axzx)/(Azzz+2*Axzx)
    return Rval

def water(wl):
    return nWat(wl) + 1j*kWat(wl)

def cyanophenol(wl):
    return nCyano(wl) + 1j*kCyano(wl)

def eL(microns, theta1):
    def mixingformulaNPrime3Phase(fractionNPrimeAir , fractionNPrimeWater):
        dcAir = (1. + 1e-10j)**2
        dcWater = (water(microns))**2
        dcCyano = (cyanophenol(microns))**2

        val = fractionNPrimeAir*((dcAir-1) / (dcAir+2)) + (fractionNPrimeWater)*((dcWater-1) / (dcWater+2))+ (1-fractionNPrimeWater-fractionNPrimeAir)*((dcCyano-1) / (dcCyano+2))
        dcNPrime = (val*2 + 1)/(1-val)
        NPrimeMix = dcNPrime**0.5

        NPrimeMixn = (NPrimeMix.real)
        NPrimeMixk = (NPrimeMix.imag)

        return NPrimeMixn + 1j*NPrimeMixk

    n1 = 1.0
    n2 = nWat(microns)

    theta2 = np.arcsin(n1*np.sin(theta1)/n2)

    rp = (n2*np.cos(theta1) - n1*np.cos(theta2)) / (n2*np.cos(theta1) + n1*np.cos(theta2))
    rs = (n1*np.cos(theta1) - n2*np.cos(theta2))/(n1*np.cos(theta1) +  n2*np.cos(theta2))

    eLx = (1. - rp)*np.cos(theta1)
    eLy = 1. + rs
    nprime = mixingformulaNPrime3Phase(facAir, facWater)
    # nprime = mixingformulaNPrime3Phase(0.75, 0.20)
    eLz = (1. + rp)*((n1/nprime)**2)*np.sin(theta1)

    # print(mixingformulaNPrime3Phase(0.56,0.18))
    return eLx, eLy, eLz

@njit
def chi2(wavenum, NR, amplitude, widthg, omega0): # recheck
    def f(x, w):
        return (np.exp(-(x-omega0)**2/(2*widthg**2))) * (amplitude/(x-w-1j*widthl))

    x0 = omega0 - 4*widthnew 
    x1 = omega0 + 4*widthnew
    sig = []
    dummy = np.arange(x0,x1+0.1,0.1) #???
    # print (sum(dummy))
    for w in wavenum:
        sig.append(np.sum(f(dummy, w)) * (dummy[1]-dummy[0]))
    return np.array(sig) + NR

def chi2ssp(wavenum, NR, amplitude, widthg, omega0):
    eLx_SFG, eLy_SFG, eLz_SFG = eL(wl_SFG, theta_SFG)
    eLx_vis, eLy_vis, eLz_vis = eL(wl_vis, theta_VIS)
    eLx_IR, eLy_IR, eLz_IR = eL(wl_IR, theta_IR)
    LLLyyz = eLy_SFG * eLy_vis * eLz_IR

    sigval = LLLyyz * chi2(wavenum, NR, amplitude, widthg, omega0)
    return abs(sigval)**2 # mag squared yyz - SSP intensity

def chi2sps(wavenum, NR, amplitude, widthg, omega0):
    eLx_SFG, eLy_SFG, eLz_SFG = eL(wl_SFG, theta_SFG)
    eLx_vis, eLy_vis, eLz_vis = eL(wl_vis, theta_VIS)
    eLx_IR, eLy_IR, eLz_IR = eL(wl_IR, theta_IR)
    LLLyzy = eLy_SFG * eLz_vis * eLy_IR

    sigval = LLLyzy*chi2(wavenum, NR, amplitude, widthg, omega0)
    return abs(sigval)**2 

def chi2ppp(wavenum, NRxxz, Axxz, NRxzx, Axzx, NRzzz, Azzz, widthg, omega0):
    eLx_SFG, eLy_SFG, eLz_SFG = eL(wl_SFG, theta_SFG)
    eLx_vis, eLy_vis, eLz_vis = eL(wl_vis, theta_VIS)
    eLx_IR, eLy_IR, eLz_IR = eL(wl_IR, theta_IR)
    LLLzzz = eLz_SFG * eLz_vis * eLz_IR
    LLLzxx = eLz_SFG * eLx_vis * eLx_IR
    LLLxzx = eLx_SFG * eLz_vis * eLx_IR
    LLLxxz = eLx_SFG * eLx_vis * eLz_IR

    sigPPP_xxz = -LLLxxz*chi2(wavenum, NRxxz, Axxz, widthg, omega0) # data from ssp
    sigPPP_zzz = LLLzzz *chi2(wavenum, NRzzz, Azzz, widthg, omega0) # zzz - unique to ppp
    sigPPP_other2 = (LLLzxx - LLLxzx)*chi2(wavenum, NRxzx, Axzx, widthg, omega0) # data from sps

    return abs(sigPPP_xxz + sigPPP_zzz + sigPPP_other2)**2 

def error(params, wavenum, exptSSP, exptSPS, exptPPP):
    modelSPS = chi2sps(wavenum, params[4], params[5], params[6], params[7])
    modelSSP = chi2ssp(wavenum, params[2], params[3], params[6], params[7])
    modelPPP = chi2ppp(wavenum, params[2], params[3], params[4], params[5], params[0], params[1], params[6], params[7])
    errorTotal = sum((modelSPS - exptSPS)**2) + sum((modelSSP - exptSSP)**2) + 100*sum((modelPPP - exptPPP)**2)
    return errorTotal


###################################################################### Quartz calibration and loading the raw data

def raw_data(pol):

    wn, norm = np.loadtxt(f'{pol}.txt', unpack=True)
    wn1, sd = np.loadtxt(f'{pol}_sd.txt', unpack=True)

    if pol == "ssp":
        ax = ax2
    if pol == "sps":
        ax = ax1
    if pol == "ppp":
        ax = ax3
    ax.fill_between(wn, norm-sd, norm+sd,
            alpha=.7, edgecolor='#3F7F4C', facecolor='#A4E9D5',
            linewidth=0)

    return (wn, norm)



def raw_data_sd(pol):

    wn1, sd = np.loadtxt(f'{pol}_sd.txt', unpack=True)

    return sd

###################################################################### Loading water refractive index data
wn, n, k = np.loadtxt('segelstein_wn.dat', unpack=True)
wl = 1.0e4/wn
nWat = interp1d(wl, n)
kWat = interp1d(wl, k)

###################################################################### Loading cyanophenol refractive index data
wn, n, k = np.loadtxt("cyanophenolDataWn.txt", unpack=True)
wl = 1.0e4/wn
nCyano = interp1d(wl, n)
kCyano = interp1d(wl, k)

###################################################################### Fixed parameters
wnFine = np.linspace(2180, 2300, 61)
widthl = 2.
widthnew = 4.
# Rval = 0.05

wl_vis = 0.532
wl_IR = 1.0e4/wnFine
wl_SFG = 1./(1./wl_vis + 1/wl_IR)

N1_vis = water(wl_vis)
N1_ir = water(wl_IR)
N1_sfg = water(wl_SFG)

theta_VIS = 60/180.*np.pi
theta_IR = 55/180.*np.pi
theta_SFG = np.arcsin((N1_vis*np.sin(theta_VIS)/wl_vis + N1_ir*np.sin(theta_IR)/wl_IR)/(N1_sfg/wl_SFG))

###################################################################### Parameters to optimize

initialTotal = [1.66727889,  0.5536648,1.56259668, 1.03315328, 9.28759536e-01, 6.48062047e-01, 4.64871167e+00, 2.23163066e+03]
pbounds = [(-5,5), (-5, 5), (-5,5),(-5,5), (-5, 5), (-5,5), (-5,5),(2220., 2240.)]


###################################################################### constants

sspwnexpt, sspexpt = raw_data('ssp')
spswnexpt, spsexpt = raw_data('sps')
pppwnexpt, pppexpt = raw_data('ppp')

###################################################################### Loop info
output = []
bestError = 9.9e9
step = 1.
# step = 0.01 # check if these data produce a flowless ternary plot
sampleSize(step=step)


for facAir in np.arange(0,1.+step,step):
    for facWater in np.arange(0,(1.-facAir+step) , step):
        if facAir+facWater < 1.+step:
            faccyano = 1. - facAir-facWater
            print(facAir, facWater, faccyano)

        # first fit SSP data
            opt = fmin_tnc(error, initialTotal, approx_grad=True, args=(pppwnexpt, sspexpt,spsexpt,pppexpt),bounds=pbounds, maxfun=1000, disp=0)[0]
            errorTotal = error(opt, pppwnexpt, sspexpt, spsexpt, pppexpt)
            ratio = R(opt[3], opt[5], opt[1])


            def mixingformulaNPrime3Phase_02(fractionNPrimeAir , fractionNPrimeWater):
                wl_vis_02 = 0.532
                wl_IR_02 = 1.0e4/wnFine
                microns_02 = 1./(1./wl_vis_02 + 1/wl_IR_02)
                dcAir = (1. + 1e-10j)**2
                dcWater = (water(microns_02))**2
                dcCyano = (cyanophenol(microns_02))**2

                val = fractionNPrimeAir*((dcAir-1) / (dcAir+2)) + (fractionNPrimeWater)*((dcWater-1) / (dcWater+2))+ (1-fractionNPrimeWater-fractionNPrimeAir)*((dcCyano-1) / (dcCyano+2))
                dcNPrime = (val*2 + 1)/(1-val)
                NPrimeMix = dcNPrime**0.5

                NPrimeMixn = (NPrimeMix.real)
                NPrimeMixk = (NPrimeMix.imag)
                return (NPrimeMixn, NPrimeMixk)


            output.append([facAir, facWater, faccyano, errorTotal, ratio, np.average(mixingformulaNPrime3Phase_02(facAir, facWater)[0]), np.average(mixingformulaNPrime3Phase_02(facAir, facWater)[1])])

            # print(output)

            if errorTotal < bestError:
                bestOpt = np.copy(opt)
                bestFacAir = facAir
                bestFacWater = facWater
                bestError = errorTotal # reset the criterion

# save the data
hndl = open(fr'C:\Users\aruna\Nextcloud\aNewBeginning\supporting\Aruna\nPrime optimizations\collective\output\faddeeva_global_{step}.txt', 'wt')

np.savetxt(hndl, output)
hndl.close()

ratioBest = round(R(bestOpt[3],bestOpt[5], bestOpt[1]),4)
facAir = bestFacAir
facWater = bestFacWater

###################################################################### Plotting

ax1.plot(spswnexpt, spsexpt,  marker='o', mec='k', mfc='w', ms=4, mew=1, lw=0, label='raw data')
ax1.plot(wnFine, chi2sps(wnFine, *bestOpt[4:8]), color='#FF5657', label=f'best fit, R = {ratioBest}')
ax1.set_title('SPS')

ax2.plot(sspwnexpt, sspexpt,  marker='o', mec='k', mfc='w', ms=4, mew=1, lw=0)
ax2.plot(wnFine, chi2ssp(wnFine, *bestOpt[2:4], *bestOpt[6:8]), color='#FF5657')
ax2.set_title('SSP')


ax3.plot(pppwnexpt, pppexpt,  marker='o', mec='k', mfc='w', ms=4, mew=1, lw=0)
ax3.plot(wnFine, chi2ppp(wnFine, *bestOpt[2:6], *bestOpt[0:2], *bestOpt[6:8]), color='#FF5657')
ax3.set_title('PPP')


ax1.set_ylabel('normalized SFG signal')
ax1.set_xlabel(r'IR wavenumber / cm$^{-1}$')
ax2.set_xlabel(r'IR wavenumber / cm$^{-1}$')
ax3.set_xlabel(r'IR wavenumber / cm$^{-1}$')
ax1.legend()

fig.set_tight_layout(True)
fig.suptitle(r'Best fit: $f_{\rm air} = %4.2f$, $f_{\rm water} = %4.2f$, $f_{\rm CP} = %4.2f$' % (bestFacAir, bestFacWater, 1-bestFacAir - bestFacWater))
pylab.savefig(fr'C:\Users\aruna\Nextcloud\aNewBeginning\supporting\Aruna\nPrime optimizations\collective\output\faddeeva_global_{step}.png', dpi=600)

print(opt)

# exporting

np.savetxt('SFGdata.txt' , np.array([wnFine, spsexpt, sspexpt, pppexpt, raw_data_sd('sps'), raw_data_sd('ssp'), raw_data_sd('ppp'), chi2sps(wnFine, *bestOpt[4:8]), chi2ssp(wnFine, *bestOpt[2:4], *bestOpt[6:8]), chi2ppp(wnFine, *bestOpt[2:6], *bestOpt[0:2], *bestOpt[6:8])]).T)

# newhandle = np.loadtxt('SFGdatanewhandle.txt', wnFine)
# newhandle02 = np.loadtxt('testdata.txt', wnFine, *bestOpt)
# alt + z