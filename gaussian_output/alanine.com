%chk=4cyano.chk
%mem=75000mb
%nproc=32
#p scf=tight opt  aug-cc-PVTZ B3LYP NoSymmetry

4cyanp Opt

0  1
C          -0.60217         2.30180         0.36354
N           0.52875         1.43156        -0.03388
H           0.51144         0.59784         0.55587
H           0.33406         1.06899        -0.97184
C          -0.39396         3.69072        -0.22478
H          -1.11563         4.39872         0.19836
H          -0.52756         3.69525        -1.31220
H           0.60719         4.07358         0.00226
C          -1.92858         1.68866        -0.10104
H          -0.61978         2.35776         1.45710
O          -2.08319         0.56164        -0.55075
O          -2.98767         2.50551         0.05662
H          -3.74474         1.98413        -0.28422

--Link1--
%chk=4cyano.chk
%mem=75000mb
%nproc=32
# CPHF=RdFreq B3LYP aug-cc-PVTZ Freq(ROA,printderivatives) Geom=Check Guess=Read polar(ROA,Raman) NoSymmetry int=superfine

4cyano

0 1

532nm
