#!/usr/bin/env python
from sympy import *

# define symbolic variables
theta, phi, psi = symbols('theta phi psi')
aa, ab, ac, bb, bc, cc = symbols('aa ab ac bb bc cc')
ba, ca, cb = symbols('ba ca cb')
lab = ['x', 'y', 'z']

# Raman tensor
molFrame = Matrix([[aa, ab, ac], [ba, bb, bc], [ca, cb, cc]])
#molFrame = Matrix([[aa, ab, ac], [ab, bb, bc], [ac, bc, cc]])
#molFrame = Matrix([[aa, 0, 0], [0, bb, 0], [0, 0, cc]])
#molFrame = Matrix([[aa, 0, 0], [0, aa, 0], [0, 0, cc]])
#molFrame = Matrix([[0,0,0], [0,0,0], [0,0,cc]])

# direction cosine matrix
DCM = Matrix([[cos(phi)*cos(theta)*cos(psi) - sin(phi)*sin(psi), -cos(phi)*cos(theta)*sin(psi) - sin(phi)*cos(psi), sin(theta)*cos(phi)],
       [sin(phi)*cos(theta)*cos(psi) + cos(phi)*sin(psi), -sin(phi)*cos(theta)*sin(psi) + cos(phi)*cos(psi), sin(theta)*sin(phi)],
        [-cos(psi)*sin(theta), sin(psi)*sin(theta), cos(theta)]])
labFrame = DCM * molFrame * transpose(DCM)

# isotropic
norm = 8*pi**2
for i in [0, 1, 2]:
    for j in [0, 1, 2]:
        intensity = integrate(labFrame[i, j] * sin(theta), (theta, 0, pi), (phi, 0, 2*pi), (psi, 0, 2*pi))
        print('%s%s   %s' % (lab[i], lab[j], simplify(intensity/norm)))
