#!/usr/bin/env python
from sympy import *

# definitions
theta, phi, psi = symbols('theta phi psi')
aaa, aab, aac, aba, abb, abc, aca, acb, acc = symbols('aaa aab aac aba abb abc aca acb acc')
baa, bab, bac, bba, bbb, bbc, bca, bcb, bcc = symbols('baa bab bac bba bbb bbc bca bcb bcc')
caa, cab, cac, cba, cbb, cbc, cca, ccb, ccc = symbols('caa cab cac cba cbb cbc cca ccb ccc')
lab = ['x', 'y', 'z']

# direction cosine matrix
R1 = Matrix([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
R2 = Matrix([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
R3 = Matrix([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])
DCM = R3*R2*R1
print(DCM)

# molecular frame
# symmetry
alpha2 = [[[aaa, aab, aac], [aba, abb, abc], [aca, acb, acc]],
        [[aba, abb, abc], [bba, bbb, bbc], [bca, bcb, bcc]],
        [[aca, acb, acc], [bca, bcb, bcc], [cca, ccb, ccc]]]
# no symmetry
# alpha2 = [[[aaa, aab, aac], [aba, abb, abc], [aca, acb, acc]],
#         [[baa, bab, bac], [bba, bbb, bbc], [bca, bcb, bcc]],
#         [[caa, cab, cac], [cba, cbb, cbc], [cca, ccb, ccc]]]

# lab frame
chi2 = MutableDenseNDimArray(zeros(27), (3,3,3))
for i in [0, 1, 2]:
    for j in [0, 1, 2]:
        for k in [0, 1, 2]:
            temp = 0
            for l in [0, 1, 2]:
                for m in [0, 1, 2]:
                    for n in [0, 1, 2]:
                        temp += DCM[i,l]*DCM[j,m]*DCM[k,n]*alpha2[l][m][n]
            chi2[i,j,k] = temp

# all elements - isotropic
norm = 4*pi**2
for i in [0, 1, 2]:
    for j in [0, 1, 2]:
        for k in [0, 1, 2]:
            new = integrate(chi2[i,j,k], (phi, 0, 2*pi), (psi, 0, 2*pi))
            result = factor(simplify(new/norm))
            print('%s%s%s:  %s' % (lab[i], lab[j], lab[k], result))
