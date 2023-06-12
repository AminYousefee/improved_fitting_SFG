from sympy import *
from time import time

t1 = time()
# introducing symbols
theta, phi, psi = symbols("theta, phi, psi")
mua, mub, muc = symbols("mua mub muc")
alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc = symbols("alphaaa alphaab alphaac alphaba alphabb alphabc alphaca alphacb alphacc")
# molecular frame derivatives.
dmudQ = Matrix([mua, mub, muc])
dalphadQ = Matrix([[alphaaa, alphaab, alphaac], [alphaba, alphabb, alphabc], [alphaca, alphacb, alphacc]])
# dalphadQ = Matrix([[alphaaa, alphaba, alphaca], [alphaba, alphabb, alphacb], [alphaca, alphacb, alphacc]])
# symbolic DCM
D = Matrix([[-sin(phi) * sin(psi) + cos(phi) * cos(psi) * cos(theta),
             -sin(phi) * cos(psi) - sin(psi) * cos(phi) * cos(theta), sin(theta) * cos(phi)],
            [sin(phi) * cos(psi) * cos(theta) + sin(psi) * cos(phi),
             -sin(phi) * sin(psi) * cos(theta) + cos(phi) * cos(psi), sin(phi) * sin(theta)],
            [-sin(theta) * cos(psi), sin(psi) * sin(theta), cos(theta)]])
# lab derivatives as a function of theta, phi, psi
lab_mu = D * dmudQ
lab_dalpha = D * dalphadQ * transpose(D)
print("lab dmu", lab_mu)
print("lab dalpha", lab_dalpha)

notations = "XYZ"
iso_norm = integrate(sin(theta), (theta, 0, pi), (phi, 0, 2 * pi), (psi, 0, 2 * pi))

# mu isotropic
print("for IR: dmu in lab")
print("isotropic (all uniform) which means we have triple integration")
lambdified_isotropic_dmu2_lab_list = [None, None, None]
for i in range(3):
    mu2_iso = factor(simplify(integrate(lab_mu[i] ** 2 * sin(theta), (theta, 0, pi), (phi, 0, 2 * pi), (psi, 0, 2 * pi)) / iso_norm))
    print("dmudQ", notations[i], mu2_iso)
    lambdified_isotropic_dmu2_lab = lambdify((mua, mub, muc), mu2_iso)
    lambdified_isotropic_dmu2_lab_list[i] = lambdified_isotropic_dmu2_lab
print("phi, psi uniform (theta unspecified)")
lambdified_theta_unspecified_dmu2_lab_list = [None, None, None]
for i in range(3):
    mu2_theta_unspecified = factor(simplify(integrate(lab_mu[i] ** 2, (phi, 0, 2 * pi), (psi, 0, 2 * pi))))
    print("dmudQ", notations[i], mu2_theta_unspecified)
    lambdified_theta_unspecified_dmu2_lab = lambdify((theta, mua, mub, muc), mu2_theta_unspecified)
    lambdified_theta_unspecified_dmu2_lab_list[i] = lambdified_theta_unspecified_dmu2_lab
print("phi is uniform (theta, psi unspecified")
lamdified_theta_psi_unspecified_dmu2_lab_list = [None, None, None]
for i in range(3):
    mu2_theta_psi_unspecified = factor(simplify(integrate(lab_mu[i] ** 2, (phi, 0, 2 * pi))))
    print("dmudQ", notations[i], mu2_theta_psi_unspecified)
    lambdified_theta_psi_unspecified_dmu2_lab = lambdify((theta, psi, mua, mub, muc), mu2_theta_psi_unspecified)
    lamdified_theta_psi_unspecified_dmu2_lab_list[i] = lambdified_theta_psi_unspecified_dmu2_lab

print("all angles are unspecified (general)")  # in here there is no integration. I just pass the formula
lambdified_general_dmu2_lab_list = [None] * 3
for i in range(3):
    mu2_general = simplify(lab_mu[i] ** 2)
    print("dmudQ", notations[i], mu2_general)
    lambdified_general_dmu2_lab = lambdify((theta, phi, psi, mua, mub, muc), mu2_general)
    lamdified_theta_psi_unspecified_dmu2_lab_list[i] = lambdified_general_dmu2_lab

# ----------------------------------- RAMAN --------------------------------------

# dalpha isotropic
print("for Raman")
print("iso tropic (all uniform)")
lambdified_isotropic_dalpha2_lab_matrix = [[None, None, None]] * 3
for i in range(3):
    for j in range(3):
        dalpha_iso = factor(simplify(integrate(lab_dalpha[i, j] ** 2 * sin(theta), (theta, 0, pi), (phi, 0, 2 * pi), (psi, 0, 2 * pi)) / iso_norm))
        print("dalphadQ", notations[i] + notations[j], dalpha_iso)
        lambdified_isotropic_dalpha_lab = lambdify((alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc), dalpha_iso)
        lambdified_isotropic_dalpha2_lab_matrix[i][j] = lambdified_isotropic_dalpha_lab

print("phi, psi uniform (theta unspecified)")
lambdified_theta_unspecified_dalpha2_lab_matrix = [[None, None, None]] * 3
for i in range(3):
    for j in range(3):
        dalpha_theta_unspecified = factor(simplify(integrate(lab_dalpha[i, j] ** 2, (phi, 0, 2 * pi), (psi, 0, 2 * pi))))
        print("dalphadQ", notations[i] + notations[j], dalpha_theta_unspecified)
        lambdified_theta_unspecified_dalpha2_lab = lambdify((theta, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc), dalpha_theta_unspecified)
        lambdified_theta_unspecified_dalpha2_lab_matrix[i][j] = lambdified_theta_unspecified_dalpha2_lab
print("phi uniform (theta, psi unspecified)")
lambdified_theta_psi_unspecified_dalpha2_lab_matrix = [[None, None, None]] * 3
for i in range(3):
    for j in range(3):
        dalpha_theta_psi_unspecified = factor(simplify(integrate(lab_dalpha[i, j] ** 2, (phi, 0, 2 * pi))))
        print("dalphadQ", notations[i] + notations[j], dalpha_theta_psi_unspecified)
        lambdified_theta_psi_unspecified_dalpha2_lab = lambdify((theta, psi, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc), dalpha_theta_psi_unspecified)
        lambdified_theta_psi_unspecified_dalpha2_lab_matrix[i][j] = lambdified_theta_psi_unspecified_dalpha2_lab

print("all angles are unspecified (general)") # in here there is no integration. I just pass the formula
lambdified_general_dalpha2_lab_matrix = [[None, None, None]]*3
for i in range(3):
    for j in range(3):
        dalpha2_general = simplify(lab_dalpha[i, j]**2)
        print("dalphadQ", notations[i]+notations[j], dalpha2_general)
        lambdified_general_dalpha2_lab = lambdify((theta, phi, psi, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc), dalpha2_general)
        lambdified_general_dalpha2_lab_matrix[i][j] = lambdified_general_dalpha2_lab

print("for SFG")
print("iso tropic (all uniform)")
lambdified_isotropic_hyperpolarizability_lab_tensor = [[[None for i in range(3)] for j in range(3)] for k in range(3)]
for i in range(3):
    for j in range(3):
        for k in range(3):
            hyperpolarizability_isotropic = expand(integrate(lab_dalpha[i, j] * lab_mu[k] * sin(theta), (theta, 0, pi), (phi, 0, 2 * pi), (psi, 0, 2 * pi)) / iso_norm)
            print(notations[i] + notations[j] + notations[k], hyperpolarizability_isotropic)
            lambdified_isotropic_hyperpolarizability_lab = lambdify((mua, mub, muc, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc), hyperpolarizability_isotropic)
            lambdified_isotropic_hyperpolarizability_lab_tensor[i][j][k] = lambdified_isotropic_hyperpolarizability_lab

print("phi, psi uniform (theta unspecified)")
lambdified_theta_unspecified_hyperpolarizability_lab_tensor = [[[None for i in range(3)] for j in range(3)] for k in range(3)]
for i in range(3):
    for j in range(3):
        for k in range(3):
            hyperpolarizability_theta_unspecified = factor(simplify(integrate(lab_dalpha[i, j] * lab_mu[k], (phi, 0, 2 * pi), (psi, 0, 2 * pi))))
            lambdified_theta_unspecified_hyperpolarizability_lab = lambdify((theta, mua, mub, muc, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc), hyperpolarizability_theta_unspecified)
            print(notations[i] + notations[j] + notations[k], hyperpolarizability_theta_unspecified)
            lambdified_theta_unspecified_hyperpolarizability_lab_tensor[i][j][k] = lambdified_theta_unspecified_hyperpolarizability_lab

print("phi uniform (theta, psi unspecified)")
lambdified_theta_psi_unspecified_hyperpolarizability_lab_tensor = [[[None for i in range(3)] for j in range(3)] for k in range(3)]
for i in range(3):
    for j in range(3):
        for k in range(3):
            hyperpolarizability_theta_psi_unspecified = factor(simplify(integrate(lab_dalpha[i, j] * lab_mu[k], (phi, 0, 2 * pi))))
            lambdified_theta_psi_unspecified_hyperpolarizability_lab = lambdify((theta, psi, mua, mub, muc, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc), hyperpolarizability_theta_psi_unspecified)
            print(notations[i] + notations[j] + notations[k], hyperpolarizability_theta_psi_unspecified)
            lambdified_theta_psi_unspecified_hyperpolarizability_lab_tensor[i][j][k] = lambdified_theta_psi_unspecified_hyperpolarizability_lab

print("all angles are unspecified (general)")
lambdified_general_hyperpolarizability_lab_tensor = [[[None for i in range(3)] for j in range(3)] for k in range(3)]
for i in range(3):
    for j in range(3):
        for k in range(3):
            hyperpolarizability_general = simplify(lab_dalpha[i, j] * lab_mu[k])
            lambdified_general_hyperpolarizability_lab = lambdify((theta, phi, psi, mua, mub, muc, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc), hyperpolarizability_general)
            print(notations[i] + notations[j] + notations[k], hyperpolarizability_general)
            lambdified_general_hyperpolarizability_lab_tensor[i][j][k] = lambdified_general_hyperpolarizability_lab

formula_obj = {"isotropic": [lambdified_isotropic_dmu2_lab_list, lambdified_isotropic_dalpha2_lab_matrix, lambdified_isotropic_hyperpolarizability_lab_tensor],
               "theta_unspecified": [lambdified_theta_unspecified_dmu2_lab_list, lambdified_theta_unspecified_dalpha2_lab_matrix, lambdified_theta_unspecified_hyperpolarizability_lab_tensor],
               "theta_psi_unspecified": [lamdified_theta_psi_unspecified_dmu2_lab_list, lambdified_theta_psi_unspecified_dalpha2_lab_matrix, lambdified_theta_psi_unspecified_hyperpolarizability_lab_tensor],
               "general":[lambdified_general_dmu2_lab_list, lambdified_general_dalpha2_lab_matrix, lambdified_general_hyperpolarizability_lab_tensor]}


import dill

dill.settings['recurse'] = True
with open("./results/sympy formula results/formula.dat", "wb") as formulafile:
    dill.dump(formula_obj, formulafile)

t2 = time()
print(t2 - t1)
