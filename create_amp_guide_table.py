from ProjectStructure import Molecule, ODF
import numpy as np
import os
from tabulate import tabulate
from copy import deepcopy


def amp_table_creator_delta(base_dir):
    alanine = Molecule("alanine", use_preset_molecular_frame_coordinates=True)
    gaussian_filepath = "./gaussian_output/{}.log".format(alanine.name)
    alanine.parse_gaussian_file(gaussian_filepath)
    mode_table_guide = [["modes"], ["frequency"]]
    for i in range(len(alanine.list_of_modes)):
        mode_table_guide[0].append("mode {}".format(i + 1))
        mode_table_guide[1].append(alanine.list_of_modes[i].frequency)
    # print(tabulate(np.array(mode_table_guide).T))
    # ---------------- DELTA, THETA ---------------
    theta0_list_in_rad = np.deg2rad(list(range(0, 185, 5)))
    odf_list = []
    for i in range(len(theta0_list_in_rad)):
        theta0 = theta0_list_in_rad[i]
        odf_parameters = {"type": "delta_dirac", "theta": {"theta0": theta0, "sigma": None}, "phi": None, "psi": None}
        odf = ODF(parameters=odf_parameters)
        odf_list.append(odf)

    alanine.draw_amp_guide_tables(base_dir, alanine.list_of_modes, odf_list)

    # ---------------- GAUSSIAN, THETA, SIGMA=0.1 ---------------


def amp_table_creator_gaussian(sigma, base_dir):
    alanine = Molecule("alanine", use_preset_molecular_frame_coordinates=True)
    gaussian_filepath = "./gaussian_output/{}.log".format(alanine.name)
    alanine.parse_gaussian_file(gaussian_filepath)
    mode_table_guide = [["modes"], ["frequency"]]
    for i in range(len(alanine.list_of_modes)):
        mode_table_guide[0].append("mode {}".format(i + 1))
        mode_table_guide[1].append(alanine.list_of_modes[i].frequency)
    # print(tabulate(np.array(mode_table_guide).T))

    theta0_list_in_rad = np.deg2rad(list(range(0, 185, 5)))
    odf_list_gaussian_sigma_small = []
    for i in range(len(theta0_list_in_rad)):
        theta0 = theta0_list_in_rad[i]
        odf_parameters = {"type": "gaussian", "theta": {"theta0": theta0, "sigma": sigma}, "phi": None, "psi": None}
        odf = ODF(parameters=odf_parameters)
        odf_list_gaussian_sigma_small.append(odf)
    alanine.draw_amp_guide_tables(base_dir, alanine.list_of_modes, odf_list_gaussian_sigma_small)



def amp_cube_creator_gaussian_theta_sigma():
    alanine = Molecule("alanine", use_preset_molecular_frame_coordinates=True)
    gaussian_filepath = "./gaussian_output/{}.log".format(alanine.name)
    alanine.parse_gaussian_file(gaussian_filepath)
    mode_table_guide = [["modes"], ["frequency"]]
    for i in range(len(alanine.list_of_modes)):
        mode_table_guide[0].append("mode {}".format(i + 1))
        mode_table_guide[1].append(alanine.list_of_modes[i].frequency)
    # print(tabulate(np.array(mode_table_guide).T))
    theta0_list_in_rad = np.deg2rad(list(range(0, 185, 5)))
    sigma_list = np.arange(1, 90, 1)
    cube = np.zeros(())
    odf_list = []
    for theta_ind, theta0_rad in enumerate(theta0_list_in_rad):
        for sigma_ind, sigma in enumerate(sigma_list):
            odf_parameters = {"type": "gaussian", "theta": {"theta0": theta0_rad, "sigma": sigma}, "phi": None, "psi": None}
            odf = ODF(parameters=odf_parameters)
            odf_list.append()


if __name__ == "__main__":
    base_path_delta = "./results/alanine/amp_tables/delta/"
    base_path_gaussian_tilt = "./results/alanine/amp_tables/gaussian/tilt/"
    if not os.path.isdir(base_path_delta):
        os.makedirs(base_path_delta)
    if not os.path.isdir(base_path_gaussian_tilt):
        os.makedirs(base_path_gaussian_tilt)
    amp_table_creator_delta(base_path_delta)
    sigma_list = np.array([np.deg2rad(sigma_deg) for sigma_deg in range(1, 90, 5)])
    for sigma in sigma_list:
        amp_table_creator_gaussian(sigma, base_path_gaussian_tilt)
        print("a table is created in gaussian mode")
