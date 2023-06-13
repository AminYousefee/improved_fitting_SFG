import itertools
import math
import os.path
import random
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import pylab
from numpy import loadtxt, exp
from scipy.optimize import fmin_tnc
import dill
import numpy as np
import re
import pickle
from math import sqrt, acos, atan
from scipy.integrate import quad, dblquad, tplquad
from tabulate import tabulate
from scipy.signal import find_peaks_cwt, find_peaks, medfilt, savgol_filter


class Molecule:
    def __init__(self, name, use_preset_molecular_frame_coordinates: bool):
        self.name = name
        self.list_of_atoms = []
        self.number_of_atoms = 0
        self.list_of_modes = []
        self.number_of_normal_modes = 0
        self.a = None
        self.b = None
        self.c = None
        self.direction_cosine_matrix = None
        if use_preset_molecular_frame_coordinates:
            self.set_molecular_frame_coordinates_a_b_c([1, 0, 0], [0, 1, 0], [0, 0, 1])
        elif (self.a is not None) or (self.b is not None) or (self.c is not None):
            self.ask_operator_to_enter_a_b_c()

    @property
    def theta(self):
        return acos(self.direction_cosine_matrix[2, 2])

    @property
    def phi(self):
        return atan(self.direction_cosine_matrix[1, 2] / self.direction_cosine_matrix[0, 2])

    @property
    def psi(self):
        return atan(-1.0 * self.direction_cosine_matrix[2, 1] / self.direction_cosine_matrix[2, 0])

    def print_parsed_data(self):
        for index, m in enumerate(self.list_of_modes):
            print(
                "mode {}, frequency = {} cm-1, reduced mass = {}, dipole_moment_derivative_vector = {}, polarizability_matrix = {}".format(
                    index + 1, m.frequency, m.red_mass, m.dipole_moment_derivatives, m.polarizability_derivatives))
        print("--" * 40)
        atoms_table = []
        atoms_header = ["Index", "Atomic Number", "X", "Y", "Z"]
        for atom in self.list_of_atoms:
            atoms_table.append([atom.index_in_molecule, atom.atomic_number, atom.coordinates[0], atom.coordinates[1], atom.coordinates[2]])
        print(tabulate(atoms_table, atoms_header))

    def parse_gaussian_file(self, filepath):
        molecule_parser = Parser(filepath, molecule=self)
        molecule_parser.determine_atoms()
        molecule_parser.frequency_parse()
        molecule_parser.reduced_mass_parse()
        molecule_parser.dipole_moment_derivatives_parse()
        molecule_parser.polarizability_derivatives_parse()
        molecule_parser.atom_coordinates_parser()
        self.calculate_hyperpolarizability_molecular_frame()

    def calculate_hyperpolarizability_molecular_frame(self):
        for mode in self.list_of_modes:
            mode.calculate_hyperpolarizability_for_each_mode_molecular()

    def dump_molecule(self):
        with open("{}.molecule".format(self.name), "wb") as wfile:
            pickle.dump(self, wfile)
            print("{} file saved successfully".format(self.name))

    @staticmethod
    def pickle_load_molecule(molecule_name):
        with open("{}.molecule".format(molecule_name), "rb") as rfile:
            molecule = pickle.load(rfile)
            molecule.ask_operator_to_enter_a_b_c()
            return molecule

    @staticmethod
    def calculate_vector_length(vector: list):
        return sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)

    def ask_operator_to_enter_a_b_c(self):
        using_preset_a_b_c = input(
            "do you like to use preset molecular frame:\na=[1,0,0]\nb=[0,1,0]\nc=[0,0,1]\n[y/n]?").lower()
        if using_preset_a_b_c in ["y", 'yes', 'true']:
            self.set_molecular_frame_coordinates_a_b_c([1, 0, 0], [0, 1, 0], [0, 0, 1])
        else:
            a = list(map(float, input("enter 'a' vector using space to enter elements").split()))
            b = list(map(float, input("enter 'b' vector using space to enter elements").split()))
            c = list(map(float, input("enter 'c' vector using space to enter elements").split()))
            self.set_molecular_frame_coordinates_a_b_c(a, b, c)

    def construct_the_molecule(self):
        # atoms_must be inserted
        self.number_of_atoms = len(self.list_of_atoms)
        if self.number_of_atoms == 0:
            raise Exception("to use this method, atoms must be inserted.")
        # calculate the number of normal modes
        self.number_of_normal_modes = 3 * self.number_of_atoms - 6
        for i in range(self.number_of_normal_modes):
            self.list_of_modes.append(Mode(self))

    def set_molecular_frame_coordinates_a_b_c(self, a: list, b: list, c: list):
        d = [a, b, c]
        for i in range(3):
            vlength = Molecule.calculate_vector_length(d[i])
            if 1.001 < vlength or 0.999 > vlength:
                d[i] = [j / vlength for j in d[i]]
        self.a = np.array(d[0])
        self.b = np.array(d[1])
        self.c = np.array(d[2])
        self.direction_cosine_matrix = np.array([self.a, self.b, self.c]).transpose()
        # convert all properties to the new version

    def transfer_lab_to_molecular_frame(self, vector):
        return np.dot(self.direction_cosine_matrix, vector)

    def transfer_molecular_to_lab_frame(self, vector):
        return np.dot(self.direction_cosine_matrix.transpose(), vector)

    def create_amp_guide_cube(self, sigma_list, theta0_list, mode_list):
        xxz_cube = np.zeros((len(sigma_list), len(theta0_list), len(mode_list)))
        xzx_cube = np.zeros((len(sigma_list), len(theta0_list), len(mode_list)))
        zzz_cube = np.zeros((len(sigma_list), len(theta0_list), len(mode_list)))
        for sigma_ind, sigma in enumerate(sigma_list):
            for theta0_ind, theta0 in enumerate(theta0_list):
                odf_parameters = {"type": "gaussian", "theta": {"theta0": theta0, "sigma": sigma}, "phi": None, "psi": None}
                odf = ODF(parameters=odf_parameters)
                for mode_ind, mode in enumerate(mode_list):
                    mode.calculate_amplitude(odf)
                    xxz_cube[sigma_ind, theta0_ind, mode_ind] = mode.amplitudes[0, 0, 2]
                    xzx_cube[sigma_ind, theta0_ind, mode_ind] = mode.amplitudes[0, 2, 0]
                    zzz_cube[sigma_ind, theta0_ind, mode_ind] = mode.amplitudes[2, 2, 2]
        desired_max_amp = 100.0
        all_amps = np.array([xxz_cube, xzx_cube, zzz_cube])
        scaled_amp = desired_max_amp / np.max(np.abs(all_amps))
        scaled_all_amps = all_amps * scaled_amp
        scaled_all_amps = scaled_all_amps.round(decimals=2)
        return scaled_all_amps

    def create_amp_guide_table(self, list_of_modes, odf_list, table_column_number, table_row_number):
        # this implementation is just for the delta dirac and one eulier angle (tilt andgle)

        xxz_table = np.zeros((table_row_number, table_column_number))
        xzx_table = np.zeros((table_row_number, table_column_number))
        zzz_table = np.zeros((table_row_number, table_column_number))
        for odf_ind, odf in enumerate(odf_list):
            # print("----------------------------------------------------")
            for mode_ind, mode in enumerate(list_of_modes):
                mode.calculate_amplitude(odf)
                xxz_table[odf_ind, mode_ind] = mode.amplitudes[0, 0, 2]
                xzx_table[odf_ind, mode_ind] = mode.amplitudes[0, 2, 0]
                zzz_table[odf_ind, mode_ind] = mode.amplitudes[2, 2, 2]
            # print(list_of_modes[0].amplitudes[0, 0, 2])
        desired_max_amp = 100.0
        all_amps = np.array([xxz_table, xzx_table, zzz_table])
        scaled_amp = desired_max_amp / np.max(np.abs(all_amps))
        scaled_all_amps = all_amps * scaled_amp
        scaled_all_amps = scaled_all_amps.round(decimals=2)
        return scaled_all_amps

    def draw_amp_guide_cube(self, filepath1, filepath2, filepath3, list_of_theta0, list_of_sigma, list_of_modes):
        ready_amp_cubes = self.create_amp_guide_cube(list_of_sigma, list_of_theta0, list_of_modes)
        theta0_list_in_degree = [round(np.rad2deg(theta0)) for theta0 in list_of_theta0]
        str_modes_in_range = [str(i + 1) for i in range(len(list_of_modes))]
        str_sigma = [str(sigma) for sigma in list_of_sigma]
        xxz_cube = ready_amp_cubes[0]
        xzx_cube = ready_amp_cubes[1]
        zzz_cube = ready_amp_cubes[2]

    def draw_amp_guide_tables(self, base_dir, list_of_modes, odf_list):
        table_row_number = len(odf_list)
        table_column_number = len(list_of_modes)  # number of modes
        theta0_list_in_degree = [round(np.rad2deg(odf.theta["theta0"])) for odf in odf_list]
        ready_amp_tables = self.create_amp_guide_table(list_of_modes, odf_list, table_column_number, table_row_number)

        xxz_table = ready_amp_tables[0]
        xzx_table = ready_amp_tables[1]
        zzz_table = ready_amp_tables[2]
        str_modes_in_range = [str(i + 1) for i in range(len(list_of_modes))]

        # print("XXZ TABLE".center(70, "-"))
        # print(tabulate(xxz_table))
        # print("XZX TABLE".center(70, "-"))
        # print(tabulate(xzx_table))
        # print("ZZZ TABLE".center(70, "-"))
        # print(tabulate(zzz_table))

        fig1 = plt.figure(figsize=(table_column_number//3 + 1, table_row_number//3))
        fig2 = plt.figure(figsize=(table_column_number//3 + 1, table_row_number//3))
        fig3 = plt.figure(figsize=(table_column_number//3 + 1, table_row_number//3))

        ax1 = fig1.add_subplot(1, 1, 1)
        ax2 = fig2.add_subplot(1, 1, 1)
        ax3 = fig3.add_subplot(1, 1, 1)

        fig1_filepath = base_dir + "XXZ.png"
        fig2_filepath = base_dir + "XZX.png"
        fig3_filepath = base_dir + "ZZZ.png"

        cmap = 'bwr'
        im1 = ax1.imshow(xxz_table, cmap=cmap, vmin=-100, vmax=100)
        cbar1 = fig1.colorbar(im1, ticks=[-100, 0, 100], ax=ax1, shrink=0.5)

        im2 = ax2.imshow(xzx_table, cmap=cmap, vmin=-100, vmax=100)
        cbar2 = fig1.colorbar(im2, ticks=[-100, 0, 100], ax=ax2, shrink=0.5)

        im3 = ax3.imshow(zzz_table, cmap=cmap, vmin=-100, vmax=100)
        cbar3 = fig1.colorbar(im3, ticks=[-100, 0, 100], ax=ax3, shrink=0.5)

        ax1.set_xticks(np.arange(table_column_number), labels=str_modes_in_range)
        ax1.set_yticks(np.arange(table_row_number), labels=theta0_list_in_degree)

        ax2.set_xticks(np.arange(table_column_number), labels=str_modes_in_range)
        ax2.set_yticks(np.arange(table_row_number), labels=theta0_list_in_degree)

        ax3.set_xticks(np.arange(table_column_number), labels=str_modes_in_range)
        ax3.set_yticks(np.arange(table_row_number), labels=theta0_list_in_degree)

        for i in range(table_row_number):
            for j in range(table_column_number):
                ax1.text(j, i, xxz_table[i, j], ha="center", va="center", fontsize=6)
                ax2.text(j, i, xzx_table[i, j], ha="center", va="center", fontsize=6)
                ax3.text(j, i, zzz_table[i, j], ha="center", va="center", fontsize=6)
        ax1.set_xlabel("Vibrational Modes  /  XXZ")
        ax1.set_ylabel("Tilt Angle \u03B8" + "\u00B0")
        ax2.set_xlabel("Vibrational Modes  /  XZX")
        ax2.set_ylabel("Tilt Angle \u03B8" + "\u00B0")
        ax3.set_xlabel("Vibrational Modes  /  ZZZ")
        ax3.set_ylabel("Tilt Angle \u03B8" + "\u00B0")


        # cbar.ax.set_yticklabels(['-100', '0', '+100'])
        fig1.tight_layout()
        fig2.tight_layout()
        fig3.tight_layout()
        dpi = 400
        fig1.savefig(fig1_filepath, dpi=dpi)
        fig2.savefig(fig2_filepath, dpi=dpi)
        fig3.savefig(fig3_filepath, dpi=dpi)
        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)
        return ready_amp_tables


class Atom:
    def __init__(self, molecule, atomic_number):
        self.atomic_number = atomic_number
        # self.atomic_symbol = symbol
        self.molecule = molecule
        self.index_in_molecule = 0
        self.coordinates = None


class Lab:
    n1 = 1
    n2 = 1.5

    class BeamLab:
        def __init__(self, theta1_degree, n1, n2, n_prime):
            self.theta1 = math.radians(theta1_degree)
            self.theta2 = math.asin(n1 / n2 * math.sin(self.theta1))
            self.rs = (n1 * math.cos(self.theta1) - n2 * math.cos(self.theta2)) / (
                    n1 * math.cos(self.theta1) + n2 * math.cos(self.theta2))
            self.rp = (n2 * math.cos(self.theta1) - n1 * math.cos(self.theta2)) / (
                    n2 * math.cos(self.theta1) + n1 * math.cos(self.theta2))
            self.Lxx = (1 - self.rp) * math.cos(self.theta1)
            self.Lyy = 1 + self.rs
            self.Lzz = (1 + self.rp) * ((n1 / n_prime) ** 2) * math.sin(self.theta1)

    def __init__(self, n1, n2, theta1_degree_sfg, theta1_degree_IR, theta1_degree_vis):
        self.n1 = n1
        self.n2 = n2
        self.n_prime = (self.n1 + self.n2) / 2.0

        self.sfg = Lab.BeamLab(theta1_degree_sfg, self.n1, self.n2, self.n_prime)
        self.vis = Lab.BeamLab(theta1_degree_vis, self.n1, self.n2, self.n_prime)
        self.ir = Lab.BeamLab(theta1_degree_IR, self.n1, self.n2, self.n_prime)


class ODF():
    def __init__(self, parameters):
        self.odf_type = parameters["type"]  # "delta_dirac" or "gaussian"
        self.parameters = parameters
        self.theta = parameters["theta"]
        self.phi = parameters["phi"]
        self.psi = parameters["psi"]
        angle_check = [self.theta, self.psi, self.phi]
        self.number_of_angles = 0
        for angle in angle_check:
            if angle is not None:
                self.number_of_angles += 1

    def odf_gaussian_theta(self, theta):
        return np.exp(-1 * (theta - self.theta["theta0"])**2 / (2 * self.theta["sigma"]))

    def odf_gaussian_theta_psi(self, theta, psi):
        return np.exp((-1 * (theta - self.theta["theta0"])**2 / (2 * self.theta["sigma"])) - ((psi - self.psi["psi0"]) / (2 * self.psi["sigma"])))

    def odf_gaussian_general(self, theta, phi, psi):
        return np.exp((-1 * (theta - self.theta["theta0"])**2 / (2 * self.theta["sigma"])) - ((psi - self.psi["psi0"]) / (2 * self.psi["sigma"])) - ((phi - self.phi["phi0"]) / (2 * self.phi["sigma"])))


#


class Spectrum:
    def __init__(self, molecule: Molecule, lab: Lab, odf: ODF, freq_range_start, freq_range_end, freq_range_step,
                 list_of_nr_chi2):
        self.freq_range_start = freq_range_start
        self.freq_range_end = freq_range_end
        self.freq_range_step = freq_range_step
        self.light_freq_range = np.arange(freq_range_start, freq_range_end, freq_range_step)
        self.molecule = molecule
        self.chi2_non_res = list_of_nr_chi2  # it contains 3 numbers for the 2nd, 6th, and 26 and they are all equal
        self.lab = lab
        self.odf = odf
        self.ir_intensity = None
        self.raman_intensity = None
        self.X2_yyz = None
        self.X2_yzy = None
        self.X2_zzz = None
        self.sps = None
        self.ssp = None
        self.ppp = None
        self.list_of_modes_in_range = self.find_modes_in_range()

    def add_mode(self, ampxxz, ampxzx, ampzzz, gamma, omega0):
        mode = Mode(molecule=self.molecule)
        mode.gamma = gamma
        mode.frequency = omega0
        mode.amplitudes = np.zeros((3, 3, 3)).tolist()
        mode.amplitudes[0, 0, 2] = ampxxz
        mode.amplitudes[0, 2, 0] = ampxzx
        mode.amplitudes[2, 2, 2] = ampzzz
        #     adding the mode in the proper position of molecule mode's list
        for i in range(len(self.molecule.list_of_modes)):
            if self.molecule.list_of_modes[i].frequency > omega0:
                self.molecule.list_of_modes.insert(i, mode)
                break
        #     adding the mode in the proper position of the spectrum
        self.list_of_modes_in_range = self.find_modes_in_range()

    def show_ir_intensity_arrays(self):
        print("IR Intensity ".center(100, "-"))
        new_freq_list = ["Frequency "] + self.light_freq_range.tolist()
        new_ir_x = ["IR Intensity X"] + self.ir_intensity[0].tolist()
        new_ir_y = ["IR Intensity Y"] + self.ir_intensity[1].tolist()
        new_ir_z = ["IR Intensity Z"] + self.ir_intensity[2].tolist()
        table = np.array([new_freq_list, new_ir_x, new_ir_y, new_ir_z]).T
        print(tabulate(table))

    def show_raman_intensity_arrays(self):
        print("Raman Intensity".center(100, "-"))
        notation = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]
        table = []
        table.append(["Frequency "] + self.light_freq_range.tolist())
        for i in range(9):
            table.append(["Raman Intensity " + notation[i]] + self.raman_intensity[i].tolist())
        table = np.array(table).T
        print(tabulate(table))

    def show_sfg_intensity_arrays(self):
        print("SFG Chi2".center(100, "-"))
        array_list = np.array([self.X2_yyz, self.X2_yzy, self.X2_zzz, self.ssp, self.sps, self.ppp])
        array_list_names = ["X2_yyz", "X2_yzy", "X2_zzz", "X2_ssp", "X2_sps", "X2_ppp"]
        table = []
        table.append(["Frequency "] + self.light_freq_range.tolist())
        for i in range(len(array_list)):
            table.append(["SFG " + array_list_names[i]] + array_list[i].tolist())
        table = np.array(table).T
        print(tabulate(table))

    def find_modes_in_range(self):
        return [mode for mode in self.molecule.list_of_modes if self.freq_range_start < mode.frequency < self.freq_range_end]

    def show_real_mode_lorentzian_info(self):
        # the main application of this function is in the fitting
        print("Real modes' information in the frequency range {} - {}".format(self.freq_range_start, self.freq_range_end).center(100, "-"))
        print("Real NR chi2 xxz", self.chi2_non_res[0])
        print("Real NR chi2 xzx", self.chi2_non_res[1])
        print("Real NR chi2 zzz", self.chi2_non_res[2])
        table = []
        headers = ["mode", "wavenumber cm-1", "width cm-1", "amp_xxz", "amp_xzx", "amp_zzz"]
        for i, mode in enumerate(self.find_modes_in_range()):
            table.append([i, mode.frequency, mode.gamma, mode.amplitudes[0, 0, 2], mode.amplitudes[0, 2, 0], mode.amplitudes[2, 2, 2]])
        print(tabulate(table, headers))
        print("-" * 100)

    def show_all_spectra_at_certain_orientaiton(self, show_normalized, filename, noise):
        pylab.rc('font', size=8)
        fig = pylab.figure(figsize=(11.7, 16.5))

        frequencies = self.light_freq_range
        xxz_mag_squared = abs(self.X2_yyz) ** 2
        xxz_im = np.imag(self.X2_yyz)
        ssp_mag_squared = abs(self.ssp) ** 2
        ssp_im = np.imag(self.ssp)
        xzx_mag_squared = abs(self.X2_yzy) ** 2
        xzx_im = np.imag(self.X2_yzy)
        sps_mag_squared = abs(self.sps) ** 2
        sps_im = np.imag(self.sps)
        zzz_mag_squared = abs(self.X2_zzz) ** 2
        zzz_im = np.imag(self.X2_zzz)
        ppp_mag_squared = abs(self.ppp) ** 2
        ppp_im = np.imag(self.ppp)
        all_y_axis = [xxz_mag_squared, xxz_im, ssp_mag_squared, ssp_im, xzx_mag_squared, xzx_im, sps_mag_squared, sps_im, zzz_mag_squared, zzz_im, ppp_mag_squared, ppp_im]
        if noise:
            for i in range(len(all_y_axis)):
                all_y_axis[i] = Calculation.add_noise_to_array(all_y_axis[i], noise)
        modes_in_range = self.find_modes_in_range()

        # ------------------  IR  ----------------------------
        # in here x and z are just unique
        # x = y
        ax1 = fig.add_subplot(7, 2, 1)  # IR
        ir_intensity = self.ir_intensity / np.max(self.ir_intensity) if show_normalized else self.ir_intensity
        for spectrum in ir_intensity:
            ax1.plot(frequencies, spectrum)
        ax1.set_ylabel("IR Intensity  /  $A.U$")
        ax1.legend(["X", "Y", "Z"], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
        # ------------------  RAMAN  ----------------------------
        # in here xx and zz are just unique
        # xx = xy = yx = yy = zx = zy
        # zz = xz = yz
        """
        1 1 2
        1 1 2
        1 1 2
        """
        ax2 = fig.add_subplot(7, 2, 2)  # Raman
        raman_intensity = self.raman_intensity / np.max(
            self.raman_intensity) if show_normalized else self.raman_intensity
        desired_raman_arrays = [raman_intensity[0], raman_intensity[-1]]
        for spectrum in desired_raman_arrays:
            ax2.plot(frequencies, spectrum)
        ax2.legend(["XX", "ZZ"], loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)
        ax2.set_ylabel("Raman Intensity  /  $A.U$")
        normalization_factor_mag_squared_polarizations = np.max([ssp_mag_squared, sps_mag_squared, ppp_mag_squared])
        normalization_factor_mag_squared = np.max([xxz_mag_squared, xzx_mag_squared, zzz_mag_squared])
        normalization_factor_im_polarizations = np.max([ssp_im, sps_im, ppp_im])
        normalization_factor_im = np.max([xzx_im, xxz_im, zzz_im])
        # ------------------  XXZ MAG SQUARED  ----------------------------
        ax3 = fig.add_subplot(7, 2, 3)  # xxz mag_squared
        all_y_axis[0] = all_y_axis[0] / normalization_factor_mag_squared if show_normalized else all_y_axis[0]
        ax3.plot(frequencies, all_y_axis[0])
        ax3.set_ylabel("$|\chi^{(2)}|^2_{xxz}$  /  $A.U$")

        ax4 = fig.add_subplot(7, 2, 4)  # xxz im
        all_y_axis[1] = all_y_axis[1] / normalization_factor_im if show_normalized else all_y_axis[1]
        ax4.plot(frequencies, all_y_axis[1])
        ax4.set_ylabel("$Imag[\chi^{(2)}]_{xxz}$  /  $A.U$")

        ax5 = fig.add_subplot(7, 2, 5)  # ssp mag_squared
        all_y_axis[2] = all_y_axis[2] / normalization_factor_mag_squared_polarizations if show_normalized else all_y_axis[2]
        ax5.plot(frequencies, all_y_axis[2])
        ax5.set_ylabel("$|\chi^{(2)}|^2_{SSP}$  /  $A.U$")

        ax6 = fig.add_subplot(7, 2, 6)  # ssp im
        all_y_axis[3] = all_y_axis[3] / normalization_factor_im_polarizations if show_normalized else all_y_axis[3]
        ax6.plot(frequencies, all_y_axis[3])
        ax6.set_ylabel("$Imag[\chi^{(2)}]_{SSP}$  /  $A.U$")

        ax7 = fig.add_subplot(7, 2, 7)  # xzx mag_squared
        all_y_axis[4] = all_y_axis[4] / normalization_factor_mag_squared if show_normalized else all_y_axis[4]
        ax7.plot(frequencies, all_y_axis[4])
        ax7.set_ylabel("$|\chi^{(2)}|^2_{xzx}$  /  $A.U$")

        ax8 = fig.add_subplot(7, 2, 8)  # xzx im
        all_y_axis[5] = all_y_axis[5] / normalization_factor_im if show_normalized else all_y_axis[5]
        ax8.plot(frequencies, all_y_axis[5])
        ax8.set_ylabel("$Imag[\chi^{(2)}]_{xzx}$  /  $A.U$")

        ax9 = fig.add_subplot(7, 2, 9)  # sps mag_squared
        all_y_axis[6] = all_y_axis[6] / normalization_factor_mag_squared_polarizations if show_normalized else all_y_axis[6]
        ax9.plot(frequencies, all_y_axis[6])
        ax9.set_ylabel("$|\chi^{(2)}|^2_{SPS}$  /  $A.U$")

        ax10 = fig.add_subplot(7, 2, 10)  # sps im
        all_y_axis[7] = all_y_axis[7] / normalization_factor_im_polarizations if show_normalized else all_y_axis[7]
        ax10.plot(frequencies, all_y_axis[7])
        ax10.set_ylabel("$Imag[\chi^{(2)}]_{SPS}$  /  $A.U$")

        ax11 = fig.add_subplot(7, 2, 11)  # zzz mag_squared
        all_y_axis[8] = all_y_axis[8] / normalization_factor_mag_squared if show_normalized else all_y_axis[8]
        ax11.plot(frequencies, all_y_axis[8])
        ax11.set_ylabel("$|\chi^{(2)}|^2_{zzz}$  /  $A.U$")

        ax12 = fig.add_subplot(7, 2, 12)  # zzz_im
        all_y_axis[9] = all_y_axis[9] / normalization_factor_im if show_normalized else all_y_axis[9]
        ax12.plot(frequencies, all_y_axis[9])
        ax12.set_ylabel("$Imag[\chi^{(2)}]_{zzz}$  /  $A.U$")

        ax13 = fig.add_subplot(7, 2, 13)  # ppp mag_squared
        all_y_axis[10] = all_y_axis[10] / normalization_factor_mag_squared_polarizations if show_normalized else all_y_axis[10]
        ax13.plot(frequencies, all_y_axis[10])
        ax13.set_ylabel("$|\chi^{(2)}|^2_{PPP}$  /  $A.U$")

        ax14 = fig.add_subplot(7, 2, 14)  # ppp im
        all_y_axis[11] = all_y_axis[11] / normalization_factor_im_polarizations if show_normalized else all_y_axis[11]
        ax14.plot(frequencies, all_y_axis[11])
        ax14.set_ylabel("$Imag[\chi^{(2)}]_{PPP}$  /  $A.U$")

        ax_list = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14]
        for ax in ax_list:
            ax.set_xlabel("IR wavenumber  /  cm$^{-1}$")
            for mode in modes_in_range:
                ax.axvline(mode.frequency, color="black", ls='dotted')

        fig.set_tight_layout(True)
        filepath = Calculation.give_filepath(self.molecule.name, show_normalized, bool(noise), filename)
        print("all-chart file is saved in {}".format(filepath))
        pylab.savefig(filepath, dpi=800)
        plt.close('all')
        print("one all chart saved")

    def calculate_sps_chi_2(self):
        # the 6th element of chi2 is used here
        sps_each_freq = lambda yzy_element: self.lab.sfg.Lyy * self.lab.vis.Lzz * self.lab.ir.Lyy * yzy_element
        self.sps = sps_each_freq(self.X2_yzy)

    def calculate_ssp_chi_2(self):
        # the 2nd element of chi2 is used here
        ssp_each_freq = lambda yyz_element: self.lab.sfg.Lyy * self.lab.vis.Lyy * self.lab.ir.Lzz * yyz_element
        self.ssp = ssp_each_freq(self.X2_yyz)

    def calculate_ppp_chi_2(self):
        zxx_each_freq = lambda \
                yzy_element: self.lab.sfg.Lzz * self.lab.vis.Lxx * self.lab.ir.Lxx * yzy_element  # 6th element
        xzx_each_freq = lambda \
                yzy_element: self.lab.sfg.Lxx * self.lab.vis.Lzz * self.lab.ir.Lxx * yzy_element  # 6th element
        xxz_each_freq = lambda \
                yyz_element: self.lab.sfg.Lxx * self.lab.vis.Lxx * self.lab.ir.Lzz * yyz_element  # 2nd element
        zzz_each_freq = lambda \
                zzz_element: self.lab.sfg.Lzz * self.lab.vis.Lzz * self.lab.ir.Lzz * zzz_element  # 26th element
        self.ppp = zxx_each_freq(self.X2_yzy) + xzx_each_freq(self.X2_yzy) + xxz_each_freq(self.X2_yyz) + zzz_each_freq(
            self.X2_zzz)

    def calculate_infrared_absorption(self):
        x_intensity_list = []
        y_intensity_list = []
        z_intensity_list = []

        with open("results/sympy formula results/formula.dat", "rb") as formulafile:
            formula = dill.load(formulafile)
            if self.odf.number_of_angles == 1:
                lambdified_dmu2_lab_list = formula["theta_unspecified"][0]
            elif self.odf.number_of_angles == 2:
                lambdified_dmu2_lab_list = formula["theta_psi_unspecified"][0]
            elif self.odf.number_of_angles == 3:  # there is no integration in sympy, but we are importing it.
                lambdified_dmu2_lab_list = formula["general"][0]
            else:
                raise Exception("number of angles is not meaningful")

        for mode in self.list_of_modes_in_range:
            mode.calculate_dipole_moment_squared_derivatives_vector_in_lab_frame(self.odf, lambdified_dmu2_lab_list)
        for w_ir in self.light_freq_range:
            summation = [0, 0, 0]
            for mode in self.list_of_modes_in_range:
                intensity_results_for_xyz_in_each_mode = mode.calculate_IR_intensity_for_each_mode(w_ir)
                summation[0] += intensity_results_for_xyz_in_each_mode[0]
                summation[1] += intensity_results_for_xyz_in_each_mode[1]
                summation[2] += intensity_results_for_xyz_in_each_mode[2]
            x_intensity_list.append(summation[0])
            y_intensity_list.append(summation[1])
            z_intensity_list.append(summation[2])
        intensity_list = [x_intensity_list, y_intensity_list, z_intensity_list]
        self.ir_intensity = np.array(intensity_list)
        print("IR intensity calculation for range {} - {} is done".format(self.freq_range_start, self.freq_range_end))

    def calculate_raman_intensity(self):
        intensities = [[] for i in range(9)]
        with open("results/sympy formula results/formula.dat", "rb") as formulafile:
            formula = dill.load(formulafile)
            if self.odf.number_of_angles == 1:
                lambdified_dalpha2_lab_matrix = formula["theta_unspecified"][1]
            elif self.odf.number_of_angles == 2:
                lambdified_dalpha2_lab_matrix = formula["theta_psi_unspecified"][1]
            elif self.odf.number_of_angles == 3:
                lambdified_dalpha2_lab_matrix = formula["general"][1]
            else:
                raise Exception("number of angles is not meaningful")
        for mode in self.list_of_modes_in_range:
            mode.calculate_polarizability_squared_derivatives_matrix_in_lab_frame(self.odf, lambdified_dalpha2_lab_matrix)

        for delta_w in self.light_freq_range:
            summations = np.array([[0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0]])
            for mode in self.list_of_modes_in_range:
                summations += mode.calculate_raman_intensity_for_each_mode(delta_w)
            for i in range(3):
                for j in range(3):
                    intensities[3 * i + j].append(summations[i, j])
        self.raman_intensity = np.array(intensities)
        print("Raman intensity calculation for range {} - {} is done".format(self.freq_range_start, self.freq_range_end))

    def calculate_suceptatability(self, extra_mode: bool):
        suceptabilities = [[] for i in range(27)]
        ims = [[] for i in range(27)]
        reals = [[] for i in range(27)]
        mag_squareds = [[] for i in range(27)]
        with open("results/sympy formula results/formula.dat", "rb") as formulafile:
            formula = dill.load(formulafile)
            if self.odf.number_of_angles == 1:
                lambdified_hyperpolarizability_lab_tensor = formula["theta_unspecified"][2]
            elif self.odf.number_of_angles == 2:
                lambdified_hyperpolarizability_lab_tensor = formula["theta_psi_unspecified"][2]
            elif self.odf.number_of_angles == 3:
                lambdified_hyperpolarizability_lab_tensor = formula["general"][2]
            else:
                raise Exception("number of angles is not meaningful")
        for mode in self.list_of_modes_in_range:
            mode.calculate_hyperpolarizability_for_each_mode_lab_frame(self.odf, lambdified_hyperpolarizability_lab_tensor)
        all_amps = np.array([[mode.amplitudes[0, 0, 2], mode.amplitudes[0, 2, 0], mode.amplitudes[2, 2, 2]] for mode in self.list_of_modes_in_range])
        desired_max_amp = 100.0
        scaling_amp = desired_max_amp / np.max(np.abs(all_amps))
        for mode in self.list_of_modes_in_range:
            mode.amplitudes = mode.amplitudes * scaling_amp
        if extra_mode:
            self.add_mode(ampxxz=100, ampxzx=100, ampzzz=100, gamma=7, omega0=2900)
        for w in self.light_freq_range:
            summations = np.array([[[complex(self.chi2_non_res[0], 0), complex(self.chi2_non_res[1], 0), complex(self.chi2_non_res[2], 0)] for j in range(3)] for i in range(3)])
            # print(str(w).center(40, "*"))
            for mode in self.list_of_modes_in_range:
                each_mode_suceptabilities = mode.calculate_suceptability_for_each_mode(w)
                # print("peak{}".format(ind), each_mode_suceptabilities[0, 0, 2])
                summations += each_mode_suceptabilities
            # print("=="*30)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        suceptabilities[9 * i + 3 * j + k].append(summations[i, j, k])
                        ims[9 * i + 3 * j + k].append(np.imag(summations[i, j, k]))
                        reals[9 * i + 3 * j + k].append(np.real(summations[i, j, k]))
                        mag_squareds[9 * i + 3 * j + k].append((abs(summations[i, j, k] ** 2)))

        self.X2_yyz = np.array(suceptabilities[2])  # the 2nd element
        self.X2_yzy = np.array(suceptabilities[6])  # the 6th element
        self.X2_zzz = np.array(suceptabilities[26])  # the 26th element
        self.calculate_ssp_chi_2()
        self.calculate_sps_chi_2()
        self.calculate_ppp_chi_2()
        print("SFG Chi2 calculation for range {} - {} is done".format(self.freq_range_start, self.freq_range_end))

    def save(self, normalize_before_save, noise, create_imag_file, filename_mag, filename_imag=None):
        if noise:
            noise_percentage = 5
            ssp_mag_squared = Calculation.add_noise_to_array(abs(self.ssp) ** 2, noise_percentage)
            ssp_im = Calculation.add_noise_to_array(np.imag(self.ssp), noise_percentage)
            sps_mag_squared = Calculation.add_noise_to_array(abs(self.sps) ** 2, noise_percentage)
            sps_im = Calculation.add_noise_to_array(np.imag(self.sps), noise_percentage)
            ppp_mag_squared = Calculation.add_noise_to_array(abs(self.ppp) ** 2, noise_percentage)
            ppp_im = Calculation.add_noise_to_array(np.imag(self.ppp), noise_percentage)
        else:
            ssp_mag_squared = abs(self.ssp) ** 2
            ssp_im = np.imag(self.ssp)
            sps_mag_squared = abs(self.sps) ** 2
            sps_im = np.imag(self.sps)
            ppp_mag_squared = abs(self.ppp) ** 2
            ppp_im = np.imag(self.ppp)
        if normalize_before_save:
            normalization_factor_mag_squared_polarizations = np.max([ssp_mag_squared, sps_mag_squared, ppp_mag_squared])
            normalization_factor_im_polarizations = np.max([ssp_im, sps_im, ppp_im])
            ssp_mag_squared = ssp_mag_squared / normalization_factor_mag_squared_polarizations if normalize_before_save else ssp_mag_squared
            ssp_im = ssp_im / normalization_factor_im_polarizations if normalize_before_save else ssp_im
            sps_mag_squared = sps_mag_squared / normalization_factor_mag_squared_polarizations if normalize_before_save else sps_mag_squared
            sps_im = sps_im / normalization_factor_im_polarizations if normalize_before_save else sps_im
            ppp_mag_squared = ppp_mag_squared / normalization_factor_mag_squared_polarizations if normalize_before_save else ppp_mag_squared
            ppp_im = ppp_im / normalization_factor_im_polarizations if normalize_before_save else ppp_im
        data_ready_to_save_mag_squared = np.array([self.light_freq_range, sps_mag_squared, ssp_mag_squared, ppp_mag_squared]).T
        filepath_mag = Calculation.give_filepath(self.molecule.name, normalize_before_save, noise, filename_mag)

        np.savetxt(filepath_mag, data_ready_to_save_mag_squared)
        if create_imag_file:
            filepath_imag = Calculation.give_filepath(self.molecule.name, normalize_before_save, noise, filename_imag)
            data_ready_to_save_im = np.array([self.light_freq_range, sps_im, ssp_im, ppp_im]).T
            np.savetxt(filepath_imag, data_ready_to_save_im)
        print("Mag squared and imag files are saved. The order is: sps, ssp, ppp")

    def create_guide_table(self):
        pass


class Calculation:

    @staticmethod
    def give_sign_combinations(number_of_combinations):
        return list(itertools.product(['+', "-"], repeat=number_of_combinations))

    @staticmethod
    def detect_certatin_possibilities(table_of_amplitudes):
        for angle in table_of_amplitudes:
            pass

    @staticmethod
    def make_sure_the_directory_exists(molecule_name: str, normalization: bool, noise: bool):
        normalization_string = "normalized" if normalization else "not_normalized"
        noise_string = "noisy" if noise else "smooth"
        directory_path = "./results/{}/experimental/{}/{}".format(molecule_name, noise_string, normalization_string)
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return directory_path

    @staticmethod
    def give_filepath(molecule_name: str, normalization: bool, noise: bool, filename: str):
        directory_path = Calculation.make_sure_the_directory_exists(molecule_name, normalization, noise)
        return directory_path + "/" + filename

    @staticmethod
    def add_noise_to_array(arr, percentage):
        arr_max = np.max(arr)
        arr_min = np.min(arr)
        max_noise = abs((arr_max - arr_min) * percentage / 100)
        noise = np.random.normal(0, max_noise, arr.shape)
        return arr + noise

    @staticmethod
    def normalized_to_max(array):
        return array / np.max(array)

    @staticmethod
    def normalized_to_min(array):
        return array / np.min(array)

    @staticmethod
    def single_integration(parameters: tuple, func, a=0, b=np.pi):
        I = quad(func, a, b, args=parameters)
        return I[0]

    @staticmethod
    def double_integration(parameters: tuple, func, a0=0, b0=np.pi, a1=0, b1=2 * np.pi):
        I = dblquad(func, a0, b0, a1, b1, args=parameters)
        return I[0]

    @staticmethod
    def triple_integration(parameters: tuple, func, a0=0, b0=np.pi, a1=0, b1=2 * np.pi, a2=0, b2=2 * np.pi):
        I = tplquad(func, a0, b0, a1, b1, a2, b2, args=parameters)
        return I[0]


class Parser:
    ATOM_PATTERN = r"Atom\s+(\d+)\shas\satomic\snumber\s+(\d+)"
    FREQUENCY_PATTERN = r"Frequencies\s+-{2}\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)"
    REDUCED_MASS_PATTERN = r"Red\.\smasses\s+-{2}\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)"
    DIPOLE_MOMENT_DERIVATIVES_PATTERN = r"Dipole\sderivative\swrt\smode\s+(\d+):\s+(.*)"
    POLARIZABILITY_DERIVATIVES_PATTERN_1 = r"Polarizability\sderivatives\swrt\smode\s+\d+\n"
    POLARIZABILITY_DERIVATIVES_PATTERN_2 = r"Polar\stensor\sderivatives"
    ATOM_COORDINATES_PATTERN_1 = r"Coordinates\s\(Angstroms\)\n.*\n.*\n"
    ATOM_COORDINATES_PATTERN_2 = r"\-*\n\s*Distance\smatrix\s"

    def __init__(self, file_name, molecule: Molecule):
        self.file_name = file_name
        self.read_file()
        self.molecule = molecule

    @staticmethod
    def convert_D_number_to_normal(string_D_type_number: str):
        D_index = string_D_type_number.find("D")
        if D_index == -1:
            raise Exception(string_D_type_number + "is not a D type number")
        base = float(string_D_type_number[:D_index])
        if -0.0001 < base < 0.0001:
            return None
        power = int(string_D_type_number[D_index + 1:])
        result = base * (10 ** power)
        # print("the D number: {}, the base: {}, the power: {}, the final float number: {}".format(
        # string_D_type_number, base, power, result))
        return result

    @staticmethod
    def convert_D_numbers_to_normals(list_of_D_strings) -> list:
        results = []
        for string_D_number in list_of_D_strings:
            converted = Parser.convert_D_number_to_normal(string_D_number)
            if converted is not None:
                results.append(converted)
        return results

    def frequency_parse(self):
        frequencies = []
        matches = re.finditer(self.FREQUENCY_PATTERN, self.file_content)
        for match in matches:
            for frequency in match.groups():
                frequencies.append(float(frequency))
        for i in range(self.molecule.number_of_normal_modes):
            self.molecule.list_of_modes[i].frequency = frequencies[i]

    def reduced_mass_parse(self):
        # returns a list of reduced masses
        redmasses = []
        matches = re.finditer(self.REDUCED_MASS_PATTERN, self.file_content)
        for match in matches:
            for redmass in match.groups():
                redmasses.append(float(redmass))
        if len(redmasses) != self.molecule.number_of_normal_modes:
            raise Exception(
                "wrong parsing, number of normal modes ({}) and reduced masses ({}) are not the same".format(
                    len(self.molecule.list_of_modes), len(redmasses)))
        for i in range(self.molecule.number_of_normal_modes):
            self.molecule.list_of_modes[i].red_mass = redmasses[i]

    def polarizability_derivatives_parse(self):
        matches_start = re.finditer(self.POLARIZABILITY_DERIVATIVES_PATTERN_1, self.file_content)
        matches_end = re.finditer(self.POLARIZABILITY_DERIVATIVES_PATTERN_2, self.file_content)
        list_match_starts = list(matches_start)
        list_match_ends = list(matches_end)
        starts_len = len(list_match_starts)
        ends_len = len(list_match_ends)

        if starts_len != ends_len:
            raise Exception("different file format for polarizability derivatives!")
        list_of_results = []
        for i in range(starts_len):
            list_of_results.append((list_match_starts[i].end(), list_match_ends[i].start()))
        final_result = []
        # print(list_of_results)
        for t in list_of_results:
            start = t[0]
            end = t[1]
            res = self.file_content[start:end]
            numbers = [self.convert_D_number_to_normal(item) for item in res.split() if "D" in item]
            new_numbers = [n for n in numbers if n is not None]
            first_line = new_numbers[:-3]
            second_line = new_numbers[-3:]
            first_line.insert(2, second_line[0])
            first_line.insert(5, second_line[1])
            first_line.append(second_line[-1])
            matrix_line1 = first_line[:3]
            matrix_line2 = first_line[3:6]
            matrix_line3 = first_line[6:]

            final_result.append([matrix_line1, matrix_line2, matrix_line3])

        for i in range(self.molecule.number_of_normal_modes):
            middle_step = np.array(final_result[i]) * 1
            self.molecule.list_of_modes[i].polarizability_derivatives = middle_step.tolist()

    def dipole_moment_derivatives_parse(self):
        dipole_moment_derivative_results = []
        matches = re.finditer(self.DIPOLE_MOMENT_DERIVATIVES_PATTERN, self.file_content)
        for match in matches:
            derivatives = self.convert_D_numbers_to_normals(match.group(2).split())
            dipole_moment_derivative_results.append(derivatives)
        if len(dipole_moment_derivative_results) != self.molecule.number_of_normal_modes:
            raise Exception("wrong parsing")
        for i in range(self.molecule.number_of_normal_modes):
            self.molecule.list_of_modes[i].dipole_moment_derivatives = [number * 1 for number in dipole_moment_derivative_results[i]]

    def read_file(self):
        # open the file and read the lines and allocate it to a variable
        file = open(self.file_name, "rt")
        self.file_content = file.read()
        file.close()

    def determine_atoms(self):
        matches = re.finditer(self.ATOM_PATTERN, self.file_content)
        for match in matches:
            atom_index, atomic_number = match.groups()
            atom = Atom(self.molecule, atomic_number)
            atom.index_in_molecule = atom_index
            self.molecule.list_of_atoms.append(atom)
        self.molecule.construct_the_molecule()

    def atom_coordinates_parser(self):
        start_parsing_list = list(re.finditer(self.ATOM_COORDINATES_PATTERN_1, self.file_content))
        end_parsing_list = list(re.finditer(self.ATOM_COORDINATES_PATTERN_2, self.file_content))
        last_start = start_parsing_list[-1].end()
        last_end = end_parsing_list[-1].start()
        raw_orientation_string = self.file_content[last_start:last_end].strip().split("\n")
        for line in raw_orientation_string:
            line_list = line.strip().split()
            atom_index, atomic_number, atomic_type, x, y, z = line_list
            # this searching is not efficient at all!!!
            for atom in self.molecule.list_of_atoms:
                if atom.index_in_molecule == atom_index and atom.atomic_number == atomic_number:
                    atom.coordinates = np.array([float(x), float(y), float(z)])
                    break


class Mode:
    def __init__(self, molecule: Molecule):
        self.mode_index = 0
        self.red_mass = 0
        self.frequency = 0
        self.polarizability_derivatives = None  # nine numbers, 3 2D list
        self.dipole_moment_derivatives = None  # 3 numbers, the type is list
        self.hyperpolarizability = None  # 27 elements in a tensor
        self.dipole_moment_squared_derivatives_in_lab_frame = None
        self.polarizability_squared_derivatives_in_lab_frame = None
        self.hyperpolarizability_for_each_mode_lab_frame = None  # 3D list
        self.gamma = 10
        self.molecule = molecule
        self.amplitudes = np.zeros((3, 3, 3))

    def calculate_amplitude(self, odf: ODF):
        with open("results/sympy formula results/formula.dat", "rb") as formulafile:
            formula = dill.load(formulafile)
            if odf.number_of_angles == 1:
                lambdified_hyperpolarizability_lab_tensor = formula["theta_unspecified"][2]
            elif odf.number_of_angles == 2:
                lambdified_hyperpolarizability_lab_tensor = formula["theta_psi_unspecified"][2]
            elif odf.number_of_angles == 3:
                lambdified_hyperpolarizability_lab_tensor = formula["general"][2]
            else:
                raise Exception("number of angles is not meaningful")
        self.calculate_hyperpolarizability_for_each_mode_lab_frame(odf, lambdified_hyperpolarizability_lab_tensor)

    def calculate_hyperpolarizability_for_each_mode_molecular(self):
        self.hyperpolarizability = np.tensordot(self.polarizability_derivatives, self.dipole_moment_derivatives, axes=0)

    def calculate_hyperpolarizability_for_each_mode_lab_frame(self, odf, formula_coming_from_sympy_hyperpolarizability_lab_ternsor):
        # if odf.odf_type == "delta_dirac" and just theta is active I assume we are working with just delta function and just theta
        dmua = self.dipole_moment_derivatives[0]
        dmub = self.dipole_moment_derivatives[1]
        dmuc = self.dipole_moment_derivatives[2]
        alphaaa = self.polarizability_derivatives[0][0]
        alphaab = self.polarizability_derivatives[0][0]
        alphaac = self.polarizability_derivatives[0][0]
        alphaba = self.polarizability_derivatives[1][1]
        alphabb = self.polarizability_derivatives[1][1]
        alphabc = self.polarizability_derivatives[1][1]
        alphaca = self.polarizability_derivatives[2][2]
        alphacb = self.polarizability_derivatives[2][2]
        alphacc = self.polarizability_derivatives[2][2]
        lab_hyperpolarizability_tensor = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if odf.odf_type == "delta_dirac":
                        theta0 = odf.theta["theta0"]
                        lab_hyperpolarizability = formula_coming_from_sympy_hyperpolarizability_lab_ternsor[i][j][k](theta0, dmua, dmub, dmuc, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc) / (4 * np.pi ** 2)
                        # notation = "{}_{}_{}".format(i, j, k)
                        # if notation == "0_0_2" or notation == "0_2_0" or notation == "2_2_2":
                        #     print("delta dirac lab polarizability {}_{}_{}".format(i, j, k), lab_hyperpolarizability)
                    elif odf.odf_type == "gaussian":
                        if odf.number_of_angles == 1:  # tilt angle
                            nominator_function = lambda theta, dmua, dmub, dmuc, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc: odf.odf_gaussian_theta(theta) * formula_coming_from_sympy_hyperpolarizability_lab_ternsor[i][j][k](theta, dmua, dmub, dmuc, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc) * np.sin(theta)
                            nominator_integration = quad(nominator_function, 0, np.pi, args=(dmua, dmub, dmuc, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc))[0]
                            denominator_function = lambda theta: odf.odf_gaussian_theta(theta) * np.sin(theta)
                            denominator_integration = quad(denominator_function, 0, np.pi)[0]
                            lab_hyperpolarizability = nominator_integration / ((4 * np.pi ** 2) * denominator_integration)
                            # notation = "{}_{}_{}".format(i, j, k)
                            # if notation == "0_0_2" or notation == "0_2_0" or notation == "2_2_2":
                            #     print("gaussian lab polarizability {}_{}_{}".format(i, j, k), lab_hyperpolarizability)
                                # print("denominator integration", denominator_integration)
                                # print("nominator integration", nominator_integration)
                                # print("ratio of integrations", lab_hyperpolarizability)
                                # print("----------------------------------------------")


                        elif odf.number_of_angles == 2:  # theta and psi
                            nominator_function = lambda theta, psi, dmua, dmub, dmuc, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc: odf.odf_gaussian_theta_psi(theta, psi) * formula_coming_from_sympy_hyperpolarizability_lab_ternsor[i][j][k](theta, psi, dmua, dmub, dmuc, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc) * np.sin(theta)
                            nominator_integration = dblquad(nominator_function, 0, np.pi, 0, 2 * np.pi, args=(dmua, dmub, dmuc, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc))[0]
                            denominator_function = lambda theta, psi: odf.odf_gaussian_theta(theta, psi) * np.sin(theta)
                            denominator_integration = dblquad(denominator_function, 0, np.pi, 0, 2 * np.pi)[0]
                            lab_hyperpolarizability = nominator_integration / ((2 * np.pi) * denominator_integration)
                        elif odf.number_of_angles == 3:  # theta, psi, and phi
                            nominator_function = lambda theta, phi, psi, dmua, dmub, dmuc, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc: odf.odf_gaussian_general(theta, phi, psi) * formula_coming_from_sympy_hyperpolarizability_lab_ternsor[i][j][k](theta, phi, psi, dmua, dmub, dmuc, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc) * np.sin(theta)
                            nominator_integration = tplquad(nominator_function, 0, np.pi, 0, 2 * np.pi, 0, 2 * np.pi, args=(dmua, dmub, dmuc, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc))[0]
                            denominator_function = lambda theta, phi, psi: odf.odf_gaussian_general(theta, phi, psi) * np.sin(theta)
                            denominator_integration = tplquad(denominator_function, 0, np.pi, 0, 2 * np.pi, 0, 2 * np.pi)[0]
                            lab_hyperpolarizability = nominator_integration / denominator_integration
                        else:
                            raise Exception("number of angles is out of range")
                    lab_hyperpolarizability_tensor[i, j, k] = lab_hyperpolarizability
        self.amplitudes = lab_hyperpolarizability_tensor / (2 * self.red_mass * self.frequency)
        # print(self.amplitudes[0, 0, 2], self.amplitudes[0, 2, 0], self.amplitudes[2, 2, 2])
        # print("----"*10)

        self.hyperpolarizability_for_each_mode_lab_frame = lab_hyperpolarizability_tensor

    def calculate_raman_intensity_for_each_mode(self, delta_w):
        result = [[None, None, None],
                  [None, None, None],
                  [None, None, None]]

        for i in range(3):
            for j in range(3):
                result[i][j] = (1 / (2 * self.red_mass * self.frequency)) * \
                               self.polarizability_squared_derivatives_in_lab_frame[i][j] * (self.gamma ** 2) / (
                                       (delta_w - self.frequency) ** 2 + self.gamma ** 2)
        return np.array(result)

    def calculate_polarizability_squared_derivatives_matrix_in_lab_frame(self, odf, formula_coming_from_sympy_dalpha2_lab_matrix):
        # load the formula
        # if odf.odf_type == "delta_dirac" and just theta is active I assume we are working with just delta function and just theta

        alphaaa = self.polarizability_derivatives[0][0]
        alphaab = self.polarizability_derivatives[0][0]
        alphaac = self.polarizability_derivatives[0][0]
        alphaba = self.polarizability_derivatives[1][1]
        alphabb = self.polarizability_derivatives[1][1]
        alphabc = self.polarizability_derivatives[1][1]
        alphaca = self.polarizability_derivatives[2][2]
        alphacb = self.polarizability_derivatives[2][2]
        alphacc = self.polarizability_derivatives[2][2]
        lab_alpha_squared_matrix = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if odf.odf_type == "delta_dirac":
                    theta0 = odf.theta["theta0"]
                    lab_alpha_squared = formula_coming_from_sympy_dalpha2_lab_matrix[i][j](theta0, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc) / (4 * np.pi ** 2)

                elif odf.odf_type == "gaussian":
                    if odf.number_of_angles == 1:
                        nominator_function = lambda theta, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc: odf.odf_gaussian_theta(theta) * formula_coming_from_sympy_dalpha2_lab_matrix[i][j](theta, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc) * np.sin(theta)
                        nominator_integration = quad(nominator_function, 0, np.pi, args=(alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc))[0]
                        denominator_function = lambda theta: odf.odf_gaussian_theta(theta) * np.sin(theta)
                        denominator_integration = quad(denominator_function, 0, np.pi)[0]
                        lab_alpha_squared = nominator_integration / ((4 * np.pi ** 2) * denominator_integration)
                    elif odf.number_of_angles == 2:
                        nominator_function = lambda theta, psi, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc: odf.odf_gaussian_theta_psi(theta, psi) * formula_coming_from_sympy_dalpha2_lab_matrix[i][j](theta, psi, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc) * np.sin(theta)
                        nominator_integration = dblquad(nominator_function, 0, np.pi, 0, 2 * np.pi, args=(alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc))[0]
                        denominator_function = lambda theta, psi: odf.odf_gaussian_theta_psi(theta, psi) * np.sin(theta)
                        denominator_integration = dblquad(denominator_function, 0, np.pi, 0, 2 * np.pi)[0]
                        lab_alpha_squared = nominator_integration / ((2 * np.pi) * denominator_integration)
                    elif odf.number_of_angles == 3:
                        nominator_function = lambda theta, phi, psi, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc: odf.odf_gaussian_general(theta, phi, psi) * formula_coming_from_sympy_dalpha2_lab_matrix[i][j](theta, phi, psi, alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc) * np.sin(theta)
                        nominator_integration = tplquad(nominator_function, 0, np.pi, 0, 2 * np.pi, 0, 2 * np.pi, args=(alphaaa, alphaab, alphaac, alphaba, alphabb, alphabc, alphaca, alphacb, alphacc))[0]
                        denominator_function = lambda theta, phi, psi: odf.odf_gaussian_general(theta, phi, psi) * np.sin(theta)
                        denominator_integration = tplquad(denominator_function, 0, np.pi, 0, 2 * np.pi, 0, 2 * np.pi)[0]
                        lab_alpha_squared = nominator_integration / denominator_integration
                    else:
                        raise Exception("number of angles is out of range")
                lab_alpha_squared_matrix[i, j] = lab_alpha_squared
        self.polarizability_squared_derivatives_in_lab_frame = lab_alpha_squared_matrix

    def calculate_dipole_moment_squared_derivatives_vector_in_lab_frame(self, odf, formula_coming_from_sympy_dmu2_lab_list):
        # if odf.odf_type == "delta_dirac" and just theta is active I assume we are working with just delta function and just theta
        # in the case that we have gaussian we have to do an integral. the integral is single integration if we just have titl angle

        lab_mu_list = []
        dmua = self.dipole_moment_derivatives[0]
        dmub = self.dipole_moment_derivatives[1]
        dmuc = self.dipole_moment_derivatives[2]
        for i in range(3):
            if odf.odf_type == "delta_dirac":  # at this stage we just assume delta dirac with just one angle (tilt angle)
                theta0 = odf.theta["theta0"]
                lab_mu_squared = formula_coming_from_sympy_dmu2_lab_list[i](theta0, dmua, dmub, dmuc) / (4 * np.pi ** 2)

            elif odf.odf_type == "gaussian":
                if odf.number_of_angles == 1:  # tilt angle
                    nominator_function = lambda theta, mua, mub, muc: odf.odf_gaussian_theta(theta) * formula_coming_from_sympy_dmu2_lab_list[i](theta, mua, mub, muc) * np.sin(theta)
                    nominator_integration = quad(nominator_function, 0, np.pi, args=(dmua, dmub, dmuc))[0]
                    denominator_function = lambda theta: odf.odf_gaussian_theta(theta) * np.sin(theta)
                    denominator_integration = quad(denominator_function, 0, np.pi)[0]
                    lab_mu_squared = nominator_integration / ((4 * np.pi ** 2) * denominator_integration)
                elif odf.number_of_angles == 2:  # theta and psi
                    nominator_function = lambda theta, psi, mua, mub, muc: odf.odf_gaussian_theta_psi(theta, psi) * formula_coming_from_sympy_dmu2_lab_list[i](theta, psi, mua, mub, muc) * np.sin(theta)
                    nominator_integration = dblquad(nominator_function, 0, np.pi, 0, 2 * np.pi, args=(dmua, dmub, dmuc))[0]
                    denominator_function = lambda theta, psi: odf.odf_gaussian_theta(theta, psi) * np.sin(theta)
                    denominator_integration = dblquad(denominator_function, 0, np.pi, 0, 2 * np.pi)[0]
                    lab_mu_squared = nominator_integration / ((2 * np.pi) * denominator_integration)
                elif odf.number_of_angles == 3:  # theta, psi, and phi
                    nominator_function = lambda theta, phi, psi, mua, mub, muc: odf.odf_gaussian_general(theta, phi, psi) * formula_coming_from_sympy_dmu2_lab_list[i](theta, phi, psi, mua, mub, muc) * np.sin(theta)
                    nominator_integration = tplquad(nominator_function, 0, np.pi, 0, 2 * np.pi, 0, 2 * np.pi, args=(dmua, dmub, dmuc))[0]
                    denominator_function = lambda theta, phi, psi: odf.odf_gaussian_general(theta, phi, psi) * np.sin(theta)
                    denominator_integration = tplquad(denominator_function, 0, np.pi, 0, 2 * np.pi, 0, 2 * np.pi)[0]
                    lab_mu_squared = nominator_integration / denominator_integration
                else:
                    raise Exception("number of angles is out of range")
            lab_mu_list.append(lab_mu_squared)
        self.dipole_moment_squared_derivatives_in_lab_frame = lab_mu_list

    def calculate_IR_intensity_for_each_mode(self, W_IR):
        intensities = []
        for i in range(3):
            intensities.append((1 / (2 * self.red_mass * self.frequency)) * self.dipole_moment_squared_derivatives_in_lab_frame[i] * self.gamma / ((W_IR - self.frequency) ** 2 + (self.gamma ** 2)))

        return intensities

    def calculate_suceptability_for_each_mode(self, w):
        result = np.array([[[complex(0, 0) for i in range(3)] for j in range(3)] for k in range(3)])
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    result[i, j, k] = complex(self.amplitudes[i, j, k], 0) / complex((w - self.frequency), self.gamma)

        return result


class Curve_Fitter:
    """
    the curve fitter should be lke it.
    parameters that I need to calculate the amplitude: Ls(come from the lab object), mode information(comes from the molecule),
    point: if mode information is None (which is none in real world) the code itself is to find the mode information (number of peaks, peak position)
    curve_fitter = Curve_Fitter(method='fmin_tnc' [ this is the default method ] and I assume there is no more methods)
    curve_fitter.set_parameters(lab, molecule, file) the molecule can be None but if it's none I should do some lineshape and peak analyzer calculations
    curve_fitter.set(parameters you want to find) this can be true and false options are: A, NR_chi2, freq0, gamma)
    for each mode we have 1 A, 1 freq0, 1 gamma and finally for each polarization we have 1 NR_chi2 ==> for alanine: 4 modes (1_gamma + 1_freq0 + 1_A) + 1_NR_chi2

    for ssp: 4 * 3(Axxz, freq0, gamma) + 1 = 13
    for sps: 4 * 3(Azxz, freq0, gamma) + 1 = 13
    for ppp: 4 * 3(Azzz, freq0, gamma) + 1 = 13
    39 parameters to find

    now in this stage I just want to find A, gamma, and the NR_chi2

    for ssp: 4 * 2(Axxz, gamma) + 1 = 9
    for sps: 4 * 2(Axzx, gamma) + 1 = 9
    for ppp: 4 * 2(Azzz, gamma) + 1 = 9

    ==> the curve fitter should be able to fit in different scenarios
    scenario 1: global fitting (all polarizations at once)
    scenario 2: find just ssp, just sps, or just ppp [ in this scenario I need just one of them]
    scenario 3: find them in order: find ssp or sps first and separately and find A, gamma or ther things
    find the average of all the same parameters that should be the same in ssp and sps polarizations (gamma, freq0) or just gamma if I work with it
    do the curve fitting again but in this step I just want to find the A and the NR_chi2. the gamma (and freq0) are fixed from the averaging
    now I have all ssp and sps information it's the time to go for ppp and I need to find A and NR_chi2 for ppp. the gamma (and freq0) is(are) fixed from previous steps.
    that's the thing I want ot implement.
    I can do another averaging (by finding gamma (and freq0) also for ppp and then do another averaging and finally go for the final fitting


    """

    def __init__(self, lab: Lab, molecule: Molecule, filepath_mag, filepath_imag):
        self.parameters = None
        self.find_gamma = None
        self.find_omega0 = None
        self.lab = lab
        self.molecule = molecule
        self.filepath_mag = filepath_mag
        self.filepath_imag = filepath_imag
        self.found_NR_chi2_xxz = 0
        self.found_NR_chi2_xzx = 0
        self.found_NR_chi2_zzz = 0
        self.freq_list = None
        self.ssp_mag_list = None
        self.ssp_im_list = None
        self.just_plotting = False

        self.sps_mag_list = None
        self.sps_im_list = None

        self.ppp_mag_list = None
        self.ppp_im_list = None

        self.read_data()
        self.original_modes = self.found_modes_in_range()
        self.found_modes = copy.deepcopy(self.original_modes)
        self.model_ssp_vectorized = np.vectorize(self.model_ssp)
        self.model_sps_vectorized = np.vectorize(self.model_sps)
        self.model_ppp_vectorized = np.vectorize(self.model_ppp)
        print("a curve fitter object is created")

    def found_modes_in_range(self):
        start = self.freq_list[0]
        end = self.freq_list[-1]
        return [mode for mode in self.molecule.list_of_modes if start <= mode.frequency <= end]

    def read_data(self):
        freq, sps, ssp, ppp = loadtxt(self.filepath_mag, unpack=True)
        freq2, im_sps, im_ssp, im_ppp = loadtxt(self.filepath_imag, unpack=True)
        base_length = len(freq)
        if all([True if length == base_length else False for length in [len(ssp), len(sps), len(ppp), len(freq2), len(im_ssp), len(im_sps), len(im_ppp)]]):
            self.freq_list = freq
            self.ssp_mag_list = ssp
            self.ssp_im_list = im_ssp
            self.sps_mag_list = sps
            self.sps_im_list = im_sps
            self.ppp_mag_list = ppp
            self.ppp_im_list = im_ppp
            # print("mag squared file path", self.filepath_abs)
            # print("im file path", self.filepath_imag)
            # print("freq", self.freq_list)
            # print("freq2", freq2)
            # print("ssp", self.ssp_mag_list)
            # print("sps", self.sps_mag_list)
            # print("ppp", self.ppp_mag_list)
            # print("im ssp", self.ssp_im_list)
            # print("sps_im", self.sps_im_list)
            # print("ppp im", self.ppp_im_list)

        else:
            raise Exception("bad file please check the file")

    def generate_initial_guess_one_polarization(self, generate_gamma: bool, generate_omega0: bool, polarization: str):
        initial_guess_list = []
        initial_guess_NR = np.sqrt(np.min(self.sps_mag_list))
        initial_guess_list.append(initial_guess_NR)
        number_of_modes = len(self.original_modes)
        for i in range(number_of_modes):
            if random.choice([True, False]):
                # for amplitude
                initial_guess_list.append(np.sqrt(np.max(self.sps_mag_list)))
            else:
                initial_guess_list.append(-1.0 * np.sqrt(np.max(self.sps_mag_list)))
            if generate_gamma:
                # for the gamma (width of Lorentzian model)
                initial_guess_list.append(np.random.normal(10, 3))
            if generate_omega0:
                # for the omega0 (the position of each peak)
                initial_guess_list.append(self.original_modes[i].frequency)
        print("random initial guess for {} created. generate gamma: {}, generate omega0: {}.".format(polarization, generate_gamma, generate_omega0))
        return initial_guess_list

    def fit_ssp(self, initial_guess: list, bounds: list, number_of_modes):
        print("fitting ssp".center(50, "-"))
        self.check_initial_bounds_find_gamma_find_omega(bounds, initial_guess, number_of_modes)
        opt_ssp = fmin_tnc(self.error_ssp, initial_guess, maxfun=900, approx_grad=True, bounds=bounds, disp=0)[0]

        self.found_NR_chi2_xxz = opt_ssp[0]
        modes_opt_ssp = opt_ssp[1:]
        for i, mode in enumerate(self.found_modes):
            if self.find_gamma and self.find_omega0:
                self.found_modes[i].amplitudes[0, 0, 2] = modes_opt_ssp[3 * i]
                self.found_modes[i].gamma = modes_opt_ssp[3 * i + 1]
                self.found_modes[i].frequency = modes_opt_ssp[3 * i + 2]
            elif self.find_gamma and not self.find_omega0:
                self.found_modes[i].amplitudes[0, 0, 2] = modes_opt_ssp[2 * i]
                self.found_modes[i].gamma = modes_opt_ssp[2 * i + 1]
            elif not (self.find_gamma and self.find_omega0):
                self.found_modes[i].amplitudes[0, 0, 2] = modes_opt_ssp[i]
            else:
                raise Exception(" Hey man! watch out!")

            print("found mode frequency", self.found_modes[i].frequency, "found mode gamma", self.found_modes[i].gamma, "found amp xxz", self.found_modes[i].amplitudes[0, 0, 2])
        print("NR xxz:", self.found_NR_chi2_xxz)
        print("SSP fitting is done.")
        print("find gamma:", self.find_gamma, "find omega0", self.find_omega0)
        print("initial guess:", initial_guess)
        print("bounds for ssp", bounds)
        print("----" * 30)
        self.just_plotting = True

        final_ssp_mag_list = self.model_ssp_vectorized(self.freq_list)
        res = self.ssp_mag_list - final_ssp_mag_list
        squared_error = sum(res ** 2)
        fig, ax = plt.subplots()

        ax.plot(self.freq_list, self.ssp_mag_list, color="blue", label="experimental")
        ax.plot(self.freq_list, final_ssp_mag_list, color="red", label="fitting")
        ax.set_ylabel("$|\chi^{(2)}|^2_{SSP}$  /  $A.U$")
        ax.set_xlabel("IR wavenumber  /  $cm^{-1}$")
        ax.legend()
        for real_mode, found_mode in zip(self.original_modes, self.found_modes):
            ax.axvline(real_mode.frequency, color="black", ls="dotted")
            ax.axvline(found_mode.frequency, color="olive", ls="dotted")
        self.just_plotting = False
        return self.found_NR_chi2_xxz, copy.deepcopy(self.found_modes), squared_error, fig

    def fit_sps(self, initial_guess: list, bounds: list, number_of_modes):
        print("fitting sps".center(50, "-"))
        self.check_initial_bounds_find_gamma_find_omega(bounds, initial_guess, number_of_modes)
        opt_sps = fmin_tnc(self.error_sps, initial_guess, maxfun=900, approx_grad=True, bounds=bounds, disp=0)[0]
        self.found_NR_chi2_xzx = opt_sps[0]
        modes_opt_sps = opt_sps[1:]
        for i, mode in enumerate(self.found_modes):
            if self.find_gamma and self.find_omega0:
                self.found_modes[i].amplitudes[0, 2, 0] = modes_opt_sps[3 * i]
                self.found_modes[i].gamma = modes_opt_sps[3 * i + 1]
                self.found_modes[i].frequency = modes_opt_sps[3 * i + 2]
            elif self.find_gamma and not self.find_omega0:
                self.found_modes[i].amplitudes[0, 2, 0] = modes_opt_sps[2 * i]
                self.found_modes[i].gamma = modes_opt_sps[2 * i + 1]
            elif not (self.find_gamma and self.find_omega0):
                self.found_modes[i].amplitudes[0, 2, 0] = modes_opt_sps[i]
            else:
                raise Exception(" Hey man! watch out!")

            print("found mode frequency", self.found_modes[i].frequency, "found mode gamma", self.found_modes[i].gamma, "found amp xzx", self.found_modes[i].amplitudes[0, 2, 0])
        print("NR xzx:", self.found_NR_chi2_xzx)
        print("SPS fitting is done.")
        print("find gamma:", self.find_gamma, "find omega0", self.find_omega0)
        print("initial guess:", initial_guess)
        print("bounds for SPS", bounds)
        print("----" * 30)
        self.just_plotting = True
        final_sps_mag_list = self.model_sps_vectorized(self.freq_list)
        res = self.sps_mag_list - final_sps_mag_list
        squared_error = sum(res ** 2)
        fig, ax = plt.subplots()
        ax.plot(self.freq_list, self.sps_mag_list, color="blue")
        ax.plot(self.freq_list, final_sps_mag_list, color="red")
        ax.set_ylabel("$|\chi^{(2)}|^2_{SPS}$  /  $A.U$")
        ax.set_xlabel("IR wavenumber  /  $cm^{-1}$")
        ax.legend(["experimental", "fitted"], loc="best")
        for real_mode, found_mode in zip(self.original_modes, self.found_modes):
            ax.axvline(real_mode.frequency, color="black", ls="dotted")
            ax.axvline(found_mode.frequency, color="olive", ls="dotted")
        self.just_plotting = False
        return self.found_NR_chi2_xzx, copy.deepcopy(self.found_modes), squared_error, fig

    def plot_single_fitting(self, filepath, fig, dpi):
        fig.savefig(filepath, dpi=dpi)
        plt.close(fig)

    def fit_ppp(self, initial_guess: list, bounds: list, number_of_modes):
        print("fitting ppp".center(50, "-"))
        self.check_initial_bounds_find_gamma_find_omega(bounds, initial_guess, number_of_modes)
        opt_ppp = fmin_tnc(self.error_ppp, initial_guess, maxfun=900, approx_grad=True, bounds=bounds, disp=0)[0]
        self.found_NR_chi2_zzz = opt_ppp[0]
        modes_opt_ppp = opt_ppp[1:]
        for i, mode in enumerate(self.found_modes):
            if self.find_gamma and self.find_omega0:
                self.found_modes[i].amplitudes[2, 2, 2] = modes_opt_ppp[3 * i]
                self.found_modes[i].gamma = modes_opt_ppp[3 * i + 1]
                self.found_modes[i].frequency = modes_opt_ppp[3 * i + 2]
            elif self.find_gamma and not self.find_omega0:
                self.found_modes[i].amplitudes[2, 2, 2] = modes_opt_ppp[2 * i]
                self.found_modes[i].gamma = modes_opt_ppp[2 * i + 1]
            elif not (self.find_gamma and self.find_omega0):
                self.found_modes[i].amplitudes[2, 2, 2] = modes_opt_ppp[i]
            else:
                raise Exception(" Hey man! watch out!")
            print("found mode frequency", self.found_modes[i].frequency, "found mode gamma", self.found_modes[i].gamma, "found amp zzz", self.found_modes[i].amplitudes[2, 2, 2])
        print("NR zzz:", self.found_NR_chi2_zzz)
        print("PPP fitting is done.")
        print("find gamma:", self.find_gamma, "find omega0", self.find_omega0)
        print("initial guess:", initial_guess)
        print("bounds for PPP", bounds)
        print("----" * 30)
        self.just_plotting = True
        final_ppp_mag_list = self.model_ppp_vectorized(self.freq_list)
        res = self.ppp_mag_list - final_ppp_mag_list
        squared_error = sum(res ** 2)
        fig, ax = plt.subplots()
        ax.plot(self.freq_list, self.ppp_mag_list, color="blue")
        ax.plot(self.freq_list, final_ppp_mag_list, color="red")
        ax.set_ylabel("$|\chi^{(2)}|^2_{PPP}$  /  $A.U$")
        ax.set_xlabel("IR wavenumber  /  $cm^{-1}$")
        ax.legend(["experimental", "fitted"], loc="best")
        for real_mode, found_mode in zip(self.original_modes, self.found_modes):
            ax.axvline(real_mode.frequency, color="black", ls="dotted")
            ax.axvline(found_mode.frequency, color="olive", ls="dotted")
        self.just_plotting = False
        return self.found_NR_chi2_zzz, self.found_modes, squared_error, fig

    def check_initial_bounds_find_gamma_find_omega(self, bounds, initial_guess, number_of_modes):
        initial_guess_length = len(initial_guess)
        bounds_length = len(bounds)
        if bounds_length != initial_guess_length and bounds_length != 0:
            raise Exception("initial guess and the bounds should have the same length")
        modes_number = number_of_modes
        if initial_guess_length == 1 + modes_number:
            self.find_gamma = False
            self.find_omega0 = False
        elif initial_guess_length == 1 + modes_number * 2:
            self.find_gamma = True
            self.find_omega0 = False
        elif initial_guess_length == 1 + modes_number * 3:
            self.find_gamma = True
            self.find_omega0 = True
        else:
            raise Exception("check the initial guess length. you may miss something")

    def model_ssp(self, freq):
        modes_parameters = self.parameters[1:]
        # if I want to find a parameter I would pick it from the parameters list
        # if I want to keep sth from the original electronic structure calculation I would pick it from the list_of_modes and mode obj
        l1 = self.lab.sfg.Lyy
        l2 = self.lab.vis.Lyy
        l3 = self.lab.ir.Lzz
        summation = complex(0, 0)
        for i in range(len(self.found_modes)):
            if self.just_plotting:
                amp = self.found_modes[i].amplitudes[0, 0, 2]
                gamma = self.found_modes[i].gamma
                mode_frequency = self.found_modes[i].frequency
            else:
                if self.find_gamma and self.find_omega0:
                    amp = modes_parameters[3 * i]
                    gamma = modes_parameters[3 * i + 1]
                    mode_frequency = modes_parameters[3 * i + 2]
                elif self.find_gamma and not self.find_omega0:
                    amp = modes_parameters[2 * i]
                    gamma = modes_parameters[2 * i + 1]
                    mode_frequency = self.found_modes[i].frequency
                elif not (self.find_gamma and self.find_omega0):
                    amp = modes_parameters[i]
                    gamma = self.found_modes[i].gamma
                    mode_frequency = self.found_modes[i].frequency
                else:
                    raise Exception("not supported condition")

            summation += (amp / complex((freq - mode_frequency), gamma))
        if self.just_plotting:
            NR = self.found_NR_chi2_xxz
        else:
            NR = self.parameters[0]
        return abs(l1 * l2 * l3 * (NR + summation)) ** 2

    def model_sps(self, freq):
        modes_parameters = self.parameters[1:]
        # if I want to find a parameter I would pick it from the parameters list
        # if I want to keep sth from the original electronic structure calculation I would pick it from the list_of_modes and mode obj
        l1 = self.lab.sfg.Lyy
        l2 = self.lab.vis.Lzz
        l3 = self.lab.ir.Lyy
        summation = complex(0, 0)
        for i in range(len(self.found_modes)):
            if self.just_plotting:
                amp = self.found_modes[i].amplitudes[0, 2, 0]
                gamma = self.found_modes[i].gamma
                mode_frequency = self.found_modes[i].frequency
            else:
                if self.find_gamma and self.find_omega0:
                    amp = modes_parameters[3 * i]
                    gamma = modes_parameters[3 * i + 1]
                    mode_frequency = modes_parameters[3 * i + 2]
                elif self.find_gamma and not self.find_omega0:
                    amp = modes_parameters[2 * i]
                    gamma = modes_parameters[2 * i + 1]
                    mode_frequency = self.found_modes[i].frequency
                elif not (self.find_gamma and self.find_omega0):
                    amp = modes_parameters[i]
                    gamma = self.found_modes[i].gamma
                    mode_frequency = self.found_modes[i].frequency
                else:
                    raise Exception("not supported condition")

            summation += (amp / complex((freq - mode_frequency), gamma))
        if self.just_plotting:
            NR = self.found_NR_chi2_xzx
        else:
            NR = self.parameters[0]
        return abs(l1 * l2 * l3 * (NR + summation)) ** 2

    def model_ppp(self, freq):
        modes_parameters = self.parameters[1:]
        #     term1
        l11 = self.lab.sfg.Lzz
        l12 = self.lab.vis.Lxx
        l13 = self.lab.ir.Lxx
        summation1 = complex(0, 0)
        #     term2
        l21 = self.lab.sfg.Lxx
        l22 = self.lab.vis.Lzz
        l23 = self.lab.ir.Lxx
        #     term3
        l31 = self.lab.sfg.Lxx
        l32 = self.lab.vis.Lxx
        l33 = self.lab.ir.Lzz
        summation3 = complex(0, 0)
        #     term4
        l41 = self.lab.sfg.Lzz
        l42 = self.lab.vis.Lzz
        l43 = self.lab.ir.Lzz
        summation4 = complex(0, 0)
        for i in range(len(self.found_modes)):  # these are just for zzz because we have xxz and xzx
            mode = self.found_modes[i]
            if self.just_plotting:
                amp = mode.amplitudes[2, 2, 2]
                gamma = mode.gamma
                mode_frequency = mode.frequency
            else:
                if self.find_gamma and self.find_omega0:
                    amp = modes_parameters[3 * i]
                    gamma = modes_parameters[3 * i + 1]
                    mode_frequency = modes_parameters[3 * i + 2]
                elif self.find_gamma and not self.find_omega0:
                    amp = modes_parameters[2 * i]
                    gamma = modes_parameters[2 * i + 1]
                    mode_frequency = mode.frequency
                elif not (self.find_gamma and self.find_omega0):
                    amp = modes_parameters[i]
                    gamma = mode.gamma
                    mode_frequency = mode.frequency
                else:
                    raise Exception("not supported condition")
            summation1 += mode.amplitudes[0, 2, 0] / complex(freq - mode_frequency, gamma)
            summation3 += mode.amplitudes[0, 0, 2] / complex(freq - mode_frequency, gamma)
            summation4 += amp / complex((freq - mode_frequency), gamma)
        summation2 = summation1
        term1 = l11 * l12 * l13 * (self.found_NR_chi2_xzx + summation1)
        term2 = l21 * l22 * l23 * (self.found_NR_chi2_xzx + summation2)
        term3 = l31 * l32 * l33 * (self.found_NR_chi2_xxz + summation3)

        if self.just_plotting:
            NR = self.found_NR_chi2_zzz
        else:
            NR = self.parameters[0]
        term4 = l41 * l42 * l43 * (NR + summation4)
        return abs(term1 + term2 + term3 + term4) ** 2

    def error_ssp(self, parameters):
        self.parameters = parameters
        # parameters variable is the same to initial guess
        res = self.ssp_mag_list - self.model_ssp_vectorized(self.freq_list)
        return sum(res ** 2)

    def error_sps(self, parameters):
        self.parameters = parameters
        res = self.sps_mag_list - self.model_sps_vectorized(self.freq_list)
        return sum(res ** 2)

    def error_ppp(self, parameters):
        self.parameters = parameters
        res = self.ppp_mag_list - self.model_ppp_vectorized(self.freq_list)
        return sum(res ** 2)

    def set_modes_constant_variables_average_ssp_sps(self, found_modes_ssp, found_modes_sps):
        print("gamma handeling:")
        for i in range(len(self.found_modes)):
            mode_ssp = found_modes_ssp[i]
            mode_sps = found_modes_sps[i]
            amp_xxz = mode_ssp.amplitudes[0, 0, 2]
            amp_xzx = mode_sps.amplitudes[0, 2, 0]
            print("+++" * 50)
            if abs(amp_xxz) > 10 and abs(amp_xzx) > 10:
                print("both were big enough", abs(amp_xxz), abs(amp_xzx))
                new_gamma = (mode_ssp.gamma + mode_sps.gamma) / 2
            elif abs(amp_xzx) > 10:
                print("just sps was big enough", abs(amp_xxz), abs(amp_xzx))
                new_gamma = mode_sps.gamma
            elif abs(amp_xxz) > 10:
                print("just ssp was big enough", abs(amp_xxz), abs(amp_xzx))
                new_gamma = mode_ssp.gamma
            new_omega0 = (mode_ssp.frequency + mode_sps.frequency) / 2
            print("gamma xxz:", mode_ssp.gamma, "gamma xzx:", mode_sps.gamma)
            print("new omega0:", new_omega0, ",new gamma:", new_gamma)
            print("+++" * 50)

            self.found_modes[i].frequency = new_omega0
            self.found_modes[i].gamma = new_gamma
        print("Gamma and Omega0 are set and will be fixed for next fittings")

    def show_chart(self, chart_filepath, fitted: bool):
        print("plotting experimental data")
        self.just_plotting = True
        pylab.rc('font', size=8)
        fig = pylab.figure(figsize=(12, 10))

        # create axs
        ax1 = fig.add_subplot(3, 2, 1)  # ssp mag
        ax2 = fig.add_subplot(3, 2, 2)  # ssp imag
        ax3 = fig.add_subplot(3, 2, 3)  # sps mag
        ax4 = fig.add_subplot(3, 2, 4)  # sps imag
        ax5 = fig.add_subplot(3, 2, 5)  # ppp mag
        ax6 = fig.add_subplot(3, 2, 6)  # ppp imag
        ax_list = [ax1, ax2, ax3, ax4, ax5, ax6]
        raw_data = [self.ssp_mag_list, self.ssp_im_list, self.sps_mag_list, self.sps_im_list, self.ppp_mag_list, self.ppp_im_list]
        # adding labels
        ax1.set_ylabel("$|\chi^{(2)}|^2 SSP$  /  A.U.")
        ax2.set_ylabel("Im[SSP]  /  A.U.")
        ax3.set_ylabel("$|\chi^{(2)}|^2 SPS$  /  A.U.")
        ax4.set_ylabel("Im[SPS]  /  A.U.")
        ax5.set_ylabel("$|\chi^{(2)}|^2 PPP$  /  A.U.")
        ax6.set_ylabel("Im[PPP]  /  A.U.")
        for ax in ax_list:
            ax.set_xlabel('IR wavenumber / cm$^{-1}$')
        # adding DFT advices
        # scaling_factor = float(input("enter the omega0 scaling factor: "))
        # DFT_scaling_factor = 1.04 * scaling_factor
        DFT_scaling_factor = 1
        for ax in ax_list:
            number_of_modes = len(self.original_modes)
            for i in range(number_of_modes - 1):
                ax.axvline(self.original_modes[i].frequency * DFT_scaling_factor, color="black", ls="dotted")
            ax.axvline(self.original_modes[-1].frequency * DFT_scaling_factor, color="black", ls="dotted", label="DFT $\omega_0$")

        # draw the noisy part
        for ax, noisy_data in zip(ax_list, raw_data):
            ax.plot(self.freq_list, noisy_data, label="exp")

        if not fitted:  # it is before fitting and we need to smooth the data
            spectra_list = [self.ssp_mag_list, self.ssp_im_list, self.sps_mag_list, self.sps_im_list, self.ppp_mag_list, self.ppp_im_list]
            med_filtered_list = []
            savgol_filtered_list = []
            all_peak_positions = []
            for spectrum in spectra_list:
                med_filtered_spectrum = medfilt(spectrum, kernel_size=9)
                med_filtered_list.append(med_filtered_spectrum)
                savgol_filtered_spectrum = savgol_filter(med_filtered_spectrum, 11, 2)
                savgol_filtered_list.append(savgol_filtered_spectrum)
                peak_inds_for_spectrum = find_peaks(savgol_filtered_spectrum)[0]
                all_peak_positions.append(self.freq_list[peak_inds_for_spectrum])
            for ax, peak_positions in zip(ax_list, all_peak_positions):
                for i in range(len(peak_positions) - 1):
                    ax.axvline(peak_positions[i], c="red", ls="dotted")
                ax.axvline(peak_positions[-1], c="red", ls="dotted", label="smoothing peak detection")

            for ax, filtered_spectrum, savgol_spectrum in zip(ax_list, med_filtered_list, savgol_filtered_list):
                ax.plot(self.freq_list, filtered_spectrum, label="med filtered")
                ax.plot(self.freq_list, savgol_spectrum, label="savgol filtered")


        else:
            fitting_ax_list = [ax1, ax3, ax5]
            fitted_data = [self.model_ssp_vectorized(self.freq_list), np.vectorize(self.model_sps)(self.freq_list), np.vectorize(self.model_ppp)(self.freq_list)]
            for ax, fitted_datum in zip(fitting_ax_list, fitted_data):
                ax.plot(self.freq_list, fitted_datum, c="red", label="fitting")
                ax.set_ylim(bottom=0)
                for i in range(len(self.found_modes) - 1):
                    ax.axvline(self.found_modes[i].frequency, color="olive", ls="dotted")
                ax.axvline(self.found_modes[-1].frequency, color="olive", ls="dotted", label="found $\omega_0$")

        for ax in ax_list:
            ax.legend()
        fig.set_tight_layout(True)
        pylab.savefig(chart_filepath, dpi=800)
        plt.close(fig)
        self.just_plotting = False

    def show_mode_lorentzian_info(self):
        print("Found from fitting information in the frequency range {} - {}".format(self.freq_list[0], self.freq_list[-1]).center(100, "-"))
        print("Found NR chi2 xxz", self.found_NR_chi2_xxz)
        print("Found NR chi2 xzx", self.found_NR_chi2_xzx)
        print("Found NR chi2 zzz", self.found_NR_chi2_zzz)
        table = []
        headers = ["mode", "wavenumber cm-1", "width cm-1", "amp_xxz", "amp_xzx", "amp_zzz"]
        for i, mode in enumerate(self.found_modes):
            table.append([i, mode.frequency, mode.gamma, mode.amplitudes[0, 0, 2], mode.amplitudes[0, 2, 0], mode.amplitudes[2, 2, 2]])
        print(tabulate(table, headers))
