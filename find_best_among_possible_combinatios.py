import os

from matplotlib import pyplot as plt
from tabulate import tabulate

from ProjectStructure import Molecule, Calculation, ODF, Lab, Curve_Fitter, Spectrum
from pprint import pprint
import numpy as np


def create_bounds(NR, gamma, comb, number_of_modes, list_of_omega0_bounds, first_time):
    bounds = [NR]
    for mode_ind in range(number_of_modes):
        if comb[mode_ind] == "+":
            bounds.append((0, 100))
        else:
            bounds.append((-100, 0))
        if first_time:
            bounds.append(gamma)
            bounds.append(list_of_omega0_bounds[mode_ind])
    return bounds


def possibilities_from_table(table):
    all_combinations_for_table = set()
    for row_ind in range(len(table)):
        row = table[row_ind]
        combination = []
        for amp in row:
            val = "+" if amp > 0 else "-"  # in here I can add another condition to see whether it's less than 10 or not then give it a ?
            combination.append(val)
        all_combinations_for_table.add(tuple(combination))
    return list(all_combinations_for_table)


def find_iterative_pattern(possible_combinations):
    arr = np.array(possible_combinations).T
    always_keep_sign = []
    for i in range(len(arr)):
        mode_signs = arr[i]
        do_mode_has_the_same_sign = all(sign == mode_signs[0] for sign in mode_signs)
        if do_mode_has_the_same_sign:
            print("mode", i + 1, "is always", mode_signs[0])
            always_keep_sign.append((i, mode_signs))
    return always_keep_sign


def molecule_possibilities(molecule, freq_start, freq_end):
    theta0_list_in_rad = np.deg2rad(list(range(0, 90, 5)))
    list_of_modes = [mode for mode in molecule.list_of_modes if freq_start < mode.frequency < freq_end]
    odf_list = []
    for i in range(len(theta0_list_in_rad)):
        theta0 = theta0_list_in_rad[i]
        odf_parameters = {"type": "delta_dirac", "theta": {"theta0": theta0, "sigma": None}, "phi": None, "psi": None}
        odf = ODF(parameters=odf_parameters)
        odf_list.append(odf)

    amp_tables = molecule.create_amp_guide_table(list_of_modes, odf_list, len(list_of_modes), len(theta0_list_in_rad))
    xxz_table = amp_tables[0]
    xzx_table = amp_tables[1]
    zzz_table = amp_tables[2]
    combinations = set(Calculation.give_sign_combinations(len(list_of_modes)))
    # I want to work from 0 to 90 degree for tilt row_ind
    # then based on the table the first mode sign should be +

    # detecting possibilities in the sign table
    print("XXZ table")
    print(tabulate(xxz_table))
    print("possible combinations for xxz")
    possible_combinations_xxz = possibilities_from_table(xxz_table)
    find_iterative_pattern(possible_combinations_xxz)
    pprint(possible_combinations_xxz)
    # ------------------------------------------
    print("XZX table")
    print(tabulate(xzx_table))
    print("possible combinations for xzx")
    possible_combinations_xzx = possibilities_from_table(xzx_table)
    pprint(possible_combinations_xzx)
    find_iterative_pattern(possible_combinations_xzx)
    # ----------------------------------
    print("ZZZ table")
    print(tabulate(zzz_table))
    print("possible combinations for zzz")
    possible_combinations_zzz = possibilities_from_table(zzz_table)
    pprint(possible_combinations_zzz)
    find_iterative_pattern(possible_combinations_zzz)

    impossible_xxz = combinations - set(possible_combinations_xxz)
    impossible_xzx = combinations - set(possible_combinations_xzx)
    impossible_zzz = combinations - set(possible_combinations_zzz)
    print("impossible situations")
    pprint(impossible_xxz)
    print("impossible xzx")
    pprint(impossible_xzx)
    print("impossible zzz")
    pprint(impossible_zzz)
    return [possible_combinations_xxz, possible_combinations_xzx, possible_combinations_zzz]


def fit_combinations():
    alanine = Molecule("alanine", use_preset_molecular_frame_coordinates=True)
    gaussian_filepath = "./gaussian_output/{}.log".format(alanine.name)
    alanine.parse_gaussian_file(gaussian_filepath)
    alanine.print_parsed_data()
    lab = Lab(n1=1, n2=1.5, theta1_degree_sfg=65, theta1_degree_IR=65, theta1_degree_vis=65)
    theta0 = np.deg2rad(35)
    odf_parameters = {"type": "delta_dirac", "theta": {"theta0": theta0, "sigma": None}, "phi": None, "psi": None}
    odf = ODF(parameters=odf_parameters)
    alanine_spectrum2 = Spectrum(molecule=alanine, lab=lab, odf=odf, freq_range_start=2800, freq_range_end=3305, freq_range_step=5, list_of_nr_chi2=[2] * 3)
    alanine_spectrum2.calculate_suceptatability(extra_mode=False)
    filepath_mag1 = "./results/alanine/experimental/noisy/not_normalized/alanine_mag_squared.txt"
    filepath_imag1 = "./results/alanine/experimental/noisy/not_normalized/alanine_imag.txt"
    curve_fitter1 = Curve_Fitter(lab=lab, molecule=alanine, filepath_mag=filepath_mag1, filepath_imag=filepath_imag1)
    base_directory = "./results/alanine/experimental/noisy/not_normalized/test/"
    first_advice_chart_filepath = base_directory + "0000_raw_data_smoothed_peak_suggestions.png"
    curve_fitter1.show_chart(first_advice_chart_filepath, fitted=False)
    bound_NR_chi2 = (0, 4)
    gamma = (5, 15)
    omega0_bounds = [(3000, 3042), (3042, 3076), (3076, 3113), (3113, 3140)]
    if len(omega0_bounds) != len(curve_fitter1.found_modes):
        raise Exception("number of omega not is not equal to number of modes which is {]".format(len(curve_fitter1.found_modes)))
    possible_combinations_xxz, possible_combinations_xzx, possible_combinations_zzz = molecule_possibilities(alanine, curve_fitter1.freq_list[0], curve_fitter1.freq_list[-1])
    squared_error_summation_ssp_sps_ppp = dict()
    for i in range(len(possible_combinations_xxz)):
        for j in range(len(possible_combinations_xzx)):
            for k in range(len(possible_combinations_zzz)):
                os.makedirs(base_directory + "{}_{}_{}".format(i, j, k))
                squared_error_summation_ssp_sps_ppp["{}_{}_{}".format(i, j, k)] = 0

    for xxz_ind, xxz_comb in enumerate(possible_combinations_xxz):

        print("xxz combination", xxz_ind, xxz_comb)
        bounds_ssp1 = create_bounds(bound_NR_chi2, gamma, xxz_comb, len(curve_fitter1.found_modes), omega0_bounds, True)
        initial_guess_ssp1 = [(item[0] + item[1]) / 2.0 for item in bounds_ssp1]
        print("initial guess ssp 1", initial_guess_ssp1)

        NR_xxz, found_modes_ssp, squared_error_ssp1, fig_ssp_1 = curve_fitter1.fit_ssp(initial_guess_ssp1, bounds_ssp1, number_of_modes=len(curve_fitter1.found_modes))
        print("SSP Fitting results:", NR_xxz, [(mode.frequency, mode.gamma, mode.amplitudes[0, 0, 2]) for mode in found_modes_ssp])
        for xzx_ind, xzx_comb in enumerate(possible_combinations_xzx):
            print("xzx combination", xzx_ind, xzx_comb)
            bounds_sps1 = create_bounds(bound_NR_chi2, gamma, xzx_comb, len(curve_fitter1.found_modes), omega0_bounds, True)
            bounds_ssp2 = create_bounds(bound_NR_chi2, gamma, xxz_comb, len(curve_fitter1.found_modes), omega0_bounds, False)
            bounds_sps2 = create_bounds(bound_NR_chi2, gamma, xzx_comb, len(curve_fitter1.found_modes), omega0_bounds, False)

            initial_guess_sps1 = [(item[0] + item[1]) / 2.0 for item in bounds_sps1]
            print("initial guess sps 1", initial_guess_sps1)

            NR_xzx, found_modes_sps, squared_error_sps1, fig_sps1 = curve_fitter1.fit_sps(initial_guess_sps1, bounds_sps1, number_of_modes=len(curve_fitter1.found_modes))
            initial_guess_ssp2 = [NR_xxz]
            initial_guess_sps2 = [NR_xzx]
            for i in range(len(found_modes_ssp)):
                initial_guess_ssp2.append(found_modes_ssp[i].amplitudes[0, 0, 2])
                initial_guess_sps2.append(found_modes_sps[i].amplitudes[0, 2, 0])

            print("SPS Fitting results:", NR_xzx, [(mode.frequency, mode.gamma, mode.amplitudes[0, 2, 0]) for mode in found_modes_sps])
            curve_fitter1.set_modes_constant_variables_average_ssp_sps(found_modes_ssp, found_modes_sps)
            NR_xxz2, found_modes_ssp2, squared_error_ssp2, fig_ssp_2 = curve_fitter1.fit_ssp(initial_guess_ssp2, bounds_ssp2, number_of_modes=4)
            NR_xzx2, found_modes_sps2, squared_error_sps2, fig_sps_2 = curve_fitter1.fit_sps(initial_guess_sps2, bounds_sps2, number_of_modes=4)
            print("second SSP Fitting results:", NR_xxz2, [(mode.frequency, mode.gamma, mode.amplitudes[0, 0, 2]) for mode in found_modes_ssp2])
            print("second SPS Fitting results:", NR_xzx2, [(mode.frequency, mode.gamma, mode.amplitudes[0, 2, 0]) for mode in found_modes_sps2])
            for zzz_ind, zzz_comb in enumerate(possible_combinations_zzz):
                bounds_ppp = create_bounds(bound_NR_chi2, gamma, zzz_comb, len(curve_fitter1.found_modes), omega0_bounds, False)
                initial_guess_ppp = [(item[0] + item[1]) / 2.0 for item in bounds_ppp]

                NR_zzz, found_modes_ppp, squared_error_ppp, fig_ppp = curve_fitter1.fit_ppp(initial_guess_ppp, bounds_ppp, number_of_modes=4)
                print("PPP Fitting results:", NR_zzz, [(mode.frequency, mode.gamma, mode.amplitudes[2, 2, 2]) for mode in found_modes_ppp])
                comprehensive_Chart_filepath = base_directory+"{}_{}_{}/".format(xxz_ind, xzx_ind, zzz_ind) + "6comprehensive_chart.png"
                curve_fitter1.show_chart(chart_filepath=comprehensive_Chart_filepath, fitted=True)
                alanine_spectrum2.show_real_mode_lorentzian_info()
                curve_fitter1.show_mode_lorentzian_info()
                squared_error_summation_ssp_sps_ppp["{}_{}_{}".format(xxz_ind, xzx_ind, zzz_ind)] = squared_error_ssp2 + squared_error_sps2 + squared_error_ppp
                dpi = 600
                ssp_plotting_filepath_1 = base_directory + "{}_{}_{}/".format(xxz_ind, xzx_ind, zzz_ind) + "1_ssp1.png"
                sps_plotting_filepath_1 = base_directory + "{}_{}_{}/".format(xxz_ind, xzx_ind, zzz_ind) + "2_sps1.png"
                ssp_plotting_filepath_2 = base_directory + "{}_{}_{}/".format(xxz_ind, xzx_ind, zzz_ind) + "3_ssp2.png"
                sps_plotting_filepath_2 = base_directory + "{}_{}_{}/".format(xxz_ind, xzx_ind, zzz_ind) + "4_sps2.png"
                ppp_plotting_filepath = base_directory + "{}_{}_{}/".format(xxz_ind, xzx_ind, zzz_ind) + "5_ppp.png"
                curve_fitter1.plot_single_fitting(ssp_plotting_filepath_1, fig_ssp_1, dpi)
                curve_fitter1.plot_single_fitting(sps_plotting_filepath_1, fig_sps1, dpi)
                curve_fitter1.plot_single_fitting(ssp_plotting_filepath_2, fig_ssp_2, dpi)
                curve_fitter1.plot_single_fitting(sps_plotting_filepath_2, fig_ssp_2, dpi)
                curve_fitter1.plot_single_fitting(ppp_plotting_filepath, fig_ppp, dpi)
    min_key = min(squared_error_summation_ssp_sps_ppp, key=squared_error_summation_ssp_sps_ppp.get)
    print(min_key)
    print("residual summation is", squared_error_summation_ssp_sps_ppp[min_key])
    for i in range(len(possible_combinations_xxz)):
        for j in range(len(possible_combinations_xzx)):
            for k in range(len(possible_combinations_zzz)):
                print(i, j, k, possible_combinations_xxz[i], possible_combinations_xzx[j], possible_combinations_zzz[k], "squared error summation all polarizations:", squared_error_summation_ssp_sps_ppp["{}_{}_{}".format(i, j, k)])


if __name__ == "__main__":
    fit_combinations()
# talking to Dennis:
"""
different molecule
different combinations for different element of chi2
how can we determine the impossibilites?
how I can consider the relative value (I mean magnitude) of amplitudes in combinations
to the next step I can determine the 
we can create the sign table as a cube with different sigma in it. at each sigma we have a table. we are going to eleiminate the rows with too small amplitudes but they may not vanished in all possibilites because they may be big in other sigmas
"""
