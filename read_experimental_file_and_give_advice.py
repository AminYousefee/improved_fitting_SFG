

from ProjectStructure import Curve_Fitter, Molecule, ODF, Lab, Spectrum
import numpy as np

def func():
    alanine = Molecule("alanine", use_preset_molecular_frame_coordinates=True)
    gaussian_filepath = "./gaussian_output/{}.log".format(alanine.name)
    alanine.parse_gaussian_file(gaussian_filepath)
    alanine.print_parsed_data()
    lab = Lab(n1=1, n2=1.5, theta1_degree_sfg=65, theta1_degree_IR=65, theta1_degree_vis=65)
    theta0 = np.deg2rad(35)
    odf_parameters = {"type": "gaussian", "theta": {"theta0": theta0, "sigma": 50}, "phi": None, "psi": None}
    odf = ODF(parameters=odf_parameters)
    alanine_spectrum2 = Spectrum(molecule=alanine, lab=lab, odf=odf, freq_range_start=2800, freq_range_end=3305, freq_range_step=5, list_of_nr_chi2=[2] * 3)

    base_directory = "./results/alanine/experimental/noisy/not_normalized/test2/"

    filepath_mag1 = "./results/alanine/experimental/noisy/not_normalized/alanine_mag_squared.txt"
    filepath_imag1 = "./results/alanine/experimental/noisy/not_normalized/alanine_imag.txt"
    curve_fitter1 = Curve_Fitter(lab=lab, molecule=alanine, filepath_mag=filepath_mag1, filepath_imag=filepath_imag1)  # scenario one: not normalized

    # I need a figure contains 6 spectra, in each one the blue lineshape shows the noisy, the orange shows the smoothed and suggested peaks from smoothing, the black dashed lines that shows the DFT after the sclaing factors
    first_advice_chart_filepath = base_directory + "01_raw_data_smoothed_DFT_suggestions.png"
    curve_fitter1.show_chart(first_advice_chart_filepath, fitted=False)
    # the smoothing is just to find the scaling factor. for this my suggestion is

    # -----------------    SSP    ------------------- JUST AMP AND NR CHI2 -------------------------------------------------------------
    # initial_guess_ssp_1 = [1, 100, 0, -100, 0]
    # bounds_ssp_find_gamma = [(0, 4), (0, 100), (-100, 0), (-100, 0), (0, 100)]
    # -------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- AMP, GAMMA, AND NR CHI2 ------------------------------------------------------
    # bounds_ssp_find_gamma = [(0, 4), (0, 100), (4, 15), (-100, 0), (4, 15), (-100, 0), (4, 15), (0, 100), (4, 15)]
    # initial_guess_ssp_1 = [(item[0] + item[1])/2.0 for item in bounds_ssp_find_gamma]
    # -------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- AMP, GAMMA, OMEGA_0, AND NR CHI2 ----------------------------------------------------
    bounds_ssp_find_gamma = [(0, 4), (0, 100), (4, 15), (3000, 3042), (-100, 0), (4, 15), (3042, 3076), (-100, 0), (4, 15), (3076, 3113), (0, 100), (4, 15), (3113, 3140)]
    initial_guess_ssp_1 = [(item[0] + item[1]) / 2.0 for item in bounds_ssp_find_gamma]
    # =====================================================================================================================================
    ssp_plotting_filepath_first = base_directory + "02_ssp_finding_gamma.png"
    NR_xxz, found_modes_ssp, squared_error_ssp = curve_fitter1.fit_ssp(initial_guess_ssp_1, bounds_ssp_find_gamma, ssp_plotting_filepath_first, number_of_modes=4)
    initial_guess_ssp_second_time = [NR_xxz]
    for mode in found_modes_ssp:
        initial_guess_ssp_second_time.append(mode.amplitudes[0, 0, 2])
    sps_plotting_filepath_first = base_directory + "02_sps_finding_gamma.png"
    # -----------------    SPS    ------------------- JUST AMP AND NR CHI2 -------------------------------------------------------------
    # initial_guess_sps_1 = [1, 0, 0, 0, 0]
    # bounds_sps_1 = [(0, 2), (-100, 0), (-100, 0), (-100, 0), (-100, 0)]
    # -------------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- AMP, GAMMA, AND NR CHI2 ------------------------------------------------------
    # bounds_sps_1 = [(0, 4), (-100, 0), (4, 15), (-100, 0), (4, 15), (-100, 0), (4, 15), (-100, 0), (4, 15)]
    # initial_guess_sps_1 = [(item[0] + item[1])/2.0 for item in bounds_sps_1]
    # -------------------------------------------------------------------------------------------------------------------------
    # --------------------------------------------- AMP, GAMMA, OMEGA_0, AND NR CHI2 ----------------------------------------------------
    bounds_sps_1 = [(0, 4), (-100, 0), (4, 15), (3000, 3042), (-100, 0), (4, 15), (3042, 3076), (-0.1, 0), (4, 15), (3076, 3113), (-100, 0), (4, 15), (3113, 3140)]
    initial_guess_sps_1 = [(item[0] + item[1]) / 2.0 for item in bounds_sps_1]
    # =====================================================================================================================================
    NR_xzx, found_modes_sps, squared_error_sps = curve_fitter1.fit_sps(initial_guess_sps_1, bounds_sps_1, sps_plotting_filepath_first, number_of_modes=4)
    initial_guess_sps_second_time = [NR_xzx]
    for mode in found_modes_sps:
        initial_guess_sps_second_time.append(mode.amplitudes[0, 2, 0])
    print("SSP Fitting results:", NR_xxz, [(mode.frequency, mode.gamma, mode.amplitudes[0, 0, 2]) for mode in found_modes_ssp])
    print("SPS Fitting results:", NR_xzx, [(mode.frequency, mode.gamma, mode.amplitudes[0, 2, 0]) for mode in found_modes_sps])
    # print(initial_guess_sps_second_time)

    curve_fitter1.set_modes_constant_variables_average_ssp_sps(found_modes_ssp, found_modes_sps)

    bounds_ssp = [(0, 4), (0, 100), (-100, 0), (-100, 0), (0, 100)]
    ssp_plotting_filepath_second = base_directory + "04_ssp_gamma_fixed_second_fitting.png"
    NR_xxz_second, found_modes_ssp_second, squared_error_ssp = curve_fitter1.fit_ssp(initial_guess_ssp_second_time, bounds_ssp, ssp_plotting_filepath_second, number_of_modes=4)

    bounds_sps = [(0, 4), (-100, 0), (-100, 0), (-100, 0), (-100, 0)]
    sps_plotting_filepath_second = base_directory + "05_sps_gamma_fixed_second_fitting.png"
    NR_xzx_second, found_modes_sps_second, squared_error_sps = curve_fitter1.fit_sps(initial_guess_sps_second_time, bounds_sps, sps_plotting_filepath_second, number_of_modes=4)
    print("second SSP Fitting results:", NR_xxz_second, [(mode.frequency, mode.gamma, mode.amplitudes[0, 0, 2]) for mode in found_modes_ssp_second])
    print("second SPS Fitting results:", NR_xzx_second, [(mode.frequency, mode.gamma, mode.amplitudes[0, 2, 0]) for mode in found_modes_sps_second])

    bounds_ppp = [(0, 4), (0, 100), (-100, 0), (-100, 0), (-100, 0)]
    initial_guess_ppp = [(item[0] + item[1]) / 2.0 for item in bounds_ppp]
    ppp_plotting_filepath = base_directory + "05ppp_gamma_fixed_fitting.png"
    print("  THIS IS THE PPP TURN  ".center(100, "#"))
    NR_zzz, found_modes_ppp, squared_error_ppp = curve_fitter1.fit_ppp(initial_guess_ppp, bounds_ppp, ppp_plotting_filepath, number_of_modes=4)
    print("PPP Fitting results:", NR_zzz, [(mode.frequency, mode.gamma, mode.amplitudes[2][2][2]) for mode in found_modes_ppp])
    comprehensive_Chart_filepath = base_directory + "06comprehensive_chart.png"
    curve_fitter1.show_chart(chart_filepath=comprehensive_Chart_filepath, fitted=True)
    alanine_spectrum2.show_real_mode_lorentzian_info()
    curve_fitter1.show_mode_lorentzian_info()

if __name__ == "__main__":
    func()