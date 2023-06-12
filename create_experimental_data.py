from ProjectStructure import Curve_Fitter, Molecule, ODF, Lab, Spectrum
import numpy as np

def func():

    # ----------------------------- READING THE MOLECULE INFORMATION ------------------------------------------
    alanine = Molecule("alanine", use_preset_molecular_frame_coordinates=True)
    gaussian_filepath = "./gaussian_output/{}.log".format(alanine.name)
    alanine.parse_gaussian_file(gaussian_filepath)
    alanine.print_parsed_data()
    relative_sign_tables_filepath = "./results/alanine/relative_sign_tables.png"

    theta0 = np.deg2rad(35)
    lab = Lab(n1=1, n2=1.5, theta1_degree_sfg=65, theta1_degree_IR=65, theta1_degree_vis=65)
    odf_parameters = {"type": "dirac", "theta": {"theta0": theta0, "sigma": 50}, "phi": None, "psi": None}
    odf = ODF(parameters=odf_parameters)
    # ----------------------------- CREATING EXPERIMENTAL DATA -----------------------------------
    NR = 2
    # spectrum 2 is going to be used as experiment-like spectrum which is in a specific region (CH region)
    alanine_spectrum2 = Spectrum(molecule=alanine, lab=lab, odf=odf, freq_range_start=2800, freq_range_end=3305, freq_range_step=5, list_of_nr_chi2=[NR] * 3)
    alanine_spectrum2.calculate_infrared_absorption()
    # alanine_spectrum2.show_ir_intensity_arrays()
    alanine_spectrum2.calculate_raman_intensity()
    # alanine_spectrum2.show_raman_intensity_arrays()

    alanine_spectrum2.calculate_suceptatability(extra_mode=False)
    # alanine_spectrum2.show_sfg_intensity_arrays()
    # alanine_spectrum2.show_real_mode_lorentzian_info()
    # ------------ADDING A MODE HERE--------------


    # ------------NOISE SHOULD BE AN OPTION--------------
    normalization = False
    noise = True
    all_chart_filename = "alanine_{}_degree_from_DFT.png".format(np.rad2deg(theta0))
    alanine_spectrum2.show_all_spectra_at_certain_orientaiton(show_normalized=normalization, filename=all_chart_filename, noise=0)

    mag_squared_filename = "alanine_mag_squared.txt"
    imag_filename = "alanine_imag.txt"
    alanine_spectrum2.show_real_mode_lorentzian_info()

    alanine_spectrum2.save(normalize_before_save=normalization, noise=noise, create_imag_file=True, filename_mag=mag_squared_filename, filename_imag=imag_filename)

if __name__ == "__main__":
    func()