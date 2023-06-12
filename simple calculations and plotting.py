from ProjectStructure import Curve_Fitter, Molecule, ODF, Lab, Spectrum
import numpy as np


def func():
    alanine = Molecule("alanine", use_preset_molecular_frame_coordinates=True)
    gaussian_filepath = "./gaussian_output/{}.log".format(alanine.name)
    alanine.parse_gaussian_file(gaussian_filepath)
    alanine.print_parsed_data()

    # ----------------------------- DEFINE OR REDEFINE THE MOLECULAR AXIS ------------------------------------------
    # this feature has not been implemented.
    # ----------------------------- PROVIDING TOOLS FOR THE CALCULATIONS: ODF, LAB, desired wavenumber range(start, end , step) ------------------------------------------

    theta0 = np.deg2rad(35)
    odf_parameters = {"type": "gaussian", "theta": {"theta0": theta0, "sigma": 50}, "phi": None, "psi": None}
    odf = ODF(parameters=odf_parameters)
    lab = Lab(n1=1, n2=1.5, theta1_degree_sfg=65, theta1_degree_IR=65, theta1_degree_vis=65)
    # spectrum 1 is going to be for the whole spectrum based on electronic structure calculation
    alanine_spectrum1 = Spectrum(molecule=alanine, lab=lab, odf=odf, freq_range_start=2800, freq_range_end=3300, freq_range_step=5, list_of_nr_chi2=[1e-5] * 3)
    # ----------------------------- CALCULATING IR INTENSITY ------------------------------------------
    alanine_spectrum1.calculate_infrared_absorption()
    alanine_spectrum1.show_ir_intensity_arrays()
    # ----------------------------- CALCULATING RAMAN INTENSITY ------------------------------------------
    alanine_spectrum1.calculate_raman_intensity()
    alanine_spectrum1.show_raman_intensity_arrays()

    # ----------------------------- CALCULATING SFG INTENSITY ------------------------------------------
    alanine_spectrum1.calculate_suceptatability(extra_mode=False)
    alanine_spectrum1.show_sfg_intensity_arrays()
    alanine_spectrum1.show_real_mode_lorentzian_info()
    noise = 0
    alanine_spectrum1.show_all_spectra_at_certain_orientaiton(show_normalized=False, filename="01simple_smoothed_all_charts_gaussian_sigma50.png", noise=noise)


if __name__ == "__main__":
    func()
