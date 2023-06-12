from ProjectStructure import Curve_Fitter, Molecule, ODF, Lab, Spectrum
import numpy as np


def func():
    alanine = Molecule("alanine", use_preset_molecular_frame_coordinates=True)
    gaussian_filepath = "./gaussian_output/{}.log".format(alanine.name)
    alanine.parse_gaussian_file(gaussian_filepath)
    alanine.print_parsed_data()
    lab = Lab(n1=1, n2=1.5, theta1_degree_sfg=65, theta1_degree_IR=65, theta1_degree_vis=65)

    for theta0_degree in range(0, 185, 5):
        print("theta0:", theta0_degree)
        theta0 = np.deg2rad(theta0_degree)
        odf_parameters = {"type": "delta_dirac", "theta": {"theta0": theta0, "sigma": None}, "phi": None, "psi": None}
        odf = ODF(parameters=odf_parameters)
        spectrum = Spectrum(molecule=alanine, lab=lab, odf=odf, freq_range_start=700, freq_range_end=3300, freq_range_step=1, list_of_nr_chi2=[1] * 3)
        spectrum.calculate_infrared_absorption()
        spectrum.calculate_raman_intensity()
        spectrum.calculate_suceptatability(extra_mode=False)
        all_chart_filename = "alanine_{}_degree.png".format(round(np.rad2deg(theta0)))
        spectrum.show_all_spectra_at_certain_orientaiton(show_normalized=False, filename=all_chart_filename, noise=0)

    if __name__ == "__main__":
        func()
