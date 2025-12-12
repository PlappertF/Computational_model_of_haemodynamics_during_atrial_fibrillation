import numpy as np
from dataclasses import dataclass
import src.heart as heart
from numba import njit
import time


def init(model, model_pars):

    # Import model parameters and assign to model class
    model.import_pars(model_pars)

    # Set compartment names
    model.compartments = {'Vp': 0, 'LA': 1, 'LV': 2, 'As': 3, ' Vs': 4, 'RA': 5, 'RV': 6, 'Ap': 7}
    model.compartment_names = list(model.compartments.keys())


def initialize_multipatch(model, fs, patch_t_delay, n_patches_atria):
    # Solver
    model.solver.dt = 1000.0 / fs  # [ms]

    # Check for the special case where we have a total of 5 patches. This case is only used if we use one patch per heart
    # wall. Up to this writing, this case only has been used for the sensitivity analysis. In this case, we don't need to
    # reformat the model parameters. In this case, we need to correct the patch_t_delay.
    if model.heart.patches.size == 5:
        patch_t_delay = np.array([patch_t_delay[0], patch_t_delay[0], patch_t_delay[0], patch_t_delay[-1],
                                  patch_t_delay[-1]])
    # Initialize the Activation dataclass and add the patch_t_delay parameter to the dataclass. This dataclass will be
    # filled with more parameters in the initialize_activations function.
    @dataclass
    class Activation:
        patch_t_delay: np.ndarray
    model.activation = Activation(patch_t_delay=patch_t_delay)

    # Initialize the Termination dataclass. This dataclass will store the information whether the simulation was terminated
    # or not. It will also store a termination message to keep track of the reason why the simulation was terminated.
    @dataclass
    class Termination:
        terminated: bool
        message: str
    model.termination = Termination(terminated=False, message='')

    # In the special case where we have a total of 5 patches, we don't need to reformat the model parameters.
    if model.heart.patches.size > 5:
        # Reformat the following model parameters to increase the number of atrial patches if n_patches_atria > 1. The
        # code does effectively nothing if n_patches_atria = 1.
        reformat_mask = np.arange(21 + 2 * n_patches_atria)
        reformat_mask[21:(21 + n_patches_atria)] = 21
        reformat_mask[(21 + n_patches_atria):] = 22
        # The following parameters for the atrial patches are duplicated.
        model.heart.c_1 = model.heart.c_1[reformat_mask]
        model.heart.c_3 = model.heart.c_3[reformat_mask]
        model.heart.c_4 = model.heart.c_4[reformat_mask]
        model.heart.ls_eiso = model.heart.ls_eiso[reformat_mask]
        model.heart.ls_ref = model.heart.ls_ref[reformat_mask]
        model.heart.lsc_0 = model.heart.lsc_0[reformat_mask]
        model.heart.sf_act = model.heart.sf_act[reformat_mask]
        model.heart.t_ad = model.heart.t_ad[reformat_mask]
        model.heart.tau_d = model.heart.tau_d[reformat_mask]
        model.heart.tau_r = model.heart.tau_r[reformat_mask]
        model.heart.v_max = model.heart.v_max[reformat_mask]
        model.activation.patch_t_delay = model.activation.patch_t_delay[reformat_mask]
        model.heart.patches = model.heart.patches[reformat_mask]
        # The parameter of one atrial patch is divided by the number of patches in the atria
        model.heart.vw[21:23] = model.heart.vw[21:23] / n_patches_atria
        model.heart.vw = model.heart.vw[reformat_mask]
        model.heart.am_ref[21:23] = model.heart.am_ref[21:23] / n_patches_atria
        model.heart.am_ref = model.heart.am_ref[reformat_mask]

    model.heart.tr = model.heart.tau_r * model.heart.t_ad
    # Other implementation of td used in Oomen 2021 that assumes repolarization is not length-dependent was not
    # copied into this code. See older version of this implementation of td. Assume repolarization is
    # length-dependent - used in Walmsley 2015
    model.heart.td = model.heart.tau_d * model.heart.t_ad

    model.heart.n_patches_tot = model.heart.patches.size
    # Find ventricular and atrial patches
    model.heart.i_ventricles = np.logical_or.reduce((model.heart.patches == 0, model.heart.patches == 1,
                                                     model.heart.patches == 2))
    model.heart.i_atria = np.logical_or(model.heart.patches == 3, model.heart.patches == 4)

    # Calculate total heart wall volume and midwall reference area
    heart.set_total_wall_volumes_areas(model)

    # Compute reference volume of heart for pericardial mechanics computations [mm^3]
    model.heart.v_tot_0 = heart.unloaded_heart_volume(model.heart.am_ref_w, model.heart.vw_w)

    model.heart.lsc = model.heart.ls_ref - model.heart.ls_eiso
    model.heart.c = np.zeros(model.heart.n_patches_tot)


def initialize_batch(model, rhythm, batch_size):
    # First, we compute the maximum number of possible iterations. With that number, we may reduce the 'batch_size' to
    # avoid unnecessary overhead. There is no need to set up matrices with millions of rows if we only simulate a few
    # iterations. It is important to note, that the simulation result is the same for any batch size. Setting the
    # maximum number of iterations is just an attempt to make the simulation faster by reducing unnecessary overhead.
    # If the batch size is higher than the number of iterations in the simulation, there will be extra rows filled with
    # zeros that are cut off at the end of the simulation. The simulation could have been faster if we hadn't needed to
    # declare extra space for these rows in the memory. If the batch size is lower than the number of iterations in
    # the simulation, the simulation will save the results of the batch to a file and overwrite the rows of the declared
    # simulation parameters. This is the exact functionality why I implemented the batch in the first place to keep
    # the memory usage low.
    # In normal sinus rhythm, we can exactly calculate the maximum number of iterations, because the RR interval between
    # each beat is fixed. In atrial fibrillation, the RR interval is drawn from a stochastic distribution, therefore we
    # can't exactly calculate the maximum number of iterations and need to use a conservative estimate.
    if rhythm == 0:  # rhythm == 'NSR'
        # The maximum number of iterations is given by the rounded up amount of heart beats times the RR interval divided
        # by the time step. The maximum number of heart beats is given by the maximum number of allowed heart beats in
        # the NSR launch phase plus the number of desired heart beats in the real simulation.
        # If we load a converged solution, then the NSR launch phase is only one beat long.
        if model.activation.use_converged:
            max_batch_size = int(np.ceil((1 + model.activation.num_beats) *
                                         model.activation.rr_char[0] / model.solver.dt))
        else:
            max_batch_size = int(np.ceil((model.solver.nsr_launch_iter_max + model.activation.num_beats) *
                                         model.activation.rr_char[0] / model.solver.dt))
    else:  # rhythm == 'AF'
        # During atrial fibrillation, it is impossible to know how many iterations the simulation will take at most. I am
        # choosing therefore an arbitrary maximum batch_size number using double the average RR interval.
        # If we load a converged solution, then the NSR launch phase is only one beat long.
        if model.activation.use_converged:
            max_batch_size = int(np.ceil((1 + model.activation.num_beats_af_launch + model.activation.num_beats) *
                                         2 * model.activation.rr_char[0] / model.solver.dt))
        else:
            max_batch_size = int(np.ceil((model.solver.nsr_launch_iter_max + model.activation.num_beats_af_launch +
                                          model.activation.num_beats) * 2 * model.activation.rr_char[
                                             0] / model.solver.dt))
    batch_size = min(batch_size, max_batch_size)

    # We declare the shape of a few parameters for which we store the values for each model iteration. The shape is
    # determined by the batch_size representing the number of iterations that are kept in the memory before the whole
    # batch is written to an output file and the memory is freed.
    # Initialize volumes and pressures for the four heart chambers and four peripheral compartments
    model.volumes = np.zeros((batch_size, 8))
    if ~np.isclose(sum(model.circulation.k), 1.0):
        raise ValueError("The sum of the volume ratio k must be 1.0.")
    model.volumes[0, :] = model.circulation.k * model.circulation.sbv / np.sum(model.circulation.k)
    model.pressures = np.zeros((batch_size, 8))

    # Set walls
    model.heart.rm = np.zeros((batch_size, 5))  # For each wall, ventricles and atria
    model.heart.xm = np.zeros((batch_size, 3))  # Only for ventricles
    model.heart.ys_store = np.zeros(batch_size)  # Septal height

    # Pericardium
    model.pericardium.lab_f = np.zeros(batch_size)
    model.pericardium.pressure = np.zeros(batch_size)

    # Solver
    model.solver.batch_size = batch_size
    model.solver.batch_inc = 0

    # Activation
    model.activation.phases = np.zeros(batch_size)
    model.activation.time = np.zeros(batch_size)

    model.heart.lab_f = np.ones((batch_size, model.heart.n_patches_tot))
    model.heart.sig_f = np.zeros((batch_size, model.heart.n_patches_tot))
    model.contractility = np.zeros((batch_size, model.heart.n_patches_tot))
    model.sarcomere_length = np.zeros((batch_size, model.heart.n_patches_tot))
    model.tension = np.zeros((batch_size, 5))

    # Get initial guesses for Vs and Ys based on LV volume at t=0 and ventricular geometry
    heart.guess_vs_ys(model)
    model.heart.dv = 0.01 * model.heart.vs
    model.heart.dy = 0.01 * model.heart.ys


def format_input_params(rhythm, fs, desired_rr_char, desired_vat_series, patch_t_delay, n_patches_atria, act_sigma_atria,
                        act_sigma_ventr, keep_simulation_data):
    """
    Here, we format the input parameters, either to speed up the computation or to enforce that numbers are floats
    instead of integers.
    """
    # We change the rhythm to a number, because comparisons with strings are slower than with numbers.
    if rhythm == 'NSR':
        rhythm = 0
    else:  # rhythm == 'AF'
        rhythm = 1

    # Make sure that the sampling frequency is a float.
    fs = float(fs)

    # Make sure that the desired RR characteristics are a (3,) ndarray with floats, and that the RR mean/ HR is written
    # as the RR mean in milliseconds.
    if isinstance(desired_rr_char, int) or isinstance(desired_rr_char, np.int64) or isinstance(desired_rr_char, float):
        desired_rr_char = float(desired_rr_char)
        if 12 <= desired_rr_char <= 200:
            desired_rr_char = np.array([60000 / desired_rr_char, 0, 0])
        else:  # 250 <= desired_rr_char <= 5000
            desired_rr_char = np.array([desired_rr_char, 0, 0])
    elif isinstance(desired_rr_char, np.ndarray):  # desired_rr_char is a (3,) ndarray
        desired_rr_char = desired_rr_char.astype(np.float64)  # Making sure the numpy array contains float and not int.

    # Make sure that the desired VAT series is a (n,) ndarray with floats.
    if isinstance(desired_vat_series, int) or isinstance(desired_vat_series, float):
        desired_vat_series = np.array([desired_vat_series], dtype=np.float64)
    elif isinstance(desired_vat_series, np.ndarray):
        desired_vat_series = desired_vat_series.astype(np.float64)

    # Make sure that the act_sigma_atria is a float.
    act_sigma_atria = float(act_sigma_atria)

    # Make sure that the act_sigma_ventr is a float.
    act_sigma_ventr = float(act_sigma_ventr)

    # We change the 'keep_simulation_data' to a number, because comparisons with strings are slower than with numbers.
    if keep_simulation_data == 'all':
        keep_simulation_data = 0
    elif keep_simulation_data == 'no_launch':
        keep_simulation_data = 1

    # For the special case, where the delay of both atria is set to infinite, we make sure that we only set up the
    # cardiogrowth model with one patch per atria, because the atria will not be activated throughout the simulation,
    # making the simulation of several atrial patches unnecessary.
    if n_patches_atria > 1 and np.all(np.isinf(patch_t_delay[21:23])):
        n_patches_atria = 1

    return (rhythm, fs, desired_rr_char, desired_vat_series, patch_t_delay, n_patches_atria, act_sigma_atria,
            act_sigma_ventr, keep_simulation_data)
