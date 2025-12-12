import numpy as np
import warnings
import os
import sys
from src.avn import check_av_node_parameters_overlapping

def check_input_params(rhythm, num_beats, num_beats_af_launch, num_draw_beats_at_once, fs, batch_size,
                       avn_p4_params, desired_rr_char, desired_vat_series, patch_t_delay, n_patches_atria,
                       act_sigma_atria, act_sigma_ventr, keep_simulation_data, use_converged):
    """
    Check if the input parameters have expected values and file types and raise an exception if the input is not as
    expected.
    :param rhythm: String, either 'NSR' or 'AF'
    :param num_beats: Positive integer including 0, number of heartbeats. num_beats=0 is only allowed if rhythm='NSR'.
                      More information can be found in the docstring of the 'rhythm_is_a_dancer' function.
    :param num_beats_af_launch: Number of beats (integer >= 0) during the 'launch' phase of the atrial fibrillation. More
                                information can be found in the docstring of the 'rhythm_is_a_dancer' function.
    :param num_draw_beats_at_once: Number of His-bundle activation times (integer >= 1) that are drawn at once. More
                                   information can be found in the docstring of the 'rhythm_is_a_dancer' function.
    :param fs: Integer or Float. A number below 20 and above 100000 will raise a ValueError. A number below 200 and above
               10000 will raise a warning.
    :param batch_size: Integer. A number below 1 will raise a ValueError.
    :param avn_p4_params: Parameters for the AV node model to simulate the AV nodal conduction of an AA series to the
                          ventricles as well as parameters for the Pearson Type IV distribution to generate an AA series
                          as input for the AVN model. Either None or NumPy array of dimension (21,). More information can
                          be found in the docstring of the 'rhythm_is_a_dancer' function.
    :param desired_rr_char: Desired RR series characteristics. Either None, a single int or float, or a NumPy array with
                            three integers or floats. More information can be found in the docstring of the
                            'rhythm_is_a_dancer' function.
    :param desired_vat_series: Desired ventricular activation time (VAT) series. Either None, a single int or float, or a
                               NumPy array of any length. More information can be found in the docstring of the
                               'rhythm_is_a_dancer' function.
    :param patch_t_delay: NumPy array of dimension (23,) with dtype=float64. More information can be found in the
                          docstring of the 'rhythm_is_a_dancer' function.
    :param n_patches_atria: Integer. A number below 1 will raise a ValueError. If batch_size is a float, it
                            will be rounded to the next integer.
    :param act_sigma_atria: Float. A number below 0 will raise a ValueError. Also, if rhythm='AF', act_sigma_atria has to
                            be set to 0.
    :param act_sigma_ventr: Float. A number below 0 will raise a ValueError.
    :param keep_simulation_data: String. The string must be 'all' or 'no_launch'. This function then
                                 transforms the strings into integers 0, 1, and 2. We use strings as model input because
                                 it is more intuitive to use, but we use integers within the simulation, because an
                                 integer comparison is much faster than a string comparison. (I haven't tested it myself,
                                 but online I found a factor of 10 difference).
    :param use_converged: Boolean. If True, the simulation will start with a converged solution. If False, no converged
                          solution is used. More information can be found in the docstring of the 'rhythm_is_a_dancer'
                          function.
    """
    # Check the rhythm parameter
    try:
        check_rhythm_parameter(rhythm)
    except ValueError as e:
        raise ValueError(e)

    # Check the fs parameter
    try:
        check_fs_parameter(fs)
    except ValueError as e:
        raise ValueError(e)

    # Check the num_beats_af_launch parameter
    try:
        check_num_beats_af_launch_parameter(rhythm, num_beats_af_launch)
    except ValueError as e:
        raise ValueError(e)

    # Check the num_beats_af_launch parameter
    try:
        check_num_draw_beats_at_once_parameter(rhythm, num_draw_beats_at_once)
    except ValueError as e:
        raise ValueError(e)

    # Check the batch_size parameter
    try:
        check_batch_size_parameter(batch_size)
    except ValueError as e:
        raise ValueError(e)

    # Check the activation time parameter combination of avn_p4_params, desired_rr_char and desired_vat_series
    try:
        check_activation_time_parameter_combination(rhythm, num_beats, avn_p4_params, desired_rr_char,
                                                    desired_vat_series)
    except ValueError as e:
        raise ValueError(e)

    # Check the patch_t_delay parameter
    try:
        check_patch_t_delay_parameter(patch_t_delay)
    except ValueError as e:
        raise ValueError(e)

    # Check the n_patches_atria parameter
    try:
        check_n_patches_atria_parameter(rhythm, n_patches_atria, patch_t_delay)
    except ValueError as e:
        raise ValueError(e)

    # Check the n_patches_atria parameter
    try:
        check_act_sigma_atria_parameter(rhythm, act_sigma_atria)
    except ValueError as e:
        raise ValueError(e)

    # Check the n_patches_atria parameter
    try:
        check_act_sigma_ventr_parameter(act_sigma_ventr)
    except ValueError as e:
        raise ValueError(e)

    # Check the keep_simulation_data parameter
    try:
        check_keep_simulation_data_parameter(keep_simulation_data)
    except ValueError as e:
        raise ValueError(e)

    # Check the use_converged parameter
    try:
        check_use_converged_parameter(use_converged)
    except ValueError as e:
        raise ValueError(e)

    return True


def check_rhythm_parameter(rhythm):
    if rhythm in ['NSR', 'AF']:
        return True
    else:
        raise ValueError("The rhythm parameter must be either 'NSR' or 'AF'.")


def check_num_beats_parameter(rhythm, num_beats):
    if isinstance(num_beats, bool):
        raise ValueError("The parameter 'num_beats' must be a positive integer including 0 or None.")
    if num_beats is None:
        return True
    if isinstance(num_beats, int) and num_beats >= 0:
        if num_beats == 0 and rhythm != 'NSR':
            raise ValueError("The parameter 'num_beats' can only be 0 if the rhythm is set to 'NSR'.")
        return True
    else:
        raise ValueError("The parameter 'num_beats' must be a positive integer including 0 or None.")


def check_num_beats_af_launch_parameter(rhythm, num_beats_af_launch):
    if isinstance(num_beats_af_launch, bool):
        raise ValueError("The parameter 'num_beats_af_launch' must be a positive integer >= 0 or None.")
    if num_beats_af_launch is None:
        if rhythm == 'AF':
            raise ValueError("The parameter 'num_beats_af_launch' must be a positive integer >= 0 if 'rhythm' is 'AF'.")
        return True
    if isinstance(num_beats_af_launch, int) and num_beats_af_launch >= 0:
        if rhythm == 'NSR':
            warnings.warn("The parameter 'num_beats_af_launch' is only relevant if 'rhythm' is 'AF'. Because " +
                          "'rhythm' is set to 'NSR', the parameter 'num_beats_af_launch' is ignored. Set the " +
                          "parameter 'num_beats_af_launch' to None to get rid of this warning.", UserWarning)
        return True
    else:
        raise ValueError("The parameter 'num_beats_af_launch' must be a positive integer >= 0 or None.")


def check_num_draw_beats_at_once_parameter(rhythm, num_draw_beats_at_once):
    if isinstance(num_draw_beats_at_once, bool):
        raise ValueError("The parameter 'num_draw_beats_at_once' must be a positive integer >= 1 or np.inf.")
    if num_draw_beats_at_once is None:
        raise ValueError("The parameter 'num_draw_beats_at_once' must be a positive integer >= 1 or np.inf.")
    if isinstance(num_draw_beats_at_once, str):
        raise ValueError("The parameter 'num_draw_beats_at_once' must be a positive integer >= 1 or np.inf.")
    if isinstance(num_draw_beats_at_once, dict):
        raise ValueError("The parameter 'num_draw_beats_at_once' must be a positive integer >= 1 or np.inf.")
    if (isinstance(num_draw_beats_at_once, int) and num_draw_beats_at_once >= 1) or np.isinf(num_draw_beats_at_once):
        if rhythm == 'NSR' and ~np.isinf(num_draw_beats_at_once):
            warnings.warn("The parameter 'num_draw_beats_at_once' is only relevant if 'rhythm' is 'AF'. Because " +
                          "'rhythm' is set to 'NSR', the parameter 'num_draw_beats_at_once' is ignored. To get rid of " +
                          "this warning, either don't specific the parameter 'num_draw_beats_at_once' or set the " +
                          "parameter 'num_draw_beats_at_once' to np.inf.", UserWarning)
        return True
    else:
        raise ValueError("The parameter 'num_beats_af_launch' must be a positive integer >= 1 or np.inf.")


def check_fs_parameter(fs):
    if not (isinstance(fs, float) or isinstance(fs, int)):
        raise ValueError("The fs parameter must be an integer or a float.")

    if not (20 <= fs <= 100000):
        raise ValueError("The fs parameter must be between 20 and 100000 Hz.")

    if fs < 200:
        warnings.warn("Sampling rate might be too low. Model could become unstable.", UserWarning)
    elif fs > 10000:
        warnings.warn("Sampling rate might be too high. Model takes too long.", UserWarning)

    return True


def check_batch_size_parameter(batch_size):
    if isinstance(batch_size, float):
        raise ValueError(u"The batch_size parameter must be an integer ≥ 1.")
    if isinstance(batch_size, bool):
        raise ValueError(u"The batch_size parameter must be an integer ≥ 1.")
    elif not isinstance(batch_size, int):
        raise ValueError(u"The batch_size parameter must be an integer ≥ 1.")
    elif batch_size < 1:
        raise ValueError(u"The batch_size parameter must be an integer ≥ 1.")

    return True


def check_activation_time_parameter_combination(rhythm, num_beats, avn_p4_params, desired_rr_char, desired_vat_series):
    """
    The heart rhythms can be set up in 4 different ways. This function checks the type and values of avn_p4_params,
    desired_rr_char and desired_vat_series based on their combination and which of the 4 ways it refers to. A detailed
    description to the parameter combinations can be found in the docstring of the 'rhythm_is_a_dancer' function.
    """
    check_num_beats_parameter(rhythm, num_beats)
    if num_beats is not None and avn_p4_params is None and desired_rr_char is None and desired_vat_series is None:
        # This is the first combination, where no parameters are given and new parameters are drawn and a new VAT series
        # is generated.
        if rhythm == 'AF':
            return True
        else:  # rhythm == 'NSR'
            raise ValueError("The combination of a given 'num_beats' and 'avn_p4_params'=None and " +
                             "'desired_rr_char'=None and 'desired_vat_series'=None is only valid if the 'rhythm' is" +
                             " set to 'AF'.")

    elif num_beats is not None and avn_p4_params is not None and desired_rr_char is None and desired_vat_series is None:
        # This is the second combination and is only allowed when the rhythm is set to 'AF'. In this case, the
        # 'avn_p4_params' parameter has to be a NumPy array of dimension (21,).
        if rhythm != 'AF':
            raise ValueError("The combination of a given 'num_beats' and given 'avn_p4_params' and " +
                             "'desired_rr_char'=None and 'desired_vat_series'=None is only valid if the 'rhythm' is " +
                             "set to 'AF'.")
        else:
            if not isinstance(avn_p4_params, np.ndarray) or avn_p4_params.shape != (21,):
                raise ValueError("The avn_p4_params parameter must be either None or a list of length 21 or a NumPy " +
                                 "array of dimension (21,).")
            else:
                check_avn_p4_params(avn_p4_params)
                return True

    elif num_beats is not None and avn_p4_params is None and desired_rr_char is not None and desired_vat_series is None:
        # This is the third combination, where num_beats and desired_rr_char are given. There are different rules for
        # the type and values of the desired_rr_char parameter based on the rhythm 'NSR' or 'AF'.
        if rhythm == 'NSR':
            if ((isinstance(desired_rr_char, int) or isinstance(desired_rr_char, float)) and
                    ((12 <= desired_rr_char <= 200) or (250 <= desired_rr_char <= 5000))):
                return True
            else:
                raise ValueError("If the 'rhythm' is 'NSR', then the desired_rr_char parameter must be a single int " +
                                 "or float between 12-200 (mean heart rate in bpm) or between 250-5000 (RR mean in ms).")
        elif rhythm == 'AF':
            # The desired_rr_char can be a single int or float and refers then to the mean heart rate or RR mean, or it
            # can be three parameters in a ndarray, and would be the RR mean, RR rmssd and RR sample entropy.
            if isinstance(desired_rr_char, int) or isinstance(desired_rr_char, float):
                if (12 <= desired_rr_char <= 200) or (250 <= desired_rr_char <= 5000):
                    return True
                else:
                    raise ValueError("If the 'rhythm' is 'AF', then the desired_rr_char parameter must be a single " +
                                     "int or float or a Numpy array with three integers or floats. You passed one " +
                                     "number, but it was outside the allowed range. It either has to be between " +
                                     "12-200 (mean heart rate in bpm) or between 250-5000 (RR mean in ms).")
            elif isinstance(desired_rr_char, np.ndarray):
                if desired_rr_char.shape != (3,):
                    raise ValueError("If the 'rhythm' is 'AF', and the desired_rr_char parameter must be a single " +
                                     "int or float or a Numpy array with three integers or floats. You passed a Numpy " +
                                     "array with the wrong shape. It has to be of shape (3,) with an RR mean between " +
                                     "250-5000ms, an RR rmssd between 0 and 1000ms, and an RR sample entropy between " +
                                     "0-10 (0.5-2.3 is a more realistic range).")
                else:
                    if not 250 <= desired_rr_char[0] <= 5000:  #
                        raise ValueError("The desired_rr_char[0] parameter describing the RR mean has to be between " +
                                         "250-5000ms.")
                    if not 0 < desired_rr_char[1] <= 1000:
                        raise ValueError("The desired_rr_char[1] parameter describing the RR rmssd has to be >0ms and " +
                                         "<=1000ms.")
                    if not 0 < desired_rr_char[2] <= 10:
                        raise ValueError("The desired_rr_char[2] parameter describing the RR sample entropy has to be " +
                                         ">0 and <=10 (0.5-2.3 is a more realistic range).")
                    return True
            else:
                raise ValueError("If the 'rhythm' is 'AF', then the desired_rr_char parameter must be a single int " +
                                 "or float representing the mean heart rate or RR mean, or a Numpy array with three " +
                                 "integers or floats representing the RR mean, RR rmssd and RR sample entropy.")

    elif num_beats is None and avn_p4_params is not None and desired_rr_char is None and desired_vat_series is not None:
        # This is the fourth combination and is only allowed when the rhythm is set to 'AF'. In this case, the
        # 'avn_p4_params' parameter has to be a NumPy array of dimension (21,).
        if rhythm != 'AF':
            raise ValueError("The combination of a given 'avn_p4_params' and 'desired_rr_char'=None and a given " +
                             "'desired_vat_series' is only valid if the 'rhythm' is set to 'AF'.")
        else:
            if isinstance(desired_vat_series, bool):
                raise ValueError("The desired_vat_series parameter must be an integer, float or non-empty " +
                                 "1-dimensional NumPy array.")
            if isinstance(desired_vat_series, int):
                desired_vat_series = np.array([desired_vat_series])
            if isinstance(desired_vat_series, float):
                desired_vat_series = np.array([desired_vat_series])
            if not isinstance(desired_vat_series, np.ndarray):
                raise ValueError("The desired_vat_series parameter must be an integer, float or non-empty " +
                                 "1-dimensional NumPy array.")
            if len(desired_vat_series.shape) != 1 or desired_vat_series.shape[0] == 0:
                raise ValueError("The desired_vat_series parameter must be an integer, float or non-empty " +
                                 "1-dimensional NumPy array.")
            if desired_vat_series[0] < 0:
                raise ValueError("The desired_vat_series parameter must be an integer, float or 1-dimensional NumPy " +
                                 "array with only positive values and where a value is always between 250-5000 larger " +
                                 "than the previous value.")
            if desired_vat_series.shape[0] > 1 and np.logical_or(~np.all(250 <= np.diff(desired_vat_series)),
                                                                 ~np.all(np.diff(desired_vat_series) <= 5000)):
                raise ValueError("The desired_vat_series parameter must be an integer, float or 1-dimensional NumPy " +
                                 "array with only positive values and where a value is always between 250-5000 larger " +
                                 "than the previous value.")

            if not isinstance(avn_p4_params, np.ndarray) or avn_p4_params.shape != (21,):
                raise ValueError("The avn_p4_params parameter must be a NumPy array of dimension (21,). Only the last " +
                                 "five parameters in avn_p4_params are used in this case. The first 16 parameters are " +
                                 "ignored.")
            else:
                check_avn_p4_params(avn_p4_params)
                return True
    elif num_beats is None and avn_p4_params is None and desired_rr_char is None and desired_vat_series is not None:
        # This is the fifth combination and is only allowed when the rhythm is set to 'NSR'. In this case, the
        # 'desired_vat_series' parameter has to be a NumPy array of minimum length 2.
        if rhythm != 'NSR':
            raise ValueError("The combination of 'num_beats'=None and 'avn_p4_params'=None and 'desired_rr_char'=None " +
                             "and a given 'desired_vat_series' is only valid if the 'rhythm' is set to 'NSR'.")
        else:
            if isinstance(desired_vat_series, bool):
                raise ValueError("The desired_vat_series parameter must be a 1-dimensional NumPy array with minimum " +
                                 "two ventricular activation times.")
            if not isinstance(desired_vat_series, np.ndarray):
                raise ValueError("The desired_vat_series parameter must be a 1-dimensional NumPy array with minimum " +
                                 "two ventricular activation times.")
            if len(desired_vat_series.shape) != 1 or desired_vat_series.shape[0] < 2:
                raise ValueError("The desired_vat_series parameter must be a 1-dimensional NumPy array with minimum " +
                                 "two ventricular activation times.")
            if desired_vat_series[0] < 0:
                raise ValueError("The desired_vat_series parameter must be a 1-dimensional NumPy array with only " +
                                 "positive values and where a value is always between 250-5000 larger " +
                                 "than the previous value.")
            if np.logical_or(~np.all(250 <= np.diff(desired_vat_series)), ~np.all(np.diff(desired_vat_series) <= 5000)):
                raise ValueError("The desired_vat_series parameter must be a 1-dimensional NumPy array with only " +
                                 "positive values and where a value is always between 250-5000 larger " +
                                 "than the previous value.")
            return True

    # This else statement will be reached if the combination of the parameters avn_p4_params, desired_rr_char and
    # desired_vat_series does not exist.
    else:
        raise ValueError("The combination of the parameters num_beats, avn_p4_params, desired_rr_char and " +
                         "desired_vat_series is not allowed. Please check the docstring of the 'rhythm_is_a_dancer' " +
                         "function for allowed combinations.")


def check_avn_p4_params(avn_p4_params):
    # check that each of the 21 values is in a realistic range
    if not 250 <= avn_p4_params[0] <= 600:  # Minimum refractory period of the slow AV nodal pathway
        raise ValueError("The avn_p4_params[0] parameter describing the minimum refractory period of the " +
                         "slow pathway has to be between 250-600ms.")
    if not 0 <= avn_p4_params[1] <= 600:  # Range of refractory period of the slow AV nodal pathway
        raise ValueError("The avn_p4_params[1] parameter describing the range of the refractory period of " +
                         "the slow pathway has to be between 0-600ms.")
    if not 50 <= avn_p4_params[2] <= 300:  # Time constant of the refractory period of the slow AV nodal
        # pathway
        raise ValueError("The avn_p4_params[2] parameter describing the time constant of the refractory " +
                         "period of the slow pathway has to be between 250-300ms.")
    if not 250 <= avn_p4_params[3] <= 600:  # Minimum refractory period of the fast AV nodal pathway
        raise ValueError("The avn_p4_params[3] parameter describing the minimum refractory period of the " +
                         "fast pathway has to be between 250-600ms.")
    if not 0 <= avn_p4_params[4] <= 600:  # Range of refractory period of the fast AV nodal pathway
        raise ValueError("The avn_p4_params[4] parameter describing the range of the refractory period of " +
                         "the fast pathway has to be between 0-600ms.")
    if not 50 <= avn_p4_params[5] <= 300:  # Time constant of the refractory period of the fast AV nodal
        # pathway
        raise ValueError("The avn_p4_params[5] parameter describing the time constant of the refractory " +
                         "period of the fast pathway has to be between 250-300ms.")
    if avn_p4_params[6] != 250:  # Fixed refractory period of the coupling node of the AV node model
        raise ValueError("The avn_p4_params[6] parameter describing the fixed refractory period of the " +
                         "AV node coupling node has been set to 250ms in previous publications. If you"
                         "want to change this, you need to rewrite this code here and possibly other "
                         "parts of the 'rhythm_is_a_dancer' function.")
    if not 0 <= avn_p4_params[7] <= 30:  # Minimum conduction delay of the slow AV nodal pathway
        raise ValueError("The avn_p4_params[7] parameter describing the minimum conduction delay of the " +
                         "slow pathway has to be between 0-30ms.")
    if not 0 <= avn_p4_params[8] <= 75:  # Range of conduction delay of the slow AV nodal pathway
        raise ValueError("The avn_p4_params[8] parameter describing the range of the conduction delay of " +
                         "the slow pathway has to be between 0-75ms.")
    if not 50 <= avn_p4_params[9] <= 300:  # Time constant of the conduction delay of the slow AV nodal
        # pathway
        raise ValueError("The avn_p4_params[9] parameter describing the time constant of the conduction " +
                         "delay of the slow pathway has to be between 50-300ms.")
    if not 0 <= avn_p4_params[10] <= 30:  # Minimum conduction delay of the fast AV nodal pathway
        raise ValueError("The avn_p4_params[10] parameter describing the minimum conduction delay of the " +
                         "fast pathway has to be between 0-30ms.")
    if not 0 <= avn_p4_params[11] <= 75:  # Range of conduction delay of the fast AV nodal pathway
        raise ValueError("The avn_p4_params[11] parameter describing the range of the conduction delay of " +
                         "the fast pathway has to be between 0-75ms.")
    if not 50 <= avn_p4_params[12] <= 300:  # Time constant of the conduction delay of the fast AV nodal
        # pathway
        raise ValueError("The avn_p4_params[12] parameter describing the time constant of the conduction " +
                         "delay of the fast pathway has to be between 50-300ms.")
    if avn_p4_params[13] != 0:  # Fixed conduction delay of the coupling node of the AV node model
        raise ValueError("The avn_p4_params[13] parameter describing the fixed conduction delay of the " +
                         "AV node coupling node has been set to 0ms in previous publications. If you want "
                         "to change this, you need to rewrite this code here and possibly other parts of "
                         "the 'rhythm_is_a_dancer' function.")
    # The next check is for the amplitude of the respiratory modulation of the AV nodal conduction
    # properties. In the publication by Plappert_2024, a neural network was trained on simulated data where
    # the amplitude was between -0.1 and 0.5. Then the trained neural network was tested on simulated data
    # where the amplitude was between 0.0 and 0.4. I would suggest to set the amplitude between 0-0.4.
    if not -0.1 <= avn_p4_params[14] <= 0.5:  # Amplitude of the respiratory modulation of the AV nodal
        # conduction properties.
        raise ValueError("The avn_p4_params[14] parameter describing the lower limit of the amplitude of " +
                         "the respiratory modulation of the AV nodal conduction properties has to be " +
                         "between -0.1 and 0.5.")
    # The next check is for the frequency of the respiratory modulation of the AV nodal conduction
    # properties. In the publication by Plappert_2024, the range of possible respiration frequencies was set
    # to 0.1-0.4Hz. 0.1Hz corresponds to deep breathing with 6 breaths per minute, and 0.4Hz corresponds to
    # faster breathing. Values outside 0.1-0.4Hz can also be realistic, so the allowed range is set here to
    # 0<freq<=10Hz.
    if not 0 < avn_p4_params[15] <= 10:  # Frequency of the respiratory modulation of the AV nodal conduction
        # properties.
        raise ValueError("The avn_p4_params[15] parameter describing the frequency of the respiratory " +
                         "modulation of the AV nodal conduction properties has to be between 0-10Hz.")
    if not 100 <= avn_p4_params[16] <= 250:  # mean arrival rate of atrial impulses used in the Pearson Type
        # IV distribution
        raise ValueError("The avn_p4_params[16] parameter describing the mean arrival rate of atrial " +
                         "impulses used in the Pearson Type IV distribution has to be between 100-250ms.")
    if not 15 <= avn_p4_params[17] <= 30:  # standard deviation of the arrival rate of atrial impulses used
        # in the Pearson Type IV distribution
        raise ValueError("The avn_p4_params[17] parameter describing the standard deviation of the " +
                         "arrival rate of atrial impulses used in the Pearson Type IV distribution has to " +
                         "be between 15-30ms.")
    if avn_p4_params[18] != 1:  # skewness of the arrival rate of atrial impulses used in the Pearson Type IV
        # distribution
        raise ValueError("The avn_p4_params[18] parameter describing the skewness of the arrival rate of " +
                         "atrial impulses used in the Pearson Type IV distribution has to be 1.")
    if avn_p4_params[19] != 6:  # kurtosis of the arrival rate of atrial impulses used in the Pearson Type IV
        # distribution
        raise ValueError("The avn_p4_params[19] parameter describing the kurtosis of the arrival rate of " +
                         "atrial impulses used in the Pearson Type IV distribution has to be 6.")
    if avn_p4_params[20] != 50:  # Lower limit of intervals between atrial impulses that are drawn from the
        # Pearson Type IV distribution. Below this limit, an interval is discarded
        # and a new interval is drawn as replacement.
        raise ValueError("The avn_p4_params[20] parameter describing the lower limit of intervals between " +
                         "atrial impulses that are drawn from the Pearson Type IV distribution has to be " +
                         "50ms.")

    # Check if the refractory period or conduction delay curves of the slow and fast pathway are overlapping
    # Import AV node model
    if os.path.exists('/Users/felix-macbook/PythonProjects/Paper-3/src'):
        sys.path.insert(0, '/Users/felix-macbook/PythonProjects/Paper-3/src')
    elif os.path.exists('/Users/felix/PycharmProjects/Paper-3/src'):
        sys.path.insert(0, '/Users/felix/PycharmProjects/Paper-3/src')
    if check_av_node_parameters_overlapping(np.concatenate((avn_p4_params[:6], avn_p4_params[7:13]))):
        raise ValueError("The refractory period or conduction delay curves of the slow and fast pathway " +
                         "of the AV node model are overlapping. This is not allowed. For all t>=0, the " +
                         "refractory period of the slow pathway has to be lower than the refractory " +
                         "period of the fast pathway. And for all t>=0, the conduction delay of the slow " +
                         "pathway has to be higher than the conduction delay of the fast pathway.")


def check_patch_t_delay_parameter(patch_t_delay):
    if not isinstance(patch_t_delay, np.ndarray) or patch_t_delay.shape != (23,) or patch_t_delay.dtype != np.float64:
        raise ValueError("The patch_t_delay parameter must be a NumPy array of dimension (23,) with dtype=float64.")
    return True


def check_n_patches_atria_parameter(rhythm, n_patches_atria, patch_t_delay):
    if isinstance(n_patches_atria, float):
        raise ValueError(u"The n_patches_atria parameter must be an integer ≥ 1.")
    if rhythm == 'AF':
        if isinstance(n_patches_atria, bool):
            raise ValueError(u"The n_patches_atria parameter must be an integer ≥ 1.")
        elif not isinstance(n_patches_atria, int):
            raise ValueError(u"The n_patches_atria parameter must be an integer ≥ 1.")
        elif n_patches_atria < 1:
            raise ValueError(u"The n_patches_atria parameter must be an integer ≥ 1.")
    elif rhythm == 'NSR':
        if isinstance(n_patches_atria, bool):
            raise ValueError("If 'rhythm' is set to 'NSR', then the 'n_patches_atria' parameter must be 1.")
        elif not isinstance(n_patches_atria, int):
            raise ValueError("If 'rhythm' is set to 'NSR', then the 'n_patches_atria' parameter must be 1.")
        elif n_patches_atria < 1:
            raise ValueError(u"The n_patches_atria parameter must be an integer ≥ 1.")

    if n_patches_atria > 1 and np.all(np.isinf(patch_t_delay[21:23])):
        warnings.warn("If the 'patch_t_delay' values for both elements are np.inf, then n_patches_atria will " +
                      "be set to 1. This is because simulating more atrial patches leads to the same results and +"
                      "is slower.", UserWarning)
    return True


def check_act_sigma_atria_parameter(rhythm, act_sigma_atria):
    if not isinstance(act_sigma_atria, (int, float)):
        raise ValueError(f"The act_sigma_atria must be an integer or float, but got {type(act_sigma_atria).__name__}")
    if act_sigma_atria < 0:
        raise ValueError("The act_sigma_atria parameter must be a positive number.")
    if isinstance(act_sigma_atria, bool):
        raise ValueError(u"The act_sigma_atria parameter must be an integer or float ≥ 0.")
    if rhythm == 'AF' and act_sigma_atria != 0:
        raise ValueError("If 'rhythm' is set to 'AF', then the 'act_sigma_atria' parameter must be 0.")
    return True


def check_act_sigma_ventr_parameter(act_sigma_ventr):
    if not isinstance(act_sigma_ventr, (int, float)):
        raise ValueError(f"The act_sigma_ventr must be an integer or float, but got {type(act_sigma_ventr).__name__}")
    if act_sigma_ventr < 0:
        raise ValueError("The act_sigma_ventr parameter must be a positive number.")
    if isinstance(act_sigma_ventr, bool):
        raise ValueError(u"The act_sigma_atria parameter must be an integer or float ≥ 0.")
    return True


def check_keep_simulation_data_parameter(keep_simulation_data):
    if isinstance(keep_simulation_data, str):
        if keep_simulation_data == 'all':
            return True
        elif keep_simulation_data == 'no_launch':
            return True
        else:
            raise ValueError(u"The keep_simulation_data parameter must be 'all' or 'no_launch'.")
    else:
        raise ValueError(u"The keep_simulation_data parameter must be 'all' or 'no_launch'.")


def check_use_converged_parameter(use_converged):
    if isinstance(use_converged, bool):
        return True
    else:
        raise ValueError(u"The use_converged parameter must be Boolean.")
