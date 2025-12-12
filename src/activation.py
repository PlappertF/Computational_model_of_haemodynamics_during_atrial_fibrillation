import numpy as np
from numba import njit
import warnings
from src.avn import get_rhythm_parameters, run_avn_model, generate_pearson4, get_pearson4_parameters


def initialize_activations(model, rhythm, num_beats, num_beats_af_launch, num_draw_beats_at_once, avn_p4_params,
                           desired_rr_char, desired_vat_series, act_sigma_atria, act_sigma_ventr, keep_simulation_data,
                           use_converged):
    """
    :param model: Class object of the model with all parameters.
    :param rhythm: The rhythm to be simulated. Options are 'NSR' for normal sinus rhythm and 'AF' for atrial
                   fibrillation.
    :param num_beats: Number of heartbeats to simulate. More information can be found in the docstring of the
                      'rhythm_is_a_dancer' function.
    :param num_beats_af_launch: Number of heartbeats to simulate during the AF launch phase. More information can be
                                found in the docstring of the 'rhythm_is_a_dancer' function.
    :param num_draw_beats_at_once: Number of His-bundle activation times (integer >= 1) that are drawn at once. More
                                   information can be found in the docstring of the 'rhythm_is_a_dancer' function.
    :param avn_p4_params: Parameters for the AV node model. More information can be found in the docstring of the
                          'rhythm_is_a_dancer' function.
    :param desired_rr_char: Characteristic of the desired RR series. More information can be found in the docstring of
                            the 'rhythm_is_a_dancer' function.
    :param desired_vat_series: Desired ventricular activation time (VAT) series. More information can be found in the
                               docstring of the 'rhythm_is_a_dancer' function.
    :param keep_simulation_data: Parameter specifying if the generated data should include all iterations, or exclude the
                                 iterations of the NSR launch phase and AF launch phase, or if only the iterations of
                                 systolic and diastolic events should be saved. More information can be found in the
                                 docstring of the 'rhythm_is_a_dancer' function.
    :param act_sigma_atria: Standard deviation of the atrial activation time. More information can be found in the
                            docstring of the 'rhythm_is_a_dancer' function.
    :param act_sigma_ventr: Standard deviation of the ventricular activation time. More information can be found in the
                            docstring of the 'rhythm_is_a_dancer' function.

    Parameters that will be added to the model.activation object:
    :model.activation.avn_rp: Parameters for the refractory period of the AV node model
        avn_rp[0]: Minimum refractory period of the slow AV nodal pathway
        avn_rp[1]: Range of refractory period of the slow AV nodal pathway
        avn_rp[2]: Time constant of the refractory period of the slow AV nodal pathway
        avn_rp[3]: Minimum refractory period of the fast AV nodal pathway
        avn_rp[4]: Range of refractory period of the fast AV nodal pathway
        avn_rp[5]: Time constant of the refractory period of the fast AV nodal pathway
        avn_rp[6]: Fixed refractory period of the coupling node of the AV node model
    :model.activation.avn_cd: Parameters for the conduction delay of the AV node model
        avn_cd[0]: Minimum conduction delay of the slow AV nodal pathway
        avn_cd[1]: Range of conduction delay of the slow AV nodal pathway
        avn_cd[2]: Time constant of the conduction delay of the slow AV nodal pathway
        avn_cd[3]: Minimum conduction delay of the fast AV nodal pathway
        avn_cd[4]: Range of conduction delay of the fast AV nodal pathway
        avn_cd[5]: Time constant of the conduction delay of the fast AV nodal pathway
        avn_cd[6]: Fixed conduction delay of the coupling node of the AV node model
    :model.activation.avn_resp: Parameters for the respiratory modulation of the AV node model
        avn_resp[0]: Amplitude of the respiratory modulation of the AV node model
        avn_resp[1]: Frequency of the respiratory modulation of the AV node model
    :model.activation.p4dist: Parameters of the Pearson Type IV distribution
        p4dist[0]: Mean arrival rate of atrial impulses used in the Pearson Type IV distribution
        p4dist[1]: Standard deviation of the arrival rate of atrial impulses used in the Pearson Type IV distribution
        p4dist[2]: Skewness of the arrival rate of atrial impulses used in the Pearson Type IV distribution
        p4dist[3]: Kurtosis of the arrival rate of atrial impulses used in the Pearson Type IV distribution
        p4dist[4]: Lower limit of the Pearson Type IV distribution
        p4dist[5]: nu parameter of the Pearson Type IV distribution
        p4dist[6]: b parameter of the Pearson Type IV distribution
        p4dist[7]: a parameter of the Pearson Type IV distribution
        p4dist[8]: lam parameter of the Pearson Type IV distribution
        p4dist[9]: M parameter of the Pearson Type IV distribution
        p4dist[10]: loggM parameter of the Pearson Type IV distribution
        p4dist[11]: invgM parameter of the Pearson Type IV distribution
    :model.activation.rr_char: RR series characteristics
        rr_char[0]: Mean of the RR series (in ms)
        rr_char[1]: Root mean square of successive RR interval differences (in ms)
        rr_char[2]: Sample entropy of the RR series

      - Matrix of activation times per MultiPatch model patch (model.activation.t_act)
        The matrix has N rows, where N is the number of patches in the MultiPatch model (model.heart.n_patches_tot).
        The matrix has M columns, where M is the number of activations in the simulation. M is not the number of all
        activations that will be simulated given by 'num_beats' or 'desired_vat_series'. Instead, it is the number of
        activations that can contribute at the same time to the contractility computation of the heart patches. With the
        time constants of the MultiPatch model and the minimum interval between atrial or ventricular activations, we can
        calculate how many activations can contribute to the contractility of the heart patches at the same time. Every
        time the 'newest' activation is triggered, the next activation will be added to the right of the matrix and the
        oldest activation on the left that has no effect anymore will be discarded. The next activation is either drawn
        using the Pearson Type IV distribution for the atrial patches or the AV node model for the ventricular patches,
        or the next activation is taken from the provided 'desired_vat_series'. For the algorithm it means that every
        time the rightmost value in a row is reduced to zero or below, the next activation time is drawn and added to the
        right, so that there is always a value >0 in the rightmost column of the matrix after the activation matrix t_act
        was updated.

        The frise calculation for the contractility could consider several previous electrical activations. An electrical
        activation has an impact on the frise value if the t value is between 0 and 8. The t values are calculated as the
        difference between the actual model time and the time of all electrical activations and these numbers are divided
        by model.heart.tr. When we use the AV node model to produce a ventricular activation time, we have a minimum RR
        interval value enforced by the constant refractory period of the coupling node. We also have a minimum AA
        interval value enforced by the fixed lower limit of the Pearson Type IV distribution. During NSR, we have the
        minimum RR interval given by the mean RR interval. With all this information, we can calculate a theoretical
        maximum of electrical activations that could have an impact on the frise value. We initialize the shape of the
        matrix of activation times for each patch with the theoretical maximum plus 1. We add one, so that in addition of
        the past activation, we can always keep track of the next activation. The theoretical maximum plus one instead of
        all activations is chosen to keep the size of the matrix as small as possible to save memory.

        Currently, the advantage of drawing the activation times of atrial and ventricular patches during the simulation
        is that the activation time matrix is small and does not increase in size with longer VAT series. In the future,
        this implementation makes it easy for adjustments where the activation times are influenced by the current state
        of the heart model, for example by hemodynamics, autonomic modulation, cardiac remodeling, or changes in hormone
        levels.

        I added the option to predefine the VAT series, because I needed it for my study where I analyze the hemodynamics
        for different RR series characteristics.

      - Array of activation time delays between patches (model.activation.patch_t_delay)
        If all patches of all heart chambers would contract at the same time, then the array would be filled with zeros.
        If we want to simulate an electrical conduction propagation through a chamber, we can use this array. This would
        assume that the conduction propagation is same beat after beat. During normal sinus rhythm, we can use a negative
        conduction delay of -120ms to account for the AV nodal delay between ventricles and atria. If 'rhythm'='AF',
        random activation times are drawn independently in the AF launch phase and real simulation for each atrial patch
        using the Pearson Type IV distribution. If an element in patch_t_delay is np.inf, then this patch will never
        contract throughout all NSR launch phase, AF launch phase and the real simulation.

      - Flags for the launch of the NSR and AF rhythm, and the real simulation
        When running the model during NSR, we first need to run the model with a fixed RR interval equal to the RR mean
        of our desired VAT series until the volume trend of two consecutive heartbeats is stable and basically the same
        (the difference of the volume at the time when the His-bundle theoretically activates has to be below a
        threshold). If that condition is met, then the 'flag_launch_nsr' flag is set to False and if both the
        'flag_launch_nsr' and 'flag_launch_af' flags are False (which it would be then directly in the NSR case), then the
        real simulation starts running. We need this launch phase for the model to settle in.
        When running the model during AF, we first run the model with a fixed RR interval equal to the RR mean of our
        desired VAT series, exactly as in the case during NSR until two consecutive heartbeats have the same volume
        trend. If that condition is met, then the 'flag_launch_nsr' flag is set to False. In the AF case, the
        'flag_launch_af' would be true, which means that an AF launch phase starts. In the AF launch phase, we use
        basically an ventricular activation time series with the same RR series characteristics as during the real
        simulation, but with a fixed number of beats. And in the AF launch phase, the atria are now fibrillating as they
        will be during the real simulation. Basically, the model simulates atrial fibrillation, but we only start
        recording the results after the AF launch phase to let the model settle in. The fibrillation of the atria is
        simulated by each atrial MultiPatch model patch being activated with an independent series of activation times
        that is drawn from a Pearson Type IV distribution. The AF launch phase is simulated for X beats, where X was
        determined in a simulation experiment showing that X is the minimum number of beats needed for the volume trend
        to lose the history information that the atria were previously beating in a regular pattern in the
        'NSR launch phase'. After the X beats, the 'flag_launch_af' flag is set to False and the real simulation starts
        running. The 'flag_simulation' is set to True and will be set to False when simulation can be terminated.

    Ideas for future modifications of this code: Currently, normal sinus rhythm is simulated in a very simple way. We
    give a mean heart rate or RR mean and a VAT series is created with the same interval between activations equal to the
    RR mean. In Scarsoglio_2014, an RR series is generated by drawing from a skewed normal distribution. We could do this
    as well. If we feel like it, we could implement a sinus node model similar to the AV node model and vary the
    conduction properties over time using the respiratory modulation that is also implemented in the AV node model. This
    model extension would require some analysis to see if the model performs better with the more complex description. We
    also would need to check the literature for already proposed methods that do similar things.
    """
    # In the 'format_input_params' function, the rhythm was set to 0 for 'NSR' and 1 for 'AF', because a comparison with
    # a number is faster than with strings. Throughout the simulation, the number stored in 'rhythm' is used, but for the
    # cardiogrowth model that is returned to the user, the rhythm is stored as a string.
    if rhythm == 0:
        model.activation.rhythm = 'NSR'
    else:  # rhythm == 1
        model.activation.rhythm = 'AF'
    """
    First, we add the AV node model parameters (avn_rp, avn_cd, avn_resp) and the Pearson Type IV distribution parameters
    (p4dist) to the activation dataclass. If the rhythm is 'NSR', we set the model parameters to np.nan, because we will 
    not use the AV node model to generate a rhythm. If the rhythm is 'AF' and the avn_p4_params are None, we draw a 
    random parameter set within predefined ranges (stated in the docstring of the 'rhythm_is_a_dancer' function) and we 
    redraw parameter sets until the refractory period and conduction delay curves of the fast and slow pathway are not 
    overlapping. If the rhythm is 'AF' and the avn_p4_params are not None, we use the provided parameters and only 
    compute some Pearson Type IV distribution parameters from the drawn mean, standard deviation, skewness and kurtosis.
    
    Second, we compute the RR mean to initialize the t_act matrix for the 'NSR launch phase'. If the rhythm is 'NSR', a 
    mean heart rate or RR mean must be provided and the 'check_input_params' function converted 'desired_rr_char' to a
    RR mean that can be directly written into the rr_char array, because we model 'NSR' currently with a constant RR 
    interval. If the rhythm is 'AF' and desired_rr_char is None, we run the AV node model and generate an RR series with 
    200 RR intervals from which the RR mean is calculated.  If the rhythm is 'AF' and desired_rr_char is not None, we 
    also run the AV node model and generate an RR series with 200 RR intervals from which the RR mean or RR mean, 
    RR rmssd and RR sample entropy are calculated. If the calculated RR series characteristics are not close enough to 
    the passed desired RR series characteristics, new AV node model parameters and Pearson Type IV distribution 
    parameters are drawn and a new RR series is generated until the RR series characteristics are close enough.
    """
    if rhythm == 0:  # rhythm == 'NSR'
        model.activation.avn_rp = np.ones(7) * np.nan
        model.activation.avn_cd = np.ones(7) * np.nan
        model.activation.avn_resp = np.ones(2) * np.nan
        model.activation.p4dist = np.ones(12) * np.nan
        if desired_rr_char is None and desired_vat_series is not None:
            model.activation.rr_char = np.array([np.mean(np.diff(desired_vat_series)), 0, 0])
        elif desired_rr_char is not None and desired_vat_series is None:
            model.activation.rr_char = desired_rr_char
    else:  # rhythm == 'AF'
        if desired_vat_series is None:
            [model.activation.avn_rp, model.activation.avn_cd, model.activation.avn_resp, model.activation.p4dist,
             model.activation.rr_char] = get_rhythm_parameters(avn_p4_params, desired_rr_char)
        # This is used if we pass an AV node model parameter set and a desired VAT series to the cardiogrowth model.
        else:  # desired_vat_series is not None
            model.activation.avn_rp = avn_p4_params[:7]
            model.activation.avn_cd = avn_p4_params[7:14]
            model.activation.avn_resp = avn_p4_params[14:16]
            p4dist = np.zeros(12)
            p4dist[0:5] = avn_p4_params[16:]
            model.activation.p4dist = get_pearson4_parameters(p4dist)
            model.activation.rr_char = np.array([np.mean(np.diff(desired_vat_series)), 0, 0])

    """
    Now we initialize the matrix of activation times per MultiPatch model patch (model.activation.t_act).
    To determine the number of columns for the t_act matrix, we calculate how many activations could contribute to the 
    computation of the contractility (the frise parameter to be exact) of a heart patch and add 1 to that number. We add
    one more column so that we can draw the next activation and keep it in the t_act matrix. If the rhythm is 'NSR', the 
    minimum time duration between activations is given by the RR mean in 'model.activation.rr_char[0]'. If the rhythm is 
    'AF', the minimum time duration between ventricular activations is given by the fixed refractory period of the 
    coupling node of the AV node model 'model.activation.avn_rp[6]' which enforces a minimum RR interval in the RR 
    series. The minimum time duration between atrial activations is given by the lower limit of the Pearson Type IV 
    distribution 'model.activation.p4dist[4]' which enforces a minimum AA interval in the AA series. In addition, we make
    sure that the activation matrix has at least 3 columns. This is because of the way the contraction function is set 
    up, tracking how much calcium was released and how much calcium is coming back into the storage.
    We fill the columns of the t_act matrix for the 'NSR launch phase', where the time duration between each activation
    is constant and equal to the RR mean and the ventricular activation times of the rightmost column are set to 0.0 to 
    mark the current time. All positive values in t_act are in the future and all negative values in t_act are in the 
    past and each iteration, a delta_t is subtracted from all values in t_act. If the rhythm is 'NSR', a Gaussian sample
    is added to the activation time of each atrial and ventricular patch to replicate the electrical excitation 
    propagation. If the rhythm is 'AF', a Gaussian sample is only added to the ventricular patches. However, in 
    the NSR launch phase, both atria and ventricles are assumed to be in 'NSR', so the Gaussian sample is added to all
    patches. The standard deviation of the Gaussian sample is 'act_sigma_ventr' for ventricular patches and 
    'act_sigma_atria' for atrial patches.
    """
    # Now set the t_act matrix and the flags for the launch of the NSR and AF rhythm.
    if rhythm == 0:  # rhythm == 'NSR'
        td_min = np.ones(model.heart.n_patches_tot) * model.activation.rr_char[0]
    else:  # rhythm == 'AF'
        td_min = np.zeros(model.heart.n_patches_tot)
        td_min[model.heart.i_ventricles] = model.activation.avn_rp[6]
        td_min[model.heart.i_atria] = model.activation.p4dist[4]
    # Making sure that t_act matrix has enough columns to have space for maximum number of activations based on min
    # refractory period
    fit_contraction_hills = int(np.ceil(max(8 * model.heart.tr / td_min)))
    # Making sure that t_act matrix has enough columns to have space for recovering of contraction hills. This is a crude
    # Approximation assuming lsc_norm never gets bigger than 4.
    fit_recover = int(np.ceil(max((np.pi/2*model.heart.td+4*model.heart.t_ad)/td_min)))
    num_t_act_columns = np.maximum(np.maximum(fit_contraction_hills + 1, fit_recover+2), 3)
    model.activation.t_act = np.array([model.activation.patch_t_delay + i * model.activation.rr_char[0]
                                       for i in np.arange(num_t_act_columns) - (num_t_act_columns - 2)]).transpose()
    act_del = np.random.normal(loc=0, scale=1, size=model.activation.t_act.shape)
    act_del[model.heart.i_ventricles, :] *= act_sigma_ventr
    model.activation.sigma_ventr = act_sigma_ventr
    act_del[model.heart.i_atria, :] *= act_sigma_atria
    model.activation.sigma_atria = act_sigma_atria
    model.activation.t_act += act_del
    model.activation.last_gauss = act_del[:, -1]

    """
    Now we set flags for the launch of the NSR and AF rhythm. During the simulation, if a flag is True, the corresponding
    launch phase will be executed at some point in the simulation. If a flag is False, the corresponding launch phase 
    either has been already finished or it will not be executed at all. First the 'flag_launch_nsr' will be checked. If
    it is True, then the simulation is in the 'NSR launch phase' where the model is run with a fixed RR interval equal to
    the RR mean until the volume trend of two consecutive RR intervals are close enough (flag_launch_nsr is set to False 
    if the difference between the volume between two consecutive beats is below a threshold). If the 'flag_launch_nsr' is
    False, then the 'flag_launch_af' will be checked. If it is True, then the simulation runs the model with fibrillating 
    atria and a irregular ventricular rhythm for a fixed number of beats (num_beats_af_launch) until 'flag_launch_af' will
    be set to False. If both 'flag_launch_nsr' and 'flag_launch_af' are False, then the simulation runs the model for a 
    number of beats (num_beats) with a regular rhythm in the 'NSR' case and an irregular rhythm in the 'AF' case. If 
    'flag_simulation' is set to False, then the current iteration will be the last for the cardiogrowth simulation.
    """
    if rhythm == 0:  # rhythm == 'NSR'
        model.activation.flag_launch_nsr = True  # Since this flag is set to True, there will be a 'NSR launch phase'.
        model.activation.flag_launch_af = False  # Since this flag is set to False, there will be no 'AF launch phase'.
        model.activation.flag_simulation = True
    else:  # rhythm == 'AF' is the only other option
        model.activation.flag_launch_nsr = True  # Since this flag is set to True, there will be a 'NSR launch phase'.
        model.activation.flag_launch_af = True  # Since this flag is set to True, there will be a 'AF launch phase'.
        model.activation.flag_simulation = True  # The simulation will end once this flag is set to False.
    model.activation.keep_simulation_data = keep_simulation_data
    model.activation.use_converged = use_converged

    """
    Now the arrays for the ventricular activation time (VAT) series and RR series are declared. If the rhythm is 'NSR', a 
    number of beats 'num_beats' had to be passed to the 'rhythm_is_a_dancer' function with which the VAT and RR series 
    can already be calculated. For 'NSR', we draw directly all the His-Bundle activation times for the VAT series and RR
    series. In the future, we could change the to draw the next RR interval from a distribution instead of always using
    a fixed RR interval. For 'AF', we have the option with a hyper-parameter to either draw one or several His-Bundle 
    activation times at once. Drawing several His-Bundle activation times at once is faster, but maybe one day, we want 
    to draw the next His-Bundle activation time only when the previous activation is triggered.
    """
    model.activation.rt = np.ones(22) * np.nan  # State variable for the refractory period times of the AV node model
    model.activation.dt = np.ones(22) * np.nan  # State variable for the conduction delay times of the AV node model
    model.activation.aat = np.nan  # State variable for the atrial activation time of the AV node model
    # State variable for the number of atrial impulses in the priority queue in the AV node model
    model.activation.node0_n_imp = np.nan
    model.activation.q = [(0.0, 0.0)]  # State variable for the impulse priority queue of the AV node model
    if rhythm == 0:  # rhythm == 'NSR'
        # Parameters for 'AF launch phase' are not used for 'NSR' because the 'AF launch phase' is skipped.
        model.activation.num_beats_af_launch = np.nan
        model.activation.vat_series_af_launch = np.nan
        model.activation.rr_series_af_launch = np.array([np.nan])
        # Parameters for the real simulation
        if desired_rr_char is None and desired_vat_series is not None:
            model.activation.num_beats = desired_vat_series.shape[0]
            model.activation.vat_series = desired_vat_series
            model.activation.rr_series = np.diff(model.activation.vat_series)
            model.activation.beats_to_draw = 0
            model.activation.num_draw_beats_at_once = np.inf
        elif desired_rr_char is not None and desired_vat_series is None:
            model.activation.num_beats = num_beats
            model.activation.vat_series = model.activation.rr_char[0] * np.arange(num_beats)
            model.activation.rr_series = np.diff(model.activation.vat_series)
            model.activation.beats_to_draw = 0
            model.activation.num_draw_beats_at_once = np.inf
    else:  # rhythm=='AF'
        # First we declare an array combining the VAT series of the 'AF launch phase' and the real simulation.
        model.activation.num_beats_af_launch = num_beats_af_launch
        if desired_vat_series is None:
            model.activation.num_beats = num_beats
            vat_series = np.ones(num_beats_af_launch + num_beats) * np.nan
            rr_series = np.ones(num_beats_af_launch + num_beats - 1) * np.nan
        else:  # desired_vat_series is not None
            model.activation.num_beats = desired_vat_series.shape[0]
            vat_series = np.concatenate((np.ones(num_beats_af_launch) * np.nan, desired_vat_series))
            rr_series = np.concatenate((np.ones(num_beats_af_launch) * np.nan, np.diff(desired_vat_series)))

        # Now we fill the declared combined array for the VAT series with activation times. We draw
        # 'num_draw_beats_at_once' of His-bundle activation times for the vat_series.
        max_beats_to_draw = sum(np.isnan(vat_series))
        # In the special case, where we have to draw beats for the 'AF launch phase', but we already have a VAT series
        # for the real simulation, we have to draw one extra beat to compute the RR interval between the last beat of the
        # 'AF launch phase' and the first beat of the real simulation.
        if desired_vat_series is not None and num_beats_af_launch > 0:
            max_beats_to_draw += 1
        # Now that we know the maximum beats we have to draw, we can compare that number to the number of beats that will
        # be drawn at once and choose the smaller number.
        num_beats_to_draw = min(max_beats_to_draw, num_draw_beats_at_once)

        # This code does not have to be executed if an RR series is passed and the AF launch phase has zero beats.
        if num_beats_to_draw > 0:
            model.activation.q = None  # State variable for the impulse priority queue of the AV node model
            # Run the AV node model to draw His-bundle activation times for the VAT series.
            fixed_avn_pars = np.concatenate((model.activation.avn_rp, model.activation.avn_cd, model.activation.avn_resp,
                                            model.activation.p4dist))
            avn_states = np.concatenate((model.activation.rt, model.activation.dt,
                                         np.array([model.activation.aat, model.activation.node0_n_imp])))
            # The default of n_aa in run_avn_model is 2, because the function 'run_avn_model' runs fastest with n_aa set
            # to 2. If n_aa is set to a larger value, the function is slower because the priority queue has to handle
            # more impulses at the same time slowing the function down. If n_aa is set to 1, the function is slower as
            # well because drawing 2 AA intervals at the same time is much faster than drawing 2 AA intervals one after
            # the other due to overhead.
            vat, avn_states, model.activation.q = (
                run_avn_model(fixed_avn_pars, avn_states, qu=model.activation.q, num_his_act=num_beats_to_draw, n_aa=2))
            # Currently, the first ventricular activation time is not at time=0.0, so we correct that.
            model.activation.rt = avn_states[0:22]
            model.activation.dt = avn_states[22:44]
            model.activation.aat = avn_states[44]
            model.activation.node0_n_imp = avn_states[45]
            model.activation.rt[1:] -= vat[0]
            model.activation.aat -= vat[0]
            for row_idx, row in enumerate(model.activation.q):
                model.activation.q[row_idx] = (row[0] - vat[0], row[1])
            vat -= vat[0]
            rr = np.diff(vat)
            # Write the drawn VAT and RR series to the previously declared arrays.
            vat_series[:num_beats_to_draw] = vat
            if desired_vat_series is not None and num_beats_af_launch > 0:
                vat_series[num_beats_af_launch+1:] += vat_series[num_beats_af_launch] - desired_vat_series[0]
            rr_series[:len(rr)] = rr

        model.activation.vat_series_af_launch = vat_series[:num_beats_af_launch]
        model.activation.rr_series_af_launch = rr_series[:num_beats_af_launch]
        model.activation.vat_series = vat_series[num_beats_af_launch:]
        model.activation.rr_series = rr_series[num_beats_af_launch:]
        # If model.activation.beats_to_draw is above 0, then more beats have to be drawn while the simulation is running.
        model.activation.beats_to_draw = max_beats_to_draw - num_beats_to_draw
        model.activation.num_draw_beats_at_once = num_draw_beats_at_once

    """
    Finally, to keep track during the simulation of how many beats have been simulated in each 'NSR launch phase', 
    'AF launch phase' and the real simulation, we initialize a 'his_activation_counter', 
    and 'next_his_activation_time'. The 'next_his_activation_time' is a scalar that is set to the amount of 
    milliseconds until next His activation that leaves the AV node model. Each iteration, the model.solver.dt is 
    subtracted from the 'next_his_activation_time'. If 'next_his_activation_time' is zero or below zero, some checks 
    are triggered based on the phase. During the 'NSR launch phase', this is the time the volume is compared with the 
    volume from the previous iteration. If the difference between the volumes is below a threshold, then the 'NSR 
    launch phase' ends. In all phases, at the time of the His activation, the 'his_activation_counter' is increased by 
    one. The 'his_activation_counter' is a (3,) NumPy array corresponding to the number of activations during the 'NSR 
    launch phase', 'AF launch phase' and the real simulation. During the 'AF launch phase' and the real simulation, 
    the number of activations in the 'his_activation_counter' is used to keep track of how many beats have been 
    simulated, and when the phase will end.
    """
    model.activation.his_activation_counter = np.zeros(3)
    model.activation.next_his_activation_time = model.activation.rr_char[0]


@njit(cache=False)
def assess_model_phase(phase_flags, his_activation_counter, v_conv, v, t_act, term_msg, const):
    """
    :param phase_flags: Flags for the launch of the NSR and AF rhythm, and the real simulation
        phase_flags[0]: Flag for the 'NSR launch phase' (model.activation.flag_launch_nsr)
        phase_flags[1]: Flag for the 'AF launch phase' (model.activation.flag_launch_af)
        phase_flags[2]: Flag for the real simulation (model.activation.flag_simulation)
        phase_flags[3]: False if rhythm is 'NSR', True if rhythm is 'AF'
        phase_flags[4]: Flag to track whether the simulation was run successfully or was terminated.
    :param his_activation_counter: His activation counter (model.activation.his_activation_counter).
    :param v_conv: Volume at the previous His-bundle activation time (model.convergence_volumes). Before the actual
        simulation is run, we first simulate one heart beat in a loop until the volumes in the first and last iteration
        are close enough. For this comparison, I store the volumes of the loop iteration in the
        'model.convergence_volumes' parameter.
    :param v: Volume at the current His-bundle activation time (model.volumes[model.solver.batch_inc, :]).
    :param t_act: Matrix of activation times per MultiPatch model patch (model.activation.t_act).
    :param term_msg: Message to inform the user why the simulation was terminated.
    :param const: Array with 4 fixed constants that don't change in the whole simulation.
        const[0]: (model.solver.cutoff) Threshold for the volume difference between two consecutive heartbeats
        const[1]: (model.solver.nsr_launch_iter_max) Maximum number of iterations for the 'NSR launch phase'
        const[2]: (model.activation.num_beats_af_launch) Number of heartbeats to simulate during the AF launch phase
        const[3]: (model.activation.num_beats) Number of heartbeats to simulate during the real simulation

    Note, that the three if-statements are not nested, but are evaluated one after the other. It can be that the 'NSR
    launch phase' is assessed and finished. Then the 'AF launch phase' is assessed and finished because the number of
    beats in the 'AF launch phase' was set to 0. Then the real simulation is assessed. All in one iteration.
    """
    """
    Code to assess the model state in the NSR launch phase.
    """
    if phase_flags[0]:  # model.activation.flag_launch_nsr
        his_activation_counter[0] += 1
        # First, check if the end of the NSR launch phase is reached by comparing the volumes of two consecutive
        # heartbeats.
        # const[0]: model.solver.cutoff
        if np.all(np.abs((v - v_conv)/v_conv) <= const[0]):
            # NSR launch phase is over.
            phase_flags[0] = False  # model.activation.flag_launch_nsr
            if phase_flags[1]:  # model.activation.flag_launch_af
                # Since flag_af is True, atrial fibrillation will be simulated. In this case, we don't need any
                # activation times in the future that correspond to NSR. Therefore, all ventricular and atrial
                # activations in the rightmost columns of the t_act matrix are set to 0.0 as long as the previous
                # activation in the column before is minimum 50 ms in the past.
                # t_act[np.where(t_act[:, -2] <= -50), -1] = 0.0
                t_act[t_act[:, -2] <= -50, -1] = 0.0
        # If the NSR launch phase is not over yet, check if the maximum number of iterations is reached.
        # const[1]: model.solver.nsr_launch_iter_max
        elif his_activation_counter[0] == const[1]:
            # Set all phase flags to false, so no more simulation iteration will be executed.
            phase_flags[0] = False  # model.activation.flag_launch_nsr
            phase_flags[1] = False  # model.activation.flag_launch_af
            phase_flags[2] = False  # model.activation.flag_simulation
            phase_flags[4] = True   # model.termination.terminated
            term_msg = ("Maximum allowed number of iterations for the 'NSR launch phase' has been reached. " +
                        "The simulation stops here and the last beat of the 'NSR launch phase' is returned.")
            return phase_flags, his_activation_counter, v_conv, t_act, term_msg

        # The NSR launch phase is not over yet, and there will be another iteration.
        else:
            # We set the convergence_volumes to the current volume, so we can compare the volumes of the next beat
            # with the current beat.
            v_conv = v
            return phase_flags, his_activation_counter, v_conv, t_act, term_msg

    """
    Code to assess the model state in the AF launch phase.
    """
    if phase_flags[1]:  # model.activation.flag_launch_af
        # First, before commiting the simulation to have at least one more heart beat in the AF launch phase, we check if
        # the 'AF launch phase' is over and whether we should start the real simulation.
        # const[2]: model.activation.num_beats_af_launch
        if his_activation_counter[1] == const[2]:
            # AF launch phase is over.
            phase_flags[1] = False  # model.activation.flag_launch_af
        else:  # his_activation_counter[1] < const[2]
            his_activation_counter[1] += 1
            return phase_flags, his_activation_counter, v_conv, t_act, term_msg

    """
    Code to assess the model state in the real simulation.
    """
    if phase_flags[2]:  # model.activation.flag_simulation
        # For the real simulation, we want to simulate 'num_beats' full heart beats. This means that the end of the last
        # beat is detected when a 'num_beats'+1 activation is initiated. But we just detect the beginning of the
        # 'num_beats'+1 beat here, but stop the simulation at this iteration, so that the extra beat is actually not
        # simulated.
        his_activation_counter[2] += 1
        # const[3]: model.activation.num_beats
        if his_activation_counter[2] > const[3]:
            his_activation_counter[2] -= 1
            phase_flags[2] = False  # model.activation.flag_simulation

    return phase_flags, his_activation_counter, v_conv, t_act, term_msg


@njit(cache=False)
def update_activation(nhat, t_act, ca_cap, phase_flags, hac, last_gauss, rr, vat, avn_states, avn_q, const, act_sigmas):
    """
    :param nhat: Time of next His-bundle activation (model.activation.next_his_activation_time).
    :param t_act: Matrix of activation times per MultiPatch model patch (model.activation.t_act).
    :param ca_cap: Capacity of the calcium storage, used in the contraction function. In this function if the oldest
        activation time in the t_act matrix is dropped to make space for a new activation, the remaining calcium storage
        will be added to the second oldest activation of that patch.
    :param phase_flags: Flags for the launch of the NSR and AF rhythm, and the real simulation
        phase_flags[0]: Flag for the 'NSR launch phase' (model.activation.flag_launch_nsr)
        phase_flags[1]: Flag for the 'AF launch phase' (model.activation.flag_launch_af)
        phase_flags[2]: Flag for the real simulation (model.activation.flag_simulation)
        phase_flags[3]: False if rhythm is 'NSR', True if rhythm is 'AF'
        phase_flags[4]: Flag to track whether the simulation was run successfully or was terminated. Not used in this
            function.
    :param hac: His activation counter (model.activation.his_activation_counter).
    :param last_gauss: The last Gaussian sample that was added to the activation time. We need to consider it when
        computing the next activation time for each patch.
    :param rr: RR series (model.activation.rr_series_af_launch, model.activation.rr_series).
    :param vat: VAT series (model.activation.vat_series_af_launch, model.activation.vat_series).
    :param avn_states: State variables of the AV node model (model.activation.rt, model.activation.dt,
        model.activation.aat, model.activation.node0_n_imp, model.activation.beats_to_draw).
    :param avn_q: Priority queue of the AV node model (model.activation.q).
    :param const:
        const[0:7]: model.activation.avn_rp
        const[7:14]: model.activation.avn_cd
        const[14:16]: model.activation.avn_resp
        const[16:28]: model.activation.p4dist
        const[28]: model.activation.rr_char[0]
        const[29]: model.activation.num_draw_beats_at_once
    :param act_sigmas: Standard deviations of the Gaussian samples added to the activation times of the MultiPatch model
        for each patch. Numpy array with one standard deviation for each patch.

    Note, if all three flags are False, then there will be no further model iterations,
    and we don't need to update the activation times.
    """
    if phase_flags[0]:  # model.activation.flag_launch_nsr
        if nhat <= 0:
            nhat += const[28]
        if np.any(t_act[:, -1] <= 0):
            mask = t_act[:, -1] <= 0
            # Move all rows one element to the left if the rightmost element has a value <= 0.
            t_act[mask] = np.roll(t_act[mask], -1)
            ca_cap[mask, 1] = np.sum(ca_cap[mask, 0:2], axis=1)
            ca_cap[mask, 0] = 0.0
            ca_cap[mask] = np.roll(ca_cap[mask], -1)
            # Overwrite the rightmost element now with a future activation time which is the activation time in the
            # second to rightmost column plus the RR mean. If a Gaussian sample is added with act_del[mask], then the
            # next time, the same values is subtracted with last_gauss[mask] to get the general activation time for that
            # chamber before adding a new Gaussian sample for that patch with act_del[mask].
            act_del = np.random.normal(loc=0, scale=1, size=act_sigmas.shape)
            act_del *= act_sigmas
            t_act[mask, -1] = t_act[mask, -2] + const[28] - last_gauss[mask] + act_del[mask]
            last_gauss[mask] = act_del[mask]

    elif phase_flags[1]:  # if model.activation.flag_launch_nsr is False and model.activation.flag_launch_af is True
        if nhat <= 0:
            if np.isnan(rr[int(hac[1])-1]):  # Draw the next RR interval if the current one has the value np.nan
                avn_states, avn_q, vat, rr = draw_next_rr_intervals(const, avn_states, avn_q, vat, rr)
            nhat += rr[int(hac[1])-1]
        if np.any(t_act[:, -1] <= 0):
            mask = t_act[:, -1] <= 0
            # Move all rows one element to the left if the rightmost element has a value <= 0.
            t_act[mask] = np.roll(t_act[mask], -1)
            ca_cap[mask, 1] = np.sum(ca_cap[mask, 0:2], axis=1)
            ca_cap[mask, 0] = 0.0
            ca_cap[mask] = np.roll(ca_cap[mask], -1)
            # Update all ventricular patches
            mask_v = mask.copy()
            mask_v[21:] = False
            if np.isnan(rr[int(hac[1])-1]):  # Draw the next RR interval if the current one has the value np.nan
                avn_states, avn_q, vat, rr = draw_next_rr_intervals(const, avn_states, avn_q, vat, rr)
            act_del = np.random.normal(loc=0, scale=1, size=act_sigmas.shape)
            act_del *= act_sigmas
            t_act[mask_v, -1] = t_act[mask_v, -2] + rr[int(hac[1]) - 1] - last_gauss[mask_v] + act_del[mask_v]
            last_gauss[mask_v] = act_del[mask_v]

            mask_a = mask.copy()
            mask_a[:21] = False
            aa_intervals = generate_pearson4(const[16:28], sum(mask_a))
            t_act[mask_a, -1] = t_act[mask_a, -2] + aa_intervals

    # if model.activation.flag_launch_nsr is False and model.activation.flag_launch_af is False and
    # model.activation.flag_simulation is True
    elif phase_flags[2]:
        if nhat <= 0:
            if sum(hac[1:3]) < vat.shape[0]:
                if np.isnan(rr[int(sum(hac[1:3])) - 1]):
                    avn_states, avn_q, vat, rr = draw_next_rr_intervals(const, avn_states, avn_q, vat, rr)
                nhat += rr[int(sum(hac[1:3])) - 1]
            else:
                # This rr mean was computed in the initialization using 200 RR intervals, and is just to get an end time
                # for the simulation.
                nhat += const[28]
        if np.any(t_act[:, -1] <= 0):
            mask = t_act[:, -1] <= 0
            # Move all rows one element to the left if the rightmost element has a value <= 0.
            t_act[mask] = np.roll(t_act[mask], -1)
            ca_cap[mask, 1] = np.sum(ca_cap[mask, 0:2], axis=1)
            ca_cap[mask, 0] = 0.0
            ca_cap[mask] = np.roll(ca_cap[mask], -1)

            if ~phase_flags[3]:  # rhythm == 'NSR'
                if sum(hac[1:3]) < vat.shape[0]:
                    act_del = np.random.normal(loc=0, scale=1, size=act_sigmas.shape)
                    act_del *= act_sigmas
                    t_act[mask, -1] = t_act[mask, -2] + rr[int(sum(hac[1:3])) - 1] - last_gauss[mask] + act_del[mask]
                    last_gauss[mask] = act_del[mask]
                else:
                    act_del = np.random.normal(loc=0, scale=1, size=act_sigmas.shape)
                    act_del *= act_sigmas
                    t_act[mask, -1] = t_act[mask, -2] + const[28] - last_gauss[mask] + act_del[mask]
                    last_gauss[mask] = act_del[mask]
            else:  # rhythm == 'AF'
                # Update all ventricular patches
                mask_v = mask.copy()
                mask_v[21:] = False
                if sum(hac[1:3]) < vat.shape[0]:
                    if np.isnan(rr[int(sum(hac[1:3])) - 1]):
                        avn_states, avn_q, vat, rr = draw_next_rr_intervals(const, avn_states, avn_q, vat, rr)
                    act_del = np.random.normal(loc=0, scale=1, size=act_sigmas.shape)
                    act_del *= act_sigmas
                    t_act[mask_v, -1] = (t_act[mask_v, -2] + rr[int(sum(hac[1:3])) - 1] - last_gauss[mask_v] +
                                         act_del[mask_v])
                    last_gauss[mask_v] = act_del[mask_v]
                else:
                    # This rr mean was computed in the initialization using 200 RR intervals, and is just to ensure that
                    # the next activation would happen after the end time of the simulation.
                    act_del = np.random.normal(loc=0, scale=1, size=act_sigmas.shape)
                    act_del *= act_sigmas
                    t_act[mask_v, -1] = t_act[mask_v, -2] + const[28] - last_gauss[mask_v] + act_del[mask_v]
                    last_gauss[mask_v] = act_del[mask_v]

                mask_a = mask.copy()
                mask_a[:21] = False
                aa_intervals = generate_pearson4(const[16:28], sum(mask_a))
                t_act[mask_a, -1] = t_act[mask_a, -2] + aa_intervals

    return nhat, t_act, ca_cap, last_gauss, rr, vat, avn_states, avn_q


@njit(cache=False)
def draw_next_rr_intervals(const, avn_states, avn_q, vat, rr):
    """
    This function is called when not all His-bundle activation times were drawn before the simulation started. In that
    case, we will draw the next RR interval(s) during the simulation with this code.
    """
    num_beats_to_draw = np.int64(min(avn_states[46], const[29]))
    avn_states[46] -= np.float64(num_beats_to_draw)
    vat_drawn, avn_states[:45], avn_q = (
        run_avn_model(const[:28], avn_states[:45], qu=avn_q, num_his_act=num_beats_to_draw,
                      n_aa=2))
    vat_start = np.argmax(np.isnan(vat))
    vat_end = vat_start + num_beats_to_draw
    vat[vat_start:vat_end] = vat_drawn
    rr[vat_start - 1:vat_end - 1] = np.diff(vat[vat_start - 1:vat_end])
    # special case where we pass an RR series for the real simulation, but we generate an RR series for the
    # AF launch phase.
    if avn_states[46] == 0 and vat_end != vat.shape[0]:
        # rr[vat_end - 1] = np.diff(vat[vat_end:vat_end + 2])
        rr[vat_end - 1] = vat[vat_end] - vat[vat_end - 1]
    return avn_states, avn_q, vat, rr



