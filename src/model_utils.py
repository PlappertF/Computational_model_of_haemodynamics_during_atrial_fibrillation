import numpy as np
import multiprocess
import matplotlib.pyplot as plt
import glob
import h5py
import pandas as pd
import seaborn as sns
import time
import os
import shutil
from collections import defaultdict
from scipy.stats import skew, kurtosis

# Import the utils module from the ocean package
from src.utils_fit import update_log, mad_outliers, mahalanobis_outliers

# Import cardiogrowth_py, needs to be added to search path in your Python environment
from src.cardiogrowth import CardioGrowth


def run_forest_run(wave, input_file, sim_dirs, constants=None, posterior=False,
                   log_file=None, n_processes=multiprocess.cpu_count() - 1, m_outlier=5.0,
                   run_sims=True, remove_outliers=False, percentile=0.95):
    """Wrapper function to run, analyze, and import model simulations"""

    # Specify directory to store simulations in
    file_path = wave.dir_sim

    # If running posterior simulations, pull according parameter sets
    if posterior:
        wave.x_sim = wave.x_posterior
        file_path = wave.dir.parent / wave.posterior_label
        remove_outliers = True
        update_log(wave.log_file, "\n---------\nPosterior\n---------")

    # Run models
    if run_sims:

        # Remove prior sim dir if rerunning simulations
        if run_sims:
            shutil.rmtree(file_path, ignore_errors=True)

        # Create sim directory
        os.makedirs(file_path, exist_ok=True)

        run_models_par(wave.x_sim, wave.x_names, file_path, input_file, constants=constants,
                       log_file=log_file, n_processes=n_processes)

    # Analyze model simulations
    analyze_model(file_path, wave.x_names, wave.y_names, constants=constants, m_outlier=m_outlier, log_file=log_file,
                  remove_outliers=remove_outliers, percentile=percentile)

    # Import results of all model simulations ran for current and all previous waves
    sim_dirs.append(file_path)
    x_sim, y_sim = import_model_results(sim_dirs, wave.x_names, wave.y_names)

    return x_sim, y_sim, sim_dirs


def run_models_par(x_model, x_names, file_path, input_file, constants=None,
                   n_processes=multiprocess.cpu_count() - 1, log_file=None):
    """
    Run model for all input parameter sets in x_model using parallel computing to reduce computational time
    """

    if constants is None:
        constants = {"rhythm": ['NSR']}

    update_log(log_file, "Running " + str(x_model.shape[0]) + " model simulations...")

    # Number of simulations to be run
    n_sims = x_model.shape[0]

    # Time simulation time
    t0 = time.time()

    with multiprocess.Pool(processes=n_processes) as pool:
        pool.starmap(run_model_par, list(zip(x_model, [x_names] * n_sims, np.arange(n_sims), [file_path] * n_sims,
                                             [input_file] * n_sims, [constants] * n_sims)))

    t1 = time.time() - t0

    update_log(log_file, "%i" % n_sims + " simulations completed in %.2f seconds" % t1)

    # Return total simulation time
    return t1


def run_model_par(x_model, x_names, i_x, file_path, input_file, constants={"rhythm": ['NSR']},
                  model_id0=0):
    """
    Run a single model for an input parameter set in x_model to calculate output parameters y: make sure they match
    with the user-defined input parameters in the main code.
    """

    # Construct dictionary of all model parameters
    pars_all = {x_names[i]: x_model[i] for i in range(len(x_names))}

    if 'NSR' in constants['rhythm']:
        # Initialize CardioGrowth
        cg = CardioGrowth(model_pars=input_file)

        # Export file name
        file_name = f'{model_id0 + i_x:05d}_nsr'

        # Update all parameters that don't have 'NSR' or 'AF' in their name
        cg.change_pars({key: val for key, val in pars_all.items() if 'NSR' not in key and 'AF' not in key})
        # Update all parameters that have 'NSR' in their name but remove the 'NSR' in the name
        cg.change_pars({key.replace(' NSR', ''): val for key, val in
                        {key: val for key, val in pars_all.items() if 'NSR' in key}.items()})
        # Update all constants that don't have 'NSR' or 'AF' in their name
        cg.change_pars({key: val for key, val in constants.items() if 'NSR' not in key and 'AF' not in key})

        cg.rhythm_is_a_dancer(rhythm='NSR', num_beats=3, num_beats_af_launch=None, avn_p4_params=None,
                              desired_rr_char=constants['HR NSR'], desired_vat_series=None,
                              patch_t_delay=np.concatenate([np.zeros(21), -120 * np.ones(2)]),  n_patches_atria=1,
                              keep_simulation_data='no_launch', use_converged=False, check_params=False,
                              file_path=file_path, file_name=file_name)

        # Save real (unscaled) parameter values x
        np.save(file_path / file_name, x_model)

    if 'AF' in constants['rhythm']:
        # Initialize CardioGrowth
        cg = CardioGrowth(model_pars=input_file)

        # Export file name
        file_name = f'{model_id0 + i_x:05d}_af'

        # Update all parameters that don't have 'NSR' or 'AF' in their name
        cg.change_pars({key: val for key, val in pars_all.items() if 'NSR' not in key and 'AF' not in key})
        # Update all parameters that have 'AF' in their name but remove the 'AF' in the name
        cg.change_pars({key.replace(' AF', ''): val for key, val in
                        {key: val for key, val in pars_all.items() if 'AF' in key}.items()})
        # Update all constants that don't have 'NSR' or 'AF' in their name
        cg.change_pars({key: val for key, val in constants.items() if 'NSR' not in key and 'AF' not in key})

        if 'CircChangeCirc_AF' in pars_all:
            cg.change_pars({'sympathetic_activity_circulation': pars_all['CircChangeCirc_AF']})

        if 'CircChangeTad_AF' in pars_all:
            cg.change_pars({'sympathetic_activity_tad': pars_all['CircChangeTad_AF']})

        rr_lookup = pd.read_pickle(
            os.getcwd()[:os.getcwd().rfind('/Computational_model_of_haemodynamics_during_atrial_fibrillation')] + '/Computational_model_of_haemodynamics_during_atrial_fibrillation/notebooks/RR_lookup.pkl')
        rr_lookup = rr_lookup[(abs(rr_lookup['HR target'] - constants['HR AF']) < 1.2) &
                              (abs(rr_lookup['RR rmssd target'] - 32.5) < 1) &
                              (abs(rr_lookup['RR sampen target'] - 1.7) < 0.025)]
        avn = np.squeeze(rr_lookup[['AVN' + str(i) for i in range(1, 22)]].values)
        rr = rr_lookup[['RR' + str(i) for i in range(1, 200)]].values
        vat = np.insert(np.cumsum(rr), 0, 0)

        cg.rhythm_is_a_dancer(rhythm="AF", num_beats=None, num_beats_af_launch=10, avn_p4_params=avn,
                              desired_rr_char=None, desired_vat_series=vat, n_patches_atria=20,
                              patch_t_delay=np.concatenate([np.zeros(21), -120 * np.ones(2)]),
                              keep_simulation_data='no_launch', use_converged=False, check_params=False,
                              file_path=file_path, file_name=file_name)

        # Save real (unscaled) parameter values x
        np.save(file_path / file_name, x_model)


def get_model_files(file_path, constants, file_extension=".hdf5"):
    """Return list if converged model output files and their corresponding parameter files"""

    # Get cardiogrowth simulation output files and temporarily remove the file extension
    sim_files = sorted(glob.glob(str(file_path) + "/*" + file_extension))
    sim_files = [sim_file.split(file_extension)[0] for sim_file in sim_files]

    # If both 'NSR' and 'AF' is in constants["rhythm"], then only keep the files for which we have a solution for both
    # rhythms
    if 'NSR' in constants["rhythm"] and 'AF' in constants["rhythm"]:
        # Step 1: Create a dictionary to store numbers and their associated endings
        ending_dict = defaultdict(set)

        # Step 2: Populate the dictionary
        for s in sim_files:
            number = s.split('_')[-2]  # Extract the number part
            ending = s.split('_')[-1]  # Extract the ending part
            ending_dict[number[-5:]].add(ending)

        # Step 3: Filter numbers that have both '_nsr' and '_af' endings
        valid_numbers = {k for k, v in ending_dict.items() if 'nsr' in v and 'af' in v}

        # Step 4: Reconstruct the list with valid strings
        sim_files = [s for s in sim_files if s.split('_')[-2][-5:] in valid_numbers]

    # Now I remove the '_nsr' and '_af' endings from the list and keep only one element per number
    filtered_strings = [s for s in sim_files if s.endswith('_af')]
    sim_files = [s[:-3] for s in filtered_strings]

    return sim_files


def analyze_model(file_path, x_labels, y_labels, sim_results_name="sim_results", constants=None,
                  m_outlier=None, file_extension=".hdf5", log_file=None, remove_outliers=False,
                  percentile=0.95):
    """
    Obtains results y_sim from all simulations with input parameters x previously stored in file_path. Unconverged solutions
    (which have no output file) are skipped and assigned NaNs. When plot_results=True, LV and RV PV loops, strain
    curves, and geometry at end-diastole are stored as well, but this makes it more time-consuming. Outliers, i.e. with
    outputs with standard deviation * m_std_exclude (default=1.96, i.e. 95% confidence interval), are omitted from analysis.
    """

    if constants is None:
        constants = {"rhythm": ['NSR']}

    # Find all converged simulations and corresponding parameter outputs
    sims = get_model_files(file_path, constants)
    update_log(log_file, str(len(sims)) + " Simulations reached convergence")

    # Pre-allocate arrays to store x and y.
    x_sims = np.empty((0, len(x_labels)))
    y_sims = np.empty((0, len(y_labels)))

    for sim in sims:
        # The npy file for 'NSR' and 'AF' contain the same model parameters, so I just need to load one of them
        if 'NSR' in constants["rhythm"]:
            x_sims = np.vstack((x_sims, np.load(sim + '_nsr.npy')))
        elif 'AF' in constants["rhythm"]:
            x_sims = np.vstack((x_sims, np.load(sim + '_af.npy')))

        # Collect output results
        y_sim = [i*0 for i in range(len(y_labels))]
        if 'NSR' in constants["rhythm"]:
            # Load model readout
            with h5py.File(sim + '_nsr.hdf5', "r", locking=False) as f:
                outputs_nsr = f['outputs'][0]
                output_names_nsr = list(f.attrs['outputs_names'])

            # Convert output names to lower case to prevent typographic mistakes from causing errors
            output_names_nsr = [output_name.lower() for output_name in output_names_nsr]

            for y_label_idx, y_label in enumerate(y_labels):
                if y_label[-3:] == 'NSR':
                    y_sim[y_label_idx] = outputs_nsr[output_names_nsr.index(y_label[:-4].lower())]

        if 'AF' in constants["rhythm"]:
            # Load model readout
            with h5py.File(sim + '_af.hdf5', "r", locking=False) as f:
                outputs_af = f['outputs'][0]
                output_names_af = list(f.attrs['outputs_names'])

            # Convert output names to lower case to prevent typographic mistakes from causing errors
            output_names_af = [output_name.lower() for output_name in output_names_af]

            for y_label_idx, y_label in enumerate(y_labels):
                if y_label[-2:] == 'AF':
                    y_sim[y_label_idx] = outputs_af[output_names_af.index(y_label[:-3].lower())]
                elif y_label[-6:] == 'Change':
                    y_sim[y_label_idx] = (outputs_af[output_names_af.index(y_label[:-7].lower())] -
                                          outputs_nsr[output_names_nsr.index(y_label[:-7].lower())])

        # Add all baseline outputs to stack
        y_sims = np.vstack((y_sims, y_sim))

    # Omit outliers, i.e. values outside m_std_outlier times the standard deviation from the median
    x_sims, y_sims = filter_outliers(x_sims, y_sims, m_outlier=m_outlier, percentile=percentile,
                                     remove=remove_outliers)
    update_log(log_file, str(x_sims.shape[0]) + " Simulations after outlier exclusion")

    # Export simulated x and y into csv file
    export_x_y(x_sims, y_sims, x_labels, y_labels, file_path, sim_results_name)

    update_log(log_file, str(x_sims.shape[0]) + " Simulations added to training data")

    return x_sims, y_sims


def export_x_y(x, y, x_labels, y_labels, file_path, file_name):
    """Export simulated or emulated x and into a csv file"""

    df = pd.DataFrame(np.concatenate((x, y), axis=1), columns=np.append(x_labels, y_labels))
    df.to_csv(file_path / str(file_name + ".csv"))


def import_model_results(sim_dirs, x_labels, y_labels, sim_results_name="sim_results"):
    """Import model results from all waves and scale y values to min and max of the simulated values"""

    # Find all simulation data, stored in .csv files
    sim_files = []
    for sim_dir in sim_dirs:
        sim_files.append(sim_dir / str(sim_results_name + '.csv'))

    df = pd.concat((pd.read_csv(f) for f in sim_files), ignore_index=True)

    x_sim = df[list(x_labels)].values
    y_sim = df[list(y_labels)].values

    return x_sim, y_sim


def filter_outliers(x, y, m_outlier=None, percentile=0.95, remove=False, sims=None, pars=None):
    """Omit outliers from simulation data using median absolute deviation and Mahalanobis distance"""

    # Initialize list of outliers indices
    outliers = []
    inliers = range(y.shape[0])

    # Step 1: MAD filtering
    if m_outlier is not None:
        if m_outlier > 0:
            outliers = mad_outliers(y, m_outlier=m_outlier)
            inliers = [i for i in range(y.shape[0]) if i not in outliers]

    # Step 2: Mahalanobis filtering, only analyze current inliers
    if percentile is not None:
        outliers_mahalanobis = mahalanobis_outliers(y[inliers, :], percentile=percentile)
    else:
        outliers_mahalanobis = []

    # Mahalanobis distance outliers are a subset of the MAD outliers, add to total set of outliers
    outliers.extend([inliers[i] for i in outliers_mahalanobis])

    # Remove outliers from x_sim and y_sim
    x = np.delete(x, outliers, axis=0)
    y = np.delete(y, outliers, axis=0)

    # Delete all outliers from simulation and parameter files if enabled
    if remove:
        if (sims is not None) and (pars is not None):
            for i in sorted(outliers, reverse=True):
                os.remove(sims[i])
                os.remove(pars[i])
                del sims[i]
                del pars[i]

    return x, y
