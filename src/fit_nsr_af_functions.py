import sys
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
from src.model_utils import get_model_files, export_x_y
import h5py

# Load the 'hrv_analysis_felix' branch from the cardiogrowth git repository
cardiogrowth_py_path = os.getcwd()[:os.getcwd().rfind('/cardiomatch')] + '/cardiogrowth_py'
sys.path.append(cardiogrowth_py_path)
from src.cardiogrowth import CardioGrowth


def gaussian_kernel(x):
    """Gaussian kernel function."""
    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2)


def compute_kde(data, x_values):
    """Compute the Kernel Density Estimate for the given data."""
    bandwidth = np.std(data) * len(data)**(-1/5)

    n = len(data)
    kde_values = np.zeros_like(x_values)

    for i in range(n):
        # Compute the kernel for each point in x_values
        kde_values += gaussian_kernel((x_values - data[i]) / bandwidth)

    # Normalize the KDE values
    kde_values /= np.max(kde_values)
    return kde_values


def set_calibration_target(patient):
    if patient == 'man':
        return pd.DataFrame([
            [0.0, 250.0, 126.83, 20.47],  # SBP NSR
            [0.0, 250.0, 130.0, 20.47],  # SBP AF
            [0.0, 200.0, 78.42, 20.47],  # DBP NSR
            [0.0, 200.0, 87.42, 20.47],  # DBP AF
            [0.0, 100.0, 32.42, 8.75],  # RVSP NSR
            [0.0, 100.0, 28.58, 8.75],  # RVSP AF
            [0.0, 100.0, 9.0, 8.75],  # RVDP NSR
            [0.0, 100.0, 7.92, 8.75],  # RVDP AF
            [0.0, 50.0, 15.5, 6.92],  # LAMP NSR
            [0.0, 50.0, 13.92, 6.92],  # LAMP AF
            [0.0, 50.0, 10.83, 6.92],  # RAMP NSR
            [0.0, 50.0, 10.08, 6.92],  # RAMP AF
            [0.0, 100.0, 53.87, 5.0],  # LAEF NSR
            [0.0, 100.0, 62.58, 5.0],  # LVEF NSR
            [79.0, 195.0, 133.0, 28.0],  # LVDV NSR
            [99.0, 240.0, 173.0, 36.0],  # RVDV NSR
            [44.0, 121.0, 81.0, 18.0],  # LAMAXV NSR
            [47.0, 162.0, 106.0, 26.0]  # RAMAXV NSR
            ],
            columns=['min', 'max', 'mu_baseline', 'sigma_baseline'],
            index=['SBP NSR', 'SBP AF', 'DBP NSR', 'DBP AF', 'RVSP NSR', 'RVSP AF', 'RVDP NSR', 'RVDP AF', 'LAMP NSR',
                   'LAMP AF', 'RAMP NSR', 'RAMP AF', 'LAEF NSR', 'LVEF NSR', 'LVDV NSR', 'RVDV NSR', 'LAMAXV NSR',
                   'RAMAXV NSR']
        )
    elif patient == 'woman':
        return pd.DataFrame([
            [0.0, 250.0, 134.4, 20.47],  # SBP NSR
            [0.0, 250.0, 135.0, 20.47],  # SBP AF
            [0.0, 200.0, 72.2, 20.47],  # DBP NSR
            [0.0, 200.0, 81.8, 20.47],  # DBP AF
            [0.0, 100.0, 35.8, 8.75],  # RVSP NSR
            [0.0, 100.0, 28.0, 8.75],  # RVSP AF
            [0.0, 100.0, 8.4, 8.75],  # RVDP NSR
            [0.0, 100.0, 6.2, 8.75],  # RVDP AF
            [0.0, 50.0, 16.4, 6.92],  # LAMP NSR
            [0.0, 50.0, 15.4, 6.92],  # LAMP AF
            [0.0, 50.0, 9.2, 6.92],  # RAMP NSR
            [0.0, 50.0, 10.2, 6.92],  # RAMP AF
            [0.0, 100.0, 50.44, 5.0],  # LAEF NSR
            [0.0, 100.0, 64.8, 5.0],  # LVEF NSR
            [73.0, 147.0, 105.0, 19.0],  # LVDV NSR
            [83.0, 180.0, 130.0, 24.0],  # RVDV NSR
            [43.0, 99.0, 67.0, 14.0],  # LAMAXV NSR
            [49.0, 126.0, 81.0, 18.0]  # RAMAXV NSR
            ],
            columns=['min', 'max', 'mu_baseline', 'sigma_baseline'],
            index=['SBP NSR', 'SBP AF', 'DBP NSR', 'DBP AF', 'RVSP NSR', 'RVSP AF', 'RVDP NSR', 'RVDP AF', 'LAMP NSR',
                   'LAMP AF', 'RAMP NSR', 'RAMP AF', 'LAEF NSR', 'LVEF NSR', 'LVDV NSR', 'RVDV NSR', 'LAMAXV NSR',
                   'RAMAXV NSR']
        )
    else:
        raise ValueError("Patient needs to be 'man' or 'woman'.")


def set_simulation_free_parameters():
    pars = {}
    # Set the stressed blood volume as free parameter
    pars['SBV'] = {'limits': [100, 2500]}

    # Set the active stress coefficient for the ventricles 'SAct' and for the atria 'SAct_a' as free parameters
    pars['SAct'] = {'limits': [0.1, 0.6]}
    pars['SAct_a'] = {'limits': [0.1, 0.6]}

    # Set the systemic arterial resistance 'Ras' that is placed between 'Cas' and 'Cvs' as a free parameter
    pars['Ras'] = {'limits': [0.5, 2.0]}  # Taken from Figure 3 in Reale_1965_Acute Effects of Countershock
    # Scaling factor for systemic capacitance 'Cas' and 'Cvs' as free parameter. This is to set how much blood the systemic arteries and veins can hold relative to the systemic blood pressure.
    pars['scale_Cs'] = {'limits': [0.8, 1.2]}
    # Set the pulmonary arterial resistance 'Rap' that is placed between 'Cap' and 'Cvp' as a free parameter
    pars['Rap'] = {'limits': [0.02, 0.6]}  # Taken from Figure 3 in Reale_1965_Acute Effects of Countershock
    # Scaling factor for pulmonary capacitance 'Cap' and 'Cvp' as free parameter. This is to set how much blood the pulmonary arteries and veins can hold relative to the pulmonary blood pressure.
    pars['scale_Cp'] = {'limits': [0.8, 1.2]}

    # Set the duration of the contraction for the ventricles 'tad' and for the atria 'tad_a' as free parameters
    pars['tad'] = {'limits': [75, 500]}
    pars['tad_a'] = {'limits': [75, 500]}

    # Set the atrial and ventricular unstressed reference wall areas
    pars['AmRefLfw'] = {'limits': [4000, 11000]}
    pars['AmRefRfw'] = {'limits': [11000, 22000]}
    pars['AmRefSw'] = {'limits': [3500, 9000]}
    pars['AmRefLA'] = {'limits': [4500, 8500]}
    pars['AmRefRA'] = {'limits': [5500, 12500]}

    # Set the scaling factor for the change between NSR and AF for the circulation resistance and capacitance as free parameter
    #   Factor change of Ras, Rap, Cvp, Cas, Cvs, Cap from NSR to AF due to change in ANS activity. Vascular resistance increases in Head-up_tilt Wieling_1998. Vascular capacitance decreases in Head-up tilt Svec_2021
    pars['CircChangeCirc_AF'] = {'limits': [0.6, 1.4]}
    # Set the scaling factor for the change between NSR and AF for the contraction duration as free parameter
    pars['CircChangeTad_AF'] = {'limits': [0.6, 1.4]}
    return pars


def set_simulation_model_constants(patient):
    constants = {}
    # Set the rhythms
    constants['rhythm'] = ['NSR', 'AF']
    # Set the heart rates
    constants['HR NSR'] = 60.0
    constants['HR AF'] = 100.0

    # Set the right atrial wall thickness
    constants['RAWth'] = 2.7  # From Varela.2017  doi: 10.1109/TMI.2017.2671839
    # Set the left atrial wall thickness
    constants['LAWth'] = 2.4  # From Varela.2017  doi: 10.1109/TMI.2017.2671839
    # Set the right ventricular free wall wall thickness
    constants['RfWth'] = 3.4  # From Matsukubo.1977  doi: 10.1161/01.cir.56.2.278
    # Set the left ventricular free wall and septal wall thicknesses in the 16 AHA segment format
    if patient == 'man':
        constants['LVWth'] = [9.21, 9.36, 8.5, 8.32, 8.68, 8.27, 7.27, 7.59, 8.10, 7.37, 7.56, 7.48, 7.53, 7.00, 6.66,
                              7.46]  # From Walpot.2019  doi: 10.1148/ryct.2019190034
        # To check, here I write which segments belong to the left free wall (lfw) and which segments belong to the septal wall (sw).
        # segment = ['lfw', 'sw', 'sw', 'lfw', 'lfw', 'lfw', 'lfw', 'sw', 'sw', 'lfw', 'lfw', 'lfw', 'lfw', 'sw', 'lfw', 'lfw']
    elif patient == 'woman':
        constants['LVWth'] = [7.47, 7.95, 7.22, 6.89, 7.03, 6.76, 5.95, 6.078, 6.65, 6.18, 6.01, 6.11, 6.51, 5.61, 5.13,
                 6.05]  # From Walpot.2019  doi: 10.1148/ryct.2019190034
        # To check, here I write which segments belong to the left free wall (lfw) and which segments belong to the septal wall (sw).
        # segment = ['lfw', 'sw', 'sw', 'lfw', 'lfw', 'lfw', 'lfw', 'sw', 'sw', 'lfw', 'lfw', 'lfw', 'lfw', 'sw', 'lfw', 'lfw']
    else:
        raise ValueError("Patient needs to be 'man' or 'woman'.")

    # Set ventricular and atrial rise and delay time
    constants['tr_a'] = 0.55
    constants['td_a'] = 0.55
    constants['tr'] = 0.37
    constants['td'] = 0.37
    return constants


def format_axis(ax, y_axis_limits, y_axis_ticks, SB_width1, fontsize=8, axis_color='k', axis_width=1):
    # Set X axis
    ax.spines['bottom'].set_visible(False)
    ax.set_xlim(-3.5, -0.5)
    ax.set_xticks(-(np.arange(1, 3)), ['man', 'woman'], rotation=270, fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize, which='both')

    # Set Y axis
    ax.spines['left'].set_linewidth(axis_width)
    ax.spines['left'].set_color(axis_color)
    ax.set_ylim(y_axis_limits[0], y_axis_limits[1])
    if y_axis_limits[1] > 100:
        ax.set_yticks([y_axis_ticks[0], 100, y_axis_ticks[1]])
        ax.yaxis.tick_left()
    else:
        ax.set_yticks(y_axis_ticks)
        ax.yaxis.tick_left()
    for label in ax.get_yticklabels():
        label.set_verticalalignment('center')
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')

    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def get_kde(data, y_axis_lims, SB_width2):
    # Initialize x values
    x = np.linspace(y_axis_lims[0], y_axis_lims[1], 500)

    # Add extra x values to illustrate a cutoff at min and max data values
    x = np.insert(x, np.argmax(x[x < np.max(data)]) + 1, np.max(data))
    x = np.insert(x, np.argmax(x[x < np.max(data)]) + 2, np.max(data) + 1e-5)
    x = np.insert(x, np.argmax(x[x < np.min(data)]) + 1, np.min(data))
    x = np.insert(x, np.argmax(x[x < np.min(data)]) + 1, np.min(data) - 1e-5)

    # Compute the y values
    y = compute_kde(data, x) * SB_width2 / 2
    # Set the kde values below the min and max data values to zero
    y[(x < np.min(data)) | (x > np.max(data))] = 0

    # Add an extra 0 at the beginning and end of y, in case the kde is not zero at the end.
    x = np.insert(x, 0, 0)
    x = np.append(x, y_axis_lims[1])
    y = np.insert(y, 0, 0)
    y = np.append(y, 0)
    return x, y


def plot_kde(ax, sim_results, y_axis_output, y_axis_lims, patient, SB_width1, SB_width2, offset, color):
    # Check if 'SBP NSR' is in the sim_results
    for y_axis_out in y_axis_output:
        if y_axis_out + ' NSR' in sim_results.columns:
            data = sim_results[y_axis_out + ' NSR'].values
            x_values, kde_values = get_kde(data, y_axis_lims, SB_width2)
            ax.plot(-(-kde_values + patient + 1 - SB_width1/2-offset), x_values, label='KDE', color='k', linewidth=0.5)
            ax.fill_between(-(-kde_values + patient + 1 - SB_width1/2-offset), x_values, alpha=0.5, color=color, linewidth=0)
        if y_axis_out + ' AF' in sim_results.columns:
            data = sim_results[y_axis_out + ' AF'].values
            x_values, kde_values = get_kde(data, y_axis_lims, SB_width2)
            ax.plot(-(kde_values + patient + 1 + SB_width1/2+offset), x_values, label='KDE', color='k', linewidth=0.5)
            ax.fill_between(-(kde_values + patient + 1 + SB_width1/2+offset), x_values, alpha=0.5, color=color, linewidth=0)
        if offset > 0:
            line, = ax.plot(
                [-(patient - SB_width1 / 2-offset), -(patient - SB_width1 / 2-offset)],
                [-1000, 1000], color='k', linewidth=1 / 2)
            line.set_solid_capstyle('butt')
            line, = ax.plot(
                [-(patient + SB_width1 / 2+offset), -(patient + SB_width1 / 2+offset)],
                [-1000, 1000], color='k', linewidth=1 / 2)
            line.set_solid_capstyle('butt')


def prepare_posterior_compare(dir_calibration):
    # Compare if the best simulation result is the same if I first exclude parameter sets that don't have an implausibility criterion of I<2
    pars = set_simulation_free_parameters()
    x_names = []
    for i_par, key in enumerate(pars):
        x_names.append(key)
    y_names = ['SBP NSR', 'SBP AF', 'DBP NSR', 'DBP AF', 'RVSP NSR', 'RVSP AF', 'RVDP NSR', 'RVDP AF', 'LAMP NSR',
               'LAMP AF', 'RAMP NSR', 'RAMP AF', 'LAEF NSR', 'LAEF AF', 'RAEF NSR', 'RAEF AF', 'LVEF NSR', 'LVEF AF',
               'RVEF NSR', 'RVEF AF', 'LAMAXV NSR', 'LAMAXV AF', 'RAMAXV NSR', 'RAMAXV AF', 'LVDV NSR', 'LVDV AF',
               'RVDV NSR', 'RVDV AF', 'CO NSR', 'CO AF']

    for patient in ['man', 'woman']:
        model_constants = set_simulation_model_constants(patient)

        if model_constants is None:
            constants = {"rhythm": ['NSR']}

        # Find all converged simulations and corresponding parameter outputs
        sims = get_model_files(dir_calibration / patient / 'Posterior', model_constants)

        # Pre-allocate arrays to store x and y.
        x_sims = np.empty((0, len(x_names)))
        y_sims = np.empty((0, len(y_names)))

        # Check for presence of acute simulations
        y_labels_acute = [y_label for y_label in y_names if "_acute" in y_label]
        y_labels = [y_label for y_label in y_names if "_acute" not in y_label]

        for sim in sims:
            # The npy file for 'NSR' and 'AF' contain the same model parameters, so I just need to load one of them
            if 'NSR' in model_constants["rhythm"]:
                x_sims = np.vstack((x_sims, np.load(sim + '_nsr.npy')))
            elif 'AF' in model_constants["rhythm"]:
                x_sims = np.vstack((x_sims, np.load(sim + '_af.npy')))

            # Collect output results
            y_sim = [i * 0 for i in range(len(y_labels))]
            if 'NSR' in model_constants["rhythm"]:
                # Load model readout
                with h5py.File(sim + '_nsr.hdf5', "r", locking=False) as f:
                    outputs_nsr = f['outputs'][0]
                    output_names_nsr = list(f.attrs['outputs_names'])

                # Convert output names to lower case to prevent typographic mistakes from causing errors
                output_names_nsr = [output_name.lower() for output_name in output_names_nsr]

                for y_label_idx, y_label in enumerate(y_labels):
                    if y_label[-3:] == 'NSR':
                        y_sim[y_label_idx] = outputs_nsr[output_names_nsr.index(y_label[:-4].lower())]

            if 'AF' in model_constants["rhythm"]:
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

            # Add all baseline and acute outputs to stack
            y_sims = np.vstack((y_sims, y_sim))

        if not os.path.exists(dir_calibration / patient / "Posterior_compare"):
            os.makedirs(dir_calibration / patient / "Posterior_compare")
        export_x_y(x_sims, y_sims, x_names, y_names + y_labels_acute, dir_calibration / patient / "Posterior_compare",
                   "sim_results")


def plot_calibration(dir_calibration):
    prepare_posterior_compare(dir_calibration)

    # Subplot with 4 rows and 1 column
    fontsize = 8
    SB_width1 = 0.5
    SB_width2 = 0.25
    SB_width3 = 0.5

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(20 / 2.54, 17 / 2.54), gridspec_kw={
        'height_ratios': [np.sqrt(190 - 23), np.sqrt(60 - 0), np.sqrt(35 - 0), np.sqrt(20 - 0)]})
    plt.subplots_adjust(hspace=0.3)
    # Set x-axis linewidth and color
    y_axis_limits = np.array([[37, 190], [0, 58], [0, 36], [0, 21]])
    y_axis_limits = np.array([[-5, 250], [-5, 100], [-5, 100], [-5, 100]])
    y_axis_limits_sims = np.array([[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]])
    y_axis_ticks_clin = np.array([[np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf], [np.inf, -np.inf]])

    # Plot the data
    y_axis_list = [['SBP', 'DBP'], ['RVSP', 'RVDP'], ['LAMP'], ['RAMP']]

    num_sim_est = np.zeros(len(['man', 'woman']))
    num_sim_kde = np.zeros(len(['man', 'woman']))
    for i, patient in enumerate(['man', 'woman']):
        clin_target = get_clin_target(patient)
        sim_results = pd.read_csv(str(dir_calibration / patient / "Posterior_compare/sim_results.csv"))
        sim_results_kde = pd.read_csv(str(dir_calibration / patient / 'Posterior/sim_results.csv'))
        num_sim_est[i] = sim_results.shape[0]
        num_sim_kde[i] = sim_results_kde.shape[0]
        # Create an empty dataframe
        sim_results_with_ECG = pd.DataFrame()
        sim_results_with_ECG_kde = pd.DataFrame()
        for axis, (y_axis_output, y_axis_lims) in enumerate(zip(y_axis_list, y_axis_limits)):
            axs[axis].add_patch(
                patches.Rectangle((-(i + 1 - SB_width1 / 2) - SB_width1, y_axis_limits[axis, 0]), SB_width1,
                                  y_axis_limits[axis, 1] - y_axis_limits[axis, 0], linewidth=1, edgecolor='none',
                                  facecolor=[0.95, 0.95, 0.95]))
            plot_kde(axs[axis], sim_results_kde, y_axis_output, y_axis_lims, i, SB_width1, SB_width2, 0,
                     plt.cm.tab10.colors[0])
            if not sim_results_with_ECG_kde.empty:
                plot_kde(axs[axis], sim_results_with_ECG_kde, y_axis_output, y_axis_lims, i, SB_width1, SB_width2, 0.2,
                         plt.cm.tab10.colors[3])
            plotdata(clin_target, y_axis_output, i, axs[axis], SB_width1, SB_width3, 'Clinical', marker_size=4)
            sim_res_without_ECG = sim_results.copy()
            sim_res_without_ECG.reset_index(drop=True, inplace=True)
            clin_normalize = pd.DataFrame(
                {'LAMAXV NSR mean': [56.24], 'LAMAXV NSR std': [14.75], 'RAMAXV NSR mean': [40.76],
                 'RAMAXV NSR std': [13.24]}, index=['population'])
            best_sim, best_sim_idx = get_best_simulation(sim_results, clin_target, clin_normalize)
            plotdata(best_sim, y_axis_output, i, axs[axis], SB_width1, SB_width3, 'Sim_NoECG', marker_size=4)
            # Check if dataframe is empty
            outputs_to_check_sim_without_ECG = [x + ' NSR' for x in y_axis_output if
                                                x + ' NSR' in sim_results_kde.columns] + [x + ' AF' for x in
                                                                                          y_axis_output if
                                                                                          x + ' AF' in sim_results_kde.columns]
            if not sim_results_kde[outputs_to_check_sim_without_ECG].empty:
                y_axis_limits_sims[axis, 0] = min(y_axis_limits_sims[axis, 0],
                                                  np.min(sim_results_kde[outputs_to_check_sim_without_ECG].values))
                y_axis_limits_sims[axis, 1] = max(y_axis_limits_sims[axis, 1],
                                                  np.max(sim_results_kde[outputs_to_check_sim_without_ECG].values))

    y_axis_ticks_clin = np.round(y_axis_ticks_clin)
    y_axis_limits_sims[:, 0] = np.floor(y_axis_limits_sims[:, 0], y_axis_ticks_clin[:, 0]) - 1
    y_axis_limits_sims[:, 1] = np.ceil(y_axis_limits_sims[:, 1], y_axis_ticks_clin[:, 1]) + 1
    y_axis_limits_sims[1:4, 0] = 0

    for axis in range(4):
        format_axis(axs[axis], y_axis_limits_sims[axis, :], y_axis_ticks_clin[axis, :], np.zeros((4, 1)), SB_width1)
        axs[axis].text(-(1 - SB_width1 / 2), y_axis_limits_sims[axis, 1] - 1.4 * np.sqrt(
            y_axis_limits_sims[axis, 1] - y_axis_limits_sims[axis, 0]), 'NSR', ha='center', va='top', fontsize=fontsize,
                       rotation=270, zorder=12)
        axs[axis].text(-(1 + SB_width1 / 2), y_axis_limits_sims[axis, 1] - 1.4 * np.sqrt(
            y_axis_limits_sims[axis, 1] - y_axis_limits_sims[axis, 0]), 'AF', ha='center', va='top', fontsize=fontsize,
                       rotation=270, zorder=12)
    axs[0].set_xlabel('Patient', fontsize=fontsize)
    axs[0].set_ylabel('DBP & SBP (mmHg)', fontsize=fontsize, rotation=270, labelpad=8)
    axs[1].set_ylabel('RVDP & RVSP (mmHg)', fontsize=fontsize, rotation=270, labelpad=13)
    axs[2].set_ylabel('LAP (mmHg)', fontsize=fontsize, rotation=270, labelpad=13)
    axs[3].set_ylabel('RAP (mmHg)', fontsize=fontsize, rotation=270, labelpad=13)

    clin_leg = axs[0].plot([-100, -99], [-100, -99], color='k', linewidth=1)
    sim_leg1 = axs[0].plot([-100, -99], [-100, -99], color=plt.cm.tab10.colors[0], linewidth=1)

    axs[0].legend([clin_leg[0], sim_leg1[0]], ['Clinical measurement', 'Simulation'], loc='best', fontsize=fontsize,
                  frameon=True, ncol=3, framealpha=1)

    for axis in range(4):
        axs[axis].invert_yaxis()
        axs[axis].xaxis.tick_top()
        axs[axis].xaxis.set_label_position('top')

    axs[0].set_xticks([-1, -2], ['Man', 'Woman'], fontsize=fontsize, ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def get_clin_target(patient):
    if patient == 'man':
        return pd.DataFrame(
            {'Sex': ['man'], 'SBP NSR': [126.83], 'SBP AF': [130.0], 'DBP NSR': [78.42], 'DBP AF': [87.42],
             'RVSP NSR': [32.42], 'RVSP AF': [28.58], 'RVDP NSR': [9.0], 'RVDP AF': [7.92], 'LAMP NSR': [15.5],
             'LAMP AF': [13.92], 'RAMP NSR': [10.83], 'RAMP AF': [10.08], 'LAEF NSR': [53.87], 'LVEF NSR': [62.58],
             'LAMAXV NSR': [60.58], 'RAMAXV NSR': [44.92]}, index=['man'])
    elif patient == 'woman':
        return pd.DataFrame(
            {'Sex': ['woman'], 'SBP NSR': [134.4], 'SBP AF': [135.0], 'DBP NSR': [72.2], 'DBP AF': [81.8],
             'RVSP NSR': [35.8], 'RVSP AF': [28.0], 'RVDP NSR': [8.4], 'RVDP AF': [6.2], 'LAMP NSR': [16.4],
             'LAMP AF': [15.4], 'RAMP NSR': [9.2], 'RAMP AF': [10.2], 'LAEF NSR': [50.44], 'LVEF NSR': [64.8],
             'LAMAXV NSR': [45.8], 'RAMAXV NSR': [30.8]}, index=['woman'])
    else:
        raise ValueError('patient must be either "man" or "woman".')


def plotdata(data, y_axis_output, patient, ax, SB_width1, SB_width3, data_type, marker_size=4):
    if data_type == 'Clinical':
        color, marker_file = 'k', 'NSR_clin.svg'
    elif data_type == 'Sim_NoECG':
        color, marker_file = plt.cm.tab10.colors[0], 'NSR_sim1.svg'
    left_marker, right_marker = load_marker(marker_file, data_type)
    if data_type == 'Clinical':
        for y_axis_out in y_axis_output:
            if y_axis_out + ' NSR' in data.columns.values or y_axis_out + ' AF' in data.columns.values:
                if y_axis_out + ' NSR' in data.columns.values:
                    x_NSR, y_NSR = patient + 1 - SB_width1 / 2 * SB_width3, data[y_axis_out + ' NSR'].values[0]
                    ax.plot(-x_NSR, y_NSR, marker=right_marker, markersize=marker_size, color=color, linestyle='None')
                else:
                    x_NSR, y_NSR = patient + 1, data[y_axis_out + ' AF'].values[0]
                if y_axis_out + ' AF' in data.columns.values:
                    x_AF, y_AF = patient + 1 + SB_width1 / 2 * SB_width3, data[y_axis_out + ' AF'].values[0]
                    ax.plot(-x_AF, y_AF, marker=left_marker, markersize=marker_size, color=color, linestyle='None')
                else:
                    x_AF, y_AF = patient + 1, data[y_axis_out + ' NSR'].values[0]
                ax.plot([-x_NSR, -x_AF], [y_NSR, y_AF], label='KDE', color=color, linewidth=1)
    elif data_type == 'Sim_NoECG':
        for y_axis_out in y_axis_output:
            if y_axis_out + ' NSR' in data.index.values or y_axis_out + ' AF' in data.index.values:
                if y_axis_out + ' NSR' in data.index.values:
                    x_NSR, y_NSR = patient + 1 - SB_width1 / 2 * SB_width3, data[y_axis_out + ' NSR']
                    ax.plot(-x_NSR, y_NSR, marker=right_marker, markersize=marker_size, color=color,
                            linestyle='None')
                else:
                    x_NSR, y_NSR = patient + 1, data[y_axis_out + ' AF']
                if y_axis_out + ' AF' in data.index.values:
                    x_AF, y_AF = patient + 1 + SB_width1 / 2 * SB_width3, data[y_axis_out + ' AF']
                    ax.plot(-x_AF, y_AF, marker=left_marker, markersize=marker_size, color=color,
                            linestyle='None')
                else:
                    x_AF, y_AF = patient + 1, data[y_axis_out + ' NSR']
                ax.plot([-x_NSR, -x_AF], [y_NSR, y_AF], label='KDE', color=color, linewidth=1)


def load_marker(file, data_type):
    planet_path, attributes = svg2paths(file)
    planet_marker = parse_path(attributes[0]['d'])
    planet_marker.vertices -= (planet_marker.vertices.max(axis=0) + planet_marker.vertices.min(axis=0)) / 2
    if data_type == 'Clinical':
        planet_marker.vertices[:, 0] -= planet_marker.vertices[:, 0].min()
    elif data_type == 'Sim_NoECG' or data_type == 'Sim_ECG':
        planet_marker.vertices[:, 0] -= planet_marker.vertices[:, 0].max()
    planet_marker.vertices /= planet_marker.vertices.max()
    planet_marker = planet_marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
    planet_marker = planet_marker.transformed(mpl.transforms.Affine2D().scale(-1, 1))
    planet_marker_rot = planet_marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
    return planet_marker, planet_marker_rot


def plot_volume_pressure(model, compartment, legend):
    # Create figure with 2 vertical subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # First subplot - Sine wave
    ax1.plot(model.activation.time * model.solver.dt /1000, model.volumes[:, compartment], 'b-', linewidth=2, label=legend + ' volume')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel(legend + ' volume (mL)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # Second subplot - Cosine wave
    ax2.plot(model.activation.time * model.solver.dt /1000, model.pressures[:, compartment], 'r-', linewidth=2, label=legend + ' pressure')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel(legend + ' pressure (mmHg)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()


def get_best_simulation(sim_data, clin_data, clin_normalize, w_ap=0.35, w_sp=0.1, w_v=5, w_e=0.1):
    # Compute the cost function component for the absolute pressure difference
    cost_abs_pres_nsr = np.zeros(len(sim_data))
    cost_abs_pres_af = np.zeros(len(sim_data))
    output_clin_counter_nsr = 0
    output_clin_counter_af = 0
    for output in ['SBP NSR', 'DBP NSR', 'RVSP NSR', 'RVDP NSR', 'LAMP NSR', 'RAMP NSR']:
        if not np.isnan(clin_data[output].values[0]):
            output_clin_counter_nsr += 1
            cost_abs_pres_nsr += abs(sim_data[output] - clin_data[output].values[0]).values ** 2
    for output in ['SBP AF', 'DBP AF', 'RVSP AF', 'RVDP AF', 'LAMP AF', 'RAMP AF']:
        if not np.isnan(clin_data[output].values[0]):
            output_clin_counter_af += 1
            cost_abs_pres_af += abs(sim_data[output] - clin_data[output].values[0]).values ** 2
    if output_clin_counter_nsr + output_clin_counter_af > 0:
        cost_abs_pres_nsr /= output_clin_counter_nsr
        cost_abs_pres_af /= output_clin_counter_af

        # Compute the cost function component for the pressure slope difference
        cost_slope_pres = np.zeros(len(sim_data))
        output_clin_counter = 0
        for output in ['SBP', 'DBP', 'RVSP', 'RVDP', 'LAMP', 'RAMP']:
            if not np.isnan(clin_data[output + ' NSR'].values[0]) and not np.isnan(clin_data[output + ' AF'].values[0]):
                cost_slope_pres += (abs((sim_data[output + ' AF'] - sim_data[output + ' NSR']) - (
                        clin_data[output + ' AF'].values[0] - clin_data[output + ' NSR'].values[0])).values) ** 2
                output_clin_counter += 1
        cost_slope_pres /= output_clin_counter

    # Compute the cost function component for the volumes
    cost_volumes = np.zeros(len(sim_data))
    output_clin_counter = 0
    for output in ['LAMAXV NSR', 'RAMAXV NSR']:
        if output == 'LAMAXV NSR':
            # Add the left atrial volume to the outputs to be fitted. I assume that the atrial volumes presented in Fuchs.2016 are the maximum atrial volumes in NSR.
            if clin_data['Sex'].values[0] == 'man':
                vol_mean = 81.0
                vol_std = 18.0
            else:  # df_patient['Sex'] == 'woman'
                vol_mean = 67.0
                vol_std = 14.0

        if output == 'RAMAXV NSR':
            # Add the right atrial volume to the outputs to be fitted.
            if clin_data['Sex'].values[0] == 'man':
                vol_mean = 106.0
                vol_std = 26.0
            else:  # df_patient['Sex'] == 'woman'
                vol_mean = 81.0
                vol_std = 18.0

        if not np.isnan(clin_data[output].values[0]):
            cost_volumes += (abs((sim_data[output] - vol_mean) / vol_std - (
                    clin_data[output].values[0] - clin_normalize[output + ' mean'].values[0]) / clin_normalize[
                                     output + ' std'].values[0]).values) ** 2
            output_clin_counter += 1
    if output_clin_counter > 0:
        cost_volumes /= output_clin_counter

    # Compute the cost function component for the ejection fraction
    cost_ef = np.zeros(len(sim_data))
    output_clin_counter = 0
    for output in ['LVEF NSR', 'LAEF NSR']:
        if not np.isnan(clin_data[output].values[0]):
            cost_ef += abs(sim_data[output] - clin_data[output].values[0]).values ** 2
            output_clin_counter += 1
    cost_ef /= output_clin_counter

    total_cost = w_ap * cost_abs_pres_nsr + w_ap * cost_abs_pres_af + w_sp * cost_slope_pres + w_v * cost_volumes + w_e * cost_ef

    return sim_data.loc[np.argmin(total_cost)], np.argmin(total_cost)
