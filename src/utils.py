import numpy as np
import pandas as pd

import src.heart as heart


def get_outputs(model):
    """Collect model outputs in Pandas dataframe."""
    # Arterial function
    sbp = np.mean(model.post_processing.pressures['sbp'])  # Systolic blood pressure
    dbp = np.mean(model.post_processing.pressures['dbp'])  # Diastolic blood pressure

    # Right ventricular pressure
    rvsp = np.mean(model.post_processing.pressures['rvsp'])  # Right ventricular systolic pressure
    rvdp = np.mean(model.post_processing.pressures['rvdp'])  # Right ventricular diastolic pressure
    rvedp = np.mean(model.post_processing.pressures['rvedp'])  # Right ventricular end-diastolic pressure

    # Left atrial pressure
    lamp = model.post_processing.pressures['lamp']  # Left atrial mean pressure

    # Right atrial pressure
    ramp = model.post_processing.pressures['ramp']  # Right atrial mean pressure

    # Left ventricular volume
    lvef = np.mean(model.post_processing.volumes['lvef'])  # Left ventricular ejection fraction
    lvdv = np.mean(model.post_processing.volumes['lvdv'])  # Left ventricular diastolic volume

    # Right ventricular volume
    rvef = np.mean(model.post_processing.volumes['rvef'])  # Right ventricular ejection fraction
    rvdv = np.mean(model.post_processing.volumes['rvdv'])  # Right ventricular diastolic volume

    # Left atrial volume
    lamaxv = np.mean(model.post_processing.volumes['lamaxv'])  # Left atrial maximum volume
    laminv = np.mean(model.post_processing.volumes['laminv'])  # Left atrial minimum volume
    lastrokev = np.mean(model.post_processing.volumes['lastrokev'])  # Left atrial total stroke volume
    laef = np.mean(model.post_processing.volumes['laef'])  # Left atrial ejection fraction

    # Right atrial volume
    ramaxv = np.mean(model.post_processing.volumes['ramaxv'])  # Right atrial maximum volume
    raminv = np.mean(model.post_processing.volumes['raminv'])  # Right atrial minimum volume
    rastrokev = np.mean(model.post_processing.volumes['rastrokev'])  # Right atrial total stroke volume
    raef = np.mean(model.post_processing.volumes['raef'])  # Right atrial ejection fraction

    # Cardiac output
    co = model.post_processing.co_lv  # Cardiac output

    # Time of first mitral valve closure
    ied = model.post_processing.time_events['mv_closes'][0]

    return pd.DataFrame(data=[[sbp, dbp,  # Arterial function
                               rvsp, rvdp, rvedp, # Right ventricular pressure
                               lvef,  # Left ventricular volume
                               rvef,  # Right ventricular volume
                               lamp, # Left atrial pressure
                               ramp, # Right atrial pressure, lapreawv, lav_middle,
                               lvdv, # Left ventricular volume
                               rvdv, # Right ventricular volume
                               laminv, lamaxv, lastrokev, laef,  # Left atrial volume
                               raminv, ramaxv, rastrokev, raef,  # Right atrial volume
                               co,  # Cardiac output
                               ied], ],
                        columns=['SBP', 'DBP', # Arterial function
                                 'RVSP', 'RVDP', 'RVEDP', # RV pressure
                                 'LVEF',  # Left ventricular volume
                                 'RVEF',  # Right ventricular volume
                                 'LAMP', # Left atrial pressure
                                 'RAMP',  # Right atrial pressure ,'LAPREAWV', 'LAVMiddle',
                                 'LVDV',  # Left ventricular volume
                                 'RVDV',  # Right ventricular volume
                                 'LAMINV', 'LAMAXV', 'LASTROKEV', 'LAEF',  # Left atrial volume
                                 'RAMINV', 'RAMAXV', 'RASTROKEV', 'RAEF',  # Right atrial volume
                                 'CO',  # Cardiac output
                                 'IED'])


def change_pars(model, pars):
    """Change model parameter values using a single dictionary input with par_name: par_value. Convenient for changing
    model parameters while fitting. Order is important: ratio-based parameter changes should occur after any absolute
    value changes"""

    # Change all key names to lowercase to prevent case inconsistencies
    pars = {key.lower(): value for key, value in pars.items()}

    for par, value in pars.items():
        # Circulation
        if par == "sbv":
            model.circulation.sbv = value
        elif par == "k_initial":
            model.circulation.k = value

        # Capacitances
        elif par == "cvp":
            model.capacitances.cvp = value
        elif par == "cas":
            model.capacitances.cas = value
        elif par == "cap":
            model.capacitances.cap = value
        elif par == "cvs":
            model.capacitances.cvs = value
        elif par == "scale_cs":
            model.capacitances.cas = model.capacitances.cas * value
            model.capacitances.cvs = model.capacitances.cvs * value
        elif par == "scale_cp":
            model.capacitances.cap = model.capacitances.cap * value
            model.capacitances.cvp = model.capacitances.cvp * value

        # Resistances
        elif par == "rvp":
            model.resistances.rvp = value
        elif par == "rcs":
            model.resistances.rcs = value
        elif par == "ras":
            model.resistances.ras = value
        elif par == "rvs":
            model.resistances.rvs = value
        elif par == "rcp":
            model.resistances.rcp = value
        elif par == "rap":
            model.resistances.rap = value
        elif par == "rav":
            model.resistances.rav = value
        elif par == "rmvb":
            model.resistances.rmvb = value
        elif par == "rtvb":
            model.resistances.rtvb = value

        # Heart parameters, ventricles and atria (with suffix _a) separately
        chamber_names = ["", "_a"]
        i_ventricles = model.heart.patches < 3
        i_atria = model.heart.patches >= 3
        i_chambers = [i_ventricles, i_atria]
        for i, chambers in enumerate(i_chambers):
            if par == "sact" + chamber_names[i] or par == "sfact" + chamber_names[i]:
                model.heart.sf_act[chambers] = value
            elif par == "tad" + chamber_names[i]:
                model.heart.t_ad[chambers] = value
            elif par == "tad_scale" + chamber_names[i]:
                model.heart.t_ad[chambers] = model.heart.t_ad[chambers] * value
            elif par == "td" + chamber_names[i]:
                model.heart.tau_d[chambers] = value
            elif par == "tr" + chamber_names[i]:
                model.heart.tau_r[chambers] = value
            elif par == "c1" + chamber_names[i]:
                model.heart.c_1[chambers] = value
            elif par == "c3" + chamber_names[i]:
                model.heart.c_3[chambers] = value
            elif par == "c4" + chamber_names[i]:
                model.heart.c_4[chambers] = value
        if par == "tad_va_ratio":
            model.heart.t_ad[i_atria] = model.heart.t_ad[i_ventricles][0] * value
        if par == "sympathetic_activity_circulation":
            model.resistances.ras = model.resistances.ras * value
            model.resistances.rap = model.resistances.rap * value
            model.capacitances.cvp = model.capacitances.cvp / value
            model.capacitances.cas = model.capacitances.cas / value
            model.capacitances.cap = model.capacitances.cap / value
            model.capacitances.cvs = model.capacitances.cvs / value
        if par == "sympathetic_activity_tad":
            model.heart.t_ad = model.heart.t_ad / value

        # Pericardium
        if par == "wth_p":
            model.pericardium.thickness = value
        elif par == "c1_p":
            model.pericardium.c_1 = value
        elif par == "c3_p":
            model.pericardium.c_3 = value
        elif par == "c4_p":
            model.pericardium.c_4 = value

        # Heart area - maintain ratio of AmRefs within each wall but scale according to total AmRef given
        elif par == "amreflfw":
            model.heart.am_ref[model.heart.patches == 0] = model.heart.am_ref[model.heart.patches == 0] * \
                                                           value / np.sum(model.heart.am_ref[model.heart.patches == 0])
        elif par == "amrefrfw":
            model.heart.am_ref[model.heart.patches == 1] = model.heart.am_ref[model.heart.patches == 1] * \
                                                           value / np.sum(model.heart.am_ref[model.heart.patches == 1])
        elif par == "amrefsw":
            model.heart.am_ref[model.heart.patches == 2] = model.heart.am_ref[model.heart.patches == 2] * \
                                                           value / np.sum(model.heart.am_ref[model.heart.patches == 2])
        elif par == "amrefla":
            model.heart.am_ref[model.heart.patches == 3] = model.heart.am_ref[model.heart.patches == 3] * \
                                                           value / np.sum(model.heart.am_ref[model.heart.patches == 3])
        elif par == "amrefra":
            model.heart.am_ref[model.heart.patches == 4] = model.heart.am_ref[model.heart.patches == 4] * \
                                                           value / np.sum(model.heart.am_ref[model.heart.patches == 4])

        # Wall volume, maintain current ratio in wall volumes between patches
        elif par == "vlfw":
            model.heart.vw[model.heart.patches == 0] = value * model.heart.vw[model.heart.patches == 0] / \
                                                       np.sum(model.heart.vw[model.heart.patches == 0])
        elif par == "vrfw":
            model.heart.vw[model.heart.patches == 1] = value * model.heart.vw[model.heart.patches == 1] / \
                                                       np.sum(model.heart.vw[model.heart.patches == 1])
        elif par == "vsw":
            model.heart.vw[model.heart.patches == 2] = value * model.heart.vw[model.heart.patches == 2] / \
                                                       np.sum(model.heart.vw[model.heart.patches == 2])
        elif par == "vla":
            model.heart.vw[model.heart.patches == 3] = value * model.heart.vw[model.heart.patches == 3] / \
                                                       np.sum(model.heart.vw[model.heart.patches == 3])
        elif par == "vra":
            model.heart.vw[model.heart.patches == 4] = value * model.heart.vw[model.heart.patches == 4] / \
                                                       np.sum(model.heart.vw[model.heart.patches == 4])

        ### The following metrics all use calculations to set wall volumes and areas, used for specific fitting schemes

        # Set total LV wall volume and distribute along left free wall and septal wall using patch number
        elif par == "lvwv":
            vw_tot_lv = np.sum(model.heart.vw[model.heart.patches == 0]) + np.sum(
                model.heart.vw[model.heart.patches == 2])
            for i_wall in [0, 2]:
                model.heart.vw[model.heart.patches == i_wall] = model.heart.vw[model.heart.patches == i_wall] * \
                                                                value / vw_tot_lv

        # Set midwall reference areas using ratio with left free wall
        elif par == "amrefrfwratio":
            am_ref_rfw = value * np.sum(model.heart.am_ref[model.heart.patches == 0])
            model.heart.am_ref[model.heart.patches == 1] = (
                    model.heart.am_ref[model.heart.patches == 1] * am_ref_rfw /
                    np.sum(model.heart.am_ref[model.heart.patches == 1]))
        elif par == "amrefswratio":
            am_ref_sw = value * np.sum(model.heart.am_ref[model.heart.patches == 0])
            model.heart.am_ref[model.heart.patches == 2] = (
                    model.heart.am_ref[model.heart.patches == 2] * am_ref_sw /
                    np.sum(model.heart.am_ref[model.heart.patches == 2]))
        elif par == "amreflaratio":
            model.heart.am_ref[model.heart.patches == 3] = value * np.sum(model.heart.am_ref[model.heart.patches == 0])
        elif par == "amrefraratio":
            model.heart.am_ref[model.heart.patches == 4] = value * np.sum(model.heart.am_ref[model.heart.patches == 0])

        # Set left atrial wall relative to right atrial wall
        elif par == "amrefralaratio":
            am_ref_la = value * np.sum(model.heart.am_ref[model.heart.patches == 4])
            model.heart.am_ref[model.heart.patches == 3] = (
                    model.heart.am_ref[model.heart.patches == 3] * am_ref_la /
                    np.sum(model.heart.am_ref[model.heart.patches == 3]))

        # Set wall volumes using ratio with left free wall
        elif par == "rfwvratio":
            vw_tot = value * np.sum(model.heart.vw[model.heart.patches == 0])
            model.heart.vw[model.heart.patches == 1] = model.heart.vw[model.heart.patches == 1] * \
                                                       vw_tot / np.sum(model.heart.vw[model.heart.patches == 1])
        elif par == "swvratio":
            vw_tot = value * np.sum(model.heart.vw[model.heart.patches == 0])
            model.heart.vw[model.heart.patches == 2] = model.heart.vw[model.heart.patches == 2] * \
                                                       vw_tot / np.sum(model.heart.vw[model.heart.patches == 2])
        elif par == "lawvratio":
            model.heart.vw[model.heart.patches == 3] = value * np.sum(
                model.heart.vw[model.heart.patches == 0])
        elif par == "rawvratio":
            model.heart.vw[model.heart.patches == 4] = value * np.sum(
                model.heart.vw[model.heart.patches == 0])

        # Set wall volumes based on specified wall thickness
        elif par == "lfwth":
            model.heart.vw[model.heart.patches == 0] = value * model.heart.am_ref[model.heart.patches == 0]
        elif par == "rfwth":
            model.heart.vw[model.heart.patches == 1] = value * model.heart.am_ref[model.heart.patches == 1]
        elif par == "swth":
            model.heart.vw[model.heart.patches == 2] = value * model.heart.am_ref[model.heart.patches == 2]
        elif par == "lawth":
            model.heart.vw[model.heart.patches == 3] = value * model.heart.am_ref[model.heart.patches == 3]
        elif par == "rawth":
            model.heart.vw[model.heart.patches == 4] = value * model.heart.am_ref[model.heart.patches == 4]

        # Set wall volume of all left ventricular patches based on specific wall thicknesses of each patch
        elif par == "lvwth":
            model.heart.vw[0:16] = value * model.heart.am_ref[0:16]

        # Update total wall volumes and areas to reflect any changes
        heart.set_total_wall_volumes_areas(model)
        model.heart.v_tot_0 = heart.unloaded_heart_volume(model.heart.am_ref_w, model.heart.vw_w)
