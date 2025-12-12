import numpy as np
import warnings
from pathlib import Path
from numba import njit
from scipy.signal import find_peaks
from scipy import interpolate
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import uuid
import shutil
from dataclasses import dataclass


def do_all_post_processing(model):
    # Compute the blood flow over time between the compartments. Use an adapted version of the RK4 solver that is used
    # during the simulation.
    @dataclass
    class PostProcessing:
        blood_flow: np.ndarray
        time_events: dict

    blood_flow = get_blood_flow(model.pressures, model.resistances, model.solver.dt)
    time_events, model.termination.terminated, model.termination.message = (
        get_valve_events(model.pressures, model.termination.terminated, model.termination.message))
    model.post_processing = PostProcessing(blood_flow=blood_flow, time_events=time_events)

    if not model.termination.terminated:
        model.post_processing.pressures = compute_pressures(model.pressures, model.post_processing.time_events,
                                                            model.activation.rhythm, model.contractility,
                                                            model.heart.patches)
        model.post_processing.volumes = compute_volumes(model.volumes, model.post_processing.time_events,
                                                        model.activation.rhythm, model.activation.vat_series,
                                                        model.activation.patch_t_delay, model.activation.time,
                                                        model.heart.patches)

        model.post_processing.shortening = compute_shortening(model.post_processing.time_events, model.activation.time,
                                                              model.heart.lab_f, model.heart.patches)

        # Compute the cardiac output in L/min
        model.post_processing.co_lv, model.post_processing.co_rv = (
            compute_cardiac_output(model.post_processing.blood_flow, model.post_processing.time_events,
                                   model.activation.time, 1000/model.solver.dt))

        # Compute the PV loop area in mmHg*ml
        model.post_processing.pv_loop_area_lv, model.post_processing.pv_loop_area_rv = (
            compute_pv_loop_area(model.post_processing.time_events, model.pressures, model.volumes))


def compute_pressures(p, time_events, rhythm, contractility, patches):
    av_closes = time_events['av_closes']
    mv_closes = time_events['mv_closes']
    rv_closes = time_events['rv_closes']
    tv_closes = time_events['tv_closes']
    pressures = {}

    # If there is only one beat, I calculate the systolic and diastolic pressures from the whole pressure signal.
    # If there are several beats, I calculate the systolic pressures between the end-diastolic and end-systolic events.
    # I calculate the diastolic pressures between the end-systolic events and the next end-diastolic event. Note, that
    # the diastolic pressure of the last heart beat is not calculated, because the end-diastolic event of the next beat
    # is not defined as it is not simulated. I could just calculate the diastolic pressure from the end-systolic event to
    # the end of the pressure signal, but I want to avoid that the simulation goes on for too long without a contraction
    # and the pressure signal drifts towards unrealistic values that are picked up in the calculation.
    if mv_closes.size == 1:
        pressures['lvesp'] = p[av_closes[0], 2]  # Systolic left-ventricular pressure
        pressures['lvedp'] = p[mv_closes[0], 2]  # Diastolic left-ventricular pressure
        pressures['lvsp'] = p[:, 2].max()  # Systolic left-ventricular pressure
        pressures['lvdp'] = p[:, 2].min()  # Diastolic left-ventricular pressure
        pressures['sbp'] = p[:, 3].max()  # Systolic blood pressure
        pressures['dbp'] = p[:, 3].min()  # Diastolic blood pressure
        pressures['lvmp'] = p[:, 2].mean()  # Mean left-ventricular pressure
    else:
        pressures['lvesp'] = p[av_closes, 2]  # Systolic left-ventricular pressure
        pressures['lvedp'] = p[mv_closes, 2]  # Diastolic left-ventricular pressure
        # Systolic LV pressure
        pressures['lvsp'] = np.array([max(p[mv_closes[i]:av_closes[i], 2]) for i in range(mv_closes.size)])
        # Diastolic LV pressure
        pressures['lvdp'] = np.array([min(p[av_closes[i]:mv_closes[i + 1], 2]) for i in range(mv_closes.size - 1)])
        # Systolic blood pressure
        pressures['sbp'] = np.array([max(p[mv_closes[i]:av_closes[i], 3]) for i in range(mv_closes.size)])
        # Diastolic blood pressure
        pressures['dbp'] = np.array([min(p[av_closes[i]:mv_closes[i + 1], 3]) for i in range(mv_closes.size - 1)])
        pressures['lvmp'] = np.mean(p[mv_closes[0]:mv_closes[-1], 2])  # Mean left-ventricular pressure

    if tv_closes.size == 1:
        pressures['rvesp'] = p[rv_closes[0], 6]  # Systolic right-ventricular pressure
        pressures['rvedp'] = p[tv_closes[0], 6]  # Diastolic right-ventricular pressure
        pressures['rvsp'] = p[:, 6].max()  # Systolic right-ventricular pressure
        pressures['rvdp'] = p[:, 6].min()  # Diastolic right-ventricular pressure
        pressures['rvmp'] = p[:, 6].mean()  # Mean right-ventricular pressure
    else:
        pressures['rvesp'] = p[rv_closes, 6]  # Systolic right-ventricular pressure
        pressures['rvedp'] = p[tv_closes, 6]  # Diastolic right-ventricular pressure
        # Systolic RV pressure
        pressures['rvsp'] = np.array([max(p[tv_closes[i]:rv_closes[i], 6]) for i in range(tv_closes.size)])
        # Diastolic RV pressure
        pressures['rvdp'] = np.array([min(p[rv_closes[i]:tv_closes[i + 1], 6]) for i in range(tv_closes.size - 1)])
        pressures['rvmp'] = np.mean(p[tv_closes[0]:tv_closes[-1], 6])  # Mean right-ventricular pressure

    if time_events['mv_closes'].size == 1:
        pressures['lamp'] = p[:, 1].mean()  # Mean LA pressure
    else:
        pressures['lamp'] = np.mean(p[mv_closes[0]:mv_closes[-1], 1])  # Mean LA pressure

    if time_events['tv_closes'].size == 1:
        pressures['ramp'] = p[:, 5].mean()  # Mean RA pressure
    else:
        pressures['ramp'] = np.mean(p[tv_closes[0]:tv_closes[-1], 5])  # Mean RA pressure

    # Compute v-wave pressure
    if time_events['mv_opens'].size == 1:
        pressures['lavwp'] = p[time_events['mv_opens'][0], 1]  # Left atrial v-wave pressure
    else:
        pressures['lavwp'] = p[time_events['mv_opens'], 1]  # Left atrial v-wave pressure

    if time_events['tv_opens'].size == 1:
        pressures['ravwp'] = p[time_events['tv_opens'][0], 5]  # Left atrial v-wave pressure
    else:
        pressures['ravwp'] = p[time_events['tv_opens'], 5]  # Left atrial v-wave pressure

    # Compute a-wave pressure. Only compute the a-wave pressure during normal sinus rhythm, because the a-wave is not
    # obvious during atrial fibrillation.
    if rhythm == "NSR":
        # Compute the index at which the contractility is maximum. I do that for each beat separately between mitral
        # valve opening and mitral valve closing.
        i_max_lac = np.array([time_events['mv_opens'][i] +
                              np.argmax(contractility[time_events['mv_opens'][i]:time_events['mv_closes'][i+1],
                                        np.where(patches == 3)[0][0]])
                              for i in range(len(time_events['mv_opens'])-1)])
        # Now we have the index of the maximum contraction, now I want to find the local maximum in the pressure signal
        # that is close to the maximum contraction. I do this, by computing the window where the pressure signal >= the
        # pressure value at maximum contractility.
        laawp = np.zeros(len(i_max_lac))
        for j, i in enumerate(i_max_lac):
            i_min = i - np.argmax(np.flip(p[:i, 1] < p[i, 1]))
            i_max = i + np.argmax(p[i:, 1] < p[i, 1])
            if i_min == i_max:  # If i_min == i_max, avoid that p[i_min:i_max, 1] will be an empty array.
                i_max += 1
            laawp[j] = p[i_min:i_max, 1].max()

        i_max_rac = np.array([time_events['tv_opens'][i] +
                              np.argmax(contractility[time_events['tv_opens'][i]:time_events['tv_closes'][i+1],
                                        np.where(patches == 4)[0][0]])
                              for i in range(len(time_events['tv_opens'])-1)])
        raawp = np.zeros(len(i_max_rac))
        for j, i in enumerate(i_max_rac):
            i_min = i - np.argmax(np.flip(p[:i, 5] < p[i, 5]))
            i_max = i + np.argmax(p[i:, 5] < p[i, 5])
            if i_min == i_max:  # If i_min == i_max, avoid that p[i_min:i_max, 5] will be an empty array.
                i_max += 1
            raawp[j] = p[i_min:i_max, 5].max()

        pressures['laawp'] = laawp
        pressures['raawp'] = raawp
    else:
        pressures['laawp'] = np.nan
        pressures['raawp'] = np.nan

    return pressures


def compute_volumes(v, time_events, rhythm, vat_series, patch_t_delay, time, patches):
    av_closes = time_events['av_closes']
    mv_closes = time_events['mv_closes']
    rv_closes = time_events['rv_closes']
    tv_closes = time_events['tv_closes']
    volumes = {}

    # If there is only one beat, I calculate the systolic and diastolic volumes from the whole volume signal.
    # If there are several beats, I calculate the diastolic volumes between the end-diastolic and end-systolic events.
    # I calculate the systolic pressures between the end-systolic events and the next end-diastolic event. Note, that
    # the systolic pressure of the last heart beat is not calculated, because the end-diastolic event of the next beat
    # is not defined as it is not simulated. I could just calculate the systolic volume from the end-systolic event to
    # the end of the volume signal, but I want to avoid that the simulation goes on for too long without a contraction
    # and the volume signal drifts towards unrealistic values that are picked up in the calculation.
    if mv_closes.size == 1:
        volumes['lvesv'] = v[av_closes[0], 2]  # Compute left-ventricular end-systolic volume
        volumes['lvedv'] = v[mv_closes[0], 2]  # Compute left-ventricular end-diastolic volume
        volumes['lvsv'] = v[:, 2].min()  # Systolic left-ventricular volume
        volumes['lvdv'] = v[:, 2].max()  # Diastolic left-ventricular volume
        volumes['lvstrokev'] = volumes['lvdv'] - volumes['lvsv']  # LV stroke volume
        volumes['lvef'] = volumes['lvstrokev'] / volumes['lvdv'] * 100  # LV ejection fraction
    else:
        volumes['lvesv'] = v[av_closes, 2]  # Compute left-ventricular end-systolic volume
        volumes['lvedv'] = v[mv_closes, 2]  # Compute left-ventricular end-diastolic volume
        # Systolic LV volume
        volumes['lvsv'] = np.array([min(v[av_closes[i]:mv_closes[i+1], 2]) for i in range(mv_closes.size - 1)])
        # Diastolic LV volume
        volumes['lvdv'] = np.array([max(v[mv_closes[i]:av_closes[i], 2]) for i in range(mv_closes.size)])
        volumes['lvstrokev'] = volumes['lvdv'][:-1] - volumes['lvsv']  # LV stroke volume
        volumes['lvef'] = volumes['lvstrokev'] / volumes['lvdv'][:-1] * 100  # LV ejection fraction

    if tv_closes.size == 1:
        volumes['rvesv'] = v[rv_closes[0], 6]  # Compute right-ventricular end-systolic volume
        volumes['rvedv'] = v[tv_closes[0], 6]  # Compute right-ventricular end-diastolic volume
        volumes['rvsv'] = v[:, 6].min()  # Systolic right-ventricular volume
        volumes['rvdv'] = v[:, 6].max()  # Diastolic right-ventricular volume
        volumes['rvstrokev'] = volumes['rvdv'] - volumes['rvsv']  # LV stroke volume
        volumes['rvef'] = volumes['rvstrokev'] / volumes['rvdv'] * 100  # LV ejection fraction
    else:
        volumes['rvesv'] = v[rv_closes, 6]  # Compute right-ventricular end-systolic volume
        volumes['rvedv'] = v[tv_closes, 6]  # Compute right-ventricular end-diastolic volume
        # Systolic RV volume
        volumes['rvsv'] = np.array([min(v[rv_closes[i]:tv_closes[i+1], 6]) for i in range(tv_closes.size - 1)])
        # Diastolic RV volume
        volumes['rvdv'] = np.array([max(v[tv_closes[i]:rv_closes[i], 6]) for i in range(tv_closes.size)])
        volumes['rvstrokev'] = volumes['rvdv'][:-1] - volumes['rvsv']  # RV stroke volume
        volumes['rvef'] = volumes['rvstrokev'] / volumes['rvdv'][:-1] * 100  # RV ejection fraction

    if rhythm == "NSR" or rhythm == 'AF':
        # First I calculate the indices of the left atrial activation times.
        i_pre_a = np.zeros(len(vat_series))
        for i, vat in enumerate(vat_series + patch_t_delay[np.where(patches == 3)[0][0]]):
            if time[0] <= vat <= time[-1]:
                i_pre_a[i] = np.argmin(np.abs(time - vat))
            else:
                # This case only happens, if the atria are not contracting. For example when modeling atrial
                # fibrillation with no atrial contraction like the Maastricht group.
                i_pre_a[i] = np.nan
        # Remove NaN values if there are any
        i_pre_a = np.int64(i_pre_a[~np.isnan(i_pre_a)])
        # Now I calculate the indices of all local maxima and minima in the left atrial volume trend.
        lamaxv_loc, _ = find_peaks(v[:, 1])
        laminv_loc, _ = find_peaks(-v[:, 1])
        # Now I get for each atrial activation time, the subsequent local maximum. And from this index, the subsequent
        # local minimum.
        sorted_lamaxv_loc = np.zeros(len(i_pre_a))
        sorted_laminv_loc = np.zeros(len(i_pre_a))
        for i, i_pre in enumerate(i_pre_a):
            possible_lamaxv_loc = lamaxv_loc[lamaxv_loc > i_pre]
            if len(possible_lamaxv_loc) > 0:
                sorted_lamaxv_loc[i] = possible_lamaxv_loc[0]
            else:
                sorted_lamaxv_loc = sorted_lamaxv_loc[:i]
                break
        # Sort out duplicates
        sorted_lamaxv_loc = list(dict.fromkeys(sorted_lamaxv_loc))
        for i, i_max in enumerate(sorted_lamaxv_loc):
            possible_laminv_loc = laminv_loc[laminv_loc > i_max]
            if len(possible_laminv_loc) > 0:
                sorted_laminv_loc[i] = possible_laminv_loc[0]
            else:
                sorted_lamaxv_loc = sorted_lamaxv_loc[:i]
                break
        sorted_laminv_loc = sorted_laminv_loc[:len(sorted_lamaxv_loc)]
        if len(sorted_lamaxv_loc) > 0:
            volumes['lastrokev'] = v[np.int64(sorted_lamaxv_loc), 1] - v[np.int64(sorted_laminv_loc), 1]  # LA stroke volume
            volumes['laef'] = volumes['lastrokev'] / v[np.int64(sorted_lamaxv_loc), 1] * 100  # LA ejection fraction
            volumes['laminv'] = v[np.int64(sorted_laminv_loc), 1]  # LA minimum volume
            volumes['lamaxv'] = v[np.int64(sorted_lamaxv_loc), 1]  # LA maximum volume
        else:
            # In this case, there were no atrial contractions. To process this case anyway, I just take every local
            # maxima and minima. It is not the perfect approach, but it will have to do.
            volumes['laminv'] = np.mean(v[np.int64(laminv_loc), 1])  # LA minimum volume
            volumes['lamaxv'] = np.mean(v[np.int64(lamaxv_loc), 1])  # LA maximum volume
            volumes['lastrokev'] = volumes['lamaxv'] - volumes['laminv']  # LA stroke volume
            volumes['laef'] = volumes['lastrokev'] / volumes['lamaxv'] * 100  # LA ejection fraction
    else:
        volumes['lastrokev'] = np.nan  # dummy value in case rhythm is not "NSR" and I need to return something.
        volumes['laef'] = np.nan  # dummy value in case rhythm is not "NSR" and I need to return something.
        volumes['laminv'] = np.nan  # dummy value in case rhythm is not "NSR" and I need to return something.
        volumes['lamaxv'] = np.nan  # dummy value in case rhythm is not "NSR" and I need to return something.

    if rhythm == "NSR" or rhythm == 'AF':
        # First I calculate the indices of the right atrial activation times.
        i_pre_a = np.zeros(len(vat_series))
        for i, vat in enumerate(vat_series + patch_t_delay[np.where(patches == 4)[0][0]]):
            if time[0] <= vat <= time[-1]:
                i_pre_a[i] = np.argmin(np.abs(time - vat))
            else:
                i_pre_a[i] = np.nan
        # Remove NaN values if there are any
        i_pre_a = np.int64(i_pre_a[~np.isnan(i_pre_a)])
        # Now I calculate the indices of all local maxima and minima in the right atrial volume trend.
        ramaxv_loc, _ = find_peaks(v[:, 5])
        raminv_loc, _ = find_peaks(-v[:, 5])
        # Now I get for each atrial activation time, the subsequent local maximum. And from this index, the subsequent
        # local minimum.
        sorted_ramaxv_loc = np.zeros(len(i_pre_a))
        sorted_raminv_loc = np.zeros(len(i_pre_a))
        for i, i_pre in enumerate(i_pre_a):
            possible_ramaxv_loc = ramaxv_loc[ramaxv_loc > i_pre]
            if len(possible_ramaxv_loc) > 0:
                sorted_ramaxv_loc[i] = possible_ramaxv_loc[0]
            else:
                sorted_ramaxv_loc = sorted_ramaxv_loc[:i]
                break
        # Sort out duplicates
        sorted_ramaxv_loc = list(dict.fromkeys(sorted_ramaxv_loc))
        for i, i_max in enumerate(sorted_ramaxv_loc):
            possible_raminv_loc = raminv_loc[raminv_loc > i_max]
            if len(possible_raminv_loc) > 0:
                sorted_raminv_loc[i] = possible_raminv_loc[0]
            else:
                sorted_ramaxv_loc = sorted_ramaxv_loc[:i]
                break
        sorted_raminv_loc = sorted_raminv_loc[:len(sorted_ramaxv_loc)]
        if len(sorted_ramaxv_loc) > 0:
            volumes['rastrokev'] = v[np.int64(sorted_ramaxv_loc), 5] - v[np.int64(sorted_raminv_loc), 5]  # RA stroke volume
            volumes['raef'] = volumes['rastrokev'] / v[np.int64(sorted_ramaxv_loc), 5] * 100  # RA ejection fraction
            volumes['raminv'] = v[np.int64(sorted_raminv_loc), 5]  # RA minimum volume
            volumes['ramaxv'] = v[np.int64(sorted_ramaxv_loc), 5]  # RA maximum volume
        else:
            volumes['raminv'] = np.mean(v[np.int64(raminv_loc), 1])  # LA minimum volume
            volumes['ramaxv'] = np.mean(v[np.int64(ramaxv_loc), 1])  # LA maximum volume
            volumes['rastrokev'] = volumes['ramaxv'] - volumes['raminv']  # LA stroke volume
            volumes['raef'] = volumes['rastrokev'] / volumes['ramaxv'] * 100  # LA ejection fraction
    else:
        volumes['rastrokev'] = np.nan  # dummy value in case rhythm is not "NSR" and I need to return something.
        volumes['raef'] = np.nan  # dummy value in case rhythm is not "NSR" and I need to return something.
        volumes['raminv'] = np.nan  # dummy value in case rhythm is not "NSR" and I need to return something.
        volumes['ramaxv'] = np.nan  # dummy value in case rhythm is not "NSR" and I need to return something.

    # Compute pre-a-wave volume. Only compute the pre-a-wave volume during normal sinus rhythm, because the a-wave is not
    # obvious during atrial fibrillation.
    if rhythm == "NSR":
        # The pre-a-wave is when the atria start contracting.
        # First I calculate the indices of the left atrial activation times.
        i_pre_a = np.zeros(len(vat_series))
        for i, vat in enumerate(vat_series + patch_t_delay[np.where(patches == 3)[0][0]]):
            if time[0] <= vat <= time[-1]:
                i_pre_a[i] = np.argmin(np.abs(time - vat))
            else:
                i_pre_a[i] = np.nan
        # Remove NaN values if there are any
        i_pre_a = np.int64(i_pre_a[~np.isnan(i_pre_a)])
        volumes['lapreawv'] = v[i_pre_a, 1]  # Left atrial pre-a-wave volume
    return volumes


def compute_shortening(time_events, time, lab_f, patches):
    mv_opens = time_events['mv_opens']
    av_closes = time_events['av_closes']
    tv_opens = time_events['tv_opens']
    rv_closes = time_events['rv_closes']
    shortening = {}

    shortening['lfw_shortening'] = 0.5 * ((lab_f[:, np.where(patches == 0)[0]])**2 - 1)

    return shortening


def compute_cardiac_output(blood_flow, time_events, time, fs):
    """
    Computes how much blood the left and right ventricle eject in liter/minute. We calculate the cardiac output by
    summing up the blood flow between the ventricles and the arteries between the first end-diastole and the last
    end-diastole event, divided by the time between the first end-diastole and the last end-diastole event.
    This function does not compute the output of the atria, because in the ventricles it is easy to determine when the
    entrance and exit are open and closed. In the atria, blood can flow into and out of the atria to the veins
    constantly, making it harder to define cardiac output. Of course, you could always sum up the blood flow and with a
    longer duration, the estimate becomes more stable.
    """
    # Compute left-ventricular output
    if time_events['mv_closes'].size == 1:
        blood_pumped_lv = np.sum(blood_flow[:, 2]) / fs / 1000  # [L]
        total_time_lv = (time[-1] - time[0]) / 60000  # [min]
    else:
        blood_pumped_lv = np.sum(blood_flow[time_events['mv_closes'][0]:time_events['mv_closes'][-1], 2]) / fs / 1000  # [L]
        total_time_lv = np.diff(time[[time_events['mv_closes'][0], time_events['mv_closes'][-1]]])[0] / 60000  # [min]
    co_lv = blood_pumped_lv / total_time_lv  # [L/min]

    # Compute right-ventricular output
    if time_events['tv_closes'].size == 1:
        blood_pumped_rv = np.sum(blood_flow[:, 6]) / fs / 1000  # [L]
        total_time_rv = (time[-1] - time[0]) / 60000  # [min]
    else:
        blood_pumped_rv = np.sum(blood_flow[time_events['tv_closes'][0]:time_events['tv_closes'][-1], 6]) / fs / 1000  # [L]
        total_time_rv = np.diff(time[[time_events['tv_closes'][0], time_events['tv_closes'][-1]]])[0] / 60000  # [min]
    co_rv = blood_pumped_rv / total_time_rv  # [L/min]

    return co_lv, co_rv


def get_valve_events(p, term_flag, term_msg):
    time_events = {
        "mv_closes": np.argwhere(np.diff(np.multiply(p[:, 2] > p[:, 1], 1)) == 1)[:, 0] + 1,
        "mv_opens": np.argwhere(np.diff(np.multiply(p[:, 2] < p[:, 1], 1)) == 1)[:, 0] + 1,
        "av_closes": np.argwhere(np.diff(np.multiply(p[:, 3] > p[:, 2], 1)) == 1)[:, 0] + 1,
        "av_opens": np.argwhere(np.diff(np.multiply(p[:, 3] < p[:, 2], 1)) == 1)[:, 0] + 1,
        "tv_closes": np.argwhere(np.diff(np.multiply(p[:, 6] > p[:, 5], 1)) == 1)[:, 0] + 1,
        "tv_opens": np.argwhere(np.diff(np.multiply(p[:, 6] < p[:, 5], 1)) == 1)[:, 0] + 1,
        "rv_closes": np.argwhere(np.diff(np.multiply(p[:, 7] > p[:, 6], 1)) == 1)[:, 0] + 1,
        "rv_opens": np.argwhere(np.diff(np.multiply(p[:, 7] < p[:, 6], 1)) == 1)[:, 0] + 1,
    }
    # Compute left-ventricular end-diastole events and left-ventricular end-systole events
    time_events['mv_opens'], time_events['mv_closes'], time_events['av_opens'], time_events['av_closes'] = (
        compute_systole_diastole(time_events['mv_opens'], time_events['mv_closes'], time_events['av_opens'],
                                 time_events['av_closes']))
    # Compute right-ventricular end-diastole events and right-ventricular end-systole events
    time_events['tv_opens'], time_events['tv_closes'], time_events['rv_opens'], time_events['rv_closes'] = (
        compute_systole_diastole(time_events['tv_opens'], time_events['tv_closes'], time_events['rv_opens'],
                                 time_events['rv_closes']))

    # Check that we have at least 2 end-diastolic and 2 end-systolic event. If not, errors will occur when computing
    # the output characteristics.
    if (len(time_events['mv_closes']) <= 1 or len(time_events['tv_closes']) <= 1 or len(time_events['av_closes']) <= 1 or
            len(time_events['rv_closes']) <= 1):
        term_flag = True
        term_msg = ("No end-diastolic or end-systolic events in the left or right ventricle. " +
                    "The simulation is terminated.")
        return time_events, term_flag, term_msg

    if len(time_events['av_closes']) != len(time_events['mv_closes']):
        raise ValueError("The number of end-systole and end-diastole events in the left ventricle are not equal.")
    if time_events['av_closes'][0] - time_events['mv_closes'][0] < 0:
        raise ValueError("End-systole happens before end-diastole in left ventricle. Check the code, this should " +
                         "never happen.")
    if len(time_events['rv_closes']) != len(time_events['tv_closes']):
        raise ValueError("The number of end-systole and end-diastole events in the right ventricle are not equal.")
    if time_events['rv_closes'][0] - time_events['tv_closes'][0] < 0:
        raise ValueError("End-systole happens before end-diastole in right ventricle. Check the code, this should " +
                         "never happen.")

    return time_events, term_flag, term_msg


def compute_systole_diastole(in_open, in_close, out_open, out_close):
    # We define the end-systole not just as the times when the aortic valve closes (pulmonary valve for the right
    # ventricle), because during atrial fibrillation the aortic valve (pulmonary valve) can open and close multiple times
    # or not at all between two mitral valve openings (tricuspid valve openings). Therefore, we start with the times of
    # mitral valve openings (tricuspid valve openings) when the ventricles start to fill and only take the last aortic
    # valve closing event (pulmonary valve closing event) before the mitral valve opening event (tricuspid valve opening
    # event) as the end-systole event.
    # I define here that a left-ventricular PV loop always starts with a mitral valve closing, then aortic valve opening,
    # then aortic valve closing and lastly a mitral valve opening. Several consecutive PV loops always follow this order.
    # All valve events before and after PV loops that don't form a complete loop will be excluded.
    # The same principle is applied to the right ventricle.
    in_open_proc = np.array([], dtype=int)
    in_close_proc = np.array([], dtype=int)
    out_open_proc = np.array([], dtype=int)
    out_close_proc = np.array([], dtype=int)

    flag_search_running = True
    while flag_search_running:
        # Diastolic Event:
        # Find last mitral valve closing before the next aortic valve opening (in case there are several)
        # If there is no mitral valve closing until the next aortic valve opening, throw away the next aortic valve
        # opening and repeat until there is no aortic valve opening. Then stop the search.
        while flag_search_running:
            if len(in_close) == 0 or len(out_open) == 0:
                flag_search_running = False
            else:
                if in_close[0] < out_open[0]:
                    in_close_proc = np.append(in_close_proc, in_close[np.where(in_close < out_open[0])[0][-1]])
                    out_open_proc = np.append(out_open_proc, out_open[0])
                    in_close = in_close[in_close > out_open[0]]
                    in_open = in_open[in_open > out_open[0]]
                    out_close = out_close[out_close > out_open[0]]
                    out_open = out_open[1:]
                    break
                else:
                    # There is no mitral valve closing event before the next aortic valve opening. Discard the next
                    # aortic valve opening event.
                    out_open = out_open[1:]
        # Systolic Event:
        # Find last aortic valve closing before the next mitral valve opening (in case there are several)
        # If there is no aortic valve closing until the next mitral valve opening, throw away the next mitral valve
        # opening and repeat until there is no mitral valve opening. Then stop the search.
        while flag_search_running:
            if len(out_close) == 0 or len(in_open) == 0:
                flag_search_running = False
            else:
                if np.any(out_close < in_open[0]):
                    out_close_proc = np.append(out_close_proc, out_close[np.where(out_close < in_open[0])[0][-1]])
                    in_open_proc = np.append(in_open_proc, in_open[0])
                    out_close = out_close[out_close > in_open[0]]
                    out_open = out_open[out_open > in_open[0]]
                    in_close = in_close[in_close > in_open[0]]
                    in_open = in_open[1:]
                    break
                else:
                    # There is no aortic valve closing event before the next mitral valve opening. Discard the next
                    # mitral valve opening event.
                    in_open = in_open[1:]
    min_len = min(len(in_close_proc), len(out_open_proc), len(out_close_proc), len(in_open_proc))
    in_close_proc = in_close_proc[:min_len]
    out_open_proc = out_open_proc[:min_len]
    out_close_proc = out_close_proc[:min_len]
    in_open_proc = in_open_proc[:min_len]
    return in_open_proc, in_close_proc, out_open_proc, out_close_proc


def get_blood_flow(p, r, dt):
    """
    Compute the blood flow between the compartments over time.
    :param model: Class object of the cardiogrowth model.
    """
    p = p.T
    # Resistances between the compartments
    res = np.array([r.rap, r.ras, r.rav, r.rcp, r.rcs, r.rmvb, r.rtvb, r.rvp, r.rvs])
    dt = dt / 1000  # [s]

    # 4th order Runge-Kutta differential equation solver to calculate the blood flow at the current time point.
    dv_k1, blood_flow_k1 = dv_combined_with_blood_flow(p, res)
    dv_k2, blood_flow_k2 = dv_combined_with_blood_flow(p + 0.5 * dt * dv_k1, res)
    dv_k3, blood_flow_k3 = dv_combined_with_blood_flow(p + 0.5 * dt * dv_k2, res)
    _, blood_flow_k4 = dv_combined_with_blood_flow(p + dt * dv_k3, res)

    blood_flow = 1 / 6 * (blood_flow_k1 + 2 * blood_flow_k2 + 2 * blood_flow_k3 + blood_flow_k4).T  # [ml/s]
    return blood_flow


def dv_combined_with_blood_flow(p, r):
    """
    Compute the blood flow and change in volume in each compartment.
    :param p: Array with the pressures in each compartment
    :param r: Array with the resistances between each compartment
    blood_flow[0]: Blood flow between pulmonary arteries and pulmonary veins
    blood_flow[1]: Blood flow between pulmonary veins and left atrium
    blood_flow[2]: Blood flow between left atrium and left ventricle
    blood_flow[3]: Blood flow between left ventricle and systemic arteries
    blood_flow[4]: Blood flow between systemic arteries and systemic veins
    blood_flow[5]: Blood flow between systemic veins and right atrium
    blood_flow[6]: Blood flow between right atrium and right ventricle
    blood_flow[7]: Blood flow between right ventricle and pulmonary arteries
    dv[0]: Change in volume of pulmonary veins
    dv[1]: Change in volume of left atrium
    dv[2]: Change in volume of left ventricle
    dv[3]: Change in volume of systemic arteries
    dv[4]: Change in volume of systemic veins
    dv[5]: Change in volume of right atrium
    dv[6]: Change in volume of right ventricle
    dv[7]: Change in volume of pulmonary arteries
    """
    blood_flow = np.array([
        (p[7] - p[0]) / r[0],  # Blood flow between pulmonary arteries and pulmonary veins
        (p[0] - p[1]) / r[7],  # Blood flow between pulmonary veins and left atrium
        (p[1] - p[2]) / r[2] * (p[1] > p[2]) -
        (p[2] - p[1]) / r[5] * (p[2] > p[1]),  # Blood flow between left atrium and left ventricle
        (p[2] - p[3]) / r[4] * (p[2] > p[3]),  # Blood flow between left ventricle and systemic arteries
        (p[3] - p[4]) / r[1],  # Blood flow between systemic arteries and systemic veins
        (p[4] - p[5]) / r[8],  # Blood flow between systemic veins and right atrium
        (p[5] - p[6]) / r[2] * (p[5] > p[6]) -
        (p[6] - p[5]) / r[6] * (p[6] > p[5]),  # Blood flow between right atrium and right ventricle
        (p[6] - p[7]) / r[3] * (p[6] > p[7]),  # Blood flow between right ventricle and pulmonary arteries
    ])
    dv = np.array([blood_flow[0] - blood_flow[1],  # Change in volume of pulmonary veins
                   blood_flow[1] - blood_flow[2],  # Change in volume of left atrium
                   blood_flow[2] - blood_flow[3],  # Change in volume of left ventricle
                   blood_flow[3] - blood_flow[4],  # Change in volume of systemic arteries
                   blood_flow[4] - blood_flow[5],  # Change in volume of systemic veins
                   blood_flow[5] - blood_flow[6],  # Change in volume of right atrium
                   blood_flow[6] - blood_flow[7],  # Change in volume of right ventricle
                   blood_flow[7] - blood_flow[0]])  # Change in volume of pulmonary arteries
    return dv, blood_flow


def compute_pv_loop_area(time_events, p, v):
    # We approximate the PV loop area by something similar than the trapezoidal rule.
    # First of all, the PV loop is most likely not a closed loop, so we start a circle at the end-diastolic event of the
    # current beat until the end-diastolic event of the next beat. We can't calculate a PV loop area for the last beat.
    # We choose a point inside the PV loop which is in the middle between the max and min pressure and max and min volume
    # of that loop. Between that center point and two adjacent samples of the PV loop, we compute the area of this
    # triangle. If we sum up all the triangles of all the adjacent samples of the PV loop with the center point, we
    # should get a good approximation of the PV loop.
    # The area of a triangle with three coordinates (x1, y1), (x2, y2), (x3, y3) is given by:
    # area = 0.5 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2))
    # Below, this equation is written as matrix multiplication.
    if time_events['mv_closes'].size == 1:
        pv_loop_area_lv = np.nan  # To calculate the PV loop area, we need at least two beats.
    else:
        pv_loop_area_lv = pv_loop_area(time_events['mv_closes'], v[:, 2], p[:, 2])

    if time_events['tv_closes'].size == 1:
        pv_loop_area_rv = np.nan  # To calculate the PV loop area, we need at least two beats.
    else:
        pv_loop_area_rv = pv_loop_area(time_events['tv_closes'], v[:, 6], p[:, 6])

    return pv_loop_area_lv, pv_loop_area_rv


def pv_loop_area(time_events, v, p):
    area = np.zeros(len(time_events) - 1)
    for i in range(len(time_events) - 1):
        v_loop = v[time_events[i]:time_events[i + 1]]
        p_loop = p[time_events[i]:time_events[i + 1]]
        v_center = (np.max(v_loop) + np.min(v_loop)) / 2
        p_center = (np.max(p_loop) + np.min(p_loop)) / 2
        x1 = v_loop[:-1]
        x2 = v_loop[1:]
        x3 = np.ones(x1.size) * v_center
        y1 = p_loop[:-1]
        y2 = p_loop[1:]
        y3 = np.ones(y1.size) * p_center
        area[i] = 0.5 * np.sum(abs(np.sum(np.diff(np.column_stack((y2, y1, y3, y2))) *
                                          np.column_stack((x3, x2, x1)), axis=1)))
    return area
