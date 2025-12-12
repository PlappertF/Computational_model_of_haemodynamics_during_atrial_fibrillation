import os
import numpy as np
import json
import jsbeautifier
import h5py
import src.heart as heart
from dataclasses import dataclass
from pathlib import Path


def import_pars(model, model_pars):
    # Load last converged solution
    with open(os.path.join(str(model_pars) + ".json")) as json_file:
        pars = json.load(json_file)

    # Assign model parameters to data classes

    @dataclass
    class Circulation:
        sbv = pars['Circulation']['sbv']  # [mL]
        k = np.array(pars['Circulation']['k'])  # [-]

    @dataclass
    class Capacitances:  # [mL/mmHg]
        cvp = pars['Capacitances']['cvp']
        cas = pars['Capacitances']['cas']
        cvs = pars['Capacitances']['cvs']
        cap = pars['Capacitances']['cap']

    @dataclass
    class Resistances:  # [mmHg * s/mL]
        rvp = pars['Resistances']['rvp']
        rcs = pars['Resistances']['rcs']
        ras = pars['Resistances']['ras']
        rvs = pars['Resistances']['rvs']
        rcp = pars['Resistances']['rcp']
        rap = pars['Resistances']['rap']
        rav = pars['Resistances']['rav']
        rmvb = pars['Resistances']['rmvb']
        rtvb = pars['Resistances']['rtvb']
        if rmvb > 1e10:
            rmvb = np.inf
        if rtvb > 1e10:
            rtvb = np.inf

    @dataclass
    class Solver:
        cutoff = pars['Solver']['cutoff']
        nsr_launch_iter_max = pars['Solver']['iter_max']

    @dataclass
    class Heart:
        # Heart geometry
        vs = pars['Heart']['vs']  # [mm^2]
        ys = pars['Heart']['ys']  # [mm]
        c = pars['Heart']['c']  # [-]
        patches = np.array(pars['Heart']['patches'])
        am_ref = np.array(pars['Heart']['am_ref'])
        vw = np.array(pars['Heart']['wv'])

        # Active and passive Material properties
        ls_ref = np.array(pars['Heart']['ls_ref'])  # [um]
        ls_eiso = np.array(pars['Heart']['ls_eiso'])  # [um]
        lsc_0 = np.array(pars['Heart']['lsc0'])  # [um]
        v_max = np.array(pars['Heart']['v_max'])  # [um/ms]
        tau_r = np.array(pars['Heart']['tr'])  # [-]
        tau_d = np.array(pars['Heart']['td'])  # [-]
        t_ad = np.array(pars['Heart']['tad'])  # [ms]
        sf_act = (np.array(pars['Heart']['sf_act']) * (
                    1 - np.array(pars['Heart']['ischemic'])))  # [MPa], accounting for ischemia
        c_1 = np.array(pars['Heart']['c_1'])  # [MPa]
        c_3 = np.array(pars['Heart']['c_3'])  # [MPa]
        c_4 = np.array(pars['Heart']['c_4'])  # [-]

    @dataclass
    class Pericardium:
        c_1 = pars['Pericardium']['c_1']  # [MPa]
        c_3 = pars['Pericardium']['c_3']  # [MPa]
        c_4 = pars['Pericardium']['c_4']  # [-]
        thickness = pars['Pericardium']['thickness']  # [mm]
        lab_pre = pars['Pericardium']['pre_stretch']  # [-]

    # Assign input classes to the model class
    model.circulation = Circulation()
    model.capacitances = Capacitances()
    model.resistances = Resistances()
    model.solver = Solver()
    model.heart = Heart()
    model.pericardium = Pericardium()


def export_sim(model, filepath, filename):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    # Calculate wall thickness
    wall_thickness, _ = heart.get_wall_thickness(model)

    with h5py.File(os.path.join(filepath, filename + ".hdf5"), "w") as f:
        # f.create_dataset("time", data=model.activation.time)
        # f.create_dataset("pressures", data=model.pressures)
        # f.create_dataset("volumes", data=model.volumes)
        # f.create_dataset("r_m", data=model.heart.rm)
        # f.create_dataset("x_m", data=model.heart.xm)
        # f.create_dataset("lab_f", data=model.heart.lab_f)
        # f.create_dataset("wall_thickness", data=wall_thickness)
        f.create_dataset("outputs", data=model.outputs.to_numpy())
        f.attrs["outputs_names"] = model.outputs.columns.to_list()
        # f.attrs["patches"] = model.heart.patches
