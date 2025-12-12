import numpy as np
from numba import njit
from numba.types import bool_, float64


def get_wall_thickness(cg):
    """
    Calculate the thickness and midwall reference area of each wall throughout the cardiac cycle
    """
    # Midwall area
    lab_f_w = np.transpose(np.array([np.mean(cg.heart.lab_f[:, cg.heart.patches == i], axis=1) for i in range(0, 5)]))
    a_m_w = cg.heart.am_ref_w * (lab_f_w ** 2)
    # Wall thickness
    wall_thickness = cg.heart.vw_w / a_m_w
    return wall_thickness, a_m_w


def set_total_wall_volumes_areas(model):
    """
    Compute total wall volume and reference areas
    """
    model.heart.vw_w = np.zeros(5)
    model.heart.am_ref_w = np.zeros(5)
    model.heart.n_patches = np.zeros(5)
    for i in range(0, 5):
        model.heart.am_ref_w[i] = sum(model.heart.am_ref[model.heart.patches == i])
        model.heart.vw_w[i] = sum(model.heart.vw[model.heart.patches == i])
        model.heart.n_patches[i] = sum(model.heart.patches == i)



def unloaded_heart_volume(am_ref_w, vw_w):
    """
    Calculate total unloaded heart volume (cavity plus wall volumes) [mm^3]
    """
    return sum(unloaded_volumes(am_ref_w, vw_w)) * 1e3 + sum(vw_w)


def unloaded_volumes(am_ref_w, vw_w):
    """
    Calculate unloaded cavity volumes given reference midwall area and wall volume. Can only be used to estimate
    unloaded (and not loaded) volumes due to geometry assumptions made in the midwall volume calculations
    """

    # Preallocate midwall volume array
    v_m_0 = np.zeros(4)

    # Left ventricle: sphere with total surface Sw plus Lfw
    v_m_0[0] = (am_ref_w[0] + am_ref_w[2]) ** (3 / 2) / (6 * np.sqrt(np.pi))

    # Right ventricle
    r = np.sqrt((am_ref_w[1] + am_ref_w[2]) / (4 * np.pi))  # Radius of RV if SW would have been directed towards the LV
    h = am_ref_w[2] / (2 * np.pi * r)  # Height of the septal wall spherical cap
    v_s = np.pi * h ** 2 / 3 * (3 * r - h)  # Septal cap midwall volume
    v_m_0[1] = (am_ref_w[1] + am_ref_w[2]) ** (3 / 2) / (
                6 * np.sqrt(np.pi)) - v_s  # Volume of RV minus septal cap volume

    # Left atrium, use spherical geometry
    v_m_0[2] = am_ref_w[3] ** (3 / 2) / (6 * np.sqrt(np.pi))

    # Right atrium, use spherical geometry
    v_m_0[3] = am_ref_w[4] ** (3 / 2) / (6 * np.sqrt(np.pi))

    # Calculate cavity volumes by subtracting half the wall volumes from midwall volumes and convert from mm^3 to mL
    v_0 = (v_m_0 - 0.5 * np.array([vw_w[0] + vw_w[2], vw_w[1] + vw_w[2], vw_w[3], vw_w[4]])) * 1e-3

    return v_0


def guess_vs_ys(model):
    """
    Estimate initial values for vs and ys based on initial LV cavity volume and wall thickness. This estimate assumes
    spherical geometry at the initial time point and a stretch of 1.05
    """

    # Get midwall area of the LV at the first time point
    v_m_lv = model.volumes[0, 2] + 0.5 * (model.heart.vw_w[0] + model.heart.vw_w[2])
    am_m_lv = np.pi ** (1 / 3) * (6 * v_m_lv) ** (2 / 3)

    # Estimate stretch at the first time point
    lab = np.sqrt(am_m_lv / (model.heart.am_ref_w[0] + model.heart.am_ref_w[2]))

    # Find spherical cap height that matches the midwall reference area of the septal wall initial stretch
    h_s = (model.heart.am_ref_w[2] * lab ** 2) / (2 * np.pi * ((3 * v_m_lv) / (4 * np.pi)) ** (1 / 3))

    # ys is twice the base radius of the spherical cap
    r_m = ((3 * v_m_lv) / (4 * np.pi)) ** (1 / 3)
    r_s = np.sqrt(h_s * (2 * r_m - h_s))
    model.heart.ys = 2 * r_s

    # vs is the volume of the spherical cap
    model.heart.vs = 1 / 6 * np.pi * h_s * (3 * r_s ** 2 + h_s ** 2)
