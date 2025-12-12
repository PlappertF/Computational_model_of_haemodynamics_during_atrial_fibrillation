import pandas as pd
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np
import networkx as nx
import os

import src.utils as utils


def pressures_volumes(cg, x_lim=None, y_lim=None, compartment=[2], legend=None,
                      show_fig=False, title=''):
    """
    pressures_volumes plots pressure-volume loops from chosen model compartments

    :param cg: cardiogrowth object
    :param x_lim: The x bounds of the plot
    :param y_lim: The y bounds of the plot
    :param compartment: Compartment of the heart being graphed
    :param legend: Legend for the plot
    :param show_fig: Should the figure be shown in a popup window
    """

    time = cg.activation.time
    volumes = cg.volumes[:, compartment]
    pressures = cg.pressures[:, compartment]

    if not legend:
        legend = compartment
        legend_switch = False
    else:
        legend_switch = True

    # Pandas format
    pv_df = pd.DataFrame()
    for i in range(np.size(volumes, 1)):
        pv_df_i = wide2long_2vars(volumes[:, i], pressures[:, i], time, [legend[i]])
        pv_df = pd.concat([pv_df, pv_df_i], ignore_index=True)

    # PV loop
    f, ax = plt.subplots()
    g = sns.lineplot(data=pv_df, x="value", y="value2", hue="variable", legend=legend_switch, errorbar=None, sort=False)
    plt.xlabel("Volume (mL)")
    plt.ylabel("Pressure (mmHg)")
    if x_lim: plt.xlim(left=x_lim[0], right=x_lim[1])
    if y_lim: plt.ylim(bottom=y_lim[0], top=y_lim[1])

    plt.title(title)

    # Remove legend title
    if legend_switch: g.legend_.set_title(None)

    if show_fig:
        plt.show()
    else:
        plt.close()


class data_linewidth_plot():
    def __init__(self, x, y, **kwargs):
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = self.ax.get_figure()
        self.lw_data = kwargs.pop("linewidth", 1)
        self.lw = 1
        self.fig.canvas.draw()

        self.ppd = 72./self.fig.dpi
        self.trans = self.ax.transData.transform
        self.linehandle, = self.ax.plot([],[],**kwargs)
        if "label" in kwargs: kwargs.pop("label")
        self.line, = self.ax.plot(x, y, **kwargs)
        self.line.set_color(self.linehandle.get_color())
        self._resize()
        self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self, event=None):
        lw =  ((self.trans((1, self.lw_data))-self.trans((0, 0)))*self.ppd)[1]
        if lw != self.lw:
            self.line.set_linewidth(lw)
            self.lw = lw
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda : self.fig.canvas.draw_idle())
        self.timer.start()


def wide2long_2vars(data1, data2, index, columns):
    # Convert two numpy arrays vs time to single long-format pandas format (e.g. pressure vs. volume)

    df0 = pd.DataFrame(data1, index=index, columns=columns)
    df = pd.DataFrame(df0, columns=columns).reset_index().melt(id_vars = 'index').rename(columns={'index':'Time'})

    df0_2 = pd.DataFrame(data2, index=index, columns=columns)
    df_2 = pd.DataFrame(df0_2, columns=columns).reset_index().melt(id_vars = 'index').rename(columns={'index':'Time'})

    df.insert(3, "value2", df_2['value'])
    return df