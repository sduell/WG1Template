"""
This module contains functions and classes for plotting various kinds of
histograms of given data.
"""
import itertools
from collections import defaultdict
from typing import Tuple, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from scipy.linalg import block_diag
from uncertainties import unumpy as unp

import wg1template.plot_style as plot_style
from wg1template.plot_utilities import get_auto_ylims

plot_style.set_matplotlibrc_params()


class HistVariable:
    """
    Helper class with properties describing the variable which will be plotted
    with HistogramPlot classes.
    """

    def __init__(self,
                 df_label: str,
                 n_bins: int,
                 scope: Union[Tuple[float, float], None] = None,
                 var_name: Union[str, None] = None,
                 unit: Union[str, None] = None,
                 use_logspace: bool = False):
        """
        HistVariable constructor.
        :param df_label: Label of the variable for the column in a pandas
        dataframe.
        :param n_bins: Number of bins used in the histogram.
        :param scope: Tuple with the scope of the variable
        :param var_name: Name of the variable used for the x axis in Plots.
                         Preferably using Latex strings like r'$\mathrm{m}_{\mu\mu}$'.
        :param unit: Unit of the variable, like GeV.
        :param use_logspace: If true, x axis will be plotted in logspace.
                             Default is False.
        """
        self._df_label = df_label
        self._scope = scope
        self._var_name = var_name
        self._x_label = var_name + f' [{unit}]' if unit else var_name
        self._unit = unit
        self._n_bins = n_bins
        self._use_logspace = use_logspace

    @property
    def df_label(self) -> str:
        """
        Column name of the variable in a pandas dataframe.
        :return: str
        """
        return self._df_label

    def has_scope(self) -> bool:
        """
        Checks if scope is set.
        :return: True if HistVariable has scope parameter set, False otherwise.
        """
        if self._scope is not None:
            return True
        else:
            return False

    @property
    def n_bins(self):
        """
        Number of bins used in the histogram.
        :return: int

        """
        return self._n_bins

    @property
    def scope(self) -> Tuple[float, float]:
        """
        The scope of the variable as (low, high).
        :return: Tuple[float, float]
        """
        return self._scope

    @scope.setter
    def scope(self, value):
        self._scope = value

    @property
    def x_label(self):
        """
        X label of the variable shown in the plot, like r'$\cos(\theta_v)$'.
        :return: str
        """
        if self._x_label is not None:
            return self._x_label
        else:
            return ""

    @x_label.setter
    def x_label(self, label):
        self._x_label = label

    @property
    def unit(self):
        """
        Physical unit of the variable, like Gev.
        :return: str
        """
        if self._unit is not None:
            return self._unit
        else:
            return ""

    @unit.setter
    def unit(self, unit):
        self._unit = unit

    @property
    def use_logspace(self):
        """
        Flag for logscale on this axis
        :return: str
        """
        return self._use_logspace


class HistComponent:
    """
    Helper class for handling components of histograms.
    """

    def __init__(self,
                 label: str,
                 data: np.ndarray,
                 weights: Union[np.ndarray, None],
                 histtype: Union[str, None],
                 color: Union[str, None],
                 ls: str,
                 weight_uncerts: Union[np.ndarray, None] = None):
        """
        HistComponent constructor.
        :param label: Component label for the histogram.
        :param data: Data to be histogramed.
        :param weights: Weights for the events in data.
        :param histtype: Specifies the histtype of the component in the
        histogram.
        :param color: Color of the histogram component.
        :param ls: Linestyle of the histogram component.
        :param weight_uncerts: Weight uncertainties for the events (used for systematics)
        """
        self._label = label
        self._data = data
        self._weights = weights
        self._weight_uncerts = weight_uncerts
        self._histtype = histtype
        self._color = color
        self._ls = ls
        self._min = np.amin(data) if len(data) > 0 else +float("inf")
        self._max = np.amax(data) if len(data) > 0 else -float("inf")
        self._stat_cov=None
        self._sys_covs=[]

    def set_stat_cov(self, cov):
        self._stat_cov=cov

    def add_sys_cov(self, cov):
        self._sys_covs.append(cov)

    def add_variation(self, data, var, bin_edges, nominal_weight, total_weight=None, nominal_weight_up=None, nominal_weight_down=None, nominal_weight_uncertainty=None):
        """
        Adds the covariance matrix resulting from an uncertainty with simple up- and down error or
        a symmetric error uncertainty.
        """
        if nominal_weight_uncertainty==None:
            if nominal_weight_up is None or nominal_weight_down is None:
                raise ValueError("ERROR! Either a total uncertainty or up- and down-weights need to be given!")
            symm_uncert_mode=False
        else:
            if nominal_weight_up is not None and nominal_weight_down is not None:
                print("WARNING! Both, a total uncertainty and up- and down-weights are given. The total uncert. will be ignored.")
                symm_uncert_mode=False
            else:
                symm_uncert_mode=True
        if (total_weight==None):
            hnom = np.histogram(data[var], bins=bin_edges, weights=data[nominal_weight])[0]
            if symm_uncert_mode:
                hvar = np.histogram(data[var], bins=bin_edges, weights=(data[nominal_weight_uncertainty]/data[nominal_weight]))[0]
            else:
                hup = np.histogram(data[var], bins=bin_edges, weights=(data[nominal_weight_up]/data[nominal_weight]))[0]
                hdown = np.histogram(data[var], bins=bin_edges, weights=(data[nominal_weight_down]/data[nominal_weight]))[0]               
        else:
            hnom = np.histogram(data[var], bins=bin_edges, weights=data[total_weight])[0]
            if symm_uncert_mode:
                hvar = np.histogram(data[var], bins=bin_edges, weights=(data[nominal_weight_uncertainty]*data[total_weight]/data[nominal_weight]))[0]
            else:
                hup = np.histogram(data[var], bins=bin_edges, weights=(data[nominal_weight_up]*data[total_weight]/data[nominal_weight]))[0]
                hdown = np.histogram(data[var], bins=bin_edges, weights=(data[nominal_weight_down]*data[total_weight]/data[nominal_weight]))[0]
        if symm_uncert_mode:
            self.add_sys_cov(np.outer(hvar, hvar))
        else:
            sign = np.ones_like(hup)
            mask = hup < hdown
            sign[mask] = -1
            diff_up = np.abs(hup - hnom)
            diff_down = np.abs(hdown - hnom)
            diff_sym = (diff_up + diff_down) / 2
            signed_diff = sign * diff_sym
            cov_mat = np.outer(signed_diff, signed_diff)
            self.add_sys_cov(cov_mat)

    #adapted from the binfit package
    def add_gaussian_variation(self, data, var, bin_edges, nominal_weight, gaussian_base, Nstart=None, Nweights=None , total_weight=None):
        """
        Adds the covariance matrix resulting from an uncertainty which is given as Nvar gaussian
        variations of the same error.
        """
        if Nweights==None:
            Nweights = len([col for col in data.columns if gaussian_base in col])
        if total_weight == None:
            nominal = np.histogram(data[var], bins=bin_edges, weights=data[nominal_weight])[0]
            bin_counts = np.array([np.histogram(data[var], bins=bin_edges, weights=data['{}_{}'.format(gaussian_base,i)])[0] for i in range(Nstart, Nweights+Nstart)])
        else:
            nominal = np.histogram(data[var], bins=bin_edges, weights=data[total_weight])[0]
            bin_counts = np.array([np.histogram(data[var], bins=bin_edges, weights=data['{}_{}'.format(gaussian_base,i)]*data[total_weight]/data[nominal_weight])[0] for i in range(Nstart,Nweights+Nstart)])
        cov_mat = np.matmul((bin_counts - nominal).T, (bin_counts - nominal))/Nweights
        self.add_sys_cov(cov_mat)    

    @property
    def get_total_cov(self):
        # without the copy, the self.stat_cov would be altered...
        cov=np.copy(self.stat_cov)
        for syscov in self.sys_covs:
            cov += syscov
        return cov
    
    @property
    def get_sys_cov(self):
        return np.sum(np.array(self.sys_covs), axis=0)

    @property
    def sys_covs(self):
        return self._sys_covs

    @property
    def stat_cov(self):
        return self._stat_cov

    @property
    def label(self):
        return self._label

    @property
    def data(self):
        return self._data

    @property
    def weights(self):
        return self._weights

    @property
    def weight_uncerts(self):
        return self._weight_uncerts

    @property
    def histtype(self):
        return self._histtype

    @property
    def color(self):
        return self._color

    @property
    def ls(self):
        return self._ls

    @property
    def min_val(self):
        return self._min

    @property
    def max_val(self):
        return self._max


class HistogramPlot:
    """
    Base class for histogram plots.
    """

    def __init__(self,
                 variable: HistVariable):
        """
        HistogramPlot constructor.
        :param variable: A HistVariable describing the variable to be
        histogramed.
        """
        self._variable = variable
        self._num_bins = variable.n_bins
        self._mc_components = defaultdict(list)
        self._data_component = None
        self._bin_edges = None
        self._bin_mids = None
        self._bin_width = None
        self._cov_mats = list()
        self._cov = None
        self._corr = None
        self._inv_corr = None

    def add_component(self,
                      label: str,
                      data: Union[pd.DataFrame, pd.Series, np.ndarray],
                      weights: Union[str, pd.Series, np.ndarray, None] = None,
                      comp_type: str = 'single',
                      histtype: str = 'step',
                      color: str = None,
                      ls: str = 'solid'):
        """
        Add components to the histogram.

        :param label: Component label for the histogram.
        :param data: Data to be histogramed.
        :param weights: Weights for the events in data.
        :param comp_type:
        :param histtype: Specifies the histtype of the component in the
        histogram.
        :param color: Color of the histogram component.
        :param ls: Linestyle of the histogram component.
        """

        if isinstance(weights, float):
            weights = np.ones(len(data)) * weights

        if isinstance(weights, str):
            weights = data[weights].values

        if isinstance(data, pd.Series):
            data = data.values

        if isinstance(data, pd.DataFrame):
            data = data[self._variable.df_label].values

        if weights is None:
            weights = np.ones_like(data)

        assert len(data) == len(weights)

        if comp_type in ['single', 'stacked']:
            self._mc_components[comp_type].append(
                HistComponent(label=label,
                              data=data,
                              weights=weights,
                              histtype=histtype,
                              color=color,
                              ls=ls)
            )
        else:
            raise ValueError(f"Component type {comp_type} not know.")

    def _find_range_from_components(self) -> Tuple[float, float]:
        """
        Finds the scope tuple from the histogram components.

        :return: scope tuple.
        """
        min_vals = list()
        max_vals = list()

        for component in itertools.chain(*self._mc_components.values()):
            min_vals.append(np.amin(component.data))
            max_vals.append(np.amax(component.data))

        return np.amin(min_vals), np.amax(max_vals)

    def _get_bin_edges(self
                       ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculates the bin edges for the histogram.
        :return: Bin edges.
        """
        if self._variable.has_scope():
            scope = self._variable.scope
        else:
            scope = self._find_range_from_components()

        low, high = scope[0], scope[1]

        if self._variable.use_logspace:
            assert low > 0, \
                f"Cannot use logspace for variable {self._variable.x_label} since the minimum value is <= 0."
            bin_edges = np.logspace(np.log10(low), np.log10(high), self._num_bins + 1)
        else:
            bin_edges = np.linspace(low, high, self._num_bins + 1)

        bin_mids = (bin_edges[1:] + bin_edges[:-1]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        return bin_edges, bin_mids, bin_width

    def _get_y_label(self, normed: bool, bin_width: float, evts_or_cand="Events", categorical: bool = False) -> str:
        """
        Creates the appropriate  y axis label for the histogram plot.

        :param normed: Whether the label is for a normalized histogram
        or not.
        :param bin_width: Width of each bin (equal binning assumed).
        :return: The y axis label,
        """

        if normed:
            return "Normalized in arb. units"
        elif categorical:
            return evts_or_cand
        elif self._variable.use_logspace:
            return f"{evts_or_cand} / Bin"
        else:
            return "{} / ({:.2g}{})".format(evts_or_cand, bin_width, " " + self._variable.unit)

    @property
    def bin_edges(self):
        return self._bin_edges

    @property
    def bin_mids(self):
        return self._bin_mids

    @property
    def bin_width(self):
        return self._bin_width


class SimpleHistogramPlot(HistogramPlot):
    def __init__(self,
                 variable: HistVariable):
        """
        HistogramPlot constructor.
        :param variable: A HistVariable describing the variable to be
        histogramed.
        """
        super().__init__(variable=variable)

    def plot_on(self,
                ax: plt.axis,
                draw_legend: bool = True,
                legend_inside: bool = True,
                yaxis_scale: Union[str, float, int] = 1.2,
                normed: bool = False,
                ylabel: str = "Events",
                log_y: bool = False,
                categorical: bool = False,
                hide_labels: bool = False) -> plt.axis:
        """
        Plots the component on a given matplotlib.pyplot.axis

        :param ax: matplotlib.pyplot.axis where the histograms will be drawn
        on.
        :param draw_legend: Draw legend on axis if True.
        :param normed: If true the histograms are normalized.

        :return: matplotlib.pyplot.axis with histogram drawn on it
        """
        bin_edges, bin_mids, bin_width = self._get_bin_edges()

        self._bin_edges = bin_edges
        self._bin_mids = bin_mids
        self._bin_width = bin_width

        for component in self._mc_components['single']:
            if component.histtype == 'stepfilled':
                alpha = 0.6
                edge_color = 'black'
            else:
                edge_color = None
                alpha = 1.0
            hMC, _, _ = ax.hist(x=component.data,
                                   bins=bin_edges,
                                   density=normed,
                                   weights=component.weights,
                                   histtype=component.histtype,
                                   label=component.label,
                                   edgecolor=edge_color if edge_color is not None else component.color,
                                   alpha=alpha,
                                   lw=1.5,
                                   ls=component.ls,
                                   color=component.color)

        if not hide_labels:
            ax.set_xlabel(self._variable.x_label, plot_style.xlabel_pos)

            y_label = self._get_y_label(normed=normed, bin_width=bin_width, evts_or_cand=ylabel, categorical=categorical)
            ax.set_ylabel(y_label, plot_style.ylabel_pos)

        if draw_legend:
            if legend_inside:
                ax.legend(frameon=False)
                ymin, ymax = get_auto_ylims(ax, hMC, log_y=log_y, yaxis_scale=yaxis_scale)
                ax.set_ylim(ymin, ymax)
            else:
                ax.legend(frameon=False, bbox_to_anchor=(1, 1))

        if log_y:
            ax.set_yscale('log')

        ax.set_xlim(self._variable._scope)

        return ax


class StackedHistogramPlot(HistogramPlot):
    def __init__(self,
                 variable: HistVariable):
        """
        HistogramPlot constructor.
        :param variable: A HistVariable describing the variable to be
        histogramed.
        """
        super().__init__(variable=variable)

    def add_data_component(self,
                           label: str,
                           data: Union[pd.DataFrame, pd.Series, np.ndarray], ):
        if isinstance(data, pd.Series):
            data = data.values

        if isinstance(data, pd.DataFrame):
            data = data[self._variable.df_label].values

        self._data_component = HistComponent(
            label=label,
            data=data,
            weights=None,
            histtype=None,
            color=None,
            ls="",
        )

    def plot_on(self,
                ax: plt.axis,
                ylabel: str = "Events",
                draw_legend: bool = True,
                legend_inside: bool = True,
                yaxis_scale: Union[str, float, int] = 1.2,
                log_y: bool = False,
                categorical: bool = False,
                sort_components: bool = True,
                hide_labels: bool = False):
        bin_edges, bin_mids, bin_width = self._get_bin_edges()

        self._bin_edges = bin_edges
        self._bin_mids = bin_mids
        self._bin_width = bin_width

        if sort_components:
            self._mc_components['stacked'].sort(key=lambda x: len(x.data))
 
        hMC, _, _ = ax.hist(x=[comp.data for comp in self._mc_components['stacked']],
                            bins=bin_edges,
                            weights=[comp.weights for comp in self._mc_components['stacked']],
                            stacked=True,
                            edgecolor="black",
                            lw=0.3,
                            color=[comp.color for comp in self._mc_components['stacked']],
                            label=[comp.label for comp in self._mc_components['stacked']],
                            histtype='stepfilled'
                            )

        if not hide_labels:
            ax.set_xlabel(self._variable.x_label, plot_style.xlabel_pos)
            y_label = self._get_y_label(False, bin_width, ylabel, categorical=categorical)
            ax.set_ylabel(y_label, plot_style.ylabel_pos)
        if draw_legend:
            if legend_inside:
                ax.legend(frameon=False, loc='upper right')
                ymin, ymax = get_auto_ylims(ax, hMC, log_y=log_y, yaxis_scale=yaxis_scale)
                ax.set_ylim(ymin, ymax)
            else:
                ax.legend(frameon=False, bbox_to_anchor=(1, 1))

        if log_y:
            ax.set_yscale('log')

        ax.set_xlim(self._variable._scope)

        return ax


class DataMCHistogramPlot(HistogramPlot):
    def __init__(self,
                 variable: HistVariable):
        """
        HistogramPlot constructor.
        :param variable: A HistVariable describing the variable to be
        histogramed.
        """
        super().__init__(variable=variable)
        # if true, the stat. and sys. cov. matrices are also stored in each component
        self.covs_to_comps = False
        # the global statistical covariance matrix
        self._stat_cov=None
        # list of global systematic covariance matrices
        self._sys_covs=[]
        # total global syst. cov. matrix (simple sum of self._sys_covs)
        self._total_sys_cov=None
        # total global cov. (stat + syst)
        self._total_cov=None
        # flag to indicate if the total sum was already properly calculated
        self._calc_tot_cov=False

    def add_data_component(self,
                           label: str,
                           data: Union[pd.DataFrame, pd.Series, np.ndarray], ):
        if isinstance(data, pd.Series):
            data = data.values

        if isinstance(data, pd.DataFrame):
            data = data[self._variable.df_label].values

        self._data_component = HistComponent(
            label=label,
            data=data,
            weights=None,
            histtype=None,
            color=None,
            ls="",
        )

    def add_mc_component(self,
                         label: str,
                         data: Union[pd.DataFrame, pd.Series, np.ndarray],
                         weights: Union[str, pd.Series, np.ndarray, None] = None,
                         weight_uncerts: Union[str, pd.Series, np.ndarray, None] = None,
                         color: str = None,
                         ls: str = 'solid'):

        if isinstance(data, pd.Series):
            data = data.values

        if isinstance(data, pd.DataFrame):
            data = data[self._variable.df_label].values

        if weights is None:
            weights = np.ones(len(data))

        if isinstance(weights, float):
            weights = np.ones(len(data)) * weights

        if isinstance(weights, str):
            weights = data[weights].values

        if isinstance(weight_uncerts, float):
            weight_uncerts = np.ones(len(data)) * weight_uncerts

        if isinstance(weight_uncerts, str):
            weight_uncerts = data[weight_uncerts].values

        assert len(data) == len(weights)
        if weight_uncerts is not None:
            assert len(data) == len(weight_uncerts)

        self._mc_components["MC"].append(
            HistComponent(
                label=label,
                data=data,
                weights=weights,
                weight_uncerts=weight_uncerts,
                histtype=None,
                color=color,
                ls=ls,
            )
        )

    def plot_on(self,
                ax1: plt.axis,
                ax2: plt.axis,
                style: str = "stacked",
                ylabel: str = "Events",
                sum_color: str = plot_style.KITColors.kit_purple,
                draw_legend: bool = True,
                legend_inside: bool = True,
                yaxis_scale: Union[str, float, int] = 1.2,
                sort_components: bool = True,
                categorical: bool = False,
                log_y: bool = False,
                lower_plot_mode: str = "ratio",
                lower_yaxis_range: Union[Tuple[float, float], bool, None] = None,
                include_systematics: bool = False
                ):
        bin_edges, bin_mids, bin_width = self._get_bin_edges()

        self._bin_edges = bin_edges
        self._bin_mids = bin_mids
        self._bin_width = bin_width

        sum_w = np.sum(
            np.array([binned_statistic(comp.data, comp.weights, statistic="sum", bins=bin_edges)[0] for comp in
                      self._mc_components["MC"]]), axis=0)

        if not self._calc_tot_cov:
            sum_w2 = np.sum(
                np.array([binned_statistic(comp.data, comp.weights ** 2, statistic="sum", bins=bin_edges)[0] for comp in
                          self._mc_components["MC"]]), axis=0)
            if include_systematics:
                # this is not taking any correlations into account and thus strongly underestimates the uncertainty!
                sum_wu2 = np.sum(
                    np.array([binned_statistic(comp.data, comp.weight_uncerts ** 2, statistic="sum", bins=bin_edges)[0] for comp in
                              self._mc_components["MC"]]), axis=0)
                sum_w2 += sum_wu2 # total^2 = [sqrt(stat^2 + syst^2)]^2
            
        else:
            include_systematics = True
            # for the total uncertainty, for each bin take every uncertainty and correlation into account:
            # self._total_cov has size (comps*bins*bins)*(comps*bins*bins)
            # the block diagonals are the covariance matrices of each component
            # the rest are cov. matrices between components. It is split in bin*bin matrices.
            # For the total error calculate: (b=bin, c=comp):
            # b1 = sum(i over comps){b1_ci^2 (stat+sys)} + sum(i over comps, j over bins){cov(b1_ci,bj_ci)} ...
            #  + sum(i over comps, j over bins){cov(b1_c1, bj_ci)} + sum(i over comps, j over bins, k over other comps){cov(b1_ci, bj_ck)}
            # much clearer with a drawing... equ. to DelC^2 = DelA^2 + DelB^2 + 2*Cov(A,B)
            mat = np.sum(np.concatenate([np.vsplit(column, len(self._mc_components["MC"])) for column in np.hsplit(self._total_cov, len(self._mc_components["MC"]))]), axis=0)
            sum_w2=np.sum(mat, axis=1)
            
        hdata, _ = np.histogram(self._data_component._data, bins=bin_edges)


        if sort_components:
            self._mc_components['stacked'].sort(key=lambda x: len(x.data))
 
        if style.lower() == "stacked":
            hMC, _, _ = ax1.hist(x=[comp.data for comp in self._mc_components['MC']],
                                 bins=bin_edges,
                                 weights=[comp.weights for comp in self._mc_components['MC']],
                                 stacked=True,
                                 edgecolor="black",
                                 lw=0.3,
                                 color=[comp.color for comp in self._mc_components['MC']],
                                 label=[comp.label for comp in self._mc_components['MC']],
                                 histtype='stepfilled'
                                 )
            if include_systematics:
                errorlabel = "MC all. unc."
            else:
                errorlabel = "MC stat. unc."
            #TODO: It is relatively easy to distinguish b/w stat. and syst. errors with self._stat_cov
            # and self._total_sys_cov. I might want to highlight both differently here:
            ax1.bar(
                x=bin_mids,
                height=2 * np.sqrt(sum_w2),
                width=self.bin_width,
                bottom=sum_w - np.sqrt(sum_w2),
                color="black",
                hatch="///////",
                fill=False,
                lw=0,
                label=errorlabel
            ) 
        if style.lower() == "summed":
            ax1.bar(
                x=bin_mids,
                height=2 * np.sqrt(sum_w2),
                width=self.bin_width,
                bottom=sum_w - np.sqrt(sum_w2),
                color=sum_color,
                lw=0,
                label="MC"
            )

        ax1.errorbar(x=bin_mids, y=hdata, yerr=np.sqrt(hdata),
                     ls="", marker=".", color="black", label=self._data_component.label)

        y_label = self._get_y_label(False, bin_width, evts_or_cand=ylabel, categorical=categorical)
        # ax1.legend(loc=0, bbox_to_anchor=(1,1))
        ax1.set_ylabel(y_label, plot_style.ylabel_pos)

        ax1.set_xlim(self._variable._scope)

        if log_y:
            ax1.set_yscale('log')

        if draw_legend:
            if legend_inside:
                ax1.legend(frameon=False, loc = 'upper right', fontsize='x-small')
                ymin, ymax = get_auto_ylims(ax1, hMC, hdata=hdata, log_y=log_y, yaxis_scale=yaxis_scale)
                ax1.set_ylim(ymin, ymax)
            else:
                ax1.legend(frameon=False, bbox_to_anchor=(1, 1))

        if lower_plot_mode == "ratio":
            lower_ylabel = r"$\frac{\mathrm{Data - MC}}{\mathrm{MC}}$"
            labelpad = -4
            lower_ymin = -0.3
            lower_ymax = 0.3
        elif lower_plot_mode == "residuals":
            lower_ylabel = r"$\frac{N_{\mathrm{Data}} - N_{\mathrm{MC}}}{\sqrt{\sigma^{2}_{\mathrm{Data}}+\sigma^{2}_{\mathrm{MC}}}}$"
            labelpad = -6
            lower_ymin = -3.
            lower_ymax = 3.
        else:
            raise ValueError(f"ERROR! Unknown lower plot mode: {lower_plot_mode}. Choose between 'ratio' and 'residuals'.")
        ax2.set_ylabel(lower_ylabel, labelpad=labelpad)
        ax2.set_xlabel(self._variable.x_label, plot_style.xlabel_pos)
        ax2.set_xlim(self._variable._scope)
        if isinstance(lower_yaxis_range, tuple):
            lower_ymin = lower_yaxis_range[0]
            lower_ymax = lower_yaxis_range[1]
        ax2.set_ylim((lower_ymin, lower_ymax))

        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                if lower_plot_mode.lower() == "ratio":
                    lower_points = (hdata - sum_w) / sum_w
                elif lower_plot_mode.lower() == "residuals":
                    lower_points = (hdata-sum_w)/np.sqrt(hdata + sum_w2)
                lower_points[lower_points == np.inf] = 0
                lower_points[lower_points == -np.inf] = 0
                lower_points = np.nan_to_num(lower_points)

            with np.errstate(divide='ignore', invalid='ignore'):
                if lower_plot_mode.lower() == "ratio":
                    #lower_errors = np.sqrt((hdata + sum_w) / (hdata - sum_w)**2 + 1/sum_w) * (hdata - sum_w) / sum_w
                    lower_errors = np.sqrt(hdata/(sum_w**2)+(hdata**2)*sum_w2/(sum_w**4)) # this should be correct, I don't know what was done above...
                elif lower_plot_mode.lower() == "residuals":
                    lower_errors = np.full_like(sum_w, 1.) # show 1 sigma bars for this case
                lower_errors[lower_errors == np.inf] = 0
                lower_errors[lower_errors == -np.inf] = 0
                lower_errors = np.nan_to_num(lower_errors)

            ax2.axhline(y=0, color=plot_style.KITColors.dark_grey, alpha=0.8)
            if isinstance(lower_yaxis_range, bool) and lower_yaxis_range == False:
                lower_ymin = lower_points.min() - lower_errors.max()
                lower_ymax = lower_points.max() + lower_errors.max()
                ax2.set_ylim((lower_ymin, lower_ymax))
            else: # draw arrows if points are out of axis range:
                low_arrow_mids = bin_mids[np.add(lower_points, lower_errors) <= lower_ymin]
                high_arrow_mids = bin_mids[np.subtract(lower_points, lower_errors) >= lower_ymax]
                arrowlength = (lower_ymax - lower_ymin) / 4.
                for arrow_pos in low_arrow_mids:
                    ax2.annotate(s='', xy=(arrow_pos, lower_ymin), xytext=(arrow_pos, lower_ymin+arrowlength), arrowprops=dict(arrowstyle="-|>", color=plot_style.TangoColors.scarlet_red))
                for arrow_pos in high_arrow_mids:
                    ax2.annotate(s='', xy=(arrow_pos, lower_ymax), xytext=(arrow_pos, lower_ymax-arrowlength), arrowprops=dict(arrowstyle="-|>", color=plot_style.TangoColors.scarlet_red))
            #ax2.errorbar(bin_mids, unp.nominal_values(ratio), yerr=unp.std_devs(ratio),
            ax2.errorbar(bin_mids, lower_points, yerr=lower_errors,
                         ls="", marker=".", color=plot_style.KITColors.kit_black)
        except ZeroDivisionError:
            #ax2.axhline(y=0, color=plot_style.KITColors.dark_grey, alpha=0.8)
            raise ZeroDivisionError
        
        plt.subplots_adjust(hspace=0.08)
    
    def add_stat_covs(self):
        """
        Adds statistical covariance matrix for all components
        """
        bin_edges, _, _ = self._get_bin_edges()
        stat_covs = []
        for comp in self._mc_components["MC"]:
            stat_errors_sq = np.histogram(comp.data,bins=bin_edges,weights=comp.weights**2)[0]
            stat_errors_sq[stat_errors_sq == 0] = 1e-14
            if self.covs_to_comps:
                comp.set_stat_cov(np.diag(stat_errors_sq))
            stat_covs.append(np.diag(stat_errors_sq))
        self._stat_cov = np.matrix(block_diag(*stat_covs))

    def print_tot_cov(self):
        """
        Prints the global covariance matrix.
        """
        if not self._calc_tot_cov:
            self.calc_tot_cov_matrix
        print(self._total_cov.shape)
        for i, elem in enumerate(self._total_cov):
            print(i, elem)
        return self._total_cov

    def calc_tot_cov_matrix(self):
        """
        Calculates total covariance matrix.
        """
        if len(self._sys_covs) == 0:
            # if plot class has no sys covs, the ones from the components are taken
            # In my opinion, this is wrong as the correlation between diff. comps is neglected.
            tot_sys_mats = [comp.get_sys_cov for comp in self._mc_components["MC"]]
            tot_mats = [comp.get_total_cov for comp in self._mc_components["MC"]]
            self._total_sys_cov = np.matrix(block_diag(*tot_sys_mats))
            self._total_cov = np.matrix(block_diag(*tot_mats))
        else:
            sys_cov = np.sum(np.array(self._sys_covs),axis=0)
            stat_cov = self._stat_cov
            # in my opinion, the following simply wrong... But I do not understand this in binfit, either...:
            #sys_cov=sys_cov*(stat_cov==0)
            self._total_sys_cov = sys_cov
            self._total_cov = np.matrix(sys_cov+stat_cov)
        self._calc_tot_cov = True

    def add_variation(self, inputdfdict, inkeys, namedict, weightdict):
        """
        Adds the covariance matrix resulting from an uncertainty with simple up- and down error or
        a symmetric error uncertainty.
        If up- and down-errors are given in the weight_dict, they are used, otherwise a symmetric error 
        is assumed.
        :param inputdfdict: dictionary of input data frames
        :param inkeys: list of keys that should actually be used
        :param namedict: dictionary of labels of the data frames given
        :param weightdict: dictionary of weights to be used
        """
        keys = inkeys
        nominal_weight=weightdict['nominal_weight'] # dataframe name of the nominal weight of the uncertainty
        total_weight=weightdict['total_weight'] # dataframe name of the total weight
        if 'nominal_weight_up' in weightdict.keys() and 'nominal_weight_down' in weightdict.keys():
            # up- and down weights need to be total size!
            nominal_weight_up=weightdict['nominal_weight_up']
            nominal_weight_down=weightdict['nominal_weight_down']
            nominal_weight_uncert=None # if both is given, the uncert. is ignored!
            use_symm_errors=False
        elif 'nominal_weight_uncertainty' in weightdict.keys():
            # uncertainty needs to be only the uncert. so nom_weight +- nom_weight_uncert gives up- and down
            nominal_weight_up=None
            nominal_weight_down=None
            nominal_weight_uncert=weightdict['nominal_weight_uncertainty']
            use_symm_errors=True

        bin_edges, _, _ = self._get_bin_edges()

        # create nominal histogram with unaltered weight
        nom = {key : np.histogram(inputdfdict[key][self._variable.df_label],bins=bin_edges,weights=inputdfdict[key][total_weight])[0] for key in keys}
        hnom = np.concatenate(list(nom.values()))
        if use_symm_errors:
            # create histogram with size of variation in each bin (meant to be same size for up and down)
            var = {key : np.histogram(inputdfdict[key][self._variable.df_label],bins=bin_edges,weights=(inputdfdict[key][nominal_weight_uncert]*inputdfdict[key][total_weight]/inputdfdict[key][nominal_weight]))[0] for key in keys}
            hvar = np.concatenate(list(var.values()))
        else:
            # create histograms with minimal weight and maximal weight
            up = {key : np.histogram(inputdfdict[key][self._variable.df_label],bins=bin_edges,weights=(inputdfdict[key][nominal_weight_up]*inputdfdict[key][total_weight]/inputdfdict[key][nominal_weight]))[0] for key in keys}
            down = {key : np.histogram(inputdfdict[key][self._variable.df_label],bins=bin_edges,weights=(inputdfdict[key][nominal_weight_down]*inputdfdict[key][total_weight]/inputdfdict[key][nominal_weight]))[0] for key in keys}
            hup = np.concatenate(list(up.values()))
            hdown = np.concatenate(list(down.values()))

        #save the single systematic covariance matrices
        #TODO: At the moment, they are calculated again... This is not necessary at all...
        if self.covs_to_comps:
            for key in keys:
                for comp in self._mc_components["MC"]:
                    if(namedict[key]==comp.label):
                        comp.add_variation(inputdfdict[key], self._variable.df_label, bin_edges, nominal_weight, total_weight, 
                                            nominal_weight_up=nominal_weight_up, nominal_weight_down=nominal_weight_down,
                                            nominal_weight_uncertainty=nominal_weight_uncert)
        if use_symm_errors:
            # for symm error case, covariance is simple due to symmetry:
            self._sys_covs.append(np.outer(hvar, hvar))
        else:
            sign = np.ones_like(hup)
            mask = hup < hdown
            sign[mask] = -1
            # calculate difference of error-histos: (in symm case, this is the direct result)
            diff_up = np.abs(hup - hnom)
            diff_down = np.abs(hdown - hnom)
            # average up- and down uncertainty
            diff_sym = (diff_up + diff_down) / 2
            signed_diff = sign * diff_sym
            covMatrix = np.outer(signed_diff, signed_diff)        
            self._sys_covs.append(covMatrix)     
            

    def add_gaussian_variations(self, inputdfdict, inkeys, namedict, weightdict, Nstart, Nvar):
        """
        Adds the covariance matrix resulting from an uncertainty which is given as Nvar gaussian
        variations of the same error.
        :param inputdfdict: dictionary of input data frames
        :param inkeys: list of keys that should actually be used
        :param namedict: dictionary of labels of the data frames given
        :param weightdict: dictionary of weights to be used.
        :param Nstart: number of gaussian error to start with
        :param Nvar: number of variations of the gaussian error to use
        """
        keys = inkeys
        total_weight=weightdict['total_weight'] # dataframe name of the total weight
        nominal_weight=weightdict['nominal_weight'] # dataframe name of the nominal weight of the uncertainty
        gaussian_base=weightdict['gaussian_basename'] # basename of the gaussian error in the dataframe.
        # It's assumed that they are further distinguished by a number (from Nstart to Nvar) divided with an underscore

        bin_edges, _, _ = self._get_bin_edges()

        # create nominal histogram with unaltered weight
        n = {key : np.histogram(inputdfdict[key][self._variable.df_label],bins=bin_edges,weights=inputdfdict[key][total_weight])[0] for key in keys}
        hnom = np.concatenate(list(n.values()))

        varMatrix=[]

        #save the single systematic covariance matrices
        #TODO: At the moment, they are calculated again... This is not necessary at all...
        if self.covs_to_comps:
            for key in keys:
                for comp in self._mc_components["MC"]:
                    if(namedict[key]==comp.label):
                        comp.add_gaussian_variation(inputdfdict[key], self._variable.df_label, bin_edges, nominal_weight, gaussian_base, Nstart, Nvar, total_weight)

        #create (Nvar-Nstart) times a varied histogram with the gaussian errors:
        for i in range(Nstart, Nstart+Nvar):
            ntemp = [np.histogram(inputdfdict[key][self._variable.df_label],bins=bin_edges,weights=(inputdfdict[key]['{}_{}'.format(gaussian_base,i)]*inputdfdict[key][total_weight]/inputdfdict[key][nominal_weight]))[0] for key in keys]
            row = np.concatenate(ntemp)
            varMatrix.append(row)
        varMatrix=np.array(varMatrix)
        # calculate the global covariance matrix:
        # check difference of nominal and varied histo.
        # Note that hnom is of shape 1*[bins*comps], varMatrix is (Nvar-Nstart)*[bins*comps]
        # The result is of shape [bins*comps]*[bins*comps]
        covMatrix=np.matmul((varMatrix-hnom).T,(varMatrix-hnom))/Nvar
        self._sys_covs.append(covMatrix)


    #adapted from Lu Cao's systematics code
    def add_PID_variations(self, inputdfdict, inkeys, namedict, weight_names, Nstart, Nvar):
        keys = inkeys
        nominal_weight=weight_names['nominal_weight']
        total_weight=weight_names['total_weight']
        new_weight=weight_names['new_weight']

        bin_edges, _, _ = self._get_bin_edges()

        # create nominal histogram with unaltered weight for every component:
        for key in keys:
            hist_central, _ = np.histogram(inputdfdict[key][self._variable.df_label], bins=bin_edges, 
                                density=False, weights=inputdfdict[key][total_weight])

            # create (Nvar-Nstart) times histograms, each with one gaussian error for every component
            hist_dict = {}
            for i in range(Nstart, Nstart+Nvar):
                hist_dict['{}_{}'.format(new_weight,i)], _ = np.histogram(
                    inputdfdict[key][self._variable.df_label], bins=bin_edges, density=False,
                    weights= inputdfdict[key]['{}_{}'.format(new_weight,i)]*inputdfdict[key][total_weight]/inputdfdict[key][nominal_weight])

            # calculate discrepancy of nominal and gaussian-error-hist as well as the standard deviation:
            stdv = [] 
            delta = np.array([])

            for bin_index in range(0, len(bin_edges)-1):
                bin_yields = []
                for w_index in hist_dict.keys():
                    bin_yields = np.append(bin_yields,hist_dict[w_index][bin_index])
                bin_delta = hist_central[bin_index] - bin_yields
                delta = np.append(delta, bin_delta)
                bin_stdv = np.sqrt((bin_delta * bin_delta).sum()/Nvar)
                stdv = np.append(stdv, bin_stdv)

            delta = delta.reshape(len(bin_edges)-1, Nvar)
            r = np.zeros((len(bin_edges)-1, len(bin_edges)-1)) 
            cov = np.zeros((len(bin_edges)-1, len(bin_edges)-1)) 

            # calculate covariance matrix now.
            # in my opinion, Will is doing slightly better in the gaussian function above as the denominator
            # and the two stdvs will cancel out. Thus, it is unnecessary to calculate them.
            # Moreover, the correlation between different components is not taken into account!
            # So, I think that this here is slightly wrong!
            for x in range(0, len(bin_edges)-1): 
                for y in range(0, len(bin_edges)-1):    
                    nom = (delta[x] * delta[y]).sum()
                    denom = np.sqrt((delta[x] * delta[x]).sum()) * np.sqrt( (delta[y] * delta[y]).sum())
                    # Pearson correltation coefficient matrix
                    if denom !=0:
                        r[x][y] = nom/denom 
                    # Covariance matrix
                    cov[x][y] = r[x][y] * stdv[x] * stdv[y]
            for comp in self._mc_components["MC"]:
                if(namedict[key]==comp.label):
                    comp.add_sys_cov(cov)



class PrebinnedDataMCHistogramPlot(DataMCHistogramPlot):
    def __init__(self,
                 variable: HistVariable):
        """
        HistogramPlot constructor for the case where we already have bin yields, *e.g.*
        from a fit.
        :param variable: A HistVariable describing the variable to be
        histogramed.
        """
        super().__init__(variable=variable)
        self._mc_bin_uncertainties = None
        self._mc_uncertainty_label = None


    def add_mc_uncertainty(self,
                           label: str = "MC stat. unc.",
                           mc_bin_uncertainties: Union[pd.DataFrame, pd.Series, np.ndarray, str, list] = None,):

        self._mc_uncertainty_label = label
        self._mc_bin_uncertainties = mc_bin_uncertainties

        if isinstance(mc_bin_uncertainties, pd.Series):
            self._mc_bin_uncertainties = mc_bin_uncertainties.values

        if isinstance(mc_bin_uncertainties, pd.DataFrame):
            self._mc_bin_uncertainties = mc_bin_uncertainties[self._variable.df_label].values

        if isinstance(mc_bin_uncertainties, str):
            self._mc_bin_uncertainties = self._data_component._data[mc_bin_uncertainties].values

        if isinstance(mc_bin_uncertainties, list):
            self._mc_bin_uncertainties = np.array(mc_bin_uncertainties)

        if mc_bin_uncertainties is not None:
            assert len(self._mc_bin_uncertainties) == self._num_bins


    def add_mc_component(self,
                         label: str,
                         bin_yields: Union[pd.DataFrame, pd.Series, np.ndarray, list],
                         color: str = None,
                         ls: str = 'solid'):

        bin_left_edges = np.linspace(*self._variable.scope,
                                     self._num_bins, endpoint=False)

        if isinstance(bin_yields, pd.Series):
            bin_yields = bin_yields.values

        if isinstance(bin_yields, pd.DataFrame):
            bin_yields = bin_yields[self._variable.df_label].values

        if isinstance(bin_yields, list):
            bin_yields = np.array(bin_yields)

        assert len(bin_yields) == self._num_bins

        self._mc_components["MC"].append(
            HistComponent(
                label=label,
                data=bin_left_edges,
                weights=bin_yields,
                histtype=None,
                color=color,
                ls=ls,
            )
        )
    
    def plot_on(self,
                ax1: plt.axis,
                ax2: plt.axis,
                style: str = "stacked",
                ylabel: str = "Events",
                sum_color: str = plot_style.KITColors.kit_purple,
                draw_legend: bool = True,
                legend_inside: bool = True,
                yaxis_scale: Union[str, float, int] = 1.2,
                sort_components: bool = True,
                categorical: bool = False,
                log_y: bool = False,
                ):
        bin_edges, bin_mids, bin_width = self._get_bin_edges()

        self._bin_edges = bin_edges
        self._bin_mids = bin_mids
        self._bin_width = bin_width

        sum_w = np.sum(
            np.array([binned_statistic(comp.data, comp.weights, statistic="sum", bins=bin_edges)[0] for comp in
                      self._mc_components["MC"]]), axis=0)

        sum_w2 = np.sum(
            np.array([binned_statistic(comp.data, comp.weights ** 2, statistic="sum", bins=bin_edges)[0] for comp in
                      self._mc_components["MC"]]), axis=0)

        hdata, _ = np.histogram(self._data_component._data, bins=bin_edges)


        if sort_components:
            self._mc_components['stacked'].sort(key=lambda x: len(x.data))
 
        if style.lower() == "stacked":
            hMC, _, _ = ax1.hist(x=[comp.data for comp in self._mc_components['MC']],
                                 bins=bin_edges,
                                 weights=[comp.weights for comp in self._mc_components['MC']],
                                 stacked=True,
                                 edgecolor="black",
                                 lw=0.3,
                                 color=[comp.color for comp in self._mc_components['MC']],
                                 label=[comp.label for comp in self._mc_components['MC']],
                                 histtype='stepfilled'
                                 )

            if self._mc_bin_uncertainties is not None:
                ax1.bar(
                    x=bin_mids,
                    height=(self._mc_bin_uncertainties + np.minimum(self._mc_bin_uncertainties, sum_w)),
                    width=self.bin_width,
                    bottom=sum_w - np.minimum(self._mc_bin_uncertainties, sum_w),
                    color="black",
                    hatch="///////",
                    fill=False,
                    lw=0,
                    label=self._mc_uncertainty_label
                )

        if style.lower() == "summed":
            ax1.bar(
                x=bin_mids,
                height=2 * np.sqrt(sum_w2),
                width=self.bin_width,
                bottom=sum_w - np.sqrt(sum_w2),
                color=sum_color,
                lw=0,
                label="MC"
            )

        ax1.errorbar(x=bin_mids, y=hdata, yerr=np.sqrt(hdata),
                     ls="", marker=".", color="black", label=self._data_component.label)

        y_label = self._get_y_label(False, bin_width, evts_or_cand=ylabel, categorical=categorical)
        # ax1.legend(loc=0, bbox_to_anchor=(1,1))
        ax1.set_ylabel(y_label, plot_style.ylabel_pos)

        ax1.set_xlim(self._variable._scope)

        if log_y:
            ax1.set_yscale('log')

        if draw_legend:
            if legend_inside:
                ax1.legend(frameon=False, loc = 'upper right', fontsize='x-small')
                ymin, ymax = get_auto_ylims(ax1, hMC, hdata=hdata, log_y=log_y, yaxis_scale=yaxis_scale)
                ax1.set_ylim(ymin, ymax)
            else:
                ax1.legend(frameon=False, bbox_to_anchor=(1, 1))

        ax2.set_ylabel(r"$\frac{\mathrm{Data - MC}}{\mathrm{MC}}$")
        ax2.set_xlabel(self._variable.x_label, plot_style.xlabel_pos)
        ax2.set_xlim(self._variable._scope)
        ax2.set_ylim((-1, 1))

        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = (hdata-sum_w)/sum_w
                ratio[ratio == np.inf] = 0
                ratio[ratio == -np.inf] = 0
                ratio = np.nan_to_num(ratio)

            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_errors = np.sqrt((hdata + sum_w)/(hdata - sum_w)**2 + 1/sum_w)*(hdata-sum_w)/sum_w
                ratio_errors[ratio_errors == np.inf] = 0
                ratio_errors[ratio_errors == -np.inf] = 0
                ratio_errors = np.nan_to_num(ratio_errors)

            ax2.axhline(y=0, color=plot_style.KITColors.dark_grey, alpha=0.8)
            ax2.set_ylim((ratio.min() - ratio_errors.max(), ratio.max() + ratio_errors.max()))
            #ax2.errorbar(bin_mids, unp.nominal_values(ratio), yerr=unp.std_devs(ratio),
            ax2.errorbar(bin_mids, ratio, yerr=ratio_errors,
                         ls="", marker=".", color=plot_style.KITColors.kit_black)
        except ZeroDivisionError:
            #ax2.axhline(y=0, color=plot_style.KITColors.dark_grey, alpha=0.8)
            raise ZeroDivisionError

        plt.subplots_adjust(hspace=0.08)


def create_hist_ratio_figure():
    """Create a matplotlib.Figure for histogram ratio plots.

    :return: A maptlotlib.Figure instance and a matplotlib.axes instance.
    """
    return plt.subplots(2, 1, figsize=(5, 5), dpi=200, sharex=True, gridspec_kw={"height_ratios": [3.5, 1]})


def create_solo_figure(figsize=(5, 5), dpi=200):
    return plt.subplots(1, 1, figsize=figsize, dpi=dpi)


def add_descriptions_to_plot(ax: plt.axis,
                             experiment: Union[str, None] = None,
                             luminosity: Union[str, None] = None,
                             additional_info: Union[str, None] = None,
                             ):
    if experiment is not None:
        ax.set_title(experiment, loc="left", fontdict={'size': 16, 'style': 'normal', 'weight': 'bold'})
    if luminosity is not None:
        ax.set_title(luminosity, loc="right")
    if additional_info is not None:
        ax.annotate(
            additional_info, (0.02, 0.98), xytext=(4, -4), xycoords='axes fraction',
            textcoords='offset points',
            fontsize=9,
            ha='left', va='top'
        )

    try:
        ax.ticklabel_format(style='plain', axis='x')
    except AttributeError:
        pass

    try:
        ax.ticklabel_format(style='plain', axis='y')
    except AttributeError:
        pass


class MultipleDataMCHistogramPlots(object):
    """
    Class to automatically plot a full list of variables.
    """
    def __init__(self,
                 additional_info: str = r"$B\rightarrow X\tau\nu$ Work in Progress",
                 cuts: Union[str, None] = None,
                 lower_plot_mode: str = "ratio",
                 columns: int = 3,
                 luminosity: Union[Tuple[float, float], float] = (1., 0.),
                 signal_index: Union[int, None] = None,
                 signal_enhancement: Union[int, None] = 10,
                 ignorelist: List[str] = []):
        """
        MultipleDataMCHistogramPlots constructor.
        :param additional_info: General string that should be shown on every plot.
        :param cuts: Optional. If selected, the cuts are applied before the plot.
        :param lower_plot_mode: Mode to decide what to show in the lower plot. Choose between 'ratio' and 'residuals'.
        :param columns: Number of columns of the multiplot. Default is 3
        :param luminosity: Luminosity with which all MC is scaled up. If tuple: (lumi, lumi_uncert)
        :param signal_index: If selected, the channel with the corresponding index is plotted in front of
                             the stackplot in the same color as the index with the signal_enhancement factor.
        :param signal_enhancement: Factor by which the extra signal plot is enhanced. Default is 10.
        :param ignorelist: list of variable names that should be ignored. They are not added to the plotbook.
        """
        self._variables = []
        self._components = defaultdict(list)
        self.cuts = cuts
        if cuts is not None:
            additional_info += "\n" + cuts.replace(' and ', '\n')
        self.additional_info = additional_info
        self.lower_plot_mode = lower_plot_mode
        if lower_plot_mode == 'ratio':
            self.lower_yaxis_range = (-0.3, 0.3) # it's possible to change that to 'False' if wanted
        elif lower_plot_mode == 'residuals':
            self.lower_yaxis_range = (-3., 3.)
        self.columns = columns
        if isinstance(luminosity, float): # if no uncert. is given, set it to zero.
            luminosity = (luminosity, 0.)
        self.luminosity = luminosity
        self.signal_index = signal_index
        self.signal_enhancement = signal_enhancement
        self.ignorelist = ignorelist
        
    def _add_component(self,
                       dict_key: str,
                       content: pd.DataFrame,
                       label: str,
                       color: str,
                       weightlabel: Union[str, None] = None,
                       weightuncert: Union[str, None] = None):
        """
        Adds component that will be plotted.
        :param dict_key: Dictionary key of the specific content.
        :param content: pd.DataFrames that will be plotted.
        :param label: Label of the specific content.
        :param color: Color of the specific content.
        :param weightlabel: Dictionary name of the weights to be applied.
        :param weightuncert: Dictionary name of the weight uncertainties.
        """
        if isinstance(self.cuts, str):
            skimmed_content = content.query(self.cuts)
        else:
            skimmed_content = content
        self._components[dict_key] = [skimmed_content, label, color, weightlabel, weightuncert]
    
    def add_mc_components(self,
                          channels: List[pd.DataFrame],
                          labels: Union[List[str], None] = None,
                          colors: Union[List[str], None] = None,
                          weightlabel: Union[str, None] = None,
                          weightuncert: Union[str, None] = None):
        """
        Adds all MC components. If previous components were added, they are deleted!
        :param channels: List of pd.DataFrames that will be plotted.
        :param labels: List of labels. Needs to be at least as long as the number of channels.
        :param colors: List of colors. Needs to be at least as long as the number of channels.
        :param weightlabel:  Dictionary name of the weights to be applied.
        :param weightuncert: Dictionary name of the weight uncertainties.
        """
        keys = list(self._components.keys())
        for key in keys: # delete all previous MC channels
            if 'MC' in key:
                del self._components[key]
        if colors is None:
            from itertools import cycle
            colors = cycle(plot_style.TangoColors.default_colors)
        if labels is None:
            labels = np.arange(len(channels))
        for i, content in enumerate(channels):
            self._add_component(dict_key='MC'+str(i),
                                content=content,
                                label=labels[i],
                                color=colors[i],
                                weightlabel=weightlabel,
                                weightuncert=weightuncert)
    
    def add_data_component(self,
                           content: pd.DataFrame,
                           dict_key: str="Data",
                           color: str="xkcd:black",
                           label: str="Data"):
        """
        Adds data component that will be plotted. Same as _add_components just with default values.
        :param content: pd.DataFrames that will be plotted as data.
        :param dict_key: Dictionary key of the specific content. Default is 'Data'.
        :param color: Color of the data component. Default is black.
        :param label: Label of the data component. Default is 'Data'.
        """
        self._add_component(dict_key=dict_key,
                            content=content,
                            color=color,
                            label=label)
                              
    def add_variable(self,
                     varname: Union[HistVariable, str],
                     vartitle: Union[str, None] = None,
                     varunit: Union[str, None] = None,
                     bins: int = 50,
                     neglecter: float = 0.01,
                     reset_varlist: bool = False):
        """
        Adds given variable to the plot.
        :param varname: The variable to plot. Can either be a complete HistVariable or a string from which
                        a HistVariable is created.
        :param vartitle: String that encodes the variable it they should be shown in latex.
        :param varunit: String that encodes the variable unit for the variable.
        :param bins: Number of bins that should be plotted.
        :param neglecter: ratio of data that should be neglected on both sides of the plot. Default is 0.01
        :param reset_varlist: If true, the variable list is resetted.
        """
        if reset_varlist:
            self._variables = []
        if isinstance(varname, str):
            if varname in self.ignorelist:
                return # don't add variable when it is in the ignorelist.
            if vartitle is None:
                vartitle = varname
            if len([*self._components.keys()]) == 0:
                raise RuntimeError("ERROR! The variable range can only be automatically deduced, if at least one component is given.")
            rangemin = np.nanmin(np.array([self._components[key][0][varname].quantile(0.00 + neglecter) for key in self._components.keys()]))
            rangemax = np.nanmax(np.array([self._components[key][0][varname].quantile(1.00 - neglecter) for key in self._components.keys()]))
            if rangemin == rangemax: # if both are the same, scipy.stats.binned_statistic leads to errors.
                rangemin -= 0.1
                rangemax += 0.1
            varobj = HistVariable(varname, n_bins=bins, scope=(rangemin, rangemax), var_name=vartitle, unit=varunit)
        elif isinstance(varname, HistVariable):
            varobj = varname
        self._variables.append(varobj)
    
    def add_list_of_variables(self,
                              varnames: Union[List[HistVariable], List[str]],
                              vartitles: Union[List[str], None] = None,
                              varunits: Union[List[str], None] = None,
                              bins: Union[List[int], int] = 50,
                              neglecter: Union[List[float], float] = 0.01,
                              reset_varlist: bool = False):
        """
        Adds given list of variables to the plotbook.
        :param varnames: The variables to plot. Can either be a complete list of HistVariable or a list
                         of strings from which HistVariables are created.
        :param vartitles: Optional list of strings which encode the variables as they should be shown in latex.
        :param varunits: List of strings which encode the variable units for the plot.
        :param bins: Number of bins that should be plotted. If a list is given, it needs to be the same size
                     as the variables. Default is 50 bins for each plot.
        :param neglecter: ratio of data that should be neglected on both sides of the plot. Default is 0.01
        :param reset_varlist: If true, the variable list is resetted.
        """
        if reset_varlist:
            self._variables = []
        if vartitles is None:
            vartitles = varnames
        if varunits is None:
            varunits = [None]*len(varnames)
        if not isinstance(bins, list):
            bins = [bins]*len(varnames)
        if not isinstance(neglecter, list):
            neglecter = [neglecter]*len(varnames)
        for i in range(len(varnames)):
            self.add_variable(varname=varnames[i],
                              vartitle=vartitles[i],
                              varunit=varunits[i],
                              bins=bins[i],
                              neglecter=neglecter[i],
                              reset_varlist=False)

    def reset_lower_plot_mode(self, new_mode:str):
        """
        Function to reset the lower plot mode to the new_mode.
        It automatically changes the default y-range for the lower plot in agreement with the new mode.
        :param new_mode: the new lower plot mode to be set.
        """
        if new_mode not in ['ratio', 'residuals']:
            raise ValueError("ERROR! Please select between 'ratio' and 'residuals' for the lower plot mode.")
        self.lower_plot_mode = new_mode
        if new_mode == 'ratio':
            self.lower_yaxis_range = (-0.3, 0.3) # it's possible to change that to 'False' if wanted
        elif new_mode == 'residuals':
            self.lower_yaxis_range = (-3., 3.)

    def create_plots(self, show_syst_errors: bool=True):
        """
        function to create all plots after everything else was specified.
        :param show_syst_errors: If true, the systematic errors are plotted as well. The weight uncertainties
                                 need to be applied to the hist components in this case.
        """
        size = 15/self.columns
        nplots = len(self._variables)
        # note that I don't use "sharex=True" here as otherwise all plots of the line will have the same range...
        for i, var in enumerate(self._variables):
            column = i % self.columns
            if column == 0:
                fig, ax = plt.subplots(2, self.columns, figsize=(self.columns*size, size), dpi=125, gridspec_kw={"height_ratios":[3.5, 1]})
            ax0 = ax[0][column]
            ax1 = ax[1][column]
            hp = DataMCHistogramPlot(var)
            for comp_key in self._components.keys():
                if 'MC' in comp_key:
                    # if channel is empty (after cuts), an error is binned_statistics again
                    if not len(self._components[comp_key][0]) == 0:
                        if isinstance(self._components[comp_key][3], str):
                            weights = self._components[comp_key][0][self._components[comp_key][3]]*self.luminosity[0]
                        else:
                            weights = np.full(len(self._components[comp_key][0]), self.luminosity[0])
                        weight_uncerts = None
                        if show_syst_errors:
                            if isinstance(self._components[comp_key][4], str):
                                weight_uncerts = np.sqrt(np.add(
                                    np.square(self._components[comp_key][0][self._components[comp_key][4]]*self.luminosity[0]),
                                    np.square(weights*self.luminosity[1])))
                            else:
                                raise ValueError("ERROR! To show the syst. uncertainties, the weight uncert. name needs to be specified in the components!")
                        hp.add_mc_component(label=self._components[comp_key][1], 
                                            data=self._components[comp_key][0][var.df_label].values,
                                            color=self._components[comp_key][2],
                                            weights=weights,
                                            weight_uncerts=weight_uncerts)
                elif 'Data' in comp_key:
                    hp.add_data_component(label=self._components[comp_key][1], 
                                          data=self._components[comp_key][0][var.df_label].values)
            hp.plot_on(ax0, ax1, style="stacked", ylabel="Events",
                       lower_plot_mode=self.lower_plot_mode, lower_yaxis_range=self.lower_yaxis_range,
                       include_systematics=show_syst_errors)
            if self.signal_index is not None:
                shp = SimpleHistogramPlot(var)
                comp_key = 'MC'+str(self.signal_index)
                if isinstance(self._components[comp_key][3], str):
                    weights = self._components[comp_key][0][self._components[comp_key][3]]*self.luminosity[0]*self.signal_enhancement
                else:
                    weights = np.full(len(self._components[comp_key][0]), self.luminosity[0]*self.signal_enhancement)
                shp.add_component(label=None,
                                  data=self._components[comp_key][0][var.df_label].values,
                                  color=self._components[comp_key][2],
                                  weights=weights)
                shp.plot_on(ax0)
                h, l = ax0.get_legend_handles_labels()
                l[0] += '[ $\\times$ ' + str(self.signal_enhancement) + ']'
                ax0.legend(h, l, frameon=False, loc='upper right', fontsize='x-small')
            # add additional information
            add_descriptions_to_plot(
                ax0,
                experiment='Belle II',
                luminosity=r"$\int \mathcal{L} \,dt="+str(round(self.luminosity[0], 1))+"\,\mathrm{fb}^{-1}$",
                additional_info=self.additional_info
            )
            # the command of shared axes is not necessarily needed as both ranges are specificly set anyway in the code.
            ax0.get_shared_x_axes().join(ax0, ax1)
            ax0.set_xticklabels([]) # otherwise, the labels are also shown between both plots...
            ax0.set_xlabel(None)
            # if it's the last plot but not the last column, hide all remaining plots:
            if nplots - i == 1 and column != self.columns-1:
                for k in range(column+1, self.columns):
                    ax[0][k].axis('off')
                    ax[1][k].axis('off')
                plt.show()
            if column == self.columns-1:
                plt.show()
