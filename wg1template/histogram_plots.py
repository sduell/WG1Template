"""
This module contains functions and classes for plotting various kinds of
histograms of given data.
"""
import itertools
from collections import defaultdict
from typing import Tuple, Union

# +
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from uncertainties import unumpy as unp

from scipy.linalg import block_diag
# -

import wg1template.plot_style as plot_style

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
        self._x_label = var_name + f' in {unit}' if unit else var_name
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
                 ls: str):
        """
        HistComponent constructor.
        :param label: Component label for the histogram.
        :param data: Data to be histogramed.
        :param weights: Weights for the events in data.
        :param histtype: Specifies the histtype of the component in the
        histogram.
        :param color: Color of the histogram component.
        :param ls: Linestyle of the histogram component.
        """
        self._label = label
        self._data = data
        self._weights = weights
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
        
    def _add_cov_mat_up_down(self, hup, hdown):
        """Helper function. Calculates a covariance matrix from
        given histogram up and down variations.
        """
        cov_mat = get_systematic_cov_mat(
            self._flat_bin_counts, hup.bin_counts.flatten(), hdown.bin_counts.flatten()
        )
        self.add_sys_cov(cov_mat)
    
    #adapted from the binfit package
    def add_gaussian_variation(self, data, var, bin_edges, nominal_weight, new_weight, Nstart=None, Nweights=None , total_weight=None):
        """Add a new covariance matrix from a given systematic variation
        of the underlying histogram to the template."""
        if Nweights==None:
            Nweights = len([col for col in data.columns if new_weight in col])
        if total_weight == None:
            nominal = np.histogram(data[var], bins = bin_edges, weights=data[nominal_weight])[0]
            bin_counts = np.array([np.histogram(data[var], bins = bin_edges, weights=data['{}_{}'.format(new_weight,i)])[0] for i in range(Nstart, Nweights+Nstart)])
        else:
            nominal = np.histogram(data[var], bins = bin_edges, weights=data[total_weight])[0]
            bin_counts = np.array([np.histogram(data[var], bins = bin_edges, weights=data['{}_{}'.format(new_weight,i)]*data[total_weight]/data[nominal_weight])[0] for i in range(Nstart,Nweights+Nstart)])
        cov_mat = np.matmul((bin_counts - nominal).T, (bin_counts - nominal))/Nweights
        self.add_sys_cov(cov_mat)    
    
    @property
    def get_cov_mat(self):
        cov=self.stat_cov
        for syscov in self.sys_covs:
            cov += syscov
        return cov
    
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

    def _get_y_label(self, normed: bool, bin_width: float, evts_or_cand="Events") -> str:
        """
        Creates the appropriate  y axis label for the histogram plot.

        :param normed: Whether the label is for a normalized histogram
        or not.
        :param bin_width: Width of each bin (equal binning assumed).
        :return: The y axis label,
        """

        if normed:
            return "Normalized in arb. units"
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
                yaxis_scale=1.3,
                normed: bool = False,
                ylabel="Events",
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
            ax.hist(x=component.data,
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

            y_label = self._get_y_label(normed=normed, bin_width=bin_width, evts_or_cand=ylabel)
            ax.set_ylabel(y_label, plot_style.ylabel_pos)

        if draw_legend:
            if legend_inside:
                ax.legend(frameon=False)
                ylims = ax.get_ylim()
                ax.set_ylim(ylims[0], yaxis_scale * ylims[1])
            else:
                ax.legend(frameon=False, bbox_to_anchor=(1, 1))

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

    def plot_on(self, ax: plt.axis, ylabel="Events", draw_legend=True, legend_inside=True, hide_labels: bool = False):
        bin_edges, bin_mids, bin_width = self._get_bin_edges()

        self._bin_edges = bin_edges
        self._bin_mids = bin_mids
        self._bin_width = bin_width

        ax.hist(x=[comp.data for comp in self._mc_components['stacked']],
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
            y_label = self._get_y_label(False, bin_width, ylabel)
            ax.set_ylabel(y_label, plot_style.ylabel_pos)
        if draw_legend:
            if legend_inside:
                ax.legend(frameon=False)
                ylims = ax.get_ylim()
                ax.set_ylim(ylims[0], 1.4 * ylims[1])

            else:
                ax.legend(frameon=False, bbox_to_anchor=(1, 1))

        return ax


# +
class DataMCHistogramPlot(HistogramPlot):
    def __init__(self,
                 variable: HistVariable):
        """
        HistogramPlot constructor.
        :param variable: A HistVariable describing the variable to be
        histogramed.
        """
        super().__init__(variable=variable)
        self._stat_cov=None
        self._global_covs=[]
        self._global_cov=None
        self._total_cov=None
        
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

        assert len(data) == len(weights)

        self._mc_components["MC"].append(
            HistComponent(
                label=label,
                data=data,
                weights=weights,
                histtype=None,
                color=color,
                ls=ls,
            )
        )

    def plot_on(self, ax1: plt.axis, ax2, style="stacked", ylabel="Events",
                sum_color=plot_style.KITColors.kit_purple,
                draw_legend: bool = True,
                legend_inside: bool = True,
                ):
        bin_edges, bin_mids, bin_width = self._get_bin_edges()

        self._bin_edges = bin_edges
        self._bin_mids = bin_mids
        self._bin_width = bin_width

        sum_w = np.sum(
            np.array([binned_statistic(comp.data, comp.weights, statistic="sum", bins=bin_edges)[0] for comp in
                      self._mc_components["MC"]]), axis=0)

        if not self._global_covs:
            sum_w2 = np.sum(
                np.array([binned_statistic(comp.data, comp.weights ** 2, statistic="sum", bins=bin_edges)[0] for comp in
                          self._mc_components["MC"]]), axis=0)
        else:
            mat = np.sum(np.array([comp.get_cov_mat for comp in self._mc_components["MC"]]), axis=0)
            #mat = np.diag(mat)
            sum_w2=np.zeros(len(mat))
            for elem in mat:
                sum_w2+=elem
        
 #       print("normal errors")
 #       print(np.array([binned_statistic(comp.data, comp.weights ** 2, statistic="sum", bins=bin_edges)[0] for comp in
 #                     self._mc_components["MC"]]))

        hdata, _ = np.histogram(self._data_component.data, bins=bin_edges)

        if style.lower() == "stacked":
            ax1.hist(x=[comp.data for comp in self._mc_components['MC']],
                     bins=bin_edges,
                     weights=[comp.weights for comp in self._mc_components['MC']],
                     stacked=True,
                     edgecolor="black",
                     lw=0.3,
                     color=[comp.color for comp in self._mc_components['MC']],
                     label=[comp.label for comp in self._mc_components['MC']],
                     histtype='stepfilled'
                     )

            ax1.bar(
                x=bin_mids,
                height=2 * np.sqrt(sum_w2),
                width=self.bin_width,
                bottom=sum_w - np.sqrt(sum_w2),
                color="black",
                hatch="///////",
                fill=False,
                lw=0,
                label="MC stat. unc."
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

        y_label = self._get_y_label(False, bin_width, evts_or_cand=ylabel)
        # ax1.legend(loc=0, bbox_to_anchor=(1,1))
        ax1.set_ylabel(y_label, plot_style.ylabel_pos, fontsize=8)

        if draw_legend:
            if legend_inside:
                ax1.legend(frameon=False, fontsize=5)
                ylims = ax1.get_ylim()
                ax1.set_ylim(ylims[0], 1.4 * ylims[1])
            else:
                ax1.legend(frameon=False, bbox_to_anchor=(1, 1), fontsize=5)

        #ax2.set_ylabel(r"$\frac{\mathrm{Data - MC}}{\mathrm{Data}}$")
        #ax2.set_xlabel(self._variable.x_label, plot_style.xlabel_pos)
        #ax2.set_ylim((-1, 1))

        #try:
        #    uhdata = unp.uarray(hdata, np.sqrt(hdata))
        #    uhmc = unp.uarray(sum_w, np.sqrt(sum_w2))
        #    ratio = (uhdata - uhmc) / uhdata

        #    ax2.axhline(y=0, color=plot_style.KITColors.dark_grey, alpha=0.8)
        #    ax2.errorbar(bin_mids, unp.nominal_values(ratio), yerr=unp.std_devs(ratio),
                         #ls="", marker=".", color=plot_style.KITColors.kit_black)
        #except ZeroDivisionError:
        #    ax2.axhline(y=0, color=plot_style.KITColors.dark_grey, alpha=0.8)

        #plt.subplots_adjust(hspace=0.08)
        
        ax2.set_ylabel(r"$\frac{N_{\mathrm{Data}} - N_{\mathrm{MC}}}{\sqrt{\sigma^{2}_{\mathrm{Data}}+\sigma^{2}_{\mathrm{MC}}}}$", fontsize=8) 
        ax2.set_xlabel(self._variable.x_label, plot_style.xlabel_pos, fontsize=8)
        
        try: 
            uhdata = unp.uarray(hdata, np.sqrt(hdata))
            uhmc = unp.uarray(sum_w, np.sqrt(sum_w2)) 
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = (hdata-sum_w)/np.sqrt(hdata + sum_w2)
                ratio[ratio == np.inf] = 0 
                ratio[ratio == -np.inf] = 0
                ratio = np.nan_to_num(ratio)
                
            #Error calculation  
            ratio_errors = np.full_like(sum_w, 1.)
        
            lowratio=-3.0
            highratio=3.0
            if(np.amin(ratio)<lowratio):
                lowratio=np.amin(ratio)-1
            if(np.amax(ratio)>highratio):
                highratio=np.amax(ratio)+1
            ax2.set_ylim(lowratio, highratio)
            ax2.axhline(y=0, color=plot_style.KITColors.dark_grey, alpha=0.8)
            ax2.errorbar(bin_mids, unp.nominal_values(ratio), yerr=ratio_errors, ls="", marker=".", color=plot_style.KITColors.kit_black)
        except ZeroDivisionError:
            ax2.axhline(y=0, color=plot_style.KITColors.dark_grey, alpha=0.8)

        plt.subplots_adjust(hspace=0.08)

    def _init_errors(self):
        """The statistical covariance matrix is initialized as diagonal
        matrix of the sum of weights squared per bin in the underlying
        histogram. For empty bins, the error is set to 1e-7. The errors
        are initialized to be 100% uncorrelated. The relative errors per
        bin are set to 1e-7 in case of empty bins.
        """
        stat_errors_sq = np.copy(self._flat_bin_errors_sq)
        stat_errors_sq[stat_errors_sq == 0] = 1e-14

        self._cov = np.diag(stat_errors_sq)

        self._cov_mats.append(np.copy(self._cov))

        self._corr = np.diag(np.ones(self._num_bins))
        self._inv_corr = np.diag(np.ones(self._num_bins))

        self._relative_errors = np.divide(
            np.sqrt(np.diag(self._cov)),
            self._flat_bin_counts,
            out=np.full(self._num_bins, 1e-7),
            where=self._flat_bin_counts != 0,
        )
    
    def add_variation(self, data, compname, weights_up, weights_down, total_weight=None, nominal_weight=None):
        """Add a new covariance matrix from a given systematic variation
        of the underlying histogram to the template."""
        if (total_weight==None or nominal_weight==None):
            hup = Hist1d(
                bins = self.bin_edges(), data=data[self._variable.df_label], weights=data[weights_up]
            )
            hdown = Hist1d(
                bins = self.bin_edges(), data=data[self._variable.df_label], weights=data[weights_down]
            )
        else:
            hup = Hist1d(
                bins = self.bin_edges(), data=data[self._variable.df_label],
                weights=data[weights_up]*data[total_weight]/data[nominal_weight]
            )
            hdown = Hist1d(
                bins = self.bin_edges(), data=data[self._variable.df_label],
                weights=data[weights_down]*data[total_weight]/data[nominal_weight]
            )
        
        for comp in self._mc_components["MC"]:
            if(compname==comp.label):
                comp.add_cov_mat_up_down(hup, hdown)
    
    def PrintGlobCov(self):
        print(type(self._global_cov))
        print(len(self._global_cov))
        
    def CalcTotCovMatrix(self):
        if len(self._global_covs) == 0:
            cov_mats = [comp.get_cov_mat for comp
                         in self._mc_components["MC"]] 
            #print(len(cov_mats[0]))
            local_cov=block_diag(*cov_mats)
            self._global_cov=np.matrix(local_cov)
            #total_corr=cov2corr(global_cov)
            #self._inv_corr = np.linalg.inv(total_corr)
        else:
            cov_mats = [comp.get_cov_mat for comp in self._mc_components["MC"]]
            #print(len(self._global_covs))
            global_cov=np.sum(np.array(self._global_covs),axis=0)
            local_cov= block_diag(*cov_mats)
            #print(len((global_cov[0])))
            #print(len(local_cov))
            global_cov=global_cov*(local_cov==0)
            self._global_cov=np.matrix(global_cov+local_cov)
        #print(len(self._total_cov))
            #total_corr=cov2corr(global_cov+local_cov)
            #self._inv_corr = np.linalg.inv(total_corr)

    def AddStatCovs(self):
        bin_edges, bin_mids, bin_width = self._get_bin_edges()
        for comp in self._mc_components["MC"]:
            stat_errors_sq = np.histogram(comp.data,bins=bin_edges,weights=comp.weights**2)[0]
            stat_errors_sq[stat_errors_sq == 0] = 1e-14
            comp.set_stat_cov(np.diag(stat_errors_sq))
            
            
    def AddGaussianVariations(self, inputdfdict, inkeys, namedict, weight_names, Nstart, Nvar):
        keys = inkeys
        #dfs = [inputdfdict[key] for key in keys]
        #titledict = dict((v,k) for k,v in namedict.items())
        nominal_weight=weight_names['nominal_weight']
        total_weight=weight_names['total_weight']
        new_weight=weight_names['new_weight']

        bin_edges, bin_mids, bin_width = self._get_bin_edges()
        
        n = {key : np.histogram(inputdfdict[key][self._variable.df_label],bins=bin_edges,weights=inputdfdict[key][total_weight])[0] for key in keys}
        #n_errs={key : np.sqrt(np.histogram(inputdfdict[key][self._variable.df_label],bins=bin_edges,weights=inputdfdict[key][total_weight]**2)[0]) for key in keys}
        
        #for key in keys:
        #    for comp in self._mc_components["MC"]:
        #        stat_errors_sq = np.copy(n_errs[key].flatten())
        #        stat_errors_sq[stat_errors_sq == 0] = 1e-14
        #        if(namedict[key]==comp.label):
        #            comp.set_stat_cov(np.diag(stat_errors_sq))
        #            #for i in range(Nstart, Nstart+Nvar): ##can add single template covs here later
        #            #comp.add_global_cov(np.diag(stat_errors_sq))
        
#        self.AddStatCovs()
        
        #if(self._stat_cov==None):
        #    self._stat_cov=np.diag(np.concatenate(list(n.values())))
            
        #for comp in self._mc_components["MC"]:
        #    print(comp.stat_cov)

        hnom = np.concatenate(list(n.values()))

        varMatrix=[]
        
        #save the single systematic covariance matrices
        for key in keys:
            for comp in self._mc_components["MC"]:
                if(namedict[key]==comp.label):
                    comp.add_gaussian_variation(inputdfdict[key], self._variable.df_label, bin_edges, nominal_weight, new_weight, Nstart, Nvar, total_weight)

        #create the global covariance matrix
        for i in range(Nstart, Nstart+Nvar):
            ntemp = [np.histogram(inputdfdict[key][self._variable.df_label],bins=bin_edges,weights=(inputdfdict[key]['{}_{}'.format(new_weight,i)]*inputdfdict[key][total_weight]/inputdfdict[key][nominal_weight]))[0] for key in keys]
            row = np.concatenate(ntemp)
            #print(len(row))
            varMatrix.append(row)
            
        varMatrix=np.array(varMatrix)
        covMatrix=np.matmul((varMatrix-hnom).T,(varMatrix-hnom))/Nvar
                
        #print(len(covMatrix))
        #print(len(covMatrix[0]))
        self._global_covs.append(covMatrix)
        #print(self._global_covs)


# -

def create_hist_ratio_figure():
    """Create a matplotlib.Figure for histogram ratio plots.

    :return: A maptlotlib.Figure instance and a matplotlib.axes instance.
    """
    return plt.subplots(2, 1, figsize=(5, 5), dpi=300, sharex=True, gridspec_kw={"height_ratios": [3.5, 1]})


def create_solo_figure(figsize=(5, 5), dpi=200):
    return plt.subplots(1, 1, figsize=figsize, dpi=dpi)


def add_descriptions_to_plot(ax: plt.axis,
                             experiment: Union[str, None] = None,
                             luminosity: Union[str, None] = None,
                             additional_info: Union[str, None] = None,
                             ):
    ax.set_title(experiment+'\n', loc="left", fontdict={'size': 10, 'style': 'normal', 'weight': 'bold'})
    ax.set_title(luminosity, loc="right")
    ax.annotate(
        additional_info, (0.02, 0.98), xytext=(4, -4), xycoords='axes fraction',
        textcoords='offset points',
        fontweight='bold', ha='left', va='top'
    )
