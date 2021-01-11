# Plotting template for WG1 (SL & Missing Energy)

Provides an excellent set of classes for producing histograms in various forms. Originally hosted on [Markus Prim's GitHub](https://github.com/MarkusPrim/WG1Template).

## Installation

This can be installed as a package and be made available to Python using the following:

```
pip install --user git+ssh://git@stash.desy.de:7999/~pgrace/wg1template.git
```

The package can then by accessed by Python:

```python
>>> import wg1template
```

## Dependencies

This package relies on the following packages (all available from PyPI):
* 

## Examples

### Imports
The following examples use these imports:
```python
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

import wg1template.histogram_plots as wg1
import wg1template.point_plots as points
from wg1template.plot_style import TangoColors
from wg1template.plot_utilities import export
```

### Defining variables

The class `wg1template.histogram_plots.HistVariable` can be used to set the properties of a variable.
```python
dummy_var = wg1.HistVariable("DummyVariable",
                             n_bins=25,
                             scope=(-0,10),
                             var_name="DummyVariable",
                             unit="GeV")
```
### Simple histogram plot

![Simple histogram plot](fig/Simple.png).

```python
hp1 = wg1.SimpleHistogramPlot(dummy_var)
hp1.add_component("Something", data, color=TangoColors.scarlet_red)
hp2 = wg1.SimpleHistogramPlot(dummy_var)
hp2.add_component("Else", bkg, color=TangoColors.aluminium)
fig, ax = wg1.create_solo_figure()
hp1.plot_on(ax, ylabel="Events")
hp2.plot_on(ax)
wg1.add_descriptions_to_plot(
    ax,
    experiment='Belle II',
    luminosity=r"$\int \mathcal{L} \,dt=5\,\mathrm{fb}^{-1}$",
    additional_info='WG1 Preliminary Plot Style\nSimpleHistogramPlot'
)
plt.show()
export(fig, 'simple', 'examples')
plt.close()
```

### Stacked histogram plot

![Stacked histogram plot](fig/Stacked.png).

```python
hp = wg1.StackedHistogramPlot(dummy_var)
hp.add_component("Continum", cont, weights=cont.__weight__, color=TangoColors.slate,
                 comp_type='stacked')
hp.add_component("Background", bkg, weights=bkg.__weight__, color=TangoColors.sky_blue,
                 comp_type='stacked')
hp.add_component("Signal", sig, weights=sig.__weight__, color=TangoColors.orange, comp_type='stacked')
fig, ax = wg1.create_solo_figure()
hp.plot_on(ax, ylabel="Candidates")
wg1.add_descriptions_to_plot(
    ax,
    experiment='Belle II',
    luminosity=r"$\int \mathcal{L} \,dt=5\,\mathrm{fb}^{-1}$",
    additional_info='WG1 Preliminary Plot Style\nStackedHistogramPlot'
)
plt.show()
export(fig, 'stacked', 'examples')
plt.close()
```

### Data-MC histogram plot

![Data-MC histogram plot](fig/DataMC.png).

```python
hp = wg1.DataMCHistogramPlot(dummy_var)
hp.add_mc_component("Continum", cont, weights=cont.__weight__, color=TangoColors.slate)
hp.add_mc_component("Background", bkg, weights=bkg.__weight__, color=TangoColors.sky_blue)
hp.add_mc_component("Signal", sig, weights=sig.__weight__, color=TangoColors.orange)
hp.add_data_component("Data", data)
fig, ax = wg1.create_hist_ratio_figure()
hp.plot_on(ax[0], ax[1], style="stacked", ylabel="Candidates")
wg1.add_descriptions_to_plot(
    ax[0],
    experiment='Belle II',
    luminosity=r"$\int \mathcal{L} \,dt=5\,\mathrm{fb}^{-1}$",
    additional_info='WG1 Preliminary Plot Style\nDataMCHistogramPlot'
)
plt.show()
export(fig, 'data-mc', 'examples')
plt.close()
```

### Data-MC histogram plot, with pre-binned MC yields

This is to be used in the case where we want to plot data and MC distributions, but we only have the bin yields for MC (such as in the case of a template fit being performed by some other program).

![Data-MC histogram plot, where MC is pre-binned.](fig/Prebinned.png).

Bin yield uncertainties are optional, but for this example we'll set a uniform uncertainty.

```python
# Make some pre-binned data
def mockup_bin_yields(mean, stdev, nPoints, bins, scope):
    data = np.random.normal(mean, stdev, nPoints)

    bin_yields, _ = np.histogram(data, bins=bins, range=scope)
    return bin_yields


scope = (-4, 6)
n_bins = 40

cont_yields = mockup_bin_yields(0, 10, 3200, n_bins, scope)
bkg_yields = mockup_bin_yields(2, 1, 1600, n_bins, scope)
sig_yields = mockup_bin_yields(1, 0.4, 800, n_bins, scope)

data = np.concatenate([
    np.random.normal(0, 10, 3200),
    np.random.normal(2, 1, 1600),
    np.random.normal(1, 0.4, 800),
])

# Make up some uniform bin yield uncertainties
bin_uncertainties = 6*np.ones(n_bins)

# Variable must have the same binning as the histogram bin values
dummy_var = wg1.HistVariable("dummy_var",
                             n_bins=n_bins,
                             scope=scope,
                             var_name="Dummy Variable",
                             unit="GeV")
```

```python
hp = wg1.PrebinnedDataMCHistogramPlot(dummy_var)
hp.add_mc_component("Continum", bin_yields=cont_yields, color=TangoColors.slate)
hp.add_mc_component("Background",  bin_yields=bkg_yields, color=TangoColors.sky_blue)
hp.add_mc_component("Signal", bin_yields=sig_yields, color=TangoColors.orange)
hp.add_mc_uncertainty("Uniform unc.", bin_uncertainties)  # optional
hp.add_data_component("Data", data)

fig, ax = wg1.create_hist_ratio_figure()
hp.plot_on(ax[0], ax[1], style='stacked', ylabel="Candidates")
wg1.add_descriptions_to_plot(
    ax[0],
    experiment="Belle II",
    luminosity=r"$\int \mathcal{L} \,dt=5\,\mathrm{fb}^{-1}$",
    additional_info="WG1 Preliminary Plot Style\nPrebinned MC"
)

plt.show()
export(fig, "Prebinned", "examples")
plt.close()
```

In this example, if we wanted to set `sqrt(N)` bin yield errors, we would use
```python
bin_uncertainties = np.sqrt(cont_yields + bgk_yields + sig_yields)
hp.add_mc_uncertainty("MC stat. unc.", bin_uncertainties)
```

### Combo plot

![Combo 1 plot](fig/Combo.png).

```python
hp1 = wg1.StackedHistogramPlot(dummy_var)
hp1.add_component("Continum", cont, weights=cont.__weight__, color=TangoColors.slate,
                  comp_type='stacked')
hp1.add_component("Background", bkg, weights=bkg.__weight__, color=TangoColors.sky_blue,
                  comp_type='stacked')
hp1.add_component("Signal", sig, weights=sig.__weight__, color=TangoColors.orange,
                  comp_type='stacked')

hp2 = wg1.SimpleHistogramPlot(dummy_var)
hp2.add_component("Signal Shape x0.5", sig, weights=sig.__weight__ * 0.5,
                  color=TangoColors.scarlet_red, ls='-.')

fig, ax = wg1.create_solo_figure()
hp1.plot_on(ax, ylabel="Candidates")
hp2.plot_on(ax, hide_labels=True)  # Hide labels to prevent overrides)
wg1.add_descriptions_to_plot(
    ax,
    experiment='Belle II',
    luminosity=r"$\int \mathcal{L} \,dt=5\,\mathrm{fb}^{-1}$",
    additional_info='WG1 Preliminary Plot Style\nStackedHistogramPlot\n+SimpleHistogramPlot'
)
plt.show()
export(fig, 'combo', 'examples')
plt.close()
```

### Combo plot 2

![Combo 2 plot](fig/Combo%202.png).

```python
hp1 = wg1.DataMCHistogramPlot(dummy_var)
hp1.add_mc_component("Continum", cont, weights=cont.__weight__, color=TangoColors.slate)
hp1.add_mc_component("Background", bkg, weights=bkg.__weight__, color=TangoColors.sky_blue)
hp1.add_mc_component("Signal", sig, weights=sig.__weight__, color=TangoColors.orange)
hp1.add_data_component("Data", data)

hp2 = wg1.SimpleHistogramPlot(dummy_var)
hp2.add_component("Signal Shape x0.5", sig, weights=sig.__weight__ * 0.5,
                  color=TangoColors.scarlet_red, ls='dotted')

fig, ax = wg1.create_hist_ratio_figure()
hp1.plot_on(ax[0], ax[1], style='stacked', ylabel="Candidates")
hp2.plot_on(ax[0], hide_labels=True)  # Hide labels to prevent overrides
wg1.add_descriptions_to_plot(
    ax[0],
    experiment='Belle II',
    luminosity=r"$\int \mathcal{L} \,dt=5\,\mathrm{fb}^{-1}$",
    additional_info='WG1 Preliminary Plot Style\nDataMCHistogramPlot\n+SimpleHistogramPlot'
)
plt.show()
export(fig, 'combo2', 'examples')
plt.close()
```

### Combo plot 3 (with function plotted)

![Combo 3 plot](fig/Combo%203.png).

```python
hp = wg1.DataMCHistogramPlot(dummy_var)
hp.add_mc_component("Continum", cont, weights=cont.__weight__, color=TangoColors.slate)
hp.add_mc_component("Background", bkg, weights=bkg.__weight__, color=TangoColors.sky_blue)
hp.add_mc_component("Signal", sig, weights=sig.__weight__, color=TangoColors.orange)
hp.add_data_component("Data", data)
fig, ax = wg1.create_hist_ratio_figure()
hp.plot_on(ax[0], ax[1], style="stacked", ylabel="Candidates")
wg1.add_descriptions_to_plot(
    ax[0],
    experiment='Belle II',
    luminosity=r"$\int \mathcal{L} \,dt=5\,\mathrm{fb}^{-1}$",
    additional_info='WG1 Preliminary Plot Style\nDataMCHistogramPlot\n+SomeFunction'
)

# Let's add some functions
ax[0].plot(
    np.linspace(*dummy_var.scope),
    500 * scipy.stats.norm(2).pdf(np.linspace(*dummy_var.scope)),
    label="Some function", color=TangoColors.chameleon,
)
ax[0].legend(frameon=False)

plt.show()
export(fig, 'combo3', 'examples')
plt.close()
```

### Data plot

![Data plot](fig/Data.png).

```python
x = np.linspace(0.5, 10.5, num=10)
y = np.array([np.random.normal(a, 1) for a in x])
x_err = 0.5 * np.ones(10)
y_err = np.ones(10)

variable = points.DataVariable(r"x-variable", r"x-units", r"y-variable", "y-units")
measured = DataPoints(
    x_values=x,
    y_values=y,
    x_errors=x_err,
    y_errors=y_err,
)

x = np.linspace(0.5, 10.5, num=10)
y = np.array([np.random.normal(a, 1) for a in x])
x_err = 0.5 * np.ones(10)
y_err = np.ones(10)*0.5
theory = DataPoints(
    x_values=x,
    y_values=y,
    x_errors=x_err,
    y_errors=y_err,
)

dp = DataPointsPlot(variable)
dp.add_component("Data Label", measured, style='point')
dp.add_component("Theory Label", theory, style='box', color=TangoColors.scarlet_red)

fig, ax = wg1.create_solo_figure(figsize=(5, 5))
dp.plot_on(ax)
wg1.add_descriptions_to_plot(
    ax,
    experiment='Can be misused',
    luminosity='This too',
    additional_info=r'Some process'
)
plt.show()
export(fig, 'data', 'examples')
plt.close()
```

## Configuration

### Align axis labels on the axes ends

To get axis labels at the axes ends, as was the default in previous versions of the WG1 template, you just
have to change some global variables that the WG1 template uses:

```python
wg1template.plot_style.xlabel_pos = {"x": 1, "ha": "right"}
wg1template.plot_style.ylabel_pos = {"x": 1, "ha": "right"}
```

### Enable/disable errorbar caps and top-right ticks

The recommendations of the [Belle2Style](https://stash.desy.de/projects/B2D/repos/belle2style) are not to have
top/right axis ticks or errorbar caps, but some users might still prefer them. They can be enabled by

```python
from wg1template plot_style
plot_style.set_matplotlibrc_params(errorbar_caps=True, top_right_ticks=True)
```

## Authors

Original authors:
* **Peter Lewis**
* **Max Welsch** 
* **Markus Prim**

Fork:
* **Phil Grace**
