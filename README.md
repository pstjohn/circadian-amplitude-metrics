These are the python source files used to generate the figures in the manuscript *P. C. St. John, S. R. Taylor, J. H. Abel, and F. J. Doyle, “Amplitude Metrics for Cellular Circadian Bioluminescence Reporters,” Biophys. J., vol. 107, no. 11, pp. 2712–2722, Dec. 2014.*
These files are provided in their raw state without
significant documentation in the hopes that they may prove useful for an
interested reader.


PRE-REQUISITES
==============

SOFTWARE   | VERSION | WEBSITE
-----------|---------|--------
python     | 2.7.3   | python.org
numpy      | 1.6.1   | numpy.org
scipy      | 0.13.3  | scipy.org
matplotlib | 1.3.1   | matplotlib.org
casadi     | 1.5.0   | casadi.org
svg.path   | 1.1     | https://pypi.python.org/pypi/svg.path


FILE DESCRIPTIONS
=================
```
fig1.py                 : python code to generate figure 1
fig2.py                 : python code to generate figure 2
fig3.py                 : python code to generate figure 3
fig4.py                 : python code to generate figure 4
fig5_degmodel_arcs.dat  : pre-calculated differential ARCs for the model from
                          Hirota et al, 2012
fig5_fit.p              : pre-calculated exponential sinusoid fit to the data
                          of Ukai et al, 2007
fig5.py                 : python code to generate figure 5
figS1.py                : python code to generate figure S1
figS2.py                : python code to generate figure S2
figS2_stochastic_data.p : pre-calculated stochastic model trajectories for the
                          15^2 cells shown in supplemental movies

data/
__init__.py       : dummy file to allow package loading
MelanopsinData.py : File to process data from Ukai et al SVG
mel_n.txt         : svg path data for figure of control trajectory
mel_p.txt         : svg path data for figure of mel-positive trajectory

tools/
__init__.py        : n/a
Amplitude.py       : main file to calculate phase pdfs and single-cell ARCS
Bioluminescence.py : utility functions to detrend and fit sinusoidal data
Odesol.py          : class to calculate features of limit-cycle models
PlotOptions.py     : various matplotlib options to control output plots
Utilities.py       : Interpolation and plotting utility functions

tools/Models/ 
__init__.py : n/a
degmodelFinal.py           : Model of PER/CRY feedback from Hirota et al, 2012
simplified_tysonmodel.py   : Simple 2-state model of mRNA-protein oscillator
simplified_tysonmodel_2.py : Modified version of previous model with exponent=2
```

