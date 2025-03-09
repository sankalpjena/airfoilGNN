import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib
from pathlib import Path
import seaborn as sns
sns.set_style("white") # sets background to white
# sns.set_theme(context='talk') # sets theme for plots

# Update Matplotlib parameters
params = {
    'legend.fontsize': 18,
    'figure.figsize': (6, 4),
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'text.usetex': True,
    'font.family': 'serif',
    'pgf.rcfonts': False,
    'axes.labelweight': 'bold',  # Make axes labels bold
    'xtick.major.width': 2,  # Make tick labels bold
    'ytick.major.width': 2,  # Make tick labels bold
    'figure.dpi': 300
}
pylab.rcParams.update(params)

# Ensure LaTeX is used for text rendering
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'text.usetex': True,
    'font.family': 'serif',
    'pgf.rcfonts': False,
})