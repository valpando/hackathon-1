import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

def read_light_curves(path_to_file):
    """Load light curves from a JSON file into a dictionary of DataFrames.

    Parameters
    ----------
    path_to_file : str or path-like
        Path to a JSON file containing a list of light-curve dictionaries.
        Each dictionary contains equal-length arrays: ``time`` (days),
        ``signal`` (magnitudes), and ``dsignal`` (signal uncertainties).

    Returns
    -------
    dict[int, pandas.DataFrame]
        Light curves indexed by their zero-based position in the file.
        Each DataFrame has one row per observation and preserves the
        original column names and values.
    """

    with open(path_to_file, encoding="utf-8") as fh:
        curves = json.load(fh)

    df_curves = {}
    for i, curve in enumerate(curves):
        df_curves[i] = pd.DataFrame(curve)

    return df_curves

def plot_light_curves(dict_of_lcs):
    """Plot light curves previously read from a JSON file.

    Parameters
    ----------
    dict_of_lcs: dict[int, pandas.DataFrame]
        Light curves indexed by their zero-based position in the file.
        Each DataFrame has one row per observation and preserves the
        original column names and values. This can be the output of the 
        'read_light_curves' function defined above.
        
    Returns
    -------
    A matplotlib plot.
    """
        
    fig, ax = plt.subplots(figsize=(12, 4))
    
    for i, df_curve in dict_of_lcs.items():
        #ax.plot(df_curve["time"], df_curve["signal"], ".-", ms=3, lw=0.6)
        ax.errorbar(df_curve["time"],
                    df_curve["signal"],
                    yerr=df_curve["dsignal"],
                    lw=0.6,
                    marker='o',
                    ms=2)
        ax.text(df_curve["time"][0]-10,
                df_curve["signal"][0],
                i,
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=22,
                color=plt.gca().lines[-1].get_color())
    
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    ax.set(xlabel="t [days]", ylabel="Mag")
    plt.tight_layout()
    plt.show()


def score(df_results, df_truth):
    """Return the root mean squared error (RMSE) in days.

    Compute the square root of the average squared error across all matched
    IDs and all three time-delay columns. Skip pairs where either value is
    missing ("NA", "NaN", or np.nan). Return NaN if no valid pairs remain.

    Parameters
    ----------
    df_results, df_truth : pandas.DataFrame
        Predicted and reference values, respectively. Each must contain
        ``ID``, ``01``, ``02``, and ``03``.
        Only IDs present in both frames are scored (inner join).

    Returns
    -------
    float
        Square root of the mean squared difference across all valid delay
        pairs, with equal weight per pair. Lower is better; zero is exact.
        Returns NaN if no valid pairs remain.

    Notes
    -----
    Numeric strings are accepted. Missing values (including the strings
    "NA" and "NaN", and np.nan) and other nonnumeric values are coerced
    to NaN. A pair is excluded if either value is missing.
    Displays the first five comparison rows with signed errors.
    """

    df_comparison = df_truth.merge(
        df_results, on="ID", suffixes=("_truth", "_result")
    )

    for col in ["01", "02", "03"]:
        truth = pd.to_numeric(df_comparison[f"{col}_truth"], errors="coerce")
        result = pd.to_numeric(df_comparison[f"{col}_result"], errors="coerce")
        df_comparison[f"{col}_error"] = result - truth

    display(df_comparison.head())

    # Pool all valid errors before taking the root mean square.
    errors = df_comparison[[f"{col}_error" for col in [
        "01", "02", "03"
    ]]].stack()
    rmse = float(np.sqrt(errors.pow(2).mean()))

    return rmse