import pandas as pd
import numpy as np
from utilfuncs import read_light_curves
from pathlib import Path
import pycs3.gen.splml
import pycs3.gen.lc_func
import pycs3.spl.topopt
import pycs3.regdiff.multiopt
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings("ignore")


DATA_DIR = Path.cwd().parent / "data"
TEST_DATA_DIR = DATA_DIR / "test"

def get_labels(filename="test.csv"):
    
    df = pd.read_csv(DATA_DIR / filename)
    labels = df["id"].values

    return labels


#--------------------------------------------------------------------------------------------
# Cross-Correlation as naive delta_t guess
#--------------------------------------------------------------------------------------------


def cross_corr(lc1, lc2):
    diff1 = lc1 - np.mean(lc1)
    diff2 = lc2 - np.mean(lc2)
    norm1 = np.sqrt(np.sum(diff1**2))
    norm2 = np.sqrt(np.sum(diff2**2))
    return np.sum(diff1*diff2)/norm1/norm2



def fit_dt_cc(t1, lc1, t2, lc2, tau_list):
    cc_list = []
    for tau in tau_list:
        lc2_shifted = np.interp(t1, t2 - tau, lc2, left=np.nan, right=np.nan)
        mask = ~np.isnan(lc2_shifted)
        if mask.sum() < 5:
            cc_list.append(-np.inf)
            continue
        cc_list.append(cross_corr(lc1[mask], lc2_shifted[mask]))
    best_tau = tau_list[np.argmax(cc_list)]
    best_cc = np.max(cc_list)
    return best_tau, best_cc



def compute_tau_guess(label):

    data = read_light_curves(TEST_DATA_DIR / f"{label}.json")
    for key in data:
        data[key] = data[key].drop_duplicates(subset=["time"])

    lc0 = data[0].signal.values
    lc1 = data[1].signal.values
    t0 = data[0].time.values
    t1 = data[1].time.values
    tau_list = np.arange(-60, 60, 0.1)

    # print("\n", label)
    tau_01, cc_01 = fit_dt_cc(t0, lc0, t1, lc1, tau_list)
    # print(tau_01, cc_01)

    if len(data) > 2:
        t2 = data[2].time.values
        lc2 = data[2].signal.values
        tau_02, cc_02 = fit_dt_cc(t0, lc0, t2, lc2, tau_list)
        # print(tau_02, cc_02)

        t3 = data[3].time.values
        lc3 = data[3].signal.values
        tau_03, cc_03 = fit_dt_cc(t0, lc0, t3, lc3, tau_list)
        # print(tau_03, cc_03)
    else:
        tau_02 = np.nan
        tau_03 = np.nan
        cc_02 = np.nan
        cc_03 = np.nan

    tau_guess = np.array([tau_01, tau_02, tau_03])
    cc = np.array([cc_01, cc_02, cc_03])

    return tau_guess, cc



#--------------------------------------------------------------------------------------------
# Python Curve Shifting functions
#--------------------------------------------------------------------------------------------


def gen_pycs3_lc(label):
    
    data = read_light_curves(TEST_DATA_DIR / f"{label}.json")
    for key in data:
        data[key] = data[key].drop_duplicates(subset=["time"])

    lcs = []
    colors = ["blue", "orange", "green", "red"]

    for key, df in data.items():
        jds = df["time"].to_numpy()
        mags = df["signal"].to_numpy()
        magerrs = df["dsignal"].to_numpy()

        l = pycs3.gen.lc_func.factory(
            jds, mags, magerrs,
            telescopename="",   # set to whatever fits your data
            object=str(key),           # e.g. "A", "B", "C", "D"
        )
        l.plotcolour = colors[key]           # optional, set/vary per curve
        lcs.append(l)

    return lcs



def spl(lcs, guess):
    lcs[1].shifttime(-guess[0])
    pycs3.gen.splml.addtolc(lcs[1], knotstep=300)
    if len(lcs) > 2:
        lcs[2].shifttime(-guess[1])
        lcs[3].shifttime(-guess[2])
        pycs3.gen.splml.addtolc(lcs[2], knotstep=300)
        pycs3.gen.splml.addtolc(lcs[3], knotstep=300)
    spline = pycs3.spl.topopt.opt_rough(lcs, nit=5, knotstep=150)
    for l in lcs:
        l.resetml()
    spline = pycs3.spl.topopt.opt_rough(lcs, nit=5, knotstep=40)
    spline = pycs3.spl.topopt.opt_fine(lcs, nit=20, knotstep=30)
    return spline



def regdiff(lcs, guess):
    lcs[1].shifttime(-guess[0])
    if len(lcs) > 2:
        lcs[2].shifttime(-guess[1])
        lcs[3].shifttime(-guess[2])
    return pycs3.regdiff.multiopt.opt_ts(lcs, pd=2, verbose=False, method="weights")



def predict_delays(lcs_opt):
    if len(lcs_opt) == 2:
        delay_1v0 = lcs_opt[0].timeshift - lcs_opt[1].timeshift
        return np.array([delay_1v0, np.nan, np.nan])
    elif len(lcs_opt) == 4:
        delay_1v0 = lcs_opt[0].timeshift - lcs_opt[1].timeshift
        delay_2v0 = lcs_opt[0].timeshift - lcs_opt[2].timeshift
        delay_3v0 = lcs_opt[0].timeshift - lcs_opt[3].timeshift
        return np.array([delay_1v0, delay_2v0, delay_3v0])
    else:
        print("Lenght of light curve object has to be either 2 or 4")
        raise ValueError()


def build_delay_lookup(filename="test.csv"):
    return pd.read_csv(DATA_DIR / filename).set_index("id")


def get_true_delays(df_lookup, label):
    row = df_lookup.loc[label]
    return row[["01", "02", "03"]].to_numpy(dtype=float)



def process_label(label, df_lookup):
    lcs = gen_pycs3_lc(label)

    guess, _ = compute_tau_guess(label)

    lcs_spl = [l.copy() for l in lcs]
    spl(lcs_spl, guess)
    dt_spline = predict_delays(lcs_spl)

    
    lcs_regdiff = [l.copy() for l in lcs]
    regdiff(lcs_regdiff, guess)
    dt_regdiff = predict_delays(lcs_regdiff)

    dt_cc = guess

    return get_true_delays(df_lookup, label), dt_cc, dt_spline, dt_regdiff



#--------------------------------------------------------------------------------------------
# Main
#--------------------------------------------------------------------------------------------



def main():

    labels = get_labels(filename="test.csv")
    df_lookup = build_delay_lookup(filename="test.csv")

    results = []
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_label, l, df_lookup): l for l in labels}
        for i, fut in enumerate(as_completed(futures)):
            print(f"Processed {futures[fut]} ({i+1}/{len(futures)})")
            results.append(fut.result())

    dts_true, dts_cc, dts_spline, dts_regdiff = map(np.array, zip(*results))

    df_cc = pd.DataFrame(np.column_stack((labels, dts_cc)), columns=["id", "01", "02", "03"])
    df_spline = pd.DataFrame(np.column_stack((labels, dts_spline)), columns=["id", "01", "02", "03"])
    df_regdiff = pd.DataFrame(np.column_stack((labels, dts_regdiff)), columns=["id", "01", "02", "03"])

    cols = ["01", "02", "03"]
    df_cc[cols] = df_cc[cols].where(df_cc[cols].abs() <= 60, np.nan)
    df_spline[cols] = df_spline[cols].where(df_spline[cols].abs() <= 60, np.nan)
    df_regdiff[cols] = df_regdiff[cols].where(df_regdiff[cols].abs() <= 60, np.nan)

    df_cc.to_csv("./solution_cc.csv", index=False)
    df_spline.to_csv("./solution_spline.csv", index=False)
    df_regdiff.to_csv("./solution_regdiff.csv", index=False)


if __name__ == "__main__":
    main()