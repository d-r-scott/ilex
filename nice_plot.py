#!/usr/bin/env python3

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cmasher as cmr
import ilex as il
from joblib import Memory
from mpl_toolkits.axes_grid1 import make_axes_locatable

memory = Memory("cache", verbose=0)

def _main():
    parser = ArgumentParser()

    # args: frb, favg, tavg, DM, crop_dur, RM, zap_chan, w_us, xlim, PA_lim, tns_name
    parser.add_argument("frb", help="FRB name")
    parser.add_argument("--favg", help="Frequency averaging factor", default=4, type=int)
    parser.add_argument("--tavg", help="Time averaging factor", default=4, type=int)
    parser.add_argument("--DM", help="DM", default=None, type=float)
    parser.add_argument("--crop", help="Crop duration", default=200, type=int)
    parser.add_argument("--RM", help="RM", default=None, type=float)
    parser.add_argument("--zap_chan", help="Channels to flag", default=None, nargs="+", type=int)
    parser.add_argument("--w_us", help="Width of boxcar", default=None, type=float)
    parser.add_argument("--xlim", help="xlim", default=(None, None), nargs=2, type=float)
    parser.add_argument("--PA_lim", help="PA limits", default=(None, None), nargs=2, type=float)
    parser.add_argument("--tns_name", help="TNS name", default=None, type=str)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 5))
    pax, sax, dax, lines = do_ax(ax, cmr.arctic_r, *vars(args).values())
    ax.set_xlabel("Time (ms)")
    pax.set_ylabel(r"$\psi$ (deg)")
    sax.set_ylabel("Intensity (S/N)")
    dax.set_ylabel("Frequency (GHz)")
    fig.legend(
            lines, 
            ["I", "L", "V"], 
            loc="center right",
            ncol=1, 
            bbox_to_anchor=(1, 0.5)
        )
    plt.savefig(f"{args.frb}.png", dpi=300, bbox_inches="tight")
    plt.show()


@memory.cache
def load_FRB(frbname, favg, tavg, DM, crop_dur, RM, flag_chans=None):
    # Load an FRB's data with Ilex and cache it for faster re-running
    print(f"{frbname} not cached!")
    
    frb = il.FRB(frbname, favg=favg, tavg=tavg, crop_dur=crop_dur, zap_above_crossing_freq=True, verbose=True)
    if frbname == "220725":
        frb.peak = int(2217 * 1000/tavg)
        frb.apply_crop(force_peak=2217)
        frb.set_crossing_freq()
        frb.zap_above_freq(frb.crossing_freq)

    if DM is not None:
        # frb.dedisperse(str(float(frb.DM)+DM))
        frb.dedisperse(str(DM), normalise=False)
    
    if flag_chans is not None:
        print(f"flagging channels {flag_chans}")
        frb.flag_dynspec(flag_chans)

    frb.set_dt_df(tavg, favg)

    frb.de_RM(RM)
    frb.calc_PA()
    frb.PA[~np.isnan(frb.PA)] = np.unwrap(frb.PA[~np.isnan(frb.PA)], discont=np.pi)

    PA = frb.PA.copy()
    PA_err = frb.PA_err.copy()

    L = np.sqrt((frb.Q/np.nanstd(frb.Q[frb.off_burst]))**2 + (frb.U/np.nanstd(frb.U[frb.off_burst]))**2)
    err = lambda S: np.nanstd(S[frb.off_burst])
    profs = [frb.I/err(frb.I), L - np.nanmean(L[frb.off_burst]), frb.V/err(frb.V)]

    if DM is not None:
        frb.dedisperse(str(DM), normalise=True)

    return (
        frb.t/1e3, 
        [PA, PA_err], 
        profs,
        frb.I_ds, 
        (frb.f0 - frb.bw/2)/1e3, 
        (frb.f0 + frb.bw/2)/1e3
    )


def plot_PA(ax, t, PA, PA_err):
    ax.errorbar(
        t, 
        PA,
        yerr=PA_err,
        fmt=".",
        ms=1,
        lw=0.5,
        capsize=1,
        capthick=0.5,
        elinewidth=0.5,
    )

def plot_time_series(ax, t, I, z, label, c):
        return ax.step(
                t, 
                I,
                where="mid",
                lw=1,
                zorder=z,
                label=label,
                c=c
            )
    
def plot_dynspec(ax, t, I_ds, fmin, fmax, cmap):
    extent = [
        t.min(),
        t.max(),
        fmin,
        fmax,
    ]

    # top channel is often broken
    I_ds[:,0] = np.nan

    ax.imshow(
        I_ds.T,
        aspect="auto", 
        interpolation="none",
        cmap=cmap,
        extent=extent,
    )

def do_ax(ax, cmap, frbname, favg, tavg, DM, crop, RM, zap_chan, w_us, xlim, PA_lim, tns_name):
    frbname = frbname.split(" ")[0]
    t, PA, ILV, I_ds, fmin, fmax = load_FRB(frbname, favg, tavg, DM, crop, RM, flag_chans=zap_chan)
    I = ILV[0]
    n_roll = np.argmax(I) - len(I)//2
    ILV = [np.roll(S, -n_roll) for S in ILV]
    I_ds = np.roll(I_ds, -n_roll, axis=0)
    PA = [np.roll(S, -n_roll) for S in PA]
    

    div = make_axes_locatable(ax)
    sax = div.append_axes("top", size="75%", pad=0, sharex=ax)
    pax = div.append_axes("top", size="30%", pad=0, sharex=ax)
    dax = ax

    if w_us is not None:
        I_temp = ILV[0].copy()
        if not isinstance(w_us, list):
            w_us = [w_us]
        for w in w_us:
            I_conv = np.convolve(I_temp, np.ones(int(w//tavg)), mode="same")
            conv_peak = t[np.argmax(I_conv)]
            w0 = conv_peak - w/2e3
            w1 = conv_peak + w/2e3
            sax.axvspan(w0, w1, color="lightblue", zorder=-10, alpha=0.5)
            t0 = np.argmin(np.abs(t - w0))
            t1 = np.argmin(np.abs(t - w1))
            I_temp[t0:t1] = 0

    if xlim is None:
        xlim = (w0 - 0.5*w_us[0]/1e3, w1 + w_us[0]/1e3)
        
    off_burst = (t < xlim[0]) | (t > xlim[1])
    PA_mask = ~off_burst & (ILV[0] > 1) & (ILV[1] > 1)
    PA[0][~PA_mask] = np.nan
    PA[1][~PA_mask] = np.nan

    # put PA in sensible range
    PA[0] += 180
    PA[0] %= 180
    PA[0] -= 90


    plot_PA(pax, t, PA[0], PA[1])
    lines = [plot_time_series(sax, t[~off_burst], S[~off_burst], -z, "ILV"[z], "krb"[z])[0] for z, S in enumerate(ILV)]
    plot_dynspec(dax, t, I_ds, fmin, fmax, cmap)


    pax.set_title(fr"{tns_name} [{tavg} $\mu$s]")
    pax.set_ylim(PA_lim)
    [label.set_visible(False) for label in pax.get_xticklabels()]

    [label.set_visible(False) for label in sax.get_xticklabels()]


    im = dax.images[0]
    clim = (
        np.nanmin(I_ds[(t > xlim[0]) & (t < xlim[1])]),
        np.nanmax(I_ds[(t > xlim[0]) & (t < xlim[1])])
    )
    im.set_clim(clim)

    dax.set_xlim(xlim)
    dax.set_facecolor("lightgrey")

    return pax, sax, dax, lines


if __name__ == "__main__":
    _main()