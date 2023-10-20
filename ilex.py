#!/usr/bin/env python3

"""Utilities for viewing and manipulating CELEBI output data
"""

import os
import sys
from glob import glob
from pathlib import Path

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
from joblib import Memory
from mpl_toolkits.axes_grid1 import make_axes_locatable
from paramiko import SSHClient
from scipy.fft import fft, ifft, dct, idct
from scp import SCPClient
from tqdm import tqdm
import sysrsync
from RMtools_1D.do_RMsynth_1D import run_rmsynth
from RMtools_1D.do_RMclean_1D import run_rmclean
from matplotlib import colors
import ScintScatMethods as scint
import astropy.units as un
from rmnest.fit_RM import RMNest
from scipy.optimize import curve_fit, minimize


data_base = "/home/drscott/data/CELEBI/"
memory = Memory(f"{data_base}cache", verbose=0)
try:
    ozstar_user = os.environ["OZSTAR_USER"]
except KeyError:
    print("WARNING: OZSTAR_USER environment variable not set.\n"
          "You will not be able to download data from ozstar.")
    ozstar_user = None

dft_cmap = cmr.arctic_r

class FRB(object):
    def __init__(
            self, 
            name: str, 
            verbose: bool = False, 
            DM: str = None, 
            bw: float = 336.0,
            tavg: int = 1, 
            favg: int = 1,
            nchan: int = 336,
            memmap: bool = True,
            crop: bool = True,
            crop_dur: float = 10.0,
            basic: bool = False,
            download_all_DMs: bool = False,
            get_crops: bool = False,
            zap_above_crossing_freq: bool = True,
        ):
        self.name = name
        self.verbose = verbose
        self.memmap = memmap
        self.crop = crop
        self.crop_dur = crop_dur
        self.data_dir = f"{data_base}{name}/"
        self.htr_dir = f"{self.data_dir}htr/"
        self.config = None
        self.basic = basic
        self.download_all_DMs = download_all_DMs
        self.get_crops = get_crops
        self.zap_above_crossing_freq = zap_above_crossing_freq

        self.get_config()

        if DM is None:
            self.DM = self.config["dm_frb"]
        else:
            self.DM = DM            # saved as string because it must 
                                    # correspond to files
        self.f0 = float(self.config["centre_freq_frb"]) # MHz
        self.bw = bw            # MHz
        self.full_dt = 1/bw     # μs
        try:
            self.nant = int(self.config["nants"])
        except:
            self.nant = int(self.config["nants_frb"])

        self.nchan = nchan
        self.ds_base_df = bw/nchan              # MHz
        self.ds_base_dt = 1/self.ds_base_df     # μs

        self.tavg = tavg
        self.favg = favg
        self.peak = None
        self.crop_time = None
        self.t = None
        self.off_burst = None
        self.crossing_freq = None
        self.flagged_chans = None

        self.dt = None
        self.df = None

        self.fig = None
        self.axes = None

        self.g2_fig = None
        self.g2_axes = None

        self.PA = None
        self.PA_err = None

        # data attributes
        for p in "XYIQUV":
            setattr(self, p, None)              # time series
            setattr(self, f"{p}_full", None)    # time series (full res)
            setattr(self, f"{p}_all", None)     # time series (all data)
            setattr(self, f"{p}_ds", None)      # dynamic spectrum
            setattr(self, f"{p}_ds_full", None) # dynamic spectrum (full res)
            setattr(self, f"{p}_ds_all", None)  # dynamic spectrum (all data)

        self.L = None
        self.L_ds = None

        self.get_data()


        try:
            self.load_data()
        except Exception as e:
            print(e)
            print(f"Could not load data for {self.name}")
            print("Trying download")
            self.download_data()
            self.load_data()

        if all (
            p is None for p in [
                self.X, self.Y, self.I, self.Q, self.U, self.V,
                self.X_ds, self.Y_ds, self.I_ds, self.Q_ds, self.U_ds, 
                self.V_ds
            ]):
            self.download_data()
            self.load_data()

        if any (
            p is None for p in [
                self.X, self.Y, self.I, self.Q, self.U, self.V,
                self.X_ds, self.Y_ds, self.I_ds, self.Q_ds, self.U_ds, 
                self.V_ds
            ]):
            print(f"Some data could not be loaded for {self.name}")
            print("Try running FRB.download_data() and FRB.load_data()")


        if self.crop:
            self.apply_crop()

        # if self.favg > 1 or self.tavg > 1:
        self.set_dt_df(self.tavg, self.favg)

        if self.nchan != 336:
            self.set_dynspec_nchan(self.nchan)

        if self.zap_above_crossing_freq:
            self.set_crossing_freq()
            self.zap_above_freq(self.crossing_freq)
            # do this again to remove noise-only data from time series
            self.set_dt_df(self.tavg, self.favg)
        else:
            self.crossing_freq = np.inf



        if self.verbose:
            print(f"Loaded data for {self.name}")
            self.data_status()

    def get_config(self):
        # Check if we have the config file, if not then try to download it
        if not os.path.isdir(self.data_dir):
            if self.verbose:
                print(f"Creating data dir for {self.name}")
            os.mkdir(self.data_dir)
        if not os.path.isfile(f"{self.data_dir}{self.name}.config"):
            if self.verbose:
                print(f"Could not find config file for {self.name}")
                print("Trying download")
            self.download_config()
        
        self.config = self.parse_config(f"{self.data_dir}{self.name}.config")

    def download_config(self):
        # Download config file from ozstar
        print(f"Downloading config file for {self.name} from ozstar")
        with SSHClient() as ssh:
            ssh.load_system_host_keys()
            ssh.connect('ozstar.swin.edu.au', username=ozstar_user)

            with SCPClient(ssh.get_transport()) as scp:
                get_fn = f"/fred/oz002/askap/craft/craco/processing/configs/{self.name}.config"
                to_fn = self.data_dir
                if self.verbose:
                    print(f"Downloading {get_fn} to {to_fn}")

                scp.get(
                    get_fn,
                    to_fn,
                )

    def parse_config(self, config_fname):
        # parse Nextflow config file
        config = {}
        with open(config_fname, "r") as f:
            for line in f.readlines():
                if "=" not in line:
                    continue
                key, val = line.split("=")
                key = key.split(".")[-1]    # remove prefix
                val = val.split("//")[0]    # remove comments
                config[key.strip()] = val.strip()
        return config

    # Data management
    def get_data(self):
        # Check if we have the data, if not then try to download it
        if not os.path.isdir(self.data_dir):
            if self.verbose:
                print(f"Creating data dir for {self.name}")
            os.mkdir(self.data_dir)
        elif os.path.isdir(self.htr_dir):
            if self.verbose:
                print(f"Found htr data for {self.name}")
            return
        else:
            if self.verbose:
                print(f"Found data dir for {self.name}, but no htr data")
        
        self.download_data()

    def download_data(self):
        # Download data from ozstar
        print(f"Downloading data for {self.name} from ozstar")
        pars = "*_I_*" if self.basic else "*"
        DMstr = self.DM if not self.download_all_DMs else ""
        get_fn = f"/fred/oz002/askap/craft/craco/processing/output/{self.name}/htr/{pars}{DMstr}.npy"
        to_fn = f"{self.htr_dir}/"
        if self.verbose:
            print(f"Downloading {get_fn} to {to_fn}")

        try:
            sysrsync.run(
                source=get_fn,
                source_ssh=f"{ozstar_user}@ozstar.swin.edu.au",
                destination=to_fn,
                sync_source_contents=False,
                verbose=self.verbose,
                options=["--progress"]
            )
        except Exception as e:
            print(e)
            print(f"Could not download {get_fn}, continuing")
    
    def load_data(self):
        # Load the data
        if self.DM is None:
            self.DM = self.get_DM()
        
        if self.verbose:
            print(f"Loading data for {self.name} at DM={self.DM}")

        # Load time series
        for p in "XYIQUV" if not self.basic else "I":
            fnames = glob(f"{self.htr_dir}{self.name}*_{p}_t_{self.DM}.npy")

            if not fnames:
                continue

            fname = fnames[0]

            if not os.path.isfile(fname):
                if self.verbose:
                    print(f"Could not find {fname}")
                continue

            if self.memmap:
                data = np.load(fname, mmap_mode="r")
            else:
                data = np.load(fname)
            setattr(self, p, data)
            setattr(self, f"{p}_full", data)
            setattr(self, f"{p}_all", data)

        # Load dynamic spectra (setting first axis to time)
        for p in "xyXYIQUV"if not self.basic else "I":
            fnames = glob(f"{self.htr_dir}{self.name}*_{p}_dynspec_{self.DM}.npy")

            if not fnames:
                continue

            fname = fnames[0]

            if not os.path.isfile(fname):
                if p not in "xyXY": # capitilsation is inconsistent
                    if self.verbose:
                        print(f"Could not find {fname}")
                continue
            if self.memmap:
                data = np.load(fname, mmap_mode="r")
            else:
                data = np.load(fname)

            # ensure time is first axis, noting we expect the longer
            # axis to be the time axis
            if data.shape[0] < data.shape[1]:
                data = data.T

            setattr(self, f"{p.upper()}_ds", data)
            setattr(self, f"{p.upper()}_ds_full", data)
            setattr(self, f"{p.upper()}_ds_all", data)
        
    def get_DM(self):
        # Figure out DM from file names
        files = glob(f"{self.htr_dir}{self.name}*")
        files = [file for file in files if "polcal" not in file]
        DMs = sorted(list(set(
                [f.split("_")[-1][:-4] for f in files]
        )))

        for i, DM in enumerate(DMs):
            try:
                _ = float(DM)
            except ValueError:
                DMs.remove(DM)

        if self.verbose:
            print(f"Found DMs for {self.name}: {DMs}")

        if len(DMs) > 1:
            print(f"Found multiple DMs for {self.name}: {DMs}")
            print("Using the config DM. To override, provide DM to FRB "
                  "constructor.")

        return self.config["dm_frb"]
    
    def data_status(self):
        # Print status of data
        print(f"FRB{self.name}")
        print(f"DM: {self.DM}")
        print(f"dt: {self.tavg} μs")
        if self.crop:
            print(f"Crop time: {self.crop_time[0]} μs → "
                  f"{self.crop_time[1]} μs")
        print(f"Peak time index: {self.peak}")
        print(f"Peak time: {self.peak*self.tavg} μs")

        loaded = [[],[]]
        for p in "XYIQUVL":
            if getattr(self, p) is None:
                loaded[0].append("×")
            else:
                loaded[0].append("✓")
        
            if getattr(self, f"{p}_ds") is None:
                loaded[1].append("×")
            else:
                loaded[1].append("✓")
        
        print(
            "              X  Y  I  Q  U  V  L\n"
            "time series:  " + "  ".join(loaded[0]) + "\n"
            "dyn spectra:  " + "  ".join(loaded[1])
        )
    
    # Data manipulation
    def apply_crop(self, force_peak=None):
        # Crop data around peak

        if force_peak is None:
        # Find rough peak
            rough_peak = np.argmax(
                scrunch(self.I_ds_full.sum(axis=1), 1000, verbose=self.verbose)
            )
        else:
            rough_peak = force_peak

        time_series_slice = slice(
            int((rough_peak-self.crop_dur/2)*336*1000), 
            int((rough_peak+self.crop_dur/2)*336*1000)
        )
        dynspec_slice = slice(
            int((rough_peak-self.crop_dur/2)*1000*(336//self.nchan)), 
            int((rough_peak+self.crop_dur/2)*1000*(336//self.nchan))
        )
        self.crop_time = [dynspec_slice.start, dynspec_slice.stop]

        if self.verbose:
            print(f"Crop time: {self.crop_time[0]} μs → "
                  f"{self.crop_time[1]} μs")

        # Crop time series
        for p in "XYIQUV":
            if getattr(self, p) is None:
                continue
            else:
                setattr(
                    self, 
                    f"{p}_full", 
                    getattr(self, f"{p}_all")[time_series_slice]
                )
            
        # Crop dynamic spectra
        for p in "XYIQUV":
            if getattr(self, f"{p}_ds") is None:
                continue
            else:
                setattr(
                    self, 
                    f"{p}_ds_full", 
                    getattr(self, f"{p}_ds_all")[dynspec_slice]
                )
        
    def set_dt(self, tavg, update_plots=True):
        # Set the time averaging to tavg
        self.set_dt_df(tavg, self.favg, update_plots=update_plots)

    def set_peak(self):
        # Set peak time index
        self.peak = np.argmax(self.I)
        if self.verbose:
            print(f"Peak time index: {self.peak}")
            print(f"Peak time: {self.peak*self.tavg*self.nchan} μs")

    def set_df(self, favg, update_plots=True):
        # Set the frequency averaging to favg
        self.set_dt_df(self.tavg, favg, update_plots=update_plots)

    def set_dt_df(self, tavg, favg, update_plots=True):
        # Set the time and frequency averaging to tavg and favg

        tscrunch = lambda x, n: np.sqrt(n) * (
            scrunch(x, n, verbose=self.verbose, func=np.mean)
            # - np.mean(x)
        ) #/ np.std(x)

        fscrunch = lambda x, n: scrunch(
            x, n, axis=1, verbose=self.verbose, func=np.mean
        )

        for p in "XYIQUV":
            if self.verbose:
                print(f"Setting {p} time resolution to {self.ds_base_dt * tavg} μs")
                print(f"Setting {p} frequency resolution to {self.ds_base_df * favg} MHz")
            
            
            # dynamic spectra - time and frequency
            if getattr(self, f"{p}_ds") is None:
                continue
            else:
                setattr(
                    self, 
                    f"{p}_ds", 
                    fscrunch(
                        tscrunch(
                            # roll so the peak is in the middle of the bin
                            np.roll(getattr(self, f"{p}_ds_full"), tavg//2, axis=0), 
                            tavg
                        ),
                        favg
                    )
                )

        if self.crossing_freq is not None and self.zap_above_crossing_freq:
            self.zap_above_freq(self.crossing_freq)
        
        if self.flagged_chans is not None:
            self.flag_dynspec(self.flagged_chans)

        for p in "XYIQUV":
            # time series - time only
            if getattr(self, p) is None:
                continue
            elif getattr(self, f"{p}_ds") is None:
                setattr(
                    self, 
                    p,
                    tscrunch(
                        np.roll(getattr(self, f"{p}_full"), tavg*self.nchan//2), 
                        tavg*self.nchan
                    )
                )
                if self.verbose and self.zap_above_crossing_freq:
                    print(f"WARNING: {p} time series may include data above crossing frequency")
            else:
                if self.verbose:
                    print(f"Getting {p} time series by summing over dynamic spectrum")
                setattr(
                    self, 
                    p,
                    np.nansum(
                        getattr(self, f"{p}_ds"),
                        axis=1
                    )/np.sqrt(self.nchan)
                )
        
        self.tavg = tavg
        self.favg = favg
        self.dt = self.ds_base_dt * tavg
        self.df = self.ds_base_df * favg
        self.t = np.arange(len(self.I))*self.dt
        self.t -= self.t[-1]/2  # centre time axis on middle of crop

        self.set_freqs()
        self.set_peak()
        

        # define off-region window as time at least 40 ms from peak
        self.off_burst = np.logical_or(
            self.t < self.t[self.peak] - 40000,
            self.t > self.t[self.peak] + 40000
        )

        if self.Q is not None and self.U is not None:
            self.set_L()

        if self.axes is not None and update_plots:
            if isinstance(self.axes, dict):
                self.plot(axes=self.axes)
        
        if self.g2_axes is not None and update_plots:
            self.calc_g2()

    def set_freqs(self):
        self.freqs = np.arange(self.nchan//self.favg)*self.df
        # make sure centre of freqs is f0
        self.freqs -= np.median(self.freqs) - self.f0
        # match frequency axis to dynamic spectrum
        self.freqs = self.freqs[::-1]

    def set_L(self):
        self.L = np.sqrt(self.Q**2 + self.U**2)
        self.L = self.L - np.mean(self.L[self.off_burst])
        self.L_ds = np.sqrt(self.Q_ds**2 + self.U_ds*2)
        T, F = self.L_ds.shape
        L_means = np.tile(np.nanmean(self.L_ds[self.off_burst], axis=0), (T, 1))
        self.L_ds = self.L_ds - L_means

    def dedisperse(self, new_DM: str, normalise: bool = True):
        # dedisperse data to new_DM (in pc/cm3)

        if self.verbose:
            print(f"Dedispersing to DM={new_DM} pc/cm3")

        delta_DM = float(new_DM) - float(self.DM)

        new_X = ifft(dedisperse_coherent(
            fft(self.X_full), delta_DM, self.f0, self.bw
        ))
        new_Y = ifft(dedisperse_coherent(
            fft(self.Y_full), delta_DM, self.f0, self.bw
        ))

        new_I, new_Q, new_U, new_V = calculate_stokes(
            new_X, new_Y, delta_phi=None, verbose=self.verbose
        )
        new_time_series = {
            "X": new_X,
            "Y": new_Y,
            "I": new_I,
            "Q": new_Q,
            "U": new_U,
            "V": new_V,
        }

        new_X_ds = generate_dynspec(new_X, verbose=self.verbose, label="X")
        new_Y_ds = generate_dynspec(new_Y, verbose=self.verbose, label="Y")

        new_I_ds, new_Q_ds, new_U_ds, new_V_ds = calculate_stokes(
            new_X_ds, new_Y_ds, delta_phi=None, normalise=normalise, verbose=self.verbose
        )
        new_dynspec = {
            "X": new_X_ds,
            "Y": new_Y_ds,
            "I": new_I_ds,
            "Q": new_Q_ds,
            "U": new_U_ds,
            "V": new_V_ds,
        }

        # for p in "XYIQUV":
        #     np.save(
        #         f"{self.htr_dir}{self.name}_{p}_t_{new_DM}.npy",
        #         new_time_series[p]
        #     )
        #     np.save(
        #         f"{self.htr_dir}{self.name}_{p}_dynspec_{new_DM}.npy",
        #         new_dynspec[p]
        #     )
        
        
        # update attributes
        if self.verbose:
            print("Updating attributes")

        self.DM = new_DM

        for p in "XYIQUV":
            setattr(self, f"{p}_full", new_time_series[p])
            setattr(self, f"{p}_ds_full", new_dynspec[p])
        
        # self.load_data()

        self.set_dt_df(self.tavg, self.favg)

        if self.verbose:
            print("Done")

    def set_dynspec_nchan(self, nchan, all_data=False):
        # create dynamic spectra with the desired number of channels
        # will reset frequency and time averaging to 1

        if self.verbose:
            print(f"Setting dynamic spectra to {nchan} channels")

        new_X_ds = generate_dynspec(
            self.X_all if all_data else self.X_full, nchan, verbose=self.verbose, label="X")
        new_Y_ds = generate_dynspec(
            self.Y_all if all_data else self.Y_full, nchan, verbose=self.verbose, label="Y")

        new_I_ds, new_Q_ds, new_U_ds, new_V_ds = calculate_stokes(
            new_X_ds, new_Y_ds, delta_phi=None, verbose=self.verbose
        )
        new_dynspec = {
            "X": new_X_ds,
            "Y": new_Y_ds,
            "I": new_I_ds,
            "Q": new_Q_ds,
            "U": new_U_ds,
            "V": new_V_ds,
        }

        # update attributes
        if self.verbose:
            print("Updating attributes")

        for p in "XYIQUV":
            setattr(self, f"{p}_ds_{'all' if all_data else 'full'}", new_dynspec[p])

        self.nchan = nchan
        self.ds_base_df = self.bw/nchan         # MHz
        self.ds_base_dt = 1/self.ds_base_df     # μs

        if self.crop:
            self.apply_crop()

        self.set_dt_df(1, 1)

        if self.verbose:
            print("Done")

    def set_crossing_freq(self):
        # find the frequency at which the FRB falls out of the data
        # (i.e. the crossing frequency)
        k_DM = 2.41e-4  #pc cm^-3 GHz^-2 us^-2

        # Find rough peak
        rough_peak_100us = np.argmax(
            scrunch(self.I_ds_all.sum(axis=1), 100, verbose=self.verbose)
        )

        # if self.crop:
        #     self.crossing_freq = (
        #         np.min(self.freqs/1e3)**(-2) - k_DM/float(self.DM) * (self.t[self.peak]+(self.crop_time[0])) 
        #     )**(-0.5)*1e3
        # else:
        self.crossing_freq = (
            # np.min(self.freqs/1e3)**(-2) - k_DM/float(self.DM) * (self.t[self.peak]-self.t[0]) 
            np.min(self.freqs/1e3)**(-2) - k_DM/float(self.DM) * rough_peak_100us*100
        )**(-0.5)*1e3

        if self.verbose:
            print(f"Crossing frequency: {self.crossing_freq:.2f} MHz")

    def zap_above_freq(self, zap_freq):
        # zap data above crossing frequency
        if self.verbose:
            print(f"Zapping data above {zap_freq:.2f} MHz")

        for p in "XYIQUV":
            for suf in ["_ds"]:
                if getattr(self, f"{p}{suf}") is None:
                    continue
                else:
                    # if any of the channel is above the crossing freq, zap
                    getattr(self, f"{p}{suf}")[:, self.freqs+self.df/2 > zap_freq] = np.nan

    def flag_dynspec(self, chans=None):
        # interactively flag channels unless list of channels provided

        flagged_chans = []

        if chans is None:
            plot_ds = self.I_ds.copy()

            fig, ax = plt.subplots()

            extent = [
                (self.t.min())/1e3,
                (self.t.max())/1e3,
                (self.f0 - self.bw/2)/1e3,
                (self.f0 + self.bw/2)/1e3,
            ]

            def draw(fig, ax):
                ax.clear()
                mask = np.ones_like(self.I_ds.T)
                mask[flagged_chans] = np.nan
                ax.imshow(
                    plot_ds.T * mask,
                    extent=extent,
                    aspect="auto",
                    interpolation="nearest",
                    cmap=cmr.arctic_r,
                )
                fig.canvas.draw()

            def on_click(event):
                freq = event.ydata*1e3

                # get channel index
                chan = np.argmin(np.abs(self.freqs - freq))

                print(freq, chan)

                if chan in flagged_chans:
                    flagged_chans.remove(chan)
                else:
                    flagged_chans.append(chan)

                print(flagged_chans)

                draw(fig, ax)

            cid = fig.canvas.mpl_connect('button_press_event', on_click)
            draw(fig, ax)

            plt.show()
        else:
            flagged_chans = chans

        if self.verbose:
            print(f"Flagged channels: {sorted(flagged_chans)}")

        self.flagged_chans = flagged_chans
        for p in "IQUV":
            getattr(self, f"{p}_ds")[:,flagged_chans] = np.nan

    def de_RM(self, RM):
        # correct dynamic spectra for Faraday rotation
        lambda_sq = (3e8/(self.freqs*1e6))**2

        psi_RM = RM * (lambda_sq - lambda_sq[0])
        psi_RM = np.repeat(psi_RM, self.I_ds.shape[0]).reshape(self.I_ds.T.shape).T

        Q_deRM = self.Q_ds * np.cos(2*psi_RM) + self.U_ds * np.sin(2*psi_RM)
        U_deRM = -self.Q_ds * np.sin(2*psi_RM) + self.U_ds * np.cos(2*psi_RM)

        # zero mean channels
        T, F = Q_deRM.shape
        Q_means = np.tile(np.nanmean(Q_deRM[self.off_burst], axis=0), (T, 1))
        U_means = np.tile(np.nanmean(U_deRM[self.off_burst], axis=0), (T, 1))

        Q_deRM -= Q_means
        U_deRM -= U_means

        self.Q_ds = Q_deRM
        self.U_ds = U_deRM
        self.Q = np.nansum(Q_deRM, axis=1)/np.sqrt(self.Q_ds.shape[1])
        self.U = np.nansum(U_deRM, axis=1)/np.sqrt(self.U_ds.shape[1])

        self.set_L()

    # Plotting
    def plot(
            self, 
            ax=None, 
            axes=None,
            show=True, 
            xlabel="Time (ms)", 
            title=True,
            xlim=None,
            legend=True,
            **kwargs
        ):
        """Plot PA, time series, and I dynamic spectrum

        If ax is provided, split into three subplots

        :param ax: _description_, defaults to None
        :type ax: _type_, optional
        """
        if ax is None and axes is None:
            fig, axes = plt.subplot_mosaic(
                "P;S;D", 
                figsize=(4, 10), 
                gridspec_kw={"height_ratios": [1, 2, 2]},
                sharex=True
            )
        elif axes is None:
            fig = ax.get_figure()
            div = make_axes_locatable(ax)
            ax_p = div.append_axes("top", size="50%", pad=0, sharex=ax)
            ax_s = ax#div.append_axes("top", 1, pad=0.1, sharex=ax)
            ax_d = div.append_axes("bottom", size="100%", pad=0, sharex=ax)
            axes = {"P": ax_p, "S": ax_s, "D": ax_d}

            ax_p.set_xticks([])
            ax_s.set_xticks([])
        else:
            fig = axes["P"].get_figure()

            #clear axes
            for ax in axes.values():
                ax.clear()


        self.plot_PA(axes["P"], xlabel=None, title=title, **kwargs)
        self.plot_time_series(axes["S"], xlabel=None, title=False, legend=legend, **kwargs)
        self.plot_dynspec(axes["D"], xlabel=xlabel, title=False, **kwargs)

        if xlim is None:
            xlim = [
                (self.t[self.peak]-100*self.dt)/1e3, 
                (self.t[self.peak]+100*self.dt)/1e3
            ]

        axes["D"].set_xlim(xlim)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)

        self.fig = fig
        self.axes = axes

        if show:
            fig.show()
        else:
            return fig, axes
        
    def plot_PA(
            self, 
            ax=None, 
            xlabel="Time (μs)",
            ylabel="PA (deg)",
            title=True,
            **kwargs
        ):
        """Plot PA vs time

        :param ax: _description_, defaults to None
        :type ax: _type_, optional
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # time axis
        t = (self.t)/1e3

        self.calc_PA()

        ax.errorbar(
            t,
            self.PA,
            yerr=self.PA_err,
            fmt=".",
            ms=0,
            **kwargs
        )

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if title is not str:
            namestr = f"FRB{self.name}"
            dtstr = f"\mathrm{{dt}}={self.dt}\,\mathrm{{\mu s}}"
            dfstr = f"\mathrm{{df}}={self.df}\,\mathrm{{MHz}}"
            DMstr = f"\mathrm{{DM}}={self.DM}\,\mathrm{{pc\,cm}}^{{-3}}"
            ax.set_title(
                rf"{namestr}$^{{{dtstr}~|~{dfstr}}}_{{{DMstr}}}$"
            )
        elif title:
            ax.set_title(title)

        return fig, ax

    def plot_time_series(
            self,
            ax=None,
            xlabel="Time (ms)",
            ylabel=r"Flux density ($\sigma$)",
            title=True,
            legend=True,
            pars="IQUV",
            **kwargs
        ):
        """Plot time series

        :param ax: _description_, defaults to None
        :type ax: _type_, optional
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # time axis
        t = (self.t)/1e3

        for i, p in enumerate(pars):
            if getattr(self, p) is None:
                continue
            ax.step(
                t, 
                getattr(self, p), 
                label=p, 
                where="mid",
                lw=1,
                zorder=-i,
                **kwargs
            )

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(fr"{self.name} [dt={self.tavg} $\mu$s]")
        if legend:
            ax.legend()

        return fig, ax
    
    def plot_dynspec(
            self,
            ax=None,
            xlabel="Time (ms)",
            ylabel="Frequency (GHz)",
            title=True,
            p="I",
            cmap=dft_cmap,
            **kwargs
        ):
        """Plot dynamic spectrum

        :param ax: _description_, defaults to None
        :type ax: _type_, optional
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        t_peak = self.t[self.peak]

        extent = [
            (self.t.min())/1e3,
            (self.t.max())/1e3,
            (self.f0 - self.bw/2)/1e3,
            (self.f0 + self.bw/2)/1e3,
        ]

        if self.I_ds is not None:
            ax.imshow(
                getattr(self, f"{p}_ds").T,
                aspect="auto", 
                interpolation="none",
                cmap=cmap,
                extent=extent,
                **kwargs
            )

        # if self.crossing_freq/1e3 < extent[3]:
        #     ax.hlines([self.crossing_freq/1e3], *extent[:2], color="r", lw=1)
        #     ax.set_ylim(None, self.crossing_freq/1e3)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(f"FRB{self.name} [dt={self.dt} μs]")

        return fig, ax

    def plot_IQUV(
              self,
              axs=None,
              xlabel="Time (ms)",
              ylabel="Frequency (MHz)",
              cmap=dft_cmap,
              figsize=(3.5, 10),
              **kwargs
    ):
        if axs is None:
            fig, axs = plt.subplots(
                nrows=4, sharex=True, sharey=True, figsize=figsize
            )
        else:
            axs = axs.flatten()
            assert len(axs) == 4, "Need 4 axes to plot IQUV"
            fig = axs[0].get_figure()
        
        for i, p in enumerate("IQUV"):
            self.plot_dynspec(
                ax=axs[i],
                xlabel=None if p != "V" else xlabel,
                ylabel=ylabel,
                title=False,
                p=p,
                cmap=cmap,
                **kwargs
            )
            ax2 = axs[i].twinx()
            ax2.set_ylabel(p, rotation=0, fontsize=20, labelpad=20)
            ax2.set_yticks([])

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        return fig, axs

    def plot_ILV(
            self,
            axs=None,
            xlabel="Time (ms)",
            **kwargs
    ):
        if axs is None:
            fig, axs = plt.subplots(
                nrows=2, sharex=True
            )
        else:
            axs=axs.flatten()
            assert len(axs) == 2, "Need two axes"
            fig = axs[0].get_figure()

        t = (self.t)/1e3

        cols = ["k", "b", "r"]

        I_err = np.nanstd(self.I[self.off_burst])
        L_err = np.nanstd(self.L[self.off_burst])
        V_err = np.nanstd(self.V[self.off_burst])

        # first ax: I, L, and V vs t
        ax = axs[0]
        for i, p in enumerate("ILV"):
            ax.step(
                t,
                getattr(self, p)/np.nanstd(getattr(self, p)[self.off_burst]),
                where="mid",
                lw=1,
                label=p,
                c=cols[i],
                **kwargs
            )
        
        ax.set_ylabel(r"Flux density ($\sigma$)")
        ax.legend()
        ax.set_title(rf"FRB{self.name} | dt={self.dt} $\mu$s")

        # second ax: pol fractions 
        ax = axs[1]
        fracs = [self.L/self.I, self.V/self.I]

        mask = (self.I/np.nanstd(self.I[self.off_burst]) > 5)

        t[~mask] = np.nan
        
        to_plot = [np.sqrt(fracs[0]**2+fracs[1]**2), fracs[0], fracs[1]]

        errs = [
            0,
            np.sqrt(L_err**2+self.L**2+I_err**2)/self.I**2,
            np.sqrt(V_err**2+self.V**2+I_err**2)/self.I**2
        ]

        errs[0] = np.sqrt(fracs[0]**2*errs[1]**2+fracs[1]**2*errs[2]**2) / np.sqrt(fracs[0]**2 + fracs[1]**2)
        
        for i, p in enumerate(to_plot):
            ax.fill_between(
                t, p-errs[i], p+errs[i],
                step="mid",
                color=cols[i],
                alpha=0.3,
                **kwargs
            )
            ax.step(
                t,
                p,
                where="mid",
                lw=1,
                label=["Total", "L", "V"][i],
                c=cols[i],
                **kwargs
            )
        ax.set_ylabel("Fraction")
        ax.legend()
        ax.set_xlabel("Time (ms)")

    # Analysis
    def calc_g2(
            self, axs=None, xlabel="Time (ms)", 
            ylabels=[r"Flux dens. ($\sigma$)", r"$g^{(2)}(0)$", "Residual (\%)"],
            xlim=None, title=None, legend=False
        ):
        # calculate g2(0) over currently plotted window

        if self.axes is None:
            self.plot()
    
        if xlim is None:
            xlim = self.axes["S"].get_xlim()
        else:
            self.axes["S"].set_xlim(xlim)

        t = (self.t)/1e3

        start_idx = np.argmin(abs(t - xlim[0])) * self.tavg*self.nchan
        end_idx = np.argmin(abs(t - xlim[1])) * self.tavg*self.nchan

        if self.verbose:
            print(f"Calculating g2 from {start_idx} to {end_idx}")

        I = self.I_full[start_idx:end_idx]
        Q = self.Q_full[start_idx:end_idx]
        U = self.U_full[start_idx:end_idx]
        V = self.V_full[start_idx:end_idx]

        binwidth = self.tavg*self.nchan

        EI, g2, expected, res = calculate_g2(
            I, Q, U, V, binwidth, verbose=self.verbose
        )

        # plot g2 and I in new figure
        if axs is None:
            if self.g2_fig is None or self.g2_axes is None:
                self.g2_fig, self.g2_axes = plt.subplots(nrows=3, sharex=True)
            else:
                #clear
                for ax in self.g2_axes:
                    ax.clear()
        else:
            assert len(axs) == 3, "Need 3 axes to plot g2"
            self.g2_axes = axs
            self.g2_fig = axs[0].get_figure()
        
        fig = self.g2_fig
        ax = self.g2_axes

        ax[0].step(t, self.I, where="mid", label=r"$\langle I\rangle$", lw=1)
        ax[0].step(t, self.L, where="mid", label=r"$\langle L\rangle$", lw=1)
        ax[0].step(t, self.V, where="mid", label=r"$\langle V\rangle$", lw=1)
        ax[0].set_title(title)
        if legend:
            ax[0].legend(fontsize="small", loc="center left", bbox_to_anchor=(1, 0.5))
        ax[1].step(
            t[start_idx//(self.tavg*self.nchan):end_idx//(self.tavg*self.nchan)],
            g2,
            c="k",
            where="mid",
            label=r"$g^{(2)}_{\mathrm{FRB}}(0)$",
            lw=1
        )
        ax[1].step(
            t[start_idx//(self.tavg*self.nchan):end_idx//(self.tavg*self.nchan)],
            expected,
            "r",
            where="mid",
            label=r"$g^{(2)}_{\mathrm{IPPR}}(0)$",
            lw=1
        )
        if legend:
            ax[1].legend(fontsize="small", loc="center left", bbox_to_anchor=(1, 0.5))
        # ax[1].legend(fontsize="small")
        ax[2].scatter(
            t[start_idx//(self.tavg*self.nchan):end_idx//(self.tavg*self.nchan)],
            res,
            c="k",
            s=1
        )
        # ax[2].axhline(0, c="k", lw=1)
        # for sgn in [-1, 1]:
        #     ax[2].axhline(sgn*sig_res, c="r", lw=1, ls="dashed")
        #     ax[2].axhline(2*sgn*sig_res, c="r", lw=1, ls="dashdot")
        #     ax[2].axhline(3*sgn*sig_res, c="r", lw=1, ls="dotted")
        # ax2t = ax[2].twinx()
        # ax2t.set_yticks([i*sig_re        else:
        # ax2t.set_yticklabels([-3, -2, -1, 0, 1, 2, 3])

        # ax2t.set_ylim(ax[2].get_ylim())

        ax[2].set_xlim(xlim)

        if xlabel is not None:
            ax[2].set_xlabel(xlabel)

        if ylabels is not None:
            for i in range(3):
                ax[i].set_ylabel(ylabels[i])

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        fig.show()
    
    def calc_g2_over_DMs(
            self, axs=None, xlabel="Time (ms)", xlim=None, DMs=None, mode="direct",
            ylabel=r"$\Delta$DM (pc cm$^{-3}$)"
        ):
        # calculate g2(0) over currently plotted window and range of DMs

        if self.axes is None:
            self.plot()
        
        if xlim is None:
            xlim = self.axes["S"].get_xlim()
        else:
            self.axes["S"].set_xlim(xlim)

        t = (self.t)/1e3

        start_idx = np.argmin(abs(t - xlim[0])) * self.tavg*self.nchan
        end_idx = np.argmin(abs(t - xlim[1])) * self.tavg*self.nchan

        if DMs is None:
            if self.verbose:
                print ("Using default DM range")
            DMs = np.arange(-0.1, 0.11, 0.01)
        
        dDM = DMs[1] - DMs[0]

        if self.verbose:
            print(f"Calculating g2 from {start_idx} to {end_idx}")
            print(f"DM range: {DMs[0]} to {DMs[-1]}, dDM={dDM}")

        X = self.X_full[start_idx:end_idx].copy()
        Y = self.Y_full[start_idx:end_idx].copy()

        binwidth = self.tavg*self.nchan

        # start X and Y at first DM
        X = ifft(dedisperse_coherent(
            fft(X), DMs[0], self.f0, self.bw, ref_func=np.median
        ))
        Y = ifft(dedisperse_coherent(
            fft(Y), DMs[0], self.f0, self.bw, ref_func=np.median
        ))

        # get dDM phases
        freqs = get_freqs(self.f0, self.bw, X.shape[0])
        dDM_phases = generate_phases(
            dDM, freqs, np.median(freqs)
        )

        EI = np.zeros((len(DMs), len(X)//binwidth))
        g2 = np.zeros((len(DMs), len(X)//binwidth))
        expected = np.zeros((len(DMs), len(X)//binwidth))
        res = np.zeros((len(DMs), len(X)//binwidth))

        for i in tqdm(range(len(DMs))):
            I = np.abs(X)**2 + np.abs(Y)**2
            Q = np.abs(X)**2 - np.abs(Y)**2
            U = 2*np.real(X*np.conj(Y))
            V = 2*np.imag(X*np.conj(Y))

            EI[i], g2[i], expected[i], res[i] = calculate_g2(
                I, Q, U, V, binwidth, mode=mode, verbose=self.verbose
            )

            X = ifft(fft(X)*dDM_phases)
            Y = ifft(fft(Y)*dDM_phases)
        
        # plot g2 and I in new figure
        if axs is None:
            if self.g2_fig is None or self.g2_axes is None:
                self.g2_fig, self.g2_axes = plt.subplots(nrows=4, sharex=True)
            else:
                #clear
                for ax in self.g2_axes:
                    ax.clear()
        else:
            axs = axs.flatten()
            assert len(axs) == 4, "Need 4 axes to plot g2"
            self.g2_axes = axs
            self.g2_fig = axs[0].get_figure()
        
        fig = self.g2_fig
        ax = self.g2_axes

        extent = [
            xlim[0]-self.dt/2e3,
            xlim[1]+self.dt/2e3,
            DMs[0]-dDM/2,
            DMs[-1]+dDM/2,
        ]

        ax[0].imshow(
            EI*1e3, # get values nice for cbar label
            aspect="auto", 
            interpolation="nearest",
            extent=extent,
            origin="lower"
        )

        ax[1].imshow(
            g2,
            aspect="auto", 
            interpolation="nearest",
            extent=extent,
            vmin=np.nanmin([g2, expected]),
            vmax=np.nanmax([g2, expected]),
            origin="lower"
        )

        ax[2].imshow(
            expected,
            aspect="auto", 
            interpolation="nearest",
            extent=extent,
            vmin=np.nanmin([g2, expected]),
            vmax=np.nanmax([g2, expected]),
            origin="lower"
        )
        # norm_res = colors.TwoSlopeNorm(
        #     vmin=min(np.nanmin(res), -1e-1),
        #     vcenter=0,
        #     vmax=max(1e-1, np.nanmax(res)),
        # )
        norm_res = colors.Normalize(
            vmin=-5,
            vmax=5,
        )
        ax[3].imshow(
            res/np.nanstd(res),
            aspect="auto", 
            interpolation="nearest",
            extent=extent,
            cmap=cmr.redshift,
            norm=norm_res,
            origin="lower"
        )

        for i in range(len(ax)):
            ax[i].set_ylabel(ylabel, fontsize="small")
        
        ax[-1].set_xlabel(xlabel)

        fig.show()

        return EI, g2, expected, res

    def mcf(self):
        autocorrelation = lambda x: sig.correlate(x, x, mode="full")
        t_full = np.arange(len(self.I_all))*self.full_dt

        xlim = self.axes["S"].get_xlim()

        start_idx = np.argmin(abs(self.t - xlim[0])) * self.tavg*self.nchan
        end_idx = np.argmin(abs(self.t - xlim[1])) * self.tavg*self.nchan

        on = slice(start_idx, end_idx)
        off = np.logical_or(
            t_full < self.crop_time[0],
            t_full > self.crop_time[1]
        )

        MCFs = []

        for P in [self.X_all, self.Y_all]:
            V_on = P[on]
            V_off = P[off]

            # Figure out tau
            tau_lim = V_on.shape[0] - 1  # Extent of tau (positive and negative)
            tau = np.arange(-tau_lim, tau_lim+1)*self.full_dt
            zero = (tau == 0)  # Mask for easy access to tau = 0

            # Treat the off-signal voltages as just the noise
            # We can do this because we really only care about the statistics of each
            P = V_on
            N = V_off

            C_V = autocorrelation(P)
            C_N = autocorrelation(N)
            C_S = C_V - C_N

            I_V = np.abs(P * np.conj(P))
            I_N = np.abs(N * np.conj(N))

            C_I_V = autocorrelation(I_V)
            C_I_N = autocorrelation(I_N)

            # Expectations of intensities
            E_I_V = np.mean(I_V)
            E_I_N = np.mean(I_N)
            E_I_S = C_S[zero]  # A handy identity!
            #E_I_S = E_I_V - E_I_N  # Above identity gives nonsensical results
            #print(f'<I_V>: {E_I_V}')
            #print(f'<I_N>: {E_I_N}')
            #print(f'<I_S>: {C_S[zero]} (C_S(0))')
            #print(f'<I_S>: {E_I_V - E_I_N} (approx)')

            # Other required quantities
            # Need to add/remove np.conj, as sig.correlate automatically uses the
            # conjugate of the second argument.
            C_VstarVstar = sig.correlate(np.conj(P), np.conj(P))
            C_NstarNstar = sig.correlate(np.conj(N), np.conj(N))
            C_NN = sig.correlate(N, N)

            C_I_S = C_I_V \
                    - C_I_N \
                    - 2*E_I_S*E_I_N \
                    - 2*np.real((C_VstarVstar - C_NstarNstar)*C_NN) \
                    - 2*np.real((C_V - C_N)*C_N)
            def quick_plot(x): plt.plot(tau, x); plt.ion(); plt.show()
            #print(f'C_I_V: [plotted]')
            #quick_plot(C_I_V)
            #print(f'C_I_N: [plotted]')
            #quick_plot(C_I_N)
            #print(f'2<I_S><I_N>: {2*E_I_S*E_I_N}')


            # Finally, calculate the MCF
            MCF = C_I_S/C_I_S[zero] - 0.5*(np.abs(C_S/C_S[zero])**2 + 1)
            MCFs.append(MCF)
        
        return MCFs

    def calc_PA(self):
        # calculate and set the PA by method in Day+2020
        I = self.I
        Q = self.Q
        U = self.U

        I = scrunch(
            np.abs(self.X_full)**2 + np.abs(self.Y_full)**2, self.tavg*self.nchan
        )

        # time axis
        t = self.t

        # define off-region window as time at least 25 ms from peak
        off_burst = self.off_burst

        I_err = np.std(I[off_burst])
        U_err = np.std(U[off_burst])
        Q_err = np.std(Q[off_burst])

        # calculate PA
        PA = np.arctan2(U, Q)/2
        PA_err = np.sqrt(
            (Q**2 * U_err**2 + U**2 * Q_err**2) / (Q**2 + U**2)**2
        )/2

        L_meas = np.sqrt(Q**2 + U**2)
        L_debias = I_err*np.sqrt((L_meas/I_err)**2 - 1)
        L_mask = (L_meas/I_err >= 1.57)
        L_debias[~L_mask] = 0
        # PA_mask = (abs(I) > I_err) & (L_debias > 0)

        # PA[~PA_mask] = np.nan
        # PA_err[~PA_mask] = np.nan

        # PA[off_burst] = np.nan
        # PA_err[off_burst] = np.nan

        # PA = np.unwrap(PA, discont=np.pi*2/3, period=np.pi)

        PA = PA % np.pi

        PA = np.rad2deg(PA)
        PA_err = np.rad2deg(PA_err)

        self.PA = PA
        self.PA_err = PA_err
        self.L_debias = L_debias
    
    def calc_RM(self, t_idx=None, method="rmtools"):
        # calculate RM - adapted from method provided by Apurba Bera
        t_idx = t_idx if t_idx else self.peak
        noise_idx = 1 #self.I_ds[self.off_burst].shape[0]//2

        rmtdata	= np.array(
            [self.freqs*1e6,
            self.I_ds[t_idx], 
            self.Q_ds[t_idx], 
            self.U_ds[t_idx], 
            np.std(self.I_ds[self.off_burst], axis=0),
            np.std(self.Q_ds[self.off_burst], axis=0), 
            np.std(self.Q_ds[self.off_burst], axis=0)]
        )

        if method == "rmtools":        
            rmd, rmad = run_rmsynth(
                rmtdata, polyOrd=3, phiMax_radm2=1.0e3, dPhi_radm2=1.0, nSamples=100.0,
                weightType='variance', fitRMSF=False, noStokesI=False, 
                phiNoise_radm2=1000000.0,
                nBits=32, showPlots=self.verbose, debug=False, verbose=self.verbose, log=print, 
                units='Jy/beam', prefixOut='prefixOut', saveFigures=None,
                fit_function='log'
            )
            
            rmc = run_rmclean(
                rmd, rmad, 0.1, maxIter=1000, gain=0.1, nBits=32, showPlots=self.verbose, 
                verbose=self.verbose, log=print
            )
            
            # print(rmc[0])
            
            res	= [
                rmc[0]['phiPeakPIfit_rm2'], rmc[0]['dPhiPeakPIfit_rm2'], 
                rmc[0]['polAngle0Fit_deg'], rmc[0]['dPolAngle0Fit_deg']
            ]
            
            return(res)	
        
        elif method == "rmnest":
            print(self.freqs, self.f0)
            print(self.Q_ds[t_idx], self.U_ds[t_idx], self.V_ds[t_idx])
            print(self.Q_ds[noise_idx], self.U_ds[noise_idx], self.V_ds[noise_idx])
            os.system("mkdir junk")
            rmn = RMNest(freqs=self.freqs, freq_cen = self.f0, 
                         s_q = self.Q_ds[t_idx], 
                         s_u = self.U_ds[t_idx], 
                         s_v = self.V_ds[t_idx],
                         rms_q = self.Q_ds[noise_idx], 
                         rms_u = self.U_ds[noise_idx],
                         rms_v = self.V_ds[noise_idx],
                        )
            rmn.fit(gfr=False, outdir='junk')
            rmn.print_summary()
                
            return(0)

    def scint(self, n_chan=336, n_subbands=1, mask=None, do_scatt=True, do_scint=True,
              tau_bounds=None, t0_bounds=None, width_bounds=None, a_bounds=None,
              nuDC_bounds=None, C_bounds=None
              ):
        # make interactive plot to select pulse timing
        print("Select start (left click) and stop (right click) times for "\
              "scintillation analysis")
        fig, ax = plt.subplots(nrows=2, sharex=True)

        # get first time profile drops below peak/e
        tau_guess_smp = np.argmax((self.I < self.I[self.peak]/np.e) & (np.arange(len(self.I)) > self.peak)) - self.peak
        tau_guess_us = int(tau_guess_smp * self.dt)

        def draw_plot():
            ax[0].plot(self.I)
            ax[0].axhline(self.I[self.peak]/np.e, c="k", ls="--")
            ax[0].axvline(self.peak, c="k", ls="--")
            ax[0].axvline(tau_guess_smp+self.peak, c="k", ls="--")

            ax[1].imshow(self.I_ds.T, aspect="auto", interpolation="nearest")

        draw_plot()

        start_idx = None
        stop_idx = None

        def onclick(event):
            nonlocal start_idx, stop_idx
            if event.button == 1:
                start_idx = event.xdata
                ax[0].clear()
                draw_plot()
                ax[0].axvline(start_idx, c="r", ls="--")
                if stop_idx is not None:
                    ax[0].axvline(stop_idx, c="r", ls="--")
                fig.canvas.draw() 
            elif event.button == 3:
                stop_idx = event.xdata
                ax[0].clear()
                draw_plot()
                if start_idx is not None:
                    ax[0].axvline(start_idx, c="r", ls="--")
                ax[0].axvline(stop_idx, c="r", ls="--")
                fig.canvas.draw() 
        
        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show()


        t_s = self.t/1e6
        t_s = t_s - t_s[0]
        start_s = t_s[int(start_idx)]
        stop_s = t_s[int(stop_idx)]
        rise_time_us = int((t_s[self.peak] - start_s)*1e6)

        print(f"Start time = {start_s} s")
        print(f"Stop time = {stop_s} s")
        print(f"Duration = {stop_s-start_s} s")
        print(f"tau guess = {tau_guess_us} us")
        print(f"rise time = {rise_time_us} us")

        # default masking: get rid zapped freqs
        if mask is None:
            mask = np.ones(self.nchan, dtype=bool)
            mask[np.isnan(self.I_ds).any(axis=0)] = 0

            if self.nchan > n_chan:
                mask = scrunch(mask, self.nchan//n_chan, axis=0, func=np.any)
            elif self.nchan < n_chan:
                mask = np.repeat(mask, n_chan//self.nchan)

        # do initial rough fit to get bounds
        # xdata = t_s[int(start_idx):int(stop_idx)]*1e6
        # ydata = self.I[int(start_idx):int(stop_idx)]
        # p0 = [0, xdata[0], rise_time_us/10, 1, tau_guess_us]
        # # popt, pcov = curve_fit(scint.pulseFit, 
        # #                        xdata, 
        # #                        ydata, 
        # #                        p0=p0,
        # #                        nan_policy="omit")

        # # print(popt)
        # # print(np.diag(pcov))

        # plt.figure()
        # plt.plot(xdata, ydata)
        # plt.plot(xdata, scint.pulseFit(xdata, *p0))
        # # plt.plot(t_s[int(start_idx):int(stop_idx)], scint.pulseFit(t_s[int(start_idx):int(stop_idx)], *popt))
        # plt.show()

        # return
        
        # default bounds
        if tau_bounds is None:
            tau_bounds = [int(tau_guess_us*0.75)*336//n_chan, int(tau_guess_us*1.25)*336//n_chan]
        if t0_bounds is None:
            t0_bounds = [0, rise_time_us*2*336//n_chan]
        if width_bounds is None:
            width_bounds = [1, rise_time_us*2*336//n_chan]
        if a_bounds is None:
            a_bounds = [1, 1000]
        if nuDC_bounds is None:
            nuDC_bounds = [1e-4, 30]
        if C_bounds is None:
            C_bounds = [0.01, 5]

        res = scint.ScatteringandScintillationMaster(
            self.X_full, self.Y_full,
            start_s, stop_s-start_s,
            n_chan, self.name, self.f0*un.MHz,
            n_subbands,
            userMask=mask,
            do_scatt=do_scatt, do_scint=do_scint,
            tau_bounds=tau_bounds, t0_bounds=t0_bounds, 
            width_bounds=width_bounds, a_bounds=a_bounds,
            nuDC_bounds=nuDC_bounds, C_bounds=C_bounds,
        )

        print(res)
        return(res)

    def fit_spectrum(self, plot=True, savefig=None):
        # fit spectrum to eq 1 of Pleunis+2021
        # https://dx.doi.org/10.3847/1538-4357/ac33ac

        spec = self.I_ds[self.peak]
        f = self.freqs
        f = f[~np.isnan(spec)]
        spec = spec[~np.isnan(spec)]
        spec_sig = np.nanstd(self.I_ds[self.off_burst], axis=0)
        spec_sig = spec_sig[~np.isnan(spec_sig)]
        print(spec)
        print(spec.shape)
        print(spec_sig)
        print(spec_sig.shape)
        # spec[np.isnan(spec)] = 0
        f_piv = self.f0 - self.bw/2

        model = lambda f, A, gamma, r: A*(f/f_piv)**(gamma+r*np.log(f/f_piv))

        p0 = [1, -2, 0]

        popt, pcov = curve_fit(model, f, spec, p0=p0, sigma=spec_sig, nan_policy="omit")

        print(f"A = {popt[0]} +- {pcov[0, 0]}")
        print(f"γ = {popt[1]} +- {pcov[1, 1]}")
        print(f"r = {popt[2]} +- {pcov[2, 2]}")

        if plot:
            fig, ax = plt.subplots()
            ax.plot(f, spec, "k", label="Data")
            ax.plot(f, spec-spec_sig, "k--", label="Data")
            ax.plot(f, spec+spec_sig, "k--", label="Data")
            ax.plot(f, model(f, *popt), "r--", label="Fit")
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Flux density")
            ax.set_title(f"FRB{self.name}")
            ax.text(0.95, 0.95,
                    f"$A = {popt[0]:.2f} \pm {pcov[0, 0]:.2f}$\n"\
                    f"$\gamma = {popt[1]:.2f} \pm {pcov[1, 1]:.2f}$\n"\
                    f"$r = {popt[2]:.2f} \pm {pcov[2, 2]:.2f}$",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    )
            # ax.legend()

            if savefig is None:
                plt.show()
            else:
                plt.savefig(savefig)

    def fit_boxcar(self, n0=None, off_burst_pad=40000, n_max=10000):
        ns = []
        vals = []
        stds = []
        means = []

        # use local peak because 1us series can get it wrong
        peak = self.t.shape[0]//2
        off_burst = np.logical_or(
            self.t < -off_burst_pad,
            self.t > off_burst_pad
        )

        I = ((self.I - np.mean(self.I[off_burst]))/np.std(self.I[off_burst]))

        def conv_boxcar(x):
            n=int(x[0])
            if n in ns:
                return vals[(np.array(ns) == n).argmax()]
            ns.append(n)
            boxcar = np.ones(n)/n
            convd = np.empty(I.shape[0])
            convd[:] = np.nan
            convd[:-(n-1)] = np.convolve(I, boxcar, mode="valid")
            convd = np.roll(convd, n//2)
            convd_std = np.nanstd(convd[off_burst])
            stds.append(convd_std)
            convd_mean = np.nanmean(convd[off_burst])
            means.append(convd_mean)
            # val = -np.nanmax((convd-convd_mean)/convd_std)
            val = -np.nanmax(convd)*np.sqrt(n)
            vals.append(val)
            return val
        
        # first pass: find initial guess (if not provided)
        if n0 is None:
            test_ns = np.logspace(0, np.log10(n_max), 1000, dtype=int)
            test_ns = np.unique(test_ns)
            test_ns = test_ns[1:]   # remove boxcar of 1
            test_vals = [conv_boxcar([n]) for n in tqdm(test_ns)]
            n0 = test_ns[np.argmin(test_vals)]
            v0 = np.min(test_vals)
    
        print(f"Initial guess: {n0}")

        # # second pass: refine
        # res = minimize(conv_boxcar, [n0], bounds=[(2, None)],
        #                method="Nelder-Mead")
        # n0 = res.x[0]
        # v0 = -res.fun
        # print(f"Best fit: {int(res.x[0])} with S/N {-res.fun}")
        
        # second pass: brute force from n0//2 to 2*n0
        res = None
        print("Refining...")
        while True:
            print(f"{n0//2} --> {2*n0}")    
            test_ns = np.arange(n0//2, 2*n0+1)
            test_vals = [conv_boxcar([n]) for n in tqdm(test_ns)]
            if min(test_vals) == v0:
                break
            else:
                n0 = test_ns[np.argmin(test_vals)]
                v0 = np.min(test_vals)


        # third pass: get error bounds by going up and down until S/N
        # reduces by 1
        thresh = v0 + 1

        def find_bound(init_step):
            n = int(n0)
            diff = 1
            step = init_step
            with tqdm(total=1) as pbar:
                while True:
                    n += step
                    if n <= 1:
                        n -= step
                        step = init_step
                        next
                    val = conv_boxcar([n])
                    old_diff = diff
                    diff = thresh - val
                    delta = old_diff - diff
                    pbar.update(delta)
                    if val > thresh:
                        if step == init_step:
                            break
                        else:
                            n -= step
                            step = init_step
                    else:
                        step *= 2
            return n
        
        n_up = find_bound(1)
        n_down = find_bound(-1)

        print(f"Upper bound: {n_up}")
        print(f"Lower bound: {n_down}")

        n = int(n0)
        boxcar = np.ones(n)/n
        convd = np.empty(I.shape[0])
        convd[:] = np.nan
        convd[:-(n-1)] = np.convolve(I, boxcar, mode="valid")
        convd = np.roll(convd, n//2)

        return(
            int(n0),
            (n_down, n_up),
            v0, 
            convd, 
            ns, 
            vals,
            off_burst,
            stds,
            means
        )
    
    def calc_pol_frac(self, window):
        # calculacte polarisation fractions in provided index window
        Q_prof = np.nansum(self.Q_ds, axis=1)
        U_prof = np.nansum(self.U_ds, axis=1)
        # L_prof = np.sqrt((Q_prof/np.nanstd(Q_prof[self.off_burst]))**2 \
        #             + (U_prof/np.nanstd(U_prof[self.off_burst]))**2)

        I_prof = np.nansum(self.I_ds, axis=1)
        V_prof = np.nansum(self.V_ds, axis=1)

        e_I = np.nanstd(I_prof[self.off_burst])
        e_V = np.nanstd(V_prof[self.off_burst])

        L_prof = e_I * np.sqrt(
            (np.sqrt(Q_prof**2 + U_prof**2)/e_I)**2 - 1
        )
        L_prof -= np.nanmean(L_prof[self.off_burst])

        e_L = np.nanstd(L_prof[self.off_burst])

        # I_prof /= e_I
        # V_prof /= e_V

        n = window[1] - window[0]
        L = np.nansum(L_prof[window[0]:window[1]])
        I = np.nansum(I_prof[window[0]:window[1]])
        V = np.nansum(V_prof[window[0]:window[1]])

        l = L/I
        v = V/I
        p = np.sqrt(l**2 + v**2)

        e_L = np.nanstd(L_prof[self.off_burst])
        e_I = np.nanstd(I_prof[self.off_burst])
        e_V = np.nanstd(V_prof[self.off_burst])

        e_l = np.sqrt((e_L/L)**2 + (e_I/I)**2) * l
        e_v = np.sqrt((e_V/V)**2 + (e_I/I)**2) * v 
        e_p = np.sqrt(
            (l/np.sqrt(l**2 + v**2) * e_l)**2 +
            (v/np.sqrt(l**2 + v**2) * e_v)**2
        )

        l_prof = L_prof/I_prof
        v_prof = V_prof/I_prof
        p_prof = np.sqrt(l_prof**2 + v_prof**2)

        return (l, e_l, l_prof), (v, e_v, v_prof), (p, e_p, p_prof)
        

# @memory.cache
def scrunch(
        x: np.ndarray, n: int, axis: int=0, verbose: bool=False, func=np.sum
    ) -> np.ndarray:
    """Scrunch data along axis (by defualt first axis) by a factor of n

    Will trim data if axis size not divisible by n

    :param x: data to scrunch
    :type x: array
    :param n: factor to scrunch by
    :type n: int
    :param axis: axis to scrunch along, defaults to 0
    :type axis: int, optional
    :param verbose: print warnings, defaults to False
    :type verbose: bool, optional
    """
    ndim = x.ndim

    if axis < 0:
        axis += ndim

    if axis < 0 or axis >= ndim:
        raise ValueError("Invalid axis")

    if n < 1:
        raise ValueError("Invalid scrunch factor")

    if n == 1:
        return x
    
    rem = x.shape[axis] % n
    if rem != 0:
        # trim data along axis
        if verbose:
            print(f"Trimming {rem} samples from axis {axis}")
        x = x.swapaxes(axis, 0)[:-rem].swapaxes(0, axis)

    shape = list(x.shape)

    if verbose:
        print(f"Shape: {shape}", end="")

    shape[axis] = shape[axis] // n

    if verbose:
        print(f" → {shape}")

    shape.insert(axis+1, n)
    return func(x.reshape(shape), axis=axis+1)


def get_freqs(f0: float, bw: float, nchan: int) -> np.ndarray:
    """Create array of frequencies.

    The returned array is the central frequency of `nchan` channels
    centred on `f0` with a bandwidth of `bw`.

    :param f0: Central frequency (arb. units, must be same as `bw`)
    :type f0: float
    :param bw: Bandwidth (arb. units, must be same as `f0`)
    :type bw: float
    :param nchan: Number of channels
    :type nchan: int
    :return: Central frequencies of `nchan` channels centred on `f0`
        over a bandwidth `bw`
    :rtype: :class:`np.ndarray`
    """
    fmin = f0 - bw / 2
    fmax = f0 + bw / 2

    chan_width = bw / nchan

    freqs = np.linspace(fmax, fmin, nchan, endpoint=False) + chan_width / 2

    return freqs


def dedisperse_coherent(
        spec: np.ndarray, DM: float, f0: float, bw: float, ref_func=np.min
) -> np.ndarray:
    """
    Coherently dedisperse the given complex spectrum.

    Coherent dedispersion is performed by applying the inverse of the
    transfer function that acts on radiation as it travels through a
    charged medium. This is detailed in Lorimer & Kramer's Handbook of
    Pulsar Astronomy (2005, Cambridge University Press).

    In practice, this is a frequency-dependent rotation of the complex
    spectrum. None of the amplitudes are altered.

    :param spec: Complex 1D-spectrum in a single polarisation
    :type spec: :class:`np.ndarray`
    :param DM: Dispersion measure to dedisperse to (pc/cm3)
    :type DM: float
    :param f0: Central frequency of the spectrum (MHz)
    :type f0: float
    :param bw: Bandwidth of the spectrum (MHz)
    :type bw: float
    :return: Coherently dedispersed complex spectrum
    :rtype: :class:`np.ndarray`
    """
    nchan = spec.shape[0]

    """
    This value of k_DM is not the most precise available. It is used 
    because to alter the commonly-used value would make pulsar timing 
    very difficult. Also, to quote Hobbs, Edwards, and Manchester 2006:
        ...ions and magnetic fields introduce a rather uncertain 
        correction of the order of a part in 10^5 (Spitzer 1962), 
        comparable to the uncertainty in some measured DM values...
    """
    k_DM = 2.41e-4

    freqs = get_freqs(f0, bw, nchan)

    f_ref = ref_func(freqs)

    dedisp_phases = np.exp(
        2j * np.pi * DM / k_DM * ((freqs - f_ref) ** 2 / f_ref ** 2 / freqs * 1e6)
    )

    new_spec = spec * dedisp_phases

    return new_spec


def generate_phases(
    DM: float, freqs, f_ref,
) -> np.ndarray:
    k_DM = 2.41e-4

    dedisp_phases = np.exp(
        2j * np.pi * DM / k_DM * ((freqs - f_ref) ** 2 / f_ref ** 2 / freqs * 1e6),
        dtype=np.complex128
    )

    return dedisp_phases


def generate_dynspec(
        t_ser: np.ndarray, nchan: int = 336, verbose: bool = False,
        label: str = "new"
    ) -> np.ndarray:
    """
    Creates a dynamic spectrum at the highest time resolution from the 
    given time series.

    :param t_ser: input time series of voltages
    :param nchan: number of frequency channels [Default = 336]
    :return: dynamic spectrum of voltages
    :rtype: :class:`np.ndarray`
    """
    dynspec = np.zeros(
        (int(t_ser.shape[0] / nchan), nchan), dtype=np.complex64
    )

    pbar = lambda x: (tqdm(x, desc=f"Generating {label} dynamic spectrum") if verbose 
                      else x)
    for i in pbar(range(int(t_ser.shape[0] / nchan))):
        dynspec[i, :] = np.fft.fft(t_ser[i * nchan : (i + 1) * nchan])

    return dynspec


def calculate_stokes(
        X: np.ndarray, 
        Y: np.ndarray, 
        delta_phi: np.ndarray = None, 
        normalise: bool = True,
        verbose: bool = False
    ) -> list:
    """Calculate Stokes parameters from X and Y arrays (time series or
    dynamic spectra)

    :param x: X polarisation array
    :type x: np.ndarray
    :param y: Y polarisation array
    :type y: np.ndarray
    :param delta_phi: Polarisation calibratoin solutions, 
                      defaults to None
    :type delta_phi: np.ndarray, optional
    :return: list of Stokes IQUV (in that order)
    :rtype: list
    """
    assert X.shape == Y.shape

    # lambda functions for each of the Stokes parameters
    stokes = {
        "I": lambda x, y: np.abs(x) ** 2 + np.abs(y) ** 2,
        "Q": lambda x, y: np.abs(x) ** 2 - np.abs(y) ** 2,
        "U": lambda x, y: 2 * np.real(np.conj(x) * y),
        "V": lambda x, y: 2 * np.imag(np.conj(x) * y),
    }

    stks = "IQUV"
    pars = []

    for stk in stks:
        if verbose:
            print(f"Calculating {stk}")
        par = stokes[stk](X, Y)
        if len(X.shape) == 2:   # dynamic spectrum -> normalise channels
            this_means, this_stds = get_norm(par)
            if stk == "I":
                stds = this_stds

            par = (par - this_means)/stds if normalise else par - this_means
        
        pars.append(par)

    # if not normalise:
    #     for i in range(len(pars)):
    #         pars[i] -= np.mean(pars[i])     #zero-mean Stokes

    if delta_phi is not None:
        if verbose:
            print("Applying polarisation calibration solutions")
        delta_phi_rpt = np.repeat(delta_phi[:, np.newaxis], X.shape[0], axis=1)

        cos_phi = np.cos(delta_phi_rpt)
        sin_phi = np.sin(delta_phi_rpt)

        U_prime = pars[2]
        V_prime = pars[3]

        # apply polcal solutions via rotation matrix
        U = U_prime * cos_phi - V_prime * sin_phi
        V = U_prime * sin_phi + V_prime * cos_phi

        pars = [pars[0], pars[1], U, V]
    
    return pars


def get_norm(ds):
    """
    Gets normalisation parameters to apply to dynamic spectra to
    normalise them

    :param ds: Input dynspec to normalise
    """
    T, F = ds.shape
    means = np.tile(np.mean(ds, axis=0), [T, 1])
    stds = np.tile(np.std(ds, axis=0), [T, 1])
    return means, stds



def calculate_g2(I, Q, U, V, binwidth, verbose=False, mode="direct"):
    # calculate g2(0) of data with bin size binwidth
    EI = np.zeros(len(I)//binwidth)
    g2 = np.zeros(len(I)//binwidth)
    p = np.zeros(len(I)//binwidth)
    expected = np.zeros(len(I)//binwidth)

    if mode == "direct":
        I_den = I
    elif "model" in mode:
        I_model = I.copy()
        I_model = np.mean(np.reshape(I_model, (-1, 336)), axis=1)
        if mode == "model":
            k = 1
        elif mode == "model2":
            k = 2
        elif mode == "model4":
            k = 4
        I_model = get_model(np.array([I_model]), k)[0]
        t_eval = np.arange(len(I))/len(I)
        t_data = np.arange(len(I_model))/len(I_model)
        I_den = np.interp(t_eval, t_data, I_model)
    else:
        raise ValueError("Invalid mode")

    pbar = lambda x: (tqdm(x, desc="Calculating g2", leave=False) if verbose else x)

    for i in pbar(range(1, len(g2)-1)):
        t0_num = int((i)*binwidth)
        t1_num = int((i+1)*binwidth)
        t0_den = t0_num-336//2 if "model" in mode else t0_num
        t1_den = t1_num-336//2 if "model" in mode else t1_num
        EI[i] = np.mean(I_den[t0_den:t1_den])
        g2[i] = (
            np.mean(I[t0_num:t1_num]**2) 
                / EI[i]**2
        )
        # pol frac
        p[i] = np.sqrt(
            Q[t0_num:t1_num].sum()**2 
            + U[t0_num:t1_num].sum()**2 
            + V[t0_num:t1_num].sum()**2
        ) / I[t0_num:t1_num].sum()

        expected[i] = 1.5 + 0.5 * p[i]**2

    res = (g2 - expected)/expected * 100

    EI[EI==0] = np.nan
    g2[g2==0] = np.nan
    p[p==0] = np.nan
    expected[expected==0] = np.nan

    return EI, g2, expected, res


# following two functions taken from Timothy Perrett's DM_optimisation
def get_model(I_data, k_mult):
    CI_data=dct(I_data, norm='ortho') #note "norm=ortho" to match MATLAB's dct
    dm_length,k_length=CI_data.shape

    #Low pass filter
    kc = get_kc(CI_data) * k_mult #cutoff k index
    O=3 #filter order
    k=np.linspace(1,k_length,k_length)
    #filter response
    fL=1/(1+(k/kc)**(2*O))

    #Pass DCT data through the combined filter to calculate structure parameter
    fL_diag=np.diag(fL) #make low-pass Filter into a diagonal matrix
    LPF_data=fL_diag@np.transpose(CI_data) #pass data through LPF
    return idct(np.transpose(LPF_data), norm='ortho') #smooth data    


def get_kc(CI_data: np.ndarray):
    """
    Determine spectral cutoff kc for the low-pass filter.
    
    :param CIdata: FRB spectrum in the discrete cosine domain, absolute values for all delta DM
    :type CIdata: :class:`np.ndarray`
    :return: Spectral cutoff `kc` for use in filtering the supplied FRB.
    :rtype: int
    """
    cumulative_window_size = 5
    noise_margin = int(CI_data.shape[1]/2) #assuming all of the higher frequency data is pure noise
    CI_data_transposed = np.transpose(np.abs(CI_data))
    CI_data_maxima = np.max(CI_data_transposed, axis = 1)
    kc = None

    #find the level where the noise flattens out
    noise_top = np.mean(CI_data_maxima[noise_margin:])

    #find where the signal dips down into the noise
    for i in range(cumulative_window_size,len(CI_data_transposed)):
        rolling_average=np.mean(CI_data_maxima[i-cumulative_window_size:i+1])
        if rolling_average <= noise_top:
            kc = i
            break
    
    # if kc is not None:
    #     print(f"Found kc of: {kc}")
    # else:
    #     raise RuntimeError('Could not find a value for kc')

    return kc