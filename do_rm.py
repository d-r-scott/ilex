import ilex as il
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

parser = ArgumentParser()
parser.add_argument("frb", help="FRB name")
parser.add_argument("--DM", help="DM", required=True, type=str)
parser.add_argument("-w", help="Boxcar width", required=True, type=int)
parser.add_argument("--crop_dur", help="Crop duration", default=200, type=int)
parser.add_argument("--idx", help="Index of peak", default=None, type=int)
args = parser.parse_args()
print(args.frb)
f = il.FRB(args.frb, tavg=args.w, favg=4, crop_dur=args.crop_dur, verbose=True)


# if args.frb == "220725":
#         f.peak = int(2217 * 1000)
#         f.apply_crop(force_peak=2217)
#         f.set_crossing_freq()
#         f.zap_above_freq(f.crossing_freq)
        
f.dedisperse(args.DM, normalise=False)


# flagging
# f.flag_dynspec([0])
f.flag_dynspec()

# plt.figure()
# plt.plot(np.nansum(f.I_ds, axis=1))
# plt.show()

if args.idx is not None:
    peak = args.idx
else:
    peak = np.argmax(np.nansum(f.I_ds, axis=1))
# idxs = range(peak-10, peak+31)
# RMs = []
# RM_errs = []

# w = args.w // 10
# boxcar = np.ones(w)/w
# I_conv = np.convolve(np.nansum(f.I_ds, axis=1), boxcar, mode="same")
# box_start = np.argmax(I_conv) - w//2
# box_end = box_start + w

# f.I_ds[peak] = np.nansum(f.I_ds[box_start:box_end], axis=0)
# f.Q_ds[peak] = np.nansum(f.Q_ds[box_start:box_end], axis=0)
# f.U_ds[peak] = np.nansum(f.U_ds[box_start:box_end], axis=0)

# for i in tqdm(idxs):
res = f.calc_RM(t_idx=peak)#t_idx=i)
    # RMs += [res[0]]
    # RM_errs += [res[1]]

if not os.path.exists(f"{f.name}"):
    os.mkdir(f"{f.name}")

# np.savetxt(f"{f.name}/RM.txt", np.array([f.t[idxs], RMs, RM_errs]).T)

# fig, axs = plt.subplots(nrows=2, sharex=True)
# axs[0].plot(f.t, np.nansum(f.I_ds, axis=1))
# axs[1].errorbar(f.t[idxs], RMs, yerr=RM_errs, fmt="o")
# axs[1].set_xlim(f.t[idxs[0]], f.t[idxs[-1]])
# plt.savefig(f"{f.name}/RM.png")

# # exit()

print(f"{f.name}\n" \
    f"RM = {res[0]} +/- {res[1]} rad/m^2\n" \
    f"PA0 = {res[2]} +/- {res[3]} deg"  
)

with open(f"{f.name}/RM_single.txt", "w") as f:
    f.write(f"{res[0]} +/- {res[1]} rad/m^2\n"
            f"{res[2]} +/- {res[3]} deg"
    )
