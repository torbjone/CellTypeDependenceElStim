"""
Inject weak (subthreshold) white-noise current into the soma of every human
spiny perisomatic Allen cell model, compute the current dipole moment, and plot
|FFT(P_z)| for all models on one figure -- to find the cell with the largest
current-dipole-moment response to somatic current input.

Usage:
    python white_noise_human_perisomatic.py            # all models
    TEST_N=2 python white_noise_human_perisomatic.py   # only first 2 (quick test)
"""
import os
import sys
import ssl
import csv
import json
import shutil
import zipfile
import subprocess
from os.path import join
from urllib.request import urlopen

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.fftpack as ff
import neuron
import LFPy
from lfpykit import CurrentDipoleMoment
import brainsignals.neural_simulations as ns

HERE = os.path.dirname(os.path.abspath(__file__))
ctrl_fig_folder = join(HERE, "human_pz_figs")
os.makedirs(ctrl_fig_folder, exist_ok=True)
ns.load_mechs_from_folder(ns.cell_models_folder)   # ISyn white-noise point process
allen_folder = ns.allen_folder
_loaded_mod_folders = set()

# ---------------------------------------------------------------------------
# Allen model loader (same as the notebook)
# ---------------------------------------------------------------------------
def _nrnivmodl_bin():
    cand = join(os.path.dirname(sys.executable), "nrnivmodl")
    if os.path.isfile(cand):
        return cand
    exe = shutil.which("nrnivmodl")
    if exe and os.path.isfile(exe):
        return exe
    raise RuntimeError("nrnivmodl not found")


def download_allen_model(model_id):
    model_id = str(model_id)
    zip_path = join(allen_folder, f"neuronal_model_{model_id}.zip")
    out_dir = join(allen_folder, f"neuronal_model_{model_id}")
    url = f"https://api.brain-map.org/neuronal_model/download/{model_id}"
    print("  downloading", model_id)
    u = urlopen(url, context=ssl._create_unverified_context())
    with open(zip_path, "wb") as f:
        f.write(u.read())
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    os.remove(zip_path)
    return out_dir


def return_allen_cell_model(model_id, dt, tstop, make_passive=False):
    model_id = str(model_id)
    model_folder = join(allen_folder, f"neuronal_model_{model_id}")
    if not os.path.isdir(model_folder):
        download_allen_model(model_id)

    # All Allen perisomatic models share the same mechanism set (Ca_HVA, NaTs,
    # ...), so we only need to compile+load it once (from the first model);
    # loading a second identically-named set errors.
    mod_folder = join(model_folder, "modfiles")
    if not hasattr(neuron.h, "NaTs"):
        if not os.path.isdir(join(mod_folder, "x86_64")):
            print("  compiling mechanisms ...")
            cwd = os.getcwd()
            os.chdir(mod_folder)
            try:
                subprocess.run([_nrnivmodl_bin()], check=True,
                               stdout=subprocess.DEVNULL)
            finally:
                os.chdir(cwd)
        neuron.load_mechanisms(mod_folder)

    params = json.load(open(join(model_folder, "fit_parameters.json")))
    manifest = json.load(open(join(model_folder, "manifest.json")))
    morph_file = join(model_folder, "reconstruction.swc")
    model_type = manifest["biophys"][0]["model_type"]

    Ra = params["passive"][0]["ra"]
    if model_type == "Biophysical - perisomatic":
        e_pas = params["passive"][0]["e_pas"]
        cms = params["passive"][0]["cm"]

    neuron.h.celsius = params["conditions"][0]["celsius"]
    reversal_potentials = params["conditions"][0]["erev"]
    active_mechs = params["genome"]

    cell_parameters = {
        'morphology': morph_file,
        'v_init': -70,
        'passive': False,
        'nsegs_method': 'fixed_length',
        'max_nsegs_length': 10.,
        'dt': dt,
        'tstart': -100,
        'tstop': tstop,
        'pt3d': True,
        'extracellular': True,
        'custom_code': [join(allen_folder, 'remove_axon.hoc')],
    }
    cell = LFPy.Cell(**cell_parameters)

    if make_passive and model_type != "Biophysical - perisomatic":
        raise RuntimeError("make_passive only implemented for perisomatic models")

    for sec in neuron.h.allsec():
        sec.insert("pas")
        sectype = sec.name().split("[")[0]
        if model_type == "Biophysical - perisomatic":
            sec.e_pas = e_pas
            for cm_dict in cms:
                if cm_dict["section"] == sectype:
                    exec("sec.cm = {}".format(cm_dict["cm"]))
        sec.Ra = Ra

        for sec_dict in active_mechs:
            if sec_dict["section"] == sectype:

                if sec_dict["mechanism"] == "":
                    # This is the passive mechanism
                    if sec_dict["name"] != "g_pas":
                        raise RuntimeError("Something wrong with model building function!")
                    exec("sec.{} = {}".format(sec_dict["name"], sec_dict["value"]))
                else:
                    if not make_passive:
                        if not sec.has_membrane(sec_dict["mechanism"]):
                            sec.insert(sec_dict["mechanism"])
                        exec("sec.{} = {}".format(sec_dict["name"], sec_dict["value"]))
        if not make_passive:
            for sec_dict in reversal_potentials:
                if sec_dict["section"] == sectype:
                    for key in sec_dict.keys():
                        if not key == "section":
                            exec("sec.{} = {}".format(key, sec_dict[key]))

    neuron.h.secondorder = 0
    ns.align_cell_to_axes(cell)
    return cell


# ---------------------------------------------------------------------------
# White-noise soma stimulus (flat-amplitude sum of sinusoids, random phase).
# Same construction as ElectricBrainSignals (Hagen & Ness 2023).
# ---------------------------------------------------------------------------
def make_white_noise_stimuli(cell, input_idx, freqs, tvec, input_scaling, rng):
    I = np.zeros(len(tvec))
    for freq in freqs:
        I += np.sin(2 * np.pi * freq * tvec / 1000. + 2 * np.pi * rng.random())
    input_array = input_scaling * I
    noise_vec = neuron.h.Vector(input_array)
    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if syn is None:
        raise RuntimeError("bad input index")
    syn.dur = 1e9
    syn.delay = 0
    noise_vec.play(syn._ref_amp, cell.dt)
    return syn, noise_vec, input_array


def amp_spectrum(tvec, sig):
    timestep = (tvec[1] - tvec[0]) / 1000.
    sample_freq = ff.fftfreq(len(sig), d=timestep)
    pidx = np.where(sample_freq >= 0)
    freqs = sample_freq[pidx]
    Y = ff.fft(sig)[pidx]
    return freqs, np.abs(Y) / len(sig)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
dt = 2**-4
tstop = 1000.        # ms
t0 = 1000.            # discard transient (ms)  -> 1000 ms analysis window (df = 1 Hz)
stim_freqs = np.arange(1., 1000.)     # white-noise band (Hz)
input_scaling = 0.004                # nA per sinusoid (weak / subthreshold)
input_idx = 0                        # soma
# explicit stimulus time vector (cell.tvec only exists after simulate); make it
# long enough to cover the whole run, including the tstart=-100 ms settling.
stim_tvec = np.arange(0, tstop + t0, dt)

csv_path = join(HERE, "allen_human_spiny_perisomatic.csv")
rows = list(csv.DictReader(open(csv_path)))
test_n = int(os.environ.get("TEST_N", "0"))
if test_n:
    rows = rows[:test_n]

results = {}
for k, row in enumerate(rows):
    mid = row["model_id"]
    name = row["name"]
    print(f"[{k+1}/{len(rows)}] model {mid} ({name}, {row['apical']})", flush=True)
    try:
        cell = return_allen_cell_model(mid, dt, tstop + t0 - dt, make_passive=True)
        rng = np.random.RandomState(1234)   # identical input waveform for every cell
        syn, nvec, iarr = make_white_noise_stimuli(cell, input_idx, stim_freqs,
                                                   stim_tvec, input_scaling, rng)
        cell.simulate(rec_vmem=True, rec_imem=True)

        cdm = CurrentDipoleMoment(cell).get_transformation_matrix() @ cell.imem
        t0_idx = np.argmin(np.abs(cell.tvec - t0))
        pz = cdm[2, t0_idx:]
        fr, amp = amp_spectrum(cell.tvec[t0_idx:], pz)

        amp = amp[(fr > 0) & (fr < 1000)]
        fr = fr[(fr > 0) & (fr < 1000)]

        fig = plt.figure()
        ax1 = fig.add_subplot(121, aspect=1)
        ax2 = fig.add_subplot(322, xlabel="time (ms)", ylabel="Vm (mV)")
        ax3 = fig.add_subplot(324, xlabel="time (ms)", ylabel="Pz (nAµm)")
        ax4 = fig.add_subplot(326, xlabel="frequency (Hz)", ylabel="nAµm")

        ax1.plot(cell.x.T, cell.z.T, c='k')
        ax2.plot(cell.tvec[t0_idx:], cell.vmem[0, t0_idx:], c='k')
        ax3.plot(cell.tvec[t0_idx:], cdm[2, t0_idx:], c='k', lw=1)
        ax3.plot(cell.tvec[t0_idx:], cdm[0, t0_idx:], c='gray', lw=0.5)
        ax3.plot(cell.tvec[t0_idx:], cdm[1, t0_idx:], c='cyan', lw=0.5)
        ax4.loglog(fr, amp, c='k')
        fig.savefig(join(ctrl_fig_folder, f"white_noise_pz_{mid}_passive2.png"), dpi=150)
        plt.close(fig)

        vmax = float(cell.somav.max())
        results[mid] = {"name": name, "apical": row["apical"], "structure": row["structure"],
                        "freqs": fr, "amp_pz": amp, "pz_rms": float(np.std(pz)),
                        "vmax": vmax, "spiking": vmax > -40.}
        print(f"    P_z RMS = {np.std(pz):.4g} nA um   somaVmax = {vmax:.1f} mV"
              f"{'  *** SPIKING ***' if vmax > -40 else ''}", flush=True)
        cell.__del__()
    except Exception as e:
        print("    FAILED:", repr(e), flush=True)

np.save(join(HERE, "white_noise_pz_results_passive2.npy"), results, allow_pickle=True)

# ---------------------------------------------------------------------------
# Plot: |FFT(P_z)| for all models; highlight the largest responders
# ---------------------------------------------------------------------------
ranking = sorted(results.items(), key=lambda kv: -kv[1]["pz_rms"])
print("\n=== ranking by P_z RMS (largest dipole response first) ===")
for rank, (mid, r) in enumerate(ranking, 1):
    print(f"{rank:2d}. {mid}  {r['name']:24s}  P_z RMS={r['pz_rms']:.4g}"
          f"  Vmax={r['vmax']:.1f}{'  SPIKING' if r['spiking'] else ''}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlabel("frequency (Hz)")
ax.set_ylabel(r"|FFT($P_z$)|  (nA$\cdot\mu$m)")
ax.set_title("Dipole-moment response to weak somatic white-noise current\n"
             "(human spiny perisomatic models)")
for mid, r in ranking:
    ax.loglog(r["freqs"][1:], r["amp_pz"][1:], color="0.8", lw=0.6, zorder=1)
colors = plt.cm.viridis(np.linspace(0, 0.9, 3))
for j, (mid, r) in enumerate(ranking[:3]):
    ax.loglog(r["freqs"][1:], r["amp_pz"][1:], color=colors[j], lw=1.5, zorder=3,
              label=f"#{j+1} {mid} (RMS={r['pz_rms']:.3g})")
ax.set_xlim(1, 500)
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(join(HERE, "white_noise_pz_fft_passive2.png"), dpi=150)
print("\nsaved white_noise_pz_fft.png and white_noise_pz_results_passive2.npy")
