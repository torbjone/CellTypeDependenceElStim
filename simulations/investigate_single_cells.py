import os
import sys # If split jobs
from os.path import join

from glob import glob
import numpy as np

import neuron
import LFPy
import brainsignals.neural_simulations as ns #From ElectricBrainSignals (Hagen and Ness 2023), see README
from brainsignals.plotting_convention import mark_subplots, simplify_axes
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


import scipy.fftpack as ff
root_folder = os.path.abspath('.')

ns.load_mechs_from_folder(ns.cell_models_folder)
np.random.seed(1534)
remove_list = ["Ca_HVA", "Ca_LVAst", "Ca", "CaDynamics_E2",
               "Ih", "Im", "K_Pst", "K_Tst", "KdShu2007", "Nap_Et2",
               "NaTa_t", "NaTs2_t", "SK_E2", "SKv3_1", "StochKv"]

ns.load_mechs_from_folder(ns.cell_models_folder)

bbp_mod_folder = join(join(ns.cell_models_folder, "bbp_mod"))
ns.load_mechs_from_folder(bbp_mod_folder)

h = neuron.h

all_cells_folder = join(root_folder,
                        'all_cells_folder')  # From the Blue Brain Project (Markram et al. 2015), see README
bbp_folder = os.path.abspath(all_cells_folder)

cell_models_folder = join(root_folder, 'brainsignals',
                          'cell_models')  # From ElectricBrainSignals (Hagen and Ness 2023), see README
bbp_mod_folder = join(cell_models_folder, "bbp_mod")


def return_BBP_neuron(cell_name, tstop, dt):

    # load some required neuron-interface files
    neuron.h.load_file("stdrun.hoc")
    neuron.h.load_file("import3d.hoc")

    CWD = os.getcwd()
    cell_folder = join(join(bbp_folder, cell_name))
    if not os.path.isdir(cell_folder):
        ns.download_BBP_model(cell_name)

    neuron.load_mechanisms(bbp_mod_folder)
    os.chdir(cell_folder)
    add_synapses = False
    # get the template name
    f = open("template.hoc", 'r')
    templatename = ns.get_templatename(f)
    f.close()

    # get biophys template name
    f = open("biophysics.hoc", 'r')
    biophysics = ns.get_templatename(f)
    f.close()

    # get morphology template name
    f = open("morphology.hoc", 'r')
    morphology = ns.get_templatename(f)
    f.close()

    # get synapses template name
    f = open(ns.posixpth(os.path.join("synapses", "synapses.hoc")), 'r')
    synapses = ns.get_templatename(f)
    f.close()

    neuron.h.load_file('constants.hoc')

    if not hasattr(neuron.h, morphology):
        """Create the cell model"""
        # Load morphology
        neuron.h.load_file(1, "morphology.hoc")
    if not hasattr(neuron.h, biophysics):
        # Load biophysics
        neuron.h.load_file(1, "biophysics.hoc")
    if not hasattr(neuron.h, synapses):
        # load synapses
        neuron.h.load_file(1, ns.posixpth(os.path.join('synapses', 'synapses.hoc')
                                       ))
    if not hasattr(neuron.h, templatename):
        # Load main cell template
        neuron.h.load_file(1, "template.hoc")

    templatefile = ns.posixpth(os.path.join(cell_folder, 'template.hoc'))

    morphologyfile = glob(os.path.join('morphology', '*'))[0]


    # Instantiate the cell(s) using LFPy
    cell = LFPy.TemplateCell(morphology=morphologyfile,
                             templatefile=templatefile,
                             templatename=templatename,
                             templateargs=1 if add_synapses else 0,
                             tstop=tstop,
                             dt=dt,
                             lambda_f = 500,
                             nsegs_method='lambda_f',
                             pt3d=True,
                             v_init = -65)
    os.chdir(CWD)
    # set view as in most other examples
    cell.set_rotation(x=np.pi / 2)
    return cell

def remove_active_mechanisms(remove_list, cell):
    # remove_list = ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
    # "SK_E2", "K_Tst", "K_Pst",
    # "Im", "Ih", "CaDynamics_E2", "Ca_LVAst", "Ca", "Ca_HVA"]

    import sys

    mt = h.MechanismType(0)
    mname = h.ref('')
    for sec in h.allsec():
        for mech in remove_list:
            mt.select(mech)
            mt.selected(mname)
            if mname[0] == mech:
                # print("Try to remove: ", mname[0])
                mt.remove(sec=sec)

    # for sec in h.allsec():
    #     print(sec.name(), sec.psection()["density_mechs"].keys())

    return cell
def make_white_noise_stimuli(cell, input_idx, freqs, tvec, input_scaling=0.005): #From ElectricBrainSignals (Hagen and Ness 2023), see README

    I = np.zeros(len(tvec))

    for freq in freqs:
        I += np.sin(2 * np.pi * freq * tvec/1000. + 2*np.pi*np.random.random())
    input_array = input_scaling * I

    noise_vec = neuron.h.Vector(input_array)

    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print("Input inserted in ", sec.name())
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if syn is None:
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noise_vec.play(syn._ref_amp, cell.dt)
    return cell, syn, noise_vec

def return_freq_amp_phase(tvec, sig):
    """ Returns the amplitude, frequency and phase of the signal"""
    import scipy.fftpack as ff
    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")
    timestep = (tvec[1] - tvec[0])/1000. if type(tvec) in [list, np.ndarray] else tvec
    sample_freq = ff.fftfreq(sig.shape[1], d=timestep)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]

    Y = ff.fft(sig, axis=1)[:, pidxs[0]]

    amplitude = np.abs(Y)/Y.shape[1]
    phase = np.angle(Y)
    return freqs, amplitude, phase



# List to store the neuron names
neurons = []

# Check if the directory exists
if os.path.exists(all_cells_folder):
    # Iterate over the directories in the all_cells_folder
    for folder_name in os.listdir(all_cells_folder):
        folder_path = os.path.join(all_cells_folder, folder_name)
        if os.path.isdir(folder_path):
            neurons.append(folder_name)
else:
    print(f"The directory {all_cells_folder} does not exist.")
# ns.compile_bbp_mechanisms()  # Compile once, before running jobs

def get_dipole_transformation_matrix(cell):  # From LFPy v.2.3.5
    return np.stack([cell.x.mean(axis=-1),
                     cell.y.mean(axis=-1),
                     cell.z.mean(axis=-1)])

tstop = 1000.
dt = 2 ** -4
cutoff = 1000


rate = 5000  # * Hz

# Common setup
num_tsteps = int(tstop / dt)

tvec = np.arange(num_tsteps) * dt
t_ = np.arange(int( (tstop + cutoff) / dt + 1)) * dt
# t0_idx = np.argmin(np.abs(tvec - cutoff))

cell_name = 'L4_BP_bIR215_5'
cell_name = "L5_MC_bAC217_1"
cell_names = ['L4_BP_bIR215_5', "L5_MC_bAC217_1"]

sample_freq = ff.fftfreq(len(tvec), d=dt / 1000)

pidxs = np.where(sample_freq >= 0)
stim_freqs = sample_freq[pidxs]

homog_pas = False
# homog_diam = True
make_passive = True

short_cell_names = {
    'L4_BP_bIR215_5': 'L4 BP',
    "L5_MC_bAC217_1": "L5 MC",
    "L5_TTPC2_cADpyr232_3": "L5 TTPC",
    "L5_NGC_bNAC219_5": "L5 NGC",
    'L4_SS_cADpyr230_1': 'L4 SS'
}

import matplotlib.pyplot as plt
fig = plt.figure(figsize=[10, 4])
fig.subplots_adjust(wspace=0.5, right=0.98, left=0.04, bottom=0.12, top=0.85)
#fig.suptitle(f"cell name: {cell_name}\nmake passive: {make_passive}\n homogeneous passive parms: {homog_pas}\n homogeneous diameter: {homog_diam}")
ax0 = fig.add_subplot(141, aspect=1, frameon=False, xticks=[], yticks=[],
                      ylim=[-1100, 300])

axd = fig.add_subplot(142, xlabel="diameter (µm)", ylabel="$z$ (µm)",
                      ylim=[-1100, 300], xlim=[0, 8], title="variable (original) diameter")
axd2 = fig.add_subplot(143, xlabel="diameter (µm)", ylabel="$z$ (µm)",
                      ylim=[-1100, 300], xlim=[0, 8], title="uniform diameter")

#ax1 = fig.add_subplot(243, xlabel="time (ms)", ylabel="soma $V_m$ (mV)")
#ax2 = fig.add_subplot(247, xlabel="time (ms)", ylabel="$P_z$ (nAµm)")

ax3 = fig.add_subplot(144, ylim=[1e-3, 0.04], xlim=[1, 2000],
                      xlabel="frequency (Hz)", ylabel="amplitude $P_z$")

cell_labels = {}
cell_vmem_colors = {
    'L4_BP_bIR215_5': "tab:grey",
    "L5_MC_bAC217_1": "tab:cyan",
    "L5_TTPC2_cADpyr232_3": "tab:olive",
    "L5_NGC_bNAC219_5": 'tab:brown',
    'L4_SS_cADpyr230_1': 'tab:pink'
}

lines = []
line_names = []

for c_idx, cell_name in enumerate(cell_names):
    for homog_diam in [True, False]:
        homog_name = "uniform dendritic diameter" if homog_diam else "variable dendritic diameter"
        sim_name = f"{short_cell_names[cell_name]}, {homog_name}"

        line_style = '-' if not homog_diam else ':'
        marker_style = 'o' if c_idx == 0 else 'x'

        clr = cell_vmem_colors[cell_name]

        cell = return_BBP_neuron(cell_name, tstop=tstop + cutoff, dt=dt)


        for sec in h.allsec():
            if "soma" in sec.name():
                d_ = 7.
            else:
                d_ = 1.
            for i in range(int(h.n3d(sec=sec))):
                if homog_diam:
                    pass
                else:
                    d_ = h.diam3d(i, sec=sec) #* scale
                h.pt3dchange(i, d_, sec=sec)

        h.define_shape()
        cell._collect_geometry()

        if homog_pas:
            for sec in cell.allseclist:
                for seg in sec:
                    seg.g_pas = 1e-5
                    seg.cm = 1
                    sec.e_pas = -70
                sec.Ra = 100

        ax0.plot(cell.x.T.copy() + 300 * c_idx, cell.z.T.copy(), c=clr)
        if homog_diam:
            ax_ = axd2
        else:
            ax_ = axd
        ax_.plot(cell.d, cell.z.mean(axis=1), ls='None', ms=6, marker=marker_style, c=clr)

        print(f"Soma diam, mean diam: {cell_name, cell.d[0], np.mean(cell.d[1:])}")

        # freq1 = np.arange(1, 10, 5)  # Shorter steplength in beginning
        # freq2 = np.arange(10, 100, 50)
        # freq3 = np.arange(100, 2200, 500)  # Longer steplength to save calculation time

        # freqs = sorted(np.concatenate((freq1, freq2, freq3)))

        # Insert noise
        cell, syn, noise_vec = make_white_noise_stimuli(cell, 0,
                                                        stim_freqs, t_, 0.0001)
        if make_passive:
            cell = remove_active_mechanisms(remove_list, cell)

        print(h.psection())

        # Run simulation
        cell.simulate(rec_imem=True, rec_vmem=True)

        # cut_tvec = cell.tvec[cell.tvec > cutoff]
        cell.vmem = cell.vmem[:, cell.tvec > cutoff]
        cell.imem = cell.imem[:, cell.tvec > cutoff]
        input_current = np.array(noise_vec)[cell.tvec > cutoff]
        cell.tvec = cell.tvec[cell.tvec > cutoff]
        cell.tvec -= cell.tvec[0]

        # ----- Compute dipole moment (z-component) -----
        cdm = get_dipole_transformation_matrix(cell) @ cell.imem
        cdm = cdm[2, :]  # 2: z-cordinate, : all timestep

        # Get frequency and amplitude of cdm
        freqs_s, amp_cdm_s, phase_cdm_s = return_freq_amp_phase(cell.tvec, cdm)

        #ax1.plot(cell.tvec, cell.vmem[0])
        #ax2.plot(cell.tvec, cdm)
        l, = ax3.loglog(freqs_s[1:], amp_cdm_s[0, 1:], ls=line_style, c=clr)

        lines.append(l)
        line_names.append(sim_name)

        del cell


fig.legend(lines, line_names, ncol=2, loc=(0.5,0.9), frameon=False)
simplify_axes(fig.axes)

plt.savefig(f"cdm_psd_comb_plot.pdf")