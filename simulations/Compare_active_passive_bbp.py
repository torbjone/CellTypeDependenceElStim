import os
import sys
from os.path import join

from glob import glob
import numpy as np

import neuron
import LFPy
import brainsignals.neural_simulations as ns    #From ElectricBrainSignals (Hagen and Ness 2023), see README

not_working_cells = []
not_working_plot_cells = []

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

root_folder = os.path.abspath('.')


import scipy.fftpack as ff
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


def return_BBP_neuron(cell_name, tstop, dt):    #Adpted froam ElectricBrainSignals (Hagen and Ness 2023), see README

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
                             v_init = -65)
    os.chdir(CWD)
    # set view as in most other examples
    cell.set_rotation(x=np.pi / 2)
    return cell


def check_existing_data(data_dict, cell_name, frequency):
    if cell_name in data_dict:
        if frequency in data_dict[cell_name]['freq']:
            return True
    return False


def run_Efield_stim_Ez(freq,
                        neurons,
                        remove_list,
                        tstop,
                        dt,
                        cutoff,
                        local_E_field=1,  # V/m
                        ):

    directory ='.'
    amp_data_filename = f'vmem_amp_data_active_passive_bbp_Ih_test.npy'
    plot_data_filename = f'plot_data_active_passive_Ih_test.npy'

    amp_data_file_path = os.path.join(directory, amp_data_filename)
    plot_data_file_path = os.path.join(directory, plot_data_filename)
    
    # Initialize or load existing data
    if os.path.exists(amp_data_file_path):
        amp_data = np.load(amp_data_file_path, allow_pickle=True).item()
    else:
        amp_data = {}

    if os.path.exists(plot_data_file_path):
        plot_data = np.load(plot_data_file_path, allow_pickle=True).item()
    else:
        plot_data = {}

    local_ext_pot = np.vectorize(lambda x, y, z: local_E_field * z / 1000)
    n_tsteps_ = int((tstop + cutoff) / dt + 1)
    t_ = np.arange(n_tsteps_) * dt
    #ns.compile_bbp_mechanisms(neurons[0]) # Compile once, before running jobs simultaneously
    
    for neuron_idx, cell_name in enumerate(neurons):
        cell_failed_simulation = False
        cell_failed_plotting = False

        cell_mechs = ['active', 'passive', "no_Ih"]

        for cell_idx in range(len(cell_mechs)):
            cell_mech = cell_mechs[cell_idx]
            cell_name_and_mech = f'{cell_name}_{cell_mech}'
        
            for f in freq:
                if check_existing_data(amp_data, cell_name_and_mech, f):
                    print(f"Skipping {cell_name_and_mech} at {f} Hz (already exists in data)")
                    continue
                
                # try:
                cell = return_BBP_neuron(cell_name, tstop + cutoff, dt)
                if cell_mech == 'passive':
                    ns.remove_active_mechanisms(remove_list, cell)
                if cell_mech == 'no_Ih':
                    ns.remove_active_mechanisms(["Ih", "Im"], cell)

                cell.extracellular = True
                for sec in cell.allseclist:
                    sec.insert("extracellular")

                base_pot = local_ext_pot(
                    cell.x.mean(axis=-1),
                    cell.y.mean(axis=-1),
                    cell.z.mean(axis=-1)
                ).reshape(cell.totnsegs, 1)

                pulse = np.sin(2 * np.pi * f * t_ / 1000)
                v_cell_ext = base_pot * pulse.reshape(1, n_tsteps_)

                cell.insert_v_ext(v_cell_ext, t_)
                cell.simulate(rec_vmem=True)

                # Calculate soma amp with fourier
                cut_tvec = cell.tvec[cell.tvec > cutoff]
                cut_soma_vmem = cell.vmem[0, cell.tvec > cutoff]
                freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_soma_vmem)
                freq_idx = np.argmin(np.abs(freqs - f))
                fourier_freq = freqs[freq_idx]
                soma_amp = vmem_amps[0, freq_idx]

                # # Store data in dictionary
                # if cell_name not in amp_data:
                #     amp_data[cell_name] = {
                #         'freq': [],
                #         'soma_amp': [],
                #     }

                # amp_data[cell_name]['freq'].append(store_freq)
                # amp_data[cell_name]['soma_amp'].append(soma_amp)
                if cell_name_and_mech not in amp_data:
                    amp_data[cell_name_and_mech] = {
                        'freq': [],
                        'fourier_freq': [],
                        'soma_amps': [],
                        'soma_vmem': [],
                        'tvec': [],
                    }

                amp_data[cell_name_and_mech]['freq'].append(f)
                amp_data[cell_name_and_mech]['fourier_freq'].append(fourier_freq)
                amp_data[cell_name_and_mech]['soma_amps'].append(soma_amp)
                amp_data[cell_name_and_mech]['tvec'].append(cut_tvec)
                amp_data[cell_name_and_mech]['soma_vmem'].append(cut_soma_vmem)


                # Save amp data to .npy file
                np.save(amp_data_file_path, amp_data)
                print(f"Amplitude data has been saved to {os.path.abspath(amp_data_file_path)}")

                # except:
                #     cell_failed_simulation = True
                #     break
            
                if f in [10, 100, 1000]:
                    try:
                        print(f"Storing data for selected frequency: {f} Hz")
                        amplitudes = []
                        for idx in range(cell.totnsegs):
                            cut_vmem = cell.vmem[idx, cell.tvec > cutoff]
                            freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_vmem)
                            freq_idx = np.argmin(np.abs(freqs - f))
                            amplitude = vmem_amps[0, freq_idx]
                            amplitudes.append(amplitude)

                        if cell_name_and_mech not in plot_data:
                            plot_data[cell_name_and_mech] = {
                                'freq': [],
                                'x': [],
                                'z': [],
                                'amplitudes': [],
                                'totnsegs': [],
                                'tvec': [],
                                'vmem': []
                            }

                        plot_data[cell_name_and_mech]['freq'].append(f)
                        plot_data[cell_name_and_mech]['x'].append(cell.x.tolist())
                        plot_data[cell_name_and_mech]['z'].append(cell.z.tolist())
                        plot_data[cell_name_and_mech]['amplitudes'].append(amplitudes)
                        plot_data[cell_name_and_mech]['totnsegs'].append(cell.totnsegs)
                        plot_data[cell_name_and_mech]['tvec'].append(cell.tvec.tolist())
                        plot_data[cell_name_and_mech]['vmem'].append(cell.vmem[0].tolist())

                        # Save plot data to .npy file
                        np.save(plot_data_file_path, plot_data)
                        print(f"Plot data has been saved to {os.path.abspath(plot_data_file_path)}")
                        
                    except:
                        cell_failed_plotting = True
                        break
                
                del cell
                print(f"{f} Hz complete for {cell_name_and_mech}")
        
        if cell_failed_simulation:
            not_working_cells.append(cell_name)
            print(f"Neuron {cell_name}, idx:{neuron_idx} does not work for simulation")
        elif cell_failed_plotting:
            not_working_plot_cells.append(cell_name)
            print(f"Neuron {cell_name}, idx:{neuron_idx} does not work for plotting")
        
        print(f"Neuron nr.{neuron_idx+1} of {len(neurons)} neurons complete \n")


if __name__=='__main__':

    h = neuron.h

    # List to store the neuron names
    #neurons = ['L4_BP_bIR215_5', "L5_MC_bAC217_1", "L5_TTPC2_cADpyr232_3", "L5_NGC_bNAC219_5", 'L4_SS_cADpyr230_1']
    neurons = ["L5_NGC_bNAC219_5"]

    remove_list = ["Ca_HVA", "Ca_LVAst", "Ca", "CaDynamics_E2", 
                   "Ih", "Im", "K_Pst", "K_Tst", "KdShu2007", "Nap_Et2",
                   "NaTa_t", "NaTs2_t", "SK_E2", "SKv3_1", "StochKv"]
    
    tstop = 1000.
    dt = 2**-6
    cutoff = 6000

    rate = 5000 # * Hz

    # Common setup

    num_tsteps = int(tstop / dt + 1)
    tvec = np.arange(num_tsteps) * dt
    #t0_idx = np.argmin(np.abs(tvec - cutoff))

    freq1 = np.arange(1, 10, 1) # Shorter steplength in beginning

    #freq2 = np.arange(10, 100, 10)
    #freq3 = np.arange(100, 2200, 100) # Longer steplength to save calculation time
    #freqs = sorted(np.concatenate((freq1, freq2, freq3)))
    run_Efield_stim_Ez(freq1, neurons, remove_list, tstop, dt, cutoff)


