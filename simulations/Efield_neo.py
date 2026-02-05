# Simulation code all neocortical neuron models from Blue Brain Project

import os
import sys # If job split
from os.path import join

from glob import glob
import numpy as np

import neuron
import LFPy
import brainsignals.neural_simulations as ns    #From ElectricBrainSignals (Hagen and Ness 2023), see README

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


not_working_cells = []
not_working_plot_cells = []


root_folder = os.path.abspath('.')



def return_BBP_neuron(cell_name, tstop, dt): #Adapted from ElectricBrainSignals (Hagen and Ness 2023), see README

    # load some required neuron-interface files
    neuron.h.load_file("stdrun.hoc")
    neuron.h.load_file("import3d.hoc")

    CWD = os.getcwd()
    cell_folder = join(join(bbp_folder, cell_name))
    if not os.path.isdir(cell_folder):
        ns.download_BBP_model(cell_name)

    # neuron.load_mechanisms(bbp_mod_folder)
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

def run_Efield_stim_Ex(freq, 
                        neurons,
                        remove_list,
                        tstop, 
                        dt, 
                        cutoff,
                        job_nr = None,
                        local_E_field=1,  # V/m
                        ):
    
    if job_nr is not None:
        directory ='/mnt/SCRATCH/susandah/output/vmem_neo_25_nov'
        amp_data_filename = f'vmem_amp_data_neo_Ex_{job_nr}.npy'
    else:
        directory ='/Users/susannedahle/CellTypeDependenceElStim/simulation_data/vmem_data_neo'
        amp_data_filename = f'vmem_amp_data_neo_Ex.npy'
    
    amp_data_file_path = os.path.join(directory, amp_data_filename)

    # Initialize or load existing data
    if os.path.exists(amp_data_file_path):
        amp_data = np.load(amp_data_file_path, allow_pickle=True).item()
    else:
        amp_data = {}

    local_ext_pot = np.vectorize(lambda x, y, z: local_E_field * x / 1000)
    n_tsteps_ = int((tstop + cutoff) / dt + 1)
    t_ = np.arange(n_tsteps_) * dt
    #ns.compile_bbp_mechanisms(neurons[0])
    
    for neuron_idx, cell_name in enumerate(neurons):
        cell_failed_simulation = False
        cell_failed_plotting = False
        
        for f in freq:
            if check_existing_data(amp_data, cell_name, f):
                print(f"Skipping {cell_name} at {f} Hz (already exists in data)")
                continue
            
            try:
                cell = return_BBP_neuron(cell_name, tstop, dt)
                ns.remove_active_mechanisms(remove_list, cell)
                
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
                cut_tvec = cell.tvec[cell.tvec > 2000]
                cut_soma_vmem = cell.vmem[0, cell.tvec > 2000]
                freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_soma_vmem)
                freq_idx = np.argmin(np.abs(freqs - f))
                store_freq = freqs[freq_idx]
                soma_amp = vmem_amps[0, freq_idx]        

                # Store data in dictionary
                if cell_name not in amp_data:
                    amp_data[cell_name] = {
                        'freq': [],
                        'soma_amp': []
                    }
                
                amp_data[cell_name]['freq'].append(store_freq)
                amp_data[cell_name]['soma_amp'].append(soma_amp)
                
                # Save amp data to .npy file
                np.save(amp_data_file_path, amp_data)
                print(f"Amplitude data has been saved to {os.path.abspath(amp_data_file_path)}")
                
            except:
                cell_failed_simulation = True
                break
            
            del cell
            print(f"{f} Hz complete for {cell_name}")
        
        if cell_failed_simulation:
            not_working_cells.append(cell_name)
            print(f"Neuron {cell_name}, idx:{neuron_idx} does not work for simulation")
        elif cell_failed_plotting:
            not_working_plot_cells.append(cell_name)
            print(f"Neuron {cell_name}, idx:{neuron_idx} does not work for plotting")
        
        print(f"Simulation with E-field in x direction complete for Neuron nr.{neuron_idx+1} of {len(neurons)} neurons\n")


def run_Efield_stim_Ey(freq, 
                        neurons,
                        remove_list,
                        tstop, 
                        dt, 
                        cutoff,
                        job_nr = None,
                        local_E_field=1,  # V/m
                        ):

    if job_nr is not None:
        directory ='/mnt/SCRATCH/susandah/output/vmem_neo_25_nov'
        amp_data_filename = f'vmem_amp_data_neo_Ey_{job_nr}.npy'
    else:
        directory ='/Users/susannedahle/CellTypeDependenceElStim/simulation_data/vmem_data_neo'
        amp_data_filename = f'vmem_amp_data_neo_Ey.npy'
    
    amp_data_file_path = os.path.join(directory, amp_data_filename)
    
    # Initialize or load existing data
    if os.path.exists(amp_data_file_path):
        amp_data = np.load(amp_data_file_path, allow_pickle=True).item()
    else:
        amp_data = {}

    local_ext_pot = np.vectorize(lambda x, y, z: local_E_field * y / 1000)
    n_tsteps_ = int((tstop + cutoff) / dt + 1)
    t_ = np.arange(n_tsteps_) * dt
    #ns.compile_bbp_mechanisms(neurons[0])
    
    for neuron_idx, cell_name in enumerate(neurons):
        cell_failed_simulation = False
        cell_failed_plotting = False
        
        for f in freq:
            if check_existing_data(amp_data, cell_name, f):
                print(f"Skipping {cell_name} at {f} Hz (already exists in data)")
                continue
            
            try:
                cell = return_BBP_neuron(cell_name, tstop, dt)
                ns.remove_active_mechanisms(remove_list, cell)
                
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
                cut_tvec = cell.tvec[cell.tvec > 2000]
                cut_soma_vmem = cell.vmem[0, cell.tvec > 2000]
                freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_soma_vmem)
                freq_idx = np.argmin(np.abs(freqs - f))
                store_freq = freqs[freq_idx]
                soma_amp = vmem_amps[0, freq_idx]

                # Store data in dictionary
                if cell_name not in amp_data:
                    amp_data[cell_name] = {
                        'freq': [],
                        'soma_amp': []   
                    }
                
                amp_data[cell_name]['freq'].append(store_freq)
                amp_data[cell_name]['soma_amp'].append(soma_amp)

                # Save amp data to .npy file
                np.save(amp_data_file_path, amp_data)
                print(f"Amplitude data has been saved to {os.path.abspath(amp_data_file_path)}")
                
            except:
                cell_failed_simulation = True
                break
            
            
            del cell
            print(f"{f} Hz complete for {cell_name}")
        
        if cell_failed_simulation:
            not_working_cells.append(cell_name)
            print(f"Neuron {cell_name}, idx:{neuron_idx} does not work for simulation")
        elif cell_failed_plotting:
            not_working_plot_cells.append(cell_name)
            print(f"Neuron {cell_name}, idx:{neuron_idx} does not work for plotting")
        
        print(f"Simulation with E-field in y direction complete for Neuron nr.{neuron_idx+1} of {len(neurons)} neurons\n")


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


def run_passive_simulation_Ez(freq,
                              neurons,
                              remove_list,
                              tstop,
                              dt,
                              cutoff,
                              # job_nr, if splitted jobs during sim
                              local_E_field=1,  # V/m
                              directory='simulation_data/vmem_data_neo'):

    # amp_data_filename = f'vmem_amp_data_neo_Ez_{job_nr}.npy' # If splitted jobs

    local_ext_pot = np.vectorize(lambda x, y, z: local_E_field * z / 1000)
    n_tsteps_ = int((tstop + cutoff) / dt + 1)
    t_ = np.arange(n_tsteps_) * dt
    
    for neuron_idx, cell_name in enumerate(neurons):
        if not divmod(neuron_idx, size)[1] == rank:
            continue

        amp_data_filename = f'vmem_amp_data_neo_Ez_{cell_name}.npy'
        amp_data_file_path = os.path.join(directory, amp_data_filename)
        # plot_data_filename = f'plot_data_neo_{job_nr}.npy' # If splitted jobs
        plot_data_filename = f'plot_data_neo_{cell_name}.npy'
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

        cell_failed_simulation = False
        cell_failed_plotting = False
        
        for f_i, f in enumerate(freq):

            if check_existing_data(amp_data, cell_name, f):
                print(f"Skipping {cell_name} at {f} Hz (already exists in data)")
                continue
            
            try:

                cell = return_BBP_neuron(cell_name, tstop + cutoff, dt)
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

                cell = remove_active_mechanisms(remove_list, cell)
                # cell = remove_active_mechanisms([], cell)

                cell.simulate(rec_vmem=True)

                # Calculate soma amp with fourier
                cut_tvec = cell.tvec[cell.tvec > cutoff]
                cut_soma_vmem = cell.vmem[0, cell.tvec > cutoff]
                freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_soma_vmem)

                freq_idx = np.argmin(np.abs(freqs - f))

                if freqs[freq_idx] != f:
                    raise RuntimeError("Frequency not as expected!")

                soma_amp = vmem_amps[0, freq_idx]

                # Length and symmetry factor
                upper_z_endpoint = cell.z.mean(axis=-1)[cell.get_closest_idx(z=10000)]
                bottom_z_endpoint = cell.z.mean(axis=-1)[cell.get_closest_idx(z=-10000)]
                closest_z_endpoint = min(upper_z_endpoint, abs(bottom_z_endpoint))
                distant_z_endpoint = max(upper_z_endpoint, abs(bottom_z_endpoint))

                total_z_len = closest_z_endpoint + distant_z_endpoint
                symmetry_factor = closest_z_endpoint/distant_z_endpoint
                asymmetry_factor = abs(upper_z_endpoint - abs(bottom_z_endpoint))/abs(upper_z_endpoint + abs(bottom_z_endpoint))

                # Soma diam
                soma_diam = cell.d[0]

                # Dendrites in z-direction:
                n_dend_z_dir = 0
                tot_x_diam_abs = 0
                tot_y_diam_abs = 0
                tot_z_diam_abs = 0
                for idx in range(cell.totnsegs):
                    dz = cell.z[idx,0] - cell.z[idx,1]
                    dx = cell.x[idx,0] - cell.x[idx,1]
                    dy = cell.y[idx,0] - cell.y[idx,1]

                    if abs(dz) > abs(dx) and abs(dz) > abs(dy):
                        tot_z_diam_abs += cell.d[idx]
                        n_dend_z_dir += 1
                    elif abs(dx) > abs(dz) and abs(dx) > abs(dy):
                        tot_x_diam_abs += cell.d[idx]
                    elif abs(dy) > abs(dx) and abs(dy) > abs(dz):
                        tot_y_diam_abs += cell.d[idx]
                
                avg_z_diam = tot_z_diam_abs/n_dend_z_dir

                # Store data in dictionary
                if cell_name not in amp_data:
                    amp_data[cell_name] = {
                        'freq': [],
                        'soma_amp': [],
                        'upper_z_endpoint': upper_z_endpoint,
                        'bottom_z_endpoint': bottom_z_endpoint,
                        'total_len': total_z_len,
                        'symmetry_factor': symmetry_factor,
                        'asymmetry_factor': asymmetry_factor,
                        'soma_diam': soma_diam,
                        'avg_z_diam': avg_z_diam       
                    }

                amp_data[cell_name]['freq'].append(f)
                amp_data[cell_name]['soma_amp'].append(soma_amp)

                # Save amp data to .npy file
                np.save(amp_data_file_path, amp_data)
                print(f"Amplitude data has been saved to {os.path.abspath(amp_data_file_path)}")
                
            except NotImplementedError:
                cell_failed_simulation = True
                print("FAILED!")
                break
            
            if f in [10, 100, 1000]:
                try:
                    print(f"Storing data for selected frequency: {f} Hz")
                    amplitudes = []
                    for idx in range(cell.totnsegs):
                        cut_vmem = cell.vmem[idx, cell.tvec > 2000]
                        freqs, vmem_amps = ns.return_freq_and_amplitude(cut_tvec, cut_vmem)
                        freq_idx = np.argmin(np.abs(freqs - f))
                        amplitude = vmem_amps[0, freq_idx]
                        amplitudes.append(amplitude)

                    if cell_name not in plot_data:
                        plot_data[cell_name] = {
                            'freq': [],
                            'x': [],
                            'z': [],
                            'amplitudes': [],
                            'totnsegs': [],
                            'tvec': [],
                            'vmem': []
                        }

                    plot_data[cell_name]['freq'].append(f)
                    plot_data[cell_name]['x'].append(cell.x.tolist())
                    plot_data[cell_name]['z'].append(cell.z.tolist())
                    plot_data[cell_name]['amplitudes'].append(amplitudes)
                    plot_data[cell_name]['totnsegs'].append(cell.totnsegs)
                    plot_data[cell_name]['tvec'].append(cell.tvec.tolist())
                    plot_data[cell_name]['vmem'].append(cell.vmem[0].tolist())

                    # Save plot data to .npy file
                    np.save(plot_data_file_path, plot_data)
                    print(f"Plot data has been saved to {os.path.abspath(plot_data_file_path)}")

                except:
                    cell_failed_plotting = True
                    break
            
            del cell
            print(f"{f} Hz complete for {cell_name}")
        
        if cell_failed_simulation:
            not_working_cells.append(cell_name)
            print(f"Neuron {cell_name}, idx:{neuron_idx} does not work for simulation")
        elif cell_failed_plotting:
            not_working_plot_cells.append(cell_name)
            print(f"Neuron {cell_name}, idx:{neuron_idx} does not work for plotting")
        
        print(f"Neuron nr.{neuron_idx+1} of {len(neurons)} neurons complete \n")


if __name__=='__main__':

    ns.load_mechs_from_folder(ns.cell_models_folder)

    bbp_mod_folder = join(join(ns.cell_models_folder, "bbp_mod"))
    ns.load_mechs_from_folder(bbp_mod_folder)

    h = neuron.h

    all_cells_folder = join(root_folder, 'all_cells_folder') #From the Blue Brain Project (Markram et al. 2015), see README
    bbp_folder = os.path.abspath(all_cells_folder)

    cell_models_folder = join(root_folder, 'brainsignals', 'cell_models') #From ElectricBrainSignals (Hagen and Ness 2023), see README
    bbp_mod_folder = join(cell_models_folder, "bbp_mod")

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
    #ns.compile_bbp_mechanisms()  # Compile once, before running jobs

    remove_list = ["Ca_HVA", "Ca_LVAst", "Ca", "CaDynamics_E2", 
                   "Ih", "Im", "K_Pst", "K_Tst", "KdShu2007", "Nap_Et2",
                   "NaTa_t", "NaTs2_t", "SK_E2", "SKv3_1", "StochKv"]
    # Simulation time 
    tstop = 1000.
    dt = 2**-6

    cutoff = 2000
    import scipy.fftpack as ff

    n_tsteps_ = int((tstop) / dt)
    t_ = np.arange(n_tsteps_) * dt
    # sample_freq = ff.fftfreq(n_tsteps_, d=dt/1000)
    # pidxs = np.where(sample_freq >= 0)
    # freqs = sample_freq[pidxs]


    # print(freqs)

    # El field frequency 
    freq1 = np.arange(1, 10, 1) # Shorter steplength in beginning
    freq2 = np.arange(10, 100, 10)
    freq3 = np.arange(100, 2200, 100) # Longer steplength to save calculation time
    freq = sorted(np.concatenate((freq1, freq2, freq3)))
    #freq = np.array([2, 20, 40, 60, 200, 400, 600, 1200, 1400, 1600, 2000])
    # freq = np.logspace(0, 2.2, 10)

    print(freq)
    # Simulation for the first neuron, full list of neurons computationally expencive, reccomend to split like shown below
    run_passive_simulation_Ez(freq, neurons, remove_list, tstop, dt, cutoff, directory="sim_results")
    # run_passive_simulation_Ex(freq, neurons[:1], remove_list, tstop, dt, cutoff)
    # run_passive_simulation_Ey(freq, neurons[:1], remove_list, tstop, dt, cutoff)

