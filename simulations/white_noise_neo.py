import os
import sys # If split jobs
from os.path import join

from glob import glob
import numpy as np

import neuron
import LFPy
import brainsignals.neural_simulations as ns #From ElectricBrainSignals (Hagen and Ness 2023), see README

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


import scipy.fftpack as ff
root_folder = os.path.abspath('.')

ns.load_mechs_from_folder(ns.cell_models_folder)
np.random.seed(1534)


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
                             v_init = -65)
    os.chdir(CWD)
    # set view as in most other examples
    cell.set_rotation(x=np.pi / 2)
    return cell


def get_dipole_transformation_matrix(cell): #From LFPy v.2.3.5
        return np.stack([cell.x.mean(axis=-1),
                         cell.y.mean(axis=-1),
                         cell.z.mean(axis=-1)])

def get_positive_dipole_transformation_matrix(cell):
    # Compute mean z-position per segment
    z_mean = cell.z.mean(axis=-1)

    # Get indices where z > 0
    pos_indices = np.where(z_mean > 0)[0]

    # Stack only the positive-z segments
    response_matrix = np.stack([
        cell.x.mean(axis=-1)[pos_indices],
        cell.y.mean(axis=-1)[pos_indices],
        cell.z.mean(axis=-1)[pos_indices]
    ])

    return response_matrix, pos_indices


def get_negative_dipole_transformation_matrix(cell):
    # Compute mean z-position per segment
    z_mean = cell.z.mean(axis=-1)

    # Get indices where z > 0
    neg_indices = np.where(z_mean < 0)[0]

    # Stack only the positive-z segments
    response_matrix = np.stack([
        cell.x.mean(axis=-1)[neg_indices],
        cell.y.mean(axis=-1)[neg_indices],
        cell.z.mean(axis=-1)[neg_indices]
    ])

    return response_matrix, neg_indices


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


def check_existing_data(cdm_data, cell_name):
    if cell_name in cdm_data:
        if cell_name in cdm_data.keys():
            return True
    return False  

def find_closest_indices(target_freqs, available_freqs):
    return [np.argmin(np.abs(available_freqs - tf)) for tf in target_freqs]


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


def run_white_noise_stim(freqs, 
                         neurons,
                         tstop, dt, cutoff
                         ):

    directory = os.path.join(root_folder, 'sim_results')
    n_tsteps_ = int((tstop + cutoff) / dt + 1)
    t_ = np.arange(n_tsteps_) * dt

    for neuron_idx, cell_name in enumerate(neurons):
        if not divmod(neuron_idx, size)[1] == rank:
            continue

        cdm_data_filename = f'cdm_data_neocortical_{cell_name}.npy'
        plot_imem_filename = f'plot_imem_{cell_name}.npy'

        cdm_data_file_path = os.path.join(directory, cdm_data_filename)
        plot_imem_file_path = os.path.join(directory, plot_imem_filename)
        failed_cells = []

        # Initialize or load existing data
        if os.path.exists(cdm_data_file_path):
            cdm_data = np.load(cdm_data_file_path, allow_pickle=True).item()

        else:
            cdm_data = {}

        # Initialize or load existing data
        if os.path.exists(plot_imem_file_path):
            plot_imem_data = np.load(plot_imem_file_path, allow_pickle=True).item()
        else:
            plot_imem_data = {}

        if check_existing_data(cdm_data, cell_name):
            print(f"Skipping {cell_name} (already exists in data)")
            continue

        try:
            cell = return_BBP_neuron(cell_name, tstop=tstop + cutoff, dt=dt)

            # Insert noise
            cell, syn, noise_vec = make_white_noise_stimuli(cell, 0, freqs, t_)
            cell = remove_active_mechanisms(remove_list, cell)

            # Run simulation
            cell.simulate(rec_imem=True, rec_vmem=True)

            #cut_tvec = cell.tvec[cell.tvec > cutoff]
            cell.vmem = cell.vmem[:, cell.tvec > cutoff]
            cell.imem = cell.imem[:, cell.tvec > cutoff]
            input_current = np.array(noise_vec)[cell.tvec > cutoff]

            # Cut initial segment
            #cell.vmem = cell.vmem[:, t0_idx:]
            #cell.imem = cell.imem[:, t0_idx:]
            cell.tvec = cell.tvec[cell.tvec > cutoff]
            cell.tvec -= cell.tvec[0]

            # ----- Compute dipole moment (z-component) -----
            cdm = get_dipole_transformation_matrix(cell) @ cell.imem
            cdm = cdm[2, :] # 2: z-cordinate, : all timestep

            # Get frequency and amplitude of cdm
            freqs_s, amp_cdm_s, phase_cdm_s = return_freq_amp_phase(cell.tvec, cdm)
            
            #Find amplitude of input currents

            freqs_input, amp_input_current, phase_input_current = return_freq_amp_phase(cell.tvec, input_current)
            cdm_per_input_current = amp_cdm_s / amp_input_current

            target_freqs = sorted(np.concatenate((np.arange(1, 10, 1), np.arange(10, 100, 10), np.arange(100, 2200, 100))))
            closest_indices = find_closest_indices(target_freqs, freqs_s)

            matched_freqs = freqs_s[closest_indices]
            matched_amp_cdm = amp_cdm_s[0, closest_indices]
            matched_phase_cdm = phase_cdm_s[0, closest_indices]
            matched_cdm_per_input_current = cdm_per_input_current[0, closest_indices]
            matched_amp_input_currens = amp_input_current[0,closest_indices]
            matched_phases_input_currens = phase_input_current[0,closest_indices]

            # ----- Compute Positive dipole moment (z-component) -----
            Tm_pos, pos_idx = get_positive_dipole_transformation_matrix(cell)
            cdm_pos = Tm_pos @ cell.imem[pos_idx, :]
            cdm_pos = cdm_pos[2, :] # 2: z-cordinate, : all timestep

            # Get frequency and amplitude of cdm
            freqs_s_pos, amp_cdm_s_pos, phase_cdm_s_pos = return_freq_amp_phase(cell.tvec, cdm_pos)

            cdm_per_input_current_pos = amp_cdm_s_pos / amp_input_current

            # The frequencies from FFT will be the same so use closest_indices for total cdm
            matched_amp_cdm_pos = amp_cdm_s_pos[0, closest_indices]
            matched_phase_cdm_pos = phase_cdm_s_pos[0, closest_indices]
            matched_cdm_per_input_current_pos = cdm_per_input_current_pos[0, closest_indices]

            # ----- Compute Negative dipole moment (z-component) -----
            Tm_neg, neg_idx = get_negative_dipole_transformation_matrix(cell)
            cdm_neg = Tm_neg @ cell.imem[neg_idx, :]
            cdm_neg = cdm_neg[2, :] # 2: z-cordinate, : all timestep

            # Get frequency and amplitude of cdm
            freqs_s_neg, amp_cdm_s_neg, phase_cdm_s_neg = return_freq_amp_phase(cell.tvec, cdm_neg)
            
            cdm_per_input_current_neg = amp_cdm_s_neg / amp_input_current

            matched_amp_cdm_neg = amp_cdm_s_neg[0, closest_indices]
            matched_phase_cdm_neg = phase_cdm_s_neg[0, closest_indices]
            matched_cdm_per_input_current_neg = cdm_per_input_current_neg[0, closest_indices]

            # ----- Morph properties -----               
            upper_z_endpoint = cell.z.mean(axis=-1)[cell.get_closest_idx(z=10000)]
            bottom_z_endpoint = cell.z.mean(axis=-1)[cell.get_closest_idx(z=-10000)]
            closest_z_endpoint = min(upper_z_endpoint, abs(bottom_z_endpoint))
            distant_z_endpoint = max(upper_z_endpoint, abs(bottom_z_endpoint))
            total_len_z_direction = closest_z_endpoint + distant_z_endpoint
            symmetry_factor_z_direction = closest_z_endpoint/distant_z_endpoint
            asymmetry_factor = abs(upper_z_endpoint - abs(bottom_z_endpoint))/upper_z_endpoint + abs(bottom_z_endpoint)

            soma_diam = cell.d[0]

            tot_z_diam_abs = 0
            numb_z_comp_abs = 0
            for idx in range(cell.totnsegs):
                dz = cell.z[idx,0] - cell.z[idx,1]
                dx = cell.x[idx,0] - cell.x[idx,1]
                dy = cell.y[idx,0] - cell.y[idx,1]

                if abs(dz) > abs(dx) and abs(dz) > abs(dy):
                    numb_z_comp_abs += 1
                    tot_z_diam_abs += cell.d[idx]
                    
            avg_z_diam = tot_z_diam_abs/numb_z_comp_abs

            # --- CALCULATION OF IMEM RETURN POSITION ---

            # 1. Calculate imem amplitudes and phases at each target frequency for each segment
            imem_amps_at_target_freqs = np.zeros((cell.totnsegs, len(matched_freqs)))
            imem_phases_at_target_freqs = np.zeros((cell.totnsegs, len(matched_freqs)))

            # The frequencies from FFT will be the same for all segments, so get them once
            freqs_imem, _ = ns.return_freq_and_amplitude(cell.tvec, cell.imem[0, :])
            imem_freq_indices = find_closest_indices(matched_freqs, freqs_imem)

            for idx in range(cell.totnsegs):
                imem_seg = cell.imem[idx, :]
                _, imem_amps, imem_phases = return_freq_amp_phase(cell.tvec, imem_seg)
                imem_amps_at_target_freqs[idx, :] = imem_amps[0, imem_freq_indices]
                imem_phases_at_target_freqs[idx, :] = imem_phases[0, imem_freq_indices]

            z_coords = cell.z.mean(axis=-1)
            soma_z_pos = z_coords[0]  # Assuming soma is at index 0

            # Get indices for segments above and below the soma once
            above_indices = np.where(z_coords > soma_z_pos)[0]
            below_indices = np.where(z_coords < soma_z_pos)[0]

            # ---- Average Imem possition AMPLITUDES ----
            avg_return_pos_above_soma_freq = []
            avg_return_pos_below_soma_freq = []
            sum_possition_times_imem_above = []
            sum_possition_times_imem_below = []

            for f_idx in range(len(matched_freqs)):
                current_amps_at_freq = imem_amps_at_target_freqs[:, f_idx]

                # For currents above the soma
                if len(above_indices) > 0:
                    amps_above = current_amps_at_freq[above_indices]
                    pos_above = z_coords[above_indices]
                    total_amp_above = np.sum(amps_above)
                    if total_amp_above > 1e-12:  # Avoid division by zero
                        avg_pos = np.sum(pos_above * amps_above) / total_amp_above
                        avg_return_pos_above_soma_freq.append(avg_pos)
                        sum_above = np.sum(pos_above * amps_above)
                        sum_possition_times_imem_above.append(sum_above)
                    else:
                        avg_return_pos_above_soma_freq.append(0)
                        sum_possition_times_imem_above.append(0)
                else:
                    avg_return_pos_above_soma_freq.append(0)
                    sum_possition_times_imem_above.append(0)

                # For currents below the soma
                if len(below_indices) > 0:
                    amps_below = current_amps_at_freq[below_indices]
                    pos_below = z_coords[below_indices]
                    total_amp_below = np.sum(amps_below)
                    if total_amp_below > 1e-12: # Avoid division by zero
                        avg_pos = np.sum(pos_below * amps_below) / total_amp_below
                        avg_return_pos_below_soma_freq.append(avg_pos)
                        sum_below = np.sum(pos_below * amps_below)
                        sum_possition_times_imem_below.append(sum_below)
                    else:
                        avg_return_pos_below_soma_freq.append(0)
                        sum_possition_times_imem_below.append(0)
                else:
                    avg_return_pos_below_soma_freq.append(0)
                    sum_possition_times_imem_below.append(0)

            # ---- Average Imem possition VALUES AT FREQ COMPONENT ---- 
            avg_return_pos_above_soma_wave_initial_value = []
            avg_return_pos_below_soma_wave_initial_value = []
            avg_return_pos_above_soma_wave_quarter_period_value = []
            avg_return_pos_below_soma_wave_quarter_period_value = []

            for f_idx, freq in enumerate(matched_freqs):
                imem_amps = imem_amps_at_target_freqs[:, f_idx]
                imem_phases = imem_phases_at_target_freqs[:, f_idx]

                # Compute wave values for each segment
                imem_wave_initial = imem_amps * np.sin(imem_phases)  # t = 0
                imem_wave_quarter = imem_amps * np.sin(2 * np.pi * freq * (1 / (4 * freq)) + imem_phases)  # t = 1/4 period
                # Simplify expression:
                # imem_wave_quarter = imem_amps * np.sin(np.pi/2 + imem_phases)

                # --- Above soma ---
                if len(above_indices) > 0:
                    # Initial wave value average
                    wave_above_initial = imem_wave_initial[above_indices]
                    pos_above = z_coords[above_indices]
                    total_wave_above_initial = np.sum(wave_above_initial)
                    if np.abs(total_wave_above_initial) > 1e-12:
                        avg_pos_initial = np.sum(pos_above * wave_above_initial) / total_wave_above_initial
                        avg_return_pos_above_soma_wave_initial_value.append(avg_pos_initial)
                    else:
                        avg_return_pos_above_soma_wave_initial_value.append(0)

                    # Quarter-period wave value average
                    wave_above_quarter = imem_wave_quarter[above_indices]
                    total_wave_above_quarter = np.sum(wave_above_quarter)
                    if np.abs(total_wave_above_quarter) > 1e-12:
                        avg_pos_quarter = np.sum(pos_above * wave_above_quarter) / total_wave_above_quarter
                        avg_return_pos_above_soma_wave_quarter_period_value.append(avg_pos_quarter)
                    else:
                        avg_return_pos_above_soma_wave_quarter_period_value.append(0)
                else:
                    avg_return_pos_above_soma_wave_initial_value.append(0)
                    avg_return_pos_above_soma_wave_quarter_period_value.append(0)

                # --- Below soma ---
                if len(below_indices) > 0:
                    # Initial wave value average
                    wave_below_initial = imem_wave_initial[below_indices]
                    pos_below = z_coords[below_indices]
                    total_wave_below_initial = np.sum(wave_below_initial)
                    if np.abs(total_wave_below_initial) > 1e-12:
                        avg_pos_initial = np.sum(pos_below * wave_below_initial) / total_wave_below_initial
                        avg_return_pos_below_soma_wave_initial_value.append(avg_pos_initial)
                    else:
                        avg_return_pos_below_soma_wave_initial_value.append(0)

                    # Quarter-period wave value average
                    wave_below_quarter = imem_wave_quarter[below_indices]
                    total_wave_below_quarter = np.sum(wave_below_quarter)
                    if np.abs(total_wave_below_quarter) > 1e-12:
                        avg_pos_quarter = np.sum(pos_below * wave_below_quarter) / total_wave_below_quarter
                        avg_return_pos_below_soma_wave_quarter_period_value.append(avg_pos_quarter)
                    else:
                        avg_return_pos_below_soma_wave_quarter_period_value.append(0)
                else:
                    avg_return_pos_below_soma_wave_initial_value.append(0)
                    avg_return_pos_below_soma_wave_quarter_period_value.append(0)

            # Store data in dictionary
            cdm_data[cell_name] = {
                'frequency': matched_freqs,
                'cdm': matched_amp_cdm,
                'cdm_phase': matched_phase_cdm, 
                'cdm_per_input_current': matched_cdm_per_input_current,
                'cdm_pos': matched_amp_cdm_pos,
                'cdm_phase_pos': matched_phase_cdm_pos, 
                'cdm_per_input_current_pos': matched_cdm_per_input_current_pos,
                'cdm_neg': matched_amp_cdm_neg,
                'cdm_phase_neg': matched_phase_cdm_neg, 
                'cdm_per_input_current_neg': matched_cdm_per_input_current_neg,
                'input_current_amp': matched_amp_input_currens,
                'input_current_phase': matched_phases_input_currens, 
                'closest_z_endpoint': closest_z_endpoint,
                'distant_z_endpoint': distant_z_endpoint,
                'upper_z_endpoint': upper_z_endpoint,
                'bottom_z_endpoint': bottom_z_endpoint,
                'total_len': total_len_z_direction,
                'symmetry_factor': symmetry_factor_z_direction,
                'asymmetry_factor': asymmetry_factor,
                'soma_diam': soma_diam,
                'tot_z_diam': tot_z_diam_abs,
                'avg_z_diam': avg_z_diam, 
                'avg_return_pos_above_soma': avg_return_pos_above_soma_freq,
                'avg_return_pos_below_soma': avg_return_pos_below_soma_freq,
                'sum_possition_times_imem_above': sum_possition_times_imem_above,
                'sum_possition_times_imem_below': sum_possition_times_imem_below, 
                'avg_return_pos_above_soma_wave_initial_value': avg_return_pos_above_soma_wave_initial_value,
                'avg_return_pos_below_soma_wave_initial_value': avg_return_pos_below_soma_wave_initial_value,
                'avg_return_pos_above_soma_wave_quarter_period_value': avg_return_pos_above_soma_wave_quarter_period_value,
                'avg_return_pos_below_soma_wave_quarter_period_value': avg_return_pos_below_soma_wave_quarter_period_value
            }


            # Save amp data to .npy file
            np.save(cdm_data_file_path, cdm_data)
            print(f"Amplitude data has been saved to {os.path.abspath(cdm_data_file_path)}")

            try:
                # Store plot data of imem amplitudes at 10, 100, 1000 Hz
                plot_imem_amplitudes_at_freqs = []
                plot_imem_phases_at_freqs = []
                plot_freqs = [10, 100, 1000]

                for idx in range(cell.totnsegs):
                    imem_seg = cell.imem[idx, :]
                    freqs_imem, imem_amps, imem_phase = return_freq_amp_phase(cell.tvec, imem_seg)

                    # Extract amplitudes for the plot frequencies
                    segment_amplitudes = []
                    segment_phases = []
                    for f in plot_freqs:
                        freq_idx = np.argmin(np.abs(freqs_imem - f))
                        amplitude = imem_amps[0, freq_idx]
                        phase = imem_phase[0, freq_idx]
                        segment_amplitudes.append(amplitude)
                        segment_phases.append(phase)

                    plot_imem_amplitudes_at_freqs.append(segment_amplitudes)
                    plot_imem_phases_at_freqs.append(segment_phases)

                # Store in dictionary
                plot_imem_data[cell_name] = {
                    'freqs': plot_freqs,
                    'x': cell.x.tolist(),
                    'z': cell.z.tolist(),
                    'totnsegs': cell.totnsegs,
                    'tvec': cell.tvec.tolist(),
                    'imem_amps': plot_imem_amplitudes_at_freqs,  # Shape: (totnsegs, len(plot_freqs))
                    'imem_phases': plot_imem_phases_at_freqs,    # Shape: (totnsegs, len(plot_freqs))
                }

                # Save plot data to .npy file
                np.save(plot_imem_file_path, plot_imem_data)
                print(f"Amplitude data has been saved to {os.path.abspath(plot_imem_file_path)}")
            except Exception as e_plot: 
                print(f'Cell failed to store plot data due to error {e_plot}')

            del cell, cdm, freqs_s, amp_cdm_s 
        
            print(f'Simulation complete for neuron {cell_name}: nr.{neuron_idx+1} of {len(neurons)}')

        except Exception as e:
            print(f"Skipping neuron {cell_name} due to error: {e}")
            failed_cells.append(cell_name)
            continue

    # Save failed cells
    if failed_cells:
        failed_path = os.path.join(directory, f"failed_cells_cdm.npy")
        np.save(failed_path, np.array(failed_cells))
        print(f"Saved list of failed cells to: {failed_path}")


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
    
    tstop = 1000.
    dt = 2**-6
    cutoff = 2000

    rate = 5000 # * Hz

    # Common setup
    num_tsteps = int(tstop / dt + 1)
    tvec = np.arange(num_tsteps) * dt
    #t0_idx = np.argmin(np.abs(tvec - cutoff))

    freq1 = np.arange(1, 10, 1) # Shorter steplength in beginning
    freq2 = np.arange(10, 100, 10)
    freq3 = np.arange(100, 2200, 100) # Longer steplength to save calculation time
    freqs = sorted(np.concatenate((freq1, freq2, freq3)))
    #cdm_amp_dict = {}  # To store amplitude spectra for each cell
    #imem_amp_dict = {}

    run_white_noise_stim(freqs, neurons, tstop, dt, cutoff)
