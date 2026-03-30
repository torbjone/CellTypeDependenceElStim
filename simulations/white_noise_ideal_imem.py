import neuron
import LFPy
import numpy as np

import os
from os.path import join
from glob import glob
import brainsignals.neural_simulations as ns #From ElectricBrainSignals (Hagen and Ness 2023), see README
import scipy.fftpack as ff

h = neuron.h
ns.load_mechs_from_folder(ns.cell_models_folder)


def return_ideal_cell(tstop, dt, apic_soma_diam = 20, apic_dend_diam=2,
                      apic_upper_len = 1000, apic_bottom_len = -200):

    tot_n_segs = 599


    nsegs_up = 500#int(tot_n_segs * np.abs(apic_upper_len) / (np.abs(apic_upper_len) + np.abs(apic_bottom_len)) )
    nsegs_down = 100#tot_n_segs - nsegs_up


    if np.abs(apic_upper_len / nsegs_up) - np.abs(apic_bottom_len / nsegs_down) > 1e-2:
        print(apic_upper_len / nsegs_up, apic_bottom_len / nsegs_down)
        raise RuntimeError("Dendritic compartments not equally long!")

    #Adapted from ElectricBrainSignals (Hagen and Ness 2023), see README
    h("forall delete_section()")
    h("""
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create soma[1], dend[2]

    proc topol() { local i
      basic_shape()
      connect dend[0](0), soma(1)
      connect dend[1](0), soma(0)
    }
    proc basic_shape() {
      soma[0] {pt3dclear()
      pt3dadd(0, 0, -10., %s)
      pt3dadd(0, 0, 10., %s)}
      dend[0] {pt3dclear()
      pt3dadd(0, 0, 10., %s)
      pt3dadd(0, 0, %s, %s)}
      dend[1] {pt3dclear()
      pt3dadd(0, 0, -10., %s)
      pt3dadd(0, 0, %s, %s)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        soma[0] all.append()
        dend[0] all.append()
        dend[1] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    soma[0] {nseg = 1}
    dend[0] {nseg = %s}
    dend[1] {nseg = %s}
    }
    proc biophys() {
    }
    celldef()

    Rm = 30000.

    forall {
        insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
        g_pas = 1 / Rm
        Ra = 150.
        cm = 1.
        }
    """ % (apic_soma_diam, apic_soma_diam, apic_dend_diam, apic_upper_len,
           apic_dend_diam, apic_dend_diam, apic_bottom_len, apic_dend_diam, nsegs_up, nsegs_down))
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
                'max_nsegs_length': 20,
                'dt': dt,
                'tstart': -100.,
                'tstop': tstop,
                'pt3d': True,
            }
    cell = LFPy.Cell(**cell_params)
    # cell.set_pos(x=-cell.xstart[0])
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

def check_existing_data(multipole_data, cell_name):
    if cell_name in multipole_data:
        if cell_name in multipole_data.keys():
            return True
    return False  

def return_freq_amp_phase(tvec, sig):
    """ Returns the amplitude and frequency of the input signal"""
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

def run_white_noise_imem(tstop,
                         dt,
                         freqs,
                         freqs_limit, 
                         soma_diam, dend_diam, upper_len, bottom_len,
                         tvec,
                         t0_idx,
                         imem_data_filename='plot_imem_data_ideal',
                         directory='sim_results'
                        ):
    
    imem_data_filename = f'{imem_data_filename}.npy'
    imem_data_file_path = os.path.join(directory, imem_data_filename)
    
    # Initialize or load existing data
    if os.path.exists(imem_data_file_path):
        imem_data = np.load(imem_data_file_path, allow_pickle=True).item()
    else:
        imem_data = {}

    i = 0

    for bot_l in bottom_len:
        for up_l in upper_len:
            for s_d in soma_diam:
                for d_d in dend_diam:

                    if d_d > s_d:
                        continue

                    i += 1
                    cell_name = f'BL_{bot_l}_UL_{up_l}_SD_{s_d}_DD_{d_d}'

                    # if check_existing_data(imem_data, cell_name):
                    #     print(f"Skipping {cell_name} (already exists in data)")
                    #     continue

                    cell = return_ideal_cell(tstop, dt,
                                             apic_soma_diam=s_d,
                                             apic_dend_diam=d_d,
                                             apic_upper_len=up_l,
                                             apic_bottom_len=bot_l)

                    print(f"Running wn simulation with {cell_name}", flush=True)

                    # White noise stimulus
                    cell, syn, noise_vec = make_white_noise_stimuli(
                        cell,
                        input_idx=0,
                        freqs=freqs[freqs < freqs_limit],
                        tvec=tvec
                    )

                    # Run simulation
                    cell.simulate(rec_imem=True, rec_vmem=True)

                    # Trim pre-t0
                    cell.vmem = cell.vmem[:, t0_idx:]
                    cell.imem = cell.imem[:, t0_idx:]
                    cell.tvec = cell.tvec[t0_idx:] - cell.tvec[t0_idx]

                    # Store data for target frequencies
                    target_freqs = [1, 5,10,50,100,500,1000]

                    cdm = get_dipole_transformation_matrix(cell) @ cell.imem
                    cdm = cdm[2, :] # 2: z-cordinate, : all timestep
                    freqs_cdm, amp_cdm, phase_cdm = return_freq_amp_phase(cell.tvec, cdm)
                    print(freqs_cdm)

                    cdm_amps = []
                    cdm_phases = []
                    for f in target_freqs:
                        freq_idx = np.argmin(np.abs(freqs_cdm - f))
                        amplitude = amp_cdm[0, freq_idx]
                        phase = phase_cdm[0, freq_idx]
                        cdm_amps.append(amplitude)
                        cdm_phases.append(phase)

                    # Store p_z amplitudes 
                    Tm_pos, pos_idx = get_positive_dipole_transformation_matrix(cell)
                    cdm_pos = Tm_pos @ cell.imem[pos_idx, :]
                    cdm_pos = cdm_pos[2, :]
                    freqs_cdm_pos, amp_cdm_pos, phase_cdm_pos = return_freq_amp_phase(cell.tvec, cdm_pos)
                    cdm_pos_amps = []
                    cdm_pos_phases = []
                    for f in target_freqs:
                        freq_idx = np.argmin(np.abs(freqs_cdm_pos - f))
                        amplitude = amp_cdm_pos[0, freq_idx]
                        phase = phase_cdm_pos[0, freq_idx]
                        cdm_pos_amps.append(amplitude)
                        cdm_pos_phases.append(phase)
                    
                    Tm_neg, neg_idx = get_negative_dipole_transformation_matrix(cell)
                    cdm_neg = Tm_neg @ cell.imem[neg_idx, :]
                    cdm_neg = cdm_neg[2, :]
                    freqs_cdm_neg, amp_cdm_neg, phase_cdm_neg = return_freq_amp_phase(cell.tvec, cdm_neg)
                    cdm_neg_amps = []
                    cdm_neg_phases = []
                    for f in target_freqs:
                        freq_idx = np.argmin(np.abs(freqs_cdm_neg - f))
                        amplitude = amp_cdm_neg[0, freq_idx]
                        phase = phase_cdm_neg[0, freq_idx]
                        cdm_neg_amps.append(amplitude)
                        cdm_neg_phases.append(phase)
                    
                    input_current = np.array(noise_vec)
                    input_current = input_current[t0_idx:len(cdm)+t0_idx]
                    freqs_input, amps_input_current, phases_input_current = return_freq_amp_phase(cell.tvec, input_current)
                    input_amps = []
                    input_phases = []
                    for f in target_freqs:
                        freq_idx = np.argmin(np.abs(freqs_input - f))
                        amplitude = amps_input_current[0, freq_idx]
                        phase = phases_input_current[0, freq_idx]
                        input_amps.append(amplitude)
                        input_phases.append(phase)

                    # Store imem amplitudes
                    imem_amplitudes_at_freqs = []
                    imem_phases_at_freqs = []
                    for idx in range(cell.totnsegs):
                        imem_seg = cell.imem[idx, :]
                        freqs_imem, imem_amps, imem_phases = return_freq_amp_phase(cell.tvec, imem_seg)

                        # Extract amplitudes for the target frequencies
                        segment_amplitudes = []
                        segment_phases = []
                        for f in target_freqs:
                            freq_idx = np.argmin(np.abs(freqs_imem - f))
                            amplitude = imem_amps[0, freq_idx]
                            phase = imem_phases[0, freq_idx]
                            segment_amplitudes.append(amplitude)
                            segment_phases.append(phase)

                        imem_amplitudes_at_freqs.append(segment_amplitudes)
                        imem_phases_at_freqs.append(segment_phases)
                    
                    # Calculate average return current positions for each target frequency
                    positive_avg_imem_pos = []
                    negative_avg_imem_pos = []
                    z_coords = cell.z.mean(axis=-1)
                    imem_amps_array = np.array(imem_amplitudes_at_freqs)

                    for f_idx in range(len(target_freqs)):
                        current_amps = imem_amps_array[:, f_idx]

                        # Positive z-direction
                        pos_indices = np.where(z_coords > 0)[0]
                        if len(pos_indices) > 0:
                            pos_z = z_coords[pos_indices]
                            pos_amps = current_amps[pos_indices]
                            sum_pos_amps = np.sum(pos_amps)
                            if sum_pos_amps > 0:
                                avg_pos_z = np.sum(pos_z * pos_amps) / sum_pos_amps
                                positive_avg_imem_pos.append(avg_pos_z)
                            else:
                                positive_avg_imem_pos.append(0)
                        else:
                            positive_avg_imem_pos.append(0)

                        # Negative z-direction
                        neg_indices = np.where(z_coords < 0)[0]
                        if len(neg_indices) > 0:
                            neg_z = z_coords[neg_indices]
                            neg_amps = current_amps[neg_indices]
                            sum_neg_amps = np.sum(neg_amps)
                            if sum_neg_amps > 0:
                                avg_neg_z = np.sum(neg_z * neg_amps) / sum_neg_amps
                                negative_avg_imem_pos.append(avg_neg_z)
                            else:
                                negative_avg_imem_pos.append(0)
                        else:
                            negative_avg_imem_pos.append(0)

                    # Store all data
                    imem_data[cell_name] = {
                        'freqs': target_freqs,
                        'x': cell.x.tolist(),
                        'z': cell.z.tolist(),
                        'totnsegs': cell.totnsegs,
                        'area': cell.area,
                        'tvec': cell.tvec.tolist(),
                        'imem_amps': imem_amplitudes_at_freqs, 
                        'imem_phases': imem_phases_at_freqs,
                        'positive_avg_imem_pos': positive_avg_imem_pos, 
                        'negative_avg_imem_pos': negative_avg_imem_pos,
                        'input_amps': input_amps,
                        'input_phases': input_phases,
                        'cdm': cdm_amps,
                        'cdm_phases': cdm_phases,
                        'cdm_pos': cdm_pos_amps,
                        'cdm_pos_phases': cdm_pos_phases,
                        'cdm_neg': cdm_neg_amps,
                        'cdm_neg_phases': cdm_neg_phases,
                    }

                    # Save to file
                    np.save(imem_data_file_path, imem_data)
                    print(f"Amplitude and PSD data saved to {os.path.abspath(imem_data_file_path)}")

                    del cell

                    print(f"Simulation complete for neuron {cell_name}: nr.{i}")



if __name__=='__main__':
    upper_len_1 = np.array([1000])
    bottom_len_1 = np.array([-500])
    dend_diam_1 = np.array([2])
    soma_diam_1 = np.array([20])

    upper_len_2 = np.array([200])
    bottom_len_2 = np.array([-100])
    dend_diam_2 = np.array([2])
    soma_diam_2 = np.array([20])

    cut_off = 2**6

    tstop = 2000 + cut_off
    dt = 2**-6

    rate = 5000 # * Hz
    freqs_limit = 10**4

    # Common setup
    num_tsteps = int(tstop / dt)
    tvec = np.arange(num_tsteps) * dt
    t0_idx = np.argmin(np.abs(tvec - cut_off))

    sample_freq = ff.fftfreq(num_tsteps - t0_idx, d=dt / 1000)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]

    print(freqs)
    print(freqs[np.argmin(np.abs(freqs - 1))])

    run_white_noise_imem(tstop, dt,freqs, freqs_limit, soma_diam_1, dend_diam_1, upper_len_1, bottom_len_1, tvec, t0_idx)
    run_white_noise_imem(tstop, dt,freqs, freqs_limit, soma_diam_2, dend_diam_2, upper_len_2, bottom_len_2, tvec, t0_idx)