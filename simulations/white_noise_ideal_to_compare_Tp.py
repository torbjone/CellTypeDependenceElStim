import os
from os.path import join

import numpy as np

import neuron
import LFPy
import brainsignals.neural_simulations as ns #From ElectricBrainSignals (Hagen and Ness 2023), see README
import scipy.fftpack as ff

np.random.seed(1534)
h = neuron.h
ns.load_mechs_from_folder(ns.cell_models_folder)

def return_ideal_cell(tstop, dt, apic_soma_diam = 20, apic_dend_diam_1=2, apic_dend_diam_2=2, apic_upper_len = 1000, apic_bottom_len = -200):
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
    """ % (apic_soma_diam, apic_soma_diam, apic_dend_diam_1, apic_upper_len, apic_dend_diam_1, apic_dend_diam_2, apic_bottom_len, apic_dend_diam_2, apic_upper_len/2, abs(apic_bottom_len/2)))
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
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

def run_white_noise_ideal(tstop,
                             dt,
                             freqs,
                             freqs_limit, 
                             soma_diam, dend_diam_1, dend_diam_2, upper_len, bottom_len,
                             tvec,
                             t0_idx,
                             multipole_data_filename='compare_wn_ideal_tf',
                             directory='sim_results'
                             ):
    
    multipole_data_filename = f'{multipole_data_filename}.npy'
    multipole_data_file_path = os.path.join(directory, multipole_data_filename)
    
    # Initialize or load existing data
    if os.path.exists(multipole_data_file_path):
        multipole_data = np.load(multipole_data_file_path, allow_pickle=True).item()
    else:
        multipole_data = {}

    i = 0

    for bot_l in bottom_len:
        for up_l in upper_len:
            for s_d in soma_diam:
                for d_d_1 in dend_diam_1:
                    for d_d_2 in dend_diam_2:
                        
                        if d_d_1 > s_d or d_d_2 > s_d:
                            continue

                        i += 1
                        cell_name = f'BL_{bot_l}_UL_{up_l}_SD_{s_d}_DD_1_{d_d_1}_DD_2_{d_d_2}'

                        if check_existing_data(multipole_data, cell_name):
                            print(f"Skipping {cell_name} (already exists in data)")
                            continue

                        cell = return_ideal_cell(tstop, dt,
                                                apic_soma_diam=s_d,
                                                apic_dend_diam_1=d_d_1,
                                                apic_dend_diam_2=d_d_2,
                                                apic_upper_len=up_l,
                                                apic_bottom_len=bot_l)

                        print(f"Running wn simulation with {cell_name}", flush=True)

                        # White noise stimulus
                        cell, syn, noise_vec = make_white_noise_stimuli(
                            cell,
                            input_idx=0,
                            freqs=freqs[freqs < freqs_limit],
                            tvec=tvec,
                            input_scaling = 0.005
                        )

                        # Run simulation
                        cell.simulate(rec_imem=True, rec_vmem=True)

                        # Trim pre-t0
                        cell.vmem = cell.vmem[:, t0_idx:]
                        cell.imem = cell.imem[:, t0_idx:]
                        cell.tvec = cell.tvec[t0_idx:] - cell.tvec[t0_idx]

                        # Compute current dipole moments (z-component)
                        cdm = get_dipole_transformation_matrix(cell) @ cell.imem
                        cdm = cdm[2, :]

                        # Get amplitude spectra
                        freqs_s, amp_cdm_s = ns.return_freq_and_amplitude(cell.tvec, cdm)

                        amp_cdm = amp_cdm_s[0, :]

                        fs = 1000.0 / cell.dt  # Sampling frequency
                        nperseg = 2**12

                        # Convert noise_vec to numpy array and trim to simulation
                        input_current = np.array(noise_vec)
                        input_current = input_current[t0_idx:len(cdm)+t0_idx]

                        _, amp_input_current = ns.return_freq_and_amplitude(cell.tvec, input_current)
                        cdm_per_input_current = amp_cdm / amp_input_current


                        # Compute geometry-based metrics
                        closest_z_endpoint = min(abs(bot_l), abs(up_l))
                        distant_z_endpoint = max(abs(bot_l), abs(up_l))
                        total_len = abs(bot_l) + abs(up_l)
                        symmetry_factor = closest_z_endpoint / distant_z_endpoint

                        # Store all data
                        multipole_data[cell_name] = {
                            'cdm_freqs': freqs_s,
                            'cdm': amp_cdm,
                            'closest_z_endpoint': closest_z_endpoint,
                            'distant_z_endpoint': distant_z_endpoint,
                            'total_len': total_len,
                            'symmetry_factor': symmetry_factor,
                            'soma_diam': s_d,
                            'tot_dend_diam': d_d_1 + d_d_2,
                            'cdm_per_input_current': cdm_per_input_current
                        }

                        # Save to file
                        np.save(multipole_data_file_path, multipole_data)
                        print(f"Amplitude and PSD data saved to {os.path.abspath(multipole_data_file_path)}")

                        # Cleanup
                        del cell, cdm, freqs_s, amp_cdm_s

                        print(f"Simulation complete for neuron {cell_name}: nr.{i}")

    print('All simulations complete')

if __name__=='__main__':
    upper_len_1 = np.array([1000])
    bottom_len_1 = np.array([-500])
    dend_diam_1_1 = np.array([2])
    dend_diam_1_2 = np.array([2])
    soma_diam_1 = np.array([20])

    upper_len_2 = np.array([600])
    bottom_len_2 = np.array([-100])
    dend_diam_2_1 = np.array([2])
    dend_diam_2_2 = np.array([2])
    soma_diam_2 = np.array([10])

    cut_off = 200
    tstop = 2**12 + cut_off
    dt = 2**-6

    rate = 5000 # * Hz
    freqs_limit = 10**4

    # Common setup
    num_tsteps = int(tstop / dt + 1)
    tvec = np.arange(num_tsteps) * dt
    t0_idx = np.argmin(np.abs(tvec - cut_off))

    sample_freq = ff.fftfreq(num_tsteps - t0_idx, d=dt / 1000)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]

    cdm_amp_dict = {}  # To store amplitude spectra for each cell
    imem_amp_dict = {}
    cdm_amp_dict = {}

    run_white_noise_ideal(tstop, dt,freqs, freqs_limit, soma_diam_1, dend_diam_1_1, dend_diam_1_2, upper_len_1, bottom_len_1, tvec, t0_idx)
    run_white_noise_ideal(tstop, dt,freqs, freqs_limit, soma_diam_2, dend_diam_2_1, dend_diam_2_2, upper_len_2, bottom_len_2, tvec, t0_idx)