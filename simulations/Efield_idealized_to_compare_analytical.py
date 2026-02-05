import os
from os.path import join

import numpy as np

import neuron
import LFPy
import brainsignals.neural_simulations as ns #From ElectricBrainSignals (Hagen and Ness 2023), see README
import scipy.fftpack as ff


h = neuron.h
ns.load_mechs_from_folder(ns.cell_models_folder)

def return_ideal_cell(tstop, dt, apic_soma_diam = 20, apic_dend_diam=2, apic_upper_len = 1000, apic_bottom_len = -200):
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
    """ % (apic_soma_diam, apic_soma_diam, apic_dend_diam, apic_upper_len, apic_dend_diam, apic_dend_diam, apic_bottom_len, apic_dend_diam, apic_upper_len/2, abs(apic_bottom_len/2)))
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


# Simulation function ---------------------------------------

def check_existing_data(file_path, cell_name, frequency):
    if not os.path.exists(file_path):
        return False
    
    data = np.load(file_path, allow_pickle=True).item()
    
    if cell_name in data:
        if frequency in data[cell_name]['freq']:
            return True
    
    return False


def run_simulation_neuron_models(freq, soma_diam, dend_diam, upper_len, bottom_len, tstop, dt, cutoff,
                                 local_E_field=1, directory='sim_results'):
    
    amp_data_file_path = os.path.join(directory, 'compare_Vm_sim_analytical.npy')
    
    if os.path.exists(amp_data_file_path):
        amp_data = np.load(amp_data_file_path, allow_pickle=True).item()
    else:
        amp_data = {}
    

    local_ext_pot = np.vectorize(lambda x, y, z: local_E_field * z / 1000)
    n_tsteps_ = int((tstop + cutoff) / dt + 1)
    t_ = np.arange(n_tsteps_) * dt

    i = 0

    for bot_l in bottom_len:
        #bot_dend_len = bot_l

        for up_l in upper_len:
            #up_dend_len = up_l

            for s_d in soma_diam:
                #s_diam = s_d

                for d_d in dend_diam:
                    #d_diam = d_d

                    if d_d > s_d:
                        continue

                    i += 1

                    cell_name = f'BL_{bot_l}_UL_{up_l}_SD_{s_d}_DD_{d_d}'

                    print(f"Running simulation with {cell_name}")

                    for f in freq:
                        if check_existing_data(amp_data_file_path, cell_name, f):
                            print(f"Skipping {cell_name} at {f} Hz (already exists in data)")
                            continue
                    
                        cell = return_ideal_cell(tstop, dt, apic_soma_diam = s_d, apic_dend_diam=d_d, apic_upper_len = up_l, apic_bottom_len = bot_l)

                        cell.extracellular = True

                        for sec in cell.allseclist:
                            sec.insert("extracellular")

                        # Calculate and insert extracellular potential
                        base_pot = local_ext_pot(
                            cell.x.mean(axis=-1),
                            cell.y.mean(axis=-1),
                            cell.z.mean(axis=-1)
                        ).reshape(cell.totnsegs, 1)

                        pulse = np.sin(2 * np.pi * f * t_ / 1000)

                        v_cell_ext = np.zeros((cell.totnsegs, n_tsteps_))
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

                        # Write data to .npy file
                        if cell_name not in amp_data:
                            amp_data[cell_name] = {
                                'freq': [],
                                'soma_vmem_amp': []
                            }
                        amp_data[cell_name]['freq'].append(store_freq)
                        amp_data[cell_name]['soma_vmem_amp'].append(soma_amp)

                        # Save amp data to .npy file
                        np.save(amp_data_file_path, amp_data)
                        print(f"Amplitude data has been saved to {os.path.abspath(amp_data_file_path)}")
                        
                        del cell

                        print(f"{f} Hz complete for {cell_name}, cell nr.{i}")



if __name__=='__main__':
    ns.load_mechs_from_folder(ns.cell_models_folder)

    # Simulation time 
    tstop = 5000.
    dt = 2**-4

    cutoff = 20

    # El field frequency 
    freq1 = np.arange(1, 10, 1) # Shorter steplength in beginning
    freq2 = np.arange(10, 100, 10)
    freq3 = np.arange(100, 2200, 200) # Longer steplength to save calculation time
    freq = sorted(np.concatenate((freq1, freq2, freq3, np.array([1000]))))
    test_freq = np.array([9,500])

    # El field strengths 
    local_E_field = 1  # V/m

    upper_len_1 = np.array([1000])
    bottom_len_1 = np.array([-500])
    dend_diam_1 = np.array([2])
    soma_diam_1 = np.array([20])

    upper_len_2 = np.array([600])
    bottom_len_2 = np.array([-100])
    dend_diam_2 = np.array([2])
    soma_diam_2 = np.array([10])

    run_simulation_neuron_models(freq, soma_diam_1, dend_diam_1, upper_len_1, bottom_len_1, tstop, dt, cutoff)
    run_simulation_neuron_models(freq, soma_diam_2, dend_diam_2, upper_len_2, bottom_len_2, tstop, dt, cutoff)

