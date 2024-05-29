"""
    ====== Main script to run all code ======
            ------- 3D version -------
      created by: Sarah Howes, Lars Reems
    =========================================  

Run this script to perform the main molecular dynamics simulation in 3D.
Using user inputs prompts, you can customize the setup for the simulation.
Or, you can choose the default values for the simulation 
and one of the three phases of matters listed below:
###################################################
Different states of matter that have been tested:
# gas: density = 0.3, temperature = 3.0
# liquid: density = 0.8, temperature = 1.0
# solid: density = 1.2, temperature = 0.5
###################################################

This script imports the class Particle_Functions3D from definitions_3D.py
to simulate the positions and velocities of one or two different types of atoms 
within a volume with periodic boundaries.

This script also will create new folders for each state and run number
in order to save the outputted data and plots that are produced.

You can customize which plots and values are calculated/saved using 
user input prompts that are provided after running the script.

"""

import numpy as np
import matplotlib.pyplot as plt
from definitions_3D import Particle_Functions3D
import os
from tqdm import tqdm
import glob
import pandas as pd



def main_simulation(number_particles:float, length:float, number_steps:float, run:float, cut_off:float, 
                    particle_mass:list=[39.9], particle_charge:float=0, ex_E_field:float=0.0, ex_M_field:float=0.0,):
    """
    Main function for running the 3D simulation through given number of time steps. Uses functions from
    Particle_Functions class to find the next positions and velocities for each time step, saves full time
    sequence to a numpy ndarray.

    Args:
        number_particles (float): number of particles in simulation
        length (float): length of box
        number_steps (float): number of time steps
        run (float): run number, for making new files
        cut_off (float): fraction of steps to rescale velocities
        particle_mass (list): masses of one or two particles in amu. Defaults to Argon mass (39.9 amu)
        particle_charge (float): positive charge value of primary particle. 
                                 Secondary particle will be equal to -particle_charge. 
                                 Defaults to zero. 
        ex_E_field (float): external electric field strength. Defaults to zero
        ex_B_field (float): external magnetic field strength. Defaults to zero

    Returns:
        np.ndarray: shape (number_steps, number_particles, (x,y,vx,vy,charge,mass)) all positions and velocities for
        each particle for each time step
    """
        
    # initialize first coordinates, velocities, charges and masses
    diff = length/(number_cubes)**(1/3)
    coordinate_base = list(np.linspace(diff/4,length-diff/4,round(2*number_cubes**(1/3))))
    initial_x_coord = coordinate_base*round(number_cubes**(2/3))*2

    initial_y_coord = []
    for i in range(round(2*(number_cubes**(1/3)))):
        for j in range(round((number_cubes)**(1/3))):
            count = 0
            while count < round((number_cubes)**(1/3)*2):
                if i%2==0:
                    if count%2==0:
                        initial_y_coord.append(coordinate_base[2*j])
                    else:
                        initial_y_coord.append(coordinate_base[2*j+1])
                else:
                    if count%2==0:
                        initial_y_coord.append(coordinate_base[2*j+1])
                    else:
                        initial_y_coord.append(coordinate_base[2*j])
                count += 1

    initial_z_coord = []
    for i in range(len(coordinate_base)):
        count = 0
        while count< round((number_cubes**(2/3)*2)):
            initial_z_coord.append(coordinate_base[i])
            count += 1


    if particle_charge>0:
        initial_charge = particle_charge*np.ones(number_particles)
        initial_charge[::2] = -1 * particle_charge
    else:
        initial_charge = np.zeros(number_particles)


    mass = np.ones(number_particles)
    for i in range(number_particles):
        if len(particle_mass) == 1:
            mass[i] = particle_mass[0]/39.9
        elif len(particle_mass) == 2:
            mass = (particle_mass[0]/39.9)*np.ones(number_particles)
            mass[::2] = particle_mass[1]/39.9
        else:
            print('Invalid input for particle mass')

    initial_velocity_x = (np.random.normal(0,np.sqrt(temperature),number_particles)) / np.sqrt(mass)
    initial_velocity_y = (np.random.normal(0,np.sqrt(temperature),number_particles)) / np.sqrt(mass)
    initial_velocity_z = (np.random.normal(0,np.sqrt(temperature),number_particles)) / np.sqrt(mass)
    
    dtypes = {'names':['x_coord', 'y_coord', 'z_coord', 
                       'velocity_x', 'velocity_y', 'velocity_z', 'charge', 'mass'], 
              'formats':[float,float,float,float,float,float,float,float]}
    
    particles = np.zeros_like(initial_x_coord,dtype=dtypes)
    
    for i in range(len(initial_x_coord)):
        particles[i]= initial_x_coord[i],initial_y_coord[i],initial_z_coord[i],\
            initial_velocity_x[i],initial_velocity_y[i],initial_velocity_z[i],initial_charge[i],mass[i]


    i_stored_particles = np.zeros_like(particles, dtype=dtypes)
    stored_particles = []
    for i in range(number_steps):
        stored_particles.append(i_stored_particles)

    stored_particles = np.array(stored_particles)

    stored_particles[0] = particles

    # start time evolution
    for t in tqdm(range(number_steps-1)):

        particle_simulator.distance_values_one_timestep(stored_particles[t])

        particle_simulator.store_total_force(stored_particles[t], time_step=time_step,
                                             ex_E_field=ex_E_field, ex_M_field=ex_M_field)
        
        x_new, y_new, z_new = particle_simulator.new_positions(stored_particles[t], time_step=time_step)

        new_data = np.zeros_like(x_new,dtype=dtypes)
        for i in range(len(x_new)):
            new_data[i]= x_new[i],y_new[i],z_new[i],0,0,0,initial_charge[i],mass[i]

        particle_simulator.distance_values_one_timestep(new_data)
        
        
        vx_new, vy_new, vz_new = particle_simulator.new_verlet_velocity(stored_particles[t], 
                                                             new_data=new_data, time_step=time_step,
                                                             ex_E_field=ex_E_field, ex_M_field=ex_M_field)

        if t <= cut_off:
            v_tot = np.sqrt(vx_new**2 + vy_new**2 + vz_new**2)
            v_tot_squared_sum = np.sum(mass*v_tot**2)
            rescaling_parameter = np.sqrt(((number_particles-1)*3*temperature)/(v_tot_squared_sum))
            vx_new = rescaling_parameter*vx_new
            vy_new = rescaling_parameter*vy_new
            vz_new = rescaling_parameter*vz_new

        for i in range(len(x_new)):
            new_data[i]= x_new[i],y_new[i],z_new[i],vx_new[i],vy_new[i],vz_new[i],initial_charge[i],mass[i]
    
        stored_particles[t+1] = new_data
    
    np.save(f'{state}/{len(particle_mass)}_particles_{particle_charge}_charge_{ex_E_field}_Efield_{ex_M_field}_Bfield/run3D_{run}/all_data.npy', stored_particles)
    
    return stored_particles



def store_pressure_one_run(stored_particles:np.ndarray, cut_off:float, run_number:int):
    """Calculate and save total 3D pressure for the system

    Args:
        stored_particles (np.ndarray): full simulation data with all time steps
        cut_off (float): fraction of steps to exclude so to only consider equilibrium values
        run_number (int): the run number of the simulation, for saving the pressure value

    Returns:
        float: total pressure of the system
    """

    path = f'{state}/{len(particle_mass)}_particles_{particle_charge}_charge_{ex_E_field}_Efield_{ex_M_field}_Bfield/pressure_3D'
    if not os.path.exists(path): 
        os.makedirs(path)

    time_summed_for_P = []
    for t in tqdm(range(number_steps-1)):
        if t > cut_off:
            rvals, diffs = particle_simulator.distance_values_one_timestep(stored_particles[t])
            dU_vals = particle_simulator.lennard_jones_potential_derivative()#stored_particles[t])
            np.fill_diagonal(rvals, 0.0)
            each_particle_sum = np.sum(rvals*dU_vals, axis=1)
            summed = np.sum(each_particle_sum)/2 # divide by 2 for repeated calculation
            time_summed_for_P.append(summed)

    time_summed_for_P = np.array(time_summed_for_P)

    pressure = (1 - (1 / (6*number_particles*temperature) )*\
                     np.mean(time_summed_for_P))*density*temperature #unitless
    print(f'Pressure: {pressure}')
    print(f'Saving pressure to: {path}')
    np.save(f'{path}/pressure_run_{run_number}.npy', float(pressure))
    return pressure



def store_plot_PCF_one_run(stored_particles:np.ndarray, run:float, cut_off:float):
    """Calculate, plot, and save the 3D pair correlation function values

    Args:
        stored_particles (np.ndarray): full simulation data with all time steps
        run (float): run number to save to folder
        cut_off (float): fraction of steps to exclude so to only consider equilibrium values
    
    Returns:
        np.array: bin values for pair correlation function
    """
    path = f'{state}/{len(particle_mass)}_particles_{particle_charge}_charge_{ex_E_field}_Efield_{ex_M_field}_Bfield/pair_correlation_function_3D'
    if not os.path.exists(path): 
        os.makedirs(path)

    number_bins = 100
    bins_data_summed = np.zeros(number_bins)

    for t in tqdm(range(number_steps-1)):
        if t > cut_off:
            particle_simulator.distance_values_one_timestep(stored_particles[t])
            bins_data, bins = particle_simulator.pair_correlation_histogram()
            bins_data_summed += bins_data

    bins_data_summed = (bins_data_summed / (number_steps-cut_off)) / 2  # exclude double counting of pairs
    bin_size = (length/2) / number_bins
    pair_correlation_function = ((2*length**3) / (number_particles*(number_particles-1))) * (bins_data_summed/(4*np.pi*(bins**2)*bin_size))

    width = bins[1] - bins[0]
    plt.bar(bins, height=pair_correlation_function, width=width, align='edge', color='green', edgecolor='k')
    plt.xlabel('Distance')
    plt.ylabel('Counts')
    plt.title('Pair Correlation Function')
    plt.savefig(f'{state}/{len(particle_mass)}_particles_{particle_charge}_charge_{ex_E_field}_Efield_{ex_M_field}_Bfield/run3D_{run}/pair_correlation_function.png',dpi=300)
    plt.close()

    print('saving pair correlation function to:', path)
    np.save(f'{path}/pcf_run_{run}.npy', pair_correlation_function)
    return pair_correlation_function



def plot_kinetic_potential_energy(stored_particles:np.ndarray, run:float, cut_off:float):
    """Calculate, plot, and save the 3D total potential and kinetic energies of the simulation
    over all time steps.

    Args:
        stored_particles (np.ndarray): full simulation data with all time steps
        run (float): run number to save to folder
        cut_off (float): fraction of steps to exclude so to only consider equilibrium values
    """
    Ekin_tot, Epot_tot = particle_simulator.sum_kinetic_potential_energy(stored_particles, cut_off=cut_off)
    E_tot = Ekin_tot+Epot_tot


    mass_argon = 6.6335209e-26 #kg
    sigma = 3.405e-10 #m
    kb = 1.380649e-23 # kg m2 s-2 K-1
    epsilon = 119.8*kb # kg m2 s-2  -> joules
    epsilon_eV  = epsilon * 6.24150907e18 #eV

    timestep_values = np.arange(0,len(E_tot), 1)
    timestep_SI = time_step*timestep_values*np.sqrt(mass_argon/(sigma**2)/epsilon) # seconds
    timestep_SI = timestep_SI*1e12 # picoseconds

    plt.grid()
    plt.plot(timestep_SI, E_tot*epsilon_eV, c='k', marker='.', alpha=0.5,label='Both')
    plt.plot(timestep_SI, Ekin_tot*epsilon_eV, c='r', marker='.', alpha=0.5,label='Kinetic')
    plt.plot(timestep_SI, Epot_tot*epsilon_eV, c='b', marker='.', alpha=0.5,label='Potential')
    plt.legend()
    plt.xlabel('Time (ps)')
    plt.ylabel('Total Energy of Whole System (eV)')
    plt.title(f'Total Energy for {state} phase')
    plt.savefig(f'{state}/{len(particle_mass)}_particles_{particle_charge}_charge_{ex_E_field}_Efield_{ex_M_field}_Bfield/run3D_{run}/total_energy.png',dpi=300)
    plt.close() 



def plot_3D_positions_one_timestep(stored_particles:np.ndarray, plot_index:int):
    """Plot one 3D position figure of simulation.

    Args:
        stored_particles (np.ndarray): full simulation data with all time steps
        plot_index (int): time step (index) of full simulation to plot in 3D
    """

    import plotly.express as px
    one_step = stored_particles[plot_index]
    xvals = one_step['x_coord']
    yvals = one_step['y_coord']
    zvals = one_step['z_coord']
    vals = {'x_coord': xvals,
            'y_coord': yvals,
            'z_coord': zvals}
    df = pd.DataFrame(vals)
    fig = px.scatter_3d(df, x='x_coord', y='y_coord', z='z_coord')
    fig.update_layout(
        scene = dict(
            xaxis = dict(range=[0,particle_simulator.length],),
                        yaxis = dict(range=[0,particle_simulator.length],),
                        zaxis = dict(range=[0,particle_simulator.length],),))
    fig.show()



def mean_pressure_all_runs(pressures_path:str):
    """Calculate the mean pressure and error of 3D simulations
    over a series of separate runs.

    Args:
        pressures_path (str): path to folder containing all saved pressure values

    Returns:
        float, float: mean and standard deviation of saved pressures
    """

    pressure_files = os.listdir(pressures_path)
    kb = 1.380649e-23 # kg m2 s-2 K-1
    epsilon = 119.8*kb # kg m2 s-2
    sigma = 3.405e-10 # m
    
    pressures_si = []
    pressures_nounit =[]
    for f in pressure_files:
        pres = np.load(f'{pressures_path}/{f}')
        pressures_si.append(pres*(epsilon/(sigma**3)))
        pressures_nounit.append(pres)

    print('mean and std pressures no units:', np.mean(pressures_nounit), np.std(pressures_nounit))
    print('mean and std pressures SI units', np.mean(pressures_si), np.std(pressures_si))

    print(f'saving mean pressure and error')
    np.save(f'{state}/{len(particle_mass)}_particles_{particle_charge}_charge_{ex_E_field}_Efield_{ex_M_field}_Bfield/mean_pressure_3D.npy', np.mean(pressures_nounit))
    np.save(f'{state}/{len(particle_mass)}_particles_{particle_charge}_charge_{ex_E_field}_Efield_{ex_M_field}_Bfield/mean_pressure_error_3D.npy', np.std(pressures_nounit))

    return np.mean(pressures_si), np.std(pressures_si)



def mean_PCF_all_runs(pcf_path:str):
    """Calculate and plot the mean and standard deviation for 
    pair correlation function for a series of separate 3D simulation runs.

    Args:
        pcf_path (str): path to saved pair correlation function values

    Returns:
        np.array, np.array: the mean and error values for each PCF bin
    """
    pcf_files = os.listdir(pcf_path)
    all_files = []
    for file in pcf_files:
        pcf = np.load(f'{pcf_path}/{file}')
        all_files.append(pcf)
    
    all_files = np.array(all_files)
    mean_pcf = np.mean(all_files, axis=0)
    stdev_pcf = np.std(all_files, axis=0)
    np.save(f'{state}/{len(particle_mass)}_particles_{particle_charge}_charge_{ex_E_field}_Efield_{ex_M_field}_Bfield/mean_pcf_3D.npy', mean_pcf)
    np.save(f'{state}/{len(particle_mass)}_particles_{particle_charge}_charge_{ex_E_field}_Efield_{ex_M_field}_Bfield/mean_pcf_std_3D.npy', stdev_pcf)
    
    bins = np.linspace(0,particle_simulator.length/2,100)
    sigma = 3.405 # angstroms

    bins_si = bins*sigma
    width = bins_si[1]-bins_si[0]
    if state == 'solid':
        color = 'forestgreen'
        edgecolor = 'darkgreen'
    elif state== 'liquid':
        color = 'royalblue'
        edgecolor = 'mediumblue'
    elif state == 'gas':
        color = 'chocolate'
        edgecolor = 'saddlebrown'
    else:
        color = 'firebrick'
        edgecolor = 'maroon'
    plt.bar(bins_si, height=mean_pcf, width=width, align='edge', edgecolor=edgecolor, color=color)
    plt.xlabel('Distance (Å)')
    plt.ylabel('Counts')
    plt.title(f'Mean pair correlation function for {state} phase')
    plt.savefig(f'{state}/{len(particle_mass)}_particles_{particle_charge}_charge_{ex_E_field}_Efield_{ex_M_field}_Bfield/mean_pair_correlation_function_3D.png',dpi=300)
    plt.close()

    return mean_pcf, stdev_pcf


######### INPUT DEFINITIONS #########
def define_temp_density():
    """User input to define temperature and density values for desired simulated phase of argon

    Returns:
        float, float, str: temperature and density values for simulation, phase label for making new files
    """
    state = input('>> Choose desired state (solid/liquid/gas/custom):')
    if state == 'solid':
        density = 1.2 #sigma^-3
        temperature = 0.5 #epsilon/kb
    elif state == 'liquid':
        density = 0.8 #sigma^-3
        temperature = 1.0 #epsilon/kb
    elif state == 'gas':
        density = 0.3 #sigma^-3
        temperature = 3.0 #epsilon/kb
    elif state == 'custom':
        density = float(input('>> Input density (units: sigma^-3):'))
        temperature = float(input('>> Input temperature (units: epsilon/kb):'))
        state = f'custom_den_{density}_temp_{temperature}'
    else:
        print('Unknown input')
        exit()

    if not os.path.exists(state): 
        print('Creating new folder for state:', state)
        os.makedirs(state)
    print('Using temperature, density:', temperature,',',density)
    return temperature, density, state



def simulation_setup():
    """Set up user inputs that will customize what each run will produce

    Returns:
        list of floats: returns all the needed variables to set up the simulation, including: 
        the number of cubes the makes up the box, the number of particles, length of each box side, 
        time step (h), number of steps, cut off for equilibrium tuning, temperature, density, and 
        state/phase of system
    """

    print(
        '===========================================================\n'
        '======== Molecular Dynamics with the Argon Atom ===========\n'
        '====== simulation by: Sarah Howes and Lars Reems ==========\n'
        '==========================================================='
        )


    default_type = input('>> Select default type:\n * original (argon, 0 charge, 0 E/B field)\n * NaCl_charged (Na, Cl, +/-1 charge, 0 E/B field)\n \
* NaCl_heavymass (Na mass/1000, Cl mass*1000, +/-1 charge, 0 E/B field)\n * NaCl_highcharge (Na, Cl, +/-10 charge, 0 E/B field)\n \
* NaCl_Efield (Na, Cl, +/-1 charge, 50 E field, 0 B field)\n * NaCl_Bfield (Na, Cl, +/-1 charge, 0 Efield, 15 B field)\n \
* custom (customize all parameters)\n >>')
    

    if default_type == 'original':
        print(
            '===== original simulation values: =====\n'
            '* number_cubes_one_direction: 3\n'
            '* timestep (units: sqrt(m*sigma^2/epsilon)): 0.001\n'
            '* number_steps: 1000\n'
            '* tuning_percent: 0.20\n'
            '* alpha: 0.7\n'
            '* grid size (reciprocal space calculation): 32\n'
            '* external electric field: 0 (units: sigma*np.sqrt(sigma/epsilon) ) \n'
            '* external magnetic field: 0 (units: sigma*np.sqrt(m*sigma)/epsilon )\n'
            '* particle charge: 0 (units: e-/np.sqrt(epsilon*sigma) )\n'
            '* number of different types of particles: 1\n'
            '* particle mass (amu): 39.9 (argon)\n'
            '======================================'
            )
        number_cubes_one_direction = 3        
        time_step = 0.001
        number_steps = 1000
        tuning_percent = 0.20
        alpha = 0.7
        grid_size = 32
        external_electric_field = 0
        external_magnetic_field = 0
        particle_charge = 0 # 2.14e-4 #1 e
        different_particles = 1
        particle_mass = [39.9]


    elif default_type == 'NaCl_charged':
        print(
            '===== argon_charged simulation values: =====\n'
            '* number_cubes_one_direction: 3\n'
            '* timestep (units: sqrt(m*sigma^2/epsilon)): 0.001\n'
            '* number_steps: 1000\n'
            '* tuning_percent: 0.20\n'
            '* alpha: 0.7\n'
            '* grid size (reciprocal space calculation): 32\n'
            '* external electric field: 0 (units: sigma*np.sqrt(sigma/epsilon) ) \n'
            '* external magnetic field: 0 (units: sigma*np.sqrt(m*sigma)/epsilon )\n'
            '* particle charge: 1 (units: e-/np.sqrt(epsilon*sigma) )\n'
            '* number of different types of particles: 2\n'
            '* particle mass (amu): 23.0 (Na), 35.45 (Cl)\n'
            '============================================='
            )
        number_cubes_one_direction = 3        
        time_step = 0.001
        number_steps = 1000
        tuning_percent = 0.20
        alpha = 0.7
        grid_size = 32
        external_electric_field = 0
        external_magnetic_field = 0
        particle_charge = 1
        different_particles = 2
        particle_mass = [23.0, 35.45]


    elif default_type == 'NaCl_heavymass':
        print(
            '===== argon_charged simulation values: =====\n'
            '* number_cubes_one_direction: 3\n'
            '* timestep (units: sqrt(m*sigma^2/epsilon)): 0.001\n'
            '* number_steps: 1000\n'
            '* tuning_percent: 0.20\n'
            '* alpha: 0.7\n'
            '* grid size (reciprocal space calculation): 32\n'
            '* external electric field: 0 (units: sigma*np.sqrt(sigma/epsilon) ) \n'
            '* external magnetic field: 0 (units: sigma*np.sqrt(m*sigma)/epsilon )\n'
            '* particle charge: 1 (units: e-/np.sqrt(epsilon*sigma) )\n'
            '* number of different types of particles: 2\n'
            '* particle mass (amu): 0.023 (Na/10), 35450 (Cl*10)\n'
            '============================================='
            )
        number_cubes_one_direction = 3        
        time_step = 0.001
        number_steps = 1000
        tuning_percent = 0.20
        alpha = 0.7
        grid_size = 32
        external_electric_field = 0
        external_magnetic_field = 0
        particle_charge = 1
        different_particles = 2
        particle_mass = [2.3, 354.5]


    elif default_type == 'NaCl_highcharge':
        print(
            '===== argon_charged simulation values: =====\n'
            '* number_cubes_one_direction: 3\n'
            '* timestep (units: sqrt(m*sigma^2/epsilon)): 0.001\n'
            '* number_steps: 1000\n'
            '* tuning_percent: 0.20\n'
            '* alpha: 0.7\n'
            '* grid size (reciprocal space calculation): 32\n'
            '* external electric field: 0 (units: sigma*np.sqrt(sigma/epsilon) ) \n'
            '* external magnetic field: 0 (units: sigma*np.sqrt(m*sigma)/epsilon )\n'
            '* particle charge: 10 (units: e-/np.sqrt(epsilon*sigma) )\n'
            '* number of different types of particles: 2\n'
            '* particle mass (amu): 23.0 (Na), 35.45 (Cl)\n'
            '============================================='
            )
        number_cubes_one_direction = 3        
        time_step = 0.001
        number_steps = 1000
        tuning_percent = 0.20
        alpha = 0.7
        grid_size = 32
        external_electric_field = 0
        external_magnetic_field = 0
        particle_charge = 10
        different_particles = 2
        particle_mass = [23.0, 35.45]


    elif default_type == 'NaCl_Efield':
        print(
            '===== argon_charged simulation values: =====\n'
            '* number_cubes_one_direction: 3\n'
            '* timestep (units: sqrt(m*sigma^2/epsilon)): 0.001\n'
            '* number_steps: 1000\n'
            '* tuning_percent: 0.20\n'
            '* alpha: 0.7\n'
            '* grid size (reciprocal space calculation): 32\n'
            '* external electric field: 50 (units: sigma*np.sqrt(sigma/epsilon) ) \n'
            '* external magnetic field: 0 (units: sigma*np.sqrt(m*sigma)/epsilon )\n'
            '* particle charge: 1 (units: e-/np.sqrt(epsilon*sigma) )\n'
            '* number of different types of particles: 2\n'
            '* particle mass (amu): 23.0 (Na), 35.45 (Cl)\n'
            '============================================='
            )
        number_cubes_one_direction = 3        
        time_step = 0.001
        number_steps = 1000
        tuning_percent = 0.20
        alpha = 0.7
        grid_size = 32
        external_electric_field = 50
        external_magnetic_field = 0
        particle_charge = 1
        different_particles = 2
        particle_mass = [23.0, 35.45]


    elif default_type == 'NaCl_Bfield':
        print(
            '===== argon_charged simulation values: =====\n'
            '* number_cubes_one_direction: 3\n'
            '* timestep (units: sqrt(m*sigma^2/epsilon)): 0.001\n'
            '* number_steps: 1000\n'
            '* tuning_percent: 0.20\n'
            '* alpha: 0.7\n'
            '* grid size (reciprocal space calculation): 32\n'
            '* external electric field: 0 (units: sigma*np.sqrt(sigma/epsilon) ) \n'
            '* external magnetic field: 15 (units: sigma*np.sqrt(m*sigma)/epsilon )\n'
            '* particle charge: 1 (units: e-/np.sqrt(epsilon*sigma) )\n'
            '* number of different types of particles: 2\n'
            '* particle mass (amu): 23.0 (Na), 35.45 (Cl)\n'
            '============================================='
            )
        number_cubes_one_direction = 3        
        time_step = 0.001
        number_steps = 1000
        tuning_percent = 0.20
        alpha = 0.7
        grid_size = 32
        external_electric_field = 0
        external_magnetic_field = 15
        particle_charge = 1
        different_particles = 2
        particle_mass = [23.0, 35.45]


    elif default_type == 'custom':
        number_cubes_one_direction = int(input('>> Input number of cubes to simulate in \none direction (number will be cubed):'))
        time_step = float(input('>> Input time step \n(time in units sqrt(m*sigma^2/epsilon)):'))
        number_steps = int(input('>> Input number of steps:'))
        tuning_percent = float(input('>> Input fraction of simulation to run equilibrium on:'))
        alpha = float(input('>> Input Ewald parameter (alpha): '))
        grid_size = float(input('>> Input grid size (for reciprocal space calculation): '))
        external_electric_field = float(input('>> Input strength of electric field (input 0 to turn off):'))
        external_magnetic_field = float(input('>> Input strength of magnetic field (input 0 to turn off):'))
        particle_charge = float(input('>> Input charge of particles (if >0, will be +/- alternating like a crystal lattice):'))
        different_particles = int(input('>> Input number of different particle types in a crystal lattice (1 or 2):'))
        if different_particles == 1:
            print('suggestions_for_mass: Ar: 39.95')
            particle_mass = float(input('>> Input the mass of the particles in atomic mass units:'))
            particle_mass = [particle_mass]
        elif different_particles == 2:
            print('suggestions_for_masses: NaCl: 23.0, 35.45')
            particle_mass1 = float(input('>> Input the mass of the first (positively charged, if applicable) particle in amu:'))
            particle_mass2 = float(input('>> Input the mass of the second (negatively charged, if applicable) particle in amu:'))
            particle_mass = [particle_mass1, particle_mass2]
        else:
            print('Invalid input, must be either 1 or 2')
    

    else:
        print('Unknown command')
        exit()

    number_cubes = number_cubes_one_direction**3
    number_particles = number_cubes*4 
    print('Number of particles for this simulation: ', number_particles)
    cut_off = tuning_percent*number_steps
    temperature, density, state = define_temp_density()
    length = (number_particles / density)**(1/3) #for 3D
    

    print(
        '==================================\n'
        '==== Final setup parameters: =====\n'
        f'* number cubes: {number_cubes}\n'
        f'* number particles: {number_particles}\n'
        f'* length (sigma): {length}\n'
        f'* time_step (units: sqrt(m*sigma^2/epsilon)): {time_step}\n'
        f'* number steps: {number_steps}\n'
        f'* tuning percent: {tuning_percent}\n'
        f'* ewald parameter (alpha): {alpha}\n'
        f'* grid size: {grid_size}\n'
        f'* temperature (units: epsilon/kb): {temperature}\n'
        f'* density (units: sigma^-3): {density}\n'
        f'* state: {state}\n'
        f'* external electric field (units: sqrt(epsilon)/(sigma*sqrt(sigma))): {external_electric_field}\n'
        f'* external magnetic field (units: epsilon/(sigma*sqrt(m*sigma))): {external_magnetic_field}\n'
        f'* particle charge (units: sqrt(epsilon*sigma)): {particle_charge}\n'
        f'* number of different particles: {different_particles}\n'
        f'* particle mass (units: amu): {particle_mass}\n'
        '=================================='
    )

    proceed = input('>> Proceed with setup? (y/n):')

    if proceed == 'n':
        exit()
    elif proceed == 'y':
        pass
    else:
        print('Unknown command')
        exit()

    return  number_cubes, number_particles, length, time_step, number_steps, cut_off, alpha, grid_size, temperature, density, state, external_electric_field, external_magnetic_field, particle_charge, particle_mass



def choose_what_to_run(state:str):
    """Select which functions from the above definitions that will run during the simulation.
    You first select the number of runs, then select which of the following you want to be produced:
        - calculate/store pressure of run
        - plot energy diagram for run
        - plot pair correlation function for run
        - plot a 3D diagram of position of particles for one time step in one run
        - calculate the mean pressure over a series of runs
        - calculate/plot the mean pair correlation function over a series of runs

    Args:
        state (str): state of the system. Used to save files to folder

    Returns:
        list of strings: strings are either `y` or `n`, indicating whether or not to run a particular function
    """

    number_runs = int(input('>> Input number of separate simulations to run:'))

    print(
        '====================================\n'
        '===== Default commands to run: =====\n'
        '* main_simulation (required): on\n'
        '* store_pressure (per run): on\n'
        '* plot_energy (per run): on\n'
        '* plot_pair_correlation_function (per run): on\n'
        '* plot_one_step_3D_position (per run): off\n'
        '* mean_pressure (average all runs): on\n'
        '* mean_pair_correlation_function (average all runs): on\n'
        '===================================='
    )

    custom_commands = input('>> Would you like to customize? (y/n):')
    if custom_commands == 'y':
        run_store_pressure = input('store_pressure (on/off):')
        run_plot_energy = input('plot_energy (on/off):')
        run_plot_pcf = input('plot_pair_correlation_function (on/off):')
        run_plot_3D_position = input('plot_one_step_3D_position (on/off):')
        run_mean_pressure = input('mean_pressure (on/off):')
        run_mean_pcf = input('mean_pair_correlation_function (on/off):')
    elif custom_commands == 'n':
        run_store_pressure, run_plot_energy, run_plot_pcf, run_mean_pressure, run_mean_pcf = ['on']*5
        run_plot_3D_position = 'off'
    else:
        print('Unknown command')
        exit()    
    
    print(
        '=======================================\n'
        '======= Chosen commands to run: =======\n'
        f'* Number of simulations to run: {number_runs}\n'
        '* main_simulation (required): on\n'
        f'* store_pressure (per run): {run_store_pressure}\n'
        f'* plot_energy (per run): {run_plot_energy}\n'
        f'* plot_pair_correlation_function (per run): {run_plot_pcf}\n'
        f'* plot_one_step_3D_position (per run): {run_plot_3D_position}\n'
        f'* mean_pressure (average all runs): {run_mean_pressure}\n'
        f'* mean_pair_correlation_function (average all runs): {run_mean_pcf}\n'
        '======================================='
    )

    proceed = input('>> Start simulation? (y/n):')

    if proceed == 'n':
        exit()
    else:
        pass

    return number_runs, run_store_pressure, run_plot_energy, run_plot_pcf, run_mean_pressure, run_mean_pcf, run_plot_3D_position

###################################


if __name__ == "__main__":

    # setting up for simulation
    number_cubes, number_particles, length, \
        time_step, number_steps, cut_off, alpha, grid_size, \
        temperature, density, state, \
        ex_E_field, ex_M_field, particle_charge, particle_mass = simulation_setup()
    
    number_runs, run_store_pressure, \
        run_plot_energy, run_plot_pcf, \
        run_mean_pressure, run_mean_pcf, run_plot_3D_position = choose_what_to_run(state)

    length = (number_particles / density)**(1/3) #for 3D

    run_store_pressure = 'on'
    run_plot_energy = 'on'
    run_plot_pcf = 'on'
    run_plot_3D_position = 'off'
    run_mean_pressure = 'on'
    run_mean_pcf = 'on'


    # start particle simulator
    particle_simulator = Particle_Functions3D(length, grid_size, alpha, number_particles)
    print('/=/=/=/=/ STARTING SIMULATION /=/=/=/=/')

    main_path = f'{state}/{len(particle_mass)}_particles_{particle_charge}_charge_{ex_E_field}_Efield_{ex_M_field}_Bfield'
    if not os.path.exists(main_path): 
        os.makedirs(main_path)
    text_file = open(f"{main_path}/list_of _parameters.txt", "w")
    text_file.write(
        '==================================\n'
        '==== Final setup parameters: =====\n'
        f'* number cubes: {number_cubes}\n'
        f'* number particles: {number_particles}\n'
        f'* length (sigma): {length}\n'
        f'* time_step (units: sqrt(m*sigma^2/epsilon)): {time_step}\n'
        f'* number steps: {number_steps}\n'
        f'* tuning percent: {cut_off/number_steps}\n'
        f'* ewald parameter (alpha): {alpha}\n'
        f'* grid size: {grid_size}\n'
        f'* temperature (units: epsilon/kb): {temperature}\n'
        f'* density (units: sigma^-3): {density}\n'
        f'* state: {state}\n'
        f'* external electric field (units: sqrt(epsilon)/(sigma*sqrt(sigma))): {ex_E_field}\n'
        f'* external magnetic field (units: epsilon/(sigma*sqrt(m*sigma))): {ex_M_field}\n'
        f'* particle charge (units: sqrt(epsilon*sigma)): {particle_charge}\n'
        f'* number of different particles: {len(particle_mass)}\n'
        f'* particle mass (units: amu): {particle_mass}\n'
        '=================================='
    )
    text_file.close()

    for i in range(number_runs):

        run_number = i+1
        print('=========== RUN', run_number, '=============')
        path = f'{main_path}/run3D_{run_number}'
        if not os.path.exists(path): 
            os.makedirs(path)


        print('Running main simulation...')
        stored_particles = main_simulation(number_particles, length, number_steps, run=run_number, 
                                           cut_off=cut_off, particle_mass=particle_mass, particle_charge=particle_charge,  
                                           ex_E_field=ex_E_field, ex_M_field=ex_M_field)
        # stored_particles = np.load(f'{path}/all_data.npy') # !! uncomment this and comment out above line only if    !!
                                                             # !! you have previously saved data that you want to plot !!

        if run_store_pressure == 'on':
            print('Finding pressure...')
            pressure = store_pressure_one_run(stored_particles, cut_off=cut_off, run_number=run_number)

        if run_plot_energy == 'on':
            print('Plotting energy...')
            plot_kinetic_potential_energy(stored_particles, run=run_number, cut_off=cut_off)

        if run_plot_pcf == 'on':
            print('plotting pair correlation function...')
            store_plot_PCF_one_run(stored_particles, run=run_number, cut_off=cut_off)

        if run_plot_3D_position == 'on':
            print('plotting positions...')
            plot_3D_positions_one_timestep(stored_particles, plot_index=0)
    
    print('========== RUNS FINISHED ==========')
    if run_mean_pressure == 'on':
        print(f'Calculating mean pressure over {number_runs} runs...')
        mean_pressure_all_runs(f'{state}/{len(particle_mass)}_particles_{particle_charge}_charge_{ex_E_field}_Efield_{ex_M_field}_Bfield/pressure_3D')
    if run_mean_pcf == 'on':
        print(f'Plotting mean PCF over {number_runs} runs...')
        mean_PCF_all_runs(f'{state}/{len(particle_mass)}_particles_{particle_charge}_charge_{ex_E_field}_Efield_{ex_M_field}_Bfield/pair_correlation_function_3D')

