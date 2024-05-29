# Computational Physics Assignment 3
# Molecular Dynamics with Coulombic Interactions

There are two main files for this code:
* `definitions_3D.py`
* `run_code_3D.py`

**`definitions_3D.py`** includes a class called `Particle_Functions3D` that is imported into `run_code_3D.py`.
It has all the functions that are used to calculate the next positions and velocities of all the 
atoms due to Newtonian and Coulombic interactions with surrounding particles. It also includes functions that calculate
the total kinetic and potential energies, as well as a function that sets up a histogram in order to 
calculate the pair correlation function. There is no need to run this script separately, it is automatially
imported and used in the second script.

**`run_code_3D.py`** is the main script. It includes a list of definitions that are used to to run the simulation, 
create plots, and calculate mean pressures and pair correlation functions. A few functions also define user 
inputs that are called in the `__main__` that are used in order for the user to customize the setup for the code
(for example, to choose the type of simulation from a pre-defined list, such as selecting an Ar or NaCl simulation). 

### Running the code
Running the code is very simple: all you have to do is run `run_code_3D.py` and a series of prompts will follow
in the terminal for you to fill out. **It is recommended to select one of the pre-defined default simulations the first time you run the code.**

The `__main__` will call all functions defined in `run_code_3D.py` depending on which are enabled from the user inputs, and the 
simulation will run. The simulation will create new folders within the directory of the scripts. The folders are organized
first based off phase of matter (solid, liquid, gas, or custom), then the primary parameters (number of particles, charge, 
presence of E and/or B field), then sub-folders will be created within these for each 
of the runs that are performed. Two folders are also created to store the pressures calculated per run, as well as the
pair correlation function (PCF) values per run. These folders are referenced later to calculate the mean pressure and PCF.
Lastly, a text file containing all of the input parameters is stored in the main folder of the simulation. 

A typical directory will look like this:

* run_code_3D.py
* definitions_3D.py
* solid
* liquid
* gas
    - 2_particles_1_charge_0_Efield_0_Bfield
        - run3D_1
        - run3D_2
        - run3D_3
        - pair_correlation_function_3D
        - pressure_3D

Each run will have the data stored for the simulation as `all_data.npy`, as well as the pair correlation function
and total energy figures (if chosen to plot). The mean PCF data and figure are saved within the main phase directory.
