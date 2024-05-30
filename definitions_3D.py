"""
    ====== Class script for running the simulations ======
              created by: Sarah Howes, Lars Reems
    ======================================================

Includes all the physics formulas that are used in calculating
the new positions and velocities of a 3D system of atoms within 
a volume with periodic boundaries.

This script is imported into run_code_3D.py in order to perform
all simulations. 

"""
import numpy as np
import math
from tqdm import tqdm
from scipy.special import erfc, erf
from scipy.fftpack import fftn, ifftn
from scipy import interpolate

class Particle_Functions3D:

    def __init__(self, length, grid_size, alpha, number_particles):
        self.length = length
        self.grid_size = grid_size
        self.alpha = alpha
        self.number_particles = number_particles
        
    def distance_values_one_timestep(self, data:np.ndarray):
        """Calculate distances and x,y,z-coordinate distances between every particle in one time step

        Args:
            data (np.ndarray): data array for one time step

        Returns:
            np.ndarray, np.ndarray: distances of shape (number_particles, number_particles)
                                    differences of shape (number_particles, number_particles, (x_coord,y_coord,z_coord))
        """
        length = self.length
        x_coordinate, y_coordinate, z_coordinate = data['x_coord'], data['y_coord'], data['z_coord']
        positions = np.array([x_coordinate,y_coordinate,z_coordinate]).T # shape: (N,3) 

        differences = (positions[:,np.newaxis,:]-positions[np.newaxis,:,:] + length/2) % length - length/2
        squared_distances = np.sum(differences**2, axis=2)

        distances = np.sqrt(squared_distances) # shape (N,N)
        np.fill_diagonal(distances, np.inf)

        self.distances = distances
        self.differences = differences

        return distances, differences



    def lennard_jones_potential(self): #, data:np.ndarray):
        """calculate total Lennard-Jones potential energy for one time step

        Args:
            data (np.ndarray): current time step data

        Returns:
            float: total potential energy
        """
        distance_values = self.distances

        lennard_jones_potential = 4 * ((1/distance_values)**12 - (1/distance_values)**6)
        potential_sum = np.sum(lennard_jones_potential)/2
        return potential_sum



    def lennard_jones_potential_derivative(self):
        """calculate derivative of total Lennard-Jones potential energy for one time step

        Args:
            data (np.ndarray): current time step data

        Returns:
            float: derivative of total potential energy
        """
        distance_values = self.distances

        dU = 4 * (-12*(1/distance_values)**13 + 6*(1/distance_values)**7)  # r in units sigma, E in units epsilon

        return dU



    def force_between_atoms(self, data:np.ndarray):
        """calculate force one particle feels for every other particle

        Args:
            data (np.ndarray): current time step data

        Returns:
            float, float, float: force in x, y, z directions
        """
        distance_values = self.distances
        difference_values = self.differences

        dU = self.lennard_jones_potential_derivative()

        force_x = (-dU/distance_values) * difference_values[:,:,0] #dx
        force_y = (-dU/distance_values) * difference_values[:,:,1] #dy
        force_z = (-dU/distance_values) * difference_values[:,:,2] #dz

        force_x_sum = np.sum(force_x, axis=1)
        force_y_sum = np.sum(force_y, axis=1)
        force_z_sum = np.sum(force_z, axis=1)


        return force_x_sum, force_y_sum, force_z_sum


    def pme_direct_energy_force(self, data:np.ndarray):
        """Calculate the real-space potential energy and force due to 
        Coulomb interactions using the PME method.

        Args:
            data (np.ndarray): current time step data

        Returns:
            real_energy(float), forces(np.array): total potential direct energy, 
                                                  direct forces between all particles in x,y,z directions shape: (Nparticles, 3)
        """
        
        alpha = self.alpha
        distances = self.distances
        differences = self.differences
        charges = data['charge']
        charge_product = np.outer(charges, charges)
        

        # Real space energy calculation
        real_energy = np.sum(np.triu(erfc(alpha * distances) / distances * charge_product, k=1))

        # real space force calculation
        erfc_alpha_r = erfc(alpha * distances)
        exp_alpha2_r2 = np.exp(-(alpha * distances) ** 2)
        term1 = erfc_alpha_r / distances**2
        term2 = 2*alpha / (np.sqrt(np.pi)* distances) * exp_alpha2_r2
        force_prefactor = charge_product * (term1 + term2)
        force_matrix = force_prefactor[..., np.newaxis] * differences
        forces = np.sum(force_matrix, axis=1) # shape (108,3)

        return real_energy, forces



    def grid_k_vectors(self, data:np.ndarray):
        """Generate the grid and k-vectors for PME reciprocal space energy and force calculations

        Args:
            data (np.ndarray): current time step data

        Returns:
            grid (np.ndarray), kvecs (np.ndarray), knots (np.ndarray): grid of assigned charges (grid_size,grid_size,grid_size),
                                                                       k-vectors for grid in x,y,z directions (3,grid_size,grid_size,grid_size) 
                                                                       grid positions of each particle in x,y,z directions (3,108)
        """

        grid_size = self.grid_size
        length = self.length

        grid = np.zeros((grid_size, grid_size, grid_size), dtype=complex) # grid


        charges = data['charge']
        x_coordinate, y_coordinate, z_coordinate = data['x_coord'], data['y_coord'], data['z_coord']
        positions = np.array([x_coordinate,y_coordinate,z_coordinate]).T # shape: (N,3) 

        

        # spline interpolation of charges
        tck, u = interpolate.splprep(positions.T)
        xknots, yknots, zknots = interpolate.splev(u, tck)  
        knots = np.array([xknots, yknots, zknots]).astype(int)
        np.add.at(grid, tuple(knots), charges)

        # FFT grid
        grid = fftn(grid)
        kvecs_idx = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
        kvecs_idx = np.where(kvecs_idx > grid_size // 2, kvecs_idx - grid_size, kvecs_idx) # periodic boundary condition -> shift indices
                                                                                           # to half of grid_size to its negative value
        kvecs = 2*np.pi * kvecs_idx / length # convert from grid indices to reciprocal space vectors

        return grid, kvecs, knots


    def pme_reciprocal_energy_force(self, data:np.ndarray):
        """Calculate the reciprocal space energy and forces due to Coulombic interactions
        using the PME method.

        Args:
            data (np.ndarray): current time step data

        Returns:
            recip_energy(float), forces(np.ndarray): total potential reciprocal energy, 
                                                     reciprocal forces between all particles in 
                                                     x,y,z directions shape: (Nparticles, 3)
        """

        length = self.length
        alpha = self.alpha
        grid_size = self.grid_size
        num_particles = self.number_particles
        volume = length**3
        charges = data['charge']

        grid, kvecs, grid_positions = self.grid_k_vectors(data)
        kvecs_squared = np.sum(kvecs**2, axis=0)
        
        kvecs_squared[0, 0, 0] = np.inf

        # reciprocal space energy calculation
        exp_term = np.exp(-np.pi**2 * kvecs_squared / alpha**2)
        scale_factor = 1/(2*np.pi*volume)
        recip_energy = scale_factor * np.sum( (exp_term/kvecs_squared) * np.abs(grid)**2)


        # calculate force
        kvec_exp_factor = exp_term / kvecs_squared
        grid_conj = np.conj(grid)
        grid_product = grid * grid_conj

        force_term = np.zeros((grid_size, grid_size, grid_size, 3), dtype=complex)
        for i in range(3):
            force_term[..., i] = (1j*kvecs[i]*grid_product*kvec_exp_factor)


        force_term = ifftn(force_term, axes=(0, 1, 2)).real

        forces = np.zeros((num_particles, 3))

        grid_positions = grid_positions.T
        grid_positions %= grid_size
        forces = scale_factor * charges[:, np.newaxis] * force_term[grid_positions[:,0], grid_positions[:,1], grid_positions[:,2]]

        return recip_energy, forces



    def pme_self_interaction_energy(self, data:np.ndarray):
        """Calculate the self-interaction Coulombic energy for the PME method (correction term)

        Args:
            data (np.ndarray): current time step data

        Returns:
            energy(float): correction to coulombic energy due to PME self interaction
        """


        alpha = self.alpha
        charges = data['charge']
        energy = -np.sum(charges**2) * alpha / np.sqrt(np.pi)

        return energy



    def pme_energy_force_total(self, data:np.ndarray):
        """Calculate the total energy and force terms due to coulombic interactions
        using the particle mesh ewald (PME) method

        Args:
            data (np.ndarray): current time step data

        Returns:
            total_energy(float), total_forces(np.ndarray): total potential coulombic energy
                                                           total forces with shape (Nparticles, 3) 
        """ 


        real_energy, real_forces = self.pme_direct_energy_force(data)
        recip_energy, recip_forces = self.pme_reciprocal_energy_force(data)
        self_energy = self.pme_self_interaction_energy(data)

        total_energy = real_energy + recip_energy + self_energy
        total_forces = real_forces + recip_forces
        return total_energy, total_forces



    def store_total_force(self, data:np.ndarray, ex_E_field:float, ex_M_field:float, time_step:float):
        """Calculate the total forces acting on all particles due to Eqns of motion (LJ potential),
        PME method, and external E/B fields. Store forces inside class.

        Args:
            data (np.ndarray): current time step data
            ex_E_field (float): magnitude of the external electric field
            ex_M_field (float): magnitude of the external magnetic field
            time_step (float): size of time step
        """
        
        force_x,force_y,force_z = self.force_between_atoms(data)
        ewald_energy, ewald_force = self.pme_energy_force_total(data)
        eforce = ewald_force.T
        eforce_x, eforce_y, eforce_z = eforce[0], eforce[1], eforce[2]

        total_force_x = force_x+eforce_x
        total_force_y = force_y+eforce_y
        total_force_z = force_z+eforce_z

        # external E-field force (x-direction)
        charges = data['charge']
        frequency = 0.1
        extern_elec = ex_E_field*np.cos(frequency*time_step)
        total_force_x += charges*extern_elec

        # external B-field force (y-direction)
        extern_mag = ex_M_field*np.cos(frequency*time_step)
        total_force_y += charges*data['velocity_y']*extern_mag
        total_force_z -= charges*data['velocity_z']*extern_mag

        self.total_forces = [total_force_x, total_force_y, total_force_z]


    def new_verlet_velocity(self, data:np.ndarray, new_data:np.ndarray, time_step:float, ex_E_field:float, ex_M_field:float):
        """calculate new velocity of all particles in the x, y, and z directions using the Verlet algorithm

        Args:
            data (np.ndarray): current time step data
            new_data (np.ndarray): new time step data
            time_step (float): size of time step
            ex_E_field (float): magnitude of the external electric field
            ex_M_field (float): magnitude of the external magnetic field

        Returns:
            velocity_x_new, velocity_y_new, velocity_z_new (np.array): new x, y, z velocities, of size Nparticles
        """

        mass = data['mass']
        
        # old forces
        total_force_x, total_force_y, total_force_z = self.total_forces
        
        # find force for current time step data
        self.store_total_force(new_data, ex_E_field, ex_M_field, time_step)
        total_force_xnew, total_force_ynew, total_force_znew = self.total_forces

        velocity_x, velocity_y, velocity_z = data['velocity_x'], data['velocity_y'], data['velocity_z']

        #Calculate the new x,y velocities
        velocity_x_new = velocity_x + 0.5*time_step*(total_force_x + total_force_xnew)/mass
        velocity_y_new = velocity_y + 0.5*time_step*(total_force_y + total_force_ynew)/mass
        velocity_z_new = velocity_z + 0.5*time_step*(total_force_z + total_force_znew)/mass

        return velocity_x_new, velocity_y_new, velocity_z_new



    def new_positions(self, data:np.ndarray, time_step:float):
        """calculate the new positions of all particles in the x, y, and z directions 
        using the Verlet algorithm
        Args:
            data (np.ndarray): data array for one time step
            time_step (float): size of time step

        Returns:
            x_coord_new, y_coord_new, z_coord_new (np.array): new x, y, z positions, of size Nparticles
        """

        mass = data['mass']

        total_force_x, total_force_y, total_force_z = self.total_forces

        data_x_coord, data_y_coord, data_z_coord = data['x_coord'], data['y_coord'], data['z_coord']
        data_velocity_x, data_velocity_y, data_velocity_z = data['velocity_x'], data['velocity_y'], data['velocity_z']

        x_coord_new = data_x_coord + data_velocity_x*time_step + 0.5*(time_step**2)*total_force_x/mass
        x_coord_new = x_coord_new%self.length

        y_coord_new = data_y_coord + data_velocity_y*time_step + 0.5*(time_step**2)*total_force_y/mass
        y_coord_new = y_coord_new%self.length

        z_coord_new = data_z_coord + data_velocity_z*time_step + 0.5*(time_step**2)*total_force_z/mass
        z_coord_new = z_coord_new%self.length

        return x_coord_new, y_coord_new, z_coord_new # units sigma



    def sum_kinetic_potential_energy(self, stored_particles:np.ndarray, cut_off:float):
        """calculate total kinetic and potential energy for each time step after equilibrium
        using lennard-jones potential and particle mesh ewald energies

        Args:
            stored_particles (np.ndarray): array of all particles at all time steps
            cut_off (float): fraction of first data values that are ignored (tuning equilibrium)

        Returns:
            Ekin_tot (np.array), Epot_tot (np.array): total kinetic and potential energy for system at each time step
        """
        # total energy plot over time
        t_step = 0
        Ekin_tot = []
        Epot_tot = []

        # loop over each time step
        for one_step in tqdm(stored_particles):
            # only calculate energies after equilibrium has reached
            if t_step > cut_off:

                total_velocity = np.sqrt(one_step['velocity_x']**2 + one_step['velocity_y']**2 + one_step['velocity_z']**2)
                mass = one_step['mass']
                Ekins = np.sum(0.5*mass*(total_velocity**2))
                
                self.distance_values_one_timestep(one_step)
                Epots = self.lennard_jones_potential()
                ewald_energy, ewald_force = self.pme_energy_force_total(one_step)
                
                Ekin_tot.append(Ekins)
                Epot_tot.append(Epots+ewald_energy)
            t_step+=1

        Ekin_tot = np.array(Ekin_tot)
        Epot_tot = np.array(Epot_tot)
        
        return Ekin_tot, Epot_tot



    def pair_correlation_histogram(self):
        """calculate the histogram bin values for the pair correlation function

        Returns:
            bins_data (np.array), bins (np.array): counts per bin, bins
        """
        length = self.length
        bins = np.linspace(0,length/2,101)

        # find all distances for one timestep
        distance_values = self.distances

        np.fill_diagonal(distance_values, 0.0)
        distance_values_flattened = distance_values.flatten().flatten()

        # exclude distances between same point
        distance_values_flattened = distance_values_flattened[distance_values_flattened != 0]

        # create histogram
        bins_data = np.histogram(distance_values_flattened, bins)[0]
        return bins_data, bins[1:]