#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Adding particles to LB3D checkpoint files
=========================================

Opening an LB3D checkpoint and adding particles to the current simulation
based on particle positions added with fibonacci's method.
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
from schiller_lab_tools.lb3d import checkpoint_io
from schiller_lab_tools.data import particles
from schiller_lab_tools.microstructure import interface
import copy
import os


# In[ ]:


path = "../../data/lb3d"
boxDims = np.array([64, 64, 128])
nx, ny, nz = boxDims
Q = 19
timestep = 10000
nprocs = 4
Rp = 8
Ro = 4
npart = 10
rhof = 0.7
radius_prop = 0.7 # assumes z is the long axis if box size is not cubic

gr_out = "restart"
npart_new = 25
## EDIT THIS ONLY ##

output_path = f"{path}/add_particles/npart_{npart_new}"
os.makedirs(output_path, exist_ok=True)


# In[ ]:


## READING OLD CHECKPOINTS ##
checkparams_files = sorted(glob.glob(f"{path}/checkparams*{timestep:08d}*.xdr"))[0]
fluid_checkpoint_files = sorted(glob.glob(f"{path}/checkpoint*{timestep:08d}*.xdr"))
checktopo_files = sorted(glob.glob(f"{path}/checktopo*{timestep:08d}*.xdr"))[0]
md_checkpoint_files = sorted(glob.glob(f"{path}/md-checkpoint*{timestep:08d}*.xdr"))[0]

curr_check_params = checkpoint_io.read_checkparams_xdr(checkparams_files)
curr_topo = checkpoint_io.read_checktopo_xdr(checktopo_files, nprocs = nprocs)
curr_fluid_params = checkpoint_io.read_checkpoint_xdr(fluid_checkpoint_files, nx, ny, nz, curr_topo, Q)
curr_md_params = checkpoint_io.read_md_checkpoint_xdr(md_checkpoint_files, use_rotation=True, interaction="ladd", n_spec = 2)
print("Checkpoints to be copied have been read")
## READING OLD CHECKPOINTS ##


# # Parameter file

# In[ ]:


## PARAMETER CHECKPOINT FILE ##
new_check_params = copy.deepcopy(curr_check_params)
# new_check_params['g_accn_max'] = 2*(nz - join_cut - other_cut) # Changing boxsize in the z direction to new size
print("New parameter file generated")
## PARAMETER CHECKPOINT FILE ##


# # Fluid files

# In[ ]:


# ## FLUID CHECKPOINT FILE ##
new_fluid_params = curr_fluid_params.copy()
print("New fluid checkpoint file generated")
# ## FLUID CHECKPOINT FILE ##


# # Topology

# In[ ]:


## TOPOLOGY CHECKPOINT FILE ##
new_topo = copy.deepcopy(curr_topo)
new_topo['cdims'] = [1,1,1]
new_topo['all_ccoords'] = [0,0,0]
new_topo['nprocs'] = 1
print("New mpi topology file generated")
## TOPOLOGY CHECKPOINT FILE ##


# # MD Files

# In[ ]:


## MD CHECKPOINT FILE ##
curr_part_locs = np.array([part['x'] for part in curr_md_params['particles']])
Rd = np.linalg.norm(curr_part_locs - np.array([32, 32, 64]), axis = -1).mean()

new_md_params = copy.deepcopy(curr_md_params)

new_particles = [] 
n = len(new_md_params["particles"])

center = np.array([nx//2, ny//2, nz//2])
new_part_locs = particles.fibonacci_sphere(npart_new, Rd, center, jitter = 0.01)
reference_dir = np.array([1,0,0])
new_part_quaternions = [particles.orientation_to_quarternion(reference_dir, pos - center) for pos in new_part_locs]

## adding particles by shifting by nz - cut and particles for new droplet.
# Particles are added on top of each other and require equilibration to be pushed apart
idx = 0
particle_id = 1

for i in range(npart):
    curr_part = copy.deepcopy(new_md_params["particles"][i])
    curr_part["uid"] = particle_id

    curr_part["x"] =  new_part_locs[idx]
    # curr_part['q'] = new_part_quaternions[idx]

    curr_part["v"]    = [0.0, 0.0, 0.0]
    curr_part["vnew"] = [0.0, 0.0, 0.0]

    curr_part["q"]    = new_part_quaternions[idx]
    curr_part["qnew"] = new_part_quaternions[idx]

    curr_part["w"]    = [0.0, 0.0, 0.0]
    curr_part["wnew"] = [0.0, 0.0, 0.0]
    new_particles.append(curr_part)
    particle_id += 1
    idx += 1

diff_part = npart_new - npart

# Copies all properties of existing particles. Substitutes the new particle locations and unique ID's
# 1st for loop accounts for if number of particles to be added is larger than number of particles currently present
for i in range(diff_part//npart):
    for j in range(npart):
        curr_part = copy.deepcopy(new_md_params["particles"][j])
        curr_part["uid"] = particle_id

        curr_part["x"] =  new_part_locs[idx]
        # curr_part['q'] = new_part_quaternions[idx]

        curr_part["v"]    = [0.0, 0.0, 0.0]
        curr_part["vnew"] = [0.0, 0.0, 0.0]

        curr_part["q"]    = new_part_quaternions[idx]
        curr_part["qnew"] = new_part_quaternions[idx]

        curr_part["w"]    = [0.0, 0.0, 0.0]
        curr_part["wnew"] = [0.0, 0.0, 0.0]

        new_particles.append(curr_part)
        particle_id += 1
        idx += 1

# 2nd loop accounts for number of particles that is not a clean multiple
for j in range(diff_part%npart):
    curr_part = copy.deepcopy(new_md_params["particles"][j])
    curr_part["uid"] = particle_id

    curr_part["x"] =  new_part_locs[idx]
    # curr_part['q'] = new_part_quaternions[idx]

    curr_part["v"]    = [0.0, 0.0, 0.0]
    curr_part["vnew"] = [0.0, 0.0, 0.0]

    curr_part["q"]    = new_part_quaternions[idx]
    curr_part["qnew"] = new_part_quaternions[idx]

    curr_part["w"]    = [0.0, 0.0, 0.0]
    curr_part["wnew"] = [0.0, 0.0, 0.0]
    new_particles.append(curr_part)
    particle_id += 1
    idx += 1

## mass correction scheme also needs to be adjusted. 
ladd_data = new_md_params['ladd_data']
mass_target = np.array(ladd_data["global_mass_target"])
new_ladd_data = copy.deepcopy(ladd_data)

## TECHNIQUE 3: Calculating mass of box components after slicing ##
particle_masses = 4/3*np.pi*Ro*Ro*Rp*npart_new*rhof
new_target_mass = np.array([np.sum(new_fluid_params[..., :Q])/curr_check_params['taubulk_r'], np.sum(new_fluid_params[..., Q:2*Q])/curr_check_params['taubulk_b']])
new_target_mass -= particle_masses/4
new_ladd_data["global_mass_target"] = new_target_mass
## TECHNIQUE 3: Calculating mass of box components after slicing ##


new_md_params['particles'] = new_particles
new_md_params['ladd_data'] = new_ladd_data
print("New MD checkpoint file generated")
## MD CHECKPOINT FILE ##


# In[ ]:


# make a 3D plot of the system
fig = plt.figure(figsize = (10, 5))

# Make data for droplet
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
droplet_coordinates = np.array([Rd*np.outer(np.cos(u), np.sin(v)),
                                Rd*np.outer(np.sin(u), np.sin(v)),
                                Rd*np.outer(np.ones(np.size(u)), np.cos(v))])

# coordinates shifted by boxDim//2 to account for center of mass location
droplet_coordinates += boxDims[:, np.newaxis, np.newaxis]//2

ax = fig.add_subplot(121, projection="3d")
ax.set_title("Old particle locations")
ax.plot_surface(*droplet_coordinates, alpha = 0.3) # Plot the surface
ax.scatter(*curr_part_locs.T, color="black")

ax = fig.add_subplot(122, projection="3d")
ax.set_title("New particle locations")
ax.plot_surface(*droplet_coordinates, alpha = 0.3) # Plot the surface
ax.scatter(*new_part_locs.T, color="black")


# # Output

# In[ ]:


## OUTPUTTING NEW CHECKPOINTS ##
uid = np.random.randint(0, 2**31, 1)[0] # Generating random number of a signed FP32 integer

checkparams_file_template = "checkparams_{0}_t{1:08d}-{2:010d}.xdr"
fluid_checkpoint_file_template = "checkpoint_{0}_t{1:08d}-{2:010d}_p{3:06d}.xdr"
checktopo_file_template = "checktopo_{0}_t{1:08d}-{2:010d}.xdr"
md_checkpoint_file_template = "md-checkpoint_{0}_t{1:08d}-{2:010d}.xdr"

output_params_path = output_path + "/" + checkparams_file_template.format(gr_out, timestep, uid)
output_fluid_path = output_path + "/" + fluid_checkpoint_file_template.format(gr_out, timestep, uid, 0)
output_topo_path = output_path + "/" + checktopo_file_template.format(gr_out, timestep, uid)
output_md_check_path = output_path + "/" + md_checkpoint_file_template.format(gr_out, timestep, uid)

checkpoint_io.write_checkparams_xdr(output_params_path, new_check_params)
checkpoint_io.write_checkpoint_xdr(output_fluid_path, new_fluid_params)
checkpoint_io.write_checktopo_xdr(output_topo_path, new_topo)
checkpoint_io.write_md_checkpoint_xdr(output_md_check_path, new_md_params["particles"], 
                                    use_rotation=True, steps_per_lbe_step=1, interaction="ladd",
                                    ladd_props=new_md_params['ladd_data'],
                                    n_spec=2)

print(f"Checkpoint output successful!. UID:{uid}")
## OUTPUTTING NEW CHECKPOINTS ##

