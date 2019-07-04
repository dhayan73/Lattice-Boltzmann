# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 16:50:17 2019

@author: Dhayan
"""

# Lattice Boltzmann for immerced rigid cylinder. D2Q9 model.
# **Uses Guo's forcing term
# The flow is periodic (generally along the x-axis)
# for this example a rigid cylinder is positioned in the flow.

# Due to the x, y array arrangement and definition of nx as rows and ny as columns the 
# Lattice molecule speed directions are as seen.

'''
    e8  e3  e5
      \ | /
    e2 -e0 - e1
      / | \
    e6  e4  e7

''' 

import numpy as np
import matplotlib.pyplot as plt

# Fluid/ lattice properties

Nx = 100
Ny = 42
cx = [0, 1, -1, 0, 0, 1, -1, 1, -1] # x speed array of particles.
cy = [0, 0, 0, 1, -1, 1, -1, -1, 1] # y speed array of particles.
opp = [0, 2, 1, 4, 3, 6, 5, 8, 7] # for bounce-back BC. 


tau = 0.55 # relation time
omega = 1/tau # relaxation frequency 
t_num = 20000 # number of time steps
gravity = 0.00001 # force density due to gravity (in positive x-direction)
wall_vel_bottom = 0 # velocity of the bottom wall
wall_vel_top = 0 # velocity of the top wall
npop = 9 # number of distribution populations.

# Particle properties.
particle_num_nodes = 36 # numebr of surface nodes
particle_radius = 8 # radius
particle_stiffness = 0.1 #stiffness modulus
particle_center_x = 20
particle_center_y = 20

# define macroscopic properties.
rho = np.ones([Nx, Ny])
ux = np.zeros([Nx, Ny])
uy = np.zeros([Nx, Ny])
force_x = np.zeros([Nx, Ny]) #fluid force (x-component)
force_y = np.zeros([Nx, Ny]) # fluid force (y-component)
weight = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36] # lattice weights.
rho0 = 1

# initialize all particle distribution arrays.
feq = np.zeros([npop,Nx,Ny]) # Equilibrium distribution.
f1 = np.zeros([npop,Nx,Ny]) # current particle distribution.
f2 = np.zeros([npop,Nx,Ny]) # modified particle distribution.
force_latt = np.zeros([npop,Nx,Ny]) # lattice force term entering the lattice Boltzmann equation.

# initialize f1 and f2 distribution.
for k in range(npop):
    feq[k] = weight[k]*(rho+rho0*(3*(ux*cx[k]+ uy*cy[k])+ 4.5*(cx[k]*ux + cy[k]*uy)**2 - 1.5*(ux**2 + uy**2)))
    f1 = feq.copy()
    f2 = feq.copy()

# structure for object, defined in a dictionary. Originally written so that code can select different types of particles.

particle = {} 

particle['num_nodes'] = particle_num_nodes
particle['radius'] = particle_radius
particle['stiffness'] = particle_stiffness
particle['centerx'] = particle_center_x
particle['centery'] = particle_center_y
particle['center_refx'] = particle_center_x
particle['center_refy'] = particle_center_y

# define node structure. As a dictionary
nn = particle_num_nodes

node = {'x':np.ravel(np.zeros([nn,1])),
        'y':np.ravel(np.zeros([nn,1])),
        'x_ref':np.ravel(np.zeros([nn,1])),
        'y_ref':np.ravel(np.zeros([nn,1])),
        'vel_x':np.ravel(np.zeros([nn,1])),
        'vel_y':np.ravel(np.zeros([nn,1])),
        'force_x':np.ravel(np.zeros([nn,1])),
        'force_y':np.ravel(np.zeros([nn,1]))}



for n in range(particle['num_nodes']):
    node['x'][n] = particle['centerx'] + particle['radius']*np.sin(2*np.pi*(n/particle['num_nodes']))
    node['x_ref'][n] = particle['centerx'] + particle['radius']*np.sin(2*np.pi*(n/particle['num_nodes']))
    node['y'][n] = particle['centery'] + particle['radius']*np.cos(2*np.pi*(n/particle['num_nodes']))
    node['y_ref'][n] = particle['centery'] + particle['radius']*np.cos(2*np.pi*(n/particle['num_nodes']))
    
# add the node dictionary to the particle

particle['node'] = node
del node


def compute_particle_forces(particle):
    #Reset forces
    # So that the force from the previous time step is deleted.
    
    particle['node']['force_x'][:] = 0
    particle['node']['force_y'][:] = 0
    
    # compute the strain forces. (Only sensitive to deformation)
    # for a rigid cylinder the node forces are proprotional to the displacement w.r.t the reference position.
    
    area = 2 * np.pi * particle['radius']/particle['num_nodes'] # area belonging to a node.
    
    particle['node']['force_x'] = -particle['stiffness'] * (particle['node']['x'] - particle['node']['x_ref']) * area
    particle['node']['force_y'] = -particle['stiffness'] * (particle['node']['y'] - particle['node']['y_ref']) * area
    
    return particle

# spread forces from the Lagrangian to the Eulerian mesh
def spread(particle, force_x, force_y): 
    
    # Reset lattice forces. 
    force_x[:,1:Ny-2] = 0
    force_y[:,1:Ny-2] = 0
        
    for n in range(particle['num_nodes']):
        
        # Identify the lowest fluid lattice node in the interpolation range.
        # Lowest means its x and y values are the smallest.
        # The other fluid node in range have co-ordinates
        # (x_int+1, y_int), (x_int, y_int+1) and (x_int+1, y_int+1).
        
        x_int = int((particle['node']['x'][n] - 0.5 + Nx)-Nx)
        y_int = int((particle['node']['y'][n] + 0.5))
        
        # Run over all neighboring fluid nodes.
        # In case of the two-point interpolation, it is 2x2 fluid nodes.
        
        X = x_int
        
        while (X <= x_int + 1):
            X += 1
            Y = y_int
            while (Y <= y_int + 1):
                                
                # Compute distance between object node and fluid lattice node.
                
                dist_x = particle['node']['x'][n] - 0.5 - (X-1)
                dist_y = particle['node']['y'][n] + 0.5 - Y
                
                # Compute interpolation weights for x and y directions based on the distance.
                
                weight_x = 1 - np.abs(dist_x)
                weight_y = 1 - np.abs(dist_y)
                
                # Compute lattice force.
                
                force_x[(X-1 + Nx)% Nx][Y] += (particle['node']['force_x'][n] * weight_x * weight_y)
                force_y[(X-1 + Nx)% Nx][Y] += (particle['node']['force_y'][n] * weight_x * weight_y)
                
                Y +=1
                
    return force_x, force_y


# interpolate the velocity
def interpolate(particle): 
    
    # Reset velocity first.
    particle['node']['vel_x'][:] = 0
    particle['node']['vel_y'][:] = 0
    
    
    # run over all particle nodes.
    for n in range(particle['num_nodes']):
        
        # identify the lowest fluid lattice node in the interpolation range (Similar to spreading force)
        
        x_int = int((particle['node']['x'][n] - 0.5 + Nx) - Nx)
        y_int = int((particle['node']['y'][n] + 0.5))
        
        # run over all neighboring fluid nodes.
        # In case of the two point interpolation, it is 2x2 fluid nodes.
        
        X = x_int
        while (X <= x_int + 1) :
            X += 1
            Y = y_int
            while (Y <= y_int + 1):
                
                # compute distance between object node and fluid lattice node.
                
                dist_x = particle['node']['x'][n] - 0.5 - (X-1)
                dist_y = particle['node']['y'][n] + 0.5 - Y
                
                # Compute interpolation weights for x and y direction based on the distance.
                
                weight_x = 1 - np.abs(dist_x)
                weight_y = 1 - np.abs(dist_y)
                
                # compute node velocities.
                
                particle['node']['vel_x'][n] += (ux[(X-1 + Nx) % Nx][Y] * weight_x * weight_y)
                particle['node']['vel_y'][n] += (uy[(X-1 + Nx) % Nx][Y] * weight_x * weight_y)
                
                Y += 1
    
    return particle

def update_particle_position(particle):
    
    # Reset center position
    particle['centerx'] = 0
    particle['centery'] = 0
    
    # Update node and center positions.
    for n in range(particle['num_nodes']):
        particle['node']['x'][n] += particle['node']['vel_x'][n]
        particle['node']['y'][n] += particle['node']['vel_y'][n]
        particle['centerx'] += particle['node']['x'][n]/ particle['num_nodes']
        particle['centery'] += particle['node']['y'][n]/ particle['num_nodes']
        
    # Check for periodicity along the x-axis
    if (particle['centerx'] < 0):
        particle['centerx'] += Nx
        
        for c in range(particle['num_nodes']):
            particle['node']['x'][c] += Nx
        
    elif (particle['centerx'] >= Nx):
        particle['centerx'] -= Nx
        
        for c in range(particle['num_nodes']):
            particle['node']['x'][c] -= Nx
            
    return particle




# Main Algorithm
for counter in range(t_num):
    
    particle = compute_particle_forces(particle)
    
    # Spread the particle force to the mesh. (Lagrangian -> Eulerian)
    force_x, force_y = spread(particle, force_x, force_y)
    
    # Apply Gou's forcing term.
    force_latt[0] = (1- 0.5 * omega) * weight[0] * (3 * ((   - ux) * (force_x + gravity) + (   - uy) * force_y))
    force_latt[1] = (1- 0.5 * omega) * weight[1] * (3 * (( 1 - ux) * (force_x + gravity) + (   - uy) * force_y) + 9 * (ux     ) * (force_x + gravity          ))
    force_latt[2] = (1- 0.5 * omega) * weight[2] * (3 * ((-1 - ux) * (force_x + gravity) + (   - uy) * force_y) + 9 * (ux     ) * (force_x + gravity          ))
    force_latt[3] = (1- 0.5 * omega) * weight[3] * (3 * ((   - ux) * (force_x + gravity) + ( 1 - uy) * force_y) + 9 * (     uy) * (                    force_y))
    force_latt[4] = (1- 0.5 * omega) * weight[4] * (3 * ((   - ux) * (force_x + gravity) + (-1 - uy) * force_y) + 9 * (     uy) * (                    force_y))
    force_latt[5] = (1- 0.5 * omega) * weight[5] * (3 * (( 1 - ux) * (force_x + gravity) + ( 1 - uy) * force_y) + 9 * (ux + uy) * (force_x + gravity + force_y))
    force_latt[6] = (1- 0.5 * omega) * weight[6] * (3 * ((-1 - ux) * (force_x + gravity) + (-1 - uy) * force_y) + 9 * (ux + uy) * (force_x + gravity + force_y))
    force_latt[7] = (1- 0.5 * omega) * weight[7] * (3 * (( 1 - ux) * (force_x + gravity) + (-1 - uy) * force_y) + 9 * (ux - uy) * (force_x + gravity - force_y))
    force_latt[8] = (1- 0.5 * omega) * weight[8] * (3 * ((-1 - ux) * (force_x + gravity) + ( 1 - uy) * force_y) + 9 * (ux - uy) * (force_x + gravity - force_y))
   

    # Apply Collision steps and streaming steps of general LBM algorithm.
    for k in range(npop):
        
        # Compute the populations equilibrium value
        feq[k] = weight[k] * rho * (1 + 3*(cx[k]*ux + cy[k]*uy) + 4.5*((cx[k]*ux + cy[k]*uy)**2) - 1.5*(ux**2 + uy**2))
        
        # Collision step
        f1[k]=f1[k]*(1-omega)+feq[k]*omega + force_latt[k]
        
        # streaming step.
        f2[k] = f1[k].copy()
        
        f2[k] = np.roll(f2[k], cx[k], axis = 0)
        f2[k] = np.roll(f2[k], cy[k], axis = 1)
        
    
    # now APPLY BOUNCE_BACK BCs.
    
    # Due to the presence of the rigid walls at y=0 and y = Ny -1, the populations have to be bounced back
    # Ladd's momentum correction term is included for moving walls (wall velocity parallel to x-axis).
    # Periodicty in the lattice in x-direction is taken into account via the %-operator.
    
    X = np.linspace(0, Nx-1, Nx, dtype=int)
    
    # for the bottom wall (y=0)
    f2[opp[4],:,1] = f2[4,:,0]
    f2[opp[6],0:Nx,1] = f2[6,(X-1+Nx)%Nx,0] + 6 * weight[6] * rho[0:Nx,1] * wall_vel_bottom
    f2[opp[7],0:Nx,1] = f2[7,(X+1)%Nx,0] - 6 * weight[7] * rho[0:Nx,1] * wall_vel_bottom
    
    # for the top wall ( y = Ny -1)
    f2[opp[3],:,Ny-2] = f2[3,:,Ny-1]
    f2[opp[5],0:Nx,Ny-2] = f2[5,(X+1)%Nx,Ny-1] - 6 * weight[5] * rho[0:Nx,Ny-2] * wall_vel_top
    f2[opp[8],0:Nx,Ny-2] = f2[8,(X-1+Nx)%Nx,Ny-1] + 6 * weight[8] * rho[0:Nx,Ny-2] * wall_vel_top
    
    f1 = f2.copy()
    
    rho = 0 # reset density.
    ux = 0 # reset macro x-velo.
    uy = 0 # reset macro y-velo.
    
    # find macroscopic terms.
    rho = np.sum(f1, axis=0) # sum all particles in distribution, to find density again.
    
    # similarly find macroscopic velocities. 
    for k in range(npop):
        ux = ux + cx[k]*f1[k]
        uy = uy + cy[k]*f1[k]
    
    # velocity correction due to body force is included.(Guo's forcing)
    ux = (ux + 0.5 * (force_x + gravity))/rho 
    uy = (uy + 0.5 * (force_y))/rho 
    
    # interpolate the surrounding mesh velocities on to the particle nodes. 
    particle = interpolate(particle)
    
    # reset particle position. (Considering a semi-rigid circle)
    particle = update_particle_position(particle)
    
    # End of main algorithm.
        



# PLOTTING
plt.contourf(((ux.T)**2 + (uy.T)**2)**0.5)
h = plt.colorbar()
plt.xlabel('Nx')
plt.ylabel('Ny')
plt.title('magnitude of fluid velocity for immerced rigid cylinder')
#xp = np.linspace(1,Nx, Nx)
#yp = np.linspace(1,Ny, Ny)    
#x_plot,y_plot = np.meshgrid(xp, yp)

#U = ux.T
#V = uy.T

#plt.streamplot(x_plot, y_plot, U, V)