# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 16:44:53 2019

@author: Dhayan
"""



# Lattice Boltzmann for lid driven flow. D2Q9 model.

# Due to the x, y array arrangement and definition of nx as rows and ny as columns the 
# Lattice molecule speed directions are as seen.

'''
    e7   e3  e6
      \ | /
    e4 -e0 - e2
      / | \
    e8  e1  e5

''' 

import numpy as np
import matplotlib.pyplot as plt

# Lattice parameters
weights = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
cx = [0, 1, 0, -1, 0, 1, -1, -1, 1] # x speed array of particles.
cy = [0, 0, 1, 0, -1, 1, 1, -1, -1] # y speed array of particles.
opp = [0, 3, 4, 1, 2, 7, 8, 5, 6] # for bounce-back BC. 

# Numerical parameters.

nx = 370 #x grid
ny = 370 #y grid

npop = 9 # number of populations used in velocity space discretization.
nsteps = 3000 # number of time steps/iterations

Re = 1000 #Reynolds number
omega = 3/2 #relaxation frequency
kvisc = 1/3*(1/omega-0.5) # kinematic viscosity.
u_analy = (Re*kvisc)/ny # velocity corresponding to Reynolds number.

# define macroscopic parameters
rho0 = 1
rho = np.ones([nx, ny])
ux = np.zeros([nx, ny])
uy = np.zeros([nx, ny])

# Define the fluid boundary.
fluid = np.ones([nx, ny])

# Define 3 walls for the confined box boundary.
fluid[0,:] = 0 # left wall
fluid[:,0] = 0 # bottom
fluid[nx-1,:] = 0# right wall

# fluid boundaries normal to the wall.
wall_normal = np.roll(fluid, 1, axis = 0) - fluid
left_wall_boundary = wall_normal < 0
# boundary definition shifted to work with Zou-He
left_wall_boundary = np.roll(left_wall_boundary, -1, axis=0) 
left_wall_boundary[0,0] = True

wall_normal = np.roll(fluid, -1, axis = 0) - fluid
right_wall_boundary = wall_normal < 0
# boundary definition shifted to work with Zou-He
right_wall_boundary = np.roll(right_wall_boundary, 1, axis=0)
right_wall_boundary[0, ny-1] = True

wall_normal = np.roll(fluid, 1, axis = 1) - fluid
bottom_wall_boundary = wall_normal < 0
# boundary definition shifted to work with Zou-He
bottom_wall_boundary = np.roll(bottom_wall_boundary, -1, axis=1)
bottom_wall_boundary[0,0] = True
bottom_wall_boundary[0, ny-1] = True

wall_normal = np.roll(fluid, -1, axis=1) - fluid
top_fluid_boundary = wall_normal < 0


# initialize all particle distribution arrays.
feq = np.zeros([npop,nx,ny]) # Equilibrium distribution.
f1 = np.zeros([npop,nx,ny]) # current particle distribution.
f2 = np.zeros([npop,nx,ny]) # modified particle distribution.

# initialize f1 and f2 distribution.
for k in range(npop):
    feq[k] = weights[k]*(rho+rho0*(3*(ux*cx[k]+ uy*cy[k])+ 4.5*(cx[k]*ux + cy[k]*uy)**2 - 1.5*(ux**2 + uy**2)))
    f1 = feq.copy()
    f2 = feq.copy()
    
#plt.ion()
#plt.figure()

#xp = np.linspace(1,nx, nx)
#yp = np.linspace(1,ny, ny)    
#x_plot,y_plot = np.meshgrid(xp, yp)

#slice_interval = 4
#skip = (slice(1, None, slice_interval), slice(1, None, slice_interval))


# Main Algorithm
for counter in range(nsteps):
    
    rho = 0 # reset density.
    ux = 0 # reset macro x-velo.
    uy = 0 # reset macro y-velo.
    
    # find macroscopic terms.
    rho = np.sum(f1, axis=0) # sum all particles in distribution, to find density again.
    
    for k in range(npop):
        ux = ux + cx[k]*f1[k] # similarly find macroscopic velocities. 
        uy = uy + cy[k]*f1[k]
        

    # Apply Collision steps and streaming steps.
    for k in range(npop):
        
        # Compute the populations equilibrium value
        feq[k]= weights[k]*(rho + 3*(ux*cx[k]+uy*cy[k])+ (9/2)*((cx[k]*cx[k]-(1/3))*ux*ux + 2*cx[k]*cy[k]*ux*uy+(cy[k]*cy[k]-(1/3))*uy*uy))
        
        # Collision step
        f1[k]=f1[k]*(1-omega)+feq[k]*omega
        
        # streaming step.
        f2[k] = f1[k].copy()
        
        f2[k] = np.roll(f2[k], cx[k], axis = 0)
        f2[k] = np.roll(f2[k], cy[k], axis = 1)

    # Set up Boundaries
    #==================================================================

    # Apply velocity on top wall using Zou-He
    #==================================================================
    # At the top. (for lid driven cavity flow top face will have a tagential flow applied to it.)

    ux[top_fluid_boundary]= u_analy 
    uy[top_fluid_boundary]= 0
    rho[top_fluid_boundary]=ux[top_fluid_boundary]+(f2[0][top_fluid_boundary]+f2[2][top_fluid_boundary]+f2[4][top_fluid_boundary] + 2*(f2[3][top_fluid_boundary]+f2[6][top_fluid_boundary]+f2[7][top_fluid_boundary]))

    f2[1][top_fluid_boundary] = f2[3][top_fluid_boundary] + (2/3)*ux[top_fluid_boundary]
    f2[5][top_fluid_boundary] = f2[7][top_fluid_boundary] + (1/6)*ux[top_fluid_boundary]+0.5*(f2[4][top_fluid_boundary]-f2[2][top_fluid_boundary])+ (1/2)*uy[top_fluid_boundary]
    f2[8][top_fluid_boundary] = f2[6][top_fluid_boundary] + (1/6)*ux[top_fluid_boundary]-0.5*(f2[4][top_fluid_boundary]-f2[2][top_fluid_boundary])- (1/2)*uy[top_fluid_boundary]
    
    
    # No-slip at the left wall.
    # =================
    
    ux[left_wall_boundary] = 0
    uy[left_wall_boundary] = 0
    
    rho[left_wall_boundary]=ux[left_wall_boundary]+(f2[0][left_wall_boundary]+f2[2][left_wall_boundary]+f2[4][left_wall_boundary]) + 2*(f2[3][left_wall_boundary]+f2[6][left_wall_boundary]+f2[7][left_wall_boundary])

    f2[1][left_wall_boundary] = f2[3][left_wall_boundary] + (2/3)*ux[left_wall_boundary]
    f2[5][left_wall_boundary] = f2[7][left_wall_boundary] + (1/6)*ux[left_wall_boundary] + 0.5*(f2[4][left_wall_boundary]-f2[2][left_wall_boundary]) + (1/2)*uy[left_wall_boundary]
    f2[8][left_wall_boundary] = f2[6][left_wall_boundary] + (1/6)*ux[left_wall_boundary] - 0.5*(f2[4][left_wall_boundary]-f2[2][left_wall_boundary]) - (1/2)*uy[left_wall_boundary]


    # No-slip at the left right wall
    # ==============================
    
    ux[right_wall_boundary] = 0
    uy[right_wall_boundary] = 0
    
    rho[right_wall_boundary]= -ux[right_wall_boundary]+(f2[0][right_wall_boundary]+f2[2][right_wall_boundary]+f2[4][right_wall_boundary]) + 2*(f2[1][right_wall_boundary]+f2[5][right_wall_boundary]+f2[8][right_wall_boundary])

    f2[3][right_wall_boundary] = f2[1][right_wall_boundary] - (2/3)*ux[right_wall_boundary]
    f2[7][right_wall_boundary] = f2[5][right_wall_boundary] - (1/6)*ux[right_wall_boundary] - 0.5*(f2[4][right_wall_boundary]-f2[2][right_wall_boundary]) - (1/2)*uy[right_wall_boundary]
    f2[6][right_wall_boundary] = f2[8][right_wall_boundary] - (1/6)*ux[right_wall_boundary] + 0.5*(f2[4][right_wall_boundary]-f2[2][right_wall_boundary]) + (1/2)*uy[right_wall_boundary]

    
    # No-slip at bottom wall
    # ======================
    
    ux[bottom_wall_boundary] = 0
    uy[bottom_wall_boundary] = 0
    
    rho[bottom_wall_boundary]=uy[bottom_wall_boundary]+(f2[0][bottom_wall_boundary]+f2[1][bottom_wall_boundary]+f2[3][bottom_wall_boundary]) + 2*(f2[4][bottom_wall_boundary]+f2[7][bottom_wall_boundary]+f2[8][bottom_wall_boundary])

    f2[2][bottom_wall_boundary] = f2[4][bottom_wall_boundary] + (2/3)*uy[bottom_wall_boundary]
    f2[5][bottom_wall_boundary] = f2[7][bottom_wall_boundary] + (1/6)*uy[bottom_wall_boundary] + 0.5*(f2[3][bottom_wall_boundary]-f2[1][bottom_wall_boundary]) + (1/2)*ux[bottom_wall_boundary]
    f2[6][bottom_wall_boundary] = f2[8][bottom_wall_boundary] + (1/6)*uy[bottom_wall_boundary] - 0.5*(f2[3][bottom_wall_boundary]-f2[1][bottom_wall_boundary]) - (1/2)*ux[bottom_wall_boundary]
   
    # Corner Treatment.
    # =========================================================================
    
    # Bottom left corner.
    x = 0
    y = 0
    rho[x,y] = rho[x,y+1] # Extrapolation 1st order.
    ux[x,y] = 0
    uy[x,y] = 0
    f2[1,x,y] = f2[3,x,y] + (2/3)*ux[x,y]
    f2[2,x,y] = f2[4,x,y] + (2/3)*uy[x,y]
    f2[5,x,y] = f2[7,x,y] + (1/6)*(ux[x,y] + uy[x,y])
    f2[6,x,y] = (1/12)*(-ux[x,y] + uy[x,y])
    f2[8,x,y] = (1/12)*(ux[x,y]-uy[x,y])
    f2[0,x,y] = 0
    f2[0,x,y] = rho[x,y] - np.sum(f2[:,x,y], axis=0)
    
    # Top left corner
    x = 0
    y = ny-1
    rho[x,y] = rho[x,y-1] # Extrapolation 1st order.
    ux[x,y] = 0
    uy[x,y] = 0
    f2[1,x,y] = f2[3,x,y] + (2/3)*ux[x,y]
    f2[4,x,y] = f2[2,x,y] - (2/3)*uy[x,y]
    f2[8,x,y] = f2[6,x,y] + (1/6)*(ux[x,y] - uy[x,y])
    f2[5,x,y] = (1/12)*(ux[x,y] + uy[x,y])
    f2[7,x,y] = -(1/12)*(ux[x,y] + uy[x,y])
    f2[0,x,y] = 0
    f2[0,x,y] = rho[x,y] - np.sum(f2[:,x,y], axis=0)
    
    
    # Bottom right corner
    x = nx-1
    y = 0
    rho[x,y] = rho[x,y+1] # Extrapolation 1st order.
    ux[x,y] = 0
    uy[x,y] = 0
    f2[3,x,y] = f2[1,x,y] - (2/3)*ux[x,y]
    f2[2,x,y] = f2[4,x,y] + (2/3)*uy[x,y]
    f2[6,x,y] = f2[8,x,y] + (1/6)*(-ux[x,y] + uy[x,y])
    f2[5,x,y] = (1/12)*(ux[x,y] + uy[x,y])
    f2[7,x,y] = -(1/12)*(ux[x,y] + uy[x,y])
    f2[0,x,y] = 0
    f2[0,x,y] = rho[x,y] - np.sum(f2[:,x,y], axis=0)
    
    
    # Top right corner
    x = nx-1
    y = ny-1
    rho[x,y] = rho[x,y-1] # Extrapolation 1st order.
    ux[x,y] = 0
    uy[x,y] = 0
    f2[3,x,y] = f2[1,x,y] - (2/3)*ux[x,y]
    f2[4,x,y] = f2[2,x,y] - (2/3)*uy[x,y]
    f2[7,x,y] = f2[5,x,y] - (1/6)*(ux[x,y] + uy[x,y])
    f2[6,x,y] = (1/12)*(-ux[x,y] + uy[x,y])
    f2[8,x,y] = (1/12)*(ux[x,y]-uy[x,y])
    f2[0,x,y] = 0
    f2[0,x,y] = rho[x,y] - np.sum(f2[:,x,y], axis=0)
    
    
    # Update the f1 distribution.
    f1=f2.copy()
    
    #plt.clf()
    #plt.streamplot(x_plot, y_plot, uy, ux)
    #plt.contourf(((ux.T)**2 + (-uy.T)**2)**0.5)
    #plt.quiver(x_plot[skip], y_plot[skip], uy[skip], ux[skip], angles='uv', scale = 0.75)
    #plt.pause(0.001)

   
# Organize data for plotting.
xp = np.linspace(1,nx, nx)
yp = np.linspace(1,ny, ny)    
x_plot,y_plot = np.meshgrid(xp, yp)

ux_plot = np.zeros([nx,ny])
uy_plot = np.zeros([nx,ny])

for x in range(nx):
    for y in range(ny):
        if fluid[x,y] == 1:
            ux_plot[x,y] = ux[x,y]
            uy_plot[x,y] = uy[x,y]
        else:
            ux_plot[x,y] = None
            uy_plot[x,y] = None
            
U = ux_plot.T
V = uy_plot.T

slice_interval = 12
skip = (slice(1, None, slice_interval), slice(1, None, slice_interval))
plt.contourf((U**2+V**2)**0.5,cmap= 'magma', vmin = 0.04, vmax = 0.16) # plot the magnitude of velocity.
cbar = plt.colorbar()
plt.quiver(x_plot[skip], y_plot[skip], U[skip], V[skip], angles='uv', scale = 1.6, color='b')
#plt.streamplot(x_plot, y_plot, U, V, color='b')
