# Lattice Boltzmann Developing poiseulle flow with Bounce-back BC. D2Q9 model.

# Due to the x, y array arrangement and definition of nx as rows and ny as columns the 
# Lattice molecule speed directions are as seen.

'''
    e7  e3  e6
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
nx = 3 #x grid
ny = 17 #y grid
npop = 9 # number of populations used in velocity space discretization.
nsteps = 850 # number of time steps/iterations

# simulation parameters
y_bottom = 0.5
y_top = ny + 0.5


Re = 20 #Reynolds number
tau = 1.0 # choose a value bigger than 0.5 for stability.
omega = 1/tau
kvisc = 1/3*(1/omega-0.5) # kinematic viscosity.
umax = (Re*kvisc)/(y_top - y_bottom)

forcex = (8*umax*kvisc)/((y_top - y_bottom)**2) # This is the force per unit volume = dp/L
forcey = 0

# define macroscopic parameters
rho0 = 1 #rho is scaled to one.
rho = np.ones([nx, ny])
ux = np.zeros([nx, ny])
uy = np.zeros([nx, ny])


# initialize all particle distribution arrays.
feq = np.zeros([npop,nx,ny]) # Equilibrium distribution.
f1 = np.zeros([npop,nx,ny]) # current particle distribution.
f2 = np.zeros([npop,nx,ny]) # modified particle distribution.
forcepop = np.zeros([npop,nx,ny])


# initialize f1 and f2 distribution.
for k in range(npop):
    feq[k] = weights[k]*(rho+rho0*(3*(ux*cx[k]+ uy*cy[k])+ 4.5*(cx[k]*ux + cy[k]*uy)**2 - 1.5*(ux**2 + uy**2)))
    f1 = feq.copy()
    f2 = feq.copy()
    

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
        
        # Compute external forcing term. (Think of this as converting pressure to density and then weighting for different directions)
        forcepop[k] = weights[k]*3*(cx[k]*forcex + cy[k]*forcey)
        
        # Collision step
        f1[k]=f1[k]*(1-omega)+feq[k]*omega + forcepop[k]
        
        # streaming step.
        f2[k] = f1[k].copy()
        
        f2[k] = np.roll(f2[k], cx[k], axis = 0)
        f2[k] = np.roll(f2[k], cy[k], axis = 1)
            
    # Apply bounce-back boundary conditions for the top and bottom walls.
    
    # At bottom-wall
    y=0
    f2[2,:,y] = f1[4,:,y]
    f2[5,:,y] = f1[7,:,y]
    f2[6,:,y] = f1[8,:,y]
    
    # At top-wall
    y = ny-1
    f2[4,:,y] = f1[2,:,y]
    f2[7,:,y] = f1[5,:,y]
    f2[8,:,y] = f1[6,:,y]
    
    # now update f1
    
    f1 = f2.copy()

# Analytical calculation for the flow profile.
# ============================================

y_b = np.array(y_bottom)
y_b = np.reshape(y_b,[1,1])
y_t = np.array(y_top)
y_t = np.reshape(y_t,[1,1])
y_s = np.linspace(1,ny,ny)
y_s = np.reshape(y_s,[1,ny])

y_plot = np.concatenate((y_b, y_s, y_t),1)
y_plot = y_plot.flatten()

del y_b, y_t, y_s


ux_analy = -1/(2*kvisc)*forcex*(y_plot-y_bottom)*(y_plot-y_top)

# Calculate L2 error
# ====================
#sum_num = 0
#sum_denom = 0

#x = nx-1
#for y in range(ny):
#    for x in range(nx):
    #sum_num = sum_num + (ux[x,y]-ux_outlet[y])**2
    #sum_denom = sum_denom + ux_outlet[y]**2
    
#error = ((sum_num)/(sum_denom))**0.5
#print('L2 relative error = ' + str(error))

'''
# calculate momentum imbalance between applied force and friction force on the 
# wall.
# =============================================================================

Fr_bot = -2*np.sum(f1[8,:,0] - f1[7,:,0], axis = 0) # Transverse force on bottom wall
Fr_top = -2*np.sum(f1[5,:,ny-1] - f1[6,:,ny-1], axis = 0) # Transverse force on top wall.

# force along top and bottom walls
F_prop = forcex*nx

delta_F_bot = Fr_bot + F_prop
delta_F_top = Fr_top + F_prop

print('Momentum imbalance bottom wall ' + str(delta_F_bot))
print('Momentum imbalance top wall ' + str(delta_F_top))
 
'''


# plot
# ====

x = np.linspace(1,nx,nx)
y = np.linspace(1,ny,ny)

X,Y = np.meshgrid(x,y)

U = np.fliplr(ux)
U = U.T

# apperently uy direction also changes with the flip.
V = np.fliplr(-uy)
V = V.T


ax = plt.axes()
l1 = ax.plot(y_plot, ux_analy, 'ko--', label='analytical')
l2 = ax.plot(Y[:,2], U[:,2], 'rs--', label = 'LBM')
ax.legend()
plt.title('Poiseuille Flow velocity profile using LBM')
plt.ylabel('Velocity ux (LBM scale)')
plt.xlabel('Height of the profile (LBM scale)')
plt.grid()
plt.show()

# Scaling factors for LBM.
# =======================

'''
The example above demonstrates poiseuille flow between two surfaces. 
Most import thing to note about this model is that the flow dependends 
on dimensionless factors such as Reynolds number.

Therefore the model has been scaled in accordance with keeping the Reynolds
number consistent. 

Remember that only Reynolds number is important to define the characteristics 
of the flow. Therefore this model can represent various different instances 
where the scaling works out to give the same scaled qtys including Re.

Example:

The above flow may model poiseuille flow of water (rho = 1000 kg/m3) across a 
chamber/tube of thickness or diameter 0.001187m (~1.2mm) flowing at a max speed
of 0.015ms-1. 

In accordance with these numbers the scaling factors are calculated to be;

[Quantity = scale_factor * Scaled_quantity]

c_nu = 5.2353e-6 for kinematic viscosity nu = 8.9e-7 m2s-1
c_h = 6.98235e-5 for height of 0.001187m
c_velo = 0.075 for a speed of 0.015ms-1
c_t = 9.31e-4 (This is supposed to be actual time for 1 timestep in the LBM)

c_rho = 1 for convenience and is a secondary scaling factor.

we generally attempt to keep the tau value above 0.5 for stability, and the 
scaled velocity less than 0.3 (to be within the Mach number.)

The scaled lattice spaceing and timesteps are considered to be 1 for convenience
therefore when scaling back to actual values;

delta_x = c_h
delta_t = c_t

Note that tau is selected by the user and the scaled viscosity can be calculated
first. Then using the Reynolds number relationship the remaining factors can be
calculated.

'''
