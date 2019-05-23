# Lattice Boltzmann for immerced cylinder. D2Q9 model.
# Based on the matlab script for LBM for immerced cylinder found http://lbmworkshop.com/?page_id=24 

import numpy as np
import matplotlib.pyplot as plt

# Lattice parameters
weights = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
cx = [0, 1, 0, -1, 0, 1, -1, -1, 1] # x speed array of particles.
cy = [0, 0, 1, 0, -1, 1, 1, -1, -1] # y speed array of particles.
opp = [0, 3, 4, 1, 2, 7, 8, 5, 6] # for bounce-back BC. 

# Numerical parameters.

nx = 50 #x grid
ny = 50 #y grid

npop = 9 # number of populations used in velocity space discretization.
nsteps = 4000 # number of time steps/iterations

# Wall parameters
y_bottom = 1.5 #bottom wall
y_top = ny - 0.5 #top wall

# cylinder parameters (not exact)
obst_r = (y_top + y_bottom)/20 + 1 # radius of the cylinder
obst_x = nx/2 + obst_r #position of the cylinder
obst_y = (y_top + y_bottom)/2 + 1 #y-symmetry 

Re = 100 #Reynolds number
omega = 1/0.6 #relaxation frequency
kvisc = 1/3*(1/omega-0.5) # kinematic viscosity
umax = Re * kvisc/ ((y_top-y_bottom)) # Mach number understood as the CFL
forcex = 8*umax*kvisc/((y_top-y_bottom)**2) # theoretical force.

# Define y range for analytical solution.
y_b = np.array(y_bottom)
y_b = np.reshape(y_b,[1,1])
y_t = np.array(y_top)
y_t = np.reshape(y_t,[1,1])
y_s = np.linspace(1,ny,ny)
y_s = np.reshape(y_s,[1,ny])

y_plot = np.concatenate((y_b, y_s, y_t),1)
y_plot = y_plot.flatten()

del y_b, y_t, y_s

# analytical value of velocity.
ux_analy = -1/(2*kvisc)*forcex*(y_plot-y_bottom)*(y_plot - y_top)

# define macroscopic parameters
rho0 = 1
rho = np.ones([nx, ny])
ux = np.zeros([nx, ny])
uy = np.zeros([nx, ny])

# Define the fluid boundary.
fluid = np.ones([nx, ny])
for x in range(nx):
    for y in range(ny):
        if (((x+1 - obst_x)**2 + (y+1 - obst_y)**2) <= obst_r**2):
            fluid[x,y]=0
            
# Define the Outer boundaries of the cylinder, as well as differentiate between top, bottom and sides.
normal_x = np.zeros([nx,ny])
normal_y = np.zeros([nx,ny])

normal_x = np.roll(fluid,-1,axis=0) - fluid # x-boundaries.
normal_y = np.roll(fluid,-1,axis=1) - fluid # y-boundaries.



# initialize all particle distribution arrays.
feq = np.zeros([npop,nx,ny]) # Equilibrium distribution.
f1 = np.zeros([npop,nx,ny]) # current particle distribution.
f2 = np.zeros([npop,nx,ny]) # modified particle distribution.

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
        
        # Collision step
        f1[k]=f1[k]*(1-omega)+feq[k]*omega
        
        # streaming step.
        f2[k] = f1[k].copy()
        # for k = 0 particles dont move so f2[k] = f2[k]
        if (k == 1):
            f2[k] = np.roll(f2[k], 1, axis = 0) # stream down
        elif (k == 2):
            f2[k] = np.roll(f2[k], 1, axis = 1) # stream right
        elif (k == 3):
            f2[k] = np.roll(f2[k], -1, axis = 0) # stream up
        elif (k == 4):
            f2[k] = np.roll(f2[k], -1, axis = 1) # stream left
        elif (k == 5):
            f2[k] = np.roll(f2[k], 1, axis = 0)
            f2[k] = np.roll(f2[k], 1, axis = 1) # stream +45
        elif (k == 6):
            f2[k] = np.roll(f2[k], -1, axis = 0)
            f2[k] = np.roll(f2[k],  1, axis = 1) # stream +135
        elif (k == 7):
            f2[k] = np.roll(f2[k], -1, axis = 0)
            f2[k] = np.roll(f2[k], -1, axis = 1) # stream 225
        elif (k == 8):
            f2[k] = np.roll(f2[k],  1, axis = 0)
            f2[k] = np.roll(f2[k], -1, axis = 1) # stream 315
        else:
            f2[k] = f2[k]


            
    # Set up Boundaries
    #==================================================================

    # Zou He at Inlet and Outlet
    #==================================================================
    x=0    # Inlet. (At inlet there will be an x velocity applied that corresponds to the analytical value.)

    ux[x,:]= ux_analy[1:ny+1].T
    uy[x,:]= 0
    rho[x,:]=ux[x,:]+(f2[0,x,:]+f2[2,x,:]+f2[4,x,:]+2*(f2[3,x,:]+f2[6,x,:]+f2[7,x,:]))

    f2[1,x,:]=f2[3,x,:]+ (2/3)*ux[x,:]
    f2[5,x,:]=f2[7,x,:]+ (1/6)*ux[x,:]+0.5*(f2[4,x,:]-f2[2,x,:])+ (1/2)*uy[x,:]
    f2[8,x,:]=f2[6,x,:]+ (1/6)*ux[x,:]-0.5*(f2[4,x,:]-f2[2,x,:])- (1/2)*uy[x,:]

    

    x=nx-1   # Outlet

    ux[x,:]=ux[x-1,:]
    uy[x,:]=0
    rho[x,:]=-ux[x,:]+(f2[0,x,:]+f2[2,x,:]+f2[4,x,:]+2*(f2[1,x,:]+f2[5,x,:]+f2[8,x,:]))

    f2[3,x,:]=f2[1,x,:]- (2/3)*ux[x,:]
    f2[7,x,:]=f2[5,x,:]- (1/6)*ux[x,:]-0.5*(f2[4,x,:]-f2[2,x,:])- (1/2)*uy[x,:]
    f2[6,x,:]=f2[8,x,:]- (1/6)*ux[x,:]+0.5*(f2[4,x,:]-f2[2,x,:])+ (1/2)*uy[x,:]
        
        
    # Bounceback Boundary Conditions at bottom and top walls and side walls
    #==================================================================
    
    # find all points on; 
    #-ve boundary of the cylinder x-direction. (Assuming standard cartician xy axes, right boundary)
    nptsx = normal_x < 0
    
    # +ve boundary of the cylinder x-direction.  (left boundary)
    pptsx = normal_x > 0
    
    # -ve bounday of the cylinder y-direction. (top boundary)
    nptsy = normal_y < 0
    
    # +ve boundary of the cylinder y-direction. (bottom boundary)
    pptsy = normal_y > 0
    
    for k in range(npop):
        
        # (assume stanard cartician xy, for a particle moving from right -> left)
        if cx[k] < 0:
            f2[k][nptsx] = f1[opp[k]][nptsx]
        elif (cx[k] > 0): #(particle moving left -> right)
            f2[k][pptsx] = f1[opp[k]][pptsx]
        
        # (particle moving top -> bottom)
        if cy[k] < 0:
            f2[k][nptsy] = f1[opp[k]][nptsy]
        elif (cy[k] > 0): # (particle moving bottom -> top)
            f2[k][pptsy] = f1[opp[k]][pptsy]

   
    f1=f2.copy() # at the end of counter loop update f1
    


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
            ux_plot[x,y] = 'nan'
            uy_plot[x,y] = 'nan'
            

# Plot a velocity fields

f = plt.figure()
plt.quiver(x_plot, y_plot, uy_plot, ux_plot, color='r')

# draw cylinder
theta = np.linspace(0,2*np.pi,51)
plt.plot(obst_y+obst_r*np.cos(theta), obst_x+obst_r*np.sin(theta))
