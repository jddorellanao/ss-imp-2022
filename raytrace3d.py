# Raytrace 3D
import numpy as np
import scipy as sp
import pandas as pd
from math import sin, cos
from scipy.integrate import solve_ivp
## RHS
def rhsf_3d(l, r, slowness, dsdx, dsdy, dsdz, xaxis, yaxis, zaxis, dx, dy, dz):
    """Right-hand side of the ODE for 3D raytracing.
    
    Parameters
    ----------
    l : float
        Independent variable (ray length)
    r : np.ndarray
        Dependent variables (x, y, z, px, py, pz, t)
    slowness : np.ndarray
        Slowness model (nz x ny x nx)
    dsdx : np.ndarray
        x-derivative of slowness
    dsdy : np.ndarray
        y-derivative of slowness
    dsdz : np.ndarray
        z-derivative of slowness
    xaxis : np.ndarray
        x-coordinate axis
    yaxis : np.ndarray
        y-coordinate axis
    zaxis : np.ndarray
        z-coordinate axis
    dx : float
        x-coordinate spacing
    dy : float
        y-coordinate spacing
    dz : float
        z-coordinate spacing
    
    Returns
    -------
    drdt : np.ndarray
        Time derivatives of the dependent variables
    """
    x, y, z, px, py, pz = r[:6]
    m, n, p = slowness.shape
    drdt = np.zeros(len(r))

    xx = (x - xaxis[0]) // dx
    yy = (y - yaxis[0]) // dy
    zz = (z - zaxis[0]) // dz
    xx = min([xx, n-1])
    xx = max([xx, 1])
    yy = min([yy, m-1])
    yy = max([yy, 1])
    zz = min([zz, p-1])
    zz = max([zz, 1]) 

    # extract s, ds/dx, ds/dz at current position (nearest-neighbour interpolation)
    s = slowness[round(zz), round(xx), round(yy)]
    dsdx = dsdx[round(zz), round(xx), round(yy)]
    dsdy = dsdy[round(zz), round(xx), round(yy)]
    dsdz = dsdz[round(zz), round(xx), round(yy)]

    drdt[0] = px/s
    drdt[1] = py/s
    drdt[2] = pz/s
    drdt[3] = dsdx
    drdt[4] = dsdy
    drdt[5] = dsdz
    drdt[6] = s

    return drdt
## Events
def event_left(l, r, slowness, dsdx, dsdy, dsdz, xaxis, yaxis, zaxis, dx, dy, dz):
    return r[0]-xaxis[0]
def event_right(l, r, slowness, dsdx, dsdy, dsdz, xaxis, yaxis, zaxis, dx, dy, dz):
    return xaxis[-1]-r[0]
def event_top(l, r, slowness, dsdx, dsdy, dsdz, xaxis, yaxis, zaxis, dx, dy, dz):
    return r[1]-zaxis[0]
def event_bottom(l, r, slowness, dsdx, dsdy, dsdz, xaxis, yaxis, zaxis, dx, dy, dz):
    return zaxis[-1]-r[1]
def event_back(l, r, slowness, dsdx, dsdy, dsdz, xaxis, yaxis, zaxis, dx, dy, dz):
    return r[2]-yaxis[0]
def event_front(l, r, slowness, dsdx, dsdy, dsdz, xaxis, yaxis, zaxis, dx, dy, dz):
    return yaxis[-1]-r[2]

for evt in [event_left, event_right, event_top, event_bottom, event_back, event_front]:
    evt.terminal = True
    evt.direction = -1
## Raytrace
def raytrace3D(vel, xaxis, yaxis, zaxis, dx, dy, dz, lstep, source, thetas, phis):
    """Raytracing for multiple rays defined by the initial conditions (source, thetas, phis)
    
    Parameters
    ----------
    vel : np.ndarray
        3D Velocity model (nz x ny x nx)
    xaxis : np.ndarray
        Horizonal x-axis 
    yaxis : np.ndarray
        Horizonal y-axis 
    zaxis : np.ndarray
        Vertical axis 
    dx : float
        Horizonal x-spacing 
    dy : float
        Horizonal y-spacing 
    dz : float
        Vertical spacing 
    lstep : np.ndarray
        Ray lenght axis
    source : tuple
        Source location (x, y, z)
    thetas : tuple
        Take-off angles in the y-z plane
    phis : tuple
        Take-off angles in the x-y plane
    """
    # Slowness and its spatial derivatives
    slowness = 1./vel
    [dsdz, dsdy, dsdx] = np.gradient(slowness, dz, dy, dx)
    df = pd.DataFrame(columns = ['Source', 'Theta', 'rx', 'ry', 'rz'])
    for theta in thetas:
        for phi in phis:
            # Initial condition
            r0=[source[0], source[1], source[2],
                sin(theta * np.pi / 180) * cos(phi * np.pi / 180) / vel[izs, iys, ixs],
                sin(theta * np.pi / 180) * sin(phi * np.pi / 180) / vel[izs, iys, ixs],
                cos(theta * np.pi / 180) / vel[izs, iys, ixs], 0]
            
            # Solve ODE
            sol = solve_ivp(rhsf_3d, [lstep[0], lstep[-1]], r0, t_eval=lstep, 
                            args=(slowness, dsdx, dsdy, dsdz, x, y, z, dx, dy, dz),
                            events=[event_right, event_left, event_top, event_bottom, event_back, event_front])
            r = sol['y'].T
            t = sol['t'].T
            # Ray coord
            rx, ry, rz = r[:,0]/1000, r[:,1]/1000, r[:,2]/1000

        for a in range(rx.size):
            append = pd.Series(
                {
                    'Source':source,
                    'Theta':theta,
                    'rx':rx[a],
                    'rx':ry[a],
                    'rz':rz[a],
                    't':t[a]
                }
            )
            df = pd.concat([df, append.to_frame().T], ignore_index=True)
    return df
## Example
# Spatial axes
dx, dy, dz,  = 100, 100, 100
x = np.arange(0, 1000, dx)
y = np.arange(0, 1000, dy)
z = np.arange(0, 1000, dz)

[xx, yy, zz]= np.meshgrid(x, y, z, indexing='ij')

# Velocity model
vel = 1000 + 0.32 * zz + 0.3 * yy

# Source location
ixs = 0
iys = 0
izs = 0
source = [x[ixs], y[iys], z[izs]] 

# Take off angles
thetas = np.arange(10, 60, 5)
phis = np.arange(10, 60, 5)
lstep = np.linspace(0, 1e5, 1000)

r = raytrace3D(vel, x, y, z, dx, dy, dz, lstep, source, thetas, phis)
