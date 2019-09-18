# Received in original form on 2019-04-18 from D. Hendel
# C. Gilhuly: modified form of potential, added mock galaxy image

import numpy as np
import matplotlib.pyplot as plt

import astropy.coordinates as coord
import astropy.units as u
from astropy import constants

import scipy.spatial
import gala.coordinates as gc
import gala.dynamics as gd
import gala.potential as gp
from gala.units import galactic
from gala.dynamics.mockstream import fardal_stream
import gala.integrate as gi


def sel( x, y, xmin, xmax, ymin, ymax ):
	"""quick function that returns True if (x,y) is between some boundaries"""
	return ( (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax) )

#####################################################################################################
#
#    First example tidal ribbon
#
#####################################################################################################

########################################
# Variables to be changed:

r_0 = 32 # kpc
v_0 = 220 # km/s

phi = 2./4. * np.pi
x_0 = r_0 * np.cos(phi)
y_0 = r_0 * np.sin(phi)
vx_0 = -1. * v_0 * np.sin(phi)
vy_0 = v_0 * np.cos(phi)

z_0 = 4 # kpc
vz_0 = 20 # km/s
M_progenitor = 1e8 # Msun
n_steps = 3e4


########################################
# Set up potential object

pot = gp.CCompositePotential()

# Using 3 Miyamoto-Nagai disc approximation to an exponential disc
# Coefficients here have been calculated using online calculator (http://astronomy.swin.edu.au/~cflynn/expmaker.php)
# but it's fairly straightforward to implement calculation following https://ui.adsabs.harvard.edu/abs/2015MNRAS.448.2934S/abstract

pot['disk1'] = gp.MiyamotoNagaiPotential( m = 10E10 * 1.9627 * u.Msun,
                                          a = 1.9746 * 5 * u.kpc, 
                                          b = 1 * u.kpc,
                                          units = galactic )

pot['disk2'] = gp.MiyamotoNagaiPotential( m = 10E10 * -1.2756 * u.Msun,
                                          a = 4.2333 * 5 * u.kpc, 
                                          b = 1 * u.kpc,
                                          units = galactic )

pot['disk3'] = gp.MiyamotoNagaiPotential( m = 10E10 * 0.2153 * u.Msun,
                                          a = 0.6354 * 5 * u.kpc, 
                                          b = 1 * u.kpc,
                                          units = galactic )

pot['bulge'] = gp.HernquistPotential( m = 3e10 * u.Msun,
                                      c = 0.5 * u.kpc, 
                                      units = galactic )

## Make grid (0.1 kpc resolution) for mock image
mock_image = np.zeros( (61,201) )

## Calculate densities along line of sight in each x-z bin
temp_index = np.indices( (1,201,1), dtype=float )
temp_index[1] = temp_index[1] / 2. - 50

# Not sure how to get properly shaped array of output from pot.density, hence for-loops
# Only run this once!!
for xind in range(0,201):
    for zind in range(0,61):

        xtemp = xind / 2. - 50
        ztemp = zind / 2. - 15

        temp_index[0] = xtemp
        temp_index[2] = ztemp

        los_mass = np.sum( pot.density( temp_index*u.kpc ) ) / 8 
        mock_image[zind, xind]  = max(0., los_mass.value / 2.)   # Suppose roughly half of disc is gas mass

## Adding halo to potential now that mock galaxy image is in place
pot['halo'] = gp.NFWPotential( m = 10E11, 
                               r_s = 25 * u.kpc, 
                               units = galactic )

########################################
# Set up progenitor initial conditions - syntax is a bit weird but I didn't come up with it!

x, y, z, vx, vy, vz = [ x_0, y_0, z_0, vx_0, vy_0, vz_0 ] # in kpc and km/s

c_gc = coord.Galactocentric( x = x * u.kpc, y = y * u.kpc, z = z * u.kpc, 
	                     v_x = vx * u.km / u.s, v_y = vy * u.km / u.s, v_z = vz * u.km / u.s )

psp = gd.PhaseSpacePosition( pos = c_gc.data.xyz, vel = [ vx, vy, vz ] * u.km / u.s )
orbit = gp.Hamiltonian( pot ).integrate_orbit( psp, dt = -0.5 * u.Myr, n_steps = n_steps )
#orbit.plot()

########################################
# Compute spray model, note that stream.plot() conveniently shows the results
# Description of the method in https://ui.adsabs.harvard.edu/abs/2015MNRAS.452..301F/abstract or gala docs

stream = fardal_stream( pot, orbit[::-1], 
                        prog_mass = M_progenitor * u.Msun,
                        release_every=1 )

# Binning stream to a histogram that matches mock galaxy image
s_binned = np.histogram2d( stream.z, stream.x, bins = [61,201], range=[ [-15,15], [-50,50] ] )
s_binned_scaled = s_binned[0] * M_progenitor / len( stream.z ) # Scaling each bin from counts to stellar mass

# Adding stream to mock galaxy image
# Noise added for aesthetic purposes
noise = np.random.rand( 61, 201 ) * 50000.0
noisy = np.log10( mock_image + s_binned_scaled + noise )
noisy_gal = np.log10( mock_image + noise )


#####################################################################################################
#
#    Second example tidal ribbon
#
#####################################################################################################

########################################
# Variables to be changed:

r_0 = 34 # kpc
v_0 = 210 # km/s

phi = 0./4. * np.pi
x_0 = r_0 * np.cos(phi)
y_0 = r_0 * np.sin(phi)
vx_0 = -1. * v_0 * np.sin(phi)
vy_0 = v_0 * np.cos(phi)

z_0 = 4 # kpc
vz_0 = 20 # km/s
M_progenitor = 1e8 # Msun
n_steps = 3e4

########################################
# Set up progenitor initial conditions - syntax is a bit weird but I didn't come up with it!

x2, y2, z2, vx2, vy2, vz2 = [ x_0, y_0, z_0, vx_0, vy_0, vz_0 ] # in kpc and km/s

c_gc2 = coord.Galactocentric( x = x2 * u.kpc, y = y2 * u.kpc, z = z2 * u.kpc, 
	                      v_x = vx2 * u.km / u.s, v_y = vy2 * u.km / u.s, v_z = vz2 * u.km / u.s)

psp2 = gd.PhaseSpacePosition( pos = c_gc2.data.xyz, vel=[ vx2,vy2,vz2 ] * u.km / u.s)
orbit2 = gp.Hamiltonian( pot ).integrate_orbit( psp2, dt = -0.5 * u.Myr, n_steps = n_steps)
#orbit.plot()

########################################
# Compute spray model, note that stream.plot() conveniently shows the results
# Description of the method in https://ui.adsabs.harvard.edu/abs/2015MNRAS.452..301F/abstract or gala docs

stream2 = fardal_stream( pot, orbit2[::-1], 
                         prog_mass = M_progenitor * u.Msun,
                         release_every = 1 )

# Binning stream to a histogram that matches mock galaxy image
s_binned2 = np.histogram2d(stream2.z, stream2.x, bins=[61,201], range=[[-15,15], [-50,50]])
s_binned_scaled2 = s_binned2[0]*M_progenitor/len(stream2.z)

# Adding stream to mock galaxy image
noise = np.random.rand( 61, 201 ) * 50000.0
noisy2 = np.log10( mock_image + s_binned_scaled2 + noise )
noisy_gal2 = np.log10( mock_image + noise )


#####################################################################################################
#
#    Plotting up both tidal ribbons
#
#####################################################################################################

########################################
#Plot stream edge on as a 2D histogram

cm_grey =plt.cm.gray

heights = [61, 61, 61]
widths = [201, 201, 201]

fig_width = 13
fig_height = (fig_width + 4.4) * sum(heights) / sum(widths)

# One column for each tidal ribbon, 3 different plots
fig, ax = plt.subplots(3, 2, figsize=(fig_width, fig_height), gridspec_kw={'height_ratios':heights})
fig.subplots_adjust(hspace= 0, wspace=0.25)

# Showing just the stream particles for first tidal ribbon
ax[0,0].imshow(s_binned_scaled,  extent=[-50,50, -15, 15], cmap=cm_grey, origin="lower")
ax[0,0].set_ylabel('z [kpc]')
ax[0,0].xaxis.tick_top()
ax[0,0].set_xlabel('x [kpc]')
ax[0,0].xaxis.set_label_position('top')

# Mock galaxy image for first tidal ribbon
# Including original contours (sans ribbon) as well as contours with ribbon
# Contour levels are in solar masses per square kpc
ax[1,0].imshow(noisy, extent=[-50,50, -15, 15], cmap=cm_grey, origin="lower")
ax[1,0].contour(noisy_gal, extent=[-50,50, -15, 15], levels=[5, 5.5, 6, 6.5, 7, 7.5, 8], colors="b", linestyles="dotted")
ax[1,0].contour(noisy, extent=[-50,50, -15, 15], levels=[5, 5.5, 6, 6.5, 7, 7.5, 8], colors="k")

ax[1,0].set_ylabel('z [kpc]')

# Bottom panel: surface density profile through midplane
SB = np.log10( (mock_image[30,:] + s_binned_scaled[30,:]) / 0.5**2 )
radii = np.arange(-50,50.5,0.5)
ax[2,0].plot(radii, SB)
ax[2,0].set_xlim(-50,50)
ax[2,0].set_ylim(3,11)
ax[2,0].set_xlabel('x [kpc]')
ax[2,0].set_ylabel(r'log $\Sigma$ [$M_\odot$ kpc$^{-2}$]')

# Just the stream particles for second tidal ribbon
ax[0,1].imshow(s_binned_scaled2,  extent=[-50,50, -15, 15], cmap=cm_grey, origin="lower")
ax[0,1].set_ylabel('z [kpc]')
ax[0,1].xaxis.tick_top()
ax[0,1].set_xlabel('x [kpc]')
ax[0,1].xaxis.set_label_position('top')

# Mock galaxy image for second tidal ribbon
ax[1,1].imshow(noisy2, extent=[-50,50, -15, 15], cmap=cm_grey, origin="lower")
ax[1,1].contour(noisy_gal2, extent=[-50,50, -15, 15], levels=[5, 5.5, 6, 6.5, 7, 7.5, 8], colors="b", linestyles="dotted")
ax[1,1].contour(noisy2, extent=[-50,50, -15, 15], levels=[5, 5.5, 6, 6.5, 7, 7.5, 8], colors="k")

ax[1,1].set_ylabel('z [kpc]')

# Surface density profile through midplane
SB2 = np.log10( (mock_image[30,:] + s_binned_scaled2[30,:]) / 0.5**2 )
radii = np.arange(-50,50.5,0.5)
ax[2,1].plot(radii, SB2)
ax[2,1].set_xlim(-50,50)
ax[2,1].set_ylim(3,11)
ax[2,1].set_xlabel('x [kpc]')
ax[2,1].set_ylabel(r'log $\Sigma$ [$M_\odot$ kpc$^{-2}$]')

plt.show()
