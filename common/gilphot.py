# C. Gilhuly's photometry utilities (gilphot)
#
# A variety of useful functions for masking and profile extraction

import numpy as np

#############################################################################################################

# Produce annular mask using given radii in pixels and center position.
# Center defaults to center of input image
# Convention: annulus = 0 and the rest of the mask image = 1
#
#!# Potential improvement: change image argument to ymax, xmax
#
def make_annulus_mask( image, rad1=800, rad2=1200, center=None ):

    ymax, xmax = np.shape( image )
    if center:
        y0 = center[0]
        x0 = center[1]
    else:
        y0 = int( ymax/2. ) # Assuming center at image center if not specified otherwise
        x0 = int( xmax/2. )

    mask = np.ones( (ymax, xmax) )
    xv,yv = np.meshgrid( np.linspace( 0, xmax-1, xmax ),
                         np.linspace( 0, ymax-1, ymax ) ) # Big list of pixel coordinates

    dist = np.sqrt( (xv - x0)**2 + (yv - y0)**2 )
    yi,xi = np.where( (dist > rad1) & (dist < rad2) ) # Smaller list of pixels that fall within annulus

    # Setting pixels within annulus to zero
    mask[yi,xi] = 0   

    return mask


#############################################################################################################

# Produce azimuthal wedge mask (like a pie slice) between two angles, given a center. 
# Center defaults to center of input image.
# Convention: wedge = 0 and rest of mask image = 1
#
#!# Potential improvements: change image argument to ymax, xmax; handle negative angles
#
def make_sector_mask( image, theta1, theta2, center=None ): 

    ymax, xmax = np.shape( image )
    if center:
        y0 = center[0]
        x0 = center[1]
    else:
        y0 = int( ymax/2. )
        x0 = int( xmax/2. )

    mask = np.ones( (ymax, xmax) )
    xv,yv = np.meshgrid( np.linspace( 0, xmax-1, xmax ),
                         np.linspace( 0, ymax-1, ymax ) ) # Big list of pixel coordinates

    # Making sure the angles given are ordered correctly
    if theta1 > theta2:
        temp = theta2
        theta2 = theta1
        theta1 = temp

    # Handling case where both angles are larger than 360 degrees
    if theta1 > 2*np.pi and theta2 > 2.*np.pi:
        theta1 = theta1 % ( 2*np.pi )
        theta2 = theta2 % ( 2*np.pi )

    phi = np.arctan2( y0-yv, xv-x0 )
    yt,xt = np.where( phi < 0 )
    phi[yt,xt] = phi[yt,xt] + 2*np.pi
    
    yi,xi = np.where( (phi > theta1) & (phi < theta2) ) # Smaller list of pixels that fall within wedge
    mask[yi,xi] = 0   

    # Handling case where only larger angle is above 360 degrees
    # Without this, wedge mask will go from theta1 - 360 degrees only
    if theta2 > 2*np.pi and theta1 < 2*np.pi:
        theta2 = theta2 % (2*np.pi)

        yi,xi = np.where( (phi >= 0) & (phi < theta2) )
        mask[yi,xi] = 0

    return mask


#############################################################################################################

# Produce rectangular slice mask given a position angle, width, and starting point (center). 
# Center defaults to center of input image.
# Convention: slice = 0 and rest of mask image = 1
#
#!# Potential improvements: change image argument to ymax, xmax
#
def make_slice_mask( image, PA, center = None, width=0, return_r = False ):

    ymax, xmax = np.shape( image )
    if center:
        y0 = center[0]
        x0 = center[1]
    else:
        y0 = int( ymax/2. )
        x0 = int( xmax/2. )

    PA = np.pi * PA / 180. # Converting PA to radians
    
    index = np.indices( image.shape )
    
    # Calculating distances from slice midline (w_temp) and distances along slice (r_temp)
    w_temp = -1.0 * np.sin( PA ) * ( index[0] - center[0] ) - np.cos( PA ) * (index[1] - center[1])
    r_temp = ( ( index[1] - center[1] ) + np.cos( PA ) * w_temp ) / np.sin( PA )
    
    r = r_temp + 0.5
    r = r.astype(int) # Rounding to nearest integer

    # Excluding pixels that either fall outside of acceptable width or that are "behind" slice
    # (at negative radii)
    r[ np.abs( w_temp ) + 0.5 > width ] = -1
    r[ r_temp < -0.5 ] = -1
    
    # Any pixel flagged with -1 is a pixel outside of the desired slice
    mask = ( r == -1 )*1
        
    if return_r:  # Flag to return 2D array of radii (for slice profiles)
        return r

    else:
        return mask


#############################################################################################################

# Calculate surface brightness profile (in linear units) within a rectangular slice
# PA is measured CLOCKWISE from East (if East is left)
#
def radial_profile_slice( data, mask, center, PA, width=0, bin_edges=[1000], bin_widths=[1,5,10,50,100] ):

    # Obtaining 2D array mapping each pixel to a profile radius (or -1)
    r_masked = make_slice_mask( data, PA, center, width, return_r = True )
                    
    # Applying mask to 2D array of slice radii
    r_masked[ mask != 0 ] = -1
        
    radialprofile = []
    errprofile = []
    radii = []
    master_i = 0  # Master index
        
    # bin_edges defines radius (in pixels) where bin width increases
    for bin_i in range( 0, len( bin_edges ) ):
         
        # Using current bin_width while next bin edge has not been passed 
        while master_i < bin_edges[bin_i]:
                
            vals = np.array([])
                
            for j in range( 0, bin_widths[bin_i] ):

                indices = np.where( r_masked == (master_i + j) )
                temp_vals = data[indices]
                vals = np.append( vals, temp_vals )
                
            if len( vals ) > 0:

                N_temp = len( vals )

                radialprofile.append( np.median( vals ) )
                radii.append( np.median( [ master_i, (master_i + bin_widths[bin_i] - 1) ] ) ) 

                # Median is not as efficient an estimator as mean, but is resistant to outliers
                median_efficiency = np.pi * N_temp / ( 2 * N_temp - 2 ) # pi/2 for large N
                errprofile.append( np.std( vals ) * np.sqrt( median_efficiency ) / np.sqrt( N_temp ) )

            # NaN entries when no unmasked pixels exist in current radial bin
            else:

                radialprofile.append( np.nan )
                errprofile.append( np.nan )
                radii.append( np.median( [ master_i, master_i + bin_widths[bin_i] ] ) ) 
                
            master_i = master_i + bin_widths[bin_i] # Starting point for next iteration       
        
    return radialprofile, radii, errprofile


#############################################################################################################

# Calculates the upper and lower surface brightness error bars in mag/arcsec^2, given SB profile and errors
# in linear units. "Flux" variables refer to these linear units (median flux per pixel)
# Assumes random SB error, random sky error, and systematic sky errors (eg. due to variations in bkgd)
# are independent. 
#
#!# Currently assumes sky_err is a single value for the entire profile
# 
def calc_profile_error( flux, flux_err, sky_val, sky_err, zeropoint, sky_err_sys = 0, pix_size = 2.5 ):
    
    mag_plus = zeropoint - 2.5*np.log10( np.array(flux) - sky_val 
                                         + np.sqrt( sky_err**2 + sky_err_sys**2 +  np.square(flux_err) ) 
                                       ) + 5*np.log10(pix_size) 

    mag_minus = zeropoint - 2.5*np.log10( np.array(flux) - sky_val 
                                          - np.sqrt( sky_err**2 + sky_err_sys**2 + np.square(flux_err) ) 
                                        ) + 5*np.log10(pix_size)

    mag_minus = [40 if np.isnan(x) else x for x in mag_minus]  # 40: arbitrarily large magnitude  
    
    return mag_plus, mag_minus


#############################################################################################################

# Calculates the upper and lower colour error bars in mag, given SB profiles and errors
# in linear units. "Flux" variables refer to these linear units (median flux per pixel)
# Assumes random SB error and systematic sky errors (eg. due to large scale variations in bkgd)
# are independent. 
#
#!# Currently neglects random sky error (usually negligible compared to random SB error) 
# 
def calc_colour_error( g_flux, g_err, r_flux, r_err, g_sky, r_sky, dg_sky = 0, dr_sky = 0):
    
    dg = 2.5 * np.array(g_err) / ( (np.array(g_flux) - g_sky) * np.log(10.) )
    dr = 2.5 * np.array(r_err) / ( (np.array(r_flux) - r_sky) * np.log(10.) )
    
    dg_sky = 2.5 * dg_sky / ( (np.array(g_flux) - g_sky) * np.log(10.) )
    dr_sky = 2.5 * dr_sky / ( (np.array(r_flux) - r_sky) * np.log(10.) )
    
    d_gmr = np.sqrt( np.square(dg) + np.square(dr) + np.square(dg_sky) + np.square(dr_sky) )
    
    return d_gmr


#############################################################################################################

# Simple function for writing SB profile to a text file
#
def write_profile( radii, SB, err=None, filename="~/Workspace/temp_profile.txt" ):
    
    fout = open( filename, 'w' )
    for i in range( 0, len(radii) ):
        fout.write( f'{radii[i]:.2f}   {SB[i]:.4f}' )
        if err:
            fout.write( f'   {err[i]:.4f}\n')
        else:
            fout.write( '\n' )
        
    fout.close()
