# C. Gilhuly's photometry utilities (gilphot)
#
# A variety of useful functions for masking and profile extraction

import numpy as np

from photutils import EllipticalAperture
from photutils.isophote import EllipseGeometry, EllipseSample, Isophote, IsophoteList
from photutils.isophote.isophote import Isophote, IsophoteList

from photutils.isophote.sample import CentralEllipseSample
from photutils.isophote.fitter import CentralEllipseFitter

from photutils.isophote import build_ellipse_model

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

# Make a source mask from ds9 regions file (for circular regions)
# If mask is for a source-subtracted image, will also identify pixels masked by MRF (== 0.000)
#

from astropy.io import fits
from skimage.morphology import erosion, dilation
from skimage.morphology import disk

def make_source_mask( image, image2, header, regionsfile, outfile, mask_zeros = True, expand = 5 ):

    ymax, xmax = np.shape( image )
    mask = np.zeros( (ymax, xmax) )

    if mask_zeros == True:
        mask[ image == 0.00 ] = 1
        mask[ image2 == 0.00 ] = 1

        if expand > 0:
            mask = dilation( mask, disk(expand) )   # Expanding masks from MRF

    # Read in file with X Y positions of stars to be masked
    f = open( regionsfile,'r')
    for line in f:
        x = int( float( line.split(',')[0] ) )
        y = int( float( line.split(',')[1] ) )
        rad = ( float( line.split(',')[2] ) )

        temp_mask = np.zeros( (ymax, xmax) )
        #!# Make this more efficient!
        for i in range( 0, ymax ):
            for j in range( 0, xmax ):
                if ( i - y )**2 + ( j - x )**2 <= rad**2:
                    temp_mask[ i,j ] = 1

        mask = mask + temp_mask

    f.close()

    # Making sure all non-zero pixels are set to 1
    mask = 1*(mask > 0)
  
    fits.writeto( outfile, mask, header, clobber=True )

    return mask


#############################################################################################################

# Update source mask with ds9 regions file (for circular regions)
#

def update_source_mask( mask, header, regionsfile, outfile ):

    ymax, xmax = np.shape( mask )

    # Read in file with X Y positions of stars to be masked
    f = open( regionsfile,'r')
    for line in f:
        x = int( float( line.split(',')[0] ) )
        y = int( float( line.split(',')[1] ) )
        rad = ( float( line.split(',')[2] ) )

        temp_mask = np.zeros( (ymax, xmax) )
        #!# Make this more efficient!
        for i in range( 0, ymax ):
            for j in range( 0, xmax ):
                if ( i - y )**2 + ( j - x )**2 <= rad**2:
                    temp_mask[ i,j ] = 1

        mask = mask + temp_mask

    f.close()

    # Making sure all non-zero pixels are set to 1
    mask = 1*(mask > 0)
  
    fits.writeto( outfile, mask, header, clobber=True )

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
                
            if len( vals ) > 1: 

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
# Assumes random errors and systematic sky errors (eg. due to large scale variations in bkgd)
# are independent. 
#
def calc_colour_error( g_flux, g_err, r_flux, r_err, g_sky, r_sky, 
                       g_sky_err, r_sky_err, g_sky_err_sys = 0, r_sky_err_sys = 0 ):
    
    dg = 2.5 * np.array(g_err) / ( (np.array(g_flux) - g_sky) * np.log(10.) )
    dr = 2.5 * np.array(r_err) / ( (np.array(r_flux) - r_sky) * np.log(10.) )
    
    dg_sky = 2.5 * g_sky_err / ( (np.array(g_flux) - g_sky) * np.log(10.) )
    dr_sky = 2.5 * r_sky_err / ( (np.array(r_flux) - r_sky) * np.log(10.) )

    dg_sky_sys = 2.5 * g_sky_err_sys / ( (np.array(g_flux) - g_sky) * np.log(10.) )
    dr_sky_sys = 2.5 * r_sky_err_sys / ( (np.array(r_flux) - r_sky) * np.log(10.) )
    
    d_gmr = np.sqrt(  np.square(dg) + np.square(dr) 
                    + np.square(dg_sky) + np.square(dr_sky) 
                    + np.square(dg_sky_sys) + np.square(dr_sky_sys) )
    
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


#############################################################################################################

# Measure sky level and uncertainty in median sky
#
import matplotlib.pyplot as plt

def sky_stats(masked_image, showHist=False):

    sky_pixels = masked_image.compressed()

    if len(sky_pixels) == 0:
        return np.nan, np.nan

    median = np.median(sky_pixels)

    med_err = np.std( sky_pixels ) / np.sqrt( len(sky_pixels) ) * np.sqrt( np.pi/2. )

    if showHist:
        plt.hist(sky_pixels, bins=100)
        plt.show()

    return median, med_err


#############################################################################################################

# Measure variation in image between azimuthal sectors
#
def measure_azimuthal_variation(image, sectors=8, initial_mask=None, center=None, theta0=None):

    ymax, xmax = np.shape(image)
    if center:
        y0 = center[0]
        x0 = center[1]
    else:
        y0 = int(ymax/2.)
        x0 = int(xmax/2.)

    if initial_mask is None:
        initial_mask = np.zeros((ymax, xmax)) # Starting with "blank" mask if there is no input mask

    if theta0 == None:
        theta0 = np.random.rand()*2.*np.pi # Random starting angle
    delta_theta = 2*np.pi / sectors

    skies = []
    sky_errs = []
    sky_noise = []
    central_theta = []

    for i in range(0,sectors):

        theta1 = theta0 + i*delta_theta
        theta2 = theta1 + delta_theta
        central_theta.append( (theta1 + theta2)/2. )

        sector_mask = make_sector_mask(image, theta1, theta2)
        temp_mask = initial_mask + sector_mask
        masked_image = np.ma.masked_where(temp_mask > 0, image)
        # plt.imshow(temp_mask); plt.show()   # For troubleshooting

        median, med_err = sky_stats(masked_image)
        sky_pixels = masked_image.compressed()

        skies.append(median)
        sky_errs.append(med_err)
        sky_noise.append(np.std(sky_pixels))

        print()
        print(f"Sector {i}: theta = {theta1*180./np.pi} - {theta2*180./np.pi} degrees")
        print(f"median sky = {median:.3f} +/- {med_err:.3f}")
        print(f"mean sky = {np.mean(sky_pixels):.3f} +/- {np.std(sky_pixels)/np.sqrt(len(sky_pixels)):.3f}")


    return skies, sky_errs, sky_noise, central_theta


#############################################################################################################

# Measure variation in image within annuli of increasing radius
#
def measure_radial_variation(image, sectors=9, rad1=600, rad2=1500, initial_mask=None, center=None):

    ymax, xmax = np.shape(image)
    if center:
        y0 = center[0]
        x0 = center[1]
    else:
        y0 = int(ymax/2.)
        x0 = int(xmax/2.)

    if np.any(np.isin(initial_mask, None)):
        initial_mask = np.zeros((ymax, xmax)) # Starting with "blank" mask if there is no input mask

    delta_rad = (rad2 - rad1)/sectors

    skies = []
    sky_errs = []
    sky_noise = []
    central_radii = []

    for i in range(0,sectors):

        r1 = rad1 + i*delta_rad
        r2 = r1 + delta_rad
        central_radii.append( (r1 + r2)/2. )

        annulus_mask = make_annulus_mask(image, r1, r2)
        temp_mask = initial_mask + annulus_mask
        masked_image = np.ma.masked_where(temp_mask > 0, image)
        #plt.imshow(temp_mask); plt.show()  # For troubleshooting

        median, med_err = sky_stats(masked_image)
        sky_pixels = masked_image.compressed()

        skies.append(median)
        sky_errs.append(med_err)
        sky_noise.append(np.std(sky_pixels))

        print()
        print(f"Sector {i}: radius {r1} - {r2} pixels")
        print(f"median sky = {median:.3f} +/- {med_err:.3f}")
        print(f"mean sky = {np.mean(sky_pixels):.3f} +/- {np.std(sky_pixels)/np.sqrt(len(sky_pixels)):.3f}")
        print(f"sky RMS = {np.std(sky_pixels):.3f}")


    return skies, sky_errs, sky_noise, central_radii


#############################################################################################################

# Measure variation in image within annuli of increasing radius
#
def find_best_sky(image, rad1=300, rad2=1000, initial_mask=None, center=None, width=20, step=20, baseline=5, thresh=1e-5, fulloutput=False, method="best_slope"):

    ymax, xmax = np.shape(image)
    if center:
        y0 = center[0]
        x0 = center[1]
    else:
        y0 = int(ymax/2.)
        x0 = int(xmax/2.)

    if np.any(np.isin(initial_mask, None)):
        initial_mask = np.zeros((ymax, xmax)) # Starting with "blank" mask if there is no input mask

    # Measure sky in each annulus
    skies = []
    sky_errs = []
    sky_noise = []
    central_radii = []

    r1 = rad1
    while r1 < rad2:

        r2 = r1 + width
        central_radii.append( (r1 + r2)/2. )

        annulus_mask = make_annulus_mask(image, r1, r2)
        temp_mask = initial_mask + annulus_mask
        masked_image = np.ma.masked_where(temp_mask > 0, image)

        median, med_err = sky_stats(masked_image)
        sky_pixels = masked_image.compressed()

        skies.append(median)
        sky_errs.append(med_err)
        sky_noise.append(np.std(sky_pixels))

        r1 = r1 + step

    # Determine local slope for each annulus
    slopes = []
    scores = []
    for i in range(0,len(skies)):

        # Want to avoid selecting missing sky values
        if np.isnan(skies[i]):
            slopes.append(99)
            scores.append(99)
            continue

        # Fit slope to sky values within +/- $baseline steps
        i_min = max(i-baseline, 0)
        i_max = min(i+baseline, len(skies)-1)
        
        subset_skies = np.array(skies[i_min:i_max])
        subset_radii = np.array(central_radii[i_min:i_max])
        subset_weights = np.array(1./np.array(sky_errs[i_min:i_max]))

        # Cleaning out any NaNs from sky values (due to masking)
        idx = np.isfinite(subset_skies) & np.isfinite(subset_weights)
        if np.sum(idx) < baseline:
            slopes.append(99)
            scores.append(99)
            continue

        # Fitting a line to selected subset of points to determine local slope
        try:
            params, res, _, _, _ = np.polyfit(subset_radii[idx], subset_skies[idx], 1, w=subset_weights[idx], full=True)
            slopes.append(np.abs(params[0]))
            scores.append(np.abs(params[0])*res/np.sum(idx))

        except ValueError:
            slopes.append(99)
            scores.append(99)

    # Checking for invalid method
    if method not in ["best_slope", "best_score", "thresh_slope", "thresh_score"]:
        print("Invalid method selected; defaulting to selecting sky by flattest local slope")
        method = "best_slope"

    # Select sky annulus with local slope closest to zero
    if method == "best_slope":
        best_i = slopes.index(min(slopes))

    # Select sky annulus with score (slope * residuals) closest to zero
    # Attempting to penalize symmetric peaks which can have "flat" local slope
    elif method == "best_score":
        best_i = scores.index(min(scores))
    
    # Select sky annulus with smallest radius that satisfies local flatness criteria
    # Defaults to selecting local slope closest to zero if none are below threshold
    elif method == "thresh_slope":
        try:
            best_sky = next(sky for sl, sky in zip(slopes, skies) if sl < thresh)
            best_i = skies.index(best_sky)
        except StopIteration:
            best_i = slopes.index(min(slopes))

    # Same as above, but threshold score (slope * residuals)
    elif method == "thresh_score":
        try:
            best_sky = next(sky for sc, sky in zip(scores, skies) if sc < thresh)
            best_i = skies.index(best_sky)
        except StopIteration:
            best_i = scores.index(min(scores))     

    # Return full lists of skies, errors, and scores, or just the selected sky value + error
    # Recommend full output as error returned is only standard error of median within radial bin
    # (Does not account for broader variation in sky background)
    if fulloutput:
        return skies, sky_errs, best_i, slopes, scores, central_radii

    else:
        return skies[best_i], sky_errs[best_i]

#############################################################################################################

# Quick plot of sky with selected radius marked

def plot_skies(skies, errs, radii, best_index):
    
    plt.errorbar(radii, skies, yerr=errs)
    plt.axvline(x=radii[best_index])
    plt.xlabel("Central annulus radius (pixels)", size=18)
    plt.ylabel("Median sky (ADU)", size=18)
    plt.show() 


#############################################################################################################

# Convenience functions for imposing contours, plotting, etc

def plot_isophotes( image, isolist, spacing = 5, log = True ):

    if log:
        plt.imshow( np.log10( image ), origin='lower' )
    else:
        plt.imshow( image, origin='lower' )
    
    rmax = isolist.sma[-1]
 
    for r in np.arange(5,rmax,spacing):
    
        iso = isolist.get_closest( r )
        x, y, = iso.sampled_coordinates()
        plt.plot(x, y, color='white')
    
    plt.show()
    
    
def fit_uniform_ellipses( image, x0, y0, phi, rmax = 0, eps = 0, integrmode = "bilinear" ):
    
    if rmax == 0:
        rmax = 1.5 * max( np.shape( image ) )
    
    # Temporary list to store instances of Isophote
    isolist_fixed_ = []

    for rad in np.arange(1,rmax):

        # Fixed ellipse geometry + current semi-major axis length
        g = EllipseGeometry(x0, y0, rad, eps, phi)

        # Sample the image on the fixed ellipse
        sample = EllipseSample(image, g.sma, geometry=g, 
                               integrmode=integrmode, sclip=3.0, nclip=3)
        sample.update(g)

        # Storing isophote in temporary list; arguments other than "sample" are arbitrary
        iso_ = Isophote(sample, 0, True, 0)
        isolist_fixed_.append(iso_)

    isolist_fixed = IsophoteList(isolist_fixed_)
    
    return isolist_fixed


def impose_isophotes( image, isolist_in, sclip=3.0, nclip=2, integrmode='bilinear', central_isophote=False ):
    
    # Temporary list to store instances of Isophote
    isolist_temp = []

    # Loop over the IsophoteList instance 

    # First isophote requires special treatment. It's an instance of CentralEllipsePixel, 
    # which requires special sampling by the CentralEllipseSample subclass.
    #!# Note that quadrant profiles can't have the central isophote sampled due to masking
    if central_isophote:
        sample = CentralEllipseSample(image, 0., geometry=isolist_in[0].sample.geometry)
        fitter = CentralEllipseFitter(sample)
        center = fitter.fit()
        isolist_temp.append(center)

    # Need to make sure that first isophote is skipped *IFF* it is a central sample (sma==0)    
    if isolist_in.sma[0] == 0:
        isolist_loop = isolist_in[1:]
    else:
        isolist_loop = isolist_in

    for iso in isolist_loop:

        g = iso.sample.geometry

        # Sample the low-S/N image at the same geometry. 
        # Should use same integration mode and same sigma-clipping settings
        sample = EllipseSample(image, g.sma, geometry=g, 
                               integrmode=integrmode, sclip=sclip, nclip=nclip)
        sample.update(g)

        iso_ = Isophote(sample, 0, True, 0)
        isolist_temp.append(iso_)

    return IsophoteList(isolist_temp)


def roundify_outer_isophotes(image, isolist_in, r_start, r_stop=None, target_eps=0.0, sclip=3.0, nclip=2, integrmode='bilinear', central_isophote=False):
    
    # Temporary list to store instances of Isophote
    isolist_temp = []
    
    if not r_stop:
        r_stop = 2*r_start

    # Updating start and stop radii according to closest existing isophotes
    iso1 = isolist_in.get_closest(r_start)
    iso2 = isolist_in.get_closest(r_stop)
    r_start = iso1.sma
    r_stop = iso2.sma
    
    eps_0 = iso1.eps
    
    # Loop over the IsophoteList instance 
    #
    # First isophote requires special treatment. It's an instance of CentralEllipsePixel, 
    # which requires special sampling by the CentralEllipseSample subclass.
    #!# Note that quadrant profiles can't have the central isophote sampled due to masking
    if central_isophote:
        sample = CentralEllipseSample(image, 0., geometry=isolist_in[0].sample.geometry)
        fitter = CentralEllipseFitter(sample)
        center = fitter.fit()
        isolist_temp.append(center)

    for iso in isolist_in[1:]:

        g = iso.sample.geometry
        
        if g.sma < r_start:

            # Sample the low-S/N image at the same geometry. 
            # Should use same integration mode and same sigma-clipping settings
            sample = EllipseSample(image, g.sma, geometry=g, 
                                   integrmode=integrmode, sclip=sclip, nclip=nclip)
            
            sample.update(g)
            
        elif g.sma >= r_start and g.sma <= r_stop:
            
            # Determine intermediate ellipticity
            # Starting off with linear decline
            new_eps = eps_0 - (eps_0 - target_eps)*np.sin((g.sma - r_start)/(r_stop - r_start)*np.pi/2.) 
            
            # Define new EllipseGeometry (all the same but new ellipticity)
            new_g = EllipseGeometry(iso.x0,
                                    iso.y0,
                                    iso.sma,
                                    new_eps,
                                    iso.pa,
                                    g.astep,
                                    g.linear_growth)
            
            sample = EllipseSample(image, new_g.sma, geometry=new_g, 
                                   integrmode=integrmode, sclip=sclip, nclip=nclip)            
            
            sample.update(new_g)
            
        else:
            
            # Define new EllipseGeometry (all the same but new ellipticity)
            new_g = EllipseGeometry(iso.x0,
                                    iso.y0,
                                    iso.sma,
                                    target_eps,
                                    iso.pa,
                                    g.astep,
                                    g.linear_growth)
            
            sample = EllipseSample(image, new_g.sma, geometry=new_g, 
                                   integrmode=integrmode, sclip=sclip, nclip=nclip)            
            
            sample.update(new_g)            
            

        iso_ = Isophote(sample, 0, True, 0)
        isolist_temp.append(iso_)

    return IsophoteList(isolist_temp)


# Get area of isophotes using unmasked reference image
def get_iso_areas(isophotes, image):

    areas = np.zeros(isophotes.intens.shape)

    # Creating dummy IsophoteList with unmasked image
    if isophotes.sma[0] == 0:
        dummy = impose_isophotes(image, isophotes, central_isophote=True)
    else:
        dummy = impose_isophotes(image, isophotes)
        areas[0] = dummy.npix_e[0]

    # npix_e[i]: area in pixels**2 contained inside ith isophote
    # Subtracting area contained in previous isophote to get annular area
    for i in range(1, len(areas)):       
        areas[i] = dummy.npix_e[i] - dummy.npix_e[i-1]
        
    return areas


def flux_to_mags(flux, sky, zeropoint, pix_size=2.5, default=np.nan):
    
    flux = np.asarray(flux)
    scalar_input = False

    if flux.ndim == 0:
        scalar_input = True
        flux = flux[np.newaxis]

    mags = np.where((flux-sky)>0, 
                    -2.5*np.log10(flux - sky) + zeropoint + 5*np.log10(pix_size),
                    default)

    # Making sure to return scalar value if input flux is scalar, not array
    if scalar_input:
        return np.squeeze(mags)
    else:
        return mags


def integrate_profile(profile, area, prof_err, area_err, valtype="log"):
    
    if valtype == "linear":      
        temp = [a*p for a, p in zip(area, profile) if a > 0 and np.isfinite(p)]
        temp_err = [(a*p)**2 * ((area_err/a)**2 + (p_err/p)**2) 
                    for a, p, p_err in zip(area, profile, prof_err) 
                    if a > 0 and np.isfinite(p)] 
        return np.sum(temp), np.sqrt(np.sum(temp_err))
    
    elif valtype == "log":
        temp = [a*np.power(10, p) for a, p in zip(area, profile) if a > 0 and np.isfinite(p)]
        temp_err = [(a*np.power(10, p))**2 * ((area_err/a)**2 + (p_err/np.log10(np.e))**2) 
                    for a, p, p_err in zip(area, profile, prof_err) 
                    if a > 0 and np.isfinite(p)] 
        return np.log10(np.sum(temp)), np.sqrt(np.sum(temp_err))*np.log10(np.e)/np.sum(temp)
        
    elif valtype == "mag":
        raise NotImplementedError 

    else:
        raise ValueError("Invalid profile value type specified; valid types are linear, log, and mag")


#############################################################################################################

# Class to manage splitting image into quadrants for profile extraction

class QuadrantProfiles:
    
    def __init__(self, image, source_mask, master_isophotes, center, theta0, verbose=True):
        
        self.quadMask = np.zeros((4,) + np.shape(image))
        self.quadIsophotes = []
        self.positionAngle = theta0
        self.center = center
        
        for i in range(0,4):
            
            theta1 = theta0 + i*np.pi/2.
            theta2 = theta1 + np.pi/2.
            
            # Generate quadrant masks
            temp_mask = make_sector_mask(image, theta1, theta2, center=center)
            temp_mask = temp_mask + source_mask
            if verbose:
                print(f"Quadrant and source mask for quad {i}:")
                plt.imshow(temp_mask, origin="lower")
                plt.show()
                
            self.quadMask[i] = 1*((temp_mask + source_mask) > 0)
        
            # Generate quadrant IsophoteList objects
            temp_image = np.ma.masked_where(self.quadMask[i] > 0, image)
            self.quadIsophotes.append(impose_isophotes(temp_image, master_isophotes))      
        
        # Sky and sky error is initialized empty
        self.quadSky = np.zeros((4))
        self.quadSkyErr = np.zeros((4))
        self.quadSkySysErr = np.zeros((4))
        
        # Other quantities to initialize
        self.quadLimits = np.ones((4))*99999    # Arbitrarily large initial limits
        #######        
    
    def measure_sky(self, image, radius_inner, radius_outer, bin_width=20, baseline=5, method="best_slope", thresh=1e-3, mask=None, default=None, tweak=False):
        
        # Tweak-mode gives access to slope/score to choose best method and/or threshold for target
        if tweak:
            slopes = []
            scores = []
            radii = []
        
        for i in range(0,4):
            
            print(f"Measuring sky for quadrant {i} ...")

            # May need to specify new mask for sky measurement
            # ex. if galaxy has a large stream
            if mask is not None:
                theta1 = self.positionAngle + i*np.pi/2.
                theta2 = theta1 + np.pi/2.
            
                # Generate quadrant masks
                temp_mask = make_sector_mask(image, theta1, theta2, center=self.center)
                temp_mask = 1*((temp_mask + mask) > 0)

            else:
                temp_mask = self.quadMask[i]           
            
            sky, err, best_i, slope, score, rad = find_best_sky(image, 
                                                                rad1=radius_inner, 
                                                                rad2=radius_outer, 
                                                                initial_mask=temp_mask, 
                                                                center=self.center, 
                                                                width=bin_width, 
                                                                step=bin_width, 
                                                                baseline=baseline, 
                                                                fulloutput=True, 
                                                                method=method,
                                                                thresh=thresh)
            if not np.isnan(sky[best_i]):
                self.quadSky[i] = sky[best_i]
                self.quadSkyErr[i] = err[best_i]
                self.quadSkySysErr[i] = np.nanstd(sky)

            elif np.isnan(sky[best_i]) and default is not None:
                self.quadSky[i] = default[0]
                self.quadSkyErr[i] = default[1]
                self.quadSkySysErr[i] = default[2]

            else:
                print("WARNING: no sky measurement available for this quadrant.")
                print("Try decreasing the inner radius limit or specify a default sky value to adopt.")
                print("eg. default=[sky, sky_err, sky_sys_err]")
            
            # Quick plot of the sky radial profile + vertical line indicating selected value
            plot_skies(sky, err, rad, best_i)
            
            if tweak:
                slopes.append(slope)
                scores.append(score)
                radii.append(rad) 
                
        if tweak:
            return slopes, scores, radii
        else:
            return
    
    
    def set_quad_limits(self, limits):
        
        self.quadLimits = limits
        
        
    def get_radii(self):
        
        return np.unique(np.concatenate((self.quadIsophotes[0].sma,
                                         self.quadIsophotes[1].sma,
                                         self.quadIsophotes[2].sma,
                                         self.quadIsophotes[3].sma)))
        
    
    def combine_quad_profiles(self, master_isophotes, master_sky, zeropoint, pix_size=2.5):
    
        radii = self.get_radii()
    
        SB_mag = []
        SB_err_p = []
        SB_err_m = []
        SB_err_ps = []
        SB_err_ms = []

        # Storing profile/error in flux units internally
        # (Important intermediates for colour error calculations
        self.combinedFlux = []
        self.combinedFluxErr = []
        self.combinedFluxSysErr = []
    
        for r in radii:
        
            temp_flux = 0
            temp_npix = 0
            temp_err = 0
            temp_err_sys = 0
        
            for i in range(0,4): 
                
                if r in self.quadIsophotes[i].sma and r < self.quadLimits[i]/pix_size:
                
                    index = np.where(self.quadIsophotes[i].sma == r)[0][0]
                    if not np.isnan(self.quadIsophotes[i].intens[index]):
                    
                        n_good = self.quadIsophotes[i].ndata[index]
                        temp_flux += (self.quadIsophotes[i].intens[index] - self.quadSky[i])*n_good
                        temp_npix += n_good
                        temp_err += n_good**2 * (self.quadIsophotes[i].int_err[index]**2 + self.quadSkyErr[i]**2)
                        temp_err_sys += n_good**2 * (self.quadIsophotes[i].int_err[index]**2 + self.quadSkyErr[i]**2 + self.quadSkySysErr[i]**2)
                  
            # If at least one quadrant profile has good pixels at current radius
            if temp_npix != 0:
            
                temp_flux = temp_flux/(temp_npix) # Renormalizing weighted SB 
                temp_SB = flux_to_mags(temp_flux, 0, zeropoint)
                SB_mag.append(temp_SB)
        
                temp_err = np.sqrt(temp_err)/(temp_npix)
                temp_err_sys = np.sqrt(temp_err_sys)/(temp_npix)
        
                # Asymmetric error bars (error in flux shown on log scale)
                # Setting default=40 mag/arcsec^2 for lower error bounds, in case they are undefined
                SB_err_m.append(flux_to_mags(temp_flux - temp_err, 0, zeropoint, default=40))
                SB_err_p.append(flux_to_mags(temp_flux + temp_err, 0, zeropoint))
                SB_err_ms.append(flux_to_mags(temp_flux - temp_err_sys, 0, zeropoint, default=40))
                SB_err_ps.append(flux_to_mags(temp_flux + temp_err_sys, 0, zeropoint))
        
                # Symmetric error bars (error in log(flux))
                #SB_err_all_r.append(2.5*np.log10(np.e)/temp_flux * temp_err)
                #SB_err_all_rs.append(2.5*np.log10(np.e)/temp_flux * temp_err_sys)

                self.combinedFlux.append(temp_flux)
                self.combinedFluxErr.append(temp_err)
                self.combinedFluxSysErr.append(temp_err_sys)
   
            # If no quadrant profiles have good pixels but not beyond max radius
            # This should be triggered for points close to the center
            elif r < max(self.quadLimits)/pix_size:
        
                i = np.where(master_isophotes.sma == r)[0][0]
            
                temp_err = np.sqrt(master_isophotes.int_err[i]**2 + master_sky[1]**2)
                temp_err_sys = np.sqrt(master_isophotes.int_err[i]**2 + master_sky[1]**2 + master_sky[2]**2)
               
                SB_mag.append(flux_to_mags(master_isophotes.intens[i], master_sky[0], zeropoint))     
                SB_err_p.append(flux_to_mags(master_isophotes.intens[i] + temp_err, master_sky[0], zeropoint))
                SB_err_m.append(flux_to_mags(master_isophotes.intens[i] - temp_err, master_sky[0], zeropoint, default=40))
                SB_err_ps.append(flux_to_mags(master_isophotes.intens[i] + temp_err_sys, master_sky[0], zeropoint))
                SB_err_ms.append(flux_to_mags(master_isophotes.intens[i] - temp_err_sys, master_sky[0], zeropoint, default=40))

                self.combinedFlux.append(master_isophotes.intens[i] - master_sky[0])
                self.combinedFluxErr.append(temp_err)
                self.combinedFluxSysErr.append(temp_err_sys)
        
            # No quadrants with good pixels, beyond maximum radius
            else:
        
                SB_mag.append(np.nan)
                SB_err_p.append(np.nan)
                SB_err_m.append(np.nan)
                SB_err_ps.append(np.nan)
                SB_err_ms.append(np.nan)

                self.combinedFlux.append(np.nan)
                self.combinedFluxErr.append(np.nan)
                self.combinedFluxSysErr.append(np.nan)
    
        # Adding in central measurement from master IsophoteList
        if radii[0] != 0 and master_isophotes.sma[0] == 0:
    
            radii = np.concatenate(([0], radii))
            temp_err = np.sqrt(master_isophotes.int_err[0]**2 + master_sky[1]**2)
            temp_err_sys = np.sqrt(master_isophotes.int_err[0]**2 + master_sky[1]**2 + master_sky[2]**2)
    
            SB_mag.insert(0, flux_to_mags(master_isophotes.intens[0], master_sky[0], zeropoint))
            SB_err_p.insert(0, flux_to_mags(master_isophotes.intens[0] + temp_err, master_sky[0], zeropoint))
            SB_err_m.insert(0, flux_to_mags(master_isophotes.intens[0] - temp_err, master_sky[0], zeropoint, default=40))
            SB_err_ps.insert(0, flux_to_mags(master_isophotes.intens[0] + temp_err_sys, master_sky[0], zeropoint))
            SB_err_ms.insert(0, flux_to_mags(master_isophotes.intens[0] - temp_err_sys, master_sky[0], zeropoint, default=40))

            self.combinedFlux.insert(0, master_isophotes.intens[0] - master_sky[0])
            self.combinedFluxErr.insert(0, temp_err)
            self.combinedFluxSysErr.insert(0, temp_err_sys)

        self.combinedFlux = np.array(self.combinedFlux)
        self.combinedFluxErr = np.array(self.combinedFluxErr)
        self.combinedFluxSysErr = np.array(self.combinedFluxSysErr)


        return radii, np.array(SB_mag), np.array(SB_err_p), np.array(SB_err_m), np.array(SB_err_ps), np.array(SB_err_ms)
    
    def show_quadrants(self):
        
        fig, ax = plt.subplots(2, 2)
        
        ax[0,0].imshow(self.quadMask[0], origin="lower")
        ax[0,1].imshow(self.quadMask[1], origin="lower")
        ax[1,0].imshow(self.quadMask[2], origin="lower")
        ax[1,1].imshow(self.quadMask[3], origin="lower")
        
        fig.set_size_inches(8,8)
        plt.show()
