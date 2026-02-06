import numpy as np
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import gridspec
import scipy.optimize as opt
import scipy.special as sp
from pathlib import Path
import os
import copy
from main_code.subroutines import fits_handling as fh
from main_code.subroutines import misc_functions as mf
from main_code.subroutines import array_calculations as ac
from os import makedirs
import copy


def read_pi_source_list_cd(path):
    """This function reads a previously generated candidate sourcelist to obtain a list of sources to examine
    and various basic parameters about them.
    
    Returns:
        pi_data (dict): A dictionary the imported source features. The keys for the dictionary are:
            'len': returns the size of the source list, or the number of sources to go through
            'lmax': the longitude coordinate of the peak of the source
            'bmax': the latitude coordinates of the peak
            'xpixmax': The x coordinate in pixel values
            'ypixmax': The y coordinate in pixel values
            'pimax': The PI value of the peak
            'simax': The stokes I value of the peak
            'stonax': The signal to noise of the pixel
            
    """
    with open(path) as pi_source_list:
        pi_source_list = pi_source_list.readlines()[6:]  # Skipping the header of the file, we only want the data

    pi_source_list = np.array(pi_source_list)
    sdata = pi_source_list.size
    print("sdata: ", sdata)

    lmax = []
    bmax = []
    xpixmax = []
    ypixmax = []
    pimax = []
    simax = []
    stonmax = []
    for line in pi_source_list:
        split = line.split()
        if len(split[0])>7:
            Lmax = split[0][:7]
            Bmax = split[0][7:]
            split[0] = Lmax
            split.insert(1, Bmax)
            # print("l53 Lmax, Bmax: ", Lmax, Bmax)
            # quit()
        
        
        lmax.append(float(split[0]))
        bmax.append(float(split[1]))
        xpixmax.append(int(split[2]))
        ypixmax.append(int(split[3]))
        pimax.append(float(split[4]))
        simax.append(float(split[5]))
        stonmax.append(float(split[6]))

    # Store each parameter as a list, where each element is a source
    pi_data = {'len': sdata,
               'lmax': np.array(lmax),
               'bmax': np.array(bmax),
               'xpixmax': np.array(xpixmax),
               'ypixmax': np.array(ypixmax),
               'pimax': np.array(pimax),
               'simax': np.array(simax),
               'stonmax': np.array(stonmax)}
    
    # Adding by Ciara Chisholm, May 1st 2024 
    # print(xpixmax,"xpixmax")
    # print("l: ", lmax)
    # print("b: ", bmax)
    return pi_data


def read_qu_data_cve(directory_path, mosaic_name):
    """This function reads data and header information from fits images from a particular mosaic.
    
    Returns:
        stokes (dict): a dictionary containing all the Stokes images for each band 
            including Stokes I, Q_A, Q_B, Q_C, Q_D, U_A, U_B, U_C, U_D
        header (dict): a dictionary that contains the header for each of the 
            Stokes images. 
    """
    stokes = {}
    header = {}

    for band in ['I', 'Q_A', 'Q_B', 'Q_C', 'Q_D', 'U_A', 'U_B', 'U_C', 'U_D']:
        stokes_header, stokes_data = fh.read_fits(f'{directory_path}{mosaic_name}/m{mosaic_name}_1420_MHz_{band}_image.fits')

        stokes[band] = stokes_data
        header[band] = stokes_header
        
    
    
    return stokes, header


def read_chitable(chitable_directory):
    """This function reads the values of the Chi table from a given file.
    """
    chitable_name = '/Chi_Table_05.dat'
    chitable_path = f'{chitable_directory}{chitable_name}'

    print(f'\n\nCHI-TABLE being used: {chitable_name}')

    with open(chitable_path) as chitable:
        chitable = chitable.readlines()

    degrees = []
    chi2 = []
    redchi2 = []
    for line in chitable:
        split = line.split()
        degrees.append(int(split[0]))
        chi2.append(float(split[1]))
        redchi2.append(float(split[2]))

    return degrees, chi2, np.array(redchi2)


def check_freq2(freq_a, freq_b, freq_c, freq_d, header_qa, header_qb, header_qc, header_qd):
    """This function checks to make sure the frequencies of the 4 Q bands equal the frequencies of the 4 U bands.
    """
    freq_qa = float(header_qa['OBSFREQ'])
    freq_qb = float(header_qb['OBSFREQ'])
    freq_qc = float(header_qc['OBSFREQ'])
    freq_qd = float(header_qd['OBSFREQ'])

    freq_ok = True

    if freq_a != freq_qa:
        print(f'\nError in frequency for Band A')
        print(f'Q: {freq_qa}')
        print(f'U: {freq_a}')
        freq_ok = False

    if freq_b != freq_qb:
        print(f'\nError in frequency for Band B')
        print(f'Q: {freq_qb}')
        print(f'U: {freq_b}')
        freq_ok = False

    if freq_c != freq_qc:
        print(f'\nError in frequency for Band C')
        print(f'Q: {freq_qc}')
        print(f'U: {freq_c}')
        freq_ok = False

    if freq_a != freq_qa:
        print(f'\nError in frequency for Band D')
        print(f'Q: {freq_qd}')
        print(f'U: {freq_d}')
        freq_ok = False

    if 27400000 > (freq_qd - freq_qa) > 27800000:  # This condition will never be met, something can't be less than 27.4M and greater than 27.8M
        print(f'Error with frequency separation')
        print(f'freq_QD: {freq_qd}')
        print(f'freq_DA: {freq_qa}')
        print(f'diff: {freq_qd - freq_qa}')

    return freq_ok


# *************************************
# POL INT MAP
# *************************************


def plot_pol_int_map(x_l_2, y_b_2, data_fit, levels, mosaic_name, num, x_gauss_rot, y_gauss_rot, x_center_gauss, y_center_gauss,
                     x_long, y_lat, x_loc, y_loc, x_fwxm_ae, y_fwxm_ae, x_fwxm_se, y_fwxm_se, rm_text, rm_err_text, chi_string, 
                     chitable_string, m_string, n_pixels, passfail, ax,light_background=True):
    """This function plots the Polarised Intensity Map.
    """

    
    if light_background:
        green, yellow, blue,black_or_white = "limegreen", "goldenrod", "royalblue", "black"
    else:
        green, yellow, blue, black_or_white = "lime", "yellow", "cyan", "white"
        
    # ax.ticklabel_format(style='plain')
    ax.contour(x_l_2, y_b_2, data_fit, levels=levels, colors='darkgrey', linewidths=1)
    # ax.set_title('Pol. Int. Map')
    ax.set_title('Polarised Intensity Map')
    ax.set_xlim(np.max(x_l_2), np.min(x_l_2))
    ax.set_ylim(np.min(y_b_2), np.max(y_b_2))
    # Units in the axis labels added by Ciara Chisholm June 4th 2024
    ax.set_xlabel('Longitude ($\degree$)')
    ax.set_ylabel('Latitude ($\degree$)')

    # ax.text(-0.15, -0.1, mosaic_name, transform=ax.transAxes, color='darkorange')  # ax.transAxes allows us to give the location of the text in a normalised range from
    # ax.text(-0.15, -0.15, num, transform=ax.transAxes, color=black_or_white)  # 0 to 1 instead of using the units of the data
    
    
    ax.scatter(x_gauss_rot + x_center_gauss, y_gauss_rot + y_center_gauss, s=2, color=green)  # psym=1, symsize=0.2
    pixel_width = x_long[y_loc, x_loc] - x_long[y_loc+1, x_loc+1]
    print("pixel_width: ", pixel_width)
    ax.text(x_long[y_loc, x_loc], y_lat[y_loc, x_loc], '*', color=green,)  # The location of this asterix is given in data space, not in the normalised 0 to 1 range


    # PLOTTING ANNULUS EDGE:
    ax.scatter(x_fwxm_ae, y_fwxm_ae, s=2, color='red')

    # PLOTTING SOURCE EDGE:
    ax.scatter(x_fwxm_se, y_fwxm_se, s=2, color=yellow) 
    

    ax.text(0.1, 0.85, 'RM:', transform=ax.transAxes, fontweight='bold')
    ax.text(0.1, 0.8, f'{rm_text}$\\pm${rm_err_text} rad/m$^2$', transform=ax.transAxes, fontweight='bold')

    # Chi square label
    ax.text(0.47, 0.85, '$\\chi^2$:', transform=ax.transAxes, fontweight='bold')
    ax.text(0.47, 0.8, chi_string, transform=ax.transAxes, color=blue, fontweight='bold')

    ax.text(0.73, 0.85, '$\\chi^2$ (table):', transform=ax.transAxes, fontweight='bold')
    ax.text(0.73, 0.8, chitable_string, transform=ax.transAxes, color=yellow, fontweight='bold') 

    # M label
    ax.text(0.1, 0.15, 'M:', transform=ax.transAxes, fontweight='bold')
    ax.text(0.1, 0.1, m_string, transform=ax.transAxes, color=blue, fontweight='bold')

    # Pixels label
    ax.text(0.73, 0.15, 'pixels', transform=ax.transAxes, fontweight='bold')
    ax.text(0.73, 0.1, str(n_pixels), transform=ax.transAxes, color=green, fontweight='bold')

    if passfail:
        ax.text(0.1, 0.5, 'PASS', transform=ax.transAxes, color=green, fontsize='large', fontweight='bold')
    else:
        ax.text(0.1, 0.5, 'FAIL', transform=ax.transAxes, color='red', fontsize='large', fontweight='bold')

    ####Added by Ciara Chisholm May 5th 2024
    ####all Lines below Were Added by Ciara Chisholm May 5th 2024
    # Used to turn off scientific notation on the x axis
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
     


# *************************************
# STOKES I MAP
# *************************************


def plot_stokes_i_map(long_arr, lat_arr, data, levels, x_gauss_rot1, y_gauss_rot1, x_gauss_rot2, y_gauss_rot2,
                      gauss_parameters, source_flag, x_long, y_lat, x_pix_max, y_pix_max, npix, ax,light_background=True):
    """This function plots the Stokes I Map.
    """
   
    
    # Setting the colors (in this case green) depending on the background. 
    if light_background:
        green = "limegreen"
    else: # dark background
        green = "lime"
    
    
    ax.set_title('Stokes I Map')
    # Units in the axis labels added by Ciara Chisholm June 4th 2024
    ax.set_xlabel('Longitude ($\degree$)')
    ax.set_ylabel('Latitude ($\degree$)')
    ax.set_xlim(np.max(long_arr), np.min(long_arr))
    ax.set_ylim(np.min(lat_arr), np.max(lat_arr))

    ax.contour(long_arr, lat_arr, data, levels=levels, colors='darkgrey', linewidths=1,)

    # Over the same plot
    ax.contour(long_arr, lat_arr, data, levels=levels, colors='darkgrey', linewidths=1)
    ax.scatter(x_gauss_rot1 + gauss_parameters['x_mean'], y_gauss_rot1 + gauss_parameters['y_mean'], s=2, color=green) 

    # Plot locations of all candidates:
    ax.text(x_long[0, x_pix_max], y_lat[y_pix_max, 0], '*', color='magenta')
    ax.text(long_arr[0, npix - 1], lat_arr[npix - 1, 0], '*', color=green) 

    # Still the same plot!
    # The following four lines were commented out by Ciara Chisholm October 22nd 2024, and replaced by the following two lines 
    # ax.scatter(x_gauss_rot2 + long_arr[npix, npix] - 0.08, y_gauss_rot2 + lat_arr[npix, npix], s=2, clip_on=False, color='red')
    # ax.text(long_arr[npix, npix] - 0.068, lat_arr[npix, npix] + 0.02, 'Beam FWHM', color='red', fontweight='bold')
    
    # ax.text(1.125, 0.2, 'Source Flag  ', transform=ax.transAxes, color=green, fontweight='bold')
    
    # ax.text(1.2, 0.1, str(source_flag), transform=ax.transAxes, size='xx-large', fontweight='bold', color=green)

    ax.text(1.025, 0.5, 'Source Flag  ', transform=ax.transAxes, color=green, fontweight='bold')
    
    ax.text(1.1, 0.4, str(source_flag), transform=ax.transAxes, size='xx-large', fontweight='bold', color=green)

    ax.grid(False)

    ####Added by Ciara Chisholm May 5th 2024
    ####all Lines below Were Added by Ciara Chisholm May 5th 2024
    # Used to turn off scientific notation on the x axis
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
      


# *************************************
# PEAK PIXEL LINEAR FIT
# *************************************


def plot_peak_pixel_linear_fit(lx, pol_ang, wrapped_pol_ang, rm_pix, drm, pol_err, predicted, probfit, ax,light_background=True):
    """This function plots the Peak Pixel Linear Fit.
    """
   
    # the following line was added by Ciara Chisholm on June 3rd 2024
    all_angles = np.concatenate((pol_ang, wrapped_pol_ang,), axis=0)
    
    if light_background:
        blue, green, error_bar_color = "royalblue", "limegreen", "black"
    else:
        blue, green, error_bar_color = "royalblue", "limegreen", "white"

    ax.set_title(f'Peak Pixel Linear Fit\nPixel RM = {mf.nround(rm_pix)}$\\pm${mf.nround(drm)}')
    # Units in the axis labels added by Ciara Chisholm June 4th 2024
    ax.set_xlabel('λ$^2$ (m$^2$)', fontsize = 13)
    ax.set_ylabel('Pol. Angle ($\degree$)', fontsize = 13)
    # ax.set_xlim(0.0435, 0.0455)
    # the following range changed from the previous line to the next by Ciara Chisholm June 3rd 2024
    ax.set_xlim(0.0435, 0.0457)
    # ax.set_ylim(np.min(pol_ang) - 5, np.max(pol_ang) + 5)
    #The line below was changed from the one above by Ciara Chisholm June 3rd 2024
    ax.set_ylim(np.min(all_angles) - 5, np.max(all_angles) + 5)
    # ax.set_xticks([0.0435, 0.0440, 0.0445, 0.0450])
    # the following lines were changed by Ciara Chisholm on June 3rd 2024
    ax.tick_params(axis='x', which='major', labelsize=9)
    ax.set_xticks(lx)
    ax.set_xticklabels([f"A\n {lx[0]:^9.4f}" , f"B\n {lx[1]:^9.4f}" ,
                        f"C\n {lx[2]:^9.4f}", f"D\n {lx[3]:^9.4f}",])
    # Scatter plots
    ax.plot(lx, wrapped_pol_ang, 'o', markersize=20, color=blue, label= "Wrapped angle")
    ax.plot(lx, pol_ang, '^', markersize=15, color=green, label= "Unwrapped angle")
    
    
   
    lgd = ax.legend( loc= "lower right",bbox_to_anchor=(1,1.02),  fontsize=6, markerscale=0.4)
    # lgd = ax.legend( loc= "lower right",  fontsize=8, markerscale=0.45)
    # lgd = ax.legend( loc= "best", fontsize=6, markerscale=0.4)


    # Error bars
    pol_err = np.array([pol_err[0][0], pol_err[1][0], pol_err[2][0], pol_err[3][0]])
    # ax.errorbar(lx, pol_ang, yerr=(2 * pol_err), capsize=5, ecolor=error_bar_color, fmt='none')
    # The following line was added by Ciara Chisholm on 23/05/23 
    ax.errorbar(lx, pol_ang, yerr=( pol_err), capsize=5, ecolor=error_bar_color, fmt='none')


    # x_range = np.arange(101) * (0.0455 - 0.0435) / 100 + 0.0435
    # The next line was changed from the previous by Ciara Chisholm on June 3rd 2024
    x_range = np.arange(101) * (0.0457 - 0.0435) / 100 + 0.0435

    # Fitted slope
    ax.plot(x_range, predicted, color='red', linestyle='--')
    
    # The following lines were removed by Ciara Chisholm on June 3rd 2024
    # ax.plot(x_range, predicted + 180, color='magenta', linestyle='--')
    # ax.plot(x_range, predicted - 180, color='yellow', linestyle='--')

    ax.text(0.35, -0.23, f'Pixel Probfit = {mf.nround(probfit * 100)}%', transform=ax.transAxes)
    


# *************************************
# RM MAP
# *************************************


def plot_rm_map(rm_data, rm_text, rm_err_text, pa_text, pa_err_text, pi_units, gauss_parameters,
                x_long, y_lat, pi, x_fwhm, y_fwhm, pass_fail, wprob_t, ax,light_background=True):
    """This function plots the Rotation Measure Map.
    """
  
    if light_background:
        green, yellow, blue, nan_color, CMAP = "limegreen", "goldenrod", "royalblue", "white", 'RdBu_r'
    else:
        green, yellow, blue, nan_color,CMAP = "lime", "yellow", "cyan", "black", "bwr"
    
    ax.set_title(f'RM Map\nRM = {rm_text}$\\pm${rm_err_text}, dPA = {pa_text}$\\pm${pa_err_text}')
    # Units in the axis labels added by Ciara Chisholm June 4th 2024
    ax.set_xlabel('Longitude ($\degree$)')
    ax.set_ylabel('Latitude ($\degree$)')
    
    
    # Added by Ciara Chisholm May 5th 2024
    # xticks= ax.get_xticklabels()
    
    # print("L382: xticsk: ", xticks)
    
    # new_ticks = []
    # for t in xticks:
    #     new_ticks+= [f"%.2f"%t]
    # ax.set_xticks(xticks)
    # ax.set_xticklabels(new_ticks)
    
    masked_array = np.ma.masked_where(rm_data == 0, rm_data)
    
    # print("masked_array: ", masked_array)
    # cmap = copy.copy(cm.get_cmap('RdBu'))
    # The previous line was changed to the following line by Ciara Chisholm on May 27th 2024
    cmap = copy.copy(cm.get_cmap(CMAP))
    cmap.set_bad(color=nan_color)

    # Heatmap
    im = ax.pcolor(x_long, y_lat, masked_array, vmin=-500, vmax=500, cmap=cmap, shading='auto')

    # Make the x-axis have the smallest values towards the right side 
    plt.gca().invert_xaxis()

    # Overplotting the contours of PI
    levels = 3.0
    if pi_units != 'mJy/beam':
        levels = 0.3

    if gauss_parameters['amp'] / 2.0 > levels:
        levels = np.array([levels, gauss_parameters['amp'] / 2.0])
    else:
        levels = np.array([gauss_parameters['amp'] / 2.0])

    ax.contour(x_long, y_lat, pi, levels=levels, colors='darkgrey')

    # Overplotting the fwhm
    ax.scatter(x_fwhm, y_fwhm, s=2, color=green)

    # pad and rotation added by Ciara Chisholm on October 22nd 2024
    cbar = ax.figure.colorbar(im, ax=ax, ticks=[-500, -250, 0, 250, 500], pad = -0.2)
    cbar.ax.set_ylabel('RM (rad/m$^2$)', rotation=270, va="bottom")
    

    # ax.text(1.25, 0.977, 'rad/m$^2$', transform=ax.transAxes )
    # ax.text(1.29, -0.025, 'rad/m$^2$', transform=ax.transAxes)

    if pass_fail:
        ax.text(-0.2, -0.22, 'PASS', transform=ax.transAxes, color=green, fontsize='large', fontweight='bold')
    else:
        ax.text(-0.2, -0.22, 'FAIL', transform=ax.transAxes, color='red', fontsize='large', fontweight='bold')

    ax.text(0.51, -0.22, f'Avg. Linfit Probfit = {round(wprob_t, 1)}%', transform=ax.transAxes, color=blue, fontsize='large', fontweight='bold', ha="center")
    
    ####all Lines below Were Added by Ciara Chisholm May 5th 2024
    # Used to turn off scientific notation on the x axis
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    
    

# PLOT 1 - POL. INT. MAP PART 1
def pss_subplot_get4_cve(data, x_arr, y_arr, x_long, y_lat, delta, data_units, x_pi_max, y_pi_max, fwxm, num, source_flag, npix=11):
    """
    This function calculates a significant amount of data, which is primarily used in plotting the Polarised Intensity Map, but in many other places too.

    ARGUMENTS:
    data (2D ndarray)    -- The polarised intensity data to be used in the calculations
    x_arr (2D ndarray)   -- The x pixel coordinates of the data
    y_arr (2D ndarray)   -- The y pixel coordinates of the data
    x_long (2D ndarray)  -- The x longitude coordinates of the data
    y_lat (2D ndarray)   -- The y latitude coordinates of the data
    delta (float)        -- The difference between each element in x_long and y_lat (should be 0.005 degrees in l/b coordinates)
    data_units (string)  -- The units of the data array
    x_pi_max (int)       -- The x pixel position of the source according to the Taylor17 source catalog
    y_pi_max (int)       -- The y pixel position of the source according to the Taylor17 source catalog
    fwxm (float)         -- The conversion factor from standard deviation or sigma to half width half maximum (unless changed in code)
    long (float)         -- The galactic longitude of the source according to the Taylor17 source catalog
    lat (float)          -- The galactic latitude of the source according to the Taylor17 source catalog
    num (int)            -- The source number (ranges from 0 to the number of sources in Taylor17_candidates.dat file generated by generate_candidate_sourcelist.py)
    mosaic_name (string) -- The name of the mosaic (e.g. MA1)
    npix (int)           -- Defines the size (in pixels) of the regions of the stamp to be examined (default 11)

    RETURNS:
    x_a_2 (2D ndarray)            -- A 2D array containing the pixel numbers in x of the region used for calculations around the source (currently set to 21 x 21)
    y_a_2 (2D ndarray)            -- A 2D array containing the pixel numbers in y of the region used for calculations around the source (currently set to 21 x 21)
    whw (2D ndarray)              -- "Where half width", a mask indicating the fwhm within the subregion around the source
    w3s (2D ndarray)              -- "Where 3 * s", a mask indicating the 'bottom' of the source
    x_loc (int)                   -- The x pixel location of the peak of the gaussian fit
    y_loc (int)                   -- The y pixel location of the peak of the gaussian fit
    x_reg (2D ndarray)            -- The x array (in pixel units) indicating the FWHM region
    y_reg (2D ndarray)            -- The y array (in pixel units) indicating the FWHM region
    x_tick_vals (list)            -- A list of floats containing the x axis labels for plotting
    y_tick_vals (list)            -- A list of floats containing the y axis labels for plotting
    w_annulus (2D ndarray)        -- A mask indicating the annulus region
    gauss_parameters (dictionary) -- The parameters of the gaussian fit: amp, xmean, ymean, xfwhm, yfwhm, theta
    source_ok (bool)              -- A boolean indicating whether the source is acceptable
    to_mask_coordinates (2d ndarray)- A array containing the pixel coordiantes to mask out after the RM of the source has been calculated. 
    """
    data_shape = data.shape
    try_max = 50
    num_pix = 20

    i_2 = x_pi_max
    if i_2 >= data_shape[0]:
        i_2 = data_shape[0] - 1
    j_2 = y_pi_max
    if j_2 >= data_shape[0]:
        j_2 = data_shape[0] - 1

    sub_id_x = np.array([i_2 - num_pix, i_2 + num_pix, j_2 - num_pix, j_2 + num_pix])

    if sub_id_x[0] < 0:
        sub_id_x[0] = 0
    if sub_id_x[1] >= data_shape[0]:
        sub_id_x[1] = data_shape[0] - 1
    if sub_id_x[2] < 0:
        sub_id_x[2] = 0
    if sub_id_x[3] >= data_shape[1]:
        sub_id_x[3] = data_shape[1] - 1

    # cut outs are 41x41 here
    data_2 = fh.cut_out_stamp(data, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    x_long_2 = fh.cut_out_stamp(x_long, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    x_a_2 = fh.cut_out_stamp(x_arr, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    y_lat_2 = fh.cut_out_stamp(y_lat, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    y_a_2 = fh.cut_out_stamp(y_arr, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])

 
    
   
    
    x_shape = x_long_2.shape
    x_arrs_2 = np.zeros(x_shape)
    y_arrs_2 = np.zeros(x_shape)
    for i in range(x_shape[0]):
        x_arrs_2[:, i] = i
    for i in range(x_shape[1]):
        y_arrs_2[i, :] = i

    max_x = x_shape[0] - 1
    max_y = x_shape[1] - 1

    if max_x > 25:
        max_x = 25
    if max_y > 25:
        max_y = 25

    x_longs = fh.cut_out_stamp(x_long_2, 15, max_x, 15, max_y)
    y_lats = fh.cut_out_stamp(y_lat_2, 15, max_x, 15, max_y)
    pi_sub = fh.cut_out_stamp(data_2, 15, max_x, 15, max_y)
    x_arrs = fh.cut_out_stamp(x_arrs_2, 15, max_x, 15, max_y)
    y_arrs = fh.cut_out_stamp(y_arrs_2, 15, max_x, 15, max_y)


    pixel_width_long = np.round(x_longs[0,0] - x_longs[0,1], 5)
    print("pixelwidth: ", pixel_width_long)
    pixels_per_degree = 1/pixel_width_long
    
    done = False
    try_num = 1

    # Initialize gaussian parameters
    amplitude = 0
    x_center_gauss = 0
    y_center_gauss = 0
    x_fwhm = 0
    y_fwhm = 0
    theta = 0

    # Initialize some arrays so the code doesn't get angry later on, these should never actually be accessed
    x_l_2 = fh.cut_out_stamp(x_long_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    y_b_2 = fh.cut_out_stamp(y_lat_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    data_fit = fh.cut_out_stamp(data_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    x_lb = np.zeros(data_fit.shape[0])
    y_lb = np.zeros(data_fit.shape[1])

    long_center_gauss = 0
    lat_center_gauss = 0

    gauss_pass = True

    try:
        while not done and try_num <= try_max:
            max_pi = np.max(pi_sub)
            eq_mask = pi_sub == max_pi
            idl_eq_mask = mf.idl_where(pi_sub == max_pi)

            # print("L554 idl_eq_mask: ", idl_eq_mask.tolist())
            # Checks to see if the 9x9 box is off the image
            data2_shape = data_2.shape

            sub_id_x[0] = x_arrs.flatten()[idl_eq_mask[0]] - npix
            sub_id_x[1] = x_arrs.flatten()[idl_eq_mask[0]] + npix
            sub_id_x[2] = y_arrs.flatten()[idl_eq_mask[0]] - npix
            sub_id_x[3] = y_arrs.flatten()[idl_eq_mask[0]] + npix

            if sub_id_x[0] < 0:
                sub_id_x[0] = 0
            if sub_id_x[1] >= data2_shape[0]:
                sub_id_x[1] = data2_shape[0] - 1
            if sub_id_x[2] < 0:
                sub_id_x[2] = 0
            if sub_id_x[3] >= data2_shape[1]:
                sub_id_x[3] = data2_shape[1] - 1

            data_fit = fh.cut_out_stamp(data_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
            x_l_2 = fh.cut_out_stamp(x_long_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
            y_b_2 = fh.cut_out_stamp(y_lat_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])

            x_pos_pix = mf.nround(x_arrs.flatten()[idl_eq_mask[0]])
            y_pos_pix = mf.nround(y_arrs.flatten()[idl_eq_mask[0]])
            x_pos = x_long_2[y_pos_pix, x_pos_pix]
            y_pos = y_lat_2[y_pos_pix, x_pos_pix]

            shape_y_2 = data_fit.shape
            x_lb = np.zeros(shape_y_2[0])
            y_lb = np.zeros(shape_y_2[1])
            x_lb[:] = x_l_2[0, :]
            y_lb[:] = y_b_2[:, 0]

            # Fit a 2D gaussian to the source, then ensure the centre is inside the correct region (defined by x_lb and y_lb)

            # est must have the form [amplitude, x_0, y_0, sigma_x, sigma_y, theta, offset], this array is the initial parameters guess. 
            est = np.array([max_pi, 0.0, 0.0, 0.01, 0.01, 0.0, 0.0])
            
            # Fitting the data to a 2D gaussian, note the x and y coordinates are shifted so the peak is at the origin (I think, I need to check later.)
            amplitude, x_center_gauss, y_center_gauss, x_stddev, y_stddev, theta, yfit = ac.gauss_fit_2d(data_fit, x_lb - x_pos, y_lb - y_pos, est)

            x_fwhm = x_stddev
            y_fwhm = y_stddev
            
            # The following two lines were added by Ciara Chisholm Oct 17 2024
            # Copying the standard deviation in the X and Y direction. 
            x_sigma = x_stddev.copy()
            y_sigma = y_stddev.copy()
            
            
            spi = yfit.shape

            long_center_gauss = x_center_gauss + x_pos
            lat_center_gauss = y_center_gauss + y_pos

            if spi[0] != 0 and (long_center_gauss < np.min(x_lb) or long_center_gauss > np.max(x_lb) or lat_center_gauss < np.min(y_lb) or lat_center_gauss > np.max(y_lb)):
                failed = True
            else:
                failed = False

            # Finding the difference between the fitted gauss centre and the longitude and latitude of every pixel in the source,
            # then locating where this diff is at a minimum.
            dif_long = np.abs(x_longs - long_center_gauss)
            dif_lat = np.abs(y_lats - lat_center_gauss)
            min_dif_long = np.min(dif_long)
            min_dif_lat = np.min(dif_lat)
            w_long = mf.idl_where(dif_long == min_dif_long)
            w_lat = mf.idl_where(dif_lat == min_dif_lat)
            xpp = x_arrs.flatten()[w_long[0]]
            ypp = y_arrs.flatten()[w_lat[0]]

            # Check the distance between the fitted gauss centre and the real centre
            # If it's too big, the source fails
            if len(spi) != 0 and not failed and (np.abs(xpp - x_pos_pix) > 3 or np.abs(ypp - y_pos_pix) > 3):
                failed = True
            else:
                failed = False

            if len(spi) != 0 and not failed:
                done = True
            else:
                try_num += 1
                pi_sub[eq_mask] = 0.0
                npix = npix
                print(f'Try #{try_num}')

        if try_num >= try_max:
            gauss_pass = False

    except RuntimeError:
        gauss_pass = False

    if gauss_pass:
        print(f'\nGaussian fit succeeded with npix: {npix}, try: {try_num}')
    else:
        print(f'Failed to fit gaussian')
        # If the 4 flag (Gaussian fit failed) is not set already, set it
        source_flag = (source_flag | 4)
        # The following line was added by Ciara Chisholm on Nov 14th
        x_sigma, y_sigma = 0,0

    
    # Pixel coordinates of the stamp cutout. 
    x_a_2 = fh.cut_out_stamp(x_a_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    y_a_2 = fh.cut_out_stamp(y_a_2, sub_id_x[0], sub_id_x[1], sub_id_x[2], sub_id_x[3])
    
    
    
    # The following two arrays where added to the code by Ciara Chisholm to find the x and y pixel coordinates to 
    #   for the mask. 
    x_cutout_2d_array_flattened = x_a_2.flatten()
    y_cutout_2d_array_flattened = y_a_2.flatten()
    
    
    
    print(f'\nGaussian Parameters:')
    print(f'Amplitude: {amplitude}')
    print(f'X Mean: {long_center_gauss}')
    print(f'Y Mean: {lat_center_gauss}')
    print(f'X Standard Deviation: {x_fwhm}')
    print(f'Y Standard Deviation: {y_fwhm}')
    print(f'Theta: {theta}\n')


    print(f"x FWHM in pixels: {x_sigma*fwxm*2*pixels_per_degree}")# converting to FWHM in pixels
    print(f"y FWHM in pixels: {y_sigma*fwxm*2*pixels_per_degree}")
    # hw_aa = fwxm * x_fwhm
    # hw_bb = fwxm * y_fwhm
    
    # The following two lines were changed from the two previously commented out lines  
    #   to the next two lines by Ciara Chisholm on Oct 17 2024
    # Converting the standard deviation in the x direction of the fitted gaussian to
    #   the Half Width Half Maximum in the x direction
    hw_aa = fwxm * x_sigma 
    hw_bb = fwxm * y_sigma # Converting the Standard Deviation to HWHM in the y direction
    
    
    
    # The following comments and 3 lines were added by Ciara Chisholm Oct 17 2024
    # Setting how much to scale half width maximum by for the masking the source after it's RM is calculated
    mask_HWHM_scale = 1.75
    hw_aa_mask = hw_aa*mask_HWHM_scale
    hw_bb_mask = hw_bb*mask_HWHM_scale
    # Following two lines of code were added in by Ciara Chisholm on Nov 5th
    hw_aa_mask_pixels = hw_aa_mask*pixels_per_degree
    hw_bb_mask_pixels = hw_bb_mask*pixels_per_degree
   
    

    annv = 100  # Indicates the edge of the source, for annulus calculations, 1/100th max
    annw = np.sqrt(2.0 * (np.log(annv))) # conversion from sigma to half width 100th max

    # three_a = annw * x_fwhm  # Edge of source radii
    # three_b = annw * y_fwhm

    # The following two lines were changed from the two previously commented out lines  
    #   to the next two lines by Ciara Chisholm on Oct 17 2024
    # Converting the standard deviation in the x direction of the fitted gaussian to
    #   the Half Width 100th Maximum in the x direction
    three_a = annw *x_sigma
    three_b = annw *y_sigma
     
    # Ciara Chisholm October 17th 2024: 
    #   This rotates the gaussian 
    x_l_2_p = (x_l_2.copy() - long_center_gauss) * np.cos(theta) - (y_b_2.copy() - lat_center_gauss) * np.sin(theta)
    y_b_2_p = (x_l_2.copy() - long_center_gauss) * np.sin(theta) + (y_b_2.copy() - lat_center_gauss) * np.cos(theta) 
    
    
    whw = mf.idl_where(((x_l_2_p / hw_aa) ** 2 + (y_b_2_p / hw_bb) ** 2) <= 1.0)
    w3s = mf.idl_where(((x_l_2_p / three_a) ** 2 + (y_b_2_p / three_b) ** 2) <= 1.0)
  
    # Establishing the location of the point source:
    w_loc = mf.idl_where(np.logical_and(np.abs(long_center_gauss - x_long) < (delta / 2.0), np.abs(lat_center_gauss - y_lat) < (delta / 2.0)))

    w_loc_shape = w_loc.shape
    if w_loc_shape[0] == 0:  # If there are no elements of x_long/y_lat that match this condition
        w_loc = mf.idl_where(np.logical_and(np.abs(long_center_gauss - x_long) < (delta / 1.99), np.abs(lat_center_gauss - y_lat) < (delta / 1.99)))
        print(f'\nWARNING: locate center retry')

    w_loc_shape = w_loc.shape
    if w_loc_shape[0] != 0:
        x_loc = x_arr.flatten()[w_loc[0]]
        y_loc = y_arr.flatten()[w_loc[0]]
    else:
        x_loc = 0
        y_loc = 0


    

    
    
    # =============================================================================
    # Creating the mask of the source    
    # =============================================================================
    # The following code was added by Ciara Chisholm on Nov 5th 2024 to try to fix the mask. 
    # Note: the rotational transformation is the inverse of the previous because the 
    #   galactic coordinates increase from right to left, and the pixel coordinates
    #   increase from left to right. 
    x_cutout_pixel_coor_rotated = (x_a_2.copy() - x_loc) * np.cos(theta) + (y_a_2.copy() - y_loc) * np.sin(theta)
    y_cutout_pixel_coor_rotated = -(x_a_2.copy() - x_loc) * np.sin(theta) + (y_a_2.copy() - y_loc) * np.cos(theta)

    
    # Finding which pixels are in the mask region. This is done by using the equation 
    #   of an ellipse. Anything in the ellipse is 
    in_mask = ((x_cutout_pixel_coor_rotated/hw_aa_mask_pixels)**2 + (y_cutout_pixel_coor_rotated/hw_bb_mask_pixels)**2) <=1.0
    
    # 
    
    # Initializing the array to store which pixels to mask. 
    where_to_mask_pixels = np.ones((len(x_a_2[in_mask]), 2))
    
    # Storing the x and y values of which pixels to mask 
    where_to_mask_pixels[:,0] = x_a_2[in_mask].copy()
    where_to_mask_pixels[:,1] = y_a_2[in_mask].copy()
    # Converting all the data to integers. 
    where_to_mask_pixels = np.array(where_to_mask_pixels, dtype=int)
    
    


    levels = 0.5 + np.arange(40) * 0.5
    if data_units != 'mJy/beam':
        levels = 0.15 + np.arange(20) * 0.05


    # Ciara Chisholm changed the lines with "x_fwhm" (currently commented out) to "x_sigma" on Oct 17 2024
    # Get fwxm region
    # x_gauss = (np.arange(1000) - 50) * x_fwhm * fwxm / 50.0
    # Ciara Chisholm changedthe range of the list to 101 because having it as 1000 
    #   doesn't make sense and later on in similar calculations it is 101. Code still runs fine
    #   Dr. Jo-Anne Brown also approved this change
    x_gauss = (np.arange(101) - 50) * x_sigma * fwxm / 50.0
    # To avoid the warnings that pop up when a calculation results in a NaN, we calculate only the good pixels, and place the NaNs manually.
    # radicand = 1.0 - (x_gauss / (x_fwhm * fwxm)) ** 2
    radicand = 1.0 - (x_gauss / (x_sigma * fwxm)) ** 2
    where_positive = radicand >= 0
    y_gauss = np.full(x_gauss.shape, np.nan)
    # y_gauss[where_positive] = np.sqrt(radicand[where_positive]) * y_fwhm * fwxm
    y_gauss[where_positive] = np.sqrt(radicand[where_positive]) * y_sigma * fwxm
    x_gauss = np.array([x_gauss, x_gauss])
    y_gauss = np.array([y_gauss, -1 * y_gauss])

    # Rotating:
    x_gauss_rot1 = x_gauss * np.cos(theta) + y_gauss * np.sin(theta)
    y_gauss_rot1 = -1.0 * x_gauss * np.sin(theta) + y_gauss * np.cos(theta)

    # # Still on the same plot

    x_fwxm = x_gauss_rot1 + long_center_gauss
    y_fwxm = y_gauss_rot1 + lat_center_gauss
    # x_tick_vals = np.array([np.max(x_l_2), x_center_gauss, np.min(x_l_2)])
    # y_tick_vals = np.array([np.min(y_b_2), y_center_gauss, np.max(y_b_2)])
    # The following line was changed from the previous two lines on Jan 22 2025 by Ciara Chisholm
    x_tick_vals = np.array([np.max(x_l_2), long_center_gauss, np.min(x_l_2)])
    y_tick_vals = np.array([np.min(y_b_2), lat_center_gauss, np.max(y_b_2)])

    whw_shape = whw.shape
    if len(whw_shape) != 0:
        x_reg = x_a_2.flatten()[whw]
        y_reg = y_a_2.flatten()[whw]
    else:
        x_reg = 0
        y_reg = 0

    if try_num <= try_max:
        source_ok = True
    else:
        source_ok = False

    diff_x = np.abs(x_arr[j_2, i_2] - x_loc)
    diff_y = np.abs(y_arr[j_2, i_2] - y_loc)

    if diff_x > 11 or diff_y > 11:
        source_ok = False
        print(f'diff_x: {diff_x}')
        print(f'diff_y: {diff_y}')

    # **************************************************************************
    # CALCULATING THE ANNULUS REGION:
    # **************************************************************************

    inc = 1.0 / 60  # One arc minute

    if x_fwhm < 0:
        sign_x_fwhm = -1.0
    else:
        sign_x_fwhm = 1.0

    if y_fwhm < 0:
        sign_y_fwhm = -1.0
    else:
        sign_y_fwhm = 1.0
        
        
    
    outer_edge_x = (annw * x_fwhm) + (inc * sign_x_fwhm) # annw is the conversion factor from sigma to half width 100th max
    outer_edge_y = (annw * y_fwhm) + (inc * sign_y_fwhm)

    
    w_annulus = mf.idl_where(
        np.logical_and((((x_l_2_p / outer_edge_x) ** 2 + (y_b_2_p / outer_edge_y) ** 2) <= 1.0), (((x_l_2_p / three_a) ** 2 + (y_b_2_p / three_b) ** 2) >= 1.0)))
  

    # **************************************************************************
    # CALCULATING ANNULUS EDGE:
    # **************************************************************************
    
    # x_gauss = (np.arange(100) - 50) * outer_edge_x / 50.0
    # The previous line was changed to the one below by Ciara Chisholm Oct 22 to make the array symmetric.
    x_gauss = (np.arange(101) - 50) * outer_edge_x / 50.0
    y_gauss = np.sqrt(1.0 - (x_gauss / outer_edge_x) ** 2) * outer_edge_y
    x_gauss = np.array([x_gauss, x_gauss])
    y_gauss = np.array([y_gauss, -1 * y_gauss])
    x_gauss_rot2 = x_gauss * np.cos(theta) + y_gauss * np.sin(theta)
    y_gauss_rot2 = -1 * x_gauss * np.sin(theta) + y_gauss * np.cos(theta)
    x_fwxm_ae = x_gauss_rot2 + long_center_gauss
    y_fwxm_ae = y_gauss_rot2 + lat_center_gauss

    # **************************************************************************
    # CALCULATING SOURCE EDGE:
    # **************************************************************************

    # x_gauss = (np.arange(100) - 50) * x_fwhm * annw / 50.0
    # The previous line was changed to the one below by Ciara Chisholm Oct 22 to make the array symmetric.
    x_gauss = (np.arange(101) - 50) * x_fwhm * annw / 50.0
    y_gauss = np.sqrt(1.0 - (x_gauss / (x_fwhm * annw)) ** 2) * y_fwhm * annw
    x_gauss = np.array([x_gauss, x_gauss])
    y_gauss = np.array([y_gauss, -1 * y_gauss])
    x_gauss_rot3 = x_gauss * np.cos(theta) + y_gauss * np.sin(theta)
    y_gauss_rot3 = -1 * x_gauss * np.sin(theta) + y_gauss * np.cos(theta)
    x_fwxm_se = x_gauss_rot3 + long_center_gauss
    y_fwxm_se = y_gauss_rot3 + lat_center_gauss

    gauss_parameters = {'amp': amplitude,
                        'x_fwhm': x_fwhm,
                        'y_fwhm': y_fwhm,
                        'x_mean': long_center_gauss,
                        'y_mean': lat_center_gauss,
                        'theta': theta}

    pi_plot_data = {'x_lb': x_lb,
                    'y_lb': y_lb,
                    'x_l_2': x_l_2,
                    'y_b_2': y_b_2,
                    'data_fit': data_fit,
                    'levels': levels,
                    'num': num,
                    'x_gauss_rot': x_gauss_rot1,
                    'y_gauss_rot': y_gauss_rot1,
                    'x_center_gauss': long_center_gauss,
                    'y_center_gauss': lat_center_gauss,
                    'x_fwxm_ae': x_fwxm_ae,
                    'y_fwxm_ae': y_fwxm_ae,
                    'x_fwxm_se': x_fwxm_se,
                    'y_fwxm_se': y_fwxm_se}



    
    
    
    
    return x_a_2, y_a_2, whw, w3s, x_loc, y_loc, x_reg, y_reg, x_tick_vals, y_tick_vals,\
        x_fwxm, y_fwxm, w_annulus, gauss_parameters, source_ok, pi_plot_data, source_flag, where_to_mask_pixels #np.array(to_mask_coordinates)


# PLOT 2 - STOKES I MAP
def pss_stokes_i_plot(stokes_i, x_long, y_lat, x_pix_max_i, y_pix_max_i, gauss_parameters, fwxm, x_pix_max, y_pix_max):
    """This function calculates values necessary for the plotting of the Stokes I Map.
    """
    npix = 11

    # data = fh.cut_out_stamp(stokes_i, x_pix_max_i - npix + 1, x_pix_max_i + npix + 1, y_pix_max_i - npix + 1, y_pix_max_i + npix + 1)
    # long_arr = fh.cut_out_stamp(x_long, x_pix_max_i - npix + 1, x_pix_max_i + npix + 1, y_pix_max_i - npix + 1, y_pix_max_i + npix + 1)
    # lat_arr = fh.cut_out_stamp(y_lat, x_pix_max_i - npix + 1, x_pix_max_i + npix + 1, y_pix_max_i - npix + 1, y_pix_max_i + npix + 1)
    
    ### The following three lines were modified from the previous three lines by Ciara Chisholm on February 6th 2026
    data = fh.cut_out_stamp(stokes_i, x_pix_max_i - npix , x_pix_max_i + npix , y_pix_max_i - npix , y_pix_max_i + npix )
    long_arr = fh.cut_out_stamp(x_long, x_pix_max_i - npix , x_pix_max_i + npix,  y_pix_max_i - npix , y_pix_max_i + npix)
    lat_arr = fh.cut_out_stamp(y_lat, x_pix_max_i - npix , x_pix_max_i + npix , y_pix_max_i - npix , y_pix_max_i + npix )


    levels = 2 ** np.arange(20)

    x_l_b = long_arr[0, :]
    y_l_b = lat_arr[:, 0]

    # Get fwxm region
    x_gauss = (np.arange(1000) - 50) * gauss_parameters['x_fwhm'] * fwxm / 50.0

    # To avoid the warnings that pop up when a calculation results in a NaN, we calculate only the good pixels, and place the NaNs manually.
    radicand = 1.0 - (x_gauss / (gauss_parameters['x_fwhm'] * fwxm)) ** 2
    where_positive = radicand >= 0
    y_gauss = np.full(x_gauss.shape, np.nan)
    y_gauss[where_positive] = np.sqrt(radicand[where_positive]) * gauss_parameters['y_fwhm'] * fwxm

    x_gauss = np.array([x_gauss, x_gauss])
    y_gauss = np.array([y_gauss, -1 * y_gauss])

    # Rotating
    x_gauss_rot1 = x_gauss * np.cos(gauss_parameters['theta']) + y_gauss * np.sin(gauss_parameters['theta'])
    y_gauss_rot1 = -1 * x_gauss * np.sin(gauss_parameters['theta']) + y_gauss * np.cos(gauss_parameters['theta'])

    # Plot the shape of the beam:
    g_coords = SkyCoord(l=gauss_parameters['x_mean'], b=gauss_parameters['y_mean'], frame='galactic', unit='degree')
    ra = g_coords.fk5.ra.deg
    dec = g_coords.fk5.dec.deg

    major = 49 / 3600 / np.sin(dec) / 2.3548  # 49"/sin(dec) FWHM in Gaussian sigma
    minor = 49 / 3600 / 2.3548  # 49" FWHM, in Gaussian sigma

    pa, gl_src, gb_src = mf.get_position_angle(ra, dec)

    # Define Beam FWHM ellipse:
    x_gauss = (np.arange(200) - 100) * minor * fwxm / 100
    y_gauss = np.sqrt(1.0 - (x_gauss / (minor * fwxm)) ** 2) * major * fwxm
    x_gauss = np.array([x_gauss, x_gauss])
    y_gauss = np.array([y_gauss, -1 * y_gauss])
    x_gauss_rot2 = x_gauss * np.cos(pa) + y_gauss * np.sin(pa)
    y_gauss_rot2 = -1 * x_gauss * np.sin(pa) + y_gauss * np.cos(pa)

    si_plot_data = {'long_arr': long_arr,
                    'lat_arr': lat_arr,
                    'data': data,
                    'levels': levels,
                    'x_l_b': x_l_b,
                    'y_l_b': y_l_b,
                    'x_gauss_rot1': x_gauss_rot1,
                    'y_gauss_rot1': y_gauss_rot1,
                    'x_gauss_rot2': x_gauss_rot2,
                    'y_gauss_rot2': y_gauss_rot2,
                    'x_pix_max': x_pix_max,
                    'y_pix_max': y_pix_max,
                    'npix': npix}

    return si_plot_data


def get_subarrays_cve(stokes_i, qa, qb, qc, qd, ua, ub, uc, ud,
                      x_long, y_lat, x_loc, y_loc, x_a_2, y_a_2,
                      delta, pi, ston, whw, w3s, noise, w_annulus):
    """This function cuts out a series of postage stamps from the input arrays,
    as well as generates some X and Y pixel arrays
    """
    min_x = mf.nround(np.min(x_a_2))
    max_x = mf.nround(np.max(x_a_2))
    min_y = mf.nround(np.min(y_a_2))
    max_y = mf.nround(np.max(y_a_2))

    cut_pi = fh.cut_out_stamp(pi, min_x, max_x, min_y, max_y)
    cut_i = fh.cut_out_stamp(stokes_i, min_x, max_x, min_y, max_y)

    cut_qa = fh.cut_out_stamp(qa, min_x, max_x, min_y, max_y)
    cut_qb = fh.cut_out_stamp(qb, min_x, max_x, min_y, max_y)
    cut_qc = fh.cut_out_stamp(qc, min_x, max_x, min_y, max_y)
    cut_qd = fh.cut_out_stamp(qd, min_x, max_x, min_y, max_y)
    cut_ua = fh.cut_out_stamp(ua, min_x, max_x, min_y, max_y)
    cut_ub = fh.cut_out_stamp(ub, min_x, max_x, min_y, max_y)
    cut_uc = fh.cut_out_stamp(uc, min_x, max_x, min_y, max_y)
    cut_ud = fh.cut_out_stamp(ud, min_x, max_x, min_y, max_y)


    
        
    cut_noise = fh.cut_out_stamp(noise, min_x, max_x, min_y, max_y)

    cut_x_long = fh.cut_out_stamp(x_long, min_x, max_x, min_y, max_y)
    cut_y_lat = fh.cut_out_stamp(y_lat, min_x, max_x, min_y, max_y)

    cut_ston = fh.cut_out_stamp(ston, min_x, max_x, min_y, max_y)

    x_loc = mf.nround(x_loc)
    y_loc = mf.nround(y_loc)

    x_pos_long = x_long[y_loc, x_loc]
    y_pos_lat = y_lat[y_loc, x_loc]

    x_shape = cut_x_long.shape

    cut_x_arr = np.zeros(x_shape)
    cut_y_arr = np.zeros(x_shape)

    for i in range(x_shape[0]):
        cut_x_arr[:, i] = i
    for i in range(x_shape[1]):
        cut_y_arr[i, :] = i

    less_mask = mf.idl_where(np.logical_and(np.abs(x_pos_long - cut_x_long) < delta / 2, np.abs(y_pos_lat - cut_y_lat) < delta / 2))

    cut_x_loc = cut_x_arr.flatten()[less_mask[0]]
    cut_y_loc = cut_y_arr.flatten()[less_mask[0]]

    cut_x_reg = cut_x_arr.flatten()[whw]
    cut_y_reg = cut_y_arr.flatten()[whw]

    x3s = cut_x_arr.flatten()[w3s]
    y3s = cut_y_arr.flatten()[w3s]

    cut_x_ann = cut_x_arr.flatten()[w_annulus]
    cut_y_ann = cut_y_arr.flatten()[w_annulus]
    
    

    return (cut_pi, cut_i, cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud, cut_x_arr, cut_y_arr,
            cut_noise, cut_x_long, cut_y_lat, cut_ston, cut_x_loc, cut_y_loc, cut_x_reg, cut_y_reg, x3s, y3s, cut_x_ann, cut_y_ann)


def sbg_av3(cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud, cut_x_ann, cut_y_ann):
    """This function removes the background noise from the input array stamps.
    """
    

    # .copy() was added by Ciara Chisholm Nov 26th 2024
    cut_qa = mf.calc_background3(cut_qa.copy(), cut_x_ann, cut_y_ann)
    cut_qb = mf.calc_background3(cut_qb.copy(), cut_x_ann, cut_y_ann)
    cut_qc = mf.calc_background3(cut_qc.copy(), cut_x_ann, cut_y_ann)
    cut_qd = mf.calc_background3(cut_qd.copy(), cut_x_ann, cut_y_ann)
    cut_ua = mf.calc_background3(cut_ua.copy(), cut_x_ann, cut_y_ann)
    cut_ub = mf.calc_background3(cut_ub.copy(), cut_x_ann, cut_y_ann)
    cut_uc = mf.calc_background3(cut_uc.copy(), cut_x_ann, cut_y_ann)
    cut_ud = mf.calc_background3(cut_ud.copy(), cut_x_ann, cut_y_ann)
    
    
    
    return cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud


def unwrap_pa_and_rm_calcs4(cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud,
                            lambda2, crd, cut_dpsi_a, cut_dpsi_b, cut_dpsi_c, cut_dpsi_d,
                            cut_x_arr, cut_y_arr, lx):
    """This function uses the polarisation angles to calculate the rotation measure of each pixel, along with its error.
    
    The following part of the doc string was edited by Ciara Chisholm, May 2nd 2024 and May 31st 2024. 
    Arguments:
        - cut_qa (2d array): cut out (or postage stamp) around the source of channel A in Stokes Q
        - cut_qb (2d array): cut out (or postage stamp) around the source of channel B in Stokes Q
        - cut_qc (2d array): cut out (or postage stamp) around the source of channel C in Stokes Q
        - cut_qd (2d array): cut out (or postage stamp) around the source of channel D in Stokes Q
        - cut_ua (2d array): cut out (or postage stamp) around the source of channel A in Stokes U
        - cut_ub (2d array): cut out (or postage stamp) around the source of channel B in Stokes U
        - cut_uc (2d array): cut out (or postage stamp) around the source of channel C in Stokes U
        - cut_ud (2d array): cut out (or postage stamp) around the source of channel D in Stokes U
        - lambda2 (float): The wavelength squared (21 cm)
        - crd (float): The conversion factor to go for radians to degrees (I don't know why this is an important parameter')
        - cut_dpsi_a(2d array): cut out containing the errors for the polarisation angles in band a
        - cut_dpsi_b(2d array): cut out containing the errors for the polarisation angles in band b
        - cut_dpsi_c(2d array): cut out containing the errors for the polarisation angles in band c
        - cut_dpsi_a(2d array): cut out containing the errors for the polarisation angles in band d
        - cut_x_arr(2d array): a 2d array containing x coordinates in every row 
        - cut_y_arr(2d array): a 2d array containing y coordinates in every column
        - lx (list): a list containing the wavelength squared of each band starting with band a
    
    Returns:
        - rm_array (2d array): The rotation measures for each pixel in the cutout
        - rm_error_array (2d array): The error in the rotation measures calculated
        - psi (2d array): the polarisation angle for every pixel in the cut out at 1420MHz based on the calculated RM/line of best fit.
        - prob_t (float): the probability the fit is correct. 
    """
    ua_shape = cut_ua.shape
    

    temp = np.zeros((2, ua_shape[0], ua_shape[1]))
    sigma_t = np.zeros((2, ua_shape[0], ua_shape[1]))
    prob_t = np.zeros(ua_shape)
    result = np.zeros((4, ua_shape[0], ua_shape[1]))

    # Unwrapping data
    u = np.array([cut_ua, cut_ub, cut_uc, cut_ud])
    q = np.array([cut_qa, cut_qb, cut_qc, cut_qd])
    #
    # =============================================================================
    # Here to the next row of equal signs is Debugging stuff Ciara Chisholm OCt 13 2024
    # =============================================================================
    u_ave = np.sum(u, axis=0)/4
    q_ave = np.sum(q, axis=0)/4
    # u_ave = (cut_ua+ cut_ub+ cut_uc+ cut_ud)/4
    # q_ave = (cut_qa+ cut_qb+ cut_qc+ cut_qd)/4
    
    

    # The following code was added by Ciara Chisholm July 30
    # PI = np.sum(u**2 +q**2)/4 # Note I know this isn't the correct PI cause it 
    PI = np.sqrt(u_ave*u_ave +q_ave*q_ave)/4 # Note I know this isn't the correct PI cause it 
    
    


    z = q + (1j * u)
    z_norm = z / np.abs(z)

    # Checking for zeros in q and u which would make z_norm fail
    
    # The following comment was added by Ciara Chisholm June 5th 2024
    # This does not quite match up with what the IDL code does, but I don't think 
    #   the difference will make any difference in the code.

    zero_mask = np.logical_and(q == 0, u == 0)
    if np.sum(zero_mask) > 0:  # If there is at least 1 zero in q or u
        print('****************************************** ZEROS IN Q AND U')
        z_norm[zero_mask] = 0.0 + (1j * 0.0)

    d_phase = z_norm[1:, :, :] * np.conj(z_norm[0:3, :, :]) # Finds the phase difference, see eqn 4.22 in page 60 Jo-Anne's thesis    
    # d_phase = np.arctan(np.imag(d_phase), np.real(d_phase)) # Find the phase difference in radians 
    # The following line was modified by Ciara Chisholm on 22/05/24 from the commented line above
    d_phase = np.arctan2(np.imag(d_phase), np.real(d_phase)) # Find the phase difference in radians 

    
    # result[0, :, :] = np.arctan(np.imag(z_norm[0, :, :]), np.real(z_norm[0, :, :])) # returns a 23x23 array
    
    
    # The following 2 lines were modified by Ciara Chisholm on 22/05/24 from the commented line above
    result[0, :, :] = np.arctan2(np.imag(z_norm[0, :, :]), np.real(z_norm[0, :, :]))# returns a 23x23 array

    
    
    
    res_0 = result[0, :, :]

    # Finding of the NaN values in the phase difference 
    w_nan = np.isnan(res_0) 
    
    if np.sum(w_nan) > 0:
        # If there is at least one NaN, we will temporarily set it equal to zero
        res_0[w_nan] = 0.0

    neg_mask = res_0 < 0.0
    if np.sum(neg_mask) > 0:
        res_0[neg_mask] += (2 * np.pi)
    if np.sum(w_nan) > 0:
        # If there used to be any NaNs, we set them equal to zero a few lines ago, and now we set them to NaNs once again.
        res_0[w_nan] = np.nan
    result[0, :, :] = res_0

    for index in [0, 1, 2]:
        result[index + 1, :, :] = result[index, :, :] + d_phase[index, :, :]

    result = result / 2.0

    psi_err = np.array([cut_dpsi_a, cut_dpsi_b, cut_dpsi_c, cut_dpsi_d])
    
    # Setting a minimum psi value of 1 degree - Ciara Chisholm Dec 3
    psi_below_min = psi_err<(np.pi/180)
    psi_err[psi_below_min]=(np.pi/180)

    # Done unwrapping, beginning rotation measure calculations
    finite_mask = mf.idl_where(np.isfinite(cut_ua))
    finite_mask_shape = finite_mask.shape
    i = cut_x_arr.flatten()[finite_mask].astype(int)
    j = cut_y_arr.flatten()[finite_mask].astype(int)
    

    nan_count = 0

    for k in np.arange(finite_mask_shape[0]):
        # Checking for nan's due to median removal in psi_err
        psi_where_nans = np.invert(np.isfinite(psi_err[:, j[k], i[k]]))

        psi_mask_no_nan = ~np.isnan(psi_err[:, j[k], i[k]])

        if np.sum(psi_where_nans) > 0:
            nan_count += 1
            print(f'Problem with nans at k = {k}')

            # Set nans equal to zero
            psi_err[psi_where_nans, j[k], i[k]] = 0
            psi_err[psi_where_nans, j[k], i[k]] = np.mean(psi_err[psi_mask_no_nan, j[k], i[k]])

        # Performing a linear fit on the polarisation angle
        def linear_function(x, m, b):
            return m * x + b

        lx = np.array(lx)
        fit_params, fit_cov = opt.curve_fit(linear_function, lx, result[:, j[k], i[k]], sigma=psi_err[:, j[k], i[k]])

        fit_slope, fit_yint = fit_params
        
        fit_data = linear_function(lx, fit_slope, fit_yint)

        weights = 1 / psi_err[:, j[k], i[k]] ** 2
        yint_sigma = np.sqrt(np.sum(weights * lx**2) / (np.sum(weights) * np.sum(weights * lx**2) - np.sum(weights * lx)**2))
        slope_sigma = np.sqrt(np.sum(weights) / (np.sum(weights) * np.sum(weights * lx**2) - np.sum(weights * lx)**2))

        # The chi-squared value is the sum of the squared difference between the observed data and the fit data divided by sigma squared
        chi_2 = np.sum((result[:, j[k], i[k]] - fit_data)**2 / psi_err[:, j[k], i[k]]**2)
        
        # # REDUCED CHI SQUARED VALUE
        # chi_2 = np.sum((result[:, j[k], i[k]] - fit_data)**2 /result[:, j[k], i[k]])
        
        # print("")
        
        
        # This shadows IDL's LINFIT() PROB parameter. From the LINFIT documentation:
        # If PROB is greater than 0.1, the model parameters are “believable”. If PROB is less than 0.1, the accuracy of the model parameters is questionable.
        # prob_good_fit = 1 - sp.gammainc(0.5 * (len(lx) - 2), 0.5 * chi_2)
        # print("dylan's prob fit:", prob_good_fit)
        
        # Ciara Chisholm commented out on October 24th 2024 the previous line and 
        #   replaced it with the following 4 lines
        #   I am not sure if the following function follows the PROB parameter from 
        #   IDL's LINFIT (info on it was hard to find), but it is the probability 
        #   that found in Dr. Brown's thesis (equation 4.30, page 64). The function
        #   gamma used a gamma distribution survival function.
        # Ciara Chisholm added the following new comments on October 28th 2024
        #   Dr. Brown get's the function in her Thesis from Dr. John Taylor's 
        #   Introduction to error analysis, appendix D. 
        d = len(lx)-2# The number of degrees of freedom is equal to the number of 
                     # data points minus the number of fitting parameters, in  
                     # this case 2. See page 269 of Taylor's error analysis book
        a = 1/2
        b = d/2
        prob_good_fit = sp.gdtrc(a,b,chi_2)
        # Fun fact in our case, prob_good_fit = np.exp(-0.5*chi_2)
        
        
        
        # !!!!!!!! Note to self:
        # Check if the function is giving the same values for different sources by printing the chi_2 value and the results prob. 
        
        
        

        temp[:, j[k], i[k]] = np.array([fit_yint, fit_slope])

        sigma_t[:, j[k], i[k]] = np.array([yint_sigma, slope_sigma])
        prob_t[j[k], i[k]] = prob_good_fit

    # Completed RM calculations
    rm_array = temp[1, :, :]
    rm_error_array = sigma_t[1, :, :]
    intercept_array = temp[0, :, :].astype(float) * crd#CRD:Convert Radians to Degrees
    
    
  

    # Polarisation angle and initial angles
    psi = (lambda2 * rm_array * crd + intercept_array).astype(float)

    # # Wrapping up the angle maps
    # psi = wrap_data(psi, 0.0, 180.0)

    return rm_array, rm_error_array, psi, prob_t


def wrap_data(data, min_data, max_data):
    """This function ensures the input data falls within the region [min_data, max_data].
    """
    diff_data = max_data - min_data

    less_mask = data < min_data
    while np.sum(less_mask) > 0:
        data[less_mask] += diff_data
        less_mask = data < min_data

    greater_mask = data > max_data
    while np.sum(greater_mask) > 0:
        data[greater_mask] -= diff_data
        greater_mask = data > max_data

    return data


def fractional_polarisation_test(cut_stokes_i, cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud, cut_noise):
    """This function calculates the fractional polarisation of each of the 4 bands, and its error,
    and returns the chi-square residuals of the fractional polarisations"""
    # Fractional polarisations:
    m_a = np.sqrt(cut_ua**2 + cut_qa**2) / cut_stokes_i
    m_b = np.sqrt(cut_ub**2 + cut_qb**2) / cut_stokes_i
    m_c = np.sqrt(cut_uc**2 + cut_qc**2) / cut_stokes_i
    m_d = np.sqrt(cut_ud**2 + cut_qd**2) / cut_stokes_i

    # Uncertainties (assuming sigma_qu = sigma_i and same noise for all channels)
    dm_a = cut_noise * np.sqrt(1 + m_a**2) / cut_stokes_i
    dm_b = cut_noise * np.sqrt(1 + m_b**2) / cut_stokes_i
    dm_c = cut_noise * np.sqrt(1 + m_c**2) / cut_stokes_i
    dm_d = cut_noise * np.sqrt(1 + m_d**2) / cut_stokes_i

    mean_m = (m_a + m_b + m_c + m_d) / 4
    chi_square_residuals = ((m_a - mean_m)**2 / dm_a**2) + ((m_b - mean_m)**2 / dm_b**2) + ((m_c - mean_m)**2 / dm_c**2) + ((m_d - mean_m)**2 / dm_d**2)

    
    return chi_square_residuals


def avg_rm_calcs8(cut_pi, rotmeas, e_rotmeas, cut_ston, edge_threshold, cut_x_reg, cut_y_reg, cut_i, prob_t, fracpol_chi2_arr):
    """This function uses the rotation measure data to calculate various parameters
    about the source to assist in our analysis of the data.
    """
    source_ok = True

    cut_x_reg = cut_x_reg.astype(int)
    cut_y_reg = cut_y_reg.astype(int)


    # ------------------------------------------------------------
    # Beginning Averaging Calculations...
    # ------------------------------------------------------------
    
    # Calculate weight maps in rotation measure:
    weights = 1.0 / (e_rotmeas * e_rotmeas)

    noise_box = cut_ston[cut_y_reg, cut_x_reg]

    temp_pi = cut_pi[cut_y_reg, cut_x_reg]
    temp_si = cut_i[cut_y_reg, cut_x_reg]
    temp_wrm = weights[cut_y_reg, cut_x_reg]
    temp_rm = rotmeas[cut_y_reg, cut_x_reg]
    temp_e_rm = e_rotmeas[cut_y_reg, cut_x_reg]
    temp = np.copy(temp_pi)
    temp_prob = prob_t[cut_y_reg, cut_x_reg]
    temp_m_chi = fracpol_chi2_arr[cut_y_reg, cut_x_reg]

    noise_box_greater = mf.idl_where(noise_box >= edge_threshold)
    noise_box_greater_python = noise_box >= edge_threshold
    noise_box_greater_shape = noise_box_greater.shape
    noise_box_less = mf.idl_where(noise_box < edge_threshold)
    noise_box_less_shape = noise_box_less.shape

    # Identifying the minimum ston in region:
    
    # # Ciara testings stuff:
    # print("line 1342 noise box greater: ", noise_box_greater)
        


    if noise_box_greater_shape[0] != 0:
        temp_noise = noise_box.flatten()[noise_box_greater]
        w_ston = np.min(temp_noise)

        # WE MAY WANT TO CONSIDER AVERAGING ALL POINTS WITHIN THE FWHM AREA!

        if len(noise_box_less_shape) != 0:
            temp[noise_box_less] = np.nan  # Makes sure that the pixels have a good S:N
            temp[noise_box_greater_python] = temp_rm[noise_box_greater_python]

        # Calculating weighted rm averages:

        n_pixels = noise_box_greater_shape[0]
        

        rm_avc = np.sum(temp_rm.flatten()[noise_box_greater]) / n_pixels
        frac_pol = np.sum(temp_pi.flatten()[noise_box_greater] / temp_si.flatten()[noise_box_greater]) / n_pixels

        w_pi = np.max(temp_pi.flatten()[noise_box_greater])
        w_si = np.max(temp_si.flatten()[noise_box_greater])

        temp[noise_box_greater_python] = 1.0 / np.sqrt(np.sum(temp_wrm[noise_box_greater_python]))
        wrm_dev = temp.flatten()[noise_box_greater[0]]

        temp[noise_box_greater_python] = np.sum(temp_rm[noise_box_greater_python] * temp_wrm[noise_box_greater_python]) / np.sum(temp_wrm[noise_box_greater_python])
        wrm_avg = temp.flatten()[noise_box_greater[0]]
        
        temp_wrm_avg = temp.copy()  # To make sure that when we update temp, it doesn't also update temp_wrm_avg
        # print("line 1374 temp_wrm_avg: ", temp_wrm_avg)
        temp[noise_box_greater_python] = np.sqrt(np.sum((temp_rm[noise_box_greater_python] - rm_avc)**2) / (n_pixels - 1))

        wrm_rms = temp.flatten()[noise_box_greater[0]]
        # print("line 1374 temp_wrm_avg[noise_box_greater_python]: ", temp_wrm_avg[noise_box_greater_python])
        temp[noise_box_greater_python] = np.sum(((temp_wrm_avg[noise_box_greater_python] - temp_rm[noise_box_greater_python])
                                                  / temp_e_rm[noise_box_greater_python])**2) / (n_pixels - 1)
        


        w_chi_2 = temp.flatten()[noise_box_greater[0]]
        if n_pixels == 1:
            w_chi_2 = 999.99
            
        
        temp[noise_box_greater_python] = np.sum(temp_e_rm[noise_box_greater_python]) / n_pixels


        # temp[noise_box_greater_python] = np.sum(temp_prob[noise_box_greater_python]) / n_pixels
        # w_prob_t = temp.flatten()[noise_box_greater[0]] * 100.0
        # Ciara Chisholm added the following line to replace the previous two lines
        #    to the optimize the line for python on Oct 31 2024
        w_prob_t = 100*np.sum(temp_prob[noise_box_greater_python]) / n_pixels
       
        
        temp[noise_box_greater_python] = np.sum(temp_m_chi[noise_box_greater_python]) / n_pixels
        fracpol_chi2_avg = temp.flatten()[noise_box_greater[0]]

    else:
        print(f'\nWARNING!!!\nNo usable pixels in this source...')
        source_ok = False

        wrm_dev = 0
        wrm_avg = 0
        wrm_rms = 0
        w_chi_2 = 0
        n_pixels = 0
        w_ston = 0
        frac_pol = 0
        w_prob_t = 0
        w_pi = 0
        w_si = 0
        fracpol_chi2_avg = 0

    return wrm_dev, wrm_avg, wrm_rms, w_chi_2, n_pixels, w_ston, source_ok, frac_pol, w_prob_t, w_pi, w_si, fracpol_chi2_avg


# PLOT 1 - POL. INT. MAP PART 2
def display_and_store2(w_rmf, w_drmf, w_chi2f, n_pixels, frac_pol_av, degrees, chitable, min_pol_threshold, w_prob_t,
                       x_long, y_lat, xpixmax, ypixmax, x_loc, y_loc, source_num, source_flag):
    """This function generates a series of strings that will be displayed in the Polarised Intensity Map.
    """
    rm_text = str(mf.truncate(w_rmf))
    rm_err_text = str(mf.truncate(w_drmf))

    chi_value = w_chi2f
    chi_string = str(chi_value)
    chi_pos = chi_string.find('.')
    chi_string = chi_string[0:chi_pos + 5]

    mask = np.array(degrees) == n_pixels - 1
    chitable_string = str(chitable[mask][0])

    m_string = str(frac_pol_av)
    ms_pos = m_string.find('.')
    m_string = m_string[0:ms_pos + 5]

    if w_chi2f < chitable.flatten()[mask] and min_pol_threshold < frac_pol_av < 1.0 and n_pixels >= 5 and w_prob_t >= 10.0:
        passfail = True
    else:
        passfail = False

    x_loc = mf.nround(x_loc)
    y_loc = mf.nround(y_loc)

    # Check if fitted location is closest to selected candidate:
    temp_array = (x_long[0, xpixmax] - x_long[y_loc, x_loc]) ** 2 + (y_lat[ypixmax, 0] - y_lat[y_loc, x_loc]) ** 2
    temp = np.min(temp_array)
    temp_index = np.where(temp_array == temp)[0][0]

    if temp_index != source_num:
        source_flag = (source_flag | 128)
        passfail = False

    pi_plot_text = {'rm_text': rm_text,
                    'rm_err_text': rm_err_text,
                    'chi_string': chi_string,
                    'chitable_string': chitable_string,
                    'm_string': m_string,
                    'n_pixels': n_pixels,
                    'passfail': passfail}

    return passfail, source_flag, pi_plot_text


def calc_av_pa_err(cut_x_reg, cut_y_reg, cut_dpsi_a, cut_dpsi_b, cut_dpsi_c, cut_dpsi_d):
    """This function calculates the average and standard deviation of the error in the polarisation angle.
    """
    cut_x_reg = cut_x_reg.astype(int)
    cut_y_reg = cut_y_reg.astype(int)
    reg_shape = cut_x_reg.shape

    crd = 180.0 / np.pi

    t_a = np.sum(cut_dpsi_a[cut_y_reg, cut_x_reg]) * crd / reg_shape[0]
    t_b = np.sum(cut_dpsi_b[cut_y_reg, cut_x_reg]) * crd / reg_shape[0]
    t_c = np.sum(cut_dpsi_c[cut_y_reg, cut_x_reg]) * crd / reg_shape[0]
    t_d = np.sum(cut_dpsi_d[cut_y_reg, cut_x_reg]) * crd / reg_shape[0]

    t_t = np.array([t_a, t_b, t_c, t_d])

    av_err = np.mean(t_t)
    dav_err = np.std(t_t) / 2.0

    return av_err, dav_err


# PLOT 4 - RM PLOT
def plot_rm_full3(data, x_long, y_lat, delta, rm_value, drm_value, pa_err, dpa_err, ston):
    """This function prepares data to be displayed in the Rotation Measure Map.
    """
    
    rm_data = data.copy()
    
    # The following four lines were commented out by Ciara Chisholm on January 6th 2026
    #   because python does not require it for colormaps. 
    
    # mask = data < -500.0
    # rm_data[mask] = -500.0

    # mask = data > 500.0
    # rm_data[mask] = 500.0

    mask = ston <= 3.0
    rm_data[mask] = 0.0


    min_x = np.min(x_long)
    max_x = np.max(x_long)
    min_y = np.min(y_lat)
    max_y = np.max(y_lat)

    d_2 = delta / 2.0

    rm_text = str(mf.truncate(rm_value))
    rm_err_text = str(mf.nround(drm_value))
    pa_text = str(mf.truncate(pa_err))
    pa_err_text = str(mf.nround(dpa_err))

    rm_plot_data = {'rm_data': rm_data,
                    'rm_text': rm_text,
                    'rm_err_text': rm_err_text,
                    'pa_text': pa_text,
                    'pa_err_text': pa_err_text,
                    'min_x': min_x,
                    'max_x': max_x,
                    'min_y': min_y,
                    'max_y': max_y,
                    'd_2': d_2}

    return rm_plot_data


# PLOT 3 - PEAK PIXEL LINEAR FIT
# def plot_strongest_linfit(pi, ua, ub, uc, ud, qa, qb, qc, qd, dpsi_c, dpsi_d, whw, rotmeas, e_rotmeas, psi, prob_t):
# the following line was changed from the previous by Ciara Chisholm Nov 26 2024
def plot_strongest_linfit(pi, ua, ub, uc, ud, qa, qb, qc, qd, dpsi_all_bands, whw, rotmeas, e_rotmeas, psi, prob_t, cut_noise):
    """This function prepares data to be displayed in the Peak Pixel Linear Fit.
    """
    # print(whw)
    #The following line added by Ciara Chisholm Nov 26 2024
    dpsi_a, dpsi_b, dpsi_c,dpsi_d = dpsi_all_bands
    
    
    
    pi_array = pi.flatten()[whw]
    temp = np.max(pi_array)
    temp_index = np.where(pi_array == temp)[0]  # imax
    
    
    
    cut_noise_peak = cut_noise.flatten()[whw[temp_index]]
    
    
    
    
    lambda_2 = (3.0 * 10**8 / (1420.4060 * 10**6))**2

    q_pix = np.array([[[qa.flatten()[whw[temp_index]]]], [[qb.flatten()[whw[temp_index]]]], [[qc.flatten()[whw[temp_index]]]], [[qd.flatten()[whw[temp_index]]]]])
    u_pix = np.array([[[ua.flatten()[whw[temp_index]]]], [[ub.flatten()[whw[temp_index]]]], [[uc.flatten()[whw[temp_index]]]], [[ud.flatten()[whw[temp_index]]]]])
    
    # qpix_flat, upix_flat = q_pix.flatten(), u_pix.flatten()
    
    
    
    # PI_pix = np.sqrt(qpix_flat**2 + upix_flat**2)
    # min_noise_bias = 0.28/1000 #Jy/beam
    # # noise_bias = cut_noise_peak
    # sigma_QUi = noise_bias
    
    # uncertainty_in_polarisation_angles = np.degrees(sigma_QUi /(2*PI_pix))
    

    pol_ang = (unwrap_angles(q_pix, u_pix) * 180 / np.pi).flatten() 

  
    
    #The following line added by Ciara Chisholm June 3rd 2024
    pol_ang_wrapped = wrap_data(pol_ang.copy(), 0, 180)
    # pol_err = np.array([dpsi_c.flatten()[whw[temp_index]], dpsi_c.flatten()[whw[temp_index]],\
    #                     dpsi_c.flatten()[whw[temp_index]], dpsi_d.flatten()[whw[temp_index]]]) * 180 / np.pi
    # The following line was changed by Ciara Chisholm on Nov 28 2024
    pol_err = np.array([dpsi_a.flatten()[whw[temp_index]], dpsi_b.flatten()[whw[temp_index]],\
                        dpsi_c.flatten()[whw[temp_index]], dpsi_d.flatten()[whw[temp_index]]]) * 180 / np.pi
    
    
    
    
    rm_pix = rotmeas.flatten()[whw[temp_index]]
    drm = e_rotmeas.flatten()[whw[temp_index]]
    pol_ang_0 = psi.flatten()[whw[temp_index]]
    probfit = prob_t.flatten()[whw[temp_index]]

    
    
    print(f"\n intercept {pol_ang_0} \n")
    
    
    # x_range = (np.arange(101) * (0.0455 - 0.0435) / 100) + 0.0435
    # The following was changed by Ciara Chisholm from the one above on June 4th 2024
    x_range = (np.arange(101) * (0.0457 - 0.0435) / 100) + 0.0435
    predicted = (rm_pix * (x_range - lambda_2) * 180 / np.pi + pol_ang_0)
    
    
    # The following currently commented code was added by Ciara Chisholm on Nov 29 2024
    debugging_prob_of_fit=False
    
    if debugging_prob_of_fit:
        print("pol ang ciara print: ", pol_ang)
    
        freq_bands = [0.0437, 0.0442, 0.0450, 0.0454][::-1]
    
    
        predict_bands = (rm_pix) *(np.array(freq_bands)- lambda_2) *180/np.pi + pol_ang_0
        print("predicted bands: ", predict_bands)
    
        chi_2 = np.sum((np.array(pol_ang)-predict_bands)**2/pol_err)
    
        print("Chi_2: ", chi_2)
    
        print("prob, exp(-chi_2/2)*100: ", np.exp(-chi_2/2)*100)
    
        predicted_angles_radians = np.radians(predict_bands)
        observed_angles_radians = np.radians(pol_ang) 
    
        chi_2_radians = np.sum((predicted_angles_radians - observed_angles_radians)**2/np.radians(pol_err))
            
        print("Chi_2 radians: ", chi_2_radians)
        print("prob_radians: ", np.exp(-chi_2_radians/2)*100)
    
    
    # The wrapped polarisation angle was added by Ciara Chisholm June 1st 2024 
    lin_fit_plot_data = {'pol_ang': pol_ang,
                          'pol_ang_wrapped': pol_ang_wrapped,
                          'rm_pix': rm_pix,
                          'drm': drm,
                          'pol_err': pol_err,
                          'predicted': predicted,
                          'probfit': probfit}

    return lin_fit_plot_data

# PLOT 3 - PEAK PIXEL LINEAR FIT


def unwrap_angles(q_arr, u_arr):
    """This function is Jo-Anne's algorithm to unwrap adjacent channels.
    """
    ## Note: Any comments with 2 pound signs were comments written by Ciara Chisholm, others also were she just forgot continue with this notation
    shape = q_arr.shape ## Getting the shape of the FITS files 
    result = np.zeros(shape) ## Creating an array to store the angles in 

    z = q_arr + (1j * u_arr) ## Calculating complex variable that used to find the PI, see page 60 of Dr. Brown's thesis for details
    z_norm = z / np.abs(z) ## Normalizing the complex variable. 

    d_phase = z_norm[1:, :, :] * np.conj(z_norm[0:shape[0] - 1, :, :]) ## Calculating the phase difference between the bands
    # d_phase = np.arctan(np.imag(d_phase), np.real(d_phase))
    # result[0, :, :] = np.arctan(np.imag(z_norm[0, :, :]), np.real(z_norm[0, :, :]))
    
    # The following two lines of codes were modified by Ciara Chisholm on 22/05/24 from the previous two commented lines
    d_phase = np.arctan2(np.imag(d_phase), np.real(d_phase)) # Finding the angle difference between bands
    result[0, :, :] = np.arctan2(np.imag(z_norm[0, :, :]), np.real(z_norm[0, :, :])) # Finding the polarsiation angle of the first band

    result_0 = result[0, :, :]
    # The following line was commented out by Ciara Chisholm on June 3rd 2024
    # result_0 = np.where(np.isnan(result_0), 0, result_0)  # Setting NaNs equal to zero
    
    ### The following section was taken from Jo-Anne Brown's unwrapping algorithm and added in on June 3rd 2024 by Ciara Chisholm
    ###########################################################################
    # Finding of the NaN values in the phase difference 
    w_nan = np.isnan(result_0) 
    
    if np.sum(w_nan) > 0:
        # If there is at least one NaN, we will temporarily set it equal to zero
        result_0[w_nan] = 0.0

    neg_mask = result_0 < 0.0
    if np.sum(neg_mask) > 0:
        result_0[neg_mask] += (2 * np.pi)
    if np.sum(w_nan) > 0:
        # If there used to be any NaNs, we set them equal to zero a few lines ago, and now we set them to NaNs once again.
        result_0[w_nan] = np.nan
    
    ### the section added ends here
    ###########################################################################
    
    result[0, :, :] = result_0

    for index in range(shape[0] - 1): # Going through bands b,c, and d to find their associated angles
        result[index + 1, :, :] = result[index, :, :] + d_phase[index, :, :]
    
 
    result = result / 2.0 # Getting the polarisation angles (see equation 4.25 on page 61 of Dr. Brown's thesis.)

    
    return result


def parse_flag(flag):
    """This flag takes in the numerical form of the source flag and returns a list of flags (in words) that were active.
    """
    flag_list = []

    # & is the python bitwise AND
    # If you & the full flag with a flag that you want to check, the result will be 0 if it isn't set
    # and equal to the flag if it is set.
    # e.g. 5 & 1 = 1, meaning the 1 flag is set, and 5 & 4 = 4, meaning the 4 flag is set,
    # while 5 & 1024 = 0, meaning the 1024 flag is not set
    if flag & 1 != 0:
        flag_list.append('Revisit later')
    if flag & 2 != 0:
        flag_list.append('Manual fail')
    if flag & 4 != 0:
        flag_list.append('Gaussian fit failed')
    if flag & 8 != 0:
        flag_list.append('Too few/many pixels with sufficiant S:N')
    if flag & 16 != 0:
        flag_list.append('Fractional polarization too low/high.')
    if flag & 32 != 0:
        flag_list.append('Failed RM-averaging Chi-square test')
    if flag & 64 != 0:
        flag_list.append('Failed average linfit Chi-square test')
    if flag & 128 != 0:
        flag_list.append('False detection from neighbour')
    if flag & 256 != 0:
        flag_list.append('Mark presence of unidenfified double')
    if flag & 512 != 0:
        flag_list.append('Go back one source')
    if flag & 1024 != 0:
        flag_list.append('Retry source')
    # The following 4 lines of code were added by Ciara Chisholm on June 25th 2024. 
    if flag & 2048 !=0:
        flag_list.append('Gradient across the source')
    if flag & 4096 !=0:
        flag_list.append('Miscellanous flag')
    if flag & 8192 !=0:
        flag_list.append('Fit two peaks as one')
    
    return flag_list


def is_mod_flag_valid(input_flag, morph):
    """This function performs some basic error validation to determine if the user input when setting/unsetting flags is valid.
    For example, '16', '5', and '13' are all valid inputs, but 'bread', '-9999', or '3.14159' are not valid inputs.
    """
    flag_valid = True

    try:  # If the input flag is a number (a float or an integer)
        float_flag = float(input_flag)

        if not float_flag.is_integer():  # If the input flag is a float but not an integer (e.g. 3.14159), the input is invalid
            flag_valid = False
        else:  # If the input flag is an integer
            # if (not morph) and (float_flag < 0 or float_flag > 2047):  # The maximum value of a flag alteration is the sum of all the possible flags
            # The following line was changed from the previous by Ciara Chisholm on June 26th 2024
            if (not morph) and (float_flag < 0 or float_flag > 8193):  # The maximum value of a flag alteration is the sum of all the possible flags
                flag_valid = False
            if morph and (float_flag < 0 or float_flag > 31):  # The maximum value of a morphology flag alteration is the sum of all the possible morphology flags
                flag_valid = False

    except ValueError:  # If the input flag is not a number (e.g. 'bread'), the input is invalid
        flag_valid = False

    return flag_valid


# The following function was added in by Ciara Chisholm on September 16th 2025
def initial_mask(mosaic,stokes, input_directory,  i, pi_data, print_mask_completed=False):
    """Function adds a small mask to the second source in a pair for when not including
    it causes the code to fit one gaussian to both sources in the pair. 
    
    Parameters:
        mosaic (str): The mosaic the pair lies in
        
        stokes (dictionary): The stokes data the code is working with
        
        input_directory (str): the input directory to the files
        
        i (int): the number of sources looked at by the code already.
        
        pi_data (dictionary): dictionary containing all the source information
        
        print_mask_completed (bool): whether to print if the mask was completed or not. Default: False
    Return:
        stokes (dictionary): A dictionary containing the new stokes maps to use, with or without masking. """
    
    # Creating a list for every stokes band: 
    mask_width = 1 #number of pixels to mask out around the peak pixel (0=only the peak pixel,)
    bands = ['I', 'Q_A', 'Q_B', 'Q_C', 'Q_D', 'U_A', 'U_B', 'U_C', 'U_D']
    mosaics_w_initial_masks = ["ij2", "g0", "g5", "g4", "v2", "ey2", "y1"]
    # gives the source number i in the orignal code when you want the mask to be placed on the next source i+1. 
    # pair_to_mask_by_mosaic = {"mij2":0, "mg0":26, "mg5":2, "mg4":0, "my1":0, "mv2":0, "mey2":2}
    
    OG_vals, header = read_qu_data_cve(input_directory, mosaic)
    
    
    
    if i%2==0:# If it's the first soure in the pair. 
        xcoord_to_mask = pi_data["xpixmax"][i+1]
        ycoord_to_mask = pi_data["ypixmax"][i+1]
        
        for b in bands:
            stokes[b][ycoord_to_mask-mask_width:ycoord_to_mask+mask_width+1, xcoord_to_mask-mask_width:xcoord_to_mask+mask_width+1]=0
            
        if print_mask_completed: print("\n Second source masked \n")
    else:# if it's the second source in the pair 
        xcoord_to_mask = pi_data["xpixmax"][i]
        ycoord_to_mask = pi_data["ypixmax"][i]
        
        for b in bands:
            stokes[b][ycoord_to_mask-mask_width:ycoord_to_mask+mask_width+1, xcoord_to_mask-mask_width:xcoord_to_mask+mask_width+1]=\
                OG_vals[b][ycoord_to_mask-mask_width:ycoord_to_mask+mask_width+1, xcoord_to_mask-mask_width:xcoord_to_mask+mask_width+1]
        if print_mask_completed: print("\n Second source mask removed\n ")
        
            
    return stokes
    
   

def main(input_directory, output_directory, chitable_directory, fig_path):
    from datetime import datetime
    from astropy.io import fits
    
    print(f'input directory: {input_directory}')
    print(f'output directory: {output_directory}')

    print(f'\n\nVERIFY THESE INPUTS: ')
    stokes_i_threshold = 1.2 / 1000
    min_pol_threshold = 0.02
    print(f'\n\nCurrent minimum polarisation threshold is: {min_pol_threshold}')
    alpha = 0.003  # fraction of I to be included in PA error
    print(f'\nAlpha factor in PA error calculations: {alpha}')
    output_ext = 'final_003I'
    print(f'\nOutput file extension: {output_ext}')
    edge_threshold = 5.0
    print(f'\nEdge threshold is {edge_threshold} sigma')
    fwxm_v = 2.0  
    print(f'\nCurrently set to FULL WIDTH, HALF MAX') # I think this should be "HALF MAX" not "MALF MAX"
    fwxm = np.sqrt(2.0 * np.log(fwxm_v))
    print(f'FWXM factor: {fwxm}')
    
    
    # Following two lines added by Ciara Chisholm on September 17th 2025
    mask_second = False
    print("Mask the second source when fitting the first in a pair: ", str(mask_second))
    
    save_figs_auto = False
    print("Save plots automatically: ", str(mask_second))
    
    
    # The following two lines were added by Ciara Chisholm on December 22nd 2025
    create_FITS = True
    get_user_input = True
    print("Getting user input on RMs: ", str(get_user_input))
    light_background =True

    # **************************************************************************
    # READING IN THE DATA:
    # **************************************************************************

    # Reading in data from PI  table

    
    mosaic_name = ''
    mosaic_caps = ''
    mosaic_path_exists = False
    while not mosaic_path_exists:
        mosaic_caps = input('\nEnter the name of the mosaic you would like to analyze in all caps [MA1]: ')
        if mosaic_caps == '':
            mosaic_caps = 'MA1'
        mosaic_lower = mosaic_caps.lower()
        if mosaic_lower[0] == 'm':
            mosaic_name = mosaic_lower[1:]
        else:
            mosaic_name = mosaic_lower

        mosaic_path = f'{input_directory}{mosaic_name}'
        if not os.path.isdir(mosaic_path):
            print(f'\nThere is no input data corresponding to this mosaic. Please try again.')
        else:
            mosaic_path_exists = True
            print(f'\nCalculations for {mosaic_caps}:')

    sourcelist_name = input('Enter the table to use [_Taylor17_candidates]: ')
    if sourcelist_name == '':
        sourcelist_name = '_Taylor17_candidates'

    mosaic_path = f'{input_directory}{mosaic_name}'
    mosaic_caps= "M"+ mosaic_name.upper()
    
    print("Beginning Calculations for", mosaic_caps)
    
    if not os.path.isdir(mosaic_path):
        print(f'\nThere is no input data corresponding to this mosaic. Please try again.')
    else:
        mosaic_path_exists = True
        print(f'\nCalculations for M{mosaic_path.upper()}:')
    
    sourcelist_name = "_Twins"

    sourcelist_path = f'{output_directory}{mosaic_name}/M{mosaic_name.upper()}{sourcelist_name}.dat'  # Default: ../Data/output_data/a1/MA1_Taylor17_candidates.dat

    if not os.path.exists(sourcelist_path):
        print(f'\nThere is no polarised candidate sourcelist for this mosaic. One must be generated before Rotation Measure analysis can begin.')
    else:
        # Ciara Chisholm October 11 2024 added line to copy the PI data. 
        # Please note that pi_data does NOT strickly contain information about the PI image,
        #   it instead contains information from the source i.e. the number of sources, 
        #   the x and y pixel coordinates, the galactic coordinates, the PI peak value,
        #   the Stokes I peak value, and the signal to noise of the pixel. 
        pi_data = read_pi_source_list_cd(sourcelist_path)
        num_sources = pi_data['len']


        
        # Reading in data from Stokes I, Q and U FITS files for each band
        # stokes, header = read_qu_data_cve(input_directory, mosaic_name)
        # The following two lines were modified from the line above by Ciara Chisholm on Oct 11 2024
        stokes_OG, header = read_qu_data_cve(input_directory, mosaic_name)
        # stokes= copy.deepcopy(stokes_OG)
        stokes= stokes_OG.copy()
        
        
        
        
        # Readying in from Chi Table

        degrees, chi2, chitable = read_chitable(chitable_directory)

        # **************************************************************************
        # DEFINING THE DATA SETS:
        # **************************************************************************

        lc = np.zeros(num_sources) # Creating longitude array
        bc = np.zeros(num_sources) # Galactic latitude array. 
        wrmc = np.zeros(num_sources) # Weighted rotation measure 
        wdrmc = np.zeros(num_sources) # Error in weighted RM
        rmsc = np.zeros(num_sources) # likely the RMS noise array
        pic = np.zeros(num_sources) # PI 
        snc = np.zeros(num_sources) # Signal to noise calculation
        mc = np.zeros(num_sources) # Fractional Polarisation PI/SI 
        chi2c = np.zeros(num_sources) # Chi squared values
        npixels = np.zeros(num_sources) # The number of pixels in FWHM
        sic = np.zeros(num_sources) # ? Stokes I value
        probchi = np.zeros(num_sources) # the array to store the probability of Chi calculation
        dpaav = np.zeros(num_sources) # Error in polarisation angle or change in PA (polarsation angle)
        flag = np.zeros(num_sources) # An array containing the flag stuff
        morphology = np.zeros(num_sources) # An array containing the morphology code for the source.
        rm_peakpix = np.zeros(num_sources) # Storing the RM of the peak pixel of each source
        drm_peakpix = np.zeros(num_sources)  # Error of RM of peak pixel
        fracpol_chi2_src = np.zeros(num_sources) # Cameron added in probably -> for something?
        
        
        # x_array and y_array return a array of pixel values, x_long and y_lat are galactic coordinates.
        x_array, y_array, x_long, y_lat = fh.make_xy_arrays(header['I'])

        # **************************************************************************
        # DETERMINING THE FREQUENCIES FOR THE FOUR BANDS:
        # **************************************************************************
        
        
        crd = 180 / np.pi # Convert to radians factor
        
        # Getting the frequencies from the fits files
        freq_a = float(header['U_A']['OBSFREQ'])
        freq_b = float(header['U_B']['OBSFREQ'])
        freq_c = float(header['U_C']['OBSFREQ'])
        freq_d = float(header['U_D']['OBSFREQ'])

        # Calculating the wavelength of each frequency.
        la2 = (3 * 10 ** 8 / freq_a) ** 2
        lb2 = (3 * 10 ** 8 / freq_b) ** 2
        lc2 = (3 * 10 ** 8 / freq_c) ** 2
        ld2 = (3 * 10 ** 8 / freq_d) ** 2
        
        
        lx = [la2, lb2, lc2, ld2]
        
        # Checking to make sure the frequencies match the theoritical frequencies. 
        freq_ok = check_freq2(freq_a, freq_b, freq_c, freq_d, header['Q_A'], header['Q_B'], header['Q_C'], header['Q_D'])
        # NOTE: Some mosaics (MM1-2, MN1-2) have a problem with the band C frequences. It's safe to skip past, because the correct numbers are used.
        
        
        # (This is mostly obsolete code) Checking if the frequencies agree 
        if not freq_ok:
            do_continue = input(f'Do you want to continue anyway? (Y/N) [N] ')
            if do_continue.lower() == 'y' or do_continue.lower() == 'yes':
                do_continue = True
            else:
                do_continue = False
        else:
            do_continue = True
        
        
        if do_continue:
            lambda_default = 3 * 10 ** 8 / (1420.4060 * 10 ** 6)
            lambda_2 = lambda_default ** 2

            # **************************************************************************
            # THINGS FOR PSS_SUB:
            # **************************************************************************
            delta = np.abs(float(header['I']['CDELT1']))

            # **************************************************************************
            print('\nBeginning Calculations...\n')
            # **************************************************************************

            box_halfwidth = 20
            pi_units = 'mJy/beam'


            i = 0
            while i < num_sources: 
                source_num = i
                source_flag = 0
                
                
                
                
                
                # Added by Ciara Chisholm on September 16th 2025
                stokes = initial_mask(mosaic_name, stokes,input_directory, i, pi_data, print_mask_completed=True)
                

                print(f'\nCalculations for source #{source_num}')

                xpix_max_i = pi_data['xpixmax'][i] # Getting the x coordinate of maximum intensity in pixel values for the source 
                ypix_max_i = pi_data['ypixmax'][i] # Getting the y coordinate of maximum intensity in pixel values for the source 
                
                
                
                # Because the box sizes are different between the noise calculation and the Gauss fit,
                # it's convenient to define a full-size PI array, fill part of it with the FG-subtracted and
                # de-biased PI, then feed that into the old code so it can extract its own box.
                # Same for S:N array and noise
                temp_pi = np.zeros(x_array.shape) # this is a an array to later contain a copy of the PI data- Ciara Chisholm 
                ston = np.zeros(x_array.shape) # Creating an array to store the Signal to Noise of the image
                noise = np.zeros(x_array.shape) # Array to store the noise  

                x_min = mf.nround(xpix_max_i) - box_halfwidth
                x_max = mf.nround(xpix_max_i) + box_halfwidth + 1 # The + 1s are because IDL indexes arrays differently than python
                y_min = mf.nround(ypix_max_i) - box_halfwidth
                y_max = mf.nround(ypix_max_i) + box_halfwidth + 1 # The + 1s are because IDL indexes arrays differently than python

                stamp = {}
                
                # Ciara Chisholm October 11 2024 modified the arrays to include the copy of the original arrays.
                OG_arrays = [x_array, y_array, stokes['I'], stokes['Q_A'], stokes['Q_B'],
                          stokes['Q_C'], stokes['Q_D'], stokes['U_A'], stokes['U_B'], stokes['U_C'], stokes['U_D']]
                arrays = OG_arrays.copy()
                array_label = ['xarr', 'yarr', 'I', 'Q_A', 'Q_B', 'Q_C', 'Q_D', 'U_A', 'U_B', 'U_C', 'U_D']
                for array in range(len(arrays)):
                    # Note: argument 2-5 are the same as x_min, x_max, y_min, y_max. 
                    #   The +1 is added in the cut_out function. 
                    stamp[array_label[array]] = fh.cut_out_stamp(arrays[array],
                                                                 mf.nround(xpix_max_i) - box_halfwidth,
                                                                 mf.nround(xpix_max_i) + box_halfwidth,
                                                                 mf.nround(ypix_max_i) - box_halfwidth,
                                                                 mf.nround(ypix_max_i) + box_halfwidth)
                    # The stamps have been stored in a dictionary, just like the fits data was. To access simply call, for example, stamp['xarr'], or stamp['I']

                # Noise calculations:
                # The main purpose of this is to generate the noise-arr. The foreground subtraction isn't used here.
                g_coords = SkyCoord(l=pi_data['lmax'][i], b=pi_data['bmax'][i], frame='galactic', unit='degree')
                ra = g_coords.fk5.ra.deg
                dec = g_coords.fk5.dec.deg

                annulus_pixels = ac.calculate_annulus(ra, dec, stamp['xarr'], stamp['yarr'], xpix_max_i, ypix_max_i, stamp['I'], stokes_i_threshold)
                foreground_pixels = annulus_pixels[0]


                
                # foreground contains an array of noise for each channel, sigma_qu is mean noise of all channels. 
                foreground_vector, sigma_qu = ac.estimate_local_noise(foreground_pixels,
                                                                      stamp['Q_A'],
                                                                      stamp['Q_B'],
                                                                      stamp['Q_C'],
                                                                      stamp['Q_D'],
                                                                      stamp['U_A'],
                                                                      stamp['U_B'],
                                                                      stamp['U_C'],
                                                                      stamp['U_D'])

                pi_debiased, noise_arr, ston_arr = ac.construct_new_ston_cutout(stamp['I'],
                                                                                stamp['Q_A'],
                                                                                stamp['Q_B'],
                                                                                stamp['Q_C'],
                                                                                stamp['Q_D'],
                                                                                stamp['U_A'],
                                                                                stamp['U_B'],
                                                                                stamp['U_C'],
                                                                                stamp['U_D'],
                                                                                foreground_vector,
                                                                                sigma_qu)

                temp_pi[y_min:y_max, x_min:x_max] = pi_debiased * 1000 # Why is the noise being multiplied by 1000? convert to mJy? - Ciara Chisholm April 15 2024
                ston[y_min:y_max, x_min:x_max] = ston_arr
                noise[y_min:y_max, x_min:x_max] = noise_arr

                
                # Ciara Chisholm May 2nd 2024: This is where the gaussian fitting is done.
                #   What is important for me atm is that x_loc, and y_loc is the location of the fitted gaussian. 
                # (x_a_2, y_a_2, whw, w3s, x_loc, y_loc, x_reg, y_reg, x_tick_vals, y_tick_vals, x_fwhm, y_fwhm,
                #  w_annulus, gauss_parameters, source_ok, pi_plot_data, source_flag) = pss_subplot_get4_cve(temp_pi,
                #                                                                                            x_array,
                #                                                                                            y_array,
                #                                                                                            x_long,
                #                                                                                            y_lat,
                #                                                                                            delta,
                #                                                                                            pi_units,
                #                                                                                            xpix_max_i,
                #                                                                                            ypix_max_i,
                #                                                                                            fwxm, # This is conversion factor to go from σ to HWHM
                #                                                                                            source_num,
                #                                                                                            source_flag)
                # Ciara Chisholm added the "to_mask_coordinates" value on October 22nd 2024
                (x_a_2, y_a_2, whw, w3s, x_loc, y_loc, x_reg, y_reg, x_tick_vals, y_tick_vals, x_fwhm, y_fwhm,
                 w_annulus, gauss_parameters, source_ok, pi_plot_data, source_flag, to_mask_coordinates) = pss_subplot_get4_cve(temp_pi,
                                                                                                           x_array,
                                                                                                           y_array,
                                                                                                           x_long,
                                                                                                           y_lat,
                                                                                                           delta,
                                                                                                           pi_units,
                                                                                                           xpix_max_i,
                                                                                                           ypix_max_i,
                                                                                                           fwxm, # This is conversion factor to go from σ to HWHM
                                                                                                           source_num,
                                                                                                           source_flag)

                                                                                                           
                si_plot_data = pss_stokes_i_plot(stokes['I'] * 1000, x_long, y_lat, xpix_max_i, ypix_max_i,
                                                 gauss_parameters, fwxm, pi_data['xpixmax'], pi_data['ypixmax'])

                # do_flagging = input(f'Do you want to activate pixel flagging? (Y/N) [N]: ')
                # do_flagging = ''
                # if do_flagging.lower() == 'y' or do_flagging.lower() == 'yes':
                #     good_fit = False
                #     while not good_fit:
                #         # Fit Gaussian to PI, get annulus
                #         (x_a_2, y_a_2, whw, w3s, x_loc, y_loc, x_reg, y_reg, x_tick_vals, y_tick_vals, x_fwhm, y_fwhm,
                #          w_annulus, gauss_parameters, source_ok, pi_plot_data, source_flag) = pss_subplot_get4_cve(temp_pi,
                #                                                                                                    x_array,
                #                                                                                                    y_array,
                #                                                                                                    x_long,
                #                                                                                                    y_lat,
                #                                                                                                    delta,
                #                                                                                                    pi_units,
                #                                                                                                    xpix_max_i,
                #                                                                                                    ypix_max_i,
                #                                                                                                    fwxm,
                #                                                                                                    pi_data["lmax"][source_num],
                #                                                                                                    pi_data["bmax"][source_num],
                #                                                                                                    source_num,
                #                                                                                                    source_flag)
                #     si_plot_data = pss_stokes_i_plot(stokes['I'] * 1000, x_long, y_lat,
                #                                      xpix_max_i, ypix_max_i, gauss_parameters, fwxm, pi_data['xpixmax'], pi_data['ypixmax'])

                if whw.shape[0] == 0:
                    source_ok = False
                if w_annulus.shape[0] == 0:
                    source_ok = False
                if not source_ok:
                    # If the 4 flag (Gaussian fit failed) is not set already, set it
                    source_flag = (source_flag | 4)

                if source_ok:
                    print('\nSource OK in Main program...')

                    (cut_pi, cut_i, cut_qa, cut_qb, cut_qc, cut_qd,
                     cut_ua, cut_ub, cut_uc, cut_ud, cut_xarr, cut_yarr,
                     cut_noise, cut_x_long, cut_y_lat, cut_ston,
                     cut_x_loc, cut_y_loc, cut_x_reg, cut_y_reg,
                     x3s, y3s, cut_x_ann, cut_y_ann) = get_subarrays_cve(stokes['I'],
                                                                         stokes['Q_A'],
                                                                         stokes['Q_B'],
                                                                         stokes['Q_C'],
                                                                         stokes['Q_D'],
                                                                         stokes['U_A'],
                                                                         stokes['U_B'],
                                                                         stokes['U_C'],
                                                                         stokes['U_D'],
                                                                         x_long,
                                                                         y_lat,
                                                                         x_loc,
                                                                         y_loc,
                                                                         x_a_2,
                                                                         y_a_2,
                                                                         delta,
                                                                         temp_pi,
                                                                         ston,
                                                                         whw,
                                                                         w3s,
                                                                         noise,
                                                                         w_annulus)
                    

                    stamp['xarr'] = cut_xarr
                    stamp['yarr'] = cut_yarr

                    if not source_ok:  # I don't think this conditional will ever be true
                        # If the 8 flag (Too few/many pixels with sufficient S:N) is not set already, set it
                        source_flag = (source_flag | 8)

                    # **************************************************************************
                    # CALCULATIONS WITH BACKGROUND REMOVED:
                    # **************************************************************************

                    print(f'\n**** BACKGROUND CALCS ****')
                    
                    

                    cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud = sbg_av3(cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud, cut_x_ann, cut_y_ann)
                    
                    
                    
                    
                    print(f'\n**** ----------------- ****')


                    # Error in polarisation angles:
                    cut_dpsi_a = cut_noise / (2 * np.sqrt(cut_ua ** 2 + cut_qa ** 2))
                    cut_dpsi_b = cut_noise / (2 * np.sqrt(cut_ub ** 2 + cut_qb ** 2))
                    cut_dpsi_c = cut_noise / (2 * np.sqrt(cut_uc ** 2 + cut_qc ** 2))
                    cut_dpsi_d = cut_noise / (2 * np.sqrt(cut_ud ** 2 + cut_qd ** 2))
                    
                    
                    
                    
                    
                    
                    
                    # The following line was added by Ciara Chisholm on Nov 26th 2024
                    cut_dpsi_all_bands = np.array([cut_dpsi_a,cut_dpsi_b,cut_dpsi_c,cut_dpsi_d])
                    
                    below_min = cut_dpsi_all_bands < np.pi/180
                    cut_dpsi_all_bands[below_min] = np.pi/180
                    # pol_err_all_bands=[]
                    # for dpsi_band in cut_dpsi_all_bands:
                    #     errors = dpsi_band.copy()
                    #     below_minimum = dpsi_band<(np.pi/180)
                    #     errors[below_minimum] = (np.pi/180)
                    #     pol_err_all_bands.append(errors)
                        
                    # cut_dpsi_all_bands = np.array(pol_err_all_bands)
                    
   
    
   
                    # Note) this does not return the polarisation angles - Ciara Chisholm June 2nd 2024
                    rm_array_rbg, rm_error_rbg, psi_rbg, prob_t = unwrap_pa_and_rm_calcs4(cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud,
                                                                                          lambda_2, crd, cut_dpsi_a, cut_dpsi_b, cut_dpsi_c, cut_dpsi_d,
                                                                                          stamp['xarr'], stamp['yarr'], lx)

                    frac_pol_chi2_arr = fractional_polarisation_test(cut_i, cut_qa, cut_qb, cut_qc, cut_qd, cut_ua, cut_ub, cut_uc, cut_ud, cut_noise)

                    (w_drmf_rbg, w_rmf_rbg, w_rm_rmsf_rbg, w_chi2f_rbg, n_pixels_rbg, w_ston_rbg, source_ok,
                     frac_pol_av, w_prob_t, w_pi, w_si, fracpol_chi2_avg) = avg_rm_calcs8(cut_pi,
                                                                                          rm_array_rbg,
                                                                                          rm_error_rbg,
                                                                                          cut_ston,
                                                                                          edge_threshold,
                                                                                          cut_x_reg,
                                                                                          cut_y_reg,
                                                                                          cut_i * 1000,
                                                                                          prob_t,
                                                                                          frac_pol_chi2_arr)

                    if n_pixels_rbg > 100:  # Sometimes this part of the code fits huge Gaussians with > 100 pixels and the chisq table can't handle it
                        source_ok = False
                        # If the 4 flag (Gaussian fit failed) is not set already, set it
                        source_flag = (source_flag | 4)

                    if not source_ok:
                        # If the 8 flag (Too few/many pixels with sufficient S:N) is not set already, set it
                        source_flag = (source_flag | 8)

                    if source_ok:
                        pass_fail, source_flag, pi_plot_text = display_and_store2(w_rmf_rbg, w_drmf_rbg, w_chi2f_rbg, n_pixels_rbg, frac_pol_av,
                                                                                  degrees, chitable, min_pol_threshold, w_prob_t, x_long, y_lat,
                                                                                  pi_data['xpixmax'], pi_data['ypixmax'], x_loc, y_loc, i, source_flag)

                        av_err, dav_err = calc_av_pa_err(cut_x_reg, cut_y_reg, cut_dpsi_a, cut_dpsi_b, cut_dpsi_c, cut_dpsi_d)

                        # rm_plot_data = plot_rm_full3(rm_array_rbg, cut_x_long, cut_y_lat, delta, w_rmf_rbg, w_drmf_rbg, av_err, dav_err, cut_ston)
                        # Ciara Chisholm added the following line to replace the previous one on October 13 2024. Numpy arrays stay linked so the RM array (rm_array_rbg)
                        #   becomes the image array. Any pixels with too low of signal to noise ratio will be zero and RM over 500 will be set to zero. 
                        #   This usually wasn't a problem because most RM weren't much greater than 500, but it was for a few. 
                        rm_plot_data = plot_rm_full3(rm_array_rbg.copy(), cut_x_long, cut_y_lat, delta, w_rmf_rbg, w_drmf_rbg, av_err, dav_err, cut_ston)

                        

                        # linear_fit_plot_data = plot_strongest_linfit(cut_pi, cut_ua, cut_ub, cut_uc, cut_ud, cut_qa, cut_qb, cut_qc, cut_qd,
                        #                                              cut_dpsi_c, cut_dpsi_d, whw, rm_array_rbg, rm_error_rbg, psi_rbg, prob_t)
                        # The following two lines were changed from the previous two by Ciara Chisholm Nov 26th 2024
                        linear_fit_plot_data = plot_strongest_linfit(cut_pi, cut_ua, cut_ub, cut_uc, cut_ud, cut_qa, cut_qb, cut_qc, cut_qd,
                                                                     cut_dpsi_all_bands, whw, rm_array_rbg, rm_error_rbg, psi_rbg, prob_t, cut_noise)


                        pi_array = cut_pi.flatten()[whw]
                        temp = np.max(pi_array)
                        temp_index = np.where(pi_array == temp)[0]

                        rm_peakpix[source_num] = rm_array_rbg.flatten()[whw[temp_index]]
                        drm_peakpix[source_num] = rm_error_rbg.flatten()[whw[temp_index]]

                        # Recording the rest of the data for output tables

                        x_loc = mf.nround(x_loc)
                        y_loc = mf.nround(y_loc)

                        lc[source_num] = x_long[y_loc, x_loc]
                        bc[source_num] = y_lat[y_loc, x_loc]
                        wrmc[source_num] = w_rmf_rbg
                        wdrmc[source_num] = w_drmf_rbg
                        rmsc[source_num] = w_rm_rmsf_rbg
                        dpaav[source_num] = av_err
                        chi2c[source_num] = w_chi2f_rbg
                        npixels[source_num] = n_pixels_rbg
                        sic[source_num] = w_si
                        mc[source_num] = frac_pol_av
                        pic[source_num] = w_pi
                        snc[source_num] = w_ston_rbg
                        probchi[source_num] = w_prob_t
                        fracpol_chi2_src[source_num] = fracpol_chi2_avg


                        
                        if n_pixels_rbg < 5:
                            # If the 8 flag (Too few/many pixels with sufficient S:N) is not set already, set it
                            source_flag = (source_flag | 8)
                        if frac_pol_av <= min_pol_threshold or frac_pol_av > 1.0:
                            # If the 16 flag (Fractional polarisation too low/high) is not set already, set it
                            source_flag = (source_flag | 16)
                        if w_chi2f_rbg >= chitable[n_pixels_rbg - 1]:
                            # If the 32 flag (Failed RM-averaging Chi-square test) is not set already, set it
                            source_flag = (source_flag | 32)
                        if w_prob_t < 10.0:
                            # If the 64 flag (Failed average linfit Chi-square test) is not set already, set it
                            source_flag = (source_flag | 64)
                        # # The following code was modified from the above by Ciara Chisholm on October 29th 2024
                        # # Failing the source if the average probability of the 
                        # #   fit failed or if the peak pixel fit failed
                        # if w_prob_t < 10.0 or linear_fit_plot_data["probfit"]*100<10.0:
                        #     # If the 64 flag (Failed average linfit Chi-square test) is not set already, set it
                        #     source_flag = (source_flag | 64)
                        
                        # The following if statement was added by Ciara Chisholm Oct 31 2024 
                        if source_flag !=0:
                            pass_fail=False # If the source is flagged, failing it. 

                        # **************************************************************************
                        # PLOTTING / FLAGGING SECTION:
                        # **************************************************************************
                        
                        
                        
                        if light_background:
                            plt.style.use('tableau-colorblind10') # for light colored background
                        else:
                            plt.style.use('dark_background') # for dark colored background.
                        
                        plt.rcParams['figure.constrained_layout.use'] = True
                        fig = plt.figure(figsize=(12, 10), dpi=80)
                        spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
                        ax1 = fig.add_subplot(spec2[0, 0])
                        ax2 = fig.add_subplot(spec2[0, 1])
                        ax3 = fig.add_subplot(spec2[1, 0])
                        ax4 = fig.add_subplot(spec2[1, 1])
                        # The following lines were added by Ciara Chisholm on June 26th 2024
                        plt.suptitle(f'Source {i+1} in {mosaic_caps.upper()}', x = 0.48, fontsize = 20,fontweight="bold")

                        # *************************************
                        # POL INT MAP
                        # *************************************
                        
                        # plot_pol_int_map(pi_plot_data['x_l_2'], pi_plot_data['y_b_2'],
                        ### The line below was changed from the line above on May 5, 2025 by Ciara Chisholm to have the same ticks for each cutout
                        plot_pol_int_map(cut_x_long, cut_y_lat,#pi_plot_data['x_l_2'], pi_plot_data['y_b_2'],
                                         pi_plot_data['data_fit'],
                                         pi_plot_data['levels'],
                                         mosaic_name,
                                         pi_plot_data['num'],
                                         pi_plot_data['x_gauss_rot'], pi_plot_data['y_gauss_rot'],
                                         pi_plot_data['x_center_gauss'], pi_plot_data['y_center_gauss'],
                                         x_long, y_lat,
                                         x_loc, y_loc,
                                         pi_plot_data['x_fwxm_ae'], pi_plot_data['y_fwxm_ae'],
                                         pi_plot_data['x_fwxm_se'], pi_plot_data['y_fwxm_se'],
                                         pi_plot_text['rm_text'], pi_plot_text['rm_err_text'],
                                         pi_plot_text['chi_string'],
                                         pi_plot_text['chitable_string'],
                                         pi_plot_text['m_string'],
                                         pi_plot_text['n_pixels'],
                                         # pi_plot_text['passfail'],
                                         pass_fail,
                                         ax1, light_background=light_background)

                        # *************************************
                        # STOKES I MAP
                        # *************************************
                        # plot_stokes_i_map(si_plot_data['long_arr'], si_plot_data['lat_arr'],
                        ### The line below was changed from the line above on May 5, 2025 by Ciara Chisholm to have the same ticks for each cutout
                        plot_stokes_i_map(cut_x_long, cut_y_lat,#si_plot_data['long_arr'], si_plot_data['lat_arr'],
                                          si_plot_data['data'],
                                          si_plot_data['levels'],
                                          si_plot_data['x_gauss_rot1'], si_plot_data['y_gauss_rot1'],
                                          si_plot_data['x_gauss_rot2'], si_plot_data['y_gauss_rot2'],
                                          gauss_parameters,
                                          source_flag,
                                          x_long, y_lat,
                                          xpix_max_i, ypix_max_i,
                                          si_plot_data['npix'],
                                          ax2, light_background=light_background)

                        # *************************************
                        # PEAK PIXEL LINEAR FIT
                        # *************************************
                        print("pol_ang data: ", linear_fit_plot_data['pol_ang'])
                        # Uncomment the line below
                        print("predicted data: ", linear_fit_plot_data['predicted'])
                        
                        
                        
                        plot_peak_pixel_linear_fit(lx,
                                                   linear_fit_plot_data['pol_ang'],
                                                    linear_fit_plot_data['pol_ang_wrapped'],
                                                   linear_fit_plot_data['rm_pix'], linear_fit_plot_data['drm'],
                                                   linear_fit_plot_data['pol_err'],
                                                   linear_fit_plot_data['predicted'],
                                                   linear_fit_plot_data['probfit'],
                                                   ax3, light_background=light_background)

                        # *************************************
                        # RM MAP
                        # *************************************

                        plot_rm_map(rm_plot_data['rm_data'],
                                    rm_plot_data['rm_text'], rm_plot_data['rm_err_text'],
                                    rm_plot_data['pa_text'], rm_plot_data['pa_err_text'],
                                    pi_units,
                                    gauss_parameters,
                                    cut_x_long, cut_y_lat,
                                    cut_pi,
                                    x_fwhm, y_fwhm,
                                    pass_fail,
                                    w_prob_t,
                                    ax4, light_background=light_background)
                        # ============================================================================= 
                        # Create_FITS added by Ciara Chisholm on December 20th 2025
                        # =============================================================================       
                        
                        out_dir = f'{output_directory}{mosaic_name}'
                        
                        # If the (mosaic specific) directory we want to put this list in doesn't exist yet, make it:
                        Path(out_dir).mkdir(parents=True, exist_ok=True)  
                        
                        out_FITS = out_dir+"/FITS/"
                        Path(out_FITS).mkdir(parents=True, exist_ok=True)  
                        
                        
                        # pair_name =  "Pair_"+ str(1+(source_num//2))
                        # pair_out_fits =out_FITS+pair_name+"/"
                        # Path(pair_out_fits).mkdir(parents=True, exist_ok=True) \
                            
                        src_name = "src_"+ str(1+(source_num//2))
                        src_out_fits =out_FITS+src_name+"/"
                        Path(src_out_fits).mkdir(parents=True, exist_ok=True)  
                        
                        # with fits.open("""/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/VLASS/VLASS_G7_SI.fits""") as VLASSFITS:
                        #     Vheader = VLASSFITS[0].header
                        
                        if create_FITS:
                            
                            headerI = header["I"]
                            PI_map = pi_plot_data['data_fit']
                            SI_map = si_plot_data['data']
                            RM_map = rm_plot_data['rm_data']
                            
                            
                            
                            hdr = fits.Header()
                            img_size = PI_map.shape
                            hdr["SIMPLE"] = "T"
                            hdr["BITPIX"] = headerI["BITPIX"]
                                 
                            # BITPIX = -64 / array data type NAXIS = 2 / number of array dimensions NAXIS1 = 2048NAXIS2 = 2048BLOCKED = T / TAPE MAY BE BLOCKED 
                            hdr['NAXIS'] = 2
                            hdr["NAXIS1"] = img_size[0]
                            hdr["NAXIS2"] = img_size[1]
                            hdr["BLOCKED"]= "T"
                            hdr["DATE"] = datetime.today().strftime('%Y-%m-%d')
                            
                            
                            # Spatial WCS (example: Galactic)
                            hdr['CTYPE1'] = headerI["CTYPE1"]
                            hdr['CTYPE2'] = headerI["CTYPE2"]
                            hdr['CUNIT1'] = 'deg'
                            hdr['CUNIT2'] = 'deg'
                            
                            hdr["CRPIX1"] = 1
                            
                            hdr["CRVAL1"]  = si_plot_data['long_arr'][0][0]
                            
                            hdr["CDELT1"]  = headerI["CDELT1"]
                            hdr["CDELT2"]  = headerI["CDELT2"]
                            
                            old_lat_coor = si_plot_data['lat_arr'][0][0]
                            
                            old_n_ref_pix = 1
                            dis_btw_pixs = headerI["CDELT2"]
                            
                            nw_coord =0 
                            change_in_n_pix = int((nw_coord - old_lat_coor)/(dis_btw_pixs))
                            nw_pix_r_n = change_in_n_pix + old_n_ref_pix
                            
                            
                            hdr["CRVAL2"] = 0
                            hdr["CRPIX2"] = nw_pix_r_n
                            
                            hdr['CTYPE3'] = 'Stokes I'
                            hdr['CUNIT3'] = None
                            
                            hdr['CRPIX3'] = 1
                            hdr['CRVAL3'] = 1
                            hdr['CDELT3'] = 1
                            
                            hdr["POLCODE"]  = "I"
                            if source_num%2 ==0:
                                src_lbl = "a"
                            else:
                                src_lbl = "b"
                                
                            object_name =  "M"+mosaic_name.upper()+"_"+ src_name+src_lbl
                            hdr["FILENAME"] = object_name +"_SI_map"
                            hdr["OBJECT"]  = object_name
                            hdr["ORIGIN"] = "DRAO"
                            hdr["INSTRUME"] = "DRAO SST"
                            hdr['BUNIT'] = 'Jy/beam'
                            # hdr["BUNIT"] = Vheader["BUNIT"]
                            hdr["EQUINOX"]  = 1950
                            hdr["OBSFREQ"] = 1.4209471E+09
                            hdr["BANDW"]  = 3E+7
                            
                            hdr["UVGRID"] = "GAUSSIAN"
                            hdr["BLGRAD"] = "GAUSSIAN TO 20%"
                            hdr["MAXBAS"] = 6E+2
                            hdr["MINBAS"] = 12.8
                            
                            #Writing the SI FITS file
                            hdu = fits.PrimaryHDU(data = SI_map/1000, header=hdr)
                            
                            
                            folder_dir = src_out_fits
                            makedirs(folder_dir[:-1], exist_ok=True)
                            filename = object_name +"_SI_map.fits"
                            hdu.writeto(folder_dir+filename, overwrite=True)
                            
                            #Writing the PI FITS file
                            
                            hdr["CRPIX1"] = 1
                            # hdr["CRPIX2"] = 0
                            # print("length x_long[0]: ", len(x_long[0]))
                            # print("WTF")
                            hdr["CRVAL1"]  = pi_plot_data['x_l_2'][0][0]
                            # hdr["CRVAL2"]  = y_lat[0]
                            hdr["CDELT1"]  = headerI["CDELT1"]
                            hdr["CDELT2"]  = headerI["CDELT2"]
                            
                            old_lat_coor = pi_plot_data['y_b_2'][0][0]
                            # print(old_lat_coor)
                            # sys.exit()
                            old_n_ref_pix = 1
                            dis_btw_pixs = headerI["CDELT2"]
                            
                            nw_coord =0 
                            change_in_n_pix = int((nw_coord - old_lat_coor)/(dis_btw_pixs))
                            nw_pix_r_n = change_in_n_pix + old_n_ref_pix
                            
                            
                            hdr["CRVAL2"] = 0
                            hdr["CRPIX2"] = nw_pix_r_n
                            
                            
                            hdr['CTYPE3'] = 'Linear Polarised Intesity'
                            hdr["FILENAME"] = object_name +"_PI_map"
                            hdr["POLCODE"]  = "PI"
                            hdu = fits.PrimaryHDU(data = PI_map/1000, header=hdr)
                            
                            # folder_dir = mosaic_path[:-8] +"output_data/"+mosaic_name+"/FITS/"
                            # makedirs(folder_dir[:-1], exist_ok=True)
                            filename = object_name +"_PI_map.fits"
                            hdu.writeto(folder_dir+filename, overwrite=True)
                            
                            #Writing the RM FITS file
                            hdr['CTYPE3'] = 'Rotation Measure'
                            hdr['BUNIT'] = 'rad/m^2'
                            hdr["POLCODE"]  = None
                            hdr["FILENAME"] = object_name +"_RM_map"
                            hdu = fits.PrimaryHDU(data = RM_map, header=hdr)
                            
                            # folder_dir = mosaic_path[:-8] +"output_data/"+mosaic_name+"/FITS/"
                            # makedirs(folder_dir[:-1], exist_ok=True)
                            filename = object_name +"_RM_map.fits"
                            hdu.writeto(folder_dir+filename, overwrite=True)
                            
                        # =============================================================================    
                        
                        
                        
                        
                        
                       
                        
                        if get_user_input:
                            # Making the plots show up, the code will keep running until the plt.show() line, at which point execution will pause
                            # In order to continue on to the next source, the plot window will have to be exited manually
                            plt.show(block=False)

          
                        
                        # Flags:
                            
                        print('\n\nFlag key: 0 - Good')
                        print('          1 - Revisit later')
                        print('          2 - Manual fail (+4 for morphology mismatch, +8 for no PI contrast, +32 for gradient)')
                        print('          4 - Gaussian fit failed')
                        print('          8 - Too few/many pixels with sufficient S:N')
                        print('         16 - Fractional polarization too low/high')
                        print('         32 - Failed RM-averaging Chi-square test')
                        print('         64 - Failed average linfit Chi-square test')
                        print('        128 - False detection from neighbour')
                        print('        256 - Mark presence of unidenfified double (please combine with "1" flag))')
                        print('        512 - Go back one source')
                        print('       1024 - Retry source')
                        # The following 3 lines of code were added by Ciara Chisholm on June 25th 2024.
                        print('       2048 - Manual flag, Gradient across the source')
                        print('       4096 - Manual flag, Fitted two peaks to one Gaussian')
                        

                        print(f'\nCurrent flag: {source_flag}')
                        for ind_flag in parse_flag(source_flag):
                            print(f'            - {ind_flag}')

                        mod_flag = ''
                        # mod_flag_valid = False
                        # The following 4 lines were added by Ciara Chisholm om Dec 22 2025 in place of the line above.
                        if get_user_input:
                            mod_flag_valid = False
                            
                        else:
                            mod_flag_valid = True
                            mod_flag = 0
                            # mod_flag_valid = True
                        while not mod_flag_valid:
                            mod_flag = input(f'Set/unset flags? [no change]: ')
                            if mod_flag == '':
                                mod_flag = 0
                                mod_flag_valid = True
                            else:
                                if is_mod_flag_valid(mod_flag, morph=False):
                                    mod_flag_valid = True
                                else:
                                    print('\nInput is invalid, please press return or enter an integer between 0 and 2047.\n')

                        # Alter flag
                        mod_flag = int(mod_flag)
                        # ^ is the python bitwise XOR, it will set any flags that the user wants set and unset any flags that the user wants unset.
                        # e.g. 1 ^ 9 = 8, as the 9 (1 + 8) will unset the 1 flag since it is already set, and set the 8 flag since it isn't set yet.
                        flag[i] = (source_flag ^ mod_flag)

                        if source_flag & 512:  # If the user set the 'Go back one source' flag
                            i -= 1
                            continue  # De-increment to previous source
                        if source_flag & 1024:  # If the user set the 'Retry source' flag
                            continue  # Don't increment to next source, meaning we run through this source again

                        if flag[i] == 0:  # If there are no issues with the source
                            print('\n\n --------------------------------')
                            print('\n Set morphology flag(s):')
                            print(' 0 - Unresolved, isolated source and polarisation matches Stokes I')
                            print(' 1 - (Clearly) Extended source')
                            print(' 2 - Resolved double/multiple')
                            print(' 4 - PI is subset of Stokes I shape')
                            print(' 8 - Additional polarized component(s) seen')
                            print('16 - Offset between PI and I')
                            
                            
                            # mod_morph_flag_valid = False
                            # The following 4 lines were added by Ciara Chisholm om Dec 22 2025 in place of the line above.
                            if get_user_input:
                                mod_morph_flag_valid = False
                                
                            else:
                                mod_morph_flag_valid = True
                                morphology_flag = 0
                                morphology[i] = morphology_flag
                         
                            while not mod_morph_flag_valid:
                                morphology_flag = input('Set flags? [0]: ')
                                if morphology_flag == '':
                                    morphology_flag = 0
                                    morphology[i] = morphology_flag
                                    mod_morph_flag_valid = True
                                else:
                                    if is_mod_flag_valid(morphology_flag, morph=True):
                                        morphology[i] = morphology_flag #added by Ciara Chisholm June 4 2024
                                        mod_morph_flag_valid = True
                                    else:
                                        print('\nInput is invalid, please press return or enter an integer between 0 and 31.\n')
                        # plt.close("all")
                        if get_user_input:
                            print('\nPlease close the plot window to continue to the next source.\n\n')
                        
                        
                        
                        # fig_folder_name = "ext_FITS_fixed"
                        # # Saving figure code:
                        # fig_dir = """/Users/ciarachisholm/Library/CloudStorage/OneDrive-UniversityofCalgary/Ciara's Research Cubby/Figures/RM_plots/"""+fig_folder_name+ "/RMs/M" + mosaic_name.upper()  
                        
                        
                        if save_figs_auto:
                            if fig_path =="":
                                fig_path = out_dir.copy()
                            fig_dir = fig_path + "Figures/" 
                        
                            # fig_dir_pdf = fig_dir +"/PDFs"
                            # fig_dir_svg = fig_dir  +"/SVGs"
                            
                            makedirs(fig_dir, exist_ok=True)
                            # makedirs(fig_dir_pdf, exist_ok=True)
                            # makedirs(fig_dir_svg, exist_ok=True)
                            
                            filename = "src_"+str(i) 
                            
                            if flag[i]==0:
                                passfail = "_passed"
                            else:
                                passfail = "_failed" + str(flag[i])
                            # filename name code below was specifically for double sources
                            # if i%2 ==0:
                            #     twin_number = "1"
                            # else:
                            #     twin_number = str(2)
                            
                            # if flag[i]==0:
                            #     passfail = "Passed"
                            # else:
                            #     passfail = "Failed _" + str(flag[i])
                            # filename = "/Pair_" +str(i//2 +1) +"_Twin" +twin_number+"_" +passfail
                            
                            
                            filename += passfail
                            
                            plt.savefig(fig_dir+ filename +".pdf")#, format="pdf")
                        
                            # plt.savefig(fig_dir_pdf + filename +".pdf")#, format="pdf")
                            # plt.savefig(fig_dir_svg + filename +".svg")#, format="svg")
                            # plt.show()
                        plt.close()
                        
                    else:
                        print('\n\nProblems with this source - IGNORED\n\n')
                        lc[source_num] = pi_data['lmax'][i]
                        bc[source_num] = pi_data['bmax'][i]
                else:
                    print('\n\nProblems with this source - IGNORED\n\n')
                    lc[source_num] = pi_data['lmax'][i]
                    bc[source_num] = pi_data['bmax'][i]
                
                # The following line of code where added by Ciara Chisholm October 11th 2024
                # Creating a list to loop through with containing all the stokes parameters and each band
                bands = ['I', 'Q_A', 'Q_B', 'Q_C', 'Q_D', 'U_A', 'U_B', 'U_C', 'U_D']
            
                # # The following four lines were added by Ciara Chisholm October 22nd 2024    
                # # =============================================================================
                # # Masking source that was just fit 
                # # =============================================================================
                
                
                for C in to_mask_coordinates: # C for coordinate, looping through all the coordinates
                    # Getting the x and y pixel values
                    x_pix,y_pix = C 
                    # Looping through all the bands
                    for b in bands:
                        # Setting the pixels of the source to zero in each band
                        stokes[b][y_pix, x_pix] = 0 
                
                
                
                
                
                i += 1  # Move on to next source

            # **************************************************************************
            # WRITING OUT THE RESULTS:
            # **************************************************************************
            if get_user_input:
                outname = input('Enter output file name [recalc]: ')
                if outname == '':
                    outname = 'recalc'
            else: outname = 'recalc'

            # out_dir = f'{output_directory}{mosaic_name}'

            # # If the (mosaic specific) directory we want to put this list in doesn't exist yet, make it:
            # Path(out_dir).mkdir(parents=True, exist_ok=True)

            file_out = f'_RMlist_{outname}.dat'
            out_path = f'{out_dir}/{mosaic_caps}{file_out}'
            out_path2 = f'{out_dir}/{mosaic_caps}_RMlist_{outname}_ONLY_GOOD.dat'

            # If either of these files already exists, we delete them so we can make them from scratch
            if os.path.exists(out_path):
                os.remove(out_path)
            if os.path.exists(out_path2):
                os.remove(out_path2)
                
                
                
            
            with open(out_path, 'w') as write_file, open(out_path2, 'w') as write_file2:
                # Starting to write the first line of  header
                write_file.write("  {0:10}{1:7}{2:9}{3:9}".format("l","b","WRM", "WdRM",))
                write_file.write("{0:10}{1:8}{2:9}{3:7}".format( "dPav", "prob", "Chi^2", "PI"))
                write_file.write("{0:10}{1:6}{2:7}{3:6}".format("SI", "M","S:N", "#pix", ))
                write_file.write("{0:5}{1:8}{2:8}{3:10}{4}\n".format( "P/F", "Morph", "mChi2", "RM_peak", "dRM_peak"))
                # Finished writing the first line of the header
                # Writing the second line, this line shows the units of the columns
                write_file.write("{0:18}{1:19}{2:27}{3:59}{4}".format("-- degrees --", "-- rad/m^2 --", "deg", pi_units, "-- rad/m^2 --"))

                # Writing the same header for the good only files
                
                write_file2.write("  {0:10}{1:7}{2:9}{3:9}".format("l","b","WRM", "WdRM",))
                write_file2.write("{0:10}{1:8}{2:9}{3:7}".format( "dPav", "prob", "Chi^2", "PI"))
                write_file2.write("{0:10}{1:6}{2:7}{3:6}".format("SI", "M","S:N", "#pix", ))
                write_file2.write("{0:5}{1:8}{2:8}{3:10}{4}\n".format( "P/F", "Morph", "mChi2", "RM_peak", "dRM_peak"))
                # Writing the second line, this line shows the units of the columns
                write_file2.write("{0:18}{1:19}{2:27}{3:59}{4}".format("-- degrees --", "-- rad/m^2 --", "deg", pi_units, "-- rad/m^2 --"))

                for i in range(num_sources):
                    
                    # The following code was added by Ciara Chisholm May 29th 2024
                    write_file.write(f'\n{lc[i]:7.3f}'
                                     f'{bc[i]:8.3f}'
                                     f'{wrmc[i]:9.2f}'
                                     f'{wdrmc[i]:9.2f}'
                                     f'{dpaav[i]:8.2f}'
                                     f'{probchi[i]:12.4f}'
                                     f'{chi2c[i]:6.2f}'
                                     f'{pic[i]:8.2f}'
                                     f'{sic[i]:8.2f}'
                                     f'{mc[i]:9.3f}'
                                     f'{snc[i]:7.2f}'
                                     f'{npixels[i]:6.0f}'
                                     f'{flag[i]:5.0f}'
                                     f'{morphology[i]:6.0f}'
                                     f'{fracpol_chi2_src[i]:10.3f}'
                                     f'{rm_peakpix[i]:10.3f}'
                                     f'{drm_peakpix[i]:9.3f}')
                    
                    
                    if flag[i] == 0 and fracpol_chi2_src[i] <= 5:
                        # The following code was added by Ciara
                        write_file2.write(f'\n{lc[i]:7.3f}'
                                         f'{bc[i]:8.3f}'
                                         f'{wrmc[i]:9.2f}'
                                         f'{wdrmc[i]:9.2f}'
                                         f'{dpaav[i]:8.2f}'
                                         f'{probchi[i]:12.4f}'
                                         f'{chi2c[i]:6.2f}'
                                         f'{pic[i]:8.2f}'
                                         f'{sic[i]:8.2f}'
                                         f'{mc[i]:9.3f}'
                                         f'{snc[i]:7.2f}'
                                         f'{npixels[i]:6.0f}'
                                         f'{flag[i]:5.0f}'
                                         f'{morphology[i]:6.0f}'
                                         f'{fracpol_chi2_src[i]:10.3f}'
                                         f'{rm_peakpix[i]:10.3f}'
                                         f'{drm_peakpix[i]:9.3f}')
            
           
            # Output a list of sources that need to be revisited:
            revisit_list = np.bitwise_and(flag.astype(int), 1) == 1
            if np.sum(revisit_list) > 0:
                lmax = pi_data['lmax'][revisit_list]
                bmax = pi_data['bmax'][revisit_list]
                xpixmax = pi_data['xpixmax'][revisit_list]
                ypixmax = pi_data['ypixmax'][revisit_list]
                pimax = pi_data['pimax'][revisit_list]
                simax = pi_data['simax'][revisit_list]
                stonmax = pi_data['stonmax'][revisit_list]

                revisit_name = input('Enter output file name [revisit]: ')
                if revisit_name == '':
                    revisit_name = 'revisit'

                revisit_file_out = f'_sourcelist_{revisit_name}.dat'
                revisit_out_path = f'{out_dir}/{mosaic_caps}{revisit_file_out}'

                # If this file already exists, we delete it so we can make it from scratch
                if os.path.exists(revisit_out_path):
                    os.remove(revisit_out_path)

                with open(revisit_out_path, 'w') as write_file:
                    write_file.write(f'Polarised Intensity source list for {mosaic_caps}')
                    write_file.write(f'\n')
                    write_file.write(f'\n   l       b     xpix  ypix       PI       SI      S/N')
                    write_file.write(f'\n-- degrees --                     --{pi_units}--')
                    write_file.write(f'\n')

                    max_long = np.max(lmax)
                    while max_long > 0.0:
                        mask = mf.idl_where(lmax == max_long)
                        # Added by Ciara Chisholm May 29th 2024
                        write_file.write(f'\n{lmax.flatten()[mask[0]]:7.3f}'
                                          f'{bmax.flatten()[mask[0]]:7.3f}'
                                          f'{xpixmax.flatten()[mask[0]]:7.0f}'
                                          f'{ypixmax.flatten()[mask[0]]:6.0f}'
                                          f'{pimax.flatten()[mask[0]]:10.2f}'
                                          f'{simax.flatten()[mask[0]]:11.2f}'
                                          f'{stonmax.flatten()[mask[0]]:7.2f}')
                        # write_file.write(f'\n{mf.string_normalise(str(round(lmax.flatten()[mask[0]], 3)), 7)}'
                        #                   f' {mf.string_normalise(str(round(bmax.flatten()[mask[0]], 3)), 6, negatives=True)}'
                        #                   f'    {mf.string_normalise(str(mf.nround(xpixmax.flatten()[mask[0]])), 4)}'
                        #                   f' {mf.string_normalise(str(mf.nround(ypixmax.flatten()[mask[0]])), 4)}'
                        #                   f'     {mf.string_normalise(str(round(pimax.flatten()[mask[0]], 2)), 6)}'
                        #                   f'    {mf.string_normalise(str(round(simax.flatten()[mask[0]], 2)), 6)}'
                        #                   f' {mf.string_normalise(str(round(stonmax.flatten()[mask[0]], 2)), 5)}')
                        where_current = np.logical_and(lmax == lmax.flatten()[mask[0]], bmax == bmax.flatten()[mask[0]])
                        lmax[where_current] = 0.0
                        max_long = np.max(lmax)

            print('\nRotation Measure analysis complete!')

