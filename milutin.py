#!/software/anaconda3/bin

""" 

Moist Energy Balance Model with a seasonal cycle of insolation and a
thermodynamic sea ice model, as described in the paper:

Feldl, N., and T. M. Merlis (2021), Polar amplification in idealized 
climates: the role of ice, moisture, and seasons

This code is based off the Dry EBM presented in the paper:

Wagner, T.J. and I. Eisenman, 2015: How Climate Model Complexity 
Influences Sea Ice Stability. J. Climate, 28, 3998–4014, 
https://doi.org/10.1175/JCLI-D-14-00654.1

with the following modifications:
- We have added the effect of latent energy on the diffusive 
  representation of atmospheric energy transport
- We have added functionality for disabling the effect of sea-ice 
  thermodynamics, the seasonal cycle of insolation, ice-albedo
  feedback, and latent energy transport.
- We use a global, rather than single-hemisphere, domain.

"""

import sys
import xarray as xr
from pandas import DataFrame as df
import pandas as pd
import numpy as np
from numpy import sqrt, deg2rad, rad2deg, sin, cos, tan, arcsin, arccos, pi
import matplotlib.pyplot as plt
import time
import scipy.io as sio
import scipy.interpolate as interp
from scipy import integrate
import mixedlayer
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from scipy.io import netcdf_file
from tabulate import tabulate
import pickle
from scipy.signal import argrelextrema
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib as mpl



start_time = time.time()
OrbitalTable = pd.read_csv('Laskar04_OrbitalTable.csv')
mpl.rcParams['axes.titlesize'] = 10 # reset global fig properties
mpl.rcParams['legend.fontsize'] = 6 # reset global fig properties
mpl.rcParams['legend.title_fontsize'] = 6 # reset global fig properties
mpl.rcParams['xtick.labelsize'] = 8 # reset global fig properties
mpl.rcParams['ytick.labelsize'] = 8 # reset global fig properties
mpl.rcParams['ytick.right'] = True # reset global fig properties
mpl.rcParams['axes.labelsize'] = 8 # reset global fig properties


def age(kyear = 0):

  params = Orbital_Insolation(kyear+1, kyear).get_orbit()
  float_params = tuple([float(i) for i in params])
  kyear ,ecc ,long_peri ,obliquity ,precession = float_params

  print('at {} kyr BP: \n eccentricity = {} \n obliquity = {} \n longitude of perihelion = {}'.format(abs(kyear), ecc, obliquity, long_peri))

  output = (ecc, obliquity, long_peri)

  return output

def wave_peaks(wave, kyear_range, maxima = 'both'):

  from_kyear, to_kyear = kyear_range
  params = Orbital_Insolation(from_kyear+1, to_kyear).get_orbit()
  kyear ,ecc ,long_peri ,obliquity ,precession = params

  if wave == 'obliquity':
    wave = obliquity
  elif wave == 'eccentricity':
    wave = ecc
  elif wave == 'long_peri':
    wave = long_peri
  elif wave == 'precession':
    wave = precession

  maxima_loc = argrelextrema(wave, np.greater)
  minima_loc = argrelextrema(wave, np.less)

  maxima_kyear = []
  minima_kyear = []
  [maxima_kyear.append(abs(kyear[i])) for i in maxima_loc]
  [minima_kyear.append(abs(kyear[i])) for i in minima_loc]
  maxima_kyear = maxima_kyear[0]
  minima_kyear = minima_kyear[0]

  if maxima == 'min':
    return minima_kyear
  elif maxima == 'max':
     return maxima_kyear
  else:
    return maxima_kyear, minima_kyear 

def obliquity(kyear = None):

  if kyear == None:

    params = Orbital_Insolation(1,0).get_orbit()
    float_params = tuple([float(i) for i in params])
    kyear ,ecc ,long_peri ,obliquity ,precession = float_params

    output = obliquity

    return output

  elif isinstance(kyear, int):

    params = Orbital_Insolation(kyear+1, kyear).get_orbit()
    float_params = tuple([float(i) for i in params])
    kyear ,ecc ,long_peri ,obliquity ,precession = float_params

    output = obliquity

    return output

  elif len(tuple(kyear)) == 2:

    from_kyear, to_kyear = kyear

    params = Orbital_Insolation(from_kyear+1, to_kyear).get_orbit()
    kyear ,ecc ,long_peri ,obliquity ,precession = params

    output = obliquity

    return output

  elif isinstance(kyear, str):

    if kyear == 'maximum':
      output = Helper_Functions().orbit_extrema('obl','max')

    elif kyear == 'minimum':
      output = Helper_Functions().orbit_extrema('obl','min')

    else:

      raise ValueError("invalid string input, try 'minimum' or 'maximum'")

    return output

def eccentricity(kyear = None):

  if kyear == None:

    params = Orbital_Insolation(1,0).get_orbit()
    float_params = tuple([float(i) for i in params])
    kyear ,ecc ,long_peri ,obliquity ,precession = float_params

    output = ecc

    return output

  elif isinstance(kyear, int):

    params = Orbital_Insolation(kyear+1, kyear).get_orbit()
    float_params = tuple([float(i) for i in params])
    kyear ,ecc ,long_peri ,obliquity ,precession = float_params

    output = ecc

    return output

  elif len(tuple(kyear)) == 2:

    from_kyear, to_kyear = kyear

    params = Orbital_Insolation(from_kyear+1, to_kyear).get_orbit()
    kyear ,ecc ,long_peri ,obliquity ,precession = params

    output = ecc

    return output

  elif isinstance(kyear, str):

    if kyear == 'maximum':
      output = Helper_Functions().orbit_extrema('ecc','max')

    elif kyear == 'minimum':
      output = Helper_Functions().orbit_extrema('ecc','min')

    else:

      raise ValueError("invalid string input, try 'minimum' or 'maximum'")

    return output

def long_peri(kyear = None):

  if kyear == None:

    params = Orbital_Insolation(1,0).get_orbit()
    float_params = tuple([float(i) for i in params])
    kyear ,ecc ,long_peri ,obliquity ,precession = float_params

    output = long_peri

    return output

  elif isinstance(kyear, int):

    params = Orbital_Insolation(kyear+1, kyear).get_orbit()
    float_params = tuple([float(i) for i in params])
    kyear ,ecc ,long_peri ,obliquity ,precession = float_params

    output = long_peri

    return output

  elif len(tuple(kyear)) == 2:

    from_kyear, to_kyear = kyear

    params = Orbital_Insolation(from_kyear+1, to_kyear).get_orbit()
    kyear ,ecc ,long_peri ,obliquity ,precession = params

    output = long_peri

    return output

  elif isinstance(kyear, str):

    if kyear == 'maximum':
      output = Helper_Functions().orbit_extrema('long','max')

    elif kyear == 'minimum':
      output = Helper_Functions().orbit_extrema('long','min')

    else:

      raise ValueError("invalid string input, try 'minimum' or 'maximum'")

    return output

def precession(kyear = None):

  if kyear == None:

    params = Orbital_Insolation(1,0).get_orbit()
    float_params = tuple([float(i) for i in params])
    kyear ,ecc ,long_peri ,obliquity ,precession = float_params

    output = precession

    return output

  elif isinstance(kyear, int):

    params = Orbital_Insolation(kyear+1, kyear).get_orbit()
    float_params = tuple([float(i) for i in params])
    kyear ,ecc ,long_peri ,obliquity ,precession = float_params

    output = precession

    return output

  elif len(tuple(kyear)) == 2:

    from_kyear, to_kyear = kyear

    params = Orbital_Insolation(from_kyear+1, to_kyear).get_orbit()
    kyear ,ecc ,long_peri ,obliquity ,precession = params

    output = precession

    return output

  elif isinstance(kyear, str):

    if kyear == 'maximum':
      output = Helper_Functions().orbit_extrema('prec','max')

    elif kyear == 'minimum':
      output = Helper_Functions().orbit_extrema('prec','min')

    else:

      raise ValueError("invalid string input, try 'minimum' or 'maximum'")

    return output

def insolation(kyear = None, latitude = None, output_type = 'array', show_plot = True, season = None, days = None, filename = 'orbkit_plot', solve_energy = None):

  if latitude is not None:
    if np.max(np.abs(latitude)) > 90:
        raise ValueError('latitude value is out of degree range (-90,90)')  
  if isinstance(latitude, list) and len(latitude) == 1:
    raise TypeError('for a single latitude value please use type int or tuple')
  if isinstance(latitude,tuple) and len(latitude)!= 2:
    raise ValueError('laitude of type(tuple) must not exceed a length of 2')

  if season == None:
    if days == None:
      day_vals = None
      day_ax = np.linspace(0,Orbital_Insolation().days_per_year_const ,365)

    elif days != None:
      if isinstance(days, tuple) and len(days) == 2 and days[0] != days[1]:
        day_vals = np.linspace(days[0],days[1], abs(days[0]-days[1])+1)
        day_ax = day_vals
      else:
        raise ValueError('day input needs to be type tuple with length 2 and elements cannot be equal')
  elif season == 'DJF':
    days_end_of_year = np.linspace(experiment().month['December'], 365, abs(experiment().month['December']-365)+1)
    days_start_of_year = np.linspace(experiment().month['January'],experiment().month['March'], abs(experiment().month['January']-experiment().month['March'])+1)
    day_vals = np.concatenate((days_end_of_year, days_start_of_year))
    day_ax = np.linspace(-32,59,np.abs(-32-59)+1)
  elif season == 'MAM':
    day_vals = np.linspace(experiment().month['March'],experiment().month['June'], abs(experiment().month['March']-experiment().month['June'])+1)
    day_ax = day_vals
  elif season == 'JJA':
    day_vals = np.linspace(experiment().month['June'],experiment().month['September'], abs(experiment().month['June']-experiment().month['September'])+1)
    day_ax = day_vals
  elif season == 'SON':
    day_vals = np.linspace(experiment().month['September'],experiment().month['December'], abs(experiment().month['September']-experiment().month['December'])+1)
    day_ax = day_vals
  
  if latitude is None:
    if kyear is None:
      if output_type == 'array':       
        output =  Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'integer', days = day_vals).T
        if show_plot == True:
          lat_ax = np.rad2deg(np.arcsin(experiment(3).config['x']))
          contour_1 = plt.contourf(day_ax,lat_ax,output,np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
          plt.xlabel('Time (days)')
          plt.ylabel('Latitude (degrees)')
          plt.title('Modern Insolation')
          plt.colorbar(contour_1, label = 'Insolation (W/m²)')
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'integer', days = day_vals).T, axis = 0)
        if show_plot == True:
          fig, axs = plt.subplots()
          axs.plot(day_ax,output, label = 'space average')
          plt.xlabel('Time (Days)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Modern Insolation')
          axs.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'integer', days = day_vals).T, axis = 1)
        if show_plot == True:
          lat_ax = np.rad2deg(np.arcsin(experiment(3).config['x'])) 
          fig, axs = plt.subplots()
          axs.plot(lat_ax,output, label = 'time average')
          plt.xlabel('Latitude (degrees)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Modern Insolation')
          plt.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'global annual mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'integer', days = day_vals).T)
        if show_plot == True:
          print('No associated plot for output')
    elif isinstance(kyear, int) or isinstance(kyear, float):
      if output_type == 'array':
        output = Orbital_Insolation(kyear+1, kyear).avg_insolation(experiment(grid_num = 3).config, lat_array = 'integer', days = day_vals).T
        if show_plot == True:
          lat_ax = np.rad2deg(np.arcsin(experiment(3).config['x']))
          contour_1 = plt.contourf(day_ax,lat_ax,output,np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
          plt.xlabel('Time (days)')
          plt.ylabel('Latitude (degrees)')
          plt.title('Insolation at {:.1f}kyr'.format(kyear))
          plt.colorbar(contour_1, label = 'Insolation (W/m²)')
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude mean':
        output = np.mean(Orbital_Insolation(kyear+1, kyear).avg_insolation(experiment(grid_num = 3).config, lat_array = 'integer', days = day_vals).T, axis = 0)
        if show_plot == True:
          fig, axs = plt.subplots()
          axs.plot(day_ax,output, label = 'space average')
          plt.xlabel('Time (Days)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation at {:.1f}kyr'.format(kyear))
          axs.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
        output = np.mean(Orbital_Insolation(kyear+1, kyear).avg_insolation(experiment(grid_num = 3).config, lat_array = 'integer', days = day_vals).T, axis = 1)
        if show_plot == True:
          lat_ax = np.rad2deg(np.arcsin(experiment(3).config['x'])) 
          fig, axs = plt.subplots()
          axs.plot(lat_ax,output, label = 'time average')
          plt.xlabel('Latitude (degrees)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation at {:.1f}kyr'.format(kyear))
          plt.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'global annual mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'integer', days = day_vals).T)
        if show_plot == True:
          print('No associated plot for output')   
    elif isinstance(kyear, tuple) and len(kyear) == 3:
      eccentricity, obliquity, long_peri = kyear
      if output_type == 'array':
        output = Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'integer', obl = obliquity, long = long_peri, ecc = eccentricity, kyear = '', days = day_vals).T
        if show_plot == True:
          lat_ax = np.rad2deg(np.arcsin(experiment(3).config['x']))
          contour_1 = plt.contourf(day_ax,lat_ax,output,np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
          plt.xlabel('Time (days)')
          plt.ylabel('Latitude (degrees)')
          plt.title('eccentricity: {:.2f}, obliquity: {:.2f} and longitude of perihelion: {:.2f}'.format(eccentricity,obliquity,long_peri))
          plt.colorbar(contour_1, label = 'Insolation (W/m²)')
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'integer', obl = obliquity, long = long_peri, ecc = eccentricity, kyear = '', days = day_vals).T, axis = 0)
        if show_plot == True:
          fig, axs = plt.subplots()
          axs.plot(day_ax,output, label = 'space average')
          plt.xlabel('Time (Days)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation with eccentricity: {:.2f}, obliquity: {:.2f} and longitude of perihelion: {:.2f}'.format(eccentricity,obliquity,long_peri))
          axs.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'integer', obl = obliquity, long = long_peri, ecc = eccentricity, kyear = '', days = day_vals).T, axis = 1)
        if show_plot == True:
          lat_ax = np.rad2deg(np.arcsin(experiment(3).config['x'])) 
          fig, axs = plt.subplots()
          axs.plot(lat_ax,output, label = 'time average')
          plt.xlabel('Latitude (degrees)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation with eccentricity: {:.2f}, obliquity: {:.2f} and longitude of perihelion: {:.2f}'.format(eccentricity,obliquity,long_peri))
          plt.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'global annual mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'integer', days = day_vals).T)
        if show_plot == True:
          print('No associated plot for output')
    elif isinstance(kyear, tuple) and len(kyear) == 2:    
      kyear_step = int(abs(kyear[1] - kyear[0]))
      kyear_range = np.linspace(kyear[0],kyear[1], kyear_step+1, dtype = int)
      lat_range = np.rad2deg(np.arcsin(experiment(grid_num = 3).config['x']))
      kyrs_inso = []
      [kyrs_inso.append(Orbital_Insolation(i+1, i).avg_insolation(experiment(grid_num = 3).config, lat_array = 'integer', days = day_vals).T) for i in kyear_range]
      output = np.array(kyrs_inso)
      output = xr.DataArray(data=output,dims=['kyr','lat','day'],coords={'kyr':kyear_range,'lat':lat_range, 'day':day_ax})
      if output_type == 'array':
          pass
          if show_plot == True:
            reshaped_output = []
            for i in range(np.shape(output)[1]):
              row = []
              for f in range(np.shape(output)[0]):
                row.append(output[f,i,:])
              row = np.hstack(row)
              reshaped_output.append(row)
            kyear_ax = np.linspace(0,len(output),np.shape(reshaped_output)[1])
            lat_ax = np.rad2deg(np.arcsin(experiment(3).config['x']))
            contour_1 = plt.contourf(kyear_ax,lat_ax,reshaped_output,np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
            plt.xlabel('Time (kyears)')
            plt.ylabel('Latitude (degrees)')
            plt.title('Insolation rangeing from {:.2f}kyr to {:.2f}kyr'.format(kyear[0],kyear[1]))
            plt.colorbar(contour_1, label = 'Insolation (W/m²)')
            plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude mean':
          output = np.mean(output,axis=1)
          if show_plot == True:

            insolation_v_time = np.hstack(output)
            kyear_ax = np.linspace(kyear[0],kyear[1],len(insolation_v_time))
            plt.plot(kyear_ax,insolation_v_time)
            plt.xlabel('Time (kyears)')
            plt.ylabel('TOA Insolation (W/m²)')
            plt.title('Insolation rangeing from {:.2f}kyr to {:.2f}kyr'.format(kyear[0],kyear[1]))
            plt.legend()
            plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
          output = np.mean(output,axis=2)
          if show_plot == True:
            fig,axs = plt.subplots()
            insolation_v_lat = np.hstack(output)
            kyear_ax = np.linspace(kyear[0],kyear[1],len(insolation_v_lat))
            x = []
            [x.append(np.linspace(-np.pi/2,np.pi/2,np.shape(output)[1])) for i in range(kyear_step)]
            x = np.rad2deg(np.hstack(x))
            axs.plot(kyear_ax,insolation_v_lat)
            axs.set_xlabel('Time (kyears)')
            axs.set_ylabel('TOA Insolation (W/m²)')
            plt.title('Insolation rangeing from {:.2f}kyr to {:.2f}kyr'.format(kyear[0],kyear[1]))
            axs.legend()
            plt.savefig('{}.png'.format(filename))   
      elif output_type == 'kyear mean':
        output = np.mean(output,axis=0)
        if show_plot == True:
          lat_ax = np.rad2deg(np.arcsin(experiment(3).config['x']))
          contour_1 = plt.contourf(day_ax,lat_ax,output,np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
          plt.xlabel('Time (days)')
          plt.ylabel('Latitude (degrees)')
          plt.title('Mean Insolation from {:.1f}kyr to {:.1f}kyr'.format(kyear[0],kyear[1]))
          plt.colorbar(contour_1, label = 'Insolation (W/m²)')
          plt.savefig('{}.png'.format(filename))
      elif output_type == 'global annual mean':
          output = np.mean(output, axis=(1,2))
          if show_plot == True:
            kyear_ax = np.linspace(kyear[0],kyear[1],len(output))
            plt.plot(kyear_ax,output)
            plt.xlabel('Time (kyears)')
            plt.ylabel('TOA Insolation (W/m²)')
            plt.title('Insolation rangeing from {:.2f}kyr to {:.2f}kyr'.format(kyear[0],kyear[1]))
            plt.savefig('{}.png'.format(filename))   
    elif isinstance(kyear, list) or isinstance(kyear, np.ndarray):
      array_at_kyears = []
      [array_at_kyears.append(Orbital_Insolation(i+1,i).avg_insolation(experiment(grid_num = 3).config, lat_array = 'integer',days = day_vals).T) for i in kyear]
      kyear_step = int(abs(kyear[1] - kyear[0]))
      lat_range = np.rad2deg(np.arcsin(experiment(grid_num = 3).config['x']))
      output = np.array(array_at_kyears)
      output = xr.DataArray(data=output,dims=['kyr','lat','day'],coords={'kyr':kyear,'lat':lat_range,'day':day_ax})
      if output_type == 'array':
        pass
        if show_plot == True:
          lat_ax = np.rad2deg(np.arcsin(experiment(3).config['x']))
          plot_rows, plot_cols = Helper_Functions.find_plot_dims(kyear)           
          fig, axs = plt.subplots(nrows=plot_rows,ncols=plot_cols)    
          count = 0
          if np.ndim(axs) == 2:
            for i in range(plot_rows):
              for f in range(plot_cols):
                if count <= len(kyear)-1:
                  im = axs[i,f].contourf(day_ax,lat_ax,output[count],np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
                  axs[i,f].set_title('kyear {}'.format(kyear[count]))
                  axs[i,f].set_xlabel('Time (days)')
                  axs[i,f].set_ylabel('Latitude (degrees)')
                  count += 1
                else:
                  axs[i,f].remove()
          elif np.ndim(axs) == 1:
            for i in range(plot_rows):
              for f in range(plot_cols):
                if count <= len(kyear)-1:
                  im = axs[f].contourf(day_ax,lat_ax,output[count],np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
                  axs[f].set_title('kyear {}'.format(kyear[count]))
                  axs[f].set_xlabel('Time (days)')
                  axs[f].set_ylabel('Latitude (degrees)')
                  count += 1
                else:
                  axs[f].remove()  
          cb_ax = fig.add_axes([.85,.124,.04,.754])
          fig.colorbar(im,label = 'Insolation (W/m²)',cax = cb_ax, anchor = (10,1))
          plt.tight_layout(rect=[0,0,.8,1])
          plt.savefig('{}.png'.format(filename))      
      elif output_type == 'latitude mean':
        output = np.mean(output, axis = 1)
        if show_plot == True:

          lat_ax = np.rad2deg(np.arcsin(experiment(3).config['x']))
          
                       
          fig, axs = plt.subplots()

          [plt.plot(day_ax,output[i], label = '{:.1f}kyear'.format(kyear[i])) for i in range(len(kyear))]
          
          plt.xlabel('Time (days)')
          plt.ylabel('Insolation (W/m²)')
          plt.title('Latitude Mean Insolation for Kyears')
          plt.legend()
          plt.tight_layout()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
        output = np.mean(output, axis = 2)
        if show_plot == True:

          lat_ax = np.rad2deg(np.arcsin(experiment(3).config['x']))
                       
          fig, axs = plt.subplots()

          [plt.plot(lat_ax,output[i], label = '{:.1f}kyear'.format(kyear[i])) for i in range(len(kyear))]
          
          plt.xlabel('Latitude (degrees)')
          plt.ylabel('Insolation (W/m²)')
          plt.title('Day Mean Insolation for Kyears')
          plt.legend()
          
          plt.tight_layout()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'kyear mean':
        output = np.mean(output, axis = 0)
        if show_plot == True:
          
          lat_ax = np.rad2deg(np.arcsin(experiment(3).config['x']))
          fig, axs = plt.subplots()
          im = axs.contourf(day_ax,lat_ax,output,np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
          kyear = [float(i) for i in kyear]
          axs.set_title('Mean Insolation for Kyears {}'.format(kyear))
          axs.set_xlabel('Time (days)')
          axs.set_ylabel('Latitude (degrees)')
          plt.colorbar(im, label = 'Insolation (W/m²)')

          plt.tight_layout()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'global annual mean':
        output = np.mean(output, axis = (1,2))
        if show_plot == True:
          kyear_ax = []
          [kyear_ax.append(str(i)) for i in kyear]
          plt.bar(kyear_ax,output)
          plt.ylim(np.min(output)-0.5,np.max(output)+0.5)
          plt.xlabel('kyear')
          plt.ylabel('Global Annual Mean Insolation (W/m²)')
          plt.tight_layout()
          plt.savefig('{}.png'.format(filename))       
  elif isinstance(latitude, int) or isinstance(latitude, float):    
    if kyear is None:
      if output_type == 'array' or output_type == 'latitude mean':
        output = Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=latitude, days = day_vals).T
        if show_plot == True:
          fig, axs = plt.subplots()
          axs.plot(day_ax,output, label = '{} degree'.format(latitude))
          plt.xlabel('Time (Days)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Modern Insolation at {} degrees'.format(latitude))
          axs.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=latitude, days = day_vals).T)
        if show_plot == True:
          print('No associated plot for output')
    elif isinstance(kyear, int) or isinstance(kyear, float):       
      if output_type == 'array' or output_type == 'latitude mean':
        output = Orbital_Insolation(kyear+1, kyear).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=latitude, days = day_vals).T
        if show_plot == True:
          fig, axs = plt.subplots()
          axs.plot(day_ax,output, label = '{} degree'.format(latitude))
          plt.xlabel('Time (Days)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Kyear {} Insolation at {} degrees'.format(kyear,latitude))
          axs.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
        output = np.mean(Orbital_Insolation(kyear+1, kyear).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=latitude, days = day_vals).T)
        if show_plot == True:
          print('No associated plot for output')
    elif isinstance(kyear, tuple) and len(kyear) == 3:
      eccentricity, obliquity, long_peri = kyear
      if output_type == 'array' or output_type == 'latitude mean':
        output = Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', obl = obliquity, long = long_peri, ecc = eccentricity, kyear = '', lat=latitude, days = day_vals).T
        if show_plot == True:
          fig, axs = plt.subplots()
          axs.plot(day_ax,output, label = '{} degree'.format(latitude))
          plt.xlabel('Time (Days)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation at {} degrees with \n eccentricity: {:.2f}, obliquity: {:.2f} and longitude of perihelion: {:.2f}'.format(latitude,eccentricity,obliquity,long_peri))
          axs.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', obl = obliquity, long = long_peri, ecc = eccentricity, kyear = '', lat=latitude, days = day_vals).T)
        if show_plot == True:
          print('No associated plot for output')
    elif isinstance(kyear, tuple) and len(kyear) == 2:
      kyear_step = int(abs(kyear[1] - kyear[0]))
      kyear_range = np.linspace(kyear[0],kyear[1], kyear_step+1, dtype = int)
      kyrs_inso = []
      [kyrs_inso.append(Orbital_Insolation(i+1, i).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=latitude, days = day_vals).T) for i in kyear_range]
      output = np.array(kyrs_inso)
      output = xr.DataArray(data=output,dims=['kyr','day'],coords={'kyr':kyear_range,'day':day_ax})
      if output_type == 'array' or output_type == 'latitude mean':
        pass
        if show_plot == True:
          insolation_v_time = np.hstack(output)
          if kyear[0] == kyear[1]:
            raise ValueError('cannot have kyear range be 0')
          else:
            kyear_ax = np.linspace(kyear[0],kyear[1],len(insolation_v_time))
            plt.plot(kyear_ax,insolation_v_time, label = '{} degree'.format(latitude))
            plt.xlabel('Time (kyears)')
            plt.ylabel('TOA Insolation (W/m²)')
            plt.title('Insolation at {} degrees for Kyear range {}'.format(latitude,kyear))
            plt.legend()
            plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
        output = np.mean(output, axis = 1)
        if show_plot == True:
          kyear_ax = np.linspace(kyear[0],kyear[1],len(output))
          plt.plot(kyear_ax,output, label = '{} degree'.format(latitude))
          plt.xlabel('Time (kyears)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Day mean insolation at {} degrees for Kyear range {}'.format(latitude, kyear))
          plt.legend()
          plt.savefig('{}.png'.format(filename))     
      elif output_type == 'kyear mean':
        output = np.mean(output, axis = 0)
        if show_plot == True:
          plt.plot(day_ax,output, label = '{} degree'.format(latitude))
          plt.xlabel('Time (days)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation at {} degrees for Kyear mean over range {}'.format(latitude, kyear))
          plt.legend()
          plt.savefig('{}.png'.format(filename))  
    elif isinstance(kyear,list) or isinstance(kyear,np.ndarray):
      array_at_kyears = []
      [array_at_kyears.append(Orbital_Insolation(i+1,i).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat',lat = latitude,days = day_vals).T) for i in kyear]
      output = np.array(array_at_kyears)
      output = xr.DataArray(data=output,dims=['kyr','day'],coords={'kyr':kyear,'day':day_ax})
      if output_type == 'array' or output_type == 'latitude mean':
        pass
        if show_plot == True:
          fig, axs = plt.subplots()
          [plt.plot(day_ax,output[i], label = '{:.1f}kyear'.format(kyear[i])) for i in range(len(kyear))]
          plt.xlabel('Time (days)')
          plt.ylabel('Insolation (W/m²)')
          plt.title('Insolation at {} degrees for Kyears'.format(latitude))
          plt.legend()
          plt.tight_layout()
          plt.savefig('{}.png'.format(filename))    
      elif output_type == 'day mean':
        output = np.mean(output, axis =1)
        if show_plot == True:
          kyear_ax = []
          [kyear_ax.append(str(i)) for i in kyear]
          plt.bar(kyear_ax,output)
          plt.ylim(np.min(output)-0.5,np.max(output)+0.5)
          plt.xlabel('kyear')
          plt.ylabel('Day Mean Insolation (W/m²)')
          plt.title('Day Mean Insoaltion at {} degrees'.format(latitude))
          plt.tight_layout()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'kyear mean':
        output = np.mean(output, axis = 0)
        if show_plot == True:
          fig, axs = plt.subplots()
          axs.plot(day_ax,output, label = '{} degrees'.format(latitude))
          plt.xlabel('Time (Days)')
          plt.ylabel('TOA Insolation (W/m²)')
          kyear = [float(i) for i in kyear]
          plt.title('Insolation at {} for mean of {}kyears'.format(latitude,kyear))
          axs.legend()
          plt.savefig('{}.png'.format(filename))   
  elif isinstance(latitude, tuple) and len(latitude) == 2:   
    if kyear is None:
      if output_type == 'array':
        output = Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=latitude, days = day_vals).T
        if show_plot == True:
          lat_ax = np.linspace(latitude[0], latitude[1], len(output))
          contour_1 = plt.contourf(day_ax,lat_ax,output,np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
          plt.xlabel('Time (days)')
          plt.ylabel('Latitude (degrees)')
          plt.title('Modern Insolation')
          plt.colorbar(contour_1, label = 'Insolation (W/m²)')
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=latitude, days = day_vals).T, axis = 0)
        if show_plot == True:
          fig, axs = plt.subplots()
          axs.plot(day_ax,output, label = 'space average')
          plt.xlabel('Time (Days)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Modern Insolation latidue mean over range {}'.format(latitude))
          axs.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=latitude, days = day_vals).T, axis = 1)
        if show_plot == True:
          lat_ax = np.linspace(latitude[0], latitude[1], (abs(latitude[0]-latitude[1]))+1)
          fig, axs = plt.subplots()
          axs.plot(lat_ax,output, label = 'day average')
          plt.xlabel('Latitude (degrees)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Modern Insolation for day mean')
          plt.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude day mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=latitude, days = day_vals).T, axis = (0,1))
        if show_plot == True:
          print('No associated plot for output')   
    elif isinstance(kyear, int) or isinstance(kyear, float):
      if output_type == 'array':
        output = Orbital_Insolation(kyear+1, kyear).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=latitude, days = day_vals).T
        if show_plot == True:
          lat_ax = np.linspace(latitude[0], latitude[1], len(output))
          contour_1 = plt.contourf(day_ax,lat_ax,output,np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
          plt.xlabel('Time (days)')
          plt.ylabel('Latitude (degrees)')
          plt.title('Insolation at {} kyear'.format(kyear))
          plt.colorbar(contour_1, label = 'Insolation (W/m²)')
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude mean':
        output = np.mean(Orbital_Insolation(kyear+1, kyear).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=latitude, days = day_vals).T, axis = 0)
        if show_plot == True:
          fig, axs = plt.subplots()
          axs.plot(day_ax,output, label = 'space average')
          plt.xlabel('Time (Days)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation at {} kyear latitude mean over range {}'.format(kyear,latitude))
          axs.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
        output = np.mean(Orbital_Insolation(kyear+1, kyear).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=latitude, days = day_vals).T, axis = 1)
        if show_plot == True:
          lat_ax = np.linspace(latitude[0], latitude[1], (abs(latitude[0]-latitude[1]))+1)
          fig, axs = plt.subplots()
          axs.plot(lat_ax,output, label = 'time average')
          plt.xlabel('Latitude (degrees)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation at {} kyear, mean over days')
          plt.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude day mean':
        output = np.mean(Orbital_Insolation(kyear+1, kyear).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=latitude, days = day_vals).T, axis = (0,1))
        if show_plot == True:
          print('No associated plot for output') 
    elif isinstance(kyear, tuple) and len(kyear) == 3:
      eccentricity, obliquity, long_peri = kyear
      if output_type == 'array':
        output = Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', obl = obliquity, long = long_peri, ecc = eccentricity, kyear = '', lat=latitude, days = day_vals).T
        if show_plot == True:
          lat_ax = np.linspace(latitude[0], latitude[1], len(output))
          contour_1 = plt.contourf(day_ax,lat_ax,output,np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
          plt.xlabel('Time (days)')
          plt.ylabel('Latitude (degrees)')
          plt.title('Insolation with \n eccentricity: {:.2f}, obliquity: {:.2f} and longitude of perihelion: {:.2f}'.format(eccentricity,obliquity,long_peri))

          plt.colorbar(contour_1, label = 'Insolation (W/m²)')
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', obl = obliquity, long = long_peri, ecc = eccentricity, kyear = '', lat=latitude, days = day_vals).T, axis = 0)
        if show_plot == True:
          fig, axs = plt.subplots()
          axs.plot(day_ax,output, label = 'space average')
          plt.xlabel('Time (Days)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation mean over latitude range {} degrees with \n eccentricity: {:.2f}, obliquity: {:.2f} and longitude of perihelion: {:.2f}'.format(latitude,eccentricity,obliquity,long_peri))
          axs.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', obl = obliquity, long = long_peri, ecc = eccentricity, kyear = '', lat=latitude, days = day_vals).T, axis = 1)
        if show_plot == True:
          lat_ax = np.linspace(latitude[0], latitude[1], abs(latitude[0]-latitude[1])+1)
          fig, axs = plt.subplots()
          axs.plot(lat_ax,output, label = 'time average')
          plt.xlabel('Latitude (degrees)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation mean over days with \n eccentricity: {:.2f}, obliquity: {:.2f} and longitude of perihelion: {:.2f}'.format(eccentricity,obliquity,long_peri))
          plt.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude day mean':
        output = np.mean(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', obl = obliquity, long = long_peri, ecc = eccentricity, kyear = '', lat=latitude, days = day_vals).T, axis = (0,1))
        if show_plot == True:
          print('No associated plot for output')
    elif isinstance(kyear, tuple) and len(kyear) == 2:
      kyear_step = int(abs(kyear[1] - kyear[0]))
      kyear_range = np.linspace(kyear[0],kyear[1], kyear_step+1, dtype = int)
      lat_range = np.linspace(latitude[0], latitude[1], 1000)
      kyrs_inso = []
      [kyrs_inso.append(Orbital_Insolation(i+1, i).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat = latitude, days = day_vals).T) for i in kyear_range]
      output = np.array(kyrs_inso)
      output = xr.DataArray(data=output,dims=['kyr','lat','day'],coords={'kyr':kyear_range,'lat':lat_range, 'day':day_ax})
      if output_type == 'array':
        pass
        if show_plot == True:
          reshaped_output = []       
          for i in range(np.shape(output)[1]):
            row = []            
            for f in range(np.shape(output)[0]):
              row.append(output[f,i,:])
            row = np.hstack(row)
            reshaped_output.append(row)
          kyear_ax = np.linspace(0,len(output),np.shape(reshaped_output)[1])
          lat_ax = np.linspace(latitude[0], latitude[1], np.shape(output)[1])
          contour_1 = plt.contourf(kyear_ax,lat_ax,reshaped_output,np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
          plt.xlabel('Time (kyears)')
          plt.ylabel('Latitude (degrees)')
          plt.title('Insolation over kyear range {}'.format(kyear))
          plt.colorbar(contour_1, label = 'Insolation (W/m²)')
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude mean':
        output = np.mean(output, axis =1)
        if show_plot == True:
          insolation_v_time = np.hstack(output)
          kyear_ax = np.linspace(kyear[0],kyear[1],len(insolation_v_time))
          plt.plot(kyear_ax,insolation_v_time, label = 'space average')
          plt.xlabel('Time (kyears)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation over kyear range {} latitude mean over range {}'.format(kyear,latitude))
          plt.legend()
          plt.savefig('{}.png'.format(filename))       
      elif output_type == 'day mean':
        output = np.mean(output, axis =2)
        if show_plot == True:
          fig,axs = plt.subplots()
          insolation_v_lat = np.hstack(output)
          kyear_ax = np.linspace(kyear[0],kyear[1],len(insolation_v_lat))
          x = []
          [x.append(np.linspace(-np.pi/2,np.pi/2,np.shape(output)[1])) for i in range(kyear_step)]
          x = np.rad2deg(np.hstack(x))
          axs.plot(kyear_ax,insolation_v_lat, label = 'day average')
          axs.set_xlabel('Time (kyears)')
          axs.set_ylabel('TOA Insolation (W/m²)')
          axs.set_title('Insolation over kyear range {} day mean'.format(kyear))
          axs.legend()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude day mean':
        output = np.mean(output, axis =(1,2))   
        if show_plot == True:
          kyear_ax = np.linspace(kyear[0],kyear[1],len(output))
          plt.plot(kyear_ax,output)
          plt.xlabel('Time (kyears)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation latitude mean over range {} and day mean'.format(latitude))
          plt.savefig('{}.png'.format(filename))     
      elif output_type == 'kyear mean':
        output = np.mean(output, axis =0)
        if show_plot == True:
          lat_ax = np.linspace(latitude[0], latitude[1], len(output))
          contour_1 = plt.contourf(day_ax,lat_ax,output,np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
          plt.xlabel('Time (days)')
          plt.ylabel('Latitude (degrees)')
          plt.title('Insolation kyear mean over range {}'.format(kyear))
          plt.colorbar(contour_1, label = 'Insolation (W/m²)')
          plt.savefig('{}.png'.format(filename))   
    elif isinstance(kyear,list) or isinstance(kyear,np.ndarray):
      array_at_kyears = []
      [array_at_kyears.append(Orbital_Insolation(i+1,i).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat',lat = latitude,days = day_vals).T) for i in kyear]    
      
      kyear_step = int(abs(kyear[1] - kyear[0]))
      lat_range = np.linspace(latitude[0], latitude[1], 1+abs(latitude[0]-latitude[1]))
      output = np.array(array_at_kyears)
      output = xr.DataArray(data=output,dims=['kyr','lat','day'],coords={'kyr':kyear,'lat':lat_range,'day':day_ax})
  
      if output_type == 'array':
        pass

        if show_plot == True:
          lat_ax = np.linspace(latitude[0], latitude[1], np.shape(output)[1])
          plot_rows, plot_cols = Helper_Functions.find_plot_dims(kyear)
                      
          fig, axs = plt.subplots(nrows=plot_rows,ncols=plot_cols)
          
          count = 0
          if np.ndim(axs) == 2:
            for i in range(plot_rows):
              for f in range(plot_cols):
                if count <= len(kyear)-1:
                  im = axs[i,f].contourf(day_ax,lat_ax,output[count],np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
                  axs[i,f].set_title('kyear {}'.format(kyear[count]))
                  axs[i,f].set_xlabel('Time (kyears)')
                  axs[i,f].set_ylabel('Latitude (degrees)')
                  count += 1
                else:
                  axs[i,f].remove()
          elif np.ndim(axs) == 1:
            for i in range(plot_rows):
              for f in range(plot_cols):
                if count <= len(kyear)-1:
                  im = axs[f].contourf(day_ax,lat_ax,output[count],np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
                  axs[f].set_title('kyear {}'.format(kyear[count]))
                  axs[f].set_xlabel('Time (kyears)')
                  axs[f].set_ylabel('Latitude (degrees)')
                  count += 1
                else:
                  axs[f].remove()
        
          cb_ax = fig.add_axes([.85,.124,.04,.754])
          fig.colorbar(im,label = 'Insolation (W/m²)',cax = cb_ax, anchor = (10,1))
          plt.tight_layout(rect=[0,0,.8,1])
          plt.savefig('{}.png'.format(filename))                
      elif output_type == 'latitude mean':

        output = np.mean(output,axis = 1)

        if show_plot == True:

          lat_ax = np.linspace(latitude[0], latitude[1], np.shape(output)[1])
          plot_rows, plot_cols = Helper_Functions.find_plot_dims(kyear)
                      
          fig, axs = plt.subplots()

          [plt.plot(day_ax,output[i], label = '{:.1f}kyear'.format(kyear[i])) for i in range(len(kyear))]
          
          plt.xlabel('Time (days)')
          plt.ylabel('Insolation (W/m²)')
          plt.title('Insolation for kyears latitude mean over range {}'.format(latitude))
          plt.legend()
          plt.tight_layout()
          plt.savefig('{}.png'.format(filename))    
      elif output_type == 'day mean':

        output = np.mean(output, axis = 2)

        if show_plot == True:

          lat_ax = np.linspace(latitude[0], latitude[1], np.shape(output)[1])
          plot_rows, plot_cols = Helper_Functions.find_plot_dims(kyear)
                      
          fig, axs = plt.subplots()

          [plt.plot(lat_ax,output[i], label = '{:.1f}kyear'.format(kyear[i])) for i in range(len(kyear))]
          
          plt.xlabel('Latitude (degrees)')
          plt.ylabel('Insolation (W/m²)')
          plt.title('Day Mean Insolation for Kyears latitude mean over range {}'.format(latitude))
          plt.legend()
          
          plt.tight_layout()
          plt.savefig('{}.png'.format(filename))       
      elif output_type == 'kyear mean':

        output = np.mean(output, axis = 0)

        if show_plot == True:

          lat_ax = np.linspace(latitude[0], latitude[1], np.shape(output)[0])
          fig, axs = plt.subplots()
          im = axs.contourf(day_ax,lat_ax,output,np.arange(0,int(np.max(output)),10), extend = 'max', cmap=plt.get_cmap('hot'))
          kyear = [float(i) for i in kyear]
          axs.set_title('Mean Insolation for Kyears {}'.format(kyear))
          axs.set_xlabel('Time (days)')
          axs.set_ylabel('Latitude (degrees)')
          plt.colorbar(im, label = 'Insolation (W/m²)')

          plt.tight_layout()
          plt.savefig('{}.png'.format(filename))          
      elif output_type == 'latitude day mean':
        output = np.mean(output, axis = (1,2))
        if show_plot == True:
          kyear_ax = np.linspace(kyear[0],kyear[1],len(output))
          plt.plot(kyear_ax,output)
          plt.xlabel('Time (kyears)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation latitude mean over range {} and day mean \n for kyears {}'.format(latitude, kyear))
          plt.savefig('{}.png'.format(filename))  
  elif isinstance(latitude, list) or isinstance(latitude,np.ndarray):     
    if kyear is None:
      array_at_lats = []
      [array_at_lats.append(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=i, days = day_vals).T) for i in latitude]
      output = np.array(array_at_lats)
      output = xr.DataArray(data=output,dims=['lat','day'],coords={'lat':latitude, 'day':day_ax})
      if output_type == 'array':
        pass
        if show_plot == True:
          [plt.plot(day_ax,output[i], label = '{} degree'.format(latitude[i])) for i in range(len(latitude))]
          plt.legend()
          plt.xlabel('Time (days)')
          lat = [float(i) for i in latitude]
          plt.title('Modern Insolation for latitudes {}'.format(lat))
          plt.ylabel('TOA Insolation (W/m²)')
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude mean':
        output = np.mean(output, axis = 0)
        if show_plot == True:
          plt.plot(day_ax,output, label = 'latitude mean') 
          plt.legend()
          plt.xlabel('Time (days)')
          plt.title('Modern Insolation latitude mean for latitudes {}'.format(latitude))
          plt.ylabel('TOA Insolation (W/m²)')
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
        output = np.mean(output, axis = 1)
        if show_plot == True:
          latitude = [str(i) for i in latitude]
          plt.bar(latitude,output)
          plt.xlabel('Latitude (degrees)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Modern Insolation day mean')
          plt.savefig('{}.png'.format(filename))     
      elif output_type == 'latitude day mean':
        output = np.mean(output, axis = (0,1))
        if show_plot == True:
          print('No associated plot for output')   
    elif isinstance(kyear, int) or isinstance(kyear,float):
      array_at_lats = []
      [array_at_lats.append(Orbital_Insolation(kyear+1, kyear).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=i, days = day_vals).T) for i in latitude]
      output = np.array(array_at_lats)
      output = xr.DataArray(data=output,dims=['lat','day'],coords={'lat':latitude, 'day':day_ax})
      if output_type == 'array':
        output = output
        if show_plot == True:
          [plt.plot(day_ax,output[i], label = '{} degree'.format(latitude[i])) for i in range(len(latitude))]
          plt.legend()
          plt.xlabel('Time (days)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('kyear {} BP'.format(kyear))
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude mean':
        output = np.mean(output, axis = 0)
        if show_plot == True:
          plt.plot(day_ax,output, label = 'latitude mean') 
          plt.legend()
          plt.xlabel('Time (days)')
          plt.title('Insolation at kyrea {} BP and latitude mean for latitudes {}'.format(kyear,latitude))
          plt.ylabel('TOA Insolation (W/m²)')
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
        output = np.mean(output, axis = 1)
        if show_plot == True:
          latitude = [str(i) for i in latitude]
          plt.bar(latitude,output)
          plt.xlabel('Latitude (degrees)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation for kyear {} BP day mean'.format(kyear))
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude day mean':
        output = np.mean(output, axis = (0,1))
        if show_plot == True:
          print('No associated plot for output')    
    elif isinstance(kyear, tuple) and len(kyear) == 3:
      eccentricity, obliquity, long_peri = kyear
      array_at_lats = []
      [array_at_lats.append(Orbital_Insolation(1,0).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', obl = obliquity, long = long_peri, ecc = eccentricity, kyear = '', lat=i, days = day_vals).T) for i in latitude]
      output = np.array(array_at_lats)
      output = xr.DataArray(data=output,dims=['lat','day'],coords={'lat':latitude, 'day':day_ax})
      if output_type == 'array':
        output = output
        if show_plot == True:
          [plt.plot(day_ax,output[i], label = '{} degree'.format(latitude[i])) for i in range(len(latitude))]
          plt.legend()
          plt.xlabel('Time (days)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation at latitudes {} with \n eccentricity: {:.2f}, obliquity: {:.2f}, longitude of perhelion: {:.2f}'.format(latitude,kyear[0],kyear[1],kyear[2]))
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude mean':
        output = np.mean(output, axis = 0)
        if show_plot == True:
          plt.plot(day_ax,output, label = 'latitude mean') 
          plt.legend()
          plt.xlabel('Time (days)')
          plt.title('Insolation latitude mean for {} with \n eccentricity: {:.2f}, obliquity: {:.2f}, longitude of perhelion: {:.2f}'.format(latitude,kyear[0],kyear[1],kyear[2]))
          plt.ylabel('TOA Insolation (W/m²)')
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'day mean':
        output = np.mean(output, axis = 1)
        if show_plot == True:
          latitude = [str(i) for i in latitude]
          plt.bar(latitude,output)
          plt.xlabel('Latitude (degrees)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation for day mean with \n eccentricity: {:.2f}, obliquity: {:.2f}, longitude of perhelion: {:.2f}'.format(kyear[0],kyear[1],kyear[2]))
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude day mean':
        output = np.mean(output, axis = (0,1))
        if show_plot == True:
          print('No associated plot for output')  
    elif isinstance(kyear, tuple) and len(kyear) == 2:
        kyear_step = int(abs(kyear[1] - kyear[0]))
        kyear_range = np.linspace(kyear[0],kyear[1], kyear_step+1, dtype = int)
        kyrs_inso = []
        for i in kyear_range:
          array_at_lats = []  
          for f in latitude:
            signle_lat_val = Orbital_Insolation(i+1, i).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat = f, days = day_vals).T
            array_at_lats.append(signle_lat_val)
          kyrs_inso.append(array_at_lats)
        output = np.array(kyrs_inso)
        output = xr.DataArray(data=output,dims=['kyr','lat','day'],coords={'kyr':kyear_range,'lat':latitude, 'day':day_ax})   
        if output_type == 'array':
          if show_plot == True:
            insolation_v_time = np.hstack(output)
            kyear_ax = np.linspace(kyear[0],kyear[1],len(insolation_v_time[0]))
            [plt.plot(kyear_ax,insolation_v_time[i], label = '{} degree'.format(latitude[i])) for i in range(len(latitude))]
            plt.xlabel('Time (kyears)')
            plt.ylabel('TOA Insolation (W/m²)')
            plt.title('Insolation over kyear range {} for latitudes {}'.format(kyear,latitude))
            plt.legend()
            plt.savefig('{}.png'.format(filename))   
        elif output_type == 'latitude mean':
          output = np.mean(output, axis =1)
          if show_plot == True:
            insolation_v_time = np.hstack(output)
            kyear_ax = np.linspace(kyear[0],kyear[1],len(insolation_v_time))
            plt.plot(kyear_ax,insolation_v_time, label = 'latitude mean')
            plt.xlabel('Time (kyears)')
            plt.ylabel('TOA Insolation (W/m²)')
            plt.title('Insolation over kyear range {} with mean of latitudes {}'.format(kyear,latitude))
            plt.legend()
            plt.savefig('{}.png'.format(filename))   
        elif output_type == 'day mean':
          output = np.mean(output, axis =2)
          if show_plot == True:
            fig,axs = plt.subplots()
            kyear_ax = np.linspace(kyear[0],kyear[1],np.shape(output)[0])
            x = []
            [x.append(np.linspace(-np.pi/2,np.pi/2,np.shape(output)[1])) for i in range(kyear_step)]
            x = np.rad2deg(np.hstack(x))
            [axs.plot(kyear_ax,output[:,i],label = '{} degrees'.format(latitude[i])) for i in range(len(latitude))]
            axs.set_xlabel('Time (kyears)')
            axs.set_ylabel('TOA Insolation (W/m²)')
            axs.set_title('Insolation of day mean at latitudes {}'.format(latitude))
            axs.legend()
            plt.savefig('{}.png'.format(filename))   
        elif output_type == 'latitude day mean':
          output = np.mean(output, axis =(1,2))
          if show_plot == True:
            kyear_ax = np.linspace(kyear[0],kyear[1],len(output))
            plt.plot(kyear_ax,output)
            plt.xlabel('Time (kyears)')
            plt.ylabel('TOA Insolation (W/m²)')
            plt.title('Insolation over kyear range {} with day mean \n and latitude mean for {} degrees'.format(kyear,latitude))
            plt.savefig('{}.png'.format(filename))     
        elif output_type == 'kyear mean':
          output = np.mean(output, axis = 0)
          if show_plot == True:
            [plt.plot(day_ax,output[i], label = '{} degree'.format(latitude[i])) for i in range(len(latitude))]
            plt.legend()
            plt.xlabel('Time (days)')
            plt.ylabel('TOA Insolation (W/m²)')
            plt.title('Insolation mean over kyear range {} BP'.format(kyear))
            plt.savefig('{}.png'.format(filename)) 
    elif isinstance(kyear,list) or isinstance(kyear,np.ndarray):
      array_at_kyears = []
      for i in kyear:
        array_at_lats = []
        [array_at_lats.append(Orbital_Insolation(i+1,i).avg_insolation(experiment(grid_num = 3).config, lat_array = 'for lat', lat=f, days = day_vals).T) for f in latitude]
        array_at_kyears.append(array_at_lats)
      
      output = np.array(array_at_kyears)
      output = xr.DataArray(data=output,dims=['kyr','lat','day'],coords={'kyr':kyear,'lat':latitude,'day':day_ax})
      
      if output_type == 'array':
        pass
        if show_plot == True:
          plot_rows, plot_cols = Helper_Functions.find_plot_dims(kyear)
                      
          fig, axs = plt.subplots(nrows=plot_rows,ncols=plot_cols)
          
          count = 0
          if np.ndim(axs) == 2:
            for i in range(plot_rows):
              for f in range(plot_cols):
                if count <= len(kyear)-1:
                  [axs[i,f].plot(day_ax,output[count,k,:], label = '{} degree'.format(latitude[k])) for k in range(len(latitude))]
                  axs[i,f].set_title('kyear {}'.format(kyear[count]))
                  axs[i,f].set_xlabel('Time (days)')
                  axs[i,f].set_ylabel('Insolation (W/m²)')
                  axs[i,f].legend()
                  count += 1
                else:
                  axs[i,f].remove()
          elif np.ndim(axs) == 1:
            for i in range(plot_rows):
              for f in range(plot_cols):
                if count <= len(kyear)-1:
                  [axs[f].plot(day_ax,output[count,k,:], label = '{} degree'.format(latitude[k])) for k in range(len(latitude))]
                  axs[f].set_title('kyear {}'.format(kyear[count]))
                  axs[f].set_xlabel('Time (days)')
                  axs[f].set_ylabel('Insolation (W/m²)')
                  axs[f].legend()
                  count += 1
                else:
                  axs[f].remove()
          plt.tight_layout()
          plt.savefig('{}.png'.format(filename))   
      elif output_type == 'latitude mean':

        output = np.mean(output, axis = 1)

        if show_plot == True:

          lat_ax = np.linspace(latitude[0], latitude[1], np.shape(output)[1])
          plot_rows, plot_cols = Helper_Functions.find_plot_dims(kyear)
                      
          fig, axs = plt.subplots()

          [plt.plot(day_ax,output[i], label = '{:.1f}kyear'.format(kyear[i])) for i in range(len(kyear))]
          
          plt.xlabel('Time (days)')
          plt.ylabel('Insolation (W/m²)')
          plt.title('Insolation for Kyears with mean of latitudes {}'.format(latitude))
          plt.legend()
          plt.tight_layout()
          plt.savefig('{}.png'.format(filename))       
      elif output_type == 'day mean':
        output = np.mean(output,axis = 2)

        if show_plot == True:

          lat_ax = np.linspace(latitude[0], latitude[1], np.shape(output)[1])
                      
          fig, axs = plt.subplots()

          [plt.plot(lat_ax,output[i], label = '{:.1f}kyear'.format(kyear[i])) for i in range(len(kyear))]
          
          plt.xlabel('Latitude (degrees)')
          plt.ylabel('Insolation (W/m²)')
          plt.title('Day Mean Insolation for latitudes {}'.format(latitude))
          plt.legend()
          
          plt.tight_layout()
          plt.savefig('{}.png'.format(filename))      
      elif output_type == 'kyear mean':

        output = np.mean(output, axis = 0)

        if show_plot == True:

          [plt.plot(day_ax,output[i], label = '{} degree'.format(latitude[i])) for i in range(len(latitude))]
          plt.legend()
          plt.xlabel('Time (days)')
          plt.title('Insolation at latitudes {} for mean of kyears {}'.format(latitude,kyear))
          plt.ylabel('TOA Insolation (W/m²)')
          plt.savefig('{}.png'.format(filename))       
      elif output_type == 'latitude day mean':
        output = np.mean(output, axis = (1,2))
        if show_plot == True:
          kyear_ax = np.linspace(kyear[0],kyear[1],len(output))
          plt.plot(kyear_ax,output)
          plt.xlabel('Time (kyears)')
          plt.ylabel('TOA Insolation (W/m²)')
          plt.title('Insolation for kyears {} with day mean \n and latitude mean for {} degrees'.format(kyear,latitude))
          plt.savefig('{}.png'.format(filename))

  else:
    raise ValueError('invalid output type') 

  if show_plot == True:
    if np.ndim(output) == 0:
      pass
    else:
      print('Plot of output has been saved as {}.png'.format(filename))

  if solve_energy == True: #needs work

    dims = output.dims
    
    if 'lat' in dims and 'day' in dims and 'kyr' not in dims:

      sum = np.sum(output)
      radius = 6378100 # m
      lat_bound_a = np.deg2rad(float(output.coords['lat'][0]))
      lat_bound_b = np.deg2rad(float(output.coords['lat'][-1]))
      area = 2 * np.pi * (radius**2) * np.abs( np.sin(lat_bound_a) - np.sin(lat_bound_b) )
      days_bound_a = float(output.coords['day'][0])
      days_bound_b = float(output.coords['day'][-1])
      days = days_bound_b - days_bound_a
      seconds = days*24*60*60

      energy = sum*area*seconds # Joules
      energy = np.array(energy)
      
      output = energy      
  
  return output

def GMT(kyear = None):

  if kyear == None:

    output = Model_Class().model(S_type = 1, grid = experiment(1).config, T = experiment().Ti, CO2_ppm = None, D = None, F = 0, moist = 1, albT = 1, seas = 2, thermo = 0, kyear = 1, hide_run = 'On')
    surface_temperature = output[2]
    GMT = np.mean(surface_temperature)
    return GMT

  elif isinstance(kyear, int):

    kyear = kyear + 1
    
    output = Model_Class().model(S_type = 1, grid = experiment(1).config, T = experiment().Ti, CO2_ppm = None, D = None, F = 0, moist = 1, albT = 1, seas = 2, thermo = 0, kyear = kyear, hide_run = 'On')
    surface_temperature = output[2]
    GMT = np.mean(surface_temperature)
    return GMT

def climate(kyear = None, latitude = None, moist = None, seas = None):

  if latitude == None:

    if kyear == None:

      output = Model_Class().model(S_type = 1, grid = experiment(1).config, T = experiment().Ti, CO2_ppm = None, D = None, F = 0, moist = 1, albT = 1, seas = 1, thermo = 0, kyear = 1, hide_run = 'On')
      surface_temperature = output[2]
      meridional_temperature_gradient = output[11]
      meridional_energy_transport = output[10]
      outgoing_longwave_radiation = output[16]
      absorbed_solar_radiation = output[4]
      southern_hemisphere_sea_ice_edge = output[14][3]
      insolation = output[5]

      return insolation, absorbed_solar_radiation, outgoing_longwave_radiation, surface_temperature, meridional_temperature_gradient, meridional_energy_transport, southern_hemisphere_sea_ice_edge

    elif isinstance(kyear, int):

      kyear = kyear + 1
      
      output = Model_Class().model(S_type = 1, grid = experiment(1).config, T = experiment().Ti, CO2_ppm = None, D = None, F = 0, moist = 1, albT = 1, seas = 1, thermo = 0, kyear = kyear, hide_run = 'On')
      surface_temperature = output[2]
      meridional_temperature_gradient = output[11]
      meridional_energy_transport = output[10]
      outgoing_longwave_radiation = output[16]
      absorbed_solar_radiation = output[4]
      southern_hemisphere_sea_ice_edge = output[14][3]
      insolation = output[5]

      return insolation, absorbed_solar_radiation, outgoing_longwave_radiation, surface_temperature, meridional_temperature_gradient, meridional_energy_transport, southern_hemisphere_sea_ice_edge

    elif len(kyear) == 3:

      eccentricity, obliquity, long_peri = kyear
  
      output = Model_Class().model(S_type = 1, grid = experiment(1).config, T = experiment().Ti, CO2_ppm = None, D = None, F = 0, moist = 1, albT = 1, seas = 1, thermo = 0, kyear = 'forced', obl = obliquity, long = long_peri, ecc = eccentricity, hide_run = 'On')
      surface_temperature = output[2]
      meridional_temperature_gradient = output[11]
      meridional_energy_transport = output[10]
      outgoing_longwave_radiation = output[16]
      absorbed_solar_radiation = output[4]
      southern_hemisphere_sea_ice_edge = output[14][3]
      insolation = output[5]

      return insolation, absorbed_solar_radiation, outgoing_longwave_radiation, surface_temperature, meridional_temperature_gradient, meridional_energy_transport, southern_hemisphere_sea_ice_edge

  elif latitude != None:
    
    if np.max(np.abs(latitude)) > 90:

      raise ValueError('latitude value is out of degree range (-90,90)')

    elif isinstance(latitude, int):

      if kyear == None:

        output = Model_Class().model(S_type = 1, grid = experiment(0).config, T = experiment(0).Ti, CO2_ppm = None, D = None, F = 0, moist = 1, albT = 1, seas = 1, thermo = 0, kyear = 1, hide_run = 'On')
        surface_temperature = output[2]
        meridional_temperature_gradient = output[11]
        meridional_energy_transport = output[10]
        outgoing_longwave_radiation = output[16]
        absorbed_solar_radiation = output[4]
        southern_hemisphere_sea_ice_edge = output[14][3]
        insolation = output[5]

        return insolation, absorbed_solar_radiation, outgoing_longwave_radiation, surface_temperature, meridional_temperature_gradient, meridional_energy_transport, southern_hemisphere_sea_ice_edge

#---------------------------------------------------------------#

class experiment(): # experimental set ups for analysis on EBM

  def __init__(self, grid_num = 1):

    self.config = self.make_config(grid_num)
    self.lat_deg = np.rad2deg(np.arcsin(self.config['x']))
    self.lat_rad = np.deg2rad(self.lat_deg)
    self.Ti = 7.5+20*(1-2*self.config['x']**2) # initial condition 
    self.F = 0
    self.month = {'January':0, "Febuary":31, "March": 59, "April": 90, "May":120, "June":151, "July":181, "August":212, "September":243, "October":273, "November":304, "December":334}
    self.r = 100
    self.control_moist = 1
    self.control_albT = 1
    self.control_seas = 2
    self.control_thermo = 0
    self.control_CO2 = 280
    self.control_orbit_type = 1
    self.control_label = 'Control'

  def make_config(self, grid): #sets the values for the grid and duration configuration

    if grid == 0: #quick run // low resolution
      n = 90
      dx = 2./n #grid box width
      x = np.linspace(-1+dx/2,1-dx/2,n) #native grid
      xb = np.linspace(-1+dx,1-dx,n-1) 
      nt = 1000
      dur= 1
      dt = 1./nt
      eb = 0.1

    elif grid == 1: #intermediate run // intermediate resolution
      n = 120
      dx = 2/n #grid box width
      x = np.linspace(-1+dx/2,1-dx/2,n) #native grid
      xb = np.linspace(-1+dx,1-dx,n-1) 
      nt = 1000
      dur= 1
      dt = 1./nt
      eb = 1e-6

    elif grid == 2: #long run // high resolution
      n = 120
      dx = 2./n #grid box width
      x = np.linspace(-1+dx/2,1-dx/2,n) #native grid
      xb = np.linspace(-1+dx,1-dx,n-1) 
      nt = 1000
      dur= 100
      dt = 1./nt
      eb = 1e-10

    elif grid == 3:

      n = 180
      dx = 2./n #grid box width
      x = np.linspace(-1+dx/2,1-dx/2,n) #native grid
      xb = np.linspace(-1+dx,1-dx,n-1) 
      nt = 1000
      dur= 100
      dt = 1./nt
      eb = 1e-10


    grid_dict = {'n': n, 'dx': dx, 'x': x, 'xb': xb, 'nt': nt, 'dur': dur, 'dt': dt, 'eb':eb}

    return grid_dict

  def saturation_specific_humidity(self,temp,press): #Calculates the saturation specific humidity form OGorman and Schneider 2008

    """
    We assume a single liquid-to-vapor phase transition with the parameter values 
    of the Clausius Clapeyron (CC) relation given in OGorman and Schneider (2008) 
    to determine the saturation specific humidity qs(T).

    """

    es0 = 610.78 # saturation vapor pressure at t0 (Pa)
    t0 = 273.16
    Rv = 461.5
    Lv = 2.5E6
    ep = 0.622 # ratio of gas constants of dry air and water vapor
    temp = temp + 273.15 # convert to Kelvin
    es = es0 * np.exp(-(Lv/Rv) * ((1/temp)-(1/t0)))
    qs = ep * es / press

    return qs

  def main(self): # compaires two runs

#--------------------Control-Panel---------------------------#
    # Toggle EBM parameters
    On_Off = {'moist':1, 'albT':1, 'seas':1,'thermo':0}

    # Manually set orbital values
    Orbitals_1 = {'obl':Helper_Functions().orbit_extrema('obl','min'), 'long':Helper_Functions().orbit_at_time('long',1), 'ecc':Helper_Functions().orbit_at_time('ecc',1)}
    Orbitals_2 = {'obl':Helper_Functions().orbit_extrema('obl','max'), 'long':Helper_Functions().orbit_at_time('long',1), 'ecc':Helper_Functions().orbit_at_time('ecc',1)}
    
    kyear = 'forced' # If forced willuse manually set values
    
    chart = 21 # generates figure # from Figures()
    
    # Radiative forcings
    kyear1 = 1 
    CO2_ppm_1 = 280
    kyear2 = 1 
    CO2_ppm_2 = 280
#------------------------------------------------------------#

    if True == True:
        #Model Runs
        if kyear == 'forced':
          if chart == 4 or chart == 5:
            output_1 = Model_Class.model(self, 0, self.config, self.Ti, CO2_ppm = CO2_ppm_1, D = None, F = 0, moist = On_Off['moist'], albT = On_Off['albT'], seas = On_Off['seas'], thermo = On_Off['thermo'])
            output_2 = Model_Class.model(self, 1, self.config, self.Ti, CO2_ppm = CO2_ppm_2, D = None, F = 0, moist = On_Off['moist'], albT = On_Off['albT'], seas = On_Off['seas'], thermo = On_Off['thermo'], obl = Orbitals_1['obl'], long = Orbitals_1['long'], ecc = Orbitals_1['ecc'], kyear = kyear)
          else:
            output_1 = Model_Class.model(self, 1, self.config, self.Ti, CO2_ppm = CO2_ppm_1, D = None, F = 0, moist = On_Off['moist'], albT = On_Off['albT'], seas = On_Off['seas'], thermo = On_Off['thermo'], obl = Orbitals_1['obl'], long = Orbitals_1['long'], ecc = Orbitals_1['ecc'], kyear = kyear)
            output_2 = Model_Class.model(self, 1, self.config, self.Ti, CO2_ppm = CO2_ppm_2, D = None, F = 0, moist = On_Off['moist'], albT = On_Off['albT'], seas = On_Off['seas'], thermo = On_Off['thermo'], obl = Orbitals_2['obl'], long = Orbitals_2['long'], ecc = Orbitals_2['ecc'], kyear = kyear)
        else:
          if chart == 4 or chart == 5:
            output_1 = Model_Class.model(self, 0, self.config, self.Ti, CO2_ppm = CO2_ppm_1, D = None, F = 0, moist = On_Off['moist'], albT = On_Off['albT'], seas = On_Off['seas'], thermo = On_Off['thermo'])
            output_2 = Model_Class.model(self, 1, self.config, self.Ti, CO2_ppm = CO2_ppm_2, D = None, F = 0, moist = On_Off['moist'], albT = On_Off['albT'], seas = On_Off['seas'], thermo = On_Off['thermo'])
          else:
            output_1 = Model_Class.model(self, 1, self.config, self.Ti, CO2_ppm = CO2_ppm_1, D = None, F = 0, moist = On_Off['moist'], albT = On_Off['albT'], seas = On_Off['seas'], thermo = On_Off['thermo'], kyear = kyear1)
            output_2 = Model_Class.model(self, 1, self.config, self.Ti, CO2_ppm = CO2_ppm_2, D = None, F = 0, moist = On_Off['moist'], albT = On_Off['albT'], seas = On_Off['seas'], thermo = On_Off['thermo'], kyear = kyear2)

        #Figure Generation
        if On_Off['seas'] == 1:
          Figures.figure(self, self.config, chart, output_1, output_2, subchart = 'seas', kyear_1 = kyear1, kyear_2 = kyear2)
        elif On_Off['seas'] == 0 or On_Off['seas'] == 2:
          Figures.figure(self, self.config, chart, output_1, output_2, output_1, output_2, 'annual', kyear1, kyear2)
        plt.tight_layout()
        plt.savefig('MEBMplots.jpg')
    else:
      pass

    return None

  def table_1_run(self, CO2_ppm = None, D = 0.3, s_type = None, hide_run = None): # runs EBM for data table 1

    On_Off = {'moist':1, 'albT':1, 'seas':1,'thermo':0}

    kyear = 1

    output = Model_Class.model(self, s_type, self.config, self.Ti, CO2_ppm, D, self.F, moist = On_Off['moist'], albT = On_Off['albT'], seas = On_Off['seas'], thermo = On_Off['thermo'], kyear = kyear, hide_run = hide_run)

    outputs = Helper_Functions().table_1_outs(self.config, output)

    outputs = list(outputs)

    if s_type == 0:
      s_type = 'def'
    elif s_type == 1:
      s_type = 'orb'
    if CO2_ppm == None:
      CO2_ppm = '280'
    if D == 0.3:
      D = 'def'

    inputs = list([CO2_ppm, D, s_type])

    return inputs, outputs

  def generate_table_1(self): # generates and tabulates data table 1

    print('------------------')
    print('')
    print('generating table 1')

    row_1 = self.table_1_run(CO2_ppm = 560, D = 0.3, s_type = 0)
    row_2 = self.table_1_run(CO2_ppm = None, D = 0.3, s_type = 0)
    row_3 = self.table_1_run(CO2_ppm = 140, D = 0.3, s_type = 0)
    
    row_4 = self.table_1_run(CO2_ppm = None, D = 0.4, s_type = 0)
    row_5 = self.table_1_run(CO2_ppm = None, D = 0.3, s_type = 0)
    row_6 = self.table_1_run(CO2_ppm = None, D = 0.2, s_type = 0)
    
    
    row_7 = self.table_1_run(CO2_ppm = 560, D = 0.3, s_type = 1)
    row_8 = self.table_1_run(CO2_ppm = None, D = 0.3, s_type = 1)
    row_9 = self.table_1_run(CO2_ppm = 140, D = 0.3, s_type = 1)
   
    row_10 = self.table_1_run(CO2_ppm = None, D = 0.4, s_type = 1)
    row_11 = self.table_1_run(CO2_ppm = None, D = 0.3, s_type = 1)
    row_12 = self.table_1_run(CO2_ppm = None, D = 0.2, s_type = 1)

    row_control = self.table_1_run(CO2_ppm = 475, D = 0.3, s_type = 1)

    row_1 = [i for sublist in row_1 for i in sublist]
    row_2 = [i for sublist in row_2 for i in sublist]
    row_3 = [i for sublist in row_3 for i in sublist]
    row_4 = [i for sublist in row_4 for i in sublist]
    row_5 = [i for sublist in row_5 for i in sublist]
    row_6 = [i for sublist in row_6 for i in sublist]
    row_7 = [i for sublist in row_7 for i in sublist]
    row_8 = [i for sublist in row_8 for i in sublist]
    row_9 = [i for sublist in row_9 for i in sublist]
    row_10 = [i for sublist in row_10 for i in sublist]
    row_11 = [i for sublist in row_11 for i in sublist]
    row_12 = [i for sublist in row_12 for i in sublist]
    row_control = [i for sublist in row_control for i in sublist]

    df = pd.DataFrame([row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8, row_9, row_10, row_11, row_12, row_control])
    df.columns = ["Atm CO2 ppm", "D", "Ins Type", "GMT", "Eq-Pole Temp Diff", "Annual Mean Ins", 'Heat Transport Max', 'Heat Tranport Max Location', "T grad Max", 'T grad Max Location', 'Wind Max', 'Wind Max Location', 'Max Ice', 'Min Ice', 'Mean Ice', 'GMAlb', 'GM_OLR']
    df = Helper_Functions().format_data(df)
    
    sup_columns = pd.MultiIndex.from_tuples([('Inputs', df.columns[0]), ('Inputs', df.columns[1]), ('Inputs', df.columns[2]),
    ('Outputs', df.columns[3]), ('Outputs', df.columns[4]), ('Outputs', df.columns[5]), ('Outputs', df.columns[6]), ('Outputs', df.columns[7]), 
    ('Outputs', df.columns[8]), ('Outputs', df.columns[9]), ('Outputs', df.columns[10]), ('Outputs', df.columns[11]), ('Outputs', df.columns[12]), ('Outputs', df.columns[13]), ('Outputs', df.columns[14]), ('Outputs', df.columns[15]), ('Outputs', df.columns[16])])

    df.columns = sup_columns

    df.to_csv("table_1.csv")

    return None

  def table_2_run(self, obl, long, ecc): # runs EBM for data table 2

    On_Off = {'moist':1, 'albT':1, 'seas':1,'thermo':0}
    
    Orbitals = {'obl':obl, 'long':long, 'ecc':ecc}

    output = Model_Class.model(self, T = self.Ti, F = self.F, grid = self.config, CO2_ppm = None, D = None, S_type = 1, moist = On_Off['moist'], albT = On_Off['albT'], seas = On_Off['seas'], thermo = On_Off['thermo'], obl = Orbitals['obl'], long = Orbitals['long'], ecc = Orbitals['ecc'], kyear = 'forced', hide_run = None)

    outputs = Helper_Functions().table_2_outs(self.config, output)
    outputs = list(outputs)

    inputs = list([Orbitals['obl'],Orbitals['long'],Orbitals['ecc']])

    return inputs, outputs

  def generate_table_2(self): # generates and tabulates data table 1

    control_row = self.table_2_run(obl = Helper_Functions().orbit_at_time('obl', 1), long = Helper_Functions().orbit_at_time('long', 1), ecc = Helper_Functions().orbit_at_time('ecc', 1))
    
    row_1 = self.table_2_run(obl = Helper_Functions().orbit_extrema('obl', 'min'), long = Helper_Functions().orbit_at_time('long', 1), ecc = Helper_Functions().orbit_at_time('ecc', 1))
    row_2 = self.table_2_run(obl = Helper_Functions().orbit_extrema('obl', 'max'), long = Helper_Functions().orbit_at_time('long', 1), ecc = Helper_Functions().orbit_at_time('ecc', 1))
    
    row_3 = self.table_2_run(obl = Helper_Functions().orbit_at_time('obl', 1), long = Helper_Functions().orbit_extrema('long', 'min'), ecc = Helper_Functions().orbit_at_time('ecc', 1))
    row_4 = self.table_2_run(obl = Helper_Functions().orbit_at_time('obl', 1), long = Helper_Functions().orbit_extrema('long', 'max'), ecc = Helper_Functions().orbit_at_time('ecc', 1))
    
    row_5 = self.table_2_run(obl = Helper_Functions().orbit_at_time('obl', 1), long = Helper_Functions().orbit_at_time('long', 1), ecc = Helper_Functions().orbit_extrema('ecc', 'min'))
    row_6 = self.table_2_run(obl = Helper_Functions().orbit_at_time('obl', 1), long = Helper_Functions().orbit_at_time('long', 1), ecc = Helper_Functions().orbit_extrema('ecc', 'max'))


    control_row = [i for sublist in control_row for i in sublist]
    row_1 = [i for sublist in row_1 for i in sublist]
    row_2 = [i for sublist in row_2 for i in sublist]
    row_3 = [i for sublist in row_3 for i in sublist]
    row_4 = [i for sublist in row_4 for i in sublist]
    row_5 = [i for sublist in row_5 for i in sublist]
    row_6 = [i for sublist in row_6 for i in sublist]

    df = pd.DataFrame([control_row, row_1, row_2, row_3, row_4, row_5, row_6])
    df.columns = ["Atm CO2 ppm", "D", "Ins Type", "GMT", "Eq-Pole Temp Diff", "Annual Mean Ins", 'Heat Transport Max', 'Heat Tranport Max Location', "T grad Max", 'T grad Max Location', 'Wind Max', 'Wind Max Location', 'Max Ice', 'Min Ice', 'Mean Ice']
    df = Helper_Functions().format_data(df)
    
    sup_columns = pd.MultiIndex.from_tuples([('Inputs', df.columns[0]), ('Inputs', df.columns[1]), ('Inputs', df.columns[2]),
    ('Outputs', df.columns[3]), ('Outputs', df.columns[4]), ('Outputs', df.columns[5]), ('Outputs', df.columns[6]), ('Outputs', df.columns[7]), 
    ('Outputs', df.columns[8]), ('Outputs', df.columns[9]), ('Outputs', df.columns[10]), ('Outputs', df.columns[11]), ('Outputs', df.columns[12]), ('Outputs', df.columns[13]), ('Outputs', df.columns[14])])

    df.columns = sup_columns

    df.to_csv("table_2.csv")

    return None

  def sensitivity_runs(self, run_type = None, orb_comp = None): # climate sensitivity analysis on various model runs
    
    if run_type == "default": # sensitivity analysis compairing default insolation runs with different sea ice dynamics to an orbital contorl run with dynamic sea ice

      control_run_settings = {'moist':self.control_moist, 'albT':self.control_albT, 'seas':self.control_seas,'thermo':self.control_thermo}
      
      no_ice_settings = {'moist':1, 'albT':0, 'seas':2,'thermo':0}
      constant_ice_settings = {'moist':1, 'albT':4, 'seas':2,'thermo':0}
      dynamic_ice_settings = {'moist':1, 'albT':3, 'seas':2,'thermo':0}

      control_run_setting_display = ''.join(list(map(str, list(control_run_settings.values()))))
      no_ice_run_setting_display = ''.join(list(map(str, list(no_ice_settings.values()))))
      constant_ice_run_setting_display = ''.join(list(map(str, list(constant_ice_settings.values()))))
      dynamic_ice_run_setting_display = ''.join(list(map(str, list(dynamic_ice_settings.values()))))

      control_run_output = Model_Class.model(self, 1, self.config, self.Ti, CO2_ppm = None, D = None, F = 0, moist = control_run_settings['moist'], albT = control_run_settings['albT'], seas = control_run_settings['seas'], thermo = control_run_settings['thermo'])
      
      no_ice_output = Model_Class.model(self, 0, self.config, self.Ti, CO2_ppm = None, D = None, F = 0, moist = no_ice_settings['moist'], albT = no_ice_settings['albT'], seas = no_ice_settings['seas'], thermo = no_ice_settings['thermo'])
      constant_ice_output = Model_Class.model(self, 0, self.config, self.Ti, CO2_ppm = None, D = None, F = 0, moist = constant_ice_settings['moist'], albT = constant_ice_settings['albT'], seas = constant_ice_settings['seas'], thermo = constant_ice_settings['thermo'])
      dynamic_ice_output = Model_Class.model(self, 0, self.config, self.Ti, CO2_ppm = None, D = None, F = 0, moist = dynamic_ice_settings['moist'], albT = dynamic_ice_settings['albT'], seas = dynamic_ice_settings['seas'], thermo = dynamic_ice_settings['thermo'])
      full_run_output = Model_Class.model(self, 0, self.config, self.Ti, CO2_ppm = None, D = None, F = 0, moist = control_run_settings['moist'], albT = control_run_settings['albT'], seas = control_run_settings['seas'], thermo = control_run_settings['thermo'])

      control_inputs = ['default', 'N/A', 'N/A', 'N/A', 280, control_run_setting_display]
      no_ice_inputs = ['orbital', Helper_Functions().orbit_at_time('obl', 1), Helper_Functions().orbit_at_time('long', 1), Helper_Functions().orbit_at_time('ecc', 1), 280, no_ice_run_setting_display]
      constant_ice_inputs = ['orbital', Helper_Functions().orbit_at_time('obl', 1), Helper_Functions().orbit_at_time('long', 1), Helper_Functions().orbit_at_time('ecc', 1), 280, constant_ice_run_setting_display]
      dynamic_ice_inputs = ['orbital', Helper_Functions().orbit_at_time('obl', 1), Helper_Functions().orbit_at_time('long', 1), Helper_Functions().orbit_at_time('ecc', 1), 280, dynamic_ice_run_setting_display]
      full_run_inputs = ['orbital', Helper_Functions().orbit_at_time('obl', 1), Helper_Functions().orbit_at_time('long', 1), Helper_Functions().orbit_at_time('ecc', 1), 280, control_run_setting_display]

      all_inputs = control_inputs, no_ice_inputs, constant_ice_inputs, dynamic_ice_inputs, full_run_inputs

      return control_run_output, no_ice_output, constant_ice_output, dynamic_ice_output, full_run_output, all_inputs

    elif run_type == 'forcing': # sensitivity analysis compairing orbital runs with different orbital params, co2 concentrations, and seaice feedback to orbital control run

      CO2_forcing = {'On':280*2, 'Off':280}
      ice_feedback = {"On":[1, 'dynamic'], "Off":[4, 'static']}
      control_orbit = {'control_obl': Helper_Functions().orbit_at_time('obl',1), "control_long":Helper_Functions().orbit_at_time('long',1), 'control_ecc': Helper_Functions().orbit_at_time('ecc',1)}
      
    
      Orbit_forcing = {'On':[1, 'orbital'], 'Off':[1, 'orbital']}
      kyear = 'forced'
      if orb_comp == 'obl':
        orbital_perturbation_settings = {'obl':Helper_Functions().orbit_extrema('obl', 'max'), 'long':Helper_Functions().orbit_at_time('long',1), 'ecc':Helper_Functions().orbit_at_time('ecc',1)}
      elif orb_comp == 'long':
        orbital_perturbation_settings = {'obl':Helper_Functions().orbit_at_time('obl',1), 'long':Helper_Functions().orbit_extrema('long','max'), 'ecc':Helper_Functions().orbit_at_time('ecc',1)}
      elif orb_comp == 'ecc':
        orbital_perturbation_settings = {'obl':Helper_Functions().orbit_at_time('obl',1), 'long':Helper_Functions().orbit_at_time('long',1), 'ecc':Helper_Functions().orbit_extrema('ecc','max')}

      run_000 = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = self.control_CO2, D = None, F = 0, moist = self.control_moist, albT = self.control_albT, seas = self.control_seas, thermo = self.control_thermo,
      obl = control_orbit['control_obl'], long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)
      
      run_010 = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = CO2_forcing["On"], D = None, F = 0, moist = self.control_moist, albT = self.control_albT, seas = self.control_seas, thermo = self.control_thermo,
      obl = control_orbit['control_obl'], long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)
      
      run_100 = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = self.control_CO2, D = None, F = 0, moist = self.control_moist, albT = self.control_albT, seas = self.control_seas, thermo = self.control_thermo,
      obl = orbital_perturbation_settings['obl'], long = orbital_perturbation_settings['long'], ecc = orbital_perturbation_settings['ecc'], kyear = kyear)
      
      run_110 = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = CO2_forcing["On"], D = None, F = 0, moist = self.control_moist, albT = self.control_albT, seas = self.control_seas, thermo = self.control_thermo,
      obl = orbital_perturbation_settings['obl'], long = orbital_perturbation_settings['long'], ecc = orbital_perturbation_settings['ecc'], kyear = kyear)

      run_001 = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = self.control_CO2, D = None, F = 0, moist = self.control_moist, albT = ice_feedback["Off"][0], seas = self.control_seas, thermo = self.control_thermo,
      obl = control_orbit['control_obl'], long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)
      
      run_011 = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = CO2_forcing["On"], D = None, F = 0, moist = self.control_moist, albT = ice_feedback["Off"][0], seas = self.control_seas, thermo = self.control_thermo,
      obl = control_orbit['control_obl'], long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)
      
      run_101 = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = self.control_CO2, D = None, F = 0, moist = self.control_moist, albT = ice_feedback["Off"][0], seas = self.control_seas, thermo = self.control_thermo,
      obl = orbital_perturbation_settings['obl'], long = orbital_perturbation_settings['long'], ecc = orbital_perturbation_settings['ecc'], kyear = kyear)
      
      run_111 = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = CO2_forcing["On"], D = None, F = 0, moist = self.control_moist, albT = ice_feedback["Off"][0], seas = self.control_seas, thermo = self.control_thermo,
      obl = orbital_perturbation_settings['obl'], long = orbital_perturbation_settings['long'], ecc = orbital_perturbation_settings['ecc'], kyear = kyear)
      
      
      run_000_inputs = [self.control_label,self.control_label,self.control_label, self.control_label, self.control_label]
      run_010_inputs = [self.control_label,self.control_label,self.control_label, CO2_forcing['On'], self.control_label]
      run_100_inputs = [orbital_perturbation_settings['obl'],orbital_perturbation_settings['long'],orbital_perturbation_settings['ecc'], self.control_label, self.control_label]
      run_110_inputs = [orbital_perturbation_settings['obl'],orbital_perturbation_settings['long'],orbital_perturbation_settings['ecc'], CO2_forcing['On'], self.control_label]
      run_001_inputs = [self.control_label,self.control_label,self.control_label, self.control_label, ice_feedback['Off'][1]]
      run_011_inputs = [self.control_label,self.control_label,self.control_label, CO2_forcing['On'], ice_feedback['Off'][1]]
      run_101_inputs = [orbital_perturbation_settings['obl'],orbital_perturbation_settings['long'],orbital_perturbation_settings['ecc'], self.control_label, ice_feedback['Off'][1]]
      run_111_inputs = [orbital_perturbation_settings['obl'],orbital_perturbation_settings['long'],orbital_perturbation_settings['ecc'], CO2_forcing['On'], ice_feedback['Off'][1]]

      all_runs = run_000,run_010,run_100,run_110,run_001,run_011,run_101,run_111
      all_run_inputs = run_000_inputs,run_010_inputs,run_100_inputs,run_110_inputs,run_001_inputs,run_011_inputs,run_101_inputs,run_111_inputs

      return all_runs, all_run_inputs

    else: # sensitivity analysis compairing min orbital param with different sea ice dynamics to control run with max orbital param

      if orb_comp == 'obl':
        orbital_control_settings = {'obl':Helper_Functions().orbit_extrema('obl', 'min'), 'long':Helper_Functions().orbit_at_time('long',1), 'ecc':Helper_Functions().orbit_at_time('ecc',1)}
        orbital_perturbation_settings = {'obl':Helper_Functions().orbit_extrema('obl', 'max'), 'long':Helper_Functions().orbit_at_time('long',1), 'ecc':Helper_Functions().orbit_at_time('ecc',1)}
      elif orb_comp == 'long':
        orbital_control_settings = {'obl':Helper_Functions().orbit_at_time('obl',1), 'long':Helper_Functions().orbit_extrema('long','min'), 'ecc':Helper_Functions().orbit_at_time('ecc',1)}
        orbital_perturbation_settings = {'obl':Helper_Functions().orbit_at_time('obl',1), 'long':Helper_Functions().orbit_extrema('long','max'), 'ecc':Helper_Functions().orbit_at_time('ecc',1)}
      elif orb_comp == 'ecc':
        orbital_control_settings = {'obl':Helper_Functions().orbit_at_time('obl',1), 'long':Helper_Functions().orbit_at_time('long',1), 'ecc':Helper_Functions().orbit_extrema('ecc','min')}
        orbital_perturbation_settings = {'obl':Helper_Functions().orbit_at_time('obl',1), 'long':Helper_Functions().orbit_at_time('long',1), 'ecc':Helper_Functions().orbit_extrema('ecc','max')}

      control_run_settings = {'moist':1, 'albT':1, 'seas':2,'thermo':0}
      
      no_ice_settings = {'moist':1, 'albT':0, 'seas':2,'thermo':0}
      constant_ice_settings = {'moist':1, 'albT':4, 'seas':2,'thermo':0}
      dynamic_ice_settings = {'moist':1, 'albT':3, 'seas':2,'thermo':0}
      
      control_run_setting_display = ''.join(list(map(str, list(control_run_settings.values()))))
      no_ice_run_setting_display = ''.join(list(map(str, list(no_ice_settings.values()))))
      constant_ice_run_setting_display = ''.join(list(map(str, list(constant_ice_settings.values()))))
      dynamic_ice_run_setting_display = ''.join(list(map(str, list(dynamic_ice_settings.values()))))

      control_run_output = Model_Class.model(self, 1, self.config, self.Ti, CO2_ppm = None, D = None, F = 0, moist = control_run_settings['moist'], albT = control_run_settings['albT'], seas = control_run_settings['seas'], thermo = control_run_settings['thermo'],
      obl = orbital_control_settings['obl'], long = orbital_control_settings['long'], ecc = orbital_control_settings['ecc'], kyear = 'forced')
      
      no_ice_output = Model_Class.model(self, 1, self.config, self.Ti, CO2_ppm = None, D = None, F = 0, moist = no_ice_settings['moist'], albT = no_ice_settings['albT'], seas = no_ice_settings['seas'], thermo = no_ice_settings['thermo'],
      obl = orbital_perturbation_settings['obl'], long = orbital_perturbation_settings['long'], ecc = orbital_perturbation_settings['ecc'], kyear = 'forced')
      
      constant_ice_output = Model_Class.model(self, 1, self.config, self.Ti, CO2_ppm = None, D = None, F = 0, moist = constant_ice_settings['moist'], albT = constant_ice_settings['albT'], seas = constant_ice_settings['seas'], thermo = constant_ice_settings['thermo'],
      obl = orbital_perturbation_settings['obl'], long = orbital_perturbation_settings['long'], ecc = orbital_perturbation_settings['ecc'], kyear = 'forced')
      
      dynamic_ice_output = Model_Class.model(self, 1, self.config, self.Ti, CO2_ppm = None, D = None, F = 0, moist = dynamic_ice_settings['moist'], albT = dynamic_ice_settings['albT'], seas = dynamic_ice_settings['seas'], thermo = dynamic_ice_settings['thermo'],
      obl = orbital_perturbation_settings['obl'], long = orbital_perturbation_settings['long'], ecc = orbital_perturbation_settings['ecc'], kyear = 'forced')
      
      full_run_output = Model_Class.model(self, 1, self.config, self.Ti, CO2_ppm = None, D = None, F = 0, moist = control_run_settings['moist'], albT = control_run_settings['albT'], seas = control_run_settings['seas'], thermo = control_run_settings['thermo'],
      obl = orbital_perturbation_settings['obl'], long = orbital_perturbation_settings['long'], ecc = orbital_perturbation_settings['ecc'], kyear = 'forced')

      control_inputs = ['orbital', orbital_control_settings['obl'], orbital_control_settings['long'], orbital_control_settings['ecc'], 280, control_run_setting_display]
      no_ice_inputs = ['orbital', orbital_perturbation_settings['obl'], orbital_perturbation_settings['long'], orbital_perturbation_settings['ecc'], 280, no_ice_run_setting_display]
      constant_ice_inputs = ['orbital', orbital_perturbation_settings['obl'], orbital_perturbation_settings['long'], orbital_perturbation_settings['ecc'], 280, constant_ice_run_setting_display]
      dynamic_ice_inputs = ['orbital', orbital_perturbation_settings['obl'], orbital_perturbation_settings['long'], orbital_perturbation_settings['ecc'], 280, dynamic_ice_run_setting_display]
      full_run_inputs = ['orbital', orbital_perturbation_settings['obl'], orbital_perturbation_settings['long'], orbital_perturbation_settings['ecc'], 280, control_run_setting_display]
      all_inputs = control_inputs, no_ice_inputs, constant_ice_inputs, dynamic_ice_inputs, full_run_inputs

      return control_run_output, no_ice_output, constant_ice_output, dynamic_ice_output, full_run_output, all_inputs

  def sensitivity_sythesis(self, run_type = None, orb_comp = None): # preparing sensitivity run output for tabulation

      if run_type == 'forcing':

        all_runs, all_run_inputs = self.sensitivity_runs(run_type = run_type, orb_comp = orb_comp)
        run_000,run_010,run_100,run_110,run_001,run_011,run_101,run_111 = all_runs

        run_000_v_run_010 = list(Helper_Functions().sensitivity_analysis(run_000,run_010))
        run_000_v_run_100 = list(Helper_Functions().sensitivity_analysis(run_000,run_100))
        run_000_v_run_110 = list(Helper_Functions().sensitivity_analysis(run_000,run_110))
        run_000_v_run_001 = list(Helper_Functions().sensitivity_analysis(run_000,run_001))
        run_000_v_run_011 = list(Helper_Functions().sensitivity_analysis(run_000,run_011))
        run_000_v_run_101 = list(Helper_Functions().sensitivity_analysis(run_000,run_101))
        run_000_v_run_111 = list(Helper_Functions().sensitivity_analysis(run_000,run_111))

        all_comparisons = run_000_v_run_010,run_000_v_run_100,run_000_v_run_110,run_000_v_run_001,run_000_v_run_011,run_000_v_run_101,run_000_v_run_111 

        return all_comparisons, all_run_inputs

      else:

        control_run_output, no_ice_output, constant_ice_output, dynamic_ice_output, full_run_output, all_inputs = self.sensitivity_runs(run_type = run_type, orb_comp = orb_comp)

        control_v_no_ice = Helper_Functions().sensitivity_analysis(control_run_output, no_ice_output)
        control_v_constant_ice = Helper_Functions().sensitivity_analysis(control_run_output, constant_ice_output)
        control_v_dynamic_ice = Helper_Functions().sensitivity_analysis(control_run_output, dynamic_ice_output)
        control_v_full = Helper_Functions().sensitivity_analysis(control_run_output, full_run_output)

        control_v_no_ice = list(control_v_no_ice)
        control_v_constant_ice = list(control_v_constant_ice)
        control_v_dynamic_ice = list(control_v_dynamic_ice)
        control_v_full = list(control_v_full)

        return control_v_no_ice, control_v_constant_ice, control_v_dynamic_ice, control_v_full, all_inputs

  def generate_sensitivity_table(self, run_type = None, orb_comp = None): # generates and tabulates sensitivity table

    if run_type == 'forcing':

      all_comparisons, all_run_inputs = self.sensitivity_sythesis(run_type = run_type, orb_comp = orb_comp)
      run_000_v_run_010,run_000_v_run_100,run_000_v_run_110,run_000_v_run_001,run_000_v_run_011,run_000_v_run_101,run_000_v_run_111 = all_comparisons
      
      run_000_inputs,run_010_inputs,run_100_inputs,run_110_inputs,run_001_inputs,run_011_inputs,run_101_inputs,run_111_inputs = all_run_inputs
      empty = [''] * len(run_000_v_run_010)

      row_1 = run_000_inputs + run_000_v_run_010
      row_2 = run_010_inputs + empty

      row_3 = run_000_inputs + run_000_v_run_100
      row_4 = run_100_inputs + empty

      row_5 = run_000_inputs + run_000_v_run_110
      row_6 = run_110_inputs + empty

      row_7 = run_000_inputs + run_000_v_run_001
      row_8 = run_001_inputs + empty

      row_9 = run_000_inputs + run_000_v_run_011
      row_10 = run_011_inputs + empty

      row_11 = run_000_inputs + run_000_v_run_101
      row_12 = run_101_inputs + empty

      row_13 = run_000_inputs + run_000_v_run_111
      row_14 = run_111_inputs + empty

      df = pd.DataFrame([row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8, row_9, row_10, row_11, row_12, row_13, row_14])
      
      df.columns = ['Obliquity','Long Peri', "Ecc", 'CO2 Forcing', 'Albedo Feedback','delta_I', 'delta_OLR', 'delta_FCO2', 'delta_GMT', 'calculated_delta_GMT', 'unnacounted_change', 'lambda ice']

      df = Helper_Functions().format_data(df)
      df = df.round(2)
      df.to_csv("table_sensitivity.csv")

      return None

    else:

      control_v_no_ice, control_v_constant_ice, control_v_dynamic_ice, control_v_full, all_inputs = self.sensitivity_sythesis(run_type = run_type, orb_comp = orb_comp)
      control_inputs, no_ice_inputs, constant_ice_inputs, dynamic_ice_inputs, full_run_inputs = all_inputs
      empty = [''] * len(control_v_no_ice)
          
      row_1 = control_inputs + control_v_no_ice
      row_2 = no_ice_inputs + empty

      row_3 = control_inputs + control_v_constant_ice
      row_4 = constant_ice_inputs + empty

      row_5 = control_inputs + control_v_dynamic_ice
      row_6 = dynamic_ice_inputs + empty

      row_7 = control_inputs + control_v_full
      row_8 = full_run_inputs + empty

      df = pd.DataFrame([row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8])
      df.columns = ['Ins Type', 'Obl', 'long', 'ecc', 'CO2 ppm', 'settings','delta_I', 'delta_ASR', 'delta_FCO2', 'delta_GMT', 'calculated_delta_GMT', 'unnacounted_change', 'lambda ice']
      df = Helper_Functions().format_data(df)
      df = df.round(2)
      df.to_csv("table_sensitivity.csv")

      return None

  def forcing_decomp_runs(self): # generates model output for min/max orbital forcings, x2 and 0.5x co2, and control run

    CO2_forcing = {'Double':[560, 'Double'], 'Half':[140, 'Half']}
    kyear = 'forced'
    control_orbit = {'control_obl': Helper_Functions().orbit_at_time('obl',1), "control_long":Helper_Functions().orbit_at_time('long',1), 'control_ecc': Helper_Functions().orbit_at_time('ecc',1)}
    obl_settings = {'max_obl': [Helper_Functions().orbit_extrema('obl', 'max'), 'Max'],'min_obl': [Helper_Functions().orbit_extrema('obl', 'min'), 'Min'] }
    ecc_settings = {'max_ecc': [Helper_Functions().orbit_extrema('ecc', 'max'), 'Max'],'min_ecc': [Helper_Functions().orbit_extrema('ecc', 'min'), 'Min'] }
    long_settings = {'max_long': [0,'Max'],'min_long': [180, 'Min'] } # orbit_extrema does not accuratly capture greatest forcings difference for longitude or perihelion due to its angular configuration


    control_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = self.control_CO2, D = None, F = 0, moist = self.control_moist, albT = self.control_albT, seas = self.control_seas, thermo = self.control_thermo,
      obl = control_orbit['control_obl'], long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)
      
    oblmax_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = self.control_CO2, D = None, F = 0, moist = self.control_moist, albT = self.control_albT, seas = self.control_seas, thermo = self.control_thermo,
      obl = obl_settings['max_obl'][0], long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)

    oblmin_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = self.control_CO2, D = None, F = 0, moist = self.control_moist, albT = self.control_albT, seas = self.control_seas, thermo = self.control_thermo,
      obl = obl_settings['min_obl'][0], long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)

    longmax_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = self.control_CO2, D = None, F = 0, moist = self.control_moist, albT = self.control_albT, seas = self.control_seas, thermo = self.control_thermo,
      obl = control_orbit['control_obl'], long = long_settings['max_long'][0], ecc = control_orbit['control_ecc'], kyear = kyear)

    longmin_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = self.control_CO2, D = None, F = 0, moist = self.control_moist, albT = self.control_albT, seas = self.control_seas, thermo = self.control_thermo,
      obl = control_orbit['control_obl'], long = long_settings['min_long'][0], ecc = control_orbit['control_ecc'], kyear = kyear)

    eccmax_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = self.control_CO2, D = None, F = 0, moist = self.control_moist, albT = self.control_albT, seas = self.control_seas, thermo = self.control_thermo,
      obl = control_orbit['control_obl'], long = control_orbit['control_long'], ecc = ecc_settings['max_ecc'][0], kyear = kyear)

    eccmin_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = self.control_CO2, D = None, F = 0, moist = self.control_moist, albT = self.control_albT, seas = self.control_seas, thermo = self.control_thermo,
      obl = control_orbit['control_obl'], long = control_orbit['control_long'], ecc = ecc_settings['min_ecc'][0], kyear = kyear)

    co2double_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = CO2_forcing['Double'][0], D = None, F = 0, moist = self.control_moist, albT = self.control_albT, seas = self.control_seas, thermo = self.control_thermo,
      obl = control_orbit['control_obl'], long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)

    co2half_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = CO2_forcing['Half'][0], D = None, F = 0, moist = self.control_moist, albT = self.control_albT, seas = self.control_seas, thermo = self.control_thermo,
      obl = control_orbit['control_obl'], long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)

    control_run_inputs = [self.control_label, self.control_label, self.control_label, self.control_label]
    oblmax_run_inputs = [obl_settings['max_obl'][1], self.control_label, self.control_label, self.control_label]
    oblmin_run_inputs = [obl_settings['min_obl'][1], self.control_label, self.control_label, self.control_label]
    longmax_run_inputs = [self.control_label, long_settings['max_long'][1], self.control_label, self.control_label]
    longmin_run_inputs = [self.control_label, long_settings['min_long'][1], self.control_label, self.control_label]
    eccmax_run_inputs = [self.control_label, self.control_label, ecc_settings['max_ecc'][1], self.control_label]
    eccmin_run_inputs = [self.control_label, self.control_label, ecc_settings['min_ecc'][1], self.control_label]
    co2double_run_inputs = [self.control_label, self.control_label, self.control_label, CO2_forcing['Double'][1]]
    co2half_run_inputs = [self.control_label, self.control_label, self.control_label, CO2_forcing['Half'][1]]

    all_runs = control_run,oblmax_run,oblmin_run,longmax_run,longmin_run,eccmax_run,eccmin_run,co2double_run,co2half_run
    all_run_inputs = control_run_inputs,oblmax_run_inputs,oblmin_run_inputs,longmax_run_inputs,longmin_run_inputs,eccmax_run_inputs,eccmin_run_inputs,co2double_run_inputs,co2half_run_inputs
    
    return all_runs, all_run_inputs

  def forcing_decomp_sensitivity_synthesis(self):  # preforms sensitivity analysis on forcing runs vs. control run

    all_runs, all_run_inputs = self.forcing_decomp_runs()
    control_run,oblmax_run,oblmin_run,longmax_run,longmin_run,eccmax_run,eccmin_run,co2double_run,co2half_run = all_runs

    control_v_oblmax = list(Helper_Functions().sensitivity_analysis(control_run,oblmax_run))
    control_v_oblmin = list(Helper_Functions().sensitivity_analysis(control_run,oblmin_run))
    control_v_longmax = list(Helper_Functions().sensitivity_analysis(control_run,longmax_run))
    control_v_longmin = list(Helper_Functions().sensitivity_analysis(control_run,longmin_run))
    control_v_eccmax = list(Helper_Functions().sensitivity_analysis(control_run,eccmax_run))
    control_v_eccmin = list(Helper_Functions().sensitivity_analysis(control_run,eccmin_run))
    control_v_co2double = list(Helper_Functions().sensitivity_analysis(control_run,co2double_run))
    control_v_co2half = list(Helper_Functions().sensitivity_analysis(control_run,co2half_run))

    all_comparisons = control_v_oblmax,control_v_oblmin,control_v_longmax,control_v_longmin,control_v_eccmax,control_v_eccmin,control_v_co2double,control_v_co2half

    return all_comparisons, all_run_inputs

  def generate_decomp_table(self): # generates and tabulates sensitivity analysis of forcing decomposition table

    all_comparisons, all_run_inputs = self.forcing_decomp_sensitivity_synthesis()
    control_v_oblmax,control_v_oblmin,control_v_longmax,control_v_longmin,control_v_eccmax,control_v_eccmin,control_v_co2double,control_v_co2half = all_comparisons
    control_run_inputs,oblmax_run_inputs,oblmin_run_inputs,longmax_run_inputs,longmin_run_inputs,eccmax_run_inputs,eccmin_run_inputs,co2double_run_inputs,co2half_run_inputs = all_run_inputs

    row_1 = oblmax_run_inputs + control_v_oblmax
    row_2 = oblmin_run_inputs + control_v_oblmin
    row_3 = longmax_run_inputs + control_v_longmax
    row_4 = longmin_run_inputs + control_v_longmin
    row_5 = eccmax_run_inputs + control_v_eccmax
    row_6 = eccmin_run_inputs + control_v_eccmin
    row_7 = co2double_run_inputs + control_v_co2double
    row_8 = co2half_run_inputs + control_v_co2half

    df = pd.DataFrame([row_1, row_2, row_3, row_4, row_5, row_6, row_7, row_8])

    df.columns = ['Obliquity','Long Peri', "Ecc", 'CO2 Forcing','delta_I', 'delta_OLR', 'delta_FCO2', 'delta_GMT', 'calculated_delta_GMT', 'unnacounted_change', 'lambda ice']
    
    df = Helper_Functions().format_data(df)
    df = df.round(2)
    df.to_csv("forcing_decomp_table.csv")

    return None

  def orbit_and_CO2_suite(self, x_type = None): # tracks EBM outputs over range of orbital or co2 forcings

    control_orbit = {'control_obl': Helper_Functions().orbit_at_time('obl',1), "control_long":Helper_Functions().orbit_at_time('long',1), 'control_ecc': Helper_Functions().orbit_at_time('ecc',1)}
    kyear = 'forced'

    CO2_val = 280

    control_output = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = self.control_CO2, D = None, F = 0, moist = self.control_moist, albT = 1, seas = self.control_seas, thermo = self.control_thermo,
        obl = control_orbit['control_obl'], long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)

    iterations = 20
    if x_type == 'CO2':
      x_range = np.linspace(100,600,iterations)
    elif x_type == 'obl':
      x_range = np.linspace(Helper_Functions().orbit_extrema('obl','min'),Helper_Functions().orbit_extrema('obl','max'),iterations)

    model_output_list = []
    if x_type == 'CO2':
      for i in x_range:
        output_of_a_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = i, D = None, F = 0, moist = self.control_moist, albT = self.control_albT, seas = self.control_seas, thermo = self.control_thermo,
        obl = control_orbit['control_obl'], long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)
        model_output_list.append(output_of_a_run)
    elif x_type == 'obl':
      for i in x_range:
        output_of_a_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = CO2_val, D = None, F = 0, moist = self.control_moist, albT = 3, seas = self.control_seas, thermo = self.control_thermo,
        obl = i, long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)
        model_output_list.append(output_of_a_run)

    GMT_of_runs = []
    for i in range(0,iterations):
      GMT_single_run = model_output_list[i][2]
      GMT_single_run = np.mean(GMT_single_run)
      GMT_of_runs.append(GMT_single_run)

    Shemi_icelines_of_runs = []
    for i in range(0,iterations):
      icelines_of_single_run = model_output_list[i][14][2]
      Shemi_icelines_of_runs.append(icelines_of_single_run)

    S65_ins = []
    for i in range(0,iterations):
      S65_ins_single_run = model_output_list[i][5][4]
      S65_ins.append(S65_ins_single_run)

    S65_temp = []
    for i in range(0,iterations):
      S65_temp_single_run = model_output_list[i][5][4]
      S65_temp.append(S65_temp_single_run)

    dASRdalpha = []
    for i in range(0,iterations):
      dASRdalpha_single_run = control_output[5]*(model_output_list[i][15] - control_output[15])
      dASRdalpha_single_run = np.mean(dASRdalpha_single_run)
      dASRdalpha.append(dASRdalpha_single_run)

    geo_wind_max_lat = []
    for i in range(0,iterations):
      geo_winds_single_run = model_output_list[i][13]
      lat_of_max_wind_single_run = Helper_Functions().find_lat_of_value(experiment().config,geo_winds_single_run,self.lat_deg,np.max(geo_winds_single_run[0:(int(len(geo_winds_single_run)/2))]))
      geo_wind_max_lat.append(lat_of_max_wind_single_run)

    t_grad_max_lat = []
    for i in range(0,iterations):
      t_grad_max_single_run = model_output_list[i][11]
      lat_of_max_t_grad_sing_run = Helper_Functions().find_lat_of_value(experiment().config,t_grad_max_single_run,self.lat_deg,np.max(t_grad_max_single_run[0:(int(len(t_grad_max_single_run)/2))]))
      t_grad_max_lat.append(lat_of_max_t_grad_sing_run)

    WSC_max_lat = []
    for i in range(0,iterations):
      WSC_max_single_run = model_output_list[i][18]
      lat_of_max_WSC_sing_run = Helper_Functions().find_lat_of_value(experiment().config,WSC_max_single_run,self.lat_deg,np.max(WSC_max_single_run[0:(int(len(WSC_max_single_run)/2))]))
      WSC_max_lat.append(lat_of_max_WSC_sing_run)

    return x_range, GMT_of_runs, Shemi_icelines_of_runs, S65_ins, S65_temp, dASRdalpha, geo_wind_max_lat, t_grad_max_lat, WSC_max_lat, CO2_val

  def obl_sensitivity_analysis(self, albT = 'On', save = None): # tracks EBM outputs for min/max obliquity forcing over range of co2 vals

    control_orbit = {'control_obl': Helper_Functions().orbit_at_time('obl',1), "control_long":Helper_Functions().orbit_at_time('long',1), 'control_ecc': Helper_Functions().orbit_at_time('ecc',1)}
    dobl = Helper_Functions().orbit_extrema('obl', 'max') - Helper_Functions().orbit_extrema('obl', 'min')
    kyear = 'forced'

    iterations = 100 # resolution
    
    x_range = np.linspace(100,600,iterations) # co2 forcing range

    max_obl_output = []
    min_obl_output = []
    for i in x_range:   
      if albT == 'On':
        max_obl_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = i, D = None, F = 0, moist = self.control_moist, albT = 3, seas = self.control_seas, thermo = self.control_thermo,
            obl = Helper_Functions().orbit_extrema('obl', "max"), long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)
        max_obl_output.append(max_obl_run)
        min_obl_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = i, D = None, F = 0, moist = self.control_moist, albT = 3, seas = self.control_seas, thermo = self.control_thermo,
          obl = Helper_Functions().orbit_extrema('obl', "min"), long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)
        min_obl_output.append(min_obl_run)
      elif albT == 'Off':
        max_obl_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = i, D = None, F = 0, moist = self.control_moist, albT = 4, seas = self.control_seas, thermo = self.control_thermo,
            obl = Helper_Functions().orbit_extrema('obl', "max"), long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)
        max_obl_output.append(max_obl_run)
        min_obl_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = i, D = None, F = 0, moist = self.control_moist, albT = 4, seas = self.control_seas, thermo = self.control_thermo,
          obl = Helper_Functions().orbit_extrema('obl', "min"), long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)
        min_obl_output.append(min_obl_run)
      elif albT == 'Static':
        co2_albT_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = i, D = None, F = 0, moist = self.control_moist, albT = 3, seas = 1, thermo = self.control_thermo,
            obl = control_orbit['control_obl'], long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear)
        
        co2_Albfin = co2_albT_run[15]
        
        max_obl_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = i, D = None, F = 0, moist = self.control_moist, albT = 5, seas = self.control_seas, thermo = self.control_thermo,
            obl = Helper_Functions().orbit_extrema('obl', "max"), long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear, Albfin_co2 = co2_Albfin)
        max_obl_output.append(max_obl_run)

        min_obl_run = Model_Class.model(self, self.control_orbit_type, self.config, self.Ti, CO2_ppm = i, D = None, F = 0, moist = self.control_moist, albT = 5, seas = self.control_seas, thermo = self.control_thermo,
          obl = Helper_Functions().orbit_extrema('obl', "min"), long = control_orbit['control_long'], ecc = control_orbit['control_ecc'], kyear = kyear, Albfin_co2 = co2_Albfin)
        min_obl_output.append(min_obl_run)
    
    if save != None:

      with open('obl_min_CO2runs_output.pkl', 'wb') as f:
              pickle.dump(min_obl_output, f)
      with open('obl_max_CO2runs_output.pkl', 'wb') as f:
              pickle.dump(max_obl_output, f)


    dASRdalpha = []
    for i in range(0,iterations):
      dASRdalpha_single_run = min_obl_output[i][5]*(max_obl_output[i][15] - min_obl_output[i][15])
      dASRdalpha_single_run = np.mean(dASRdalpha_single_run)
      dASRdalpha.append(dASRdalpha_single_run)

    dASR = []
    for i in range(0,iterations):
      dASR_single_run = max_obl_output[i][4] - min_obl_output[i][4]
      dASR_single_run = np.mean(dASR_single_run)
      dASR.append(dASR_single_run)

    dGMT = []
    for i in range(0,iterations):
      dGMT_single_run = max_obl_output[i][2] - min_obl_output[i][2]
      dGMT_single_run = np.mean(dGMT_single_run)
      dGMT.append(dGMT_single_run)

    obl_sensitivity_GMT = []
    for i in range(0,iterations):
      obl_sense_singel_run = dGMT[i] / dobl
      obl_sensitivity_GMT.append(obl_sense_singel_run)

    obl_sensitivity_ASR = []
    for i in range(0,iterations):
      obl_sense_singel_run = dASR[i] / dobl
      obl_sensitivity_ASR.append(obl_sense_singel_run)
    
    GMT_obl_max = []
    for i in range(0,iterations):
      GMT_obl_max_single_run = max_obl_output[i][2]
      GMT_obl_max_single_run = np.mean(GMT_obl_max_single_run)
      GMT_obl_max.append(GMT_obl_max_single_run)

    GMT_obl_min = []
    for i in range(0,iterations):
      GMT_obl_min_single_run = min_obl_output[i][2]
      GMT_obl_min_single_run = np.mean(GMT_obl_min_single_run)
      GMT_obl_min.append(GMT_obl_min_single_run)
  
    if True == True: # ICELINES
      mean_icelines_obl_max = []
      for i in range(0,iterations):
        icelines_obl_max_single_run = max_obl_output[i][14][2]
        icelines_obl_max_single_run = np.mean(icelines_obl_max_single_run)
        mean_icelines_obl_max.append(icelines_obl_max_single_run)

      mean_icelines_obl_min = []
      for i in range(0,iterations):
        icelines_obl_min_single_run = min_obl_output[i][14][2]
        icelines_obl_min_single_run = np.mean(icelines_obl_min_single_run)
        mean_icelines_obl_min.append(icelines_obl_min_single_run)

      max_icelines_obl_max = []
      for i in range(0,iterations):
        icelines_obl_max_single_run = max_obl_output[i][14][0]
        icelines_obl_max_single_run = np.mean(icelines_obl_max_single_run)
        max_icelines_obl_max.append(icelines_obl_max_single_run)

      max_icelines_obl_min = []
      for i in range(0,iterations):
        icelines_obl_min_single_run = min_obl_output[i][14][0]
        icelines_obl_min_single_run = np.mean(icelines_obl_min_single_run)
        max_icelines_obl_min.append(icelines_obl_min_single_run)

      min_icelines_obl_max = []
      for i in range(0,iterations):
        icelines_obl_max_single_run = max_obl_output[i][14][1]
        icelines_obl_max_single_run = np.mean(icelines_obl_max_single_run)
        min_icelines_obl_max.append(icelines_obl_max_single_run)

      min_icelines_obl_min = []
      for i in range(0,iterations):
        icelines_obl_min_single_run = min_obl_output[i][14][1]
        icelines_obl_min_single_run = np.mean(icelines_obl_min_single_run)
        min_icelines_obl_min.append(icelines_obl_min_single_run)

    if True == True: # TGRADS
      EQP_obl_max = []
      for i in range(0,iterations):
        EQP_obl_max_single_run = np.ptp(max_obl_output[i][2])
        EQP_obl_max.append(EQP_obl_max_single_run)

      EQP_obl_min = []
      for i in range(0,iterations):
        EQP_obl_min_single_run = np.ptp(min_obl_output[i][2])
        EQP_obl_min.append(EQP_obl_min_single_run)

      Tgrad_30_40_min = []
      for i in range(0,iterations):
        Tgrad_30_40_min_single_run = Helper_Functions().find_value_at_lat(self.config, min_obl_output[i][2], -30) - Helper_Functions().find_value_at_lat(self.config, min_obl_output[i][2], -40)
        Tgrad_30_40_min.append(Tgrad_30_40_min_single_run)

      Tgrad_30_40_max = []
      for i in range(0,iterations):
        Tgrad_30_40_max_single_run = Helper_Functions().find_value_at_lat(self.config, max_obl_output[i][2], -30) - Helper_Functions().find_value_at_lat(self.config, max_obl_output[i][2], -40)
        Tgrad_30_40_max.append(Tgrad_30_40_max_single_run)

      Tgrad_40_50_min = []
      for i in range(0,iterations):
        Tgrad_40_50_min_single_run = Helper_Functions().find_value_at_lat(self.config, min_obl_output[i][2], -40) - Helper_Functions().find_value_at_lat(self.config, min_obl_output[i][2], -50)
        Tgrad_40_50_min.append(Tgrad_40_50_min_single_run)

      Tgrad_40_50_max = []
      for i in range(0,iterations):
        Tgrad_40_50_max_single_run = Helper_Functions().find_value_at_lat(self.config, max_obl_output[i][2], -40) - Helper_Functions().find_value_at_lat(self.config, max_obl_output[i][2], -50)
        Tgrad_40_50_max.append(Tgrad_40_50_max_single_run)

      Tgrad_50_60_min = []
      for i in range(0,iterations):
        Tgrad_50_60_min_single_run = Helper_Functions().find_value_at_lat(self.config, min_obl_output[i][2], -50) - Helper_Functions().find_value_at_lat(self.config, min_obl_output[i][2], -60)
        Tgrad_50_60_min.append(Tgrad_50_60_min_single_run)

      Tgrad_50_60_max = []
      for i in range(0,iterations):
        Tgrad_50_60_max_single_run = Helper_Functions().find_value_at_lat(self.config, max_obl_output[i][2], -50) - Helper_Functions().find_value_at_lat(self.config, max_obl_output[i][2], -60)
        Tgrad_50_60_max.append(Tgrad_50_60_max_single_run)
      
      Tgrad_60_70_min = []
      for i in range(0,iterations):
        Tgrad_60_70_min_single_run = Helper_Functions().find_value_at_lat(self.config, min_obl_output[i][2], -60) - Helper_Functions().find_value_at_lat(self.config, min_obl_output[i][2], -70)
        Tgrad_60_70_min.append(Tgrad_60_70_min_single_run)

      Tgrad_60_70_max = []
      for i in range(0,iterations):
        Tgrad_60_70_max_single_run = Helper_Functions().find_value_at_lat(self.config, max_obl_output[i][2], -60) - Helper_Functions().find_value_at_lat(self.config, max_obl_output[i][2], -70)
        Tgrad_60_70_max.append(Tgrad_60_70_max_single_run)
    
    full_tgrad_max_obl = []
    for i in range(0,iterations):
      t_grad_max_obl_single_run = max_obl_output[i][11]
      full_tgrad_max_obl.append(t_grad_max_obl_single_run)

    full_tgrad_min_obl = []
    for i in range(0,iterations):
      t_grad_min_obl_single_run = min_obl_output[i][11]
      full_tgrad_min_obl.append(t_grad_min_obl_single_run)


    #packing some outputs for org
    packed_Tgrad = Tgrad_30_40_min, Tgrad_30_40_max, Tgrad_40_50_min, Tgrad_40_50_max, Tgrad_50_60_min, Tgrad_50_60_max, Tgrad_60_70_min, Tgrad_60_70_max
    packed_icelines =  mean_icelines_obl_max, mean_icelines_obl_min, max_icelines_obl_max, max_icelines_obl_min, min_icelines_obl_max, min_icelines_obl_min
    
    return x_range, obl_sensitivity_GMT, obl_sensitivity_ASR, GMT_obl_max, GMT_obl_min, EQP_obl_max, EQP_obl_min, packed_icelines , packed_Tgrad, full_tgrad_max_obl, full_tgrad_min_obl

class Orbital_Insolation(): # computes insolation values from orbital parameters 

  def __init__(self, from_ktime=2000, until_ktime = 0):

    
    self.from_ktime = from_ktime
    self.until_ktime = until_ktime
    self.days_per_year_const = 365.2422 #days
    self.day_type = 1

    if from_ktime < 0 or until_ktime < 0:
      raise ValueError('kyear BP must be positive value')

  def get_orbit(self, control_obliquity = None): #retrieves and packs data from Laskar 2004

      kyear0 = OrbitalTable['kyear']
      ecc0 = OrbitalTable['ecc']
      long_peri = OrbitalTable['long_peri']
      obliquity = OrbitalTable['obliquity']
      precession = OrbitalTable['precession']

      kyear0 = kyear0.to_numpy()
      ecc0 = ecc0.to_numpy()
      long_peri = long_peri.to_numpy()
      obliquity = obliquity.to_numpy()
      precession = precession.to_numpy()
      
      if isinstance(self.until_ktime, int) or isinstance(self.until_ktime, (int, np.integer)):
        kyear0 = kyear0[self.until_ktime:self.from_ktime]
        ecc0 = ecc0[self.until_ktime:self.from_ktime]
        long_peri = long_peri[self.until_ktime:self.from_ktime]
        obliquity = obliquity[self.until_ktime:self.from_ktime]
        precession = precession[self.until_ktime:self.from_ktime]
      
      elif isinstance(self.until_ktime, float):
        ecc0 = np.interp(self.until_ktime,kyear0,ecc0)
        long_peri = np.interp(self.until_ktime,kyear0,long_peri)
        obliquity = np.interp(self.until_ktime,kyear0,obliquity)
        precession = np.interp(self.until_ktime,kyear0,precession)
        kyear0 = self.until_ktime

      return kyear0,ecc0,long_peri,obliquity,precession

  def solar_long(self, day, days_per_year = None, obl = None, long = None, ecc = None, kyear = None): #https://climlab.readthedocs.io/en/latest/_modules/climlab/solar/insolation.html#daily_insolation

    if type(kyear) == type(None):

      kyear0,ecc0,long_peri,obliquity,precession = self.get_orbit('on')
      

    elif type(kyear) == type('string'):

      obliquity = obl
      long_peri = long
      ecc0 = ecc
    
    if days_per_year is None:
        days_per_year = self.days_per_year_const
    ecc = ecc0
    
    long_peri_rad = deg2rad(long_peri)
    delta_lambda = (day - 80.) * 2*pi/days_per_year
    beta = sqrt(1 - ecc**2)

    lambda_long_m = -2*((ecc/2 + (ecc**3)/8 ) * (1+beta) * sin(-long_peri_rad) -
        (ecc**2)/4 * (1/2 + beta) * sin(-2*long_peri_rad) + (ecc**3)/8 *
        (1/3 + beta) * sin(-3*long_peri_rad)) + delta_lambda
    lambda_long = ( lambda_long_m + (2*ecc - (ecc**3)/4)*sin(lambda_long_m - long_peri_rad) +
        (5/4)*(ecc**2) * sin(2*(lambda_long_m - long_peri_rad)) + (13/12)*(ecc**3)
        * sin(3*(lambda_long_m - long_peri_rad)) )
    return lambda_long
  
  def insolation(self, day, lat = 0, obl = None, long = None, ecc = None, kyear = None):  #returns insolation based on day and lat #https://climlab.readthedocs.io/en/latest/_modules/climlab/solar/insolation.html#daily_insolation

    S0 = 1365.2 # W/m^2
    
    oldsettings = np.seterr(invalid='ignore')

    if type(kyear) != type('string'):

      kyear0,ecc0,long_peri,obliquity,precession = self.get_orbit('on')

    elif type(kyear) == type('string'):

      obliquity = obl
      long_peri = long
      ecc0 = ecc

    lat_is_xarray = True
    day_is_xarray = True
    
    if type(lat) is np.ndarray:
          lat_is_xarray = False
          lat = xr.DataArray(lat, coords=[lat], dims=['lat'])
    if type(day) is np.ndarray:
          day_is_xarray = False
          day = xr.DataArray(day, coords=[day], dims=['day'])

    phi = deg2rad(lat)
    
    if self.day_type==1: # calendar days
          lambda_long = self.solar_long(day, obl = obl, long = long, ecc = ecc, kyear = kyear)
    elif self.day_type==2: #solar longitude (1-360) is specified in input, no need to convert days to longitude
          lambda_long = deg2rad(day)
    else:
          raise ValueError('Invalid day_type.')

    # Compute declination angle of the sun
    delta = arcsin(sin(deg2rad(obliquity)) * sin(lambda_long))

    Ho = xr.where( abs(delta)-pi/2+abs(phi) < 0., # there is sunset/sunrise
                arccos(-tan(phi)*tan(delta)),
                # otherwise figure out if it's all night or all day
                xr.where((phi*delta)>0., pi, 0.) )
    
    coszen = Ho*sin(phi)*sin(delta) + cos(phi)*cos(delta)*sin(Ho)

    Fsw = (S0/pi) *( (1+ecc0*cos(lambda_long -deg2rad(long_peri)))**2 / (1-ecc0**2)**2 * coszen)

    return Fsw

  def avg_insolation(self, grid, lat_array = 0, lat = 0, from_lat = None, to_lat = None, from_month = None, to_month = None, obl = None, long = None, ecc = None, kyear = None, days = None): #returns annual insolation at lat

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

    # converting x to latitude for insolation method input
    x = np.rad2deg(np.arcsin(x))
    
    if lat_array == 0:

      avg_day = []   
      
      for f in range(0,int(self.days_per_year_const)):
        
        avg_each_day_of_lat = self.insolation(f, lat,  obl = obl, long = long, ecc = ecc, kyear = kyear)
        
        avg_day.append(avg_each_day_of_lat) 
      
      sum_days = sum(avg_day)
      
      avg_for_year = sum_days / len(range(0,int(self.days_per_year_const)))
      
      return avg_for_year

    elif lat_array == 'seas':

      if isinstance(days, type(None)):
        day = np.linspace(0,self.days_per_year_const,nt)
      else:
        day = days
      
      avg_lat = self.insolation(day, x, obl = obl, long = long, ecc = ecc, kyear = kyear)

      avg_lat = np.array(avg_lat)

      return avg_lat

    elif lat_array == 'local':
      
      avg_lat = []

      for i in range(from_lat,to_lat,int(180/n)):

          avg_day = []

          for f in range(from_month, to_month):
              
              day_at_lat = self.insolation(f,i, obl = obl, long = long, ecc = ecc, kyear = kyear)

              day_at_lat = list(day_at_lat)

              avg_day.append(day_at_lat)

          avg_day = np.mean(avg_day, axis=0) # average insoaltion of all days at a single lattitude for 2Mya

          avg_lat.append(avg_day)

      avg_lat = np.array(avg_lat)

      avg_lat = np.mean(avg_lat, axis= (0))

      return avg_lat

    elif lat_array == 'annual':
      
      if isinstance(days, type(None)):
        day = np.linspace(0,365,nt)
      else:
        day = days
      
      avg_lat = []

      for i in x:

        day_at_lat = self.insolation(day,i, obl = obl, long = long, ecc = ecc, kyear = kyear)
       
        avg_lat.append(day_at_lat)

      avg_lat = np.array(avg_lat)

      avg_lat = avg_lat.T

      avg_lat = np.mean(avg_lat, axis = 0)

      avg_lat = np.tile(avg_lat, (nt, 1))

      return avg_lat

    elif lat_array == 'integer':

      if isinstance(days, type(None)):
        day = np.linspace(0,self.days_per_year_const,365)
      else:
        day = days
      
      avg_lat = self.insolation(day, x, obl = obl, long = long, ecc = ecc, kyear = kyear)

      return avg_lat

    elif lat_array == 'for lat':

      if isinstance(days, type(None)):
        day = np.linspace(0,self.days_per_year_const,365)
      else:
        day = days
      
      avg_lat = []

      if isinstance(lat, int) or isinstance(lat, float):

        avg_lat = self.insolation(day, lat = lat, obl = obl, long = long, ecc = ecc, kyear = kyear)
    
      else:

        sin_grid = np.linspace(sin(np.deg2rad(lat[0])),sin(np.deg2rad(lat[1])),1000) #native grid
        lat_range = np.rad2deg(np.arcsin(sin_grid))
        
        avg_lat = self.insolation(day, lat = lat_range, obl = obl, long = long, ecc = ecc, kyear = kyear)
          
        avg_lat = np.array(avg_lat)

        day_ax = np.linspace(0,self.days_per_year_const,365)
        avg_lat = xr.DataArray(data=avg_lat,dims=['day','lat'],coords={'day':day_ax,'lat':lat_range})

      return avg_lat

  def get_insolation(self, grid, kyear, at_lat): #returns annual average insolation at kyear and lattitude

    if kyear-1 >= 0:
      return self.avg_insolation(grid, lat = at_lat)[kyear-1]
    else:
      return "get_insolation input must be positive integer"

  def s_array(self, grid, lat_array = 0, from_lat = None, to_lat = None, obl = None, long = None, ecc = None, kyear = None): #takes in a kyear and returns an insolation-lattitude array for seasonal or annual avg

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']
    
    if kyear == 'forced':
      return Orbital_Insolation(1,0).avg_insolation(grid, lat_array, from_lat = from_lat, to_lat = to_lat, obl = obl, long = long, ecc = ecc, kyear = kyear)

    elif type(kyear) == type(0):
      return Orbital_Insolation(kyear, kyear-1).avg_insolation(grid, lat_array, from_lat = from_lat, to_lat = to_lat)

    elif kyear-1 < 0:
      return "kyear cannot be a negative value"

  def display_orbit(self, kyear): #returns orbital values for a single kyear

    if kyear != 'forced':
      kyear0,ecc0,long_peri,obliquity,precession = self.get_orbit()

      kyear0 = kyear0[kyear-1:kyear]
      ecc0 = ecc0[kyear-1:kyear]
      long_peri = long_peri[kyear-1:kyear]
      obliquity = obliquity[kyear-1:kyear]

      kyear0, ecc0, long_peri, obliquity = float(kyear0), float(ecc0), float(long_peri), float(obliquity)

      return kyear0,ecc0,long_peri,obliquity
    
    else:

      kyear0 = 0
      ecc0 = 0
      long_peri = 0
      obliquity = 0
      
      return kyear0,ecc0,long_peri,obliquity

class Helper_Functions(): # miscellaneous methods for analysis on EBM output

  def __init__(self, output = None):
    
    self.output = output

  def find_plot_dims(kyear):

    least_square = len(kyear)
    for i in range(-least_square,0):
      if (np.sqrt(abs(i))).is_integer():
        least_square = np.abs(i)
        plot_rows = int(np.sqrt(least_square))
        plot_cols = int(np.sqrt(least_square))
        if least_square == len(kyear):
          break
        else:
          while plot_rows*plot_cols < len(kyear):
            plot_cols = plot_cols + 1
          break
    
    return plot_rows, plot_cols

  def area_array(self, grid, r = 6378100): # calculates the area of a zonal band on earth

      n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
      nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']
      
      #lat = np.linspace(-90,90,n+1)
      #lat = np.deg2rad(lat)
      lat = x
      Area_bands = []
      
      if type(r) == np.ndarray:

        for i in range(0,len(lat)-1): #geiod earth
            
            A = 2 * np.pi * (r[i]**2) * np.abs( np.sin(lat[i]) - np.sin(lat[i+1]) ) 

            Area_bands.append(A)

      else:
        for i in range(0,len(lat)-1): #perfect sphere earth
            
            A = 2 * np.pi * (r**2) * np.abs( np.sin(lat[i]) - np.sin(lat[i+1]) ) 

            Area_bands.append(A)

      return Area_bands

  def geocentric_rad(self,lat): # returns an array of radii for each latitude for geiod earth

    r_eq = 6378137.0 #m
    r_pole = 6356752.3142 #m

    geo_r = ((((r_eq**2)*np.cos(lat))**2 + ((r_pole**2)*np.sin(lat))**2) / ((r_eq * np.cos(lat))**2 + (r_pole * np.sin(lat))**2))**(1/2)

    return geo_r

  def weight_area(self, grid, array): # returns area weighted annual mean insolation

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

    r = 6378100 #m

    lat = np.linspace(-90,90,n)
    lat = np.deg2rad(lat)

    total_area = np.pi * 4 * (r**2)

    array = np.mean(array, axis = (0))

    zonal_bands = self.area_array(grid, r)

    energy_in_band = array * zonal_bands #watts

    total_energy = np.sum(energy_in_band)

    annual_mean_insolation = total_energy / total_area

    return annual_mean_insolation

  def make_contours(self, array_1, array_2, array_diff): # returns bounds and step for arrays to contour plot, so nothing is missed and steps are approporaite size

    res_param = 20

    plot_1_max = np.max(array_1)
    plot_1_min = np.min(array_1)
    plot_1_step = (abs(plot_1_min) + abs(plot_1_max)) / res_param

    plot_2_max = np.max(array_2)
    plot_2_min = np.min(array_2)
    plot_2_step = (abs(plot_2_min) + abs(plot_2_max)) / res_param

    plot_diff_max = np.max(array_diff)
    plot_diff_min = np.min(array_diff)
    plot_diff_step = (abs(plot_diff_min) + abs(plot_diff_max)) / res_param

    bar_max_list = [plot_1_max,plot_2_max]
    bar_max = np.max(bar_max_list)
    bar_min_list = [plot_1_min, plot_2_min]
    bar_min = np.max(bar_min_list)
    bar_step = (abs(bar_min) + abs(bar_max)) / res_param

    return bar_min, bar_max, bar_step, plot_diff_max, plot_diff_min, plot_diff_step

  def find_extrema(self, grid, array, position_array, type = None): # finds kyear of max/min values

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

    if array.ndim == 1:
      
      if type == None:
        
        array = list(array)
        array_max = np.max(array)
        max_index = array.index(array_max)
        max_position = position_array[max_index]
        
        array_min = np.min(array)
        min_index = array.index(array_min)
        min_position = position_array[min_index]

        both_position = (min_position, max_position)

        return both_position

      elif type == 'min':
        
        array = list(array)
        array_min = np.min(array)
        min_index = array.index(array_min)
        min_position = position_array[min_index]

        return min_position

      elif type == 'max':

        array = list(array)
        array_max = np.max(array)
        max_index = array.index(array_max)
        max_position = position_array[max_index]

        return max_position

    elif array.ndim == 2:
      
      seasonal_maximum_loc = []

      for i in range(0,nt):

        array_in_loop = abs(array[:,i])
        array_in_loop = list(array_in_loop)
        array_in_loop_max = np.max(array_in_loop)
        max_index = array_in_loop.index(array_in_loop_max)
        max_position = position_array[max_index]

        seasonal_maximum_loc.append(max_position)

  def find_zeros(self, grid, array, position_array, any = None): # finds position of value closest to 0

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

    if array.ndim == 1:
    
      array = abs(array)
      array = list(array)
      closest_to_zero = np.min(array)
      zero_index = array.index(closest_to_zero)
      zero_position = position_array[zero_index]
      output = zero_position

      if any == 'both': # finds positions for the the closest two values to 0

        sorted_list = sorted(array)
        p_min_ultimate = sorted_list[1]
        second_zero = array.index(p_min_ultimate)
        zero_position_2 = position_array[second_zero]

        output = (zero_position, zero_position_2)

      return output

    elif array.ndim == 2:

      #Southern Hemi
      array = array[0:(int(n/2)),:]
      position_array = position_array[0:(int(n/2))]

      seasonal_iceline = []
  
      for i in range(0,nt):

        array_in_loop = abs(array[:,i])
        array_in_loop = list(array_in_loop)
        closest_to_zero = np.min(array_in_loop)
        zero_index = array_in_loop.index(closest_to_zero)
        zero_position = position_array[zero_index]
        output = zero_position

        seasonal_iceline.append(output)

      max_ice_extent = np.max(seasonal_iceline)
      min_ice_extent = np.min(seasonal_iceline)
      mean_ice_extent = np.mean(seasonal_iceline)

      return max_ice_extent, min_ice_extent, mean_ice_extent

  def find_lat_of_value(self, grid, array, position_array, value = 0.4, save_icelines = 'Off', filename = None): # finds the max, min, mean, and all latitudes where specified value occurs

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

    if array.ndim == 1:

      positions = position_array[array == value]
      southern_hemi_positions = [x for x in positions if x <= 0]
      val_ext = np.max(southern_hemi_positions)

      return val_ext
    
    elif array.ndim == 2:

      positions_S = []
      positions_N = []
  
      for i in range(0, nt):
        array_in_loop = array[:,i]

        if value == 'maximum':
          position = position_array[array_in_loop == np.max(array_in_loop)]
        elif value == 'minimum':
          position = position_array[array_in_loop == np.min(array_in_loop)]
        else:
          position = position_array[array_in_loop == value]
        
        southern_hemi_positions = [x for x in position if x <= 0]
        northern_hemi_positions = [x for x in position if x >= 0]
        if len(southern_hemi_positions) > 0:
          val_ext_S = np.max(southern_hemi_positions)
          positions_S.append(val_ext_S)
        elif len(southern_hemi_positions) <= 0:
          positions_S.append(-90.0)
        if len(northern_hemi_positions) > 0:
          val_ext_N = np.min(northern_hemi_positions)
          positions_N.append(val_ext_N)
        elif len(northern_hemi_positions) <= 0:
          positions_N.append(90.0)
      positions_NS = list(zip(positions_S,positions_N))
      if save_icelines == 'On':
        np.save(filename, positions_NS)
      elif save_icelines == 'Off':
        pass

      max_val = np.max(positions_S)
      min_val = np.min(positions_S)
      mean_val = np.mean(positions_S)
      
      return max_val, min_val, mean_val, positions_S 

  def find_nearest_value(self, array, value): # finds nearest latitude to specified value 
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

  def find_value_at_lat(self,grid,array,lat): # finds the value at a specified latitude

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

    lat_array = list(np.rad2deg(np.arcsin(x)))

    target_lat = self.find_nearest_value(lat_array, lat)

    value_at_lat = array[lat_array.index(target_lat)]
    
    return value_at_lat

  def find_kyear_extrema(self, orbit_type, extrema_type): # finds kyear where orbital extrema occur
    
    kyear,ecc,long_peri,obl,prec = Orbital_Insolation().get_orbit()

    if orbit_type == 'ecc':
      orbit_type = ecc
    elif orbit_type == 'long':
      orbit_type = long_peri
    elif orbit_type == 'obl':
      orbit_type = obl
    elif orbit_type == 'prec':
      orbit_type = prec

    orbit_type = list(orbit_type)

    if extrema_type == 'min':

      min_orb = np.min(orbit_type)
      min_index = orbit_type.index(min_orb)
      min_kyear = kyear[min_index]
      min_kyear = abs(int(min_kyear))
      
      return min_kyear
    
    elif extrema_type == 'max':

      max_orb = np.max(orbit_type)
      max_index = orbit_type.index(max_orb)
      max_kyear = kyear[max_index]
      max_kyear = abs(int(max_kyear))

      return max_kyear

  def orbit_extrema(self, orbit_type, extrema_type): # finds orbital parameter extrema in range of Laskar 2004 look up table

    kyear,ecc,long_peri,obl,prec = Orbital_Insolation().get_orbit()

    if orbit_type == 'ecc':
      orbit_type = ecc
    elif orbit_type == 'long':
      orbit_type = long_peri
    elif orbit_type == 'obl':
      orbit_type = obl
    elif orbit_type == 'prec':
      orbit_type = prec

    orbit_type = list(orbit_type)

    if extrema_type == 'min':

      return float(np.min(orbit_type))
    
    elif extrema_type == 'max':

      return float(np.max(orbit_type))
  
  def orbit_at_time(self, orbit_type, kyear): # finds obrital parameter float value at specified kyear

    kyear0,ecc,long_peri,obl,prec = Orbital_Insolation().get_orbit()

    if orbit_type == 'ecc':
      orbit_type = ecc
    elif orbit_type == 'long':
      orbit_type = long_peri
    elif orbit_type == 'obl':
      orbit_type = obl
    elif orbit_type == 'prec':
      orbit_type = prec

    orbit = orbit_type[kyear-1:kyear]
    
    return float(orbit)

  def band_width(self, grid, units = 'km'): # finds distance between grid cells in specified units of km ,latitude, or sine of latitude

      n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
      nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

      lat_distance = np.gradient(np.rad2deg(np.arcsin(x)))

      if units == 'km':

        output = lat_distance * 110

        return output

      elif units == 'deg':

        output = lat_distance

        return output
      
      elif units == 'sinlat':

        output = np.sin(lat_distance)

        return output

  def take_gradient(self, grid, arr_1, diffx = 'deg'): # takes gradient of 1 or 2 dimentional matrices

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

    lat_dist = self.band_width(grid, units = diffx)
    
    if arr_1.ndim == 1:
        gradient = []
        for i in range(0, nt):
            T_slice = np.gradient(arr_1[i])
            gradient.append(T_slice)
        gradient = np.array(gradient)
        gradient = gradient / lat_dist 
        return gradient
    elif arr_1.ndim == 2: # gradient along axis 1
        gradient = []
        for i in range(0, nt):
            T_slice = np.gradient(arr_1[i, :])
            gradient.append(T_slice)
        gradient = np.array(gradient)
        gradient = gradient / lat_dist  
        return gradient

  def table_1_outs(self, grid, model_output, Hemi = 'S'): # finds values from EBM output for data table 1

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

    lat = np.rad2deg(np.arcsin(x))

    tfin, Efin, Tfin, T0fin, ASRfin, S, Tg, mean_S, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind, ice_lines, Albfin, OLRfin, CO2_ppm, wind_stress_curl = model_output

    global_mean_ins = np.mean(S)
    GMT = np.mean(Tfin)
    GMA = np.mean(Albfin)
    GM_OLR = np.mean(OLRfin)
        
    if Hemi == "S":

      S_hemi_output = []

      for i in range(len(model_output)):

        if type(model_output[i]) == np.ndarray:
          S_hemi_values = model_output[i][0:45]

        else:
          S_hemi_values = model_output[i]

        S_hemi_output.append(S_hemi_values)

      tfin, Efin, Tfin, T0fin, ASRfin, S, Tg, mean_S, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind, ice_lines, Albfin, OLRfin, CO2_ppm, wind_stress_curl = S_hemi_output

    if Efin.ndim == 2:
      eq_pole_diff = np.ptp(Tfin, axis=(0,1))
    else:
      eq_pole_diff = np.ptp(Tfin, axis=(0))

    E_transport_mag = np.min(E_transport)

    if E_transport.ndim == 2:
      E_transport_loc = self.find_extrema(grid, np.mean(E_transport, axis = 1), lat, 'min')
    else:     
      E_transport_loc = self.find_extrema(grid,E_transport, lat, 'min')

    T_grad_mag = np.max(T_grad)

    if T_grad.ndim == 2:
      T_grad_loc = self.find_extrema(grid, np.mean(T_grad, axis = 1), lat, 'max')
    else:
      T_grad_loc = self.find_extrema(grid, T_grad, lat, 'max')

    geo_wind_mag = np.max(geo_wind)

    if geo_wind.ndim == 2:
      geo_wind_loc = self.find_extrema(grid, np.mean(geo_wind, axis = 1), lat, 'max')
    else:
      geo_wind_loc = self.find_extrema(grid, geo_wind, lat, 'max')

    if Tfin.ndim == 2:
      max_ice, min_ice, mean_ice, positions = ice_lines
    else:
      max_ice, min_ice, mean_ice, positions = ice_lines

    return GMT, eq_pole_diff, global_mean_ins, E_transport_mag, E_transport_loc, T_grad_mag, T_grad_loc, geo_wind_mag, geo_wind_loc, max_ice, min_ice, mean_ice, GMA, GM_OLR

  def table_2_outs(self, grid, model_output, Hemi = "S"): # finds values from EBM output for data table 2

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

    lat = np.rad2deg(np.arcsin(x))

    tfin, Efin, Tfin, T0fin, ASRfin, S, Tg, mean_S, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind, ice_lines, Albfin, OLRfin, CO2_ppm, wind_stress_curl = model_output

    global_mean_ins = np.mean(S)
    GMT = np.mean(Tfin)
        
    if Hemi == "S":

      S_hemi_output = []

      for i in range(len(model_output)):

        if type(model_output[i]) == np.ndarray:
          S_hemi_values = model_output[i][0:45]

        else:
          S_hemi_values = model_output[i]

        S_hemi_output.append(S_hemi_values)

      tfin, Efin, Tfin, T0fin, ASRfin, S, Tg, mean_S, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind, ice_lines, Albfin, OLRfin, CO2_ppm, wind_stress_curl = S_hemi_output

    if Efin.ndim == 2:
      eq_pole_diff = np.ptp(Tfin, axis=(0,1))
    else:
      eq_pole_diff = np.ptp(Tfin, axis=(0))

    E_transport_mag = np.min(E_transport)

    if E_transport.ndim == 2:
      E_transport_loc = self.find_extrema(grid, np.mean(E_transport, axis = 1), lat, 'min')
    else:     
      E_transport_loc = self.find_extrema(grid,E_transport, lat, 'min')

    T_grad_mag = np.max(T_grad)

    if T_grad.ndim == 2:
      T_grad_loc = self.find_extrema(grid, np.mean(T_grad, axis = 1), lat, 'max')
    else:
      T_grad_loc = self.find_extrema(grid, T_grad, lat, 'max')

    geo_wind_mag = np.max(geo_wind)

    if geo_wind.ndim == 2:
      geo_wind_loc = self.find_extrema(grid, np.mean(geo_wind, axis = 1), lat, 'max')
    else:
      geo_wind_loc = self.find_extrema(grid, geo_wind, lat, 'max')

    if Tfin.ndim == 2:
      max_ice, min_ice, mean_ice, S_positions = ice_lines
    else:
      max_ice, min_ice, mean_ice, S_positions = ice_lines 

    return GMT, eq_pole_diff, global_mean_ins, E_transport_mag, E_transport_loc, T_grad_mag, T_grad_loc, geo_wind_mag, geo_wind_loc, max_ice, min_ice, mean_ice

  def format_data(self, df): # formats data table numerical values
    
    for i in range(0, len(df.iloc[0,:])):

      for f in range(0, len(df.iloc[:,i])):

        if type(df.iloc[f,i]) == type('str'):
         
          pass

        elif type(df.iloc[f,i]) == type(1.) or type(df.iloc[f,i]) == type(1) or isinstance(df.iloc[f,i], np.float64):

          if abs(df.iloc[f,i]) > 999:

            df.iloc[f,i] = "{:.2e}".format(df.iloc[f,i])

          else:

            df.iloc[f,i] = "{:.2f}".format(df.iloc[f,i])
      
            pass

    return df

  def CO2_to_A(self,CO2_ppm): # CO2 radiative forcing equation from Myrhe 1998

      S_efolding = 4 # Sherwood, 2020
      
      A = 196 # From default MEBM

      A = A - np.log2((CO2_ppm/280)) * S_efolding

      return A

  def sensitivity_analysis(self, run_1, run_2): # determines radiative forcings of model, determines climate sensitivity feedback parameters
  
    tfin_1, Efin_1, Tfin_1, T0fin_1, ASRfin_1, S_1, Tg_1, mean_S_1, OLR_1, kyear_1, E_transport_1, T_grad_1, alpha_1, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = run_1
    tfin_2, Efin_2, Tfin_2, T0fin_2, ASRfin_2, S_2, Tg_2, mean_S_2, OLR_2, kyear_2, E_transport_2, T_grad_2, alpha_2, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = run_2

    B = 1.8  # model lambda no-feedback [C/(W/m2)]
    model_GMT_changd_2xCO2 = 4 / B #2.22 #C
    CO2_Rforce_double = 4 #W/m^2
    sensitivity_parameter = model_GMT_changd_2xCO2 / CO2_Rforce_double
    
    delta_FCO2 = np.log2(CO2_2 / CO2_1) * CO2_Rforce_double
    delta_I = S_2 - S_1
    delta_I = (np.mean(delta_I))
    delta_ASR = ASRfin_2 - ASRfin_1
    delta_ASR = np.mean(delta_ASR)
    delta_GMT = Tfin_2 - Tfin_1
    delta_GMT = np.mean(delta_GMT)

    delta_OLR = OLRfin_2 - OLRfin_1
    delta_OLR = np.mean(delta_OLR)

    delta_alpha = Albfin_2 - Albfin_1
    delta_alpha = np.mean(delta_alpha)

    delta_N = (ASRfin_2-OLRfin_2) - (ASRfin_1-OLRfin_1)
    delta_N = np.mean(delta_N)

    dASRdI = Albfin_1*(S_2 - S_1)
    dASRdI = np.mean(dASRdI)
    
    dASRdalpha = S_1*(Albfin_2-Albfin_1)
    dASRdalpha = np.mean(dASRdalpha)

    net_r = delta_I + dASRdI + dASRdalpha + delta_FCO2

    calculated_delta_GMT = sensitivity_parameter * (net_r)

    experiemntally_determined_sensitivity_parameter = delta_GMT / net_r
    
    lambda_ice = (-delta_FCO2 / delta_GMT) + 1.8

    calculated_delta_GMT = experiemntally_determined_sensitivity_parameter * (net_r)

    unnacounted_change = delta_GMT - calculated_delta_GMT

    return dASRdI, dASRdalpha, delta_FCO2, delta_GMT, calculated_delta_GMT, unnacounted_change, lambda_ice

  def calculate_zonal_wind(self, T_grad): # uses linear regression from CAM to find geostrophic winds velocities

    #np.save(arr = T_grad, file = 'T_grad_save_for_reg.npy')

    from CESM_data import total_interp_values

    m_interp , b_interp = total_interp_values

    ## Units of CESM are wind[m/s] and T grad[C/lat] ##

    zonal_wind = m_interp * T_grad + b_interp
    zonal_wind = zonal_wind.T

    return zonal_wind

  def sea_ice_ext_to_area(self,extent_lat): # converts sea ice extent (latitude) to a surface area (m)

    r = 6378100 #m

    lat_extent_rad = np.deg2rad(extent_lat)

    area = 2 * np.pi * (r**2) * (1 - -np.cos(lat_extent_rad))

    return area

class Model_Class(): # MEBM from (Feldl & Merlis, 2021) with added orbital insolation and sea ice parameters

  def __init__(self):

    pass
  
  def model(self, S_type, grid, T, CO2_ppm = None, D = None, F=0, moist = 1, albT = 1, seas = 0, thermo = 0, hide_run = None, S = None, obl = None, long = None, ecc = None, kyear = 1, save_Alb = None, load_Albfin = 'testfile_Alb.npy', Albfin_co2 = None): #atmospheric moist energy balance

    if D == None:
      if moist==0:
        D = 0.6 # diffusivity for heat transport (W m^-2 K^-1)
      elif moist==1:
        D = 0.3 # diffusivity for heat transport (W m^-2 K^-1)
    else:
      D = D
    
    if hide_run == None:
      print("")
      print("-------------------------------------------------")
      print("")
      print(f'diffusivity for heat transport is {D} W m^-2 K^-1')
    else:
      pass
   
    S1 = 338 # insolation seasonal dependence (W m^-2)

    if CO2_ppm == None:
      CO2_ppm = 280  

    A = Helper_Functions().CO2_to_A(CO2_ppm)
    B = 1.8 # OLR temperature dependence (W m^-2 K^-1)
    #cw = 9.8 # ocean mixed layer heat capacity (W yr m^-2 K^-1)
    cw = mixedlayer.heatcapacity(60) # ocean mixed layer heat capacity (W yr m^-2 K^-1)
    S0 = 420 # insolation at equator (W m^-2)
    S2 = 240 # insolation spatial dependence (W m^-2)
    a0 = 0.7 # ice-free co-albedo at equator
    a2 = 0.1 # ice=free co-albedo spatial dependence
    ai = 0.4 # co-albedo where there is sea ice
    Fb = 0 # heat flux from ocean below (W m^-2)
    k = 2 # sea ice thermal conductivity (W m^-1 K^-1)
    Lf = 9.5 # sea ice latent heat of fusion (W yr m^-3)
    cg = 0.098 #0.01*cw # ghost layer heat capacity(W yr m^-2 K^-1)
    tau = 1e-5 # ghost layer coupling timescale (yr)
    Lv = 2.5E6 # latent heat of vaporization (J kg^-1)
    cp = 1004.6 # heat capacity of air at constant pressure (J kg^`-`1 K^-1)
    RH = 0.8 # relative humidity
    Ps = 1E5 # surface pressure (Pa)
    sigma = 5.67e-8 # stephan boltzmann constant (W m^-2 K^-4)
    omega = 7.2921159e-5 # angular velocity of earth (s^-1)
    roh = 1.225 # density of air (kg m^-3)
    drag_coeff = 1.25e-3 # air-sea drag coefficient from "Wind Stress Drag Coefficient over the Global Ocean" 2007
    ice_ext = 70
    air_therm_expansion = 3.4e-3
    south_pole_alb = -77
    south_pole_outter_alb = -68

    # Read grid dict
    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

    coriolis_param = 2*omega*x
    lat = np.rad2deg(np.arcsin(x))

    if hide_run == None:
      if S_type == 0:
        print(f'CO2 ppm is {CO2_ppm} and Insolation is Default')
      elif S_type == 1:
        if kyear == 'forced':
          print(f"CO2 ppm is {CO2_ppm} and Insolation is forced")
          print(f'Obl is {obl:.2f}, Long is {long:.2f}, Ecc is {ecc:.2f}')
        else:
          print(f"CO2 ppm is {CO2_ppm} and Insolation is Orbital @kyr {kyear}")
      print(f'Running model for {n} grid cells')
      print(f'Model has a {moist}{albT}{seas}{thermo} configurtion')
    else:
      pass

    # Diffusion Operator (WE15, Appendix A) 
    lam = D/dx**2*(1-xb**2)
    L1=np.append(0, -lam) 
    L2=np.append(-lam, 0) 
    L3=-L1-L2
    diffop = - np.diag(L3) - np.diag(L2[:n-1],1) - np.diag(L1[1:n],-1)

  
    # Definitions for implicit scheme on Tg
    cg_tau = cg/tau
    dt_tau = dt/tau
    dc = dt_tau*cg_tau
    kappa = (1+dt_tau)*np.identity(n)-dt*diffop/cg
    
    # Seasonal forcing (WE15 eq.3)
    if seas == 0:
      S1 = 0.0

    ty = np.arange(dt/2,1+dt/2,dt)
    if S_type == 0: # Uses seasonal Insolation given by (North & Coakley 1979)
 
      # Seasonal forcing (WE15 eq.3)
      if seas == 0:
        S1 = 0.0
      
      elif seas == 2:
        pass

      S = (np.tile(S0-S2*x**2,[nt,1])-
          np.tile(S1*np.cos(2*np.pi*ty),[n,1]).T*np.tile(x,[nt,1]))
    
      # zero out negative insolation
      S = np.where(S<0,0,S)

    elif S_type == 1: # Uses orbital parameters to determine seasonal TOA insolation

      if kyear != 'forced': # non-forced will use orbital parameter values at a specifed kyear from Laskar 2004 look up table

        if seas == 0:
          
          S = Orbital_Insolation().s_array(grid, kyear = kyear, lat_array = 'annual')

          S = np.where(S<0,0,S)
        
        elif seas == 1:
          
          S = Orbital_Insolation().s_array(grid, kyear = kyear, lat_array = 'seas')

          S = np.where(S<0,0,S) 

        elif seas == 2:

          S = Orbital_Insolation().s_array(grid, kyear = kyear, lat_array = 'seas')

          S = np.where(S<0,0,S)
        
      elif kyear == 'forced': # forced will use indivudual orbital parameter values


        if seas == 0:
          
          S = Orbital_Insolation().s_array(grid, lat_array = 'annual', obl = obl, long = long, ecc = ecc, kyear = kyear)

          S = np.where(S<0,0,S)
        
        elif seas == 1:
          
          S = Orbital_Insolation().s_array(grid, lat_array = 'seas', obl = obl, long = long, ecc = ecc, kyear = kyear)

          S = np.where(S<0,0,S) 

        elif seas == 2:

          S = Orbital_Insolation().s_array(grid, lat_array = 'seas', obl = obl, long = long, ecc = ecc, kyear = kyear)

          S = np.where(S<0,0,S)

    # Further definitions
    M = B+cg_tau

    aw = a0-a2*x**2# open water albedo

    kLf = k*Lf

    # Set up output arrays
    Efin = np.zeros((n,nt)) 
    Tfin = np.zeros((n,nt))
    T0fin = np.zeros((n,nt))
    ASRfin = np.zeros((n,nt))
    Albfin = np.zeros((n,nt))
    OLRfin = np.zeros((n,nt))
    tfin = np.linspace(0,1,nt)

    # Initial conditions
    Tg = T
    E = cw*T
    Energy_Balance = 2
    while abs(Energy_Balance) > eb: # model runs to specified energy balance (eb)

      # Integration (see WE15_NumericIntegration.pdf)
      # Loop over Years 

      for years in range(dur):
 
        # Loop within One Year
        for i in range(nt):

          # forcing
          if albT == 1: #dynamic ice
            alpha = aw * (E>0) + ai * (E<0) #WE15, eq.4
          
          elif albT == 2: # constant ice lines
            alpha = aw * (abs(lat) < ice_ext) + ai * (abs(lat) > ice_ext)

          elif albT == 3: #dynamic ice with antarctic land ice

            alpha = aw * (E>0) + ai * (E<0)
            
            alpha = alpha * (lat > south_pole_outter_alb) + (ai + 0.08) * (lat < south_pole_outter_alb)

            alpha = ai * (lat <= south_pole_alb) + alpha * (lat > south_pole_alb)

            transient_ice = Helper_Functions().find_lat_of_value(grid, alpha, lat, ai)

            if transient_ice  >  south_pole_outter_alb:

              alpha = alpha * (lat > transient_ice) + ai * (lat <= transient_ice)
            
            else:
              pass

          elif albT == 4: #loaded seasonal icelines

            if n == 90:
              alpha_loaded = np.load('Albedo_90cells_A=196_Co2=280.npy')
            else:
              alpha_loaded = np.load('Albedo_120cells_A=196_Co2=280.npy')

            alpha = alpha_loaded[:,i]

          elif albT == 5: #load iterated seasonal icelines

            alpha_loaded = Albfin_co2

            alpha = alpha_loaded[:,i]

          else: #no ice
            alpha = aw

          C = alpha*S[i,:] + cg_tau * Tg - A + F

          OLR = A + B*T

          # surface temperature
          if thermo == 1:
            T0 = C/(M-kLf/E) #WE15, eq.A3
          else:
            T0 = E/cw
        
          # store final year
          if years==(dur-1): 
            Efin[:,i] = E
            Tfin[:,i] = T
            T0fin[:,i] = T0
            ASRfin[:,i] = alpha*S[i,:]
            Albfin[:,i] = alpha
            OLRfin[:,i] = OLR

          
          T = E/cw*(E>=0)+T0*(E<0)*(T0<0) #WE15, eq.9

          # Forward Euler on E˚º
          E = E+dt*(C-M*T+Fb) #WE15, eq.A2

          # Implicit Euler on Tg
          if moist == 1:

            # Forward Euler on diffusion of latent heat
            q = RH * experiment().saturation_specific_humidity(Tg,Ps)
            rhs1 = np.matmul(dt*diffop/cg, Lv*q/cp)

            if thermo == 1:
            # FM21, eq. 3
              Tg = np.linalg.solve(kappa-np.diag(dc/(M-kLf/E)*(T0<0)*(E<0)),
                                  Tg + rhs1 + (dt_tau*(E/cw*(E>=0)+(ai*S[i,:]-A+F)/(M-kLf/E)*(T0<0)*(E<0))))
            else:
              Tg = np.linalg.solve(kappa,
                                  Tg + rhs1 + dt_tau*(E/cw) )

          elif moist == 0:
            if thermo ==1:
            #WE15, eq. A1
              Tg = np.linalg.solve(kappa-np.diag(dc/(M-kLf/E)*(T0<0)*(E<0)),
                                  Tg + (dt_tau*(E/cw*(E>=0)+(ai*S[i,:]-A+F)/(M-kLf/E)*(T0<0)*(E<0))))
            else:
              Tg = np.linalg.solve(kappa,
                                  Tg + dt_tau*(E/cw) )
    
      Energy_Balance = np.mean(ASRfin, axis=(0,1)) - A - B*np.mean(Tfin, axis=(0,1))


    # analyzing ice lines
    max_ice_extent, min_ice_extent, mean_ice_extent, ice_positions = Helper_Functions().find_lat_of_value(grid, Albfin, lat, ai)
    ice_lines = max_ice_extent, min_ice_extent, mean_ice_extent, ice_positions
    if save_Alb == 'On':
      np.save(file = "put file name here", arr = Albfin)
 
    # calculating energy transport
    R_TOA = ASRfin - (A + B*Tfin)
    area = Helper_Functions().area_array(grid)[0]
    dHdlat = area * R_TOA

    E_transport = []
    for i in range(0,nt):
      E_slice = np.cumsum(dHdlat[:,i])
      E_transport.append(E_slice)

    E_transport = np.array(E_transport)

    # caclulating surface temperature gradient
    T_grad = Helper_Functions().take_gradient(grid,Tfin.T, diffx = 'deg') # C /deg

    # geostrophic wind
    #geo_wind =  Helper_Functions().calculate_zonal_wind(T_grad)
    #max_geo_wind, min_geo_wind, mean_geo_wind, wind_lines = Helper_Functions().find_lat_of_value(grid, geo_wind, lat, 'maximum')

    # coriolis parameter
    f = 2*omega*sin(np.arcsin(x))
    f = np.tile(f, (nt,1))

    # WSC
    #wind_stress = (geo_wind ** 2) * roh * drag_coeff

    #wind_stress_curl = Helper_Functions().take_gradient(grid,wind_stress.T) # m / s deg

    wind_stress_curl = Efin
    geo_wind = Efin

    mean_S = np.mean(S, axis=(0,1))
    OLR = A - B*Tfin
    OLR = np.mean(OLR,axis = (1))
  
    
    if hide_run == None:
      print(f'{np.mean(Tfin, axis=(0,1))} global mean temp')
      print(f'{np.ptp(np.mean(Tfin, axis=1))} equator-pole temp difference')
      print(f'{np.mean(S, axis=(0,1))} global mean annual mean inso')
      print(f'{mean_ice_extent} mean ice line')
      print(f'{Energy_Balance} energy balance')
    else:
      pass

    #make arrays 1D for annual, have shape of n
    wind_stress_curl = wind_stress_curl.T
    E_transport = E_transport.T
    T_grad = T_grad.T
    S = S.T

    if seas == 0:
      Efin = np.mean(Efin, axis = 1)
      Tfin = np.mean(Tfin, axis = 1)
      T0fin = np.mean(T0fin, axis = 1)
      ASRfin = np.mean(ASRfin, axis = 1)
      OLR = np.mean(OLR, axis = 1)
      S = np.mean(S, axis = 1)
      E_transport = np.mean(E_transport, axis = 1)
      T_grad = np.mean(T_grad, axis = 1)
      geo_wind = np.mean(geo_wind, axis = 1)
      Albfin = np.mean(Albfin, axis = 1)
      OLRfin = np.mean(OLRfin, axis = 1)
      wind_stress_curl = np.mean(wind_stress_curl, axis = 1)

    if seas == 2:
      Efin = np.mean(Efin, axis = 1)
      Tfin = np.mean(Tfin, axis = 1)
      T0fin = np.mean(T0fin, axis = 1)
      ASRfin = np.mean(ASRfin, axis = 1)
      OLR = np.mean(OLR, axis = 1)
      S = np.mean(S, axis = 1)
      E_transport = np.mean(E_transport, axis = 1)
      T_grad = np.mean(T_grad, axis = 1)
      geo_wind = np.mean(geo_wind, axis = 1)
      Albfin = np.mean(Albfin, axis = 1)
      OLRfin = np.mean(OLRfin, axis = 1)
      wind_stress_curl = np.mean(wind_stress_curl, axis = 1)

    return tfin, Efin, Tfin, T0fin, ASRfin, S, Tg, mean_S, OLR, kyear, E_transport, T_grad, alpha, geo_wind, ice_lines, Albfin, OLRfin, CO2_ppm, wind_stress_curl

class Figures(): # generates specified figures

  def __init__(self):
    pass

  def figure(self, grid, chart, output_1 = None, output_2 = None, output_3 = None, output_4 = None, subchart = None, kyear_1 = None, kyear_2 = None, kyear3 = None, kyear4 = None): #generates multiple figures based on chart/subchart condition

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

    print("")
    print("-------------------------------------------------")
    print("")
    print("generating chart {}".format(chart))
    print("")
    
    if chart == 1: #default charts of northern hemi plots showing energy and temp and more

      # northern hemi only for plot
      n_2 = int(n/2)
      x_n = x[-n_2:]
      Tfin = Tfin[-n_2:,:]
      Efin = Efin[-n_2:,:]
      T0fin = T0fin[-n_2:,:]

      # seasons and ice edge
      # winter/summer occur when hemispheric T is min/max
      winter = np.argmin(np.mean(Tfin, axis=0))
      summer = np.argmax(np.mean(Tfin, axis=0))
      ice = np.where(Efin<0,np.expand_dims(x_n,1),1)
      xi = np.min(ice, axis=0)
      Lf = 9.5 # sea ice latent heat of fusion (W yr m^-3)
      icethick = -Efin/Lf*(Efin<0)

      # plot enthalpy (Fig 2a)
      plt.subplot(1,4,1)
      clevsE = np.arange(-300,301,50)
      plt.contourf(tfin,x_n,Efin,clevsE)
      plt.colorbar()
      # plot ice edge on E
      plt.contour(tfin,x_n,icethick,[0],colors='k')
      plt.xlabel('t (final year)')
      plt.ylabel('x')
      plt.ylim(0,1)
      plt.title(r'E (J m$^{-2}$)')

      # plot temperature (Fig 2b)
      plt.subplot(1,4,2)
      clevsT = np.arange(-30,31,5)
      plt.contourf(tfin,x_n,Tfin,clevsT)
      plt.colorbar()
      # plot T=0 contour (the region between ice edge and T=0 contour is the
      # reegion of summer ice surface melt)
      plt.contour(tfin,x_n,icethick,[0],colors='k')
      plt.contour(tfin,x_n,T0fin,[0],colors='r')
      plt.xlabel('t (final year)')
      plt.ylabel('x')
      plt.ylim(0,1)
      plt.title(r'T ($^\circ$C)')
      
      # plot the ice thickness (Fig 2c)
      plt.subplot(1,4,3)
      clevsh = np.arange(0.00001,5.5,0.5)
      plt.contourf(tfin,x_n,icethick,clevsh)
      plt.colorbar()
      # plot ice edge on h
      plt.contour(tfin,x_n,icethick,[0],colors='k')
      plt.plot([tfin[winter], tfin[winter]],[0,max(x_n)],'k')
      plt.plot([tfin[summer], tfin[summer]],[0,max(x_n)],'k--')
      plt.xlabel('t (final year)')
      plt.ylabel('x')
      plt.ylim(0,1)
      plt.title('h (m)')
      
      # plot temperature profiles (Fig 2d)
      plt.subplot(4,4,4)
      Summer, = plt.plot(x_n,Tfin[:,summer],'k--',label='summer')
      Winter, = plt.plot(x_n,Tfin[:,winter],'k',label='winter')
      plt.plot([0,1],[0,0],'k')
      plt.xlabel('x')
      plt.ylabel(r'T ($^\circ$C)')
      plt.legend(handles = [Summer,Winter],loc=0, fontsize=8)
      
      # plot ice thickness profiles (Fig 2e)
      plt.subplot(4,4,8)
      plt.plot(x_n,icethick[:,summer],'k--')
      plt.plot(x_n,icethick[:,winter],'k')
      plt.plot([0,1], [0,0],'k')
      plt.xlim([0.7,1])
      plt.xlabel('x')
      plt.ylabel('h (m)')

      # plot seasonal thickness cycle at pole (Fig 2f)
      plt.subplot(4,4,12)
      plt.plot(tfin,icethick[-1,:],'k')
      plt.xlabel('t (final year)')
      plt.ylabel(r'h$_p$ (m)')
      
      # plot ice edge seasonal cycle (Fig 2g)
      plt.subplot(4,4,16)
      xideg = np.degrees(np.arcsin(xi))
      plt.plot(tfin,xideg,'k-')
      plt.ylim([40,90])
      plt.xlabel('t (final year)')
      plt.ylabel(r'$\theta_i$ (deg)')

      plt.savefig('MEBMplots.jpg')

    elif chart == 2: #chart of orbital values from Laskar 2004

      kyear0,ecc0,long_peri,obliquity,precession = Orbital_Insolation().get_orbit()

      fig,axs = plt.subplots(ncols=1,nrows=4, figsize = (10,6))

      plt.suptitle('Orbital Parameters from Laskar 2004')

      axs[0].plot(-kyear0,ecc0, color = 'red')
      axs[0].set_ylabel('ecc')
      axs[0].get_xaxis().set_visible(False)
      axs[1].plot(-kyear0,long_peri, color = 'blue')
      axs[1].set_ylabel('long_peri')
      axs[1].get_xaxis().set_visible(False)
      axs[2].plot(-kyear0,obliquity, color = 'green')
      axs[2].set_ylabel('obliquity')
      axs[2].get_xaxis().set_visible(False)
      axs[3].plot(-kyear0,precession, color = 'purple')
      axs[3].set_xlabel('kyear')
      axs[3].set_ylabel('precession')

      fig.savefig("OrbitOverview.jpg")

    elif chart == 3: # obliquity and insolation comparison
        
        kyear0,ecc0,long_peri,obliquity,precession = Orbital_Insolation(100,0).get_orbit()

        fig,axs = plt.subplots(nrows = 2, figsize = (10,3))

        axs[0].plot(-kyear0, Orbital_Insolation(100,0).avg_insolation(grid,'local',from_lat = 65, to_lat = 66, from_month = experiment().month["June"]+21,to_month=experiment().month['June']+22))
        axs[0].set_ylabel('Insolation')
        axs[0].get_xaxis().set_visible(False)

        axs[1].plot(-kyear0, obliquity)
        axs[1].set_ylabel('Obliquity')
        axs[1].set_xlabel('kyear')


        fig.savefig('OrbitalInsolation.jpg')

    elif chart == 4: #compares default and orbital insolations in model run 

      if subchart == 'seas':

        tfin, Efin, Tfin, T0fin, ASRfin, S_def_seas, Tg, mean_S_def, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
        tfin, Efin, Tfin, T0fin, ASRfin, S_orb_seas, Tg, mean_S_orb, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

        S_def_seas = S_def_seas.T
        S_orb_seas = S_orb_seas.T
        inso_difference = S_orb_seas-S_def_seas

        bar_min, bar_max, bar_step, plot_diff_max, plot_diff_min, plot_diff_step = Helper_Functions().make_contours(S_def_seas, S_orb_seas, inso_difference)
        
        dur_plt = np.linspace(0,365,nt)

        lat = np.rad2deg(np.arcsin(x))

        fig,axs = plt.subplots(ncols = 3, figsize = (12,8))

        axs[0].set_title('Seasonal TOA Insolation EBM Insolation \n Global Mean Insoaltion = {:.2f}W/m²'.format(mean_S_def))
        contour_1 = axs[0].contourf(dur_plt,lat,S_def_seas.T,np.arange(0,500,10), extend = 'max', cmap=plt.get_cmap('hot'))

        axs[1].set_title('Seasonal TOA Insolation Orbital Insolation \n Global Mean Insolation {:.2f}W/m²'.format(mean_S_orb))
        contour_2 = axs[1].contourf(dur_plt,lat,S_orb_seas.T,np.arange(0,500,10), extend = 'max', cmap=plt.get_cmap('hot'))

        axs[2].set_title('Seasonal Difference \n Global Mean Difference {:.2f}W/m²'.format((mean_S_orb - mean_S_def)))
        contour_3 = axs[2].contourf(dur_plt,lat,inso_difference.T,np.arange(-190,190,1), extend = 'both', cmap=plt.get_cmap('jet'))
        
        axs[0].set_xlabel('time (days)')
        axs[1].set_xlabel('time (days)')
        axs[2].set_xlabel('time (days)')
        axs[0].set_ylabel('latitude (ϕ)')

        axs[0].text(1.2, 0.975, "(a)", transform=axs[0].transAxes,
        fontsize=16)
        axs[1].text(1.2, 0.975, "(b)", transform=axs[1].transAxes,
        fontsize=16)
        axs[2].text(1.2, 0.975, "(c)", transform=axs[2].transAxes,
        fontsize=16)
       

        plt.colorbar(contour_1,ax = axs[0], label = 'W/m²')
        plt.colorbar(contour_2,ax = axs[1], label = 'W/m²')
        plt.colorbar(contour_3,ax = axs[2], label = 'W/m²')

        fig.savefig('ContourCompare.png')
      
      elif subchart == "annual":

        tfin, Efin, Tfin, T0fin, ASRfin, S_def_annual, Tg, mean_S_def, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
        tfin, Efin, Tfin, T0fin, ASRfin, S_orb_annual, Tg, mean_S_orb, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

        inso_difference = S_orb_annual.T-S_def_annual.T

        dur_plt = np.linspace(0,365,nt)
        lat = np.rad2deg(np.arcsin(x))

        fig,axs = plt.subplots(ncols = 3, figsize = (12,8))

        plt.suptitle('annual mean default insoaltion = {:.2f}, annual mean orbital insolation {:.2f}, annual mean difference {:.2f}'.format(mean_S_def, mean_S_orb, (mean_S_def - mean_S_orb)))

        axs[0].set_title('Default Annual Insolation')
        axs[0].plot(lat,S_def_annual)

        axs[1].set_title('Orbital Annual Insolation')
        axs[1].plot(lat,S_orb_annual)
        
        axs[2].set_title('Seas Difference')
        axs[2].plot(lat,inso_difference)

        axs[0].set_xlabel('grid')
        axs[1].set_xlabel('grid')
        axs[2].set_xlabel('grid')
        axs[0].set_ylabel('Insolation W/m^2')

        fig.savefig('ContourCompare.jpg')

    elif chart == 5: #compares default and orbital ins model Temperature output

      if subchart == 'seas':

        tfin, Efin, Tfin_def_seas, T0fin, ASRfin, S_def_seas, Tg, mean_S_def, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
        tfin, Efin, Tfin_orb_seas, T0fin, ASRfin, S_orb_seas, Tg, mean_S_orb, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

        mean_Tfin_def_seas = np.mean(Tfin_def_seas)
        mean_Tfin_orb_seas = np.mean(Tfin_orb_seas)

        temp_difference = Tfin_orb_seas-Tfin_def_seas

        bar_min, bar_max, bar_step, plot_diff_max, plot_diff_min, plot_diff_step = Helper_Functions().make_contours(Tfin_def_seas, Tfin_orb_seas, temp_difference)

        plot_extrema = np.max(np.abs([plot_diff_min,plot_diff_max])) 

        dur_plt = np.linspace(0,365,nt)
        lat = np.rad2deg(np.arcsin(x))

        fig,axs = plt.subplots(ncols = 3, figsize = (12,8))
        
        contour_1 = axs[0].contourf(dur_plt,lat,Tfin_def_seas,np.arange(-25,25,1), extend = 'both')
        axs[0].set_title('Seasonal Temperature EBM Insolation \n Global Mean Temperature = {:.2f}'.format(mean_Tfin_def_seas))

        contour_2 = axs[1].contourf(dur_plt,lat,Tfin_orb_seas,np.arange(-25,25,1),  extend = 'both')
        axs[1].set_title('Seasonal Temperature Orbital Insolation \n Global Mean Temperature {:.2f}'.format(mean_Tfin_orb_seas))

        contour_3 = axs[2].contourf(dur_plt,lat,temp_difference,np.arange(-3,3,0.01),  extend = 'both', cmap=plt.get_cmap('bwr'))
        axs[2].set_title('Seasonal Difference \n Global Mean Difference {:.2f}'.format((mean_Tfin_orb_seas - mean_Tfin_def_seas)))

        iceline_contour_0 = axs[0].contour(dur_plt,lat,Tfin_def_seas,levels = [0], colors = 'red', label = 'Sea Ice Extent')
        iceline_contour_1 = axs[1].contour(dur_plt,lat,Tfin_orb_seas,levels = [0], colors = 'red', label = 'Sea Ice Extent')

        axs[0].text(1.2, 0.975, "(a)", transform=axs[0].transAxes,
        fontsize=16)
        axs[1].text(1.2, 0.975, "(b)", transform=axs[1].transAxes,
        fontsize=16)
        axs[2].text(1.2, 0.975, "(c)", transform=axs[2].transAxes,
        fontsize=16)
        
        plt.colorbar(contour_1,ax = axs[0], label = '°C')
        plt.colorbar(contour_2,ax = axs[1], label = '°C')
        plt.colorbar(contour_3,ax = axs[2], label = 'Δ°C')

        axs[0].set_xlabel('time (days)')
        axs[1].set_xlabel('time (days)')
        axs[2].set_xlabel('time (days)')
        axs[0].set_ylabel('latitude (ϕ)')
        

        axs[0].clabel(iceline_contour_0)
        axs[1].clabel(iceline_contour_1)
        

        if True == True:
          plt.savefig('ContourCompare_Temp.png')

      elif subchart == 'annual':

        tfin, Efin, Tfin_def_annual, T0fin, ASRfin, S_def_annual, Tg, mean_S_def, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
        tfin, Efin, Tfin_orb_annual, T0fin, ASRfin, S_orb_annual, Tg, mean_S_orb, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

        mean_Tfin_def_annual = np.mean(Tfin_def_annual)
        mean_Tfin_orb_annual = np.mean(Tfin_orb_annual)

        temp_difference = Tfin_orb_annual-Tfin_def_annual

        dur_plt = np.linspace(0,365,nt)
        lat = np.rad2deg(np.arcsin(x))


        fig,axs = plt.subplots(ncols = 3, figsize = (12,8))

        plt.suptitle('global mean default Temperature = {:.2f}, global mean orbital Temperature {:.2f}, global mean difference {:.2f}'.format(mean_Tfin_def_annual, mean_Tfin_orb_annual, (mean_Tfin_def_annual - mean_Tfin_orb_annual)))

        axs[0].set_title('Annual Temp EBM Insolation')
        axs[0].plot(lat,Tfin_def_annual)

        axs[1].set_title('Annual Temp Orbital Insolation')
        axs[1].plot(lat,Tfin_orb_annual)

        axs[2].set_title('Annual Difference')
        axs[2].plot(lat,temp_difference)

        axs[0].set_xlabel('lat')
        axs[1].set_xlabel('lat')
        axs[2].set_xlabel('lat')
        axs[0].set_ylabel('Temperature C')

        fig.savefig('ContourCompare_Temp.jpg')
  
    elif chart == 6: #compares two orbital runs insolation at different kyears
      
      if subchart == 'seas':

          tfin, Efin, Tfin_def_seas, T0fin, ASRfin, S_seas_kyear1, Tg, mean_S_kyear1, OLR, S_kyear1, E_transport_kyr1, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
          tfin, Efin, Tfin_orb_seas, T0fin, ASRfin, S_seas_kyear2, Tg, mean_S_kyear2, OLR, S_kyear2, E_transport_kyr2, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

          kyear0_1,ecc0_1,long_peri_1,obliquity_1 = Orbital_Insolation().display_orbit(S_kyear1)
          kyear0_2,ecc0_2,long_peri_2,obliquity_2 = Orbital_Insolation().display_orbit(S_kyear2)

          dur_plt = np.linspace(0,365,nt)
          lat = np.rad2deg(np.arcsin(x))

          inso_difference = S_seas_kyear1.T - S_seas_kyear2.T

          bar_min, bar_max, bar_step, plot_diff_max, plot_diff_min, plot_diff_step = Helper_Functions().make_contours(S_seas_kyear1, S_seas_kyear2, inso_difference)

          kyr1_display = kyear_1-1
          kyr2_display = kyear_2-1
        
          fig,axs = plt.subplots(ncols = 3, figsize = (12,8))

          if kyr1_display == 0:
            plt.suptitle('Global Mean Insolation {} = {:.2f}, Global Mean Insolation {}kyr = {:.2f}, Global Mean Difference = {:.2f} \n Obliquity {} = {:.2f}deg, Obliquity {} = {:.2f}deg'.format("Modern", mean_S_kyear1, kyr2_display, mean_S_kyear2, (mean_S_kyear1 - mean_S_kyear2), "Modern", obliquity_1, kyr2_display, obliquity_2))
            axs[0].set_title('Orbital Seasonal Insolation {}'.format("Modern"))
            axs[1].set_title('Orbital Seasonal Insolation {}kyr'.format(kyr2_display))
            axs[2].set_title('Seasonal Difference')
          elif kyr2_display == 0:
            plt.suptitle('Global Mean Insolation {}kyr = {:.2f}, Global Mean Insolation {} = {:.2f}, Global Mean Difference = {:.2f}, \n Obliquity {} = {:.2f}deg, Obliquity {} = {:.2f}deg'.format(kyr1_display, mean_S_kyear1, "Modern", mean_S_kyear2, (mean_S_kyear1 - mean_S_kyear2), kyr1_display, obliquity_1, "Modern", obliquity_2))
            axs[1].set_title('Orbital Seasonal Insolation {}'.format("Modern"))
            axs[0].set_title('Orbital Seasonal Insolation {}kyr'.format(kyr1_display))
            axs[2].set_title('Seasonal Difference')
          else:
            plt.suptitle('Global Mean Insolation {}kyr = {:.2f}, Global Mean Insolation {}kyr = {:.2f}, Global Mean Difference = {:.2f} \n Obliquity {} = {:.2f}deg, Obliquity {} = {:.2f}deg'.format(kyr1_display, mean_S_kyear1, kyr2_display, mean_S_kyear2, (mean_S_kyear1 - mean_S_kyear2), kyr1_display, obliquity_1, kyr2_display, obliquity_2))
            axs[0].set_title('Orbital Seasonal Insolation {}kyr'.format(kyr1_display))
            axs[1].set_title('Orbital Seasonal Insolation {}kyr'.format(kyr2_display))
            axs[2].set_title('Seasonal Difference')
          
          contour_1 = axs[0].contourf(dur_plt,lat,S_seas_kyear1.T,np.arange(0,525,15), extend = 'max', cmap=plt.get_cmap('hot'))
          contour_2 = axs[1].contourf(dur_plt,lat,S_seas_kyear2.T,np.arange(0,525,15), extend = 'max', cmap=plt.get_cmap('hot'))
          contour_3 = axs[2].contourf(dur_plt,lat,inso_difference,np.arange(-50,50,1), extend = 'both', cmap=plt.get_cmap('bwr'))

          plt.colorbar(contour_1,ax = axs[0])
          plt.colorbar(contour_2,ax = axs[1])
          plt.colorbar(contour_3,ax = axs[2])

          axs[0].set_xlabel('days')
          axs[1].set_xlabel('days')
          axs[2].set_xlabel('days')
          axs[0].set_ylabel('latitude')

          fig.savefig('ContourCompare_Paleo.jpg')

      if subchart == "annual":

          tfin, Efin, Tfin_annual_kyear1, T0fin, ASRfin, S_annual_kyear1, Tg, mean_S_kyear1, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
          tfin, Efin, Tfin_annual_kyear2, T0fin, ASRfin, S_annual_kyear2, Tg, mean_S_kyear2, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

          dur_plt = np.linspace(0,365,nt)
          lat = np.rad2deg(np.arcsin(x))

          inso_difference = S_annual_kyear1 - S_annual_kyear2

          kyr1_display = kyear_1-1
          kyr2_display = kyear_2-1

          min_obl_S = list(zip(S_annual_kyear1, lat))

          max_obl_S = list(zip(S_annual_kyear2, lat))

          diff_S_lats = list(zip(inso_difference, lat))

          fig,axs = plt.subplots(ncols = 3, figsize = (12,8))

          plt.suptitle('global mean Insolation {}kyr = {:.2f}, global mean Insolation {}kyr = {:.2f}, global mean difference = {:.2f}'.format(kyr1_display, mean_S_kyear1, kyr2_display, mean_S_kyear2, (mean_S_kyear1 - mean_S_kyear2)))

          axs[0].set_title('Orbital Annual Insolation kyear {}'.format(kyr1_display))
          axs[0].plot(lat,S_annual_kyear1)

          axs[1].set_title('Orbital Annual Insolation kyear {}'.format(kyr2_display))
          axs[1].plot(lat,S_annual_kyear2)

          axs[2].set_title('Annual Difference')
          axs[2].plot(lat,inso_difference)
          
          axs[0].set_xlabel('lat')
          axs[1].set_xlabel('lat')
          axs[2].set_xlabel('lat')
          axs[0].set_ylabel('Insolation (W/m²)')

          fig.savefig('ContourCompare_Paleo.jpg')

      if subchart == 'forcing': # obliquity forcing

        output_1 = np.load('control_1110_low_obl.npy', allow_pickle=True)
        output_2 = np.load('control_1110_high_obl.npy', allow_pickle=True)
        tfin, Efin, Tfin_1, T0fin, ASRfin, S_seas_kyear1, Tg, mean_S_kyear1, OLR, S_kyear1, E_transport_kyr1, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
        tfin, Efin, Tfin_2, T0fin, ASRfin, S_seas_kyear2, Tg, mean_S_kyear2, OLR, S_kyear2, E_transport_kyr2, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

        dur_plt = np.linspace(0,365,nt)

        inso_difference = S_seas_kyear2 - S_seas_kyear1
        temp_difference = Tfin_2 - Tfin_1

        fig, axs = plt.subplots(ncols = 2)

        contour_1 = axs[0].contourf(dur_plt, experiment().lat_deg, inso_difference, np.arange(-50,50,0.5), extend = 'max', cmap=plt.get_cmap('jet'))
        contour_2 = axs[1].contourf(dur_plt, experiment().lat_deg, temp_difference, np.arange(-2,2,0.005), extend = 'max', cmap=plt.get_cmap('seismic'))
  
        plt.colorbar(contour_1,ax = axs[0])
        plt.colorbar(contour_2,ax = axs[1])
        axs[0].set_xlabel('Time (Days)')
        axs[1].set_xlabel('Time (Days)')

        axs[0].set_ylabel('Latitude (φ)')

        axs[0].set_title('Insolation Difference Between \n Minimum and Maximum Obliquity')
        axs[1].set_title('Temperature Difference From \n Obliquity Forcing')

        fig.tight_layout()

        plt.savefig('ins_force_temp_response.png')

    elif chart == 7: #compares two orb runs temp output
      
      if subchart == 'seas':
        
        tfin, Efin, Tfin_seas_kyear1, T0fin, ASRfin, S_seas_kyear1, Tg, mean_S_kyear1, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
        tfin, Efin, Tfin_seas_kyear2, T0fin, ASRfin, S_seas_kyear2, Tg, mean_S_kyear2, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

        Tfin_kyear1_mean = np.mean(Tfin_seas_kyear1)
        Tfin_kyear2_mean = np.mean(Tfin_seas_kyear2)

        dur_plt = np.linspace(0,365,nt)
        lat = np.rad2deg(np.arcsin(x))

        temp_difference = Tfin_seas_kyear2 - Tfin_seas_kyear1

        bar_min, bar_max, bar_step, plot_diff_max, plot_diff_min, plot_diff_step = Helper_Functions().make_contours(Tfin_seas_kyear1, Tfin_seas_kyear2, temp_difference)

        kyr1_display = kyear_1-1
        kyr2_display = kyear_2-1

        fig,axs = plt.subplots(ncols = 3, figsize = (12,8))

        if kyr1_display == 0:
          plt.suptitle('Global Mean Temperature {} = {:.2f}, Global Mean Temperature {}kyr = {:.2f}, Global Mean Difference = {:.2f}'.format("Modern", Tfin_kyear1_mean, kyr2_display, Tfin_kyear2_mean, (np.mean(temp_difference))))
          axs[0].set_title('Seasonal Temperature {}'.format("Modern"))
          axs[1].set_title('Seasonal Temperature {}kyr'.format(kyr2_display))
          axs[2].set_title('Seasonal Difference')
        elif kyr2_display == 0:
          plt.suptitle('Global Mean Temperature {}kyr = {:.2f}, Global Mean Temperature {} = {:.2f}, Global Mean Difference = {:.2f}'.format(kyr1_display, Tfin_kyear1_mean, "Modern", Tfin_kyear2_mean, (np.mean(temp_difference))))
          axs[1].set_title('Seasonal Temperature {}'.format("Modern"))
          axs[0].set_title('Seasonal Temperature {}kyr'.format(kyr1_display))
          axs[2].set_title('Seasonal Difference')
        else:
          plt.suptitle('Global Mean Temperature {}kyr = {:.2f}, Global Mean Temperature {}kyr = {:.2f}, Global Mean Difference = {:.2f}'.format(kyr1_display, Tfin_kyear1_mean, kyr2_display, Tfin_kyear2_mean, (np.mean(temp_difference))))
          axs[0].set_title('Seasonal Temperature {}kyr'.format(kyr1_display))
          axs[1].set_title('Seasonal Temperature {}kyr'.format(kyr2_display))
          axs[2].set_title('Seasonal Difference')

        contour_1 = axs[0].contourf(dur_plt,lat,Tfin_seas_kyear1,np.arange(bar_min,bar_max,bar_step), extend = 'both')
        contour_2 = axs[1].contourf(dur_plt,lat,Tfin_seas_kyear2,np.arange(bar_min,bar_max,bar_step), extend = 'both')
        contour_3 = axs[2].contourf(dur_plt,lat,temp_difference,np.arange(plot_diff_min,plot_diff_max,plot_diff_step), extend = 'both', cmap=plt.get_cmap('bwr'))

        axs[0].contour(dur_plt,lat,Tfin_seas_kyear1,levels = [0], colors = 'red')
        axs[1].contour(dur_plt,lat,Tfin_seas_kyear1,levels = [0], colors = 'red')

        plt.colorbar(contour_1,ax = axs[0])
        plt.colorbar(contour_2,ax = axs[1])
        plt.colorbar(contour_3,ax = axs[2])
      
        axs[0].set_xlabel('days')
        axs[1].set_xlabel('days')
        axs[2].set_xlabel('days')
        axs[0].set_ylabel('lat(φ)')
        axs[1].set_ylabel('lat(φ)')
        axs[2].set_ylabel('lat(φ)')

        fig.savefig('ContourCompare_Paleo.jpg')
        
      elif subchart == "annual":

        tfin, Efin, Tfin_annual_kyear1, T0fin, ASRfin, S_annual_kyear1, Tg, mean_S_kyear1, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
        tfin, Efin, Tfin_annual_kyear2, T0fin, ASRfin, S_annual_kyear2, Tg, mean_S_kyear2, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

        Tfin_kyear1_mean = np.mean(Tfin_annual_kyear1)
        Tfin_kyear2_mean = np.mean(Tfin_annual_kyear2)

        dur_plt = np.linspace(0,365,nt)
        lat = np.rad2deg(np.arcsin(x))

        temp_difference = Tfin_annual_kyear1 - Tfin_annual_kyear2

        kyr1_display = kyear_1-1
        kyr2_display = kyear_2-1

        fig,axs = plt.subplots(ncols = 3, figsize = (12,8))

        plt.suptitle('global mean Temperature {}kyr = {:.2f}, global mean Temperature {}kyr = {:.2f}, global mean difference = {:.2f}'.format(kyr1_display, Tfin_kyear1_mean, kyr2_display, Tfin_kyear2_mean, (Tfin_kyear1_mean - Tfin_kyear2_mean)))

        axs[0].set_title('Annual Temperature kyear {}'.format(kyr1_display))
        axs[0].plot(lat,Tfin_annual_kyear1)

        axs[1].set_title('Annual Temperature kyear {}'.format(kyr2_display))
        axs[1].plot(lat,Tfin_annual_kyear2)

        axs[2].set_title('Annual Difference')
        axs[2].plot(lat,temp_difference)
        
        axs[0].set_xlabel('lat(φ)')
        axs[1].set_xlabel('lat(φ)')
        axs[2].set_xlabel('lat(φ)')
        axs[0].set_ylabel('Temperature (C)')

        fig.savefig('ContourCompare_Paleo.jpg')

    elif chart == 8: #plots kyear vs annual avg insolation at a given latitude  

      kyear0,ecc0,long_peri,obliquity,precession = Orbital_Insolation(10).get_orbit()

      lat = -65

      fig,axs = plt.subplots()

      axs.plot(-kyear0,Orbital_Insolation(10).avg_insolation(grid, lat = lat))
      axs.set_xlabel('kyear')
      axs.set_ylabel('W/m^2')
      if lat > 0:
        axs.set_title("Annual Average Insolation at {}N".format(lat))
      elif lat < 0:
        axs.set_title("Annual Average Insolation at {}S".format(abs(lat)))
      else:
        axs.set_title("Annual Average Insolation at the Equator")

    elif chart == 9: #plots to mimic MPT paper, insolation and orbit

      kyear0,ecc0,long_peri,obliquity,precession = Orbital_Insolation().get_orbit()

      fig,axs = plt.subplots(nrows = 3, figsize = (10,5))

      axs[0].plot(-kyear0,obliquity, color = "green")
      axs[1].plot(-kyear0,ecc0, label = "eccentricity", color = 'red')
      axs[1].plot(-kyear0, precession, label = 'precession', color = 'darkblue')
      axs[2].plot(-kyear0, Orbital_Insolation().avg_insolation(experiment().config,lat_array= "local", from_lat = 65, to_lat = 66, from_month = experiment().month['June'], to_month = experiment().month['September']), color = 'darkorange')

      axs[2].set_xlabel('kyear')

      axs[0].set_ylabel('obliquity (deg)')
      axs[0].grid()
      axs[1].set_ylabel('eccentricity / precession')
      axs[1].legend()
      axs[1].grid()
      axs[2].set_ylabel('65N JJA insolation (W/m2)')
      axs[2].grid()

      fig.tight_layout()

      plt.savefig('orbtail_ins_paper_comp.jpg')

    elif chart == 10: #Summer Solstice at 65N vs kyear

      kyear0,ecc0,long_peri,obliquity,precession = Orbital_Insolation(140,0).get_orbit()

      fig,axs = plt.subplots(figsize = (14,5))

      axs.plot(-kyear0,Orbital_Insolation(140,0).avg_insolation(experiment().config,lat_array= "local", from_lat = 65, to_lat = 66, from_month = experiment().month['June']+21, to_month = experiment().month['June']+22))

      plt.savefig('65N_2Mya_Summer.jpg')

    elif chart == 11: #OLR and ASR plots

      tfin, Efin, Tfin, T0fin, ASRfin_1, S_def_annual, Tg, mean_S_def, OLR_def, S_kyear, E_transport, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
      tfin, Efin, Tfin, T0fin, ASRfin_2, S_orb_annual, Tg, mean_S_orb, OLR_orb, S_kyear, E_transport, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

      ASRfin_1 = np.mean(ASRfin_1, axis =1)
      ASRfin_2 = np.mean(ASRfin_2, axis =1)

      inso_difference = S_orb_annual-S_def_annual
      E_max = np.max(E_transport)
      E_transport = list(E_transport)
      where = E_transport.index(E_max)
      fig,axs = plt.subplots(nrows = 2, figsize = (8,10))

      axs[0].plot(x,ASRfin_1, label = 'default ASR')
      axs[0].plot(x, OLR_def, label = 'default OLR')
      axs2 = axs[0].twinx()
      axs2.plot(x, E_transport, color = 'green')
      axs[0].set_xlabel('lattitude')
      axs[0].set_ylabel('zonally averaged ASR and OLR')
      axs[0].set_title("Default Energy Balance")
      axs2.plot(x[where], E_max, 'o',color ='red', markersize = (10))
      axs[0].legend()
      axs[1].plot(x,ASRfin_2, label = 'orbital ASR')
      axs[1].plot(x,OLR_orb, label = 'orbital OLR')
      axs[1].set_xlabel('lattitude')
      axs[1].set_ylabel('zonally averaged ASR and OLR')
      axs[1].set_title("Orbital Energy Balance")
      axs[1].legend()

    elif chart == 12: # Energy Transport Compare
      
      tfin, Efin, Tfin_annual_kyear2, T0fin, ASRfin, S_annual_kyear2, Tg, mean_S_kyear2, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2
      lat = np.rad2deg(np.arcsin(x))
     
      if subchart == 'annual':
        
        fig, axs = plt.subplots(nrows = 1)
        axs.set_xlabel('latitude')
        axs.set_ylabel('Watts')
        axs.plot(ax,E_transport)
      
      elif subchart == 'seas':
        
        dur_plt = np.linspace(0,365,nt)
        fig, axs = plt.subplots(nrows = 1)
        axs.set_xlabel('latitude')
        axs.set_ylabel('Watts')
        axs.plot(ax,E_transport[0,:])
        axs.plot(ax,E_transport[100,:])
        axs.plot(ax,E_transport[200,:])
        axs.plot(ax,E_transport[300,:])
        axs.plot(ax,E_transport[400,:])
        axs.plot(ax,E_transport[500,:])
        axs.plot(ax,E_transport[600,:])
        axs.plot(ax,E_transport[700,:])
        axs.plot(ax,E_transport[800,:])
        axs.plot(ax,E_transport[900,:])
        axs.plot(ax,np.mean(E_transport, axis = 0))

    elif chart == 13: #Energy Transport Paleo Compare
      
      if subchart == 'seas':

        tfin, Efin, Tfin_def_seas, T0fin, ASRfin, S_seas_kyear1, Tg, mean_S_kyear1, OLR, S_kyear1, E_transport_kyr1, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
        tfin, Efin, Tfin_orb_seas, T0fin, ASRfin, S_seas_kyear2, Tg, mean_S_kyear2, OLR, S_kyear2, E_transport_kyr2, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

        kyear0_1,ecc0_1,long_peri_1,obliquity_1 = Orbital_Insolation().display_orbit(kyear_1)
        kyear0_2,ecc0_2,long_peri_2,obliquity_2 = Orbital_Insolation().display_orbit(kyear_2)

        dur_plt = np.linspace(0,365,nt)

        trans_difference = E_transport_kyr1 - E_transport_kyr2

        bar_min, bar_max, bar_step, plot_diff_max, plot_diff_min, plot_diff_step = Helper_Functions().make_contours(E_transport_kyr1, E_transport_kyr2, trans_difference)

        kyr1_display = kyear_1-1
        kyr2_display = kyear_2-1

        E_max_kyr1 = np.max(E_transport_kyr1)
        E_max_kyr2 = np.max(E_transport_kyr2)

        ax = np.linspace(-90,90,90)
      
        fig,axs = plt.subplots(ncols = 3, figsize = (12,8))

        if kyr1_display == 0:
          plt.suptitle('Maximum Transport {} = {:.2e}, Maximum Transport {}kyr = {:.2e}, Maximum Transport Difference = {:.2e} \n Obliquity {} = {:.2f}deg, Obliquity {} = {:.2f}deg'.format("Modern", E_max_kyr1, kyr2_display, E_max_kyr2, (E_max_kyr1 - E_max_kyr2), "Modern", obliquity_1, kyr2_display, obliquity_2))
          axs[0].set_title('Seasonal Energy Transport {}'.format("Modern"))
          axs[1].set_title('Seasonal Energy Transport {}kyr'.format(kyr2_display))
          axs[2].set_title('Seasonal Difference')
        elif kyr2_display == 0:
          plt.suptitle('Maximum Transport {}kyr = {:.2e}, Maximum Transport {} = {:.2e}, Maximum Transport Difference = {:.2e}, \n Obliquity {} = {:.2f}deg, Obliquity {} = {:.2f}deg'.format(kyr1_display, E_max_kyr1, "Modern", E_max_kyr2, (E_max_kyr1 - E_max_kyr2), kyr1_display, obliquity_1, "Modern", obliquity_2))
          axs[1].set_title('Seasonal Energy Transport {}'.format("Modern"))
          axs[0].set_title('Seasonal Energy Transport {}kyr'.format(kyr1_display))
          axs[2].set_title('Seasonal Difference')
        else:
          plt.suptitle('Global Mean Insolation {}kyr = {:.2e}, Global Mean Insolation {}kyr = {:.2e}, Global Mean Difference = {:.2e} \n Obliquity {} = {:.2f}deg, Obliquity {} = {:.2f}deg'.format(kyr1_display, E_max_kyr1, kyr2_display, E_max_kyr2, (E_max_kyr1 - E_max_kyr2), kyr1_display, obliquity_1, kyr2_display, obliquity_2))
          axs[0].set_title('Seasonal Energy Transport {}kyr'.format(kyr1_display))
          axs[1].set_title('Seasonal Energy Transport {}kyr'.format(kyr2_display))
          axs[2].set_title('Seasonal Difference')

        contour_1 = axs[0].contourf(dur_plt,ax,E_transport_kyr1,np.arange(bar_min,bar_max,bar_step), extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_2 = axs[1].contourf(dur_plt,ax,E_transport_kyr2,np.arange(bar_min,bar_max,bar_step), extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_3 = axs[2].contourf(dur_plt,ax,trans_difference,np.arange(plot_diff_min,plot_diff_max,plot_diff_step), extend = 'both', cmap=plt.get_cmap('bwr'))

        plt.colorbar(contour_1,ax = axs[0])
        plt.colorbar(contour_2,ax = axs[1])
        plt.colorbar(contour_3,ax = axs[2])

        axs[0].set_xlabel('days')
        axs[1].set_xlabel('days')
        axs[2].set_xlabel('days')
        axs[0].set_ylabel('sin(φ)')

      if subchart == "annual":

        tfin, Efin, Tfin_annual_kyear1, T0fin, ASRfin, S_annual_kyear1, Tg, mean_S_kyear1, OLR, S_kyear1, E_transport_kyr1, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
        tfin, Efin, Tfin_annual_kyear2, T0fin, ASRfin, S_annual_kyear2, Tg, mean_S_kyear2, OLR, S_kyear2, E_transport_kyr2, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

        trans_difference = E_transport_kyr1 - E_transport_kyr2
        
        lat_grid = np.rad2deg(np.arcsin(x))

        E_max_kyr1 = np.max(E_transport_kyr1)
        E_max_kyr2 = np.max(E_transport_kyr2)
        E_min_kyr1 = np.min(E_transport_kyr1)
        E_min_kyr2 = np.min(E_transport_kyr2)

        E_transport_kyr2 = list(E_transport_kyr2)
        E_transport_kyr1 = list(E_transport_kyr1)

        where_E_1_min = E_transport_kyr1.index(E_min_kyr1)
        where_E_2_min = E_transport_kyr2.index(E_min_kyr2)

        where_1 = lat_grid[where_E_1_min]
        where_2 = lat_grid[where_E_2_min]

        print('kyr1', where_1)
        print('kyr2', where_2)

        E_transport_kyr1 = np.array(E_transport_kyr1)
        E_transport_kyr2 = np.array(E_transport_kyr2)


        S_annual_kyear1 = np.mean(S_annual_kyear1, axis = 0)
        S_annual_kyear2 = np.mean(S_annual_kyear2, axis = 0)

        kyear0_1,ecc0_1,long_peri_1,obliquity_1 = Orbital_Insolation().display_orbit(kyear_1)
        kyear0_2,ecc0_2,long_peri_2,obliquity_2 = Orbital_Insolation().display_orbit(kyear_2)

        dur_plt = np.linspace(0,365,nt)
        ax = np.linspace(-90,90,90)

        trans_difference = E_transport_kyr1 - E_transport_kyr2

        kyr1_display = kyear_1-1
        kyr2_display = kyear_2-1

        fig,axs = plt.subplots(ncols = 3, figsize = (12,8))

        plt.suptitle('Maximum Transport {}kyr = {:.2e}, Maximum Transport {}kyr = {:.2e}, Max Transport Difference = {:.2e}'.format(kyr1_display, E_max_kyr1, kyr2_display, E_max_kyr2, (E_max_kyr1 - E_max_kyr2)))

        if kyr1_display == 0:
          plt.suptitle('Maximum Transport {} = {:.2e}, Maximum Transport {}kyr = {:.2e}, Maximum Transport Difference = {:.2e} \n Obliquity {}kyr = {:.2f}deg, Obliquity {}kyr = {:.2f}deg'.format("Modern", E_max_kyr1, kyr2_display, E_max_kyr2, (E_max_kyr1 - E_max_kyr2), "Modern", obliquity_1, kyr2_display, obliquity_2))
          axs[0].set_title('Annual Energy Transport {}'.format("Modern"))
          axs[1].set_title('Annual Energy Transport {}kyr'.format(kyr2_display))
          axs[2].set_title('Annual Difference')
        elif kyr2_display == 0:
          plt.suptitle('Maximum Transport {}kyr = {:.2e}, Maximum Transport {} = {:.2e}, Maximum Transport Difference = {:.2e}, \n Obliquity {} = {:.2f}deg, Obliquity {} = {:.2f}deg'.format(kyr1_display, E_max_kyr1, "Modern", E_max_kyr2, (E_max_kyr1 - E_max_kyr2), kyr1_display, obliquity_1, "Modern", obliquity_2))
          axs[1].set_title('Annual Energy Transport {}'.format("Modern"))
          axs[0].set_title('Annual Energy Transport {}kyr'.format(kyr1_display))
          axs[2].set_title('Annual Difference')
        else:
          plt.suptitle('Maximum Transport {}kyr = {:.2e}, Maximum Transport{}kyr = {:.2e}, Maximum Transport Difference = {:.2e} \n Obliquity {} = {:.2f}deg, Obliquity {} = {:.2f}deg'.format(kyr1_display, E_max_kyr1, kyr2_display, E_max_kyr2, (E_max_kyr1 - E_max_kyr2), kyr1_display, obliquity_1, kyr2_display, obliquity_2))
          axs[0].set_title('Annual Energy Transport {}kyr'.format(kyr1_display))
          axs[1].set_title('Annual Energy Transport {}kyr'.format(kyr2_display))
          axs[2].set_title('Annual Difference')

        axs[0].plot(lat_grid,E_transport_kyr1)
        axs[0].plot(lat_grid,E_transport_kyr2)
        axs[2].plot(lat_grid,trans_difference)
        
        axs[0].set_xlabel('latitude')
        axs[1].set_xlabel('latitude')
        axs[2].set_xlabel('latitude')
        axs[0].set_ylabel('Energy Transport (W)')

    elif chart == 14: #Temp gradient Paleo

      if subchart == 'seas':

        tfin, Efin, Tfin_def_seas, T0fin, ASRfin, S_seas_kyear1, Tg, mean_S_kyear1, OLR, S_kyear1, E_transport_kyr1, T_grad1, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
        tfin, Efin, Tfin_orb_seas, T0fin, ASRfin, S_seas_kyear2, Tg, mean_S_kyear2, OLR, S_kyear2, E_transport_kyr2, T_grad2, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

        SO_wind, SO_lats, SO_Tgrad = Helper_Functions().seasonal_zonal_regression(T_grad1.T)
        SO_wind, SO_lats, SO_geo_wind = Helper_Functions().seasonal_zonal_regression(geo_wind_1.T)

        kyear0_1,ecc0_1,long_peri_1,obliquity_1 = Orbital_Insolation().display_orbit(kyear_1)
        kyear0_2,ecc0_2,long_peri_2,obliquity_2 = Orbital_Insolation().display_orbit(kyear_2)

        dur_plt = np.linspace(0,365,nt)

        trans_difference = T_grad1 - T_grad2

        bar_min, bar_max, bar_step, plot_diff_max, plot_diff_min, plot_diff_step = Helper_Functions().make_contours(T_grad1, T_grad2, trans_difference)

        kyr1_display = kyear_1-1
        kyr2_display = kyear_2-1

        T_max_kyr1 = np.max(T_grad1)
        T_max_kyr2 = np.max(T_grad2)

        lat = np.rad2deg(np.arcsin(x))
      
        fig,axs = plt.subplots(ncols = 3, figsize = (12,8))

        if kyr1_display == 0:
          plt.suptitle('Maximum Transport {} = {:.2e}, Maximum Transport {}kyr = {:.2e}, Maximum Transport Difference = {:.2e} \n Obliquity {} = {:.2f}deg, Obliquity {} = {:.2f}deg'.format("Modern", T_max_kyr1, kyr2_display, T_max_kyr2, (T_max_kyr1 - T_max_kyr2), "Modern", obliquity_1, kyr2_display, obliquity_2))
          axs[0].set_title('Seasonal Energy Transport {}'.format("Modern"))
          axs[1].set_title('Seasonal Energy Transport {}kyr'.format(kyr2_display))
          axs[2].set_title('Seasonal Difference')
        elif kyr2_display == 0:
          plt.suptitle('Maximum Transport {}kyr = {:.2e}, Maximum Transport {} = {:.2e}, Maximum Transport Difference = {:.2e}, \n Obliquity {} = {:.2f}deg, Obliquity {} = {:.2f}deg'.format(kyr1_display, T_max_kyr1, "Modern", T_max_kyr2, (T_max_kyr1 - T_max_kyr2), kyr1_display, obliquity_1, "Modern", obliquity_2))
          axs[1].set_title('Seasonal Energy Transport {}'.format("Modern"))
          axs[0].set_title('Seasonal Energy Transport {}kyr'.format(kyr1_display))
          axs[2].set_title('Seasonal Difference')
        else:
          plt.suptitle('Global Mean Insolation {}kyr = {:.2e}, Global Mean Insolation {}kyr = {:.2e}, Global Mean Difference = {:.2e} \n Obliquity {} = {:.2f}deg, Obliquity {} = {:.2f}deg'.format(kyr1_display, T_max_kyr1, kyr2_display, T_max_kyr2, (T_max_kyr1 - T_max_kyr2), kyr1_display, obliquity_1, kyr2_display, obliquity_2))
          axs[0].set_title('Seasonal Energy Transport {}kyr'.format(kyr1_display))
          axs[1].set_title('Seasonal Energy Transport {}kyr'.format(kyr2_display))
          axs[2].set_title('Seasonal Difference')

        contour_1 = axs[0].contourf(dur_plt,lat,T_grad1.T,np.arange(bar_min,bar_max,bar_step), extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_2 = axs[1].contourf(dur_plt,lat,geo_wind_1,np.arange(bar_min,bar_max,bar_step), extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_3 = axs[2].contourf(dur_plt,lat,trans_difference,np.arange(plot_diff_min,plot_diff_max,plot_diff_step), extend = 'both', cmap=plt.get_cmap('bwr'))
        
        plt.colorbar(contour_1,ax = axs[0])
        plt.colorbar(contour_2,ax = axs[1])

        axs[0].set_xlabel('days')
        axs[1].set_xlabel('days')
        axs[2].set_xlabel('days')
        axs[0].set_ylabel('sin(φ)')

      if subchart == "annual":

        tfin, Efin, Tfin_annual_kyear1, T0fin, ASRfin, S_annual_kyear1, Tg, mean_S_kyear1, OLR, S_kyear1, E_transport_kyr1, T_grad1, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
        tfin, Efin, Tfin_annual_kyear2, T0fin, ASRfin, S_annual_kyear2, Tg, mean_S_kyear2, OLR, S_kyear2, E_transport_kyr2, T_grad2, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2
        lat_grid = np.rad2deg(np.arcsin(x))

        T_max_kyr1 = np.max(T_grad1)
        T_max_kyr2 = np.max(T_grad2)
        T_min_kyr1 = np.min(T_grad1)
        T_min_kyr2 = np.min(T_grad2)

        T_grad1 = list(T_grad1)
        where_min_1 = T_grad1.index(T_min_kyr1)
        where_min_1 = lat_grid[where_min_1]
        print('kyr1',where_min_1)

        T_grad2 = list(T_grad2)
        where_min_2 = T_grad2.index(T_min_kyr2)
        where_min_2 = lat_grid[where_min_2]
        print('kyr2',where_min_2)

        kyear0_1,ecc0_1,long_peri_1,obliquity_1 = Orbital_Insolation().display_orbit(kyear_1)
        kyear0_2,ecc0_2,long_peri_2,obliquity_2 = Orbital_Insolation().display_orbit(kyear_2)

        dur_plt = np.linspace(0,365,nt)
        ax = np.linspace(-90,90,90)

        T_grad2 = np.array(T_grad2)
        T_grad1 = np.array(T_grad1)

        trans_difference = T_grad1 - T_grad2

        kyr1_display = kyear_1-1
        kyr2_display = kyear_2-1

        change = list(zip(trans_difference[0:45], lat_grid[0:45]))

        fig,axs = plt.subplots(ncols = 3, figsize = (12,8))

        plt.suptitle('Maximum Temp Change {}kyr = {:.2e}, Maximum Temp Change {}kyr = {:.2e}, Maximum Kyr Difference = {:.2e}'.format(kyr1_display, T_max_kyr1, kyr2_display, T_max_kyr2, (T_max_kyr1 - T_max_kyr2)))

        if kyr1_display == 0:
          plt.suptitle('Maximum Temp Change {} = {:.2e}, Maximum Temp Change {}kyr = {:.2e}, Maximum Kyr Difference = {:.2e} \n Obliquity {}kyr = {:.2f}deg, Obliquity {}kyr = {:.2f}deg'.format("Modern", T_max_kyr1, kyr2_display, T_max_kyr2, (T_max_kyr1 - T_max_kyr2), "Modern", obliquity_1, kyr2_display, obliquity_2))
          axs[0].set_title('Annual Temp Gradient {}'.format("Modern"))
          axs[1].set_title('Annual Temp Gradient {}kyr'.format(kyr2_display))
          axs[2].set_title('Annual Difference')
        elif kyr2_display == 0:
          plt.suptitle('Maximum Temp Change {}kyr = {:.2e}, Maximum Temp Change {} = {:.2e}, Maximum Kyr Difference = {:.2e}, \n Obliquity {} = {:.2f}deg, Obliquity {} = {:.2f}deg'.format(kyr1_display, T_max_kyr1, "Modern", T_max_kyr2, (T_max_kyr1 - T_max_kyr2), kyr1_display, obliquity_1, "Modern", obliquity_2))
          axs[1].set_title('Annual Temp Gradient {}'.format("Modern"))
          axs[0].set_title('Annual Temp Gradient {}kyr'.format(kyr1_display))
          axs[2].set_title('Annual Difference')
        else:
          plt.suptitle('Maximum Temp Change {}kyr = {:.2e}, Maximum Temp Change {}kyr = {:.2e}, Maximum Kyr Difference = {:.2e} \n Obliquity {} = {:.2f}deg, Obliquity {} = {:.2f}deg'.format(kyr1_display, T_max_kyr1, kyr2_display, T_max_kyr2, (T_max_kyr1 - T_max_kyr2), kyr1_display, obliquity_1, kyr2_display, obliquity_2))
          axs[0].set_title('Annual Temp Gradient {}kyr'.format(kyr1_display))
          axs[1].set_title('Annual Temp Gradient {}kyr'.format(kyr2_display))
          axs[2].set_title('Annual Difference')

        axs[0].plot(lat_grid,T_grad1)
        axs2 = axs[0].twinx()
        axs2.plot(lat_grid, Tfin_annual_kyear1, color = 'orange')
        axs[1].plot(lat_grid,T_grad1)
        axs[2].plot(lat_grid,trans_difference)
        
        axs[0].set_xlabel('latitude')
        axs[1].set_xlabel('latitude')
        axs[2].set_xlabel('latitude')
        axs[0].set_ylabel('Temperature Gradient')

    elif chart == 15: #albedo, insolaiton, and temp

      tfin_1, Efin_1, Tfin_1, T0fin_1, ASRfin_1, S_1, Tg_1, mean_S_1, OLR_1, S_kyear_1, E_transport_1, T_grad_1, alpha_1, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
      tfin_2, Efin_2, Tfin_2, T0fin_2, ASRfin_2, S_2, Tg_2, mean_S_2, OLR_2, S_kyear_2, E_transport_2, T_grad_2, alpha_2, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

      lat = np.rad2deg(np.arcsin(x))

      extrema_2 = Helper_Functions().find_extrema(grid, T_grad_2, lat, 'min')
      extrema_1 = Helper_Functions().find_extrema(grid, T_grad_1, lat, 'min')
      ice_line_1 = Helper_Functions().find_zeros(grid, Tfin_1, lat, 'both')
      ice_line_2 = Helper_Functions().find_zeros(grid, Tfin_2, lat, 'both')
      wind_extrema_1 = Helper_Functions().find_extrema(grid, geo_wind_1, lat, 'max')
      wind_extrema_2 = Helper_Functions().find_extrema(grid, geo_wind_2, lat, 'max')

      fig,axs = plt.subplots(figsize = (8,4))


      plt.suptitle('extrema_1 : {} \n extrema_2 : {} \n wind_extrema_1 : {} \n wind_extrema_2 : {}'.format(extrema_1, extrema_2, wind_extrema_1, wind_extrema_2))
      axs.plot(lat,T_grad_2, color = 'blue', label = "Obl Max")
      axs_1 = axs.twinx()
      axs_1.plot(lat,T_grad_1, color = "red", label = 'Obl Min')
      axs.set_ylabel('Temperature (deg C)') 
      axs_1.set_ylabel('Temperature Difference (deg C)') 
      axs.set_xlabel("Lattitude")
      axs.set_title("MEBM Temp and Temp Gradient vs Latitude")
      axs_2 = axs.twinx()
      axs_2.plot(lat, Tfin_1, color = 'green', label = 'Temp')
      axs_3 = axs.twinx()
      axs_3.plot(lat,geo_wind_1)
      axs_3.plot(lat,geo_wind_2)
      axs.legend(loc = (0.813,0.88))
      axs_1.legend()

    elif chart == 16:

      tfin_1, Efin_1, Tfin_1, T0fin_1, ASRfin_1, S_1, Tg_1, mean_S_1, OLR_1, S_kyear_1, E_transport_1, T_grad_1, alpha_1, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
      tfin_2, Efin_2, Tfin_2, T0fin_2, ASRfin_2, S_2, Tg_2, mean_S_2, OLR_2, S_kyear_2, E_transport_2, T_grad_2, alpha_2, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2
     
      lat = np.rad2deg(np.arcsin(x))
      extrema_1 = Helper_Functions().find_extrema(grid, T_grad_1, lat, 'max')
      extrema_2 = Helper_Functions().find_extrema(grid, T_grad_2, lat, 'max')

      wind_peak_1 = Helper_Functions().find_extrema(grid, geo_wind_1[0:45], lat, 'max')
      wind_peak_2 = Helper_Functions().find_extrema(grid, geo_wind_2[0:45], lat, 'max')

      wind_diff = geo_wind_1 - geo_wind_2
      T_grad_diff = T_grad_1 - T_grad_2

      freeze = Helper_Functions().find_zeros(grid, Tfin_1,lat, any ="both")
      
      fig,axs = plt.subplots(figsize = (7,8))

      axs.set_title('Min Obl Tgrad peak at {:.2f}, Max Obl Tgrad peak at {:.2f} \n Min Obl Tgrad Mag {:.2f}, Max Obl Tgrad Mag {:.2f} \n Min obl wind peak : {:.2f}, Max obl wind peak 2 : {:.2f}'.format(extrema_1, extrema_2, np.max(T_grad_1), np.max(T_grad_2), wind_peak_1, wind_peak_2))
      axs.plot(lat,T_grad_1, label = 'Min obl T grad', color = 'blue')
      axs.plot(lat,T_grad_2, label = 'Max obl T grad', color = 'red')
     
      axs_1 = axs.twinx()
      axs_1.plot(lat, geo_wind_1, label = 'Min obl wind speed', color = 'green')
      axs_1.plot(lat, geo_wind_2, label = 'Max obl wind speed', color = 'orange')
      axs_1.set_ylabel('wind speed')
  
      axs.set_ylabel("dT/dy")
      axs.set_xlabel('latitude')

      lines, labels = axs.get_legend_handles_labels()
      lines2, labels2 = axs_1.get_legend_handles_labels()
      axs_1.legend(lines + lines2, labels + labels2, loc=0) 


      axs.set_xlim(-90,0)
    
      plt.tight_layout()

    elif chart == 17: # obl sensitivity w/ GMT-xaxis

      x_range, GMT_ouput, iceline_output, S65_ins, S65_temp, dASRdalpha, geo_wind_max_lat, t_grad_max_lat, WSC_max_lat, CO2_val = experiment().orbit_and_CO2_suite(x_type = 'obl')

      fig, axs_2 = plt.subplots()

  
      axs_2.plot(GMT_ouput, iceline_output, color = 'blue', label = 'mean ice edge')
      axs_2.plot(GMT_ouput, geo_wind_max_lat, label = 'Zwind max lat', color = 'red')
      axs_2.plot(GMT_ouput, t_grad_max_lat, label = 'tgrad max lat', color = 'orange')
      axs_2.plot(GMT_ouput, WSC_max_lat, label = 'WSC max lat', color = 'purple')
      axs_2.legend()
     
     
      axs_2.set_xlabel('GMT [C]')
      axs_2.set_ylabel('latitude')
      axs_2.set_title('GMT vs S Hemi Mean Ice Edge, Tgrad, Zwind Max, and Wind Stress Curl'.format(CO2_val))

      plt.savefig('ForcingSuite.jpg')

    elif chart == 18: #obliquity sensitivity plots

      x_range, obl_sense_GMT, obl_sensitivity_ASR, GMT_obl_max, GMT_obl_min, EQP_obl_max, EQP_obl_min, packed_icelines, packed_Tgrad, full_tgrad_obl_max, full_tgrad_obl_min = np.load('HD_obl_analysis_dynamic.npy', allow_pickle=True)
      mean_icelines_obl_max, mean_icelines_obl_min, max_icelines_obl_max, max_icelines_obl_min, min_icelines_obl_max, min_icelines_obl_min = packed_icelines
      Tgrad_30_40_min, Tgrad_30_40_max, Tgrad_40_50_min, Tgrad_40_50_max, Tgrad_50_60_min, Tgrad_50_60_max, Tgrad_60_70_min, Tgrad_60_70_max = packed_Tgrad
      
      x_range_nof, obl_sense_GMT_nof, obl_sensitivity_ASR_nof, GMT_obl_max_nof, GMT_obl_min_nof, EQP_obl_max_nof, EQP_obl_min_nof, packed_icelines_nof, packed_Tgrad_nof, full_tgrad_obl_max_nof, full_tgrad_obl_min_nof = np.load('HD_obl_analysis_nofeedback.npy', allow_pickle=True)
      mean_icelines_obl_max_nof, mean_icelines_obl_min_nof, max_icelines_obl_max_nof, max_icelines_obl_min_nof, min_icelines_obl_max_nof, min_icelines_obl_min_nof = packed_icelines_nof
      Tgrad_30_40_min_nof, Tgrad_30_40_max_nof, Tgrad_40_50_min_nof, Tgrad_40_50_max_nof, Tgrad_50_60_min_nof, Tgrad_50_60_max_nof, Tgrad_60_70_min_nof, Tgrad_60_70_max_nof = packed_Tgrad_nof

      x_range_static, obl_sense_GMT_static, obl_sensitivity_ASR_static, GMT_obl_max_static, GMT_obl_min_static, EQP_obl_max_static, EQP_obl_min_static, packed_icelines_static, packed_Tgrad_static, full_tgrad_obl_max_static, full_tgrad_obl_min_static = np.load('HD_obl_analysis_static.npy', allow_pickle=True)
      mean_icelines_obl_max_static, mean_icelines_obl_min_static, max_icelines_obl_max_static, max_icelines_obl_min_static, min_icelines_obl_max_static, min_icelines_obl_min_static = packed_icelines_static
      Tgrad_30_40_min_static, Tgrad_30_40_max_static, Tgrad_40_50_min_static, Tgrad_40_50_max_static, Tgrad_50_60_min_static, Tgrad_50_60_max_static, Tgrad_60_70_min_static, Tgrad_60_70_max_static = packed_Tgrad_static

      ### Comment in to re-generate output ###
      # output_LGM = Model_Class.model(self, 1, experiment().config, experiment().Ti, CO2_ppm = 190, D = None, F = 0, moist = 1, albT = 3, seas = 1, thermo = 0, kyear = 21)
      # output_modern = Model_Class.model(self, 1, experiment().config, experiment().Ti, CO2_ppm = 280, D = None, F = 0, moist = 1, albT = 3, seas = 1, thermo = 0, kyear = 1)
      # output_dryas = Model_Class.model(self, 1, experiment().config, experiment().Ti, CO2_ppm = 240, D = None, F = 0, moist = 1, albT = 3, seas = 1, thermo = 0, kyear = 12)

      from CESM_data import pdf_analysis

      tfin_1, Efin_1, Tfin_1, T0fin_1, ASRfin_1, S_def_seas_1, Tg_1, mean_S_def_1, OLR_1, S_kyear_1, E_transport_1, T_grad_1, alpha_1, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = np.load('LGM_output.npy', allow_pickle=True)
      tfin_2, Efin_2, Tfin_2, T0fin_2, ASRfin_2, S_def_seas_2, Tg_2, mean_S_def_2, OLR_2, S_kyear_2, E_transport_2, T_grad_2, alpha_2, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = np.load('modern_output.npy', allow_pickle=True)
      tfin_3, Efin_3, Tfin_3, T0fin_3, ASRfin_3, S_def_seas_3, Tg_3, mean_S_def_3, OLR_3, S_kyear_3, E_transport_3, T_grad_3, alpha_3, geo_wind_3, ice_lines_3, Albfin_3, OLRfin_3, CO2_3, wind_stress_curl_3 = np.load('dryas_output.npy', allow_pickle=True)

      pdf1_1, pdf2_1, pdf3_1, x_range_1, array_vals_1 = pdf_analysis(T_grad_1.T, experiment().lat_deg, ice_lines_1[3])
      pdf1_2, pdf2_2, pdf3_2, x_range_2, array_vals_2 = pdf_analysis(T_grad_2.T, experiment().lat_deg, ice_lines_2[3])
      pdf1_3, pdf2_3, pdf3_3, x_range_3, array_vals_3 = pdf_analysis(T_grad_3.T, experiment().lat_deg, ice_lines_3[3])


      ### Comment in to re-generate output ###

      # obl_analysis_dynamic = x_range, obl_sense_GMT, obl_sensitivity_ASR, GMT_obl_max, GMT_obl_min, EQP_obl_max, EQP_obl_min, packed_icelines, packed_Tgrad, full_tgrad_obl_max, full_tgrad_obl_min
      # np.save('HD_obl_analysis_dynamic', obl_analysis_dynamic)

      # obl_analysis_nofeedback = x_range_nof, obl_sense_GMT_nof, obl_sensitivity_ASR_nof, GMT_obl_max_nof, GMT_obl_min_nof, EQP_obl_max_nof, EQP_obl_min_nof, packed_icelines_nof, packed_Tgrad_nof, full_tgrad_obl_max_nof, full_tgrad_obl_min_nof
      # np.save('HD_obl_analysis_nofeedback', obl_analysis_nofeedback)

      # obl_analysis_static = x_range_static, obl_sense_GMT_static, obl_sensitivity_ASR_static, GMT_obl_max_static, GMT_obl_min_static, EQP_obl_max_static, EQP_obl_min_static, packed_icelines_static, packed_Tgrad_static, full_tgrad_obl_max_static, full_tgrad_obl_min_static
      # np.save('HD_obl_analysis_static', obl_analysis_static)

      # np.save('LGM_output.npy', output_LGM)
      # np.save('modern_output.npy', output_modern)
      # np.save('dryas_output.npy', output_dryas)

      ###-----------------------------------------###

      fig,axs = plt.subplots(4,8, figsize = (12,8))

      gs = axs[0,0].get_gridspec()

      sensitivity_ax = fig.add_subplot(gs[0:4,0:2])
      iceline_ax = fig.add_subplot(gs[0:2,2:5])
      t_grad_ax = fig.add_subplot(gs[0:2,5:8])
      LGM_ax = fig.add_subplot(gs[2:4,2:4])
      modern_ax = fig.add_subplot(gs[2:4,6:8])
      dryas_ax = fig.add_subplot(gs[2:4,4:6])


      iceline_ax.plot(x_range, mean_icelines_obl_max, label = 'Mean Ice Extent ε Max')
      iceline_ax.plot(x_range, mean_icelines_obl_min, label = 'Mean Ice Extent ε Min', color = 'red')

      iceline_ax.plot(x_range, max_icelines_obl_max, label = 'Max Ice Extent ε Max', color = 'cyan', alpha=0.4)
      iceline_ax.plot(x_range, max_icelines_obl_min, label = 'Max Ice Extent ε Min', color = 'lightsalmon', alpha=0.4)

      iceline_ax.plot(x_range, min_icelines_obl_max, label = 'Min Ice Extent ε Max', color = 'dodgerblue', alpha=0.4)
      iceline_ax.plot(x_range, min_icelines_obl_min, label = 'Min Ice Extent ε Min', color = 'orangered', alpha=0.4)

      iceline_ax.fill_between(x_range,mean_icelines_obl_max,mean_icelines_obl_min, color = 'gray', alpha = 0.7)
      iceline_ax.fill_between(x_range,max_icelines_obl_max,max_icelines_obl_min, color = 'gray', alpha = 0.2)
      iceline_ax.fill_between(x_range,min_icelines_obl_max,min_icelines_obl_min, color = 'gray', alpha = 0.2)

      iceline_ax.set_ylabel('Latitude (φ)')
      iceline_ax.set_xlabel('Atmospheric CO2 ppm')
      iceline_ax.set_title('Sea Ice Edge v CO2 & Minimum/Maximum Obliquity')

      
      
      LGM_ax.plot(x_range_1,pdf1_1, color = 'blue', label = 'over ice')
      LGM_ax.plot(x_range_1,pdf2_1, color = 'red', label = 'over water')
      LGM_ax.plot(x_range_1,pdf3_1, color = 'green',label = 'at ice edge')
      LGM_ax.set_xlabel('Meridional Temperature Gradient [ΔK/Δφ]')
      LGM_ax.set_ylabel('Probability Density')
      LGM_ax.set_title('LGM Temperature Gradients \n Mean Sea Ice Edge at {:.2f}'.format(ice_lines_1[2]))
      LGM_ax.axvline(1, linestyle = 'dotted', color = 'black')
      LGM_ax.legend(loc = 'upper left')

      modern_ax.plot(x_range_2,pdf1_2, color = 'blue', label = 'over ice')
      modern_ax.plot(x_range_2,pdf2_2, color = 'red', label = 'over water')
      modern_ax.plot(x_range_2,pdf3_2, color = 'green',label = 'at ice edge')
      modern_ax.set_xlabel('Meridional Temperature Gradient [ΔK/Δφ]')
      modern_ax.set_title('Modern Temperature Gradients \n Mean Sea Ice Edge at {:.2f}'.format(ice_lines_2[2]))
      modern_ax.axvline(1, linestyle = 'dotted', color = 'black')
      modern_ax.legend()


      dryas_ax.plot(x_range_3,pdf1_3, color = 'blue', label = 'over ice')
      dryas_ax.plot(x_range_3,pdf2_3, color = 'red', label = 'over water')
      dryas_ax.plot(x_range_3,pdf3_3, color = 'green',label = 'at ice edge')
      dryas_ax.set_xlabel('Meridional Temperature Gradient [ΔK/Δφ]')
      dryas_ax.set_title('Early Holocene Temperature Gradients \n Mean Sea Ice Edge at {:.2f}'.format(ice_lines_3[2]))
      dryas_ax.axvline(1, linestyle = 'dotted', color = 'black')
      dryas_ax.legend()

      
      t_grad_ax.plot(x_range, EQP_obl_max, label = '0°-90°S ε Max', color = 'blue')
      t_grad_ax.plot(x_range, EQP_obl_min, label = '0°-90°S ε Min', color = 'red')

      t_grad_ax.fill_between(x_range,EQP_obl_max,EQP_obl_min, color = 'gray', alpha = 0.7)


      t_grad_ax.set_xlabel('Atmospheric CO2 ppm')
      t_grad_ax.set_ylabel('Meridional Temperature Gradients [ΔK/Δφ]')
      t_grad_ax.set_title('Equator-Pole Temperature Gradient v CO2')

      if True == True:
        axs[0,0].remove()
        axs[0,1].remove()
        axs[0,2].remove()
        axs[0,3].remove()
        axs[0,4].remove()
        axs[0,5].remove()
        axs[0,6].remove()
        axs[0,7].remove()
        axs[1,0].remove()
        axs[1,1].remove()
        axs[1,2].remove()
        axs[1,3].remove()
        axs[1,4].remove()
        axs[1,5].remove()
        axs[1,6].remove()
        axs[1,7].remove()
        axs[2,0].remove()
        axs[2,1].remove()
        axs[2,2].remove()
        axs[2,3].remove()
        axs[2,4].remove()
        axs[2,5].remove()
        axs[2,6].remove()
        axs[2,7].remove()
        axs[3,0].remove()
        axs[3,1].remove()
        axs[3,2].remove()
        axs[3,3].remove()
        axs[3,4].remove()
        axs[3,5].remove()
        axs[3,6].remove()
        axs[3,7].remove()
    
      fig.suptitle('GMT, Sea Ice, and Meridional Temperature Gradient Responses to Radiative Forcings in Idealized EBM')
      fig.tight_layout()

      plt.savefig('ForcingSuite.png')

    elif chart == 19: # CAM and MEBM regression compare

      tfin, Efin, Tfin, T0fin, ASRfin, S_def_seas, Tg, mean_S_def, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
      lat = np.rad2deg(np.arcsin(x))

      from CESM_data import CAM_vars

      CAM_T_grad , CAM_Z_wind, CAM_surf_temp, CAM_lat = CAM_vars

      fig, axs = plt.subplots(nrows = 3)

      axs[0].plot(lat,T_grad, color = 'green', label = 'Tgrad')
      axs1 = axs[0].twinx()
      axs1.plot(lat, geo_wind_1, color = 'blue')
      axs[0].set_title('MEBM Tgrad and Interpolated Zwind v Lat')
      axs[0].set_ylabel('T grad [C/km]')
      axs1.set_ylabel('Z wind [m/s]')
      axs[0].legend()      

      axs[1].plot(CAM_lat,CAM_T_grad, color = 'green', label = 'Tgrad')
      axs2 = axs[1].twinx()
      axs2.plot(CAM_lat, CAM_Z_wind, color = 'blue')
      axs[1].set_title('GCM Tgrad and Zwind v Lat')
      axs[1].set_ylabel('T grad [C/km]')
      axs2.set_ylabel('Z wind [m/s]')
      axs[2].set_xlabel('latitude')
      axs[1].legend()

      axs[2].plot(lat,Tfin, color = 'blue', label = 'MEBM')
      axs[2].plot(CAM_lat,CAM_surf_temp, color = 'red', label = 'GCM')
      axs[2].set_ylabel('Surface Temp [C]')
      axs[2].set_title('MEBM v GCM Surface Temperatures')
      axs[2].legend()

    elif chart == 20: # contour plot WSC, wind, and Tgrad

      tfin, Efin, Tfin, T0fin, ASRfin, S_def_seas, Tg, mean_S_def, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1

  
      lat = np.rad2deg(np.arcsin(x))

      fig, axs = plt.subplots(ncols = 3)

      contour_1 = axs[0].contourf(np.linspace(0,365,nt),lat,wind_stress_curl_1,np.arange(-.06,.06,0.01), extend = 'both',cmap=plt.get_cmap('bwr'))
      contour_2 = axs[1].contourf(np.linspace(0,365,nt),lat,geo_wind_1,np.arange(-15,15,0.01), extend = 'both',cmap=plt.get_cmap('bwr'))
      contour_3 = axs[2].contourf(np.linspace(0,365,nt),lat,T_grad,np.arange(-1,1,0.1), extend = 'both',cmap=plt.get_cmap('bwr'))

      ice_contour = axs[0].plot(np.linspace(0,365,nt),ice_lines_1[3], color = 'black', label = 'Sea Ice Extent')
      ice_contour = axs[1].plot(np.linspace(0,365,nt),ice_lines_1[3], color = 'black', label = 'Sea Ice Extent')
      ice_contour = axs[2].plot(np.linspace(0,365,nt),ice_lines_1[3], color = 'black', label = 'Sea Ice Extent')

      plt.colorbar(contour_1,ax = axs[0])
      plt.colorbar(contour_2,ax = axs[1])
      plt.colorbar(contour_3,ax = axs[2])

    elif chart == 21: #PDF analysis on model output

      from CESM_data import pdf_analysis
     
      tfin_1, Efin_1, Tfin_1, T0fin_1, ASRfin_1, S_def_seas_1, Tg_1, mean_S_def_1, OLR_1, S_kyear_1, E_transport_1, T_grad_1, alpha_1, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
      tfin_2, Efin_2, Tfin_2, T0fin_2, ASRfin_2, S_def_seas_2, Tg_2, mean_S_def_2, OLR_2, S_kyear_2, E_transport_2, T_grad_2, alpha_2, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2


      pdf1_1, pdf2_1, pdf3_1, x_range_1, array_vals_1 = pdf_analysis(wind_stress_curl_1.T, experiment().lat_deg, ice_lines_1[3])
      pdf1_2, pdf2_2, pdf3_2, x_range_2, array_vals_2 = pdf_analysis(wind_stress_curl_2.T, experiment().lat_deg, ice_lines_2[3])

      fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize = (10,6))

      axs[0,0].plot(x_range_1,pdf1_1, color = 'blue', label = 'over ice')
      axs[0,0].plot(x_range_1,pdf2_1, color = 'red', label = 'over water')
      axs[0,0].plot(x_range_1,pdf3_1, color = 'green',label = 'at ice edge')
      axs[0,0].set_xlabel('Wind Stress Curl [N / m^2 deg]')
      axs[0,0].set_ylabel('Probability Density')
      axs[0,0].set_title('Modern Wind Stress Curl \n mean sea ice edge at {:.2f}°S'.format(abs(ice_lines_1[2])))

      axs[0,0].legend()
      axs[0,0].hist(array_vals_1[1],color = 'red', label = 'sub arctic tgrad', density = True, alpha = 0.1, histtype='stepfilled')
      axs[0,0].hist(array_vals_1[0],color = 'blue', label = 'over ice tgrad', density = True, alpha = 0.1, histtype='stepfilled')
      axs[0,0].hist(array_vals_1[2],color = 'green', label = 'around icelines tgrad', density = True, alpha = 0.1, histtype='stepfilled')

      axs[1,0].plot(x_range_2,pdf1_2, color = 'blue', label = 'over ice')
      axs[1,0].plot(x_range_2,pdf2_2, color = 'red', label = 'over water')
      axs[1,0].plot(x_range_2,pdf3_2, color = 'green',label = 'at ice edge')
      axs[1,0].set_xlabel('Wind Stress Curl [N / m^2 deg]')
      axs[1,0].set_ylabel('Probability Density')
      axs[1,0].set_title('LGM Wind Stress Curl\n mean sea ice edge at {:.2f}'.format(ice_lines_2[2]))

      axs[1,0].legend()
      axs[1,0].hist(array_vals_2[1],color = 'red', label = 'sub arctic tgrad', density = True, alpha = 0.1, histtype='stepfilled')
      axs[1,0].hist(array_vals_2[0],color = 'blue', label = 'over ice tgrad', density = True, alpha = 0.1, histtype='stepfilled')
      axs[1,0].hist(array_vals_2[2],color = 'green', label = 'around icelines tgrad', density = True, alpha = 0.1, histtype='stepfilled')

      pdf1_3, pdf2_3, pdf3_3, x_range_3, array_vals_3 = pdf_analysis(geo_wind_1.T, experiment().lat_deg, ice_lines_1[3])
      pdf1_4, pdf2_4, pdf3_4, x_range_4, array_vals_4 = pdf_analysis(geo_wind_2.T, experiment().lat_deg, ice_lines_2[3])

      axs[0,1].plot(x_range_3,pdf1_3, color = 'blue', label = 'over ice')
      axs[0,1].plot(x_range_3,pdf2_3, color = 'red', label = 'over water')
      axs[0,1].plot(x_range_3,pdf3_3, color = 'green',label = 'at ice edge')
      axs[0,1].set_xlabel('Surface Wind Speed [m/s]')
      axs[0,1].set_ylabel('Probability Density')
      axs[0,1].set_title('Modern Surface Wind Speed \n mean sea ice edge at {:.2f}°S'.format(abs(ice_lines_1[2])))

      axs[0,1].legend()
      axs[0,1].hist(array_vals_3[1],color = 'red', label = 'sub arctic tgrad', density = True, alpha = 0.1, histtype='stepfilled')
      axs[0,1].hist(array_vals_3[0],color = 'blue', label = 'over ice tgrad', density = True, alpha = 0.1, histtype='stepfilled')
      axs[0,1].hist(array_vals_3[2],color = 'green', label = 'around icelines tgrad', density = True, alpha = 0.1, histtype='stepfilled')

      axs[1,1].plot(x_range_4,pdf1_4, color = 'blue', label = 'over ice')
      axs[1,1].plot(x_range_4,pdf2_4, color = 'red', label = 'over water')
      axs[1,1].plot(x_range_4,pdf3_4, color = 'green',label = 'at ice edge')
      axs[1,1].set_xlabel('Surface Wind Speed [m/s]')
      axs[1,1].set_ylabel('Probability Density')
      axs[1,1].set_title('LGM Surface Wind Speed \n mean sea ice edge at {:.2f}'.format(ice_lines_2[2]))

      axs[1,1].legend()
      axs[1,1].hist(array_vals_4[1],color = 'red', label = 'sub arctic tgrad', density = True, alpha = 0.1, histtype='stepfilled')
      axs[1,1].hist(array_vals_4[0],color = 'blue', label = 'over ice tgrad', density = True, alpha = 0.1, histtype='stepfilled')
      axs[1,1].hist(array_vals_4[2],color = 'green', label = 'around icelines tgrad', density = True, alpha = 0.1, histtype='stepfilled')


      pdf1_5, pdf2_5, pdf3_5, x_range_5, array_vals_5 = pdf_analysis(T_grad_1.T, experiment().lat_deg, ice_lines_1[3])
      pdf1_6, pdf2_6, pdf3_6, x_range_6, array_vals_6 = pdf_analysis(T_grad_2.T, experiment().lat_deg, ice_lines_2[3])

      axs[0,2].plot(x_range_5,pdf1_5, color = 'blue', label = 'over ice')
      axs[0,2].plot(x_range_5,pdf2_5, color = 'red', label = 'over water')
      axs[0,2].plot(x_range_5,pdf3_5, color = 'green',label = 'at ice edge')
      axs[0,2].set_xlabel('Surface Temperature Gradient [K/deg]')
      axs[0,2].set_ylabel('Probability Density')
      axs[0,2].set_title('Modern Surface Temp Gradients \n mean sea ice edge at {:.2f}°S'.format(abs(ice_lines_1[2])))

      axs[0,2].legend()
      axs[0,2].hist(array_vals_5[1],color = 'red', label = 'sub arctic tgrad', density = True, alpha = 0.1, histtype='stepfilled')
      axs[0,2].hist(array_vals_5[0],color = 'blue', label = 'over ice tgrad', density = True, alpha = 0.1, histtype='stepfilled')
      axs[0,2].hist(array_vals_5[2],color = 'green', label = 'around icelines tgrad', density = True, alpha = 0.1, histtype='stepfilled')

      axs[1,2].plot(x_range_6,pdf1_6, color = 'blue', label = 'over ice')
      axs[1,2].plot(x_range_6,pdf2_6, color = 'red', label = 'over water')
      axs[1,2].plot(x_range_6,pdf3_6, color = 'green',label = 'at ice edge')
      axs[1,2].set_xlabel('Surface Temperature Gradient [K/deg]')
      axs[1,2].set_ylabel('Probability Density')
      axs[1,2].set_title('LGM Surface Temp Gradients \n mean sea ice edge at {:.2f}'.format(ice_lines_2[2]))

      axs[1,2].legend()
      axs[1,2].hist(array_vals_6[1],color = 'red', label = 'sub arctic tgrad', density = True, alpha = 0.1, histtype='stepfilled')
      axs[1,2].hist(array_vals_6[0],color = 'blue', label = 'over ice tgrad', density = True, alpha = 0.1, histtype='stepfilled')
      axs[1,2].hist(array_vals_6[2],color = 'green', label = 'around icelines tgrad', density = True, alpha = 0.1, histtype='stepfilled')

      fig.suptitle('PDF of Surface Wind Stress Curl, Wind Speed, & Temperature Gradients for Modern and LGM')
      fig.tight_layout()
      plt.savefig('PDF_analysis.png')

    elif chart == 22: #Insolation and Temperautre Difference Plots

      tfin, Efin, Tfin_kyear_1, T0fin, ASRfin, S_seas_kyear1, Tg, mean_S_kyear1, OLR, S_kyear1, E_transport_kyr1, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
      tfin, Efin, Tfin_kyear_2, T0fin, ASRfin, S_seas_kyear2, Tg, mean_S_kyear2, OLR, S_kyear2, E_transport_kyr2, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

      insolation_difference = S_seas_kyear2.T - S_seas_kyear1.T
      temperature_difference = Tfin_kyear_2 -Tfin_kyear_1

      dur_plt = np.linspace(0,365,nt)
      lat = np.rad2deg(np.arcsin(x))

      fig,axs = plt.subplots(ncols = 2, figsize = (12,8))

      contour_1 = axs[0].contourf(dur_plt,lat,insolation_difference.T,np.arange(-50,50,0.1), extend = 'both', cmap=plt.get_cmap('jet'))
      contour_2 = axs[1].contourf(dur_plt,lat,temperature_difference,np.arange(-1.5,1.5,0.01), extend = 'both', cmap=plt.get_cmap('bwr')) 

      plt.colorbar(contour_1,ax = axs[0])
      plt.colorbar(contour_2,ax = axs[1])

      axs[0].set_xlabel('time (days)')
      axs[1].set_xlabel('time (days)')

      axs[0].set_ylabel('latitude (ϕ)')

      plt.suptitle('Seasonal Response to Minimum-Maximum Obliquity Forcing')

      axs[0].set_title('TOA Insolation Difference \n Global Mean Insolation Difference: {:.2f}'.format(np.abs(np.mean(S_seas_kyear2-S_seas_kyear1))))
      axs[1].set_title('Temperature Difference \n Global Mean Temperature Difference: {:.2f}'.format(np.mean(Tfin_kyear_2-Tfin_kyear_1)))

      axs[0].text(1.2, 0.975, "(a)", transform=axs[0].transAxes,
        fontsize=16)
      axs[1].text(1.2, 0.975, "(b)", transform=axs[1].transAxes,
        fontsize=16)
      

      fig.savefig('MinMaxOrbParamDiff.png')

    elif chart == 23: #Obliquity Sensitivity

      x_range, obl_sense_GMT, obl_sensitivity_ASR, GMT_obl_max, GMT_obl_min, EQP_obl_max, EQP_obl_min, packed_icelines, packed_Tgrad, full_tgrad_obl_max, full_tgrad_obl_min = np.load('HD_obl_analysis_dynamic.npy', allow_pickle=True)
      mean_icelines_obl_max, mean_icelines_obl_min, max_icelines_obl_max, max_icelines_obl_min, min_icelines_obl_max, min_icelines_obl_min = packed_icelines
      Tgrad_30_40_min, Tgrad_30_40_max, Tgrad_40_50_min, Tgrad_40_50_max, Tgrad_50_60_min, Tgrad_50_60_max, Tgrad_60_70_min, Tgrad_60_70_max = packed_Tgrad
      
      x_range_nof, obl_sense_GMT_nof, obl_sensitivity_ASR_nof, GMT_obl_max_nof, GMT_obl_min_nof, EQP_obl_max_nof, EQP_obl_min_nof, packed_icelines_nof, packed_Tgrad_nof, full_tgrad_obl_max_nof, full_tgrad_obl_min_nof = np.load('HD_obl_analysis_nofeedback.npy', allow_pickle=True)
      mean_icelines_obl_max_nof, mean_icelines_obl_min_nof, max_icelines_obl_max_nof, max_icelines_obl_min_nof, min_icelines_obl_max_nof, min_icelines_obl_min_nof = packed_icelines_nof
      Tgrad_30_40_min_nof, Tgrad_30_40_max_nof, Tgrad_40_50_min_nof, Tgrad_40_50_max_nof, Tgrad_50_60_min_nof, Tgrad_50_60_max_nof, Tgrad_60_70_min_nof, Tgrad_60_70_max_nof = packed_Tgrad_nof

      x_range_static, obl_sense_GMT_static, obl_sensitivity_ASR_static, GMT_obl_max_static, GMT_obl_min_static, EQP_obl_max_static, EQP_obl_min_static, packed_icelines_static, packed_Tgrad_static, full_tgrad_obl_max_static, full_tgrad_obl_min_static = np.load('HD_obl_analysis_static.npy', allow_pickle=True)
      mean_icelines_obl_max_static, mean_icelines_obl_min_static, max_icelines_obl_max_static, max_icelines_obl_min_static, min_icelines_obl_max_static, min_icelines_obl_min_static = packed_icelines_static
      Tgrad_30_40_min_static, Tgrad_30_40_max_static, Tgrad_40_50_min_static, Tgrad_40_50_max_static, Tgrad_50_60_min_static, Tgrad_50_60_max_static, Tgrad_60_70_min_static, Tgrad_60_70_max_static = packed_Tgrad_static

      fig, axs = plt.subplots()

      axs.plot(x_range, obl_sense_GMT, label = 'Temperature Dependant Sea Ice', color = 'cadetblue')
      axs.plot(x_range, obl_sense_GMT_static, label = 'Static Sea Ice', color = 'lightcoral')
      axs.plot(x_range, obl_sense_GMT_nof, label = 'No Feedback', color = 'thistle')

      axs.set_ylabel('Sε [°C/°ε]')
      axs.set_xlabel('Atmospheric CO₂ [ppmv]')
      axs.set_title('Obliquity Induced ΔGMT v Atmospheric CO₂ concentrations')

      plt.legend()

      fig.savefig('ObliquitySensitivity.png')

    elif chart == 24: # Sea Ice Ext over CO2 & Obliquity

      
      x_range, obl_sense_GMT, obl_sensitivity_ASR, GMT_obl_max, GMT_obl_min, EQP_obl_max, EQP_obl_min, packed_icelines, packed_Tgrad, full_tgrad_obl_max, full_tgrad_obl_min = np.load('HD_obl_analysis_dynamic.npy', allow_pickle=True)
      mean_icelines_obl_max, mean_icelines_obl_min, max_icelines_obl_max, max_icelines_obl_min, min_icelines_obl_max, min_icelines_obl_min = packed_icelines
      Tgrad_30_40_min, Tgrad_30_40_max, Tgrad_40_50_min, Tgrad_40_50_max, Tgrad_50_60_min, Tgrad_50_60_max, Tgrad_60_70_min, Tgrad_60_70_max = packed_Tgrad
      
      x_range_nof, obl_sense_GMT_nof, obl_sensitivity_ASR_nof, GMT_obl_max_nof, GMT_obl_min_nof, EQP_obl_max_nof, EQP_obl_min_nof, packed_icelines_nof, packed_Tgrad_nof, full_tgrad_obl_max_nof, full_tgrad_obl_min_nof = np.load('HD_obl_analysis_nofeedback.npy', allow_pickle=True)
      mean_icelines_obl_max_nof, mean_icelines_obl_min_nof, max_icelines_obl_max_nof, max_icelines_obl_min_nof, min_icelines_obl_max_nof, min_icelines_obl_min_nof = packed_icelines_nof
      Tgrad_30_40_min_nof, Tgrad_30_40_max_nof, Tgrad_40_50_min_nof, Tgrad_40_50_max_nof, Tgrad_50_60_min_nof, Tgrad_50_60_max_nof, Tgrad_60_70_min_nof, Tgrad_60_70_max_nof = packed_Tgrad_nof

      x_range_static, obl_sense_GMT_static, obl_sensitivity_ASR_static, GMT_obl_max_static, GMT_obl_min_static, EQP_obl_max_static, EQP_obl_min_static, packed_icelines_static, packed_Tgrad_static, full_tgrad_obl_max_static, full_tgrad_obl_min_static = np.load('HD_obl_analysis_static.npy', allow_pickle=True)
      mean_icelines_obl_max_static, mean_icelines_obl_min_static, max_icelines_obl_max_static, max_icelines_obl_min_static, min_icelines_obl_max_static, min_icelines_obl_min_static = packed_icelines_static
      Tgrad_30_40_min_static, Tgrad_30_40_max_static, Tgrad_40_50_min_static, Tgrad_40_50_max_static, Tgrad_50_60_min_static, Tgrad_50_60_max_static, Tgrad_60_70_min_static, Tgrad_60_70_max_static = packed_Tgrad_static


      mean_sea_ice_area_obl_max = Helper_Functions().sea_ice_ext_to_area(mean_icelines_obl_max)
      mean_sea_ice_area_obl_min = Helper_Functions().sea_ice_ext_to_area(mean_icelines_obl_min)

      min_sea_ice_area_obl_max = Helper_Functions().sea_ice_ext_to_area(min_icelines_obl_max)
      min_sea_ice_area_obl_min = Helper_Functions().sea_ice_ext_to_area(min_icelines_obl_min)
      
      max_sea_ice_area_obl_max = Helper_Functions().sea_ice_ext_to_area(max_icelines_obl_max)
      max_sea_ice_area_obl_min = Helper_Functions().sea_ice_ext_to_area(max_icelines_obl_min)


      fig, axs = plt.subplots()

      axs.plot(x_range, mean_icelines_obl_max, label = 'Mean Ice Extent ε Max')
      axs.plot(x_range, min_icelines_obl_max, label = 'Min Ice Extent ε Max', color = 'dodgerblue', alpha=0.4)
      axs.plot(x_range, max_icelines_obl_max, label = 'Max Ice Extent ε Max', color = 'cyan', alpha=0.4)
      
      axs.plot(x_range, mean_icelines_obl_min, label = 'Mean Ice Extent ε Min', color = 'red')
      axs.plot(x_range, min_icelines_obl_min, label = 'Min Ice Extent ε Min', color = 'orangered', alpha=0.4)
      axs.plot(x_range, max_icelines_obl_min, label = 'Max Ice Extent ε Min', color = 'lightsalmon', alpha=0.4)

      axs.fill_between(x_range,mean_icelines_obl_max,mean_icelines_obl_min, color = 'gray', alpha = 0.7)
      axs.fill_between(x_range,max_icelines_obl_max,max_icelines_obl_min, color = 'gray', alpha = 0.2)
      axs.fill_between(x_range,min_icelines_obl_max,min_icelines_obl_min, color = 'gray', alpha = 0.2)

      axs.set_ylabel('Latitude of Sea Ice Extent [ϕ]')
      axs.set_xlabel('Atmospheric CO2 [ppmv]')
      axs.set_title('Sea Ice Edge v CO2 & Minimum/Maximum Obliquity')
      axs.legend(fontsize=10)

      fig.savefig('ObliquitySeaIceEdge.png')

    elif chart == 25: #Tgrad changes over different CO2 and Obliquity 

      
      x_range, obl_sense_GMT, obl_sensitivity_ASR, GMT_obl_max, GMT_obl_min, EQP_obl_max, EQP_obl_min, packed_icelines, packed_Tgrad, full_tgrad_obl_max, full_tgrad_obl_min = np.load('HD_obl_analysis_dynamic.npy', allow_pickle=True)
      mean_icelines_obl_max, mean_icelines_obl_min, max_icelines_obl_max, max_icelines_obl_min, min_icelines_obl_max, min_icelines_obl_min = packed_icelines
      Tgrad_30_40_min, Tgrad_30_40_max, Tgrad_40_50_min, Tgrad_40_50_max, Tgrad_50_60_min, Tgrad_50_60_max, Tgrad_60_70_min, Tgrad_60_70_max = packed_Tgrad
      
      x_range_nof, obl_sense_GMT_nof, obl_sensitivity_ASR_nof, GMT_obl_max_nof, GMT_obl_min_nof, EQP_obl_max_nof, EQP_obl_min_nof, packed_icelines_nof, packed_Tgrad_nof, full_tgrad_obl_max_nof, full_tgrad_obl_min_nof = np.load('HD_obl_analysis_nofeedback.npy', allow_pickle=True)
      mean_icelines_obl_max_nof, mean_icelines_obl_min_nof, max_icelines_obl_max_nof, max_icelines_obl_min_nof, min_icelines_obl_max_nof, min_icelines_obl_min_nof = packed_icelines_nof
      Tgrad_30_40_min_nof, Tgrad_30_40_max_nof, Tgrad_40_50_min_nof, Tgrad_40_50_max_nof, Tgrad_50_60_min_nof, Tgrad_50_60_max_nof, Tgrad_60_70_min_nof, Tgrad_60_70_max_nof = packed_Tgrad_nof

      x_range_static, obl_sense_GMT_static, obl_sensitivity_ASR_static, GMT_obl_max_static, GMT_obl_min_static, EQP_obl_max_static, EQP_obl_min_static, packed_icelines_static, packed_Tgrad_static, full_tgrad_obl_max_static, full_tgrad_obl_min_static = np.load('HD_obl_analysis_static.npy', allow_pickle=True)
      mean_icelines_obl_max_static, mean_icelines_obl_min_static, max_icelines_obl_max_static, max_icelines_obl_min_static, min_icelines_obl_max_static, min_icelines_obl_min_static = packed_icelines_static
      Tgrad_30_40_min_static, Tgrad_30_40_max_static, Tgrad_40_50_min_static, Tgrad_40_50_max_static, Tgrad_50_60_min_static, Tgrad_50_60_max_static, Tgrad_60_70_min_static, Tgrad_60_70_max_static = packed_Tgrad_static

      fig, axs = plt.subplots()

      axs.plot(x_range, EQP_obl_max, label = 'ε Max', color = 'blue')
      axs.plot(x_range, EQP_obl_min, label = 'ε Min', color = 'red')

      axs.plot(x_range, EQP_obl_max_nof, label = 'ε Max nof', color = 'lightblue')
      axs.plot(x_range, EQP_obl_min_nof, label = 'ε Min nof', color = 'pink')

      axs.plot(x_range, EQP_obl_max_static, label = 'ε Max static', color = 'darkblue')
      axs.plot(x_range, EQP_obl_min_static, label = 'ε Min static', color = 'darkred')

      axs.set_xlabel('Atmospheric CO2 ppm')
      axs.set_ylabel('Meridional Temperature Gradients [ΔK/Δφ]')
      axs.set_title('Equator-Pole Temperature Gradient v CO2')

      axs.legend(fontsize = 10)

      fig.savefig('ObliquityTgrad.png')

    elif chart == 26: # EBM Radiative PDF Analysis (LGM, Early Holocene, Modern)

      
      x_range, obl_sense_GMT, obl_sensitivity_ASR, GMT_obl_max, GMT_obl_min, EQP_obl_max, EQP_obl_min, packed_icelines, packed_Tgrad, full_tgrad_obl_max, full_tgrad_obl_min = np.load('HD_obl_analysis_dynamic.npy', allow_pickle=True)
      mean_icelines_obl_max, mean_icelines_obl_min, max_icelines_obl_max, max_icelines_obl_min, min_icelines_obl_max, min_icelines_obl_min = packed_icelines
      Tgrad_30_40_min, Tgrad_30_40_max, Tgrad_40_50_min, Tgrad_40_50_max, Tgrad_50_60_min, Tgrad_50_60_max, Tgrad_60_70_min, Tgrad_60_70_max = packed_Tgrad
      
      x_range_nof, obl_sense_GMT_nof, obl_sensitivity_ASR_nof, GMT_obl_max_nof, GMT_obl_min_nof, EQP_obl_max_nof, EQP_obl_min_nof, packed_icelines_nof, packed_Tgrad_nof, full_tgrad_obl_max_nof, full_tgrad_obl_min_nof = np.load('HD_obl_analysis_nofeedback.npy', allow_pickle=True)
      mean_icelines_obl_max_nof, mean_icelines_obl_min_nof, max_icelines_obl_max_nof, max_icelines_obl_min_nof, min_icelines_obl_max_nof, min_icelines_obl_min_nof = packed_icelines_nof
      Tgrad_30_40_min_nof, Tgrad_30_40_max_nof, Tgrad_40_50_min_nof, Tgrad_40_50_max_nof, Tgrad_50_60_min_nof, Tgrad_50_60_max_nof, Tgrad_60_70_min_nof, Tgrad_60_70_max_nof = packed_Tgrad_nof

      x_range_static, obl_sense_GMT_static, obl_sensitivity_ASR_static, GMT_obl_max_static, GMT_obl_min_static, EQP_obl_max_static, EQP_obl_min_static, packed_icelines_static, packed_Tgrad_static, full_tgrad_obl_max_static, full_tgrad_obl_min_static = np.load('HD_obl_analysis_static.npy', allow_pickle=True)
      mean_icelines_obl_max_static, mean_icelines_obl_min_static, max_icelines_obl_max_static, max_icelines_obl_min_static, min_icelines_obl_max_static, min_icelines_obl_min_static = packed_icelines_static
      Tgrad_30_40_min_static, Tgrad_30_40_max_static, Tgrad_40_50_min_static, Tgrad_40_50_max_static, Tgrad_50_60_min_static, Tgrad_50_60_max_static, Tgrad_60_70_min_static, Tgrad_60_70_max_static = packed_Tgrad_static

      from CESM_data import pdf_analysis

      tfin_1, Efin_1, Tfin_1, T0fin_1, ASRfin_1, S_def_seas_1, Tg_1, mean_S_def_1, OLR_1, S_kyear_1, E_transport_1, T_grad_1, alpha_1, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = np.load('LGM_output.npy', allow_pickle=True)
      tfin_2, Efin_2, Tfin_2, T0fin_2, ASRfin_2, S_def_seas_2, Tg_2, mean_S_def_2, OLR_2, S_kyear_2, E_transport_2, T_grad_2, alpha_2, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = np.load('modern_output.npy', allow_pickle=True)
      tfin_3, Efin_3, Tfin_3, T0fin_3, ASRfin_3, S_def_seas_3, Tg_3, mean_S_def_3, OLR_3, S_kyear_3, E_transport_3, T_grad_3, alpha_3, geo_wind_3, ice_lines_3, Albfin_3, OLRfin_3, CO2_3, wind_stress_curl_3 = np.load('dryas_output.npy', allow_pickle=True)

      pdf1_1, pdf2_1, pdf3_1, x_range_1, array_vals_1 = pdf_analysis(T_grad_1.T, experiment().lat_deg, ice_lines_1[3])
      pdf1_2, pdf2_2, pdf3_2, x_range_2, array_vals_2 = pdf_analysis(T_grad_2.T, experiment().lat_deg, ice_lines_2[3])
      pdf1_3, pdf2_3, pdf3_3, x_range_3, array_vals_3 = pdf_analysis(T_grad_3.T, experiment().lat_deg, ice_lines_3[3])

      fig, axs = plt.subplots(ncols = 3, figsize = (10,6))

      linethick = 2

      axs[0].plot(x_range_1,pdf1_1, color = 'thistle', label = 'over ice', linewidth=linethick)
      axs[0].plot(x_range_1,pdf2_1, color = 'lightcoral', label = 'over water', linewidth=linethick)
      axs[0].plot(x_range_1,pdf3_1, color = 'cadetblue',label = 'at ice edge', linewidth=linethick)
      axs[0].set_xlabel('Meridional Temperature Gradient [K/φ]')
      axs[0].set_ylabel('Probability Density')
      axs[0].set_title('LGM Surface MTGs \n Mean Sea Ice Edge at {:.2f}°S'.format(abs(ice_lines_1[2])))
      axs[0].axvline(1, linestyle = 'dotted', color = 'black')
      axs[0].legend(loc = 'upper left')

      axs[1].plot(x_range_3,pdf1_3, color = 'thistle', label = 'over ice', linewidth=linethick)
      axs[1].plot(x_range_3,pdf2_3, color = 'lightcoral', label = 'over water', linewidth=linethick)
      axs[1].plot(x_range_3,pdf3_3, color = 'cadetblue',label = 'at ice edge', linewidth=linethick)
      axs[1].set_xlabel('Meridional Temperature Gradient [K/φ]')
      axs[1].set_title('Early Holocene Surface MTGs \n Mean Sea Ice Edge at {:.2f}°S'.format(abs(ice_lines_3[2])))
      axs[1].axvline(1, linestyle = 'dotted', color = 'black')

      axs[2].plot(x_range_2,pdf1_2, color = 'thistle', label = 'over ice', linewidth=linethick)
      axs[2].plot(x_range_2,pdf2_2, color = 'lightcoral', label = 'over water', linewidth=linethick)
      axs[2].plot(x_range_2,pdf3_2, color = 'cadetblue',label = 'at ice edge', linewidth=linethick)
      axs[2].set_xlabel('Meridional Temperature Gradient [K/φ]')
      axs[2].set_title('Modern Surface MTGs \n Mean Sea Ice Edge at {:.2f}°S'.format(abs(ice_lines_2[2])))
      axs[2].axvline(1, linestyle = 'dotted', color = 'black')

      axs[0].text(0.89, 0.95, "(a)", transform=axs[0].transAxes,
        fontsize=16)
      axs[1].text(0.89, 0.95, "(b)", transform=axs[1].transAxes,
        fontsize=16)
      axs[2].text(0.89, 0.95, "(c)", transform=axs[2].transAxes,
        fontsize=16)

      plt.tight_layout()

      fig.savefig('EBM_Radiation_PDF.png')

    elif chart == 27: # chart in development

      x_range, obl_sense_GMT, obl_sensitivity_ASR, GMT_obl_max, GMT_obl_min, EQP_obl_max, EQP_obl_min, packed_icelines, packed_Tgrad, full_tgrad_obl_max, full_tgrad_obl_min = np.load('HD_obl_analysis_dynamic.npy', allow_pickle=True)
      mean_icelines_obl_max, mean_icelines_obl_min, max_icelines_obl_max, max_icelines_obl_min, min_icelines_obl_max, min_icelines_obl_min = packed_icelines
      Tgrad_30_40_min, Tgrad_30_40_max, Tgrad_40_50_min, Tgrad_40_50_max, Tgrad_50_60_min, Tgrad_50_60_max, Tgrad_60_70_min, Tgrad_60_70_max = packed_Tgrad
      
      x_range_nof, obl_sense_GMT_nof, obl_sensitivity_ASR_nof, GMT_obl_max_nof, GMT_obl_min_nof, EQP_obl_max_nof, EQP_obl_min_nof, packed_icelines_nof, packed_Tgrad_nof, full_tgrad_obl_max_nof, full_tgrad_obl_min_nof = np.load('HD_obl_analysis_nofeedback.npy', allow_pickle=True)
      mean_icelines_obl_max_nof, mean_icelines_obl_min_nof, max_icelines_obl_max_nof, max_icelines_obl_min_nof, min_icelines_obl_max_nof, min_icelines_obl_min_nof = packed_icelines_nof
      Tgrad_30_40_min_nof, Tgrad_30_40_max_nof, Tgrad_40_50_min_nof, Tgrad_40_50_max_nof, Tgrad_50_60_min_nof, Tgrad_50_60_max_nof, Tgrad_60_70_min_nof, Tgrad_60_70_max_nof = packed_Tgrad_nof

      x_range_static, obl_sense_GMT_static, obl_sensitivity_ASR_static, GMT_obl_max_static, GMT_obl_min_static, EQP_obl_max_static, EQP_obl_min_static, packed_icelines_static, packed_Tgrad_static, full_tgrad_obl_max_static, full_tgrad_obl_min_static = np.load('HD_obl_analysis_static.npy', allow_pickle=True)
      mean_icelines_obl_max_static, mean_icelines_obl_min_static, max_icelines_obl_max_static, max_icelines_obl_min_static, min_icelines_obl_max_static, min_icelines_obl_min_static = packed_icelines_static
      Tgrad_30_40_min_static, Tgrad_30_40_max_static, Tgrad_40_50_min_static, Tgrad_40_50_max_static, Tgrad_50_60_min_static, Tgrad_50_60_max_static, Tgrad_60_70_min_static, Tgrad_60_70_max_static = packed_Tgrad_static

      pass

    elif chart == 28: # surface meridional temperature gradeint contour plot for obliquity sensitivity forcings

      with open('obl_max_CO2runs_output.pkl', 'rb') as f:
            obl_max_CO2runs_output = pickle.load(f)
      with open('obl_min_CO2runs_output.pkl', 'rb') as f:
            obl_min_CO2runs_output = pickle.load(f)

      CO2_range = np.linspace(100,600,100)
      lat = np.rad2deg(np.arcsin(x))
      MTG_obl_max = []
      MTG_obl_min = []

      for i in range(len(obl_max_CO2runs_output)):

        MTG_at_CO2_val_oblmax = obl_max_CO2runs_output[i][11]
        MTG_at_CO2_val_oblmin = obl_min_CO2runs_output[i][11]

        MTG_obl_max.append(MTG_at_CO2_val_oblmax)
        MTG_obl_min.append(MTG_at_CO2_val_oblmin)


      MTG_obl_max = np.array(MTG_obl_max)
      MTG_obl_min = np.array(MTG_obl_min)

      MTG_diff = abs(MTG_obl_max) - abs(MTG_obl_min)


      fig, axs = plt.subplots(ncols = 3)

      contour_1 = axs[0].contourf(CO2_range,lat,MTG_diff.T,np.arange(-0.1,0.1,0.001), extend = 'max', cmap=plt.get_cmap('bwr'))
      contour_2 = axs[1].contourf(CO2_range,lat,MTG_obl_max.T,np.arange(-2,2,0.01), extend = 'max', cmap=plt.get_cmap('bwr'))
      contour_3 = axs[2].contourf(CO2_range,lat,MTG_obl_min.T,np.arange(-2,2,0.01), extend = 'max', cmap=plt.get_cmap('bwr'))

      plt.colorbar(contour_1,ax = axs[0], label = 'W/m²')
      plt.colorbar(contour_2,ax = axs[1], label = 'W/m²')
      plt.colorbar(contour_3,ax = axs[2], label = 'W/m²')

      fig.tight_layout()
      plt.savefig('MTG_contour_FCO2.png')

if __name__ == '__main__':
  experiment().main()
  #experiment().generate_table_2()
  #experiment().generate_sensitivity_table(run_type = 'no', orb_comp='obl')
  #experiment().generate_decomp_table()
  #experiment().orbit_and_CO2_suite(x_type = 'CO2')
  #Figures().figure(experiment().config, chart = 23)
  #Figures().figure(experiment().config, chart = 18, subchart='forcing')
 
  print("-------------------------------------------------")
  print("--- %s seconds ---" % (time.time() - start_time))
  print("-------------------------------------------------")
  print("")