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
from climlab.solar.orbital.long import OrbitalTable as OrbitalTable #The Laskar 2004 orbital data table 2Mya-present
from tabulate import tabulate


start_time = time.time()

mpl.rcParams['axes.titlesize'] = 10 # reset global fig properties
mpl.rcParams['legend.fontsize'] = 6 # reset global fig properties
mpl.rcParams['legend.title_fontsize'] = 6 # reset global fig properties
mpl.rcParams['xtick.labelsize'] = 8 # reset global fig properties
mpl.rcParams['ytick.labelsize'] = 8 # reset global fig properties
mpl.rcParams['ytick.right'] = True # reset global fig properties
mpl.rcParams['axes.labelsize'] = 8 # reset global fig properties

class experiment():

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

    elif grid == 1: #intermediate run // low resolution
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

  def analytics(self, type): #returns numerical information such as shape, average, and more

    tfin, Efin, Tfin, T0fin, ASRfin, S, Tg = self.main()

    if type == "shape":
      print('tfin is',np.shape(tfin))
      print('Efin is',np.shape(Efin))
      print('Tfin is',np.shape(Tfin))
      print('T0fin is',np.shape(T0fin))
      print('ASRfin is',np.shape(ASRfin))
      print('S is',np.shape(S))
      print('Tg is',np.shape(Tg))

    elif type == "avg":
      print('tfin is',np.mean(tfin))
      print('Efin is',np.mean(Efin))
      print('Tfin is',np.mean(Tfin))
      print('T0fin is',np.mean(T0fin))
      print('ASRfin is',np.mean(ASRfin))
      print('S is',np.mean(S))
      print('Tg is',np.mean(Tg))
    
    else:  
      print('tfin is',tfin)
      print('Efin is',Efin)
      print('Tfin is',Tfin)
      print('T0fin is',T0fin)
      print('ASRfin is',ASRfin)
      print('S is',S)
      print('Tg is',Tg)
  
  def main(self): #executes script, generates output, and figures

#--------------------Control-Panel---------------------------#
    On_Off = {'moist':1, 'albT':1, 'seas':1,'thermo':0}
    
    Orbitals_1 = {'obl':Helper_Functions().orbit_extrema('obl','min'), 'long':Helper_Functions().orbit_at_time('long',1), 'ecc':Helper_Functions().orbit_at_time('ecc',1)}
    Orbitals_2 = {'obl':Helper_Functions().orbit_extrema('obl','max'), 'long':Helper_Functions().orbit_at_time('long',1), 'ecc':Helper_Functions().orbit_at_time('ecc',1)}
    
    kyear = 'noforced'
    
    chart = 5
    
    kyear1 = 1#Helper_Functions().find_kyear_extrema('obl', 'min')
    CO2_ppm_1 = 280
    kyear2 = 2#Helper_Functions().find_kyear_extrema('obl', 'max')
    CO2_ppm_2 = 280
#------------------------------------------------------------#

    #Model_Class.model(self, 1, self.config, self.Ti, CO2_ppm = CO2_ppm_1, D = None, F = 0, moist = On_Off['moist'], albT = On_Off['albT'], seas = On_Off['seas'], thermo = On_Off['thermo'], kyear = kyear1, save_Alb= 'On')    

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

        np.save('control_1110_low_obl.npy', output_1)
        np.save('control_1110_high_obl.npy', output_2)

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

  def two_runs(self): #executes multiple main functions
    
    experiment(0).main()
    experiment(1).main()

  def table_1_run(self, CO2_ppm = None, D = 0.3, s_type = None, hide_run = None):

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

  def generate_table_1(self):

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

  def table_2_run(self, obl, long, ecc):

    On_Off = {'moist':1, 'albT':1, 'seas':2,'thermo':1}
    
    Orbitals = {'obl':obl, 'long':long, 'ecc':ecc}

    output = Model_Class.model(self, T = self.Ti, F = self.F, grid = self.config, CO2_ppm = None, D = None, S_type = 1, moist = On_Off['moist'], albT = On_Off['albT'], seas = On_Off['seas'], thermo = On_Off['thermo'], obl = Orbitals['obl'], long = Orbitals['long'], ecc = Orbitals['ecc'], kyear = 'forced', hide_run = None)

    outputs = Helper_Functions().table_2_outs(self.config, output)
    outputs = list(outputs)

    inputs = list([Orbitals['obl'],Orbitals['long'],Orbitals['ecc']])

    return inputs, outputs

  def generate_table_2(self):

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

  def sensitivity_runs(self, run_type = None, orb_comp = None, def_v_orb = 'Off'):
    
    if run_type == "default":

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

    elif run_type == 'forcing':

      CO2_forcing = {'On':560, 'Off':280}
      ice_feedback = {"On":[1, 'dynamic'], "Off":[4, 'static']}
      control_orbit = {'control_obl': Helper_Functions().orbit_at_time('obl',1), "control_long":Helper_Functions().orbit_at_time('long',1), 'control_ecc': Helper_Functions().orbit_at_time('ecc',1)}
      
      if def_v_orb == 'On':
        Orbit_forcing = {'On':[1, 'orbital'], 'Off':[0, 'default']}
        kyear = 1
        orbital_perturbation_settings = {'obl':0,'long':0,'ecc':0}
      else:
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
      
      if def_v_orb == 'On':
        run_000_inputs = [Orbit_forcing['Off'][1], CO2_forcing['Off'], self.control_label]
        run_010_inputs = [Orbit_forcing['Off'][1], CO2_forcing['On'], self.control_label]
        run_100_inputs = [Orbit_forcing['On'][1], CO2_forcing['Off'], self.control_label]
        run_110_inputs = [Orbit_forcing['On'][1], CO2_forcing['On'], self.control_label]
        run_001_inputs = [Orbit_forcing['Off'][1], CO2_forcing['Off'], ice_feedback['Off'][1]]
        run_011_inputs = [Orbit_forcing['Off'][1], CO2_forcing['On'], ice_feedback['Off'][1]]
        run_101_inputs = [Orbit_forcing['On'][1], CO2_forcing['Off'], ice_feedback['Off'][1]]
        run_111_inputs = [Orbit_forcing['On'][1], CO2_forcing['On'], ice_feedback['Off'][1]]

      elif def_v_orb != 'On':

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

    else:

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

  def sensitivity_sythesis(self, run_type = None, orb_comp = None, def_v_orb = "Off"):

      if run_type == 'forcing':

        all_runs, all_run_inputs = self.sensitivity_runs(run_type = run_type, orb_comp = orb_comp, def_v_orb = def_v_orb)
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

        control_run_output, no_ice_output, constant_ice_output, dynamic_ice_output, full_run_output, all_inputs = self.sensitivity_runs(run_type = run_type, orb_comp = orb_comp, def_v_orb = def_v_orb)

        control_v_no_ice = Helper_Functions().sensitivity_analysis(control_run_output, no_ice_output)
        control_v_constant_ice = Helper_Functions().sensitivity_analysis(control_run_output, constant_ice_output)
        control_v_dynamic_ice = Helper_Functions().sensitivity_analysis(control_run_output, dynamic_ice_output)
        control_v_full = Helper_Functions().sensitivity_analysis(control_run_output, full_run_output)

        control_v_no_ice = list(control_v_no_ice)
        control_v_constant_ice = list(control_v_constant_ice)
        control_v_dynamic_ice = list(control_v_dynamic_ice)
        control_v_full = list(control_v_full)

        return control_v_no_ice, control_v_constant_ice, control_v_dynamic_ice, control_v_full, all_inputs

  def generate_sensitivity_table(self, run_type = None, orb_comp = None, def_v_orb = "Off"):

    if run_type == 'forcing':

      all_comparisons, all_run_inputs = self.sensitivity_sythesis(run_type = run_type, orb_comp = orb_comp, def_v_orb = def_v_orb)
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
      if def_v_orb == 'On':
        df.columns = ['Orbital Forcing', 'CO2 Forcing', 'Albedo Feedback','delta_I', 'delta_OLR', 'delta_FCO2', 'delta_GMT', 'calculated_delta_GMT', 'unnacounted_change']
      else:
        df.columns = ['Obliquity','Long Peri', "Ecc", 'CO2 Forcing', 'Albedo Feedback','delta_I', 'delta_OLR', 'delta_FCO2', 'delta_GMT', 'calculated_delta_GMT', 'unnacounted_change']

      df = Helper_Functions().format_data(df)
      df = df.round(2)
      df.to_csv("table_sensitivity.csv")

      return None

    else:

      control_v_no_ice, control_v_constant_ice, control_v_dynamic_ice, control_v_full, all_inputs = self.sensitivity_sythesis(run_type = run_type, orb_comp = orb_comp, def_v_orb = def_v_orb)
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
      df.columns = ['Ins Type', 'Obl', 'long', 'ecc', 'CO2 ppm', 'settings','delta_I', 'delta_ASR', 'delta_FCO2', 'delta_GMT', 'calculated_delta_GMT', 'unnacounted_change']
      df = Helper_Functions().format_data(df)
      df = df.round(2)
      df.to_csv("table_sensitivity.csv")

      return None

  def forcing_decomp_runs(self):

    CO2_forcing = {'Double':[560, 'Double'], 'Half':[140, 'Half']}
    kyear = 'forced'
    control_orbit = {'control_obl': Helper_Functions().orbit_at_time('obl',1), "control_long":Helper_Functions().orbit_at_time('long',1), 'control_ecc': Helper_Functions().orbit_at_time('ecc',1)}
    obl_settings = {'max_obl': [Helper_Functions().orbit_extrema('obl', 'max'), 'Max'],'min_obl': [Helper_Functions().orbit_extrema('obl', 'min'), 'Min'] }
    #long_settings = {'max_long': [Helper_Functions().orbit_extrema('long', 'max'),'Max'],'min_long': [Helper_Functions().orbit_extrema('long', 'min'), 'Min'] }
    ecc_settings = {'max_ecc': [Helper_Functions().orbit_extrema('ecc', 'max'), 'Max'],'min_ecc': [Helper_Functions().orbit_extrema('ecc', 'min'), 'Min'] }
    long_settings = {'max_long': [0,'Max'],'min_long': [180, 'Min'] }


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

  def forcing_decomp_sensitivity_synthesis(self):

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

  def generate_decomp_table(self):

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

    df.columns = ['Obliquity','Long Peri', "Ecc", 'CO2 Forcing','delta_I', 'delta_OLR', 'delta_FCO2', 'delta_GMT', 'calculated_delta_GMT', 'unnacounted_change']
    
    df = Helper_Functions().format_data(df)
    df = df.round(2)
    df.to_csv("forcing_decomp_table.csv")

    return None

  def orbit_and_CO2_suite(self, x_type = None):

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
      #icelines_of_single_run = np.mean(icelines_of_single_run)
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

    #()
    return x_range, GMT_of_runs, Shemi_icelines_of_runs, S65_ins, S65_temp, dASRdalpha, geo_wind_max_lat, t_grad_max_lat, WSC_max_lat, CO2_val

  def obl_sensitivity_analysis(self, albT = 'On'):

    control_orbit = {'control_obl': Helper_Functions().orbit_at_time('obl',1), "control_long":Helper_Functions().orbit_at_time('long',1), 'control_ecc': Helper_Functions().orbit_at_time('ecc',1)}
    dobl = Helper_Functions().orbit_extrema('obl', 'max') - Helper_Functions().orbit_extrema('obl', 'min')
    kyear = 'forced'

    iterations = 100
    
    x_range = np.linspace(100,600,iterations)

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

class Orbital_Insolation():

  def __init__(self, from_ktime=2000, until_ktime = 0):

    self.from_ktime = from_ktime
    self.until_ktime = until_ktime
    self.days_per_year_const = 365.2422 #days
    self.day_type = 1

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

      kyear0 = kyear0[self.until_ktime:self.from_ktime]
      ecc0 = ecc0[self.until_ktime:self.from_ktime]
      long_peri = long_peri[self.until_ktime:self.from_ktime]
      obliquity = obliquity[self.until_ktime:self.from_ktime]
      precession = precession[self.until_ktime:self.from_ktime]

     
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
  
  def insolation(self, day, lat = 0, obl_run = None, obl = None, long = None, ecc = None, kyear = None):  #returns insolation based on day and lat #https://climlab.readthedocs.io/en/latest/_modules/climlab/solar/insolation.html#daily_insolation

    S0 = 1365.2 # W/m^2
    
    oldsettings = np.seterr(invalid='ignore')

    if type(kyear) != type('string'):

      kyear0,ecc0,long_peri,obliquity,precession = self.get_orbit('on')

    elif type(kyear) == type('string'):

      obliquity = obl
      long_peri = long
      ecc0 = ecc

    if obl_run != None:
      obliquity = obl_run

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

  def avg_insolation(self, grid, lat_array = 0, lat = 0, from_lat = None, to_lat = None, from_month = None, to_month = None, obl_run = None, obl = None, long = None, ecc = None, kyear = None): #returns annual insolation at lat

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']
    # converting x to lattitude for insolation method input
    x = np.rad2deg(np.arcsin(x))

    #breakpoint()
    
    if lat_array == 0:

      avg_day = []   
      
      for f in range(0,int(self.days_per_year_const)):
        
        avg_each_day_of_lat = self.insolation(f, lat,  obl = obl, long = long, ecc = ecc, kyear = kyear)
        
        avg_day.append(avg_each_day_of_lat) 
      
      sum_days = sum(avg_day)
      
      avg_for_year = sum_days / len(range(0,int(self.days_per_year_const)))
      
      return avg_for_year

    elif lat_array == 'seas':

      day = np.linspace(0,self.days_per_year_const,nt)
      
      avg_lat = []

      for i in x:
            
        day_at_lat = self.insolation(day,i, obl = obl, long = long, ecc = ecc, kyear = kyear)
       
        avg_lat.append(day_at_lat)

      avg_lat = np.array(avg_lat)

      avg_lat = avg_lat.T

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

      day = np.linspace(0,365,nt)
      
      avg_lat = []

      for i in x:

        day_at_lat = self.insolation(day,i, obl = obl, long = long, ecc = ecc, kyear = kyear)
       
        avg_lat.append(day_at_lat)

      avg_lat = np.array(avg_lat)

      avg_lat = avg_lat.T

      avg_lat = np.mean(avg_lat, axis = 0)

      avg_lat = np.tile(avg_lat, (nt, 1))

      return avg_lat

  def get_insolation(self, grid, kyear, at_lat): #returns annual average insolation at kyear and lattitude

    if kyear-1 >= 0:
      return self.avg_insolation(grid, lat = at_lat)[kyear-1]
    else:
      return "get_insolation input must be positive integer"

  def get_energy(self, grid, lat_array = 0, lat = 0, from_lat = None, to_lat = None, from_month = None, to_month = None): #returns Watts for a given latitude band

      inso = self.avg_insolation(grid, lat_array, lat, from_lat, to_lat, from_month, to_month)

      area = Helper_Functions().area_array(grid)

      area = area[from_lat:to_lat]

      area = np.tile(area, (2000,1)).T

      energy = inso * area

      return energy

  def s_array(self, grid, lat_array = 0, from_lat = None, to_lat = None, obl_run = None, obl = None, long = None, ecc = None, kyear = None): #takes in a kyear and returns an insolation-lattitude array for seasonal or annual avg

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']
    
    if kyear == 'forced':
      return Orbital_Insolation(1,0).avg_insolation(grid, lat_array, from_lat = from_lat, to_lat = to_lat, obl_run = obl_run, obl = obl, long = long, ecc = ecc, kyear = kyear)

    elif type(kyear) == type(0):
      return Orbital_Insolation(kyear, kyear-1).avg_insolation(grid, lat_array, from_lat = from_lat, to_lat = to_lat, obl_run = obl_run)

    elif kyear-1 < 0:
      return "kyear cannot be a negative value"

  def get_mean_insolation(self, grid, lat_array = 0, from_lat = None, to_lat = None):

    mean_inso_list = []

    for i in range(self.until_ktime+1, self.from_ktime):

      mean_inso = np.mean(self.s_array(i, grid, lat_array, from_lat = from_lat, to_lat = to_lat))

      mean_inso_list.append(mean_inso)
  
    return mean_inso_list

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

  def obl_suite(self,grid):
    
    kyear0,ecc0,long_peri,obliquity,precession = self.get_orbit()

    min_obl = int(np.min(obliquity))
    max_obl = int(np.max(obliquity))
    resolution = 10
    obl_values = np.linspace(min_obl,max_obl,resolution)

    obl_out = []

    for i in obl_values:

      run = self.s_array(1, grid, obl_run = i, lat_array = "annual")

      obl_out.append(run)

    obl_out = np.array(np.mean(obl_out, axis = 1))

    return obl_out, obl_values

class Helper_Functions():

  def __init__(self, output = None):
    
    self.output = output

  def area_array(self, grid, r = 6378100): #calculates the area of a zonal band on earth

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

  def geocentric_rad(self,lat): #returns an array of radii for each latitude for geiod earth

    r_eq = 6378137.0 #m
    r_pole = 6356752.3142 #m

    geo_r = ((((r_eq**2)*np.cos(lat))**2 + ((r_pole**2)*np.sin(lat))**2) / ((r_eq * np.cos(lat))**2 + (r_pole * np.sin(lat))**2))**(1/2)

    return geo_r

  def weight_area(self, grid, array): #returns area weighted annual mean insolation

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

  def make_contours(self, array_1, array_2, array_diff): #returns bounds and step for arrays to contour plot, so nothing is missed and steps are approporaite size

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

  def find_extrema(self, grid, array, position_array, type = None):

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

  def find_zeros(self, grid, array, position_array, any = None):

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

    if array.ndim == 1:
    
      array = abs(array)
      array = list(array)
      closest_to_zero = np.min(array)
      zero_index = array.index(closest_to_zero)
      zero_position = position_array[zero_index]
      output = zero_position

      if any == 'both':

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

  def find_lat_of_value(self, grid, array, position_array, value = 0.4, save_icelines = 'Off', filename = None):

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

  def find_nearest_value(self, array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

  def find_value_at_lat(self,grid,array,lat):

    n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
    nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

    lat_array = list(np.rad2deg(np.arcsin(x)))

    target_lat = self.find_nearest_value(lat_array, lat)

    value_at_lat = array[lat_array.index(target_lat)]
    
    return value_at_lat

  def find_kyear_extrema(self, orbit_type, extrema_type):
    
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

  def orbit_extrema(self, orbit_type, extrema_type):

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
  
  def orbit_at_time(self, orbit_type, kyear):

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

  def band_width(self, grid, units = 'km'):

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

  def take_gradient(self, grid, arr_1, diffx = 'deg'):

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
    elif arr_1.ndim == 2:
        gradient = []
        for i in range(0, nt):
            T_slice = np.gradient(arr_1[i, :])
            gradient.append(T_slice)
        gradient = np.array(gradient)
        gradient = gradient / lat_dist  
        return gradient

  def table_1_outs(self, grid, model_output, Hemi = 'S'):

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

  def table_2_outs(self, grid, model_output, Hemi = "S"):

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
      max_ice, min_ice, mean_ice = ice_lines
    else:
      max_ice, min_ice, mean_ice = ice_lines 

    return GMT, eq_pole_diff, global_mean_ins, E_transport_mag, E_transport_loc, T_grad_mag, T_grad_loc, geo_wind_mag, geo_wind_loc, max_ice, min_ice, mean_ice

  def format_data(self, df):
    
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

  def CO2_to_A(self,CO2_ppm):

      S_efolding = 5.35 # Myhre, 1998
      
      #A = 196
      A = 192.9499973

      A = A - np.log((CO2_ppm/280)) * S_efolding

      return A

  def CO2_forcing(self, CO2):

    S_efolding = 5.35 # W/ m^2 2xCO2

    forcing = np.log(CO2/280) * S_efolding

    return forcing

  def sensitivity_analysis(self, run_1, run_2):
  
    tfin_1, Efin_1, Tfin_1, T0fin_1, ASRfin_1, S_1, Tg_1, mean_S_1, OLR_1, kyear_1, E_transport_1, T_grad_1, alpha_1, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = run_1
    tfin_2, Efin_2, Tfin_2, T0fin_2, ASRfin_2, S_2, Tg_2, mean_S_2, OLR_2, kyear_2, E_transport_2, T_grad_2, alpha_2, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = run_2

    B = 1.8#4/2.6
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

    net_r = dASRdI + dASRdalpha + delta_FCO2

    calculated_delta_GMT = sensitivity_parameter * (net_r)

    unnacounted_change = delta_GMT - calculated_delta_GMT

    B_model = (dASRdI + dASRdalpha + delta_FCO2) / delta_GMT

    #breakpoint()

    return dASRdI, dASRdalpha, delta_FCO2, delta_GMT, calculated_delta_GMT, unnacounted_change

  def calculate_zonal_wind(self, T_grad):

    #np.save(arr = T_grad, file = 'T_grad_save_for_reg.npy')

    from CESM_data import total_interp_values

    m_interp , b_interp = total_interp_values

    ## Units of CESM are wind[m/s] and T grad[C/lat] ##

    zonal_wind = m_interp * T_grad + b_interp

    zonal_wind = zonal_wind.T

    return zonal_wind

  def seasonal_zonal_regression(self, T_grad):

    from CESM_data import all_popts

    popt1, popt2, popt3, popt4, popt5, popt6, popt7, popt8, popt9, popt10, popt11, popt12 = all_popts

    # Getting Tgrad within southern ocean bounds
    lats = experiment().lat_deg
    lat_pole_bound = self.find_nearest_value(lats, -70)
    lat_eq_bound = self.find_nearest_value(lats, -40)
    index_pole_lat = np.where(lats == lat_pole_bound)[0][0]
    index_eq_lat = np.where(lats == lat_eq_bound)[0][0]
    
    SO_lats = lats[index_pole_lat:index_eq_lat]
    SO_Tgrad = T_grad[:,index_pole_lat:index_eq_lat]

    # Breaking up Tgrad into 12 sections and storing
    Tgrad1 = SO_Tgrad[0:83,:]
    Tgrad2 = SO_Tgrad[83:167,:]
    Tgrad3 = SO_Tgrad[167:251,:]
    Tgrad4 = SO_Tgrad[251:335,:]
    Tgrad5 = SO_Tgrad[335:419,:]
    Tgrad6 = SO_Tgrad[419:503,:]
    Tgrad7 = SO_Tgrad[503:587,:]
    Tgrad8 = SO_Tgrad[587:671,:]
    Tgrad9 = SO_Tgrad[671:755,:]
    Tgrad10 = SO_Tgrad[755:839,:]
    Tgrad11 = SO_Tgrad[839:923,:]
    Tgrad12 = SO_Tgrad[923:1000,:]
    Tgrads = Tgrad1, Tgrad2, Tgrad3, Tgrad4, Tgrad5, Tgrad6, Tgrad7, Tgrad8, Tgrad9, Tgrad10, Tgrad11, Tgrad12

    # Compute Winds

    SO_Z_winds = []
    for i in range(0,12):
      single_run_winds = (all_popts[i][0]*Tgrads[i]*SO_lats + all_popts[i][1]*SO_lats**2)*Tgrads[i] + all_popts[i][2]+all_popts[i][3]
      SO_Z_winds.append(single_run_winds)

    SO_all = [item for SO_Z_winds in SO_Z_winds for item in SO_Z_winds]    
    
    return SO_all, SO_lats, SO_Tgrad

class Model_Class(): 

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
    if S_type == 0:
 
      # Seasonal forcing (WE15 eq.3)
      if seas == 0:
        S1 = 0.0
      
      elif seas == 2:
        pass

      S = (np.tile(S0-S2*x**2,[nt,1])-
          np.tile(S1*np.cos(2*np.pi*ty),[n,1]).T*np.tile(x,[nt,1]))
    
      # zero out negative insolation
      S = np.where(S<0,0,S)

    elif S_type == 1:

      if kyear != 'forced':

        if seas == 0:
          
          S = Orbital_Insolation().s_array(grid, kyear = kyear, lat_array = 'annual')

          S = np.where(S<0,0,S)
        
        elif seas == 1:
          
          S = Orbital_Insolation().s_array(grid, kyear = kyear, lat_array = 'seas')
          #S = Orbital_Insolation().s_array(grid, lat_array = 'seas', obl = Helper_Functions().orbit_at_time('obl', kyear), long = Helper_Functions().orbit_at_time('long', kyear), ecc = Helper_Functions().orbit_at_time('ecc', kyear), kyear = kyear)


          S = np.where(S<0,0,S) 

        elif seas == 2:

          S = Orbital_Insolation().s_array(grid, kyear = kyear, lat_array = 'seas')
          #S = Orbital_Insolation().s_array(grid, lat_array = 'seas', obl = Helper_Functions().orbit_at_time('obl', kyear), long = Helper_Functions().orbit_at_time('long', kyear), ecc = Helper_Functions().orbit_at_time('ecc', kyear), kyear = kyear)


          S = np.where(S<0,0,S)
        
      elif kyear == 'forced':

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

    # Set up preliminary output arrays
    Epre = np.zeros((n,nt)) 
    Tpre = np.zeros((n,nt))
    T0pre = np.zeros((n,nt))
    ASRpre = np.zeros((n,nt))
    Albpre = np.zeros((n,nt))
    tpre = np.linspace(0,1,nt)


    # Initial conditions
    Tg = T
    E = cw*T
    Energy_Balance = 2
    while abs(Energy_Balance) > eb:
      # Integration (see WE15_NumericIntegration.pdf)
      # Loop over Years 
      for years in range(dur):
      # e_balance = 100
      # while e_balance > 0.001:
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
              alpha_loaded = np.load('testfile_Alb_90cells.npy')
            else:
              alpha_loaded = np.load('testfile_Alb.npy')

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

    #---------------------------------------------------------


    max_ice_extent, min_ice_extent, mean_ice_extent, ice_positions = Helper_Functions().find_lat_of_value(grid, Albfin, lat, ai)
    ice_lines = max_ice_extent, min_ice_extent, mean_ice_extent, ice_positions

    if save_Alb == 'On':
      np.save(file = 'testfile_Alb_90cells', arr = Albfin)
 
    
    R_TOA = ASRfin - (A + B*Tfin)
    area = Helper_Functions().area_array(grid)[0]
    dHdlat = area * R_TOA

    E_transport = []
    for i in range(0,nt):
      E_slice = np.cumsum(dHdlat[:,i])
      E_transport.append(E_slice)

    E_transport = np.array(E_transport)

    T_grad = Helper_Functions().take_gradient(grid,Tfin.T, diffx = 'deg') # C /deg

    #Helper_Functions().seasonal_zonal_regression(T_grad)

    geo_wind =  Helper_Functions().calculate_zonal_wind(T_grad)
    max_geo_wind, min_geo_wind, mean_geo_wind, wind_lines = Helper_Functions().find_lat_of_value(grid, geo_wind, lat, 'maximum')

    f = 2*omega*sin(np.arcsin(x))
    f = np.tile(f, (nt,1))

    T_grad_m =  1000*Helper_Functions().take_gradient(grid,Tfin.T,diffx='km')

    #T_fin_K = [Tfin +273.15 for i in Tfin]
    T_fin_K = Tfin + 273.15
   # T_fin_K = np.asarray(T_fin_K)

    # breakpoint()
    analytic_geo_wind = (-9.8* T_grad_m * 0.01) / (roh * f * T_fin_K.T)
    #breakpoint()
    #geo_wind = analytic_geo_wind.T

    wind_stress = (geo_wind ** 2) * roh * drag_coeff

    wind_stress_curl = Helper_Functions().take_gradient(grid,wind_stress.T) # m / s deg


    mean_S = np.mean(S, axis=(0,1))
    OLR = A - B*Tfin
    OLR = np.mean(OLR,axis = (1))
    OLR = R_TOA
    
 #---------------------------------------------------------

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

class Figures():

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

        plt.suptitle('Annual Mean Default Insoaltion = {:.2f}, Annual Mean Orbital Insolation {:.2f}, Annual Mean Difference {:.2f}'.format(mean_S_def, mean_S_orb, (mean_S_def - mean_S_orb)))

        axs[0].set_title('Default Seasonal Insolation')
        contour_1 = axs[0].contourf(dur_plt,lat,S_def_seas,np.arange(bar_min,bar_max,bar_step), extend = 'max', cmap=plt.get_cmap('hot'))

        axs[1].set_title('Orbital Seasonal Insolation')
        contour_2 = axs[1].contourf(dur_plt,lat,S_orb_seas,np.arange(bar_min,bar_max,bar_step), extend = 'max', cmap=plt.get_cmap('hot'))

        axs[2].set_title('Seasonal Difference')
        contour_3 = axs[2].contourf(dur_plt,lat,inso_difference,np.arange(plot_diff_min,plot_diff_max,plot_diff_step), extend = 'max', cmap=plt.get_cmap('hot'))
        
        axs[0].set_xlabel('days')
        axs[1].set_xlabel('days')
        axs[2].set_xlabel('days')
        axs[0].set_ylabel('lat')
        axs[1].set_ylabel('lat')
        axs[2].set_ylabel('lat')

        plt.colorbar(contour_1,ax = axs[0])
        plt.colorbar(contour_2,ax = axs[1])
        plt.colorbar(contour_3,ax = axs[2])

        fig.savefig('ContourCompare.jpg')
      
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

        dur_plt = np.linspace(0,365,nt)
        lat = np.rad2deg(np.arcsin(x))

        fig,axs = plt.subplots(ncols = 3, figsize = (12,8))

        plt.suptitle('global mean default Temperature = {:.2f}, global mean orbital Temperature {:.2f}, global mean difference {:.2f}'.format(mean_Tfin_def_seas, mean_Tfin_orb_seas, (mean_Tfin_def_seas - mean_Tfin_orb_seas)))
        
        contour_1 = axs[0].contourf(dur_plt,lat,Tfin_def_seas,np.arange(bar_min,bar_max,bar_step), extend = 'both')
        axs[0].set_title('Seasonal Temp Default ins')

        contour_2 = axs[1].contourf(dur_plt,lat,Tfin_orb_seas,np.arange(bar_min,bar_max,bar_step),  extend = 'both')
        axs[1].set_title('Seasonal Temp Orbital ins')

        contour_3 = axs[2].contourf(dur_plt,lat,temp_difference,np.arange(plot_diff_min,plot_diff_max,plot_diff_step),  extend = 'both')
        axs[2].set_title('Seasonal Difference')

        test = axs[0].contour(dur_plt,lat,Tfin_def_seas,levels = [0], colors = 'red', label = 'Sea Ice Extent')
        test = axs[1].contour(dur_plt,lat,Tfin_orb_seas,levels = [0], colors = 'red', label = 'Sea Ice Extent')

        plt.colorbar(contour_1,ax = axs[0])
        plt.colorbar(contour_2,ax = axs[1])
        plt.colorbar(contour_3,ax = axs[2])

        axs[0].set_xlabel('dur')
        axs[1].set_xlabel('dur')
        axs[2].set_xlabel('dur')
        axs[0].set_ylabel('lat')
        axs[1].set_ylabel('lat')
        axs[2].set_ylabel('lat')

        if True == True:
          print('savefig')
          plt.savefig('ContourCompare_Temp.jpg')

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

        axs[0].set_title('Annual Temp Default ins')
        axs[0].plot(lat,Tfin_def_annual)

        axs[1].set_title('Annual Temp Orbital ins')
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

          inso_difference = S_seas_kyear1 - S_seas_kyear2

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
          
          contour_1 = axs[0].contourf(dur_plt,x,S_seas_kyear1,np.arange(bar_min,bar_max,bar_step), extend = 'max', cmap=plt.get_cmap('hot'))
          contour_2 = axs[1].contourf(dur_plt,x,S_seas_kyear2,np.arange(bar_min,bar_max,bar_step), extend = 'max', cmap=plt.get_cmap('hot'))
          contour_3 = axs[2].contourf(dur_plt,x,inso_difference,np.arange(plot_diff_min,plot_diff_max,plot_diff_step), extend = 'max', cmap=plt.get_cmap('hot'))

          plt.colorbar(contour_1,ax = axs[0])
          plt.colorbar(contour_2,ax = axs[1])
          plt.colorbar(contour_3,ax = axs[2])

          axs[0].set_xlabel('days')
          axs[1].set_xlabel('days')
          axs[2].set_xlabel('days')
          axs[0].set_ylabel('sin(φ)')

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

         # breakpoint()

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

      if subchart == 'forcing':

        output_1 = np.load('control_1110_low_obl.npy', allow_pickle=True)
        output_2 = np.load('control_1110_high_obl.npy', allow_pickle=True)
        tfin, Efin, Tfin_1, T0fin, ASRfin, S_seas_kyear1, Tg, mean_S_kyear1, OLR, S_kyear1, E_transport_kyr1, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
        tfin, Efin, Tfin_2, T0fin, ASRfin, S_seas_kyear2, Tg, mean_S_kyear2, OLR, S_kyear2, E_transport_kyr2, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2

        dur_plt = np.linspace(0,365,nt)

        inso_difference = S_seas_kyear2 - S_seas_kyear1
        temp_difference = Tfin_2 - Tfin_1

        #breakpoint()

        fig, axs = plt.subplots(ncols = 2)

        #contour_1 = axs[0].contourf(dur_plt, experiment().lat_deg, inso_difference, np.arange(-20,50,0.5), extend = 'max', cmap=plt.get_cmap('hot'))
        #contour_1 = axs[1].contour(dur_plt, experiment().lat_deg, temp_difference, 0, cmap=plt.get_cmap('hot'))
        #contour_1 = axs[0].contour(dur_plt, experiment().lat_deg, inso_difference, 0, color = 'black')
        contour_2 = axs[1].contourf(dur_plt, experiment().lat_deg, temp_difference, np.arange(-2,2,0.005), extend = 'max', cmap=plt.get_cmap('seismic'))
        contour_1 = axs[0].contourf(dur_plt, experiment().lat_deg, inso_difference, np.arange(-50,50,0.5), extend = 'max', cmap=plt.get_cmap('jet'))

        #test1 = axs[0].contour(dur_plt,experiment().lat_deg,0,levels = [0], colors = 'black')
        #test2 = axs[1].contour(dur_plt,experiment().lat_deg,temp_difference,levels = [0], colors = 'black')

        #test_1 = axs[1].plot(dur_plt, ice_lines_1[3], color = 'red', label = 'Low Obliquity Mean Ice Line')
        #test_2 = axs[1].plot(dur_plt, ice_lines_2[3], color = 'blue', label = 'High Obliquity Mean Ice Line')

        #axs[1].legend()

        plt.colorbar(contour_1,ax = axs[0])#, label = '[W/m²]')
        plt.colorbar(contour_2,ax = axs[1])#, label = 'Celcius')

        axs[0].set_xlabel('Time (Days)')
        axs[1].set_xlabel('Time (Days)')

        axs[0].set_ylabel('Latitude (φ)')

        #plt.suptitle('Energy Balance Model Reponse to Obliquity Forcing')
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

    elif chart == 11: #OLR and ASR plots BROKEN

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

    elif chart == 12: #fake?Energy Transport Paleo Compare2
      
      tfin, Efin, Tfin_annual_kyear2, T0fin, ASRfin, S_annual_kyear2, Tg, mean_S_kyear2, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2
      lat = np.rad2deg(np.arcsin(x))
     
      if subchart == 'annual':

        #kyear0_1,ecc0_1,long_peri_1,obliquity_1 = Orbital_Insolation().display_orbit(S_kyear)

        
        fig, axs = plt.subplots(nrows = 1)
        axs.set_xlabel('latitude')
        axs.set_ylabel('Watts')
        #axs.set_title('obl {}'.format(obliquity_1))
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

        #breakpoint()

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
        #breakpoint()
        # contour_1 = axs[0].contourf(dur_plt,SO_lats,SO_Tgrad.T,np.arange(-0,1,0.01), extend = 'both', cmap=plt.get_cmap('bwr'))
        # contour_2 = axs[1].contourf(dur_plt,SO_lats,SO_geo_wind.T,np.arange(14,20,0.01), extend = 'both', cmap=plt.get_cmap('bwr'))

        plt.colorbar(contour_1,ax = axs[0])
        plt.colorbar(contour_2,ax = axs[1])
        #plt.colorbar(contour_3,ax = axs[2])

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

      # axs[0].plot(lat,alpha)
      # axs_0 = axs[0].twinx()
      # axs_0.plot(lat,Tfin, color = 'green')

      # axs[1].plot(lat,OLR, color = 'green')
      # axs_1 = axs[1].twinx()
      # axs_1.plot(lat,E_transport, color = "blue")
      # #axs_2 = axs[1].twinx()
      # #axs_2.plot(lat,S, color = 'red')
      #plt.suptitle('extrema_1 : {} \n extrema_2 : {} \n ice_line_1 : {} \n ice_line_2 : {} \n wind_extrema_1 : {} \n wind_extrema_2 : {}'.format(extrema_1, extrema_2, ice_line_1, ice_line_2, wind_extrema_1, wind_extrema_2))
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
      #tfin_3, Efin_3, Tfin_3, T0fin_3, ASRfin_3, S_3, Tg_3, mean_S_3, OLR_3, S_kyear_3, E_transport_3, T_grad_3, alpha_3, geo_wind_3 = output_3
      #tfin_4, Efin_4, Tfin_4, T0fin_4, ASRfin_4, S_4, Tg_4, mean_S_4, OLR_4, S_kyear_4, E_transport_4, T_grad_4, alpha_4, geo_wind_4 = output_4

      lat = np.rad2deg(np.arcsin(x))
      extrema_1 = Helper_Functions().find_extrema(grid, T_grad_1, lat, 'max')
      extrema_2 = Helper_Functions().find_extrema(grid, T_grad_2, lat, 'max')
      #extrema_3 = Helper_Functions().find_extrema(T_grad_3, lat, 'max')
      #extrema_4 = Helper_Functions().find_extrema(T_grad_4, lat, 'max')
      wind_peak_1 = Helper_Functions().find_extrema(grid, geo_wind_1[0:45], lat, 'max')
      wind_peak_2 = Helper_Functions().find_extrema(grid, geo_wind_2[0:45], lat, 'max')

      wind_diff = geo_wind_1 - geo_wind_2
      T_grad_diff = T_grad_1 - T_grad_2

      freeze = Helper_Functions().find_zeros(grid, Tfin_1,lat, any ="both")
      
      fig,axs = plt.subplots(figsize = (7,8))

      #axs.set_title('Modern extrema at {:.2f}, LGM extrema at {:.2f}, Min Obl at {:.2f}, Max Obl at {:.2f} \n Modern Mag {:.2f}, LGM Mag {:.2f}, Min Obl Mag {:.2f}, Max Obl Mag {:.2f}'.format(extrema_1, extrema_2, extrema_3, extrema_4, np.max(T_grad_1), np.max(T_grad_2), np.max(T_grad_3), np.max(T_grad_4)))
      axs.set_title('Min Obl Tgrad peak at {:.2f}, Max Obl Tgrad peak at {:.2f} \n Min Obl Tgrad Mag {:.2f}, Max Obl Tgrad Mag {:.2f} \n Min obl wind peak : {:.2f}, Max obl wind peak 2 : {:.2f}'.format(extrema_1, extrema_2, np.max(T_grad_1), np.max(T_grad_2), wind_peak_1, wind_peak_2))
      axs.plot(lat,T_grad_1, label = 'Min obl T grad', color = 'blue')
      axs.plot(lat,T_grad_2, label = 'Max obl T grad', color = 'red')
      #axs.plot(lat,T_grad_3, label = 'Obliquity Minimum')
      #axs.plot(lat,T_grad_4, label = 'Obliquity Maximum')
      axs_1 = axs.twinx()
      axs_1.plot(lat, geo_wind_1, label = 'Min obl wind speed', color = 'green')
      axs_1.plot(lat, geo_wind_2, label = 'Max obl wind speed', color = 'orange')
      axs_1.set_ylabel('wind speed')
     # axs_11 = axs[0].twinx()
     # axs_11.plot(lat, alpha_1)
      axs.set_ylabel("dT/dy")
      axs.set_xlabel('latitude')

      lines, labels = axs.get_legend_handles_labels()
      lines2, labels2 = axs_1.get_legend_handles_labels()
      axs_1.legend(lines + lines2, labels + labels2, loc=0) 

      # axs[1].plot(lat, wind_diff)
      # axs_2 = axs[1].twinx()
      # axs_2.plot(lat, T_grad_diff, color = 'red')
      axs.set_xlim(-90,0)
      #axs[1].set_xlim(-90,0)
      #axs.set_ylim(1,5.5)
      # axs.legend()
      # axs_1.legend()
      plt.tight_layout()

    elif chart == 17: # obl sensitivity w/ GMT-xaxis

      x_range, GMT_ouput, iceline_output, S65_ins, S65_temp, dASRdalpha, geo_wind_max_lat, t_grad_max_lat, WSC_max_lat, CO2_val = experiment().orbit_and_CO2_suite(x_type = 'obl')

      fig, axs_2 = plt.subplots()

      #axs.plot(x_range, GMT_ouput, label = 'GMT', color = 'green')
      #axs_2 = axs.twinx()
      axs_2.plot(GMT_ouput, iceline_output, color = 'blue', label = 'mean ice edge')
      axs_2.plot(GMT_ouput, geo_wind_max_lat, label = 'Zwind max lat', color = 'red')
      axs_2.plot(GMT_ouput, t_grad_max_lat, label = 'tgrad max lat', color = 'orange')
      axs_2.plot(GMT_ouput, WSC_max_lat, label = 'WSC max lat', color = 'purple')
      axs_2.legend()
     
      #axs.axvline(x = Helper_Functions().orbit_at_time('obl',1), color = 'b', label = 'control')
      #axs.set_xlabel('CO2')
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


      ###-----------------------------------------###

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

      # sensitivity_ax.plot(x_range, obl_sensitivity_ASR, label = 'CO2 and Obliquity Dependant Sea Ice Feedback')
      # sensitivity_ax.plot(x_range, obl_sensitivity_ASR_nof, color = 'gray', label = 'No Sea Ice Feedback')
      # sensitivity_ax.plot(x_range, obl_sensitivity_ASR_static, color = 'red', label = 'CO2 Dependent Sea Ice Feedback')
      # sensitivity_ax.axhline(0, color = 'black', linestyle = 'dotted')
      # sensitivity_ax.set_ylabel('ΔGMT/Δε')
      # sensitivity_ax.set_xlabel('Atmospheric CO2 ppm')
      # sensitivity_ax.set_title('GMT Response to Obliquity (ε) and CO2 Forcing \n for different Sea Ice Dynamics')

      #fig.text(0.07,0.87, 'CO2 and Obliquity Dependant \n Sea Ice Feedback', color = 'blue', fontsize = 'x-small')
      #fig.text(0.1,0.57, 'CO2 Dependent \n Sea Ice Feedback', color = 'red', fontsize = 'x-small')

      #sensitivity_ax.legend()

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
      #iceline_ax.legend()

      
      
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

      # t_grad_ax.plot(x_range, Tgrad_50_60_max, label = '50°S-60°S ε Max', color = 'dodgerblue', alpha=0.4)
      # t_grad_ax.plot(x_range, Tgrad_50_60_min, label = '50°S-60°S ε Min', color = 'orangered', alpha=0.4)

      # t_grad_ax.plot(x_range, Tgrad_60_70_max, label = '60°S-70°S ε Max', color = 'blue', alpha=0.4)
      # t_grad_ax.plot(x_range, Tgrad_60_70_min, label = '60°S-70°S ε Min', color = 'firebrick', alpha=0.4)

      t_grad_ax.set_xlabel('Atmospheric CO2 ppm')
      t_grad_ax.set_ylabel('Meridional Temperature Gradients [ΔK/Δφ]')
      t_grad_ax.set_title('Equator-Pole Temperature Gradient v CO2')

      #t_grad_ax.legend()

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

      plt.savefig('ForcingSuite.pdf')

    elif chart == 19: # CAM and MEBM regression compare

      tfin, Efin, Tfin, T0fin, ASRfin, S_def_seas, Tg, mean_S_def, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1
      #tfin, Efin, Tfin_orb_seas, T0fin, ASRfin, S_orb_seas, Tg, mean_S_orb, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_2, ice_lines_2, Albfin_2, OLRfin_2, CO2_2, wind_stress_curl_2 = output_2
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
      #axs3 = axs[2].twinx()
      axs[2].plot(CAM_lat,CAM_surf_temp, color = 'red', label = 'GCM')
      axs[2].set_ylabel('Surface Temp [C]')
      axs[2].set_title('MEBM v GCM Surface Temperatures')
      axs[2].legend()

    elif chart == 20: # contour plot WSC, wind, and Tgrad

      tfin, Efin, Tfin, T0fin, ASRfin, S_def_seas, Tg, mean_S_def, OLR, S_kyear, E_transport, T_grad, alpha, geo_wind_1, ice_lines_1, Albfin_1, OLRfin_1, CO2_1, wind_stress_curl_1 = output_1

      # SO_wind, SO_lats = Helper_Functions().seasonal_zonal_regression(T_grad.T)
      # SO_wind = np.array(SO_wind)
      #breakpoint()
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
      #from CESM_data import pdf

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
      axs[0,0].set_title('Modern Wind Stress Curl \n mean sea ice edge at {:.2f}'.format(ice_lines_1[2]))

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
      axs[0,1].set_title('Modern Surface Wind Speed \n mean sea ice edge at {:.2f}'.format(ice_lines_1[2]))

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
      axs[0,2].set_title('Modern Surface Temp Gradients \n mean sea ice edge at {:.2f}'.format(ice_lines_1[2]))

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
      breakpoint()
      fig.tight_layout()

if __name__ == '__main__':
  experiment().main()
  #experiment().generate_table_1()
  #experiment().generate_sensitivity_table(run_type = 'forcing', orb_comp='obl', def_v_orb="Off")
  #experiment().generate_decomp_table()
  #experiment().orbit_and_CO2_suite(x_type = 'CO2')
  #Figures().figure(experiment().config, chart = 18)
  #Figures().figure(experiment().config, chart = 18, subchart='forcing')
 
  print("-------------------------------------------------")
  print("--- %s seconds ---" % (time.time() - start_time))
  print("-------------------------------------------------")
  print("")


