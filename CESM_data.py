import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import lognorm
import scipy as sp
import matplotlib.mlab as mlab
from sklearn.metrics import r2_score 
from sea_ice_MEBM_ghost import experiment
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d

# Define functions
def take_gradient(arr_1): # takes the gradient of an array
    
    if arr_1.ndim == 2:
        gradient = []
        for i in range(0, 240):
            T_slice = np.gradient(arr_1[i, :])
            gradient.append(T_slice)
        gradient = np.array(gradient)
        gradient = gradient / 1.9  # deg #210.9 kmeters
        return gradient
    elif arr_1.ndim == 1:
        gradient = []
        for i in range(0, 240):
            T_slice = np.gradient(arr_1[i])
            gradient.append(T_slice)
        gradient = np.array(gradient)
        gradient = gradient / 1.9  # deg #210.9 kmeters
        return gradient
    elif arr_1.ndim == 3:

        gradients = []
        gradient_at_lon = []
        for i in range(0,240):
            for f in range(0,144):
                T_slice = np.gradient(arr_1[i,:,f])
                gradient_at_lon.append(T_slice)
            #gradients.append(gradient_at_lon)
        gradients = np.array(gradients)
        gradients = gradients / 1.9  # deg #210.9 kmeters

        return gradients

def find_lat_of_max(array, position_array, ice='Off', ice_frac = 'non-zero', avg = None): # finds the corresponding latitude of the maximum value of an array
    positions = []
    if avg == 'On':
        for i in range(0, 12):
            southern_hemi_lats = position_array[position_array < 0]
    
            array_in_loop = array[i, :]
            array_in_loop_southern_hemi = array_in_loop[position_array < 0]
            if ice == "On":
              
                if ice_frac == 'non-zero':
                    southern_hemi_positions = southern_hemi_lats[np.nonzero(array_in_loop_southern_hemi)[0][-1]]
                    southern_hemi_positions = np.array([southern_hemi_positions])
                elif ice_frac == '0.5':
                    southern_hemi_positions = southern_hemi_lats[array_in_loop_southern_hemi == find_nearest_value(array_in_loop_southern_hemi,0.5)]
                elif ice_frac == '0.75':
                    southern_hemi_positions = southern_hemi_lats[array_in_loop_southern_hemi == find_nearest_value(array_in_loop_southern_hemi,0.75)]
                elif ice_frac == 'full':
                    southern_hemi_positions = southern_hemi_lats[array_in_loop_southern_hemi == find_nearest_value(array_in_loop_southern_hemi,1)]
                elif ice_frac == 'contintental boundry':
                    southern_hemi_positions = southern_hemi_lats[np.nonzero(array_in_loop_southern_hemi)[0][0]]
                    southern_hemi_positions = np.array([southern_hemi_positions])
            else:
                southern_hemi_positions = southern_hemi_lats[array_in_loop_southern_hemi == np.max(array_in_loop_southern_hemi)]
            if len(southern_hemi_positions) > 0:
                ice_ext = np.max(southern_hemi_positions)
                positions.append(ice_ext)
            elif len(southern_hemi_positions) <= 0:
                positions.append(-90.0)
    else:
        for i in range(0, 240):
            southern_hemi_lats = position_array[position_array < 0]
            array_in_loop = array[i, :]
            array_in_loop_southern_hemi = array_in_loop[position_array < 0]
            
            if ice == "On":
               
                if ice_frac == 'non-zero':                    
                    southern_hemi_positions = southern_hemi_lats[np.nonzero(array_in_loop_southern_hemi)[0][-1]]
                    southern_hemi_positions = np.array([southern_hemi_positions])
                elif ice_frac == '0.5':
                    southern_hemi_positions = southern_hemi_lats[array_in_loop_southern_hemi == find_nearest_value(array_in_loop_southern_hemi,0.5)]
                elif ice_frac == '0.75':
                    southern_hemi_positions = southern_hemi_lats[array_in_loop_southern_hemi == find_nearest_value(array_in_loop_southern_hemi,0.75)]
                elif ice_frac == 'full':
                    southern_hemi_positions = southern_hemi_lats[array_in_loop_southern_hemi == find_nearest_value(array_in_loop_southern_hemi,1)]
                    
                elif ice_frac == 'contintental boundry':
                    southern_hemi_positions = southern_hemi_lats[np.nonzero(array_in_loop_southern_hemi)[0][0]]
                    southern_hemi_positions = np.array([southern_hemi_positions])
                    
            else:
                southern_hemi_positions = southern_hemi_lats[array_in_loop_southern_hemi == np.max(array_in_loop_southern_hemi)]
            if len(southern_hemi_positions) > 0:
                ice_ext = np.max(southern_hemi_positions)
                positions.append(ice_ext)
            elif len(southern_hemi_positions) <= 0:
                positions.append(-90.0)


    max_ice = np.max(positions)
    min_ice = np.min(positions)
    mean_ice = np.mean(positions)

    return max_ice, min_ice, mean_ice, positions

def correlation(var_1, var_2, lat=None): # determines the correlation value of two data sets
    if lat is not None:
        corr = np.corrcoef(var_1[:, lat], var_2[:, lat])[1][0]
    else:
        corr = np.corrcoef(var_1, var_2)[1][0]
    return corr    

def linear_regression(arr_1, arr_2, lat): # least square linear regression

    x , y = arr_1[:, lat], arr_2[:, lat]

    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]

    return m,b

def regression_line(x,lat): # line equation using linear regression

    y = all_m[lat] * x[:,lat] + all_b[lat]

    return y

def calc_z_wind(T_grad, all_m, all_b): # reconstructs zonal wind from MTG regression

    if T_grad.ndim == 2:
        
        z_wind = all_m * T_grad + all_b
        
        return z_wind
    
    elif T_grad.ndim == 3:

        z_wind_at_each_lon = []

        for i in range(0, len(lon)):
            
            T_grad_at_lon = T_grad[:,:,i]#take_gradient(T_grad[:,:,i])
            
            z_wind_at_lon = all_m[i][:] * T_grad_at_lon + all_b[i][:]
            z_wind_at_each_lon.append(z_wind_at_lon)

        return z_wind_at_each_lon

def poly_fit(x,y,deg): # fits coefficients of n order polynomial
    
    poly_coeff = np.polyfit(x, y, deg)
    deg_arr = np.flip(np.linspace(0,deg))

    fit = 0

    for i in range(0,deg):

       fit += poly_coeff[i]*x**deg_arr[i]

    return fit

def reg_func(y,a,b,c): # simple regression equation

    output = (a + b)*y + c

    return output

def opt_func(x,a,b,c,d): # ad hoc optimization function
    
    lat, xdata = x

    roh = 1.2 #kg /m^3
    omega = 7.27e-5 #s^-1

    ### comment in and out equations of choice ###

    #output = -(1/(a*np.sin(np.deg2rad(lat)))) * xdata + b+c+d
    #output = -(1/(a*roh*2*omega*np.sin(np.deg2rad(lat)))) * xdata*b + c
    #output = (a*xdata**3 + b*np.sin(np.deg2rad(lat)))*xdata + c + d
    output = (a*xdata*lat + b*lat**2)*xdata + c+d
    #output = a*xdata + b*lat*xdata + c*np.sin(d)*xdata
    
    return output

def prep_multivar_data(xdata, xdata_2, y_data, month, lat_bounds = None): # multivariate regression preparation of model data
    
    months = {'January':0,'February':1,'March':2,'April':3,'May':4,'June':5,'July':6,'August':7,'September':8,'October':9,'November':10,'December':11}
    
    init_slice = months[month]

    annual_steps = [init_slice]
    while annual_steps[-1] < len(time_ax):
        annual_steps.append(annual_steps[-1]+11)
    annual_steps = [x for x in annual_steps if x < len(time_ax)]

    mean_of_month_xdata = []
    [mean_of_month_xdata.append(xdata_2[i,:]) for i in annual_steps]
    mean_of_month_xdata = np.mean(mean_of_month_xdata, axis=0)
    
    mean_of_month_ydata = []
    [mean_of_month_ydata.append(y_data[i,:]) for i in annual_steps]
    mean_of_month_ydata = np.mean(mean_of_month_ydata, axis=0)

    if lat_bounds == "SO":

        xdata = xdata[10:26]
        mean_of_month_xdata = mean_of_month_xdata[10:26]
        mean_of_month_ydata = mean_of_month_ydata[10:26]

    xdata_full = []
    xdata_full.append(xdata)
    xdata_full.append(mean_of_month_xdata)

    return xdata_full, mean_of_month_ydata

def contour_opt(popt, xdata_full): # preparation for contour plotting

    lat, xdata = xdata_full

    Z_wind_2d = []

    for i in range(0,len(lat)):

        xdata_lat = xdata_full[0][i]
        xdata_gradT = xdata_full[1][i]
        xdata_in_loop = []
        xdata_in_loop.append(xdata_lat)
        xdata_in_loop.append(xdata_gradT)

        val = opt_func(xdata_in_loop, popt[0], popt[1], popt[2], popt[3])

        Z_wind_2d.append(val)

    return Z_wind_2d

def find_nearest_value(array, value): # finds nearest specified value in array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def contour_for_a_month(xdata_1, xdata_2, y_data, month): # prepares contoru plot for single month

    xdata_full, ydata = prep_multivar_data(xdata_1,xdata_2,y_data,month)

    lat, xdata = xdata_full

    popts = []

    for i in range(0,21):

        lat_in_loop = lat[i,:]
        xdata_in_loop = xdata[i,:]

        xdata_full_in_loop = []
        xdata_full_in_loop.append(lat_in_loop)
        xdata_full_in_loop.append(xdata_in_loop)

        popt, pcov = curve_fit(opt_func,xdata_full_in_loop, ydata)

def stitch_regression(xdata, xdata_2, y_data, lat_bounds = None): # meshes individual months into full year contour plot

    months = {'January':0,'February':1,'March':2,'April':3,'May':4,'June':5,'July':6,'August':7,'September':8,'October':9,'November':10,'December':11}

    full_time_reg = []
    full_y_data = []
    f_constant = []
    popts = []

    for i in months:
    
        xdata_full, ydata = prep_multivar_data(xdata,xdata_2,y_data,i,lat_bounds)

        popt, pcov = curve_fit(opt_func,xdata_full, ydata)

        Z_reg = opt_func(xdata_full, popt[0], popt[1], popt[2], popt[3])

        full_time_reg.append(Z_reg)
        full_y_data.append(ydata)
        f_constant.append(popt[0])
        popts.append(list(popt))
    
    popts = np.reshape(popts, (12,4))
    
    return full_time_reg, full_y_data, f_constant, xdata_full, popts

def T_grad_ice_line_analysis(array,position_array,icelines): # compartmentalizes MTGs relative to ice lines

    over_ice_vals = []
    lat_of_over_ice_vals = []

    over_water_vals = []
    lat_of_water_ice_vals = []

    around_ice_vals = []
    lat_of_around_ice_vals = []
    
    for i in range(0,len(icelines)):

        southern_hemi_lats = position_array[position_array < 0]

        array_in_loop = array[i, :]

        array_in_loop_southern_hemi = array_in_loop[position_array < 0]

        over_ice_array_in_loop = array_in_loop_southern_hemi[southern_hemi_lats < icelines[i]]        
        lat_of_over_ice_array_in_loop = southern_hemi_lats[southern_hemi_lats < icelines[i]]
        lat_of_over_ice_array_in_loop = list(lat_of_over_ice_array_in_loop)
        over_ice_array_in_loop = list(over_ice_array_in_loop)

        over_surface_array_in_loop = array_in_loop_southern_hemi[southern_hemi_lats > icelines[i]]
        over_surface_array_in_loop = array_in_loop_southern_hemi[southern_hemi_lats < icelines[i]+21]
        lat_of_over_surface_array_in_loop = southern_hemi_lats[southern_hemi_lats > icelines[i]]
        lat_of_over_surface_array_in_loop = list(lat_of_over_surface_array_in_loop)
        over_surface_array_in_loop = list(over_surface_array_in_loop)

        around_iceline_array_in_loop = array_in_loop_southern_hemi[southern_hemi_lats > icelines[i]-1]
        around_iceline_array_in_loop = array_in_loop_southern_hemi[southern_hemi_lats < icelines[i]+1]
        around_iceline_array_in_loop = array_in_loop_southern_hemi[southern_hemi_lats == icelines[i]]
        lat_of_around_iceline_array_in_loop = southern_hemi_lats[southern_hemi_lats == icelines[i]]
        lat_of_around_iceline_array_in_loop = list(lat_of_around_iceline_array_in_loop)
        around_iceline_array_in_loop = list(around_iceline_array_in_loop)

        over_ice_vals.append(over_ice_array_in_loop)
        lat_of_over_ice_vals .append(lat_of_over_ice_array_in_loop)   

        over_water_vals.append(over_surface_array_in_loop)
        lat_of_water_ice_vals.append(lat_of_over_surface_array_in_loop)

        around_ice_vals.append(around_iceline_array_in_loop)
        lat_of_around_ice_vals.append(lat_of_around_iceline_array_in_loop)

    over_ice_vals = [item for over_ice_vals in over_ice_vals for item in over_ice_vals]
    lat_of_over_ice_vals = [item for lat_of_over_ice_vals in lat_of_over_ice_vals for item in lat_of_over_ice_vals]

    over_water_vals = [item for over_water_vals in over_water_vals for item in over_water_vals]
    lat_of_water_ice_vals = [item for lat_of_water_ice_vals in lat_of_water_ice_vals for item in lat_of_water_ice_vals]

    around_ice_vals = [item for around_ice_vals in around_ice_vals for item in around_ice_vals]
    lat_of_around_ice_vals = [item for lat_of_around_ice_vals in lat_of_around_ice_vals for item in lat_of_around_ice_vals]

    index_pole_lat = np.where(position_array == np.min(lat_of_over_ice_vals))[0][0]
    index_eq_lat = np.where(position_array == np.max(lat_of_water_ice_vals))[0][0]

    lat_range = position_array[index_pole_lat:index_eq_lat]

    over_ice_lats_and_vals = list(zip(lat_of_over_ice_vals,over_ice_vals))
    over_water_lats_and_vals = list(zip(lat_of_water_ice_vals,over_water_vals))
    around_ice_lats_and_vals = list(zip(lat_of_around_ice_vals,around_ice_vals))

    value_groups = over_ice_lats_and_vals,over_water_lats_and_vals,around_ice_lats_and_vals


    return over_ice_vals, over_water_vals, around_ice_vals, lat_of_over_ice_vals, lat_of_water_ice_vals, lat_of_around_ice_vals, lat_range

def lognorm_pdf(x,mu,sigma): # computs lognormal probability density functions

    output = (1 / (x*sigma*np.sqrt(2*np.pi))) * np.exp(-((np.log(x)-mu)**2) / (2*(sigma**2)))

    return output

def pdf_analysis(array,position_array,icelines): # finds pdfs

    array_vals = T_grad_ice_line_analysis(array, position_array, icelines)

    mu1, sd1 = norm.fit(array_vals[0])
    mu2, sd2 = norm.fit(array_vals[1])
    mu3, sd3 = norm.fit(array_vals[2])
    mu = [mu1,mu2,mu3]
    mu = np.mean(mu)
    sd = [sd1,sd2,sd3]
    sd = np.max(sd)

    x_range = np.linspace(mu - 4*sd,mu + 4*sd,1000)
    
    pdf1 = norm.pdf(x_range, mu1, sd1)
    pdf2 = norm.pdf(x_range, mu2, sd2)
    pdf3 = norm.pdf(x_range, mu3, sd3)

    return pdf1, pdf2, pdf3, x_range, array_vals

def lon_regression(x1,x2,y): # linear regression for longitude values

    m_and_b_at_each_lon = []

    for i in range(0,len(lon)):

        T_grad_at_lon = take_gradient(x2[:,:,i])
        Z_wind_at_lon = y[:,:,i]

        m_and_b_at_each_lat_at_lon = [linear_regression(T_grad_at_lon, Z_wind_at_lon, lat = i) for i in range(0,len(lat))]

        all_m_at_lon = []
        [all_m_at_lon.append(m_and_b_at_each_lat_at_lon[f][0]) for f in range(0,len(lat))] 
        all_b_at_lon = []
        [all_b_at_lon.append(m_and_b_at_each_lat_at_lon[f][1]) for f in range(0,len(lat))]

        m_and_b_of_regression_at_lon = all_m_at_lon, all_b_at_lon

        m_and_b_at_each_lon.append(m_and_b_of_regression_at_lon)
        #breakpoint()

    m_and_b_at_each_lon = np.array(m_and_b_at_each_lon)

    return m_and_b_at_each_lon

def monthly_average(array, month): # computes monthly average of data values

    months = {'January':0,'February':1,'March':2,'April':3,'May':4,'June':5,'July':6,'August':7,'September':8,'October':9,'November':10,'December':11}

    if month == 'annual':

        annual_monthly_avg = []

        for g in months.keys():
            
            init_slice = months[g]

            annual_steps = [init_slice]
            while annual_steps[-1] < len(time_ax):
                annual_steps.append(annual_steps[-1]+11)
            annual_steps = [x for x in annual_steps if x < len(time_ax)]

            mean_of_month_xdata = []
            [mean_of_month_xdata.append(array[i,:]) for i in annual_steps]
            mean_of_month_xdata = np.mean(mean_of_month_xdata, axis=0)

            annual_monthly_avg.append(mean_of_month_xdata)

        annual_monthly_avg = np.array(annual_monthly_avg)

        return annual_monthly_avg

    else:

        init_slice = months[month]

        annual_steps = [init_slice]
        while annual_steps[-1] < len(time_ax):
            annual_steps.append(annual_steps[-1]+11)
        annual_steps = [x for x in annual_steps if x < len(time_ax)]

        mean_of_month_xdata = []
        [mean_of_month_xdata.append(array[i,:]) for i in annual_steps]
        mean_of_month_xdata = np.mean(mean_of_month_xdata, axis=0)

        return mean_of_month_xdata

def coveriance(var1,var2): # returns covariance matrix of two variables

    array = np.vstack((var1,var2))

    return np.cov(array)

# Importing and processing data and model configuration
data = xr.open_dataset('cpl_1850_f19.cam.h0.nc')
variables = data.data_vars
grid = experiment().config
n = grid['n']; dx = grid['dx']; x = grid['x']; xb = grid['xb']
nt = grid['nt']; dur = grid['dur']; dt = grid['dt']; eb = grid['eb']

# Extracting processing and defining model variables
time_dim = int(list(np.shape(data.time))[0]) # Time Step = 1 month ... time = 0 is January
time_ax = np.linspace(0, time_dim, time_dim)
lat = data.lat.values
lon = data.lon.values
lev = data.lev.values
cor_param = 2*7.2921159e-5*np.sin(np.deg2rad(lat))
cor_grad = 1.9*2*7.2921159e-5*np.cos(np.deg2rad(lat))

M_heat_trans_all = data['VT'].values # meridional heat transport [Km/s]
M_heat_trans = M_heat_trans_all[:, -1, :, :]  # Getting surface values only
#M_heat_trans = np.cumsum(M_heat_trans_all, axis = 1)
M_heat_trans = np.mean(M_heat_trans, axis=2)  # zonal mean

Z_wind = data['U'].values # zonal wind velocity [m/s]
Z_wind_lon = Z_wind[:,-1,:] # Getting surface values only
Z_wind = np.mean(Z_wind_lon, axis = 2) # zonal mean

U10 = data['U10'].values # zonal wind velocity at 10m [m/s]
U10 = np.mean(U10, axis = 2) # zonal mean

U2d = data['U2d'].values # zonal mean wind speed defined on ilev
U2d = np.mean(U2d, axis = 2) # zonal mean

sea_ice = data['ICEFRAC'].values # sea ice surface area fraction 
sea_ice_lon = sea_ice
sea_ice = np.mean(sea_ice, axis=2)  # zonal mean

Temp = data['TS'].values  # surface temperature [K]
Temp_lon = Temp
Temp = np.mean(Temp, axis=2)  # zonal mean

wind_stress = data['TAUX'].values # zonal surface stress [N/m2]
wind_stress_lon = wind_stress
wind_stress = np.mean(wind_stress, axis = 2) # zonal mean

Ins = data['SOLIN'].values  # solar insolation [W/m2]
Ins = np.mean(Ins, axis=2)  # zonal mean

surf_preassure = data['PS'].values # surface preassure [Pa]
surf_preassure = np.mean(surf_preassure, axis = 2) # zonal mean

lev_temp = data['T'].values # vertical temperature field
lev_temp = np.mean(lev_temp, axis = 3) # zonal mean

# Processed data
T_grad = take_gradient(Temp) # meridional surface temperature gradient
M_grad = take_gradient(M_heat_trans)
P_grad = take_gradient(surf_preassure) # meridional surface preassure gradient
WSC = take_gradient(wind_stress) # meridionla wind stress curl

lev_T_grad = []
for i in range(0,len(lev)): # solving meridional temperature gradient at each level
    t_grad_at_lev = take_gradient(lev_temp[:,i,:])
    lev_T_grad.append(t_grad_at_lev)


CAM_Tgrad = np.mean(T_grad, axis = 0) # time average
CAM_Z_wind = np.mean(Z_wind, axis = 0) # time average
CAM_lat = lat
CAM_surf_temp = np.mean(Temp,axis = 0) # time average
CAM_vars = CAM_Tgrad, CAM_Z_wind, CAM_surf_temp, CAM_lat

# Analyzing data
max_wind = find_lat_of_max(Z_wind, lat)
max_T_grad = find_lat_of_max(T_grad, lat)[3]
max_WSC_SO = find_lat_of_max(WSC[:, 5:25], lat[5:25])[3]
max_heat_transport = find_lat_of_max(M_heat_trans, lat)[3]

max_lev_tgrad = []
for i in range(0,len(lev)): # finding latutude of maximum temperature gradient at each level
    max_tgrad_at_lev = find_lat_of_max(lev_T_grad[i][:][:], lat)[3]
    max_lev_tgrad.append(max_tgrad_at_lev)

non_zero_ice_line = find_lat_of_max(sea_ice, lat, ice = 'On', ice_frac = 'non-zero')[3]
ice_line_half = find_lat_of_max(sea_ice, lat, ice = 'On', ice_frac = '0.5')[3]
ice_line_three_q = find_lat_of_max(sea_ice, lat, ice = 'On', ice_frac = '0.75')[3]
full_ice_line = find_lat_of_max(sea_ice, lat, ice = 'On', ice_frac = 'full')[3]
boundry_ice_line = find_lat_of_max(sea_ice, lat, ice = "On", ice_frac = "contintental boundry")[3]

corr_80 = correlation(Z_wind[:, 5], T_grad[:, 5])
corr_70 = correlation(Z_wind[:, 10], T_grad[:, 10])
corr_60 = correlation(Z_wind[:, 16], T_grad[:, 16])
corr_50 = correlation(Z_wind[:, 21], T_grad[:, 21])
corr_40 = correlation(Z_wind[:, 26], T_grad[:, 26])
corr_30 = correlation(Z_wind[:, 31], T_grad[:, 31])

net_corr = np.corrcoef(Z_wind[:, 5:31], T_grad[:, 5:31])

corr_for_each_lat = [correlation(Z_wind, T_grad, lat=i) for i in range(0, len(lat))]

x_80, y_80 = T_grad[:, 5], Z_wind[:, 5]
x_70, y_70 = T_grad[:, 10], Z_wind[:, 10]
x_60, y_60 = T_grad[:, 16], Z_wind[:, 16]
x_50, y_50 = T_grad[:, 21], Z_wind[:, 21]
x_40, y_40 = T_grad[:, 26], Z_wind[:, 26]
x_30, y_30 = T_grad[:, 31], Z_wind[:, 31]

# Linear regression
m_and_b_at_each_lat = [linear_regression(M_heat_trans, Z_wind, lat = i) for i in range(0,len(lat))]

all_m = []
[all_m.append(m_and_b_at_each_lat[i][0]) for i in range(0,len(lat))] 
all_b = []
[all_b.append(m_and_b_at_each_lat[i][1]) for i in range(0,len(lat))]

m_and_b_of_regression = all_m, all_b

m_80, b_80 = m_and_b_at_each_lat[5][0], m_and_b_at_each_lat[5][1]
m_70, b_70 = m_and_b_at_each_lat[10][0], m_and_b_at_each_lat[10][1]
m_60, b_60 = m_and_b_at_each_lat[16][0], m_and_b_at_each_lat[16][1]
m_50, b_50 = m_and_b_at_each_lat[21][0], m_and_b_at_each_lat[21][1]
m_40, b_40 = m_and_b_at_each_lat[26][0], m_and_b_at_each_lat[26][1]
m_30, b_30 = m_and_b_at_each_lat[31][0], m_and_b_at_each_lat[31][1]

regression_line_80 = regression_line(T_grad, 5)
regression_line_70 = regression_line(T_grad, 10)
regression_line_60 = regression_line(T_grad, 16)
regression_line_50 = regression_line(T_grad, 21)
regression_line_40 = regression_line(T_grad, 26)
regression_line_30 = regression_line(T_grad, 31)

# R^2 Correlation for Line Fit
R2_80 = r2_score(y_80, regression_line_80)
R2_70 = r2_score(y_70, regression_line_70)
R2_60 = r2_score(y_60, regression_line_60)
R2_50 = r2_score(y_50, regression_line_50)
R2_40 = r2_score(y_40, regression_line_40)
R2_30 = r2_score(y_30, regression_line_30)

# Interpolation on m and b
model_lat = np.rad2deg(np.arcsin(experiment().config['x']))

m_interp = np.interp(model_lat, lat, all_m)
b_interp = np.interp(model_lat, lat, all_b)

total_interp_values = m_interp, b_interp

# Calculate Zonal Wind from Regression
Z_wind_reg = calc_z_wind(sea_ice, all_m, all_b)

MEBM_Tgrad = np.load('T_grad_save_for_reg.npy')

func_m = interp1d(lat,all_m, kind = 'linear')
func_b = interp1d(lat,all_b, kind = 'linear')

Z_wind_MEBM_reg_scipy = calc_z_wind(MEBM_Tgrad, func_m(model_lat), func_b(model_lat))
Z_wind_MEBM_reg_numpy = calc_z_wind(MEBM_Tgrad, m_interp, b_interp)

# Multivariate Regression

# xdata_full, ydata = prep_multivar_data(lat,T_grad,Z_wind,'November')
# popt, pcov = curve_fit(opt_func,xdata_full, ydata)

Z_year_reg, Z_y_data, f_constant, xdata_full, all_popts = stitch_regression(lat,T_grad, Z_wind, lat_bounds='SO')

    # % difference

coriolis_const = 2*7.29e-5
f_array = np.array(f_constant)
normalized_diff = ( abs(f_array - coriolis_const) ) / np.max(f_array) - coriolis_const

    # 2D R^2

R2_contour = r2_score(Z_y_data, Z_year_reg)

# Longitudinal Linear Regression
m_and_b_for_each_lon = lon_regression(lat,sea_ice_lon,Z_wind_lon)

m_for_each_lon = m_and_b_for_each_lon[:, 0, :]
b_for_each_lon = m_and_b_for_each_lon[:, 1, :]

z_wind_at_each_lon = calc_z_wind(sea_ice_lon,m_for_each_lon,b_for_each_lon)

### Lon Avg ###

SO_upper = 25

T_grad_avg = monthly_average(T_grad, 'annual')
sea_ice_avg = monthly_average(sea_ice, 'annual')
WSC_avg = monthly_average(WSC, 'annual')
WSC_avg = -1*WSC_avg
wind_speed_avg = monthly_average(Z_wind, 'annual')

max_T_grad_avg = find_lat_of_max(T_grad_avg, lat, avg = 'On')[3]
max_WSC_SO_avg = find_lat_of_max(WSC_avg[:, 5:SO_upper], lat[5:SO_upper], avg = 'On')[3]
max_wind_speed_SO_avg = find_lat_of_max(wind_speed_avg[:, 5:SO_upper], lat[5:SO_upper], avg = 'On')[3]

non_zero_ice_line_avg = find_lat_of_max(sea_ice_avg, lat, ice = 'On', ice_frac = 'non-zero', avg = 'On')[3]
ice_line_half_avg = find_lat_of_max(sea_ice_avg, lat, ice = 'On', ice_frac = '0.5', avg = 'On')[3]
ice_line_three_q_avg = find_lat_of_max(sea_ice_avg, lat, ice = 'On', ice_frac = '0.75', avg = 'On')[3]
full_ice_line_avg = find_lat_of_max(sea_ice_avg, lat, ice = 'On', ice_frac = 'full', avg = 'On')[3]
boundry_ice_line_avg = find_lat_of_max(sea_ice_avg, lat, ice = "On", ice_frac = "contintental boundry", avg = 'On')[3]

### Pacific ###

Pacific_Temps = Temp_lon[:,:,68:108]
Pacific_sea_ice = sea_ice_lon[:,:,68:108]
Pacific_wind_stress = wind_stress_lon[:,:,68:108]

Pacific_temp_mean = np.mean(Pacific_Temps, axis = 2)
Pacific_ice_mean = np.mean(Pacific_sea_ice, axis = 2)
Pacific_stress_mean = np.mean(Pacific_wind_stress, axis = 2)

Pacific_Tgrad = take_gradient(Pacific_temp_mean)

Pacific_Tgrad_avg = monthly_average(Pacific_Tgrad, 'annual')
Pacific_sea_ice_avg = monthly_average(Pacific_ice_mean, 'annual')
Pacific_WSC_avg = monthly_average(Pacific_stress_mean, 'annual')
Pacific_WSC_avg = -1*Pacific_WSC_avg

Pacific_max_T_grad_avg = find_lat_of_max(Pacific_Tgrad_avg[:, 5:SO_upper], lat[5:SO_upper], avg = 'On')[3]
Pacific_max_WSC_SO_avg = find_lat_of_max(Pacific_WSC_avg[:, 5:SO_upper], lat[5:SO_upper], avg = 'On')[3]

Pacific_non_zero_ice_line_avg = find_lat_of_max(Pacific_sea_ice_avg, lat, ice = 'On', ice_frac = 'non-zero', avg = 'On')[3]
Pacific_ice_line_half_avg = find_lat_of_max(Pacific_sea_ice_avg, lat, ice = 'On', ice_frac = '0.5', avg = 'On')[3]
Pacific_ice_line_three_q_avg = find_lat_of_max(Pacific_sea_ice_avg, lat, ice = 'On', ice_frac = '0.75', avg = 'On')[3]
Pacific_full_ice_line_avg = find_lat_of_max(Pacific_sea_ice_avg, lat, ice = 'On', ice_frac = 'full', avg = 'On')[3]
Pacific_boundry_ice_line_avg = find_lat_of_max(Pacific_sea_ice_avg, lat, ice = "On", ice_frac = "contintental boundry", avg = 'On')[3]


### Atlantic ###

Atlantic_Temps = Temp_lon[:,:,124:143]
Atlantic_sea_ice = sea_ice_lon[:,:,124:143]
Atlantic_wind_stress = wind_stress_lon[:,:,124:143]

Atlantic_temp_mean = np.mean(Atlantic_Temps, axis = 2)
Atlantic_ice_mean = np.mean(Atlantic_sea_ice, axis = 2)
Atlantic_stress_mean = np.mean(Atlantic_wind_stress, axis = 2)

Atlantic_Tgrad = take_gradient(Atlantic_temp_mean)

Atlantic_Tgrad_avg = monthly_average(Atlantic_Tgrad, 'annual')
Atlantic_sea_ice_avg = monthly_average(Atlantic_ice_mean, 'annual')
Atlantic_WSC_avg = monthly_average(Atlantic_stress_mean, 'annual')
Atlantic_WSC_avg = -1*Atlantic_WSC_avg

Atlantic_max_T_grad_avg = find_lat_of_max(Atlantic_Tgrad_avg[:, 5:SO_upper], lat[5:SO_upper], avg = 'On')[3]
Atlantic_max_WSC_SO_avg = find_lat_of_max(Atlantic_WSC_avg[:, 5:SO_upper], lat[5:SO_upper], avg = 'On')[3]

Atlantic_non_zero_ice_line_avg = find_lat_of_max(Atlantic_sea_ice_avg, lat, ice = 'On', ice_frac = 'non-zero', avg = 'On')[3]
Atlantic_ice_line_half_avg = find_lat_of_max(Atlantic_sea_ice_avg, lat, ice = 'On', ice_frac = '0.5', avg = 'On')[3]
Atlantic_ice_line_three_q_avg = find_lat_of_max(Atlantic_sea_ice_avg, lat, ice = 'On', ice_frac = '0.75', avg = 'On')[3]
Atlantic_full_ice_line_avg = find_lat_of_max(Atlantic_sea_ice_avg, lat, ice = 'On', ice_frac = 'full', avg = 'On')[3]
Atlantic_boundry_ice_line_avg = find_lat_of_max(Atlantic_sea_ice_avg, lat, ice = "On", ice_frac = "contintental boundry", avg = 'On')[3]


### Indian ###

Indian_Temps = Temp_lon[:,:,20:44]
Indian_sea_ice = sea_ice_lon[:,:,20:44]
Indian_wind_stress = wind_stress_lon[:,:,20:44]

Indian_temp_mean = np.mean(Indian_Temps, axis = 2)
Indian_ice_mean = np.mean(Indian_sea_ice, axis = 2)
Indian_stress_mean = np.mean(Indian_wind_stress, axis = 2)

Indian_Tgrad = take_gradient(Indian_temp_mean)

Indian_Tgrad_avg = monthly_average(Indian_Tgrad, 'annual')
Indian_sea_ice_avg = monthly_average(Indian_ice_mean, 'annual')
Indian_WSC_avg = monthly_average(Indian_stress_mean, 'annual')
Indian_WSC_avg = -1*Indian_WSC_avg

Indian_max_T_grad_avg = find_lat_of_max(Indian_Tgrad_avg[:, 5:SO_upper], lat[5:SO_upper], avg = 'On')[3]
Indian_max_WSC_SO_avg = find_lat_of_max(Indian_WSC_avg[:, 5:SO_upper], lat[5:SO_upper], avg = 'On')[3]

Indian_non_zero_ice_line_avg = find_lat_of_max(Indian_sea_ice_avg[:, 5:SO_upper], lat[5:SO_upper], ice = 'On', ice_frac = 'non-zero', avg = 'On')[3]
Indian_ice_line_half_avg = find_lat_of_max(Indian_sea_ice_avg[:, 5:SO_upper], lat[5:SO_upper], ice = 'On', ice_frac = '0.5', avg = 'On')[3]
Indian_ice_line_three_q_avg = find_lat_of_max(Indian_sea_ice_avg[:, 5:SO_upper], lat[5:SO_upper], ice = 'On', ice_frac = '0.75', avg = 'On')[3]
Indian_full_ice_line_avg = find_lat_of_max(Indian_sea_ice_avg[:, 5:SO_upper], lat[5:SO_upper], ice = 'On', ice_frac = 'full', avg = 'On')[3]
Indian_boundry_ice_line_avg = find_lat_of_max(Indian_sea_ice_avg[:, 5:SO_upper], lat[5:SO_upper], ice = "On", ice_frac = "contintental boundry", avg = 'On')[3]

# Creating figures
def generate_figures(which_figure):

    if which_figure == 1: # linear regression on zonal wind and surface Meridional Heat Transport at different latitudes

        fig, axs = plt.subplots(figsize=(15, 10), nrows=6)
        fig.supylabel('Zonal Wind Speed [m/s]')
        fig.supxlabel('Meridional Heat Transport [Km/s]')
        axs[0].set_title('Meridional Heat Transport vs Surface Zonal Wind at 80°S')
        axs[0].scatter(x_80, y_80, label='80°S', color='red')
        axs[0].plot(x_80, regression_line_80, label='Fitted line', color='black')
        axs[0].annotate('Correlation = {:.2f}'.format(corr_80), (0, 76), xycoords='axes points')
        axs[0].annotate('R² = {:.2f}'.format(R2_80), (100, 76), xycoords='axes points')
        axs[0].legend(loc='upper right')
        
        # ... Repeat for other latitudes ...

        axs[1].set_title('Meridional Heat Transport vs Surface Zonal Wind at 70°S')
        axs[1].scatter(x_70,y_70, label = '70°S', color = 'orange')
        axs[1].plot(x_70, regression_line_70, label='Fitted line', color = 'black')
        axs[1].annotate('Correlation = {:.2f}'.format(corr_70), (0,76), xycoords='axes points')
        axs[1].annotate('R² = {:.2f}'.format(R2_70), (100, 76), xycoords='axes points')
        axs[1].legend(loc = 'upper right')


        axs[2].set_title('Meridional Heat Transport vs Surface Zonal Wind at 60°S')
        axs[2].scatter(x_60,y_60, label = '60°S', color = 'yellow')
        axs[2].plot(x_60, regression_line_60, label='Fitted line', color = 'black')
        axs[2].annotate('Correlation = {:.2f}'.format(corr_60), (0,76), xycoords='axes points')
        axs[2].annotate('R² = {:.2f}'.format(R2_60), (100, 76), xycoords='axes points')
        axs[2].legend(loc = 'upper right')


        axs[3].set_title('Meridional Heat Transport vs Surface Zonal Wind at 50°S')
        axs[3].scatter(x_50,y_50, label = '50°S', color = 'green')
        axs[3].plot(x_50, regression_line_50, label='Fitted line', color = 'black')
        axs[3].annotate('Correlation = {:.2f}'.format(corr_50), (0,76), xycoords='axes points')
        axs[3].annotate('R² = {:.2f}'.format(R2_50), (100, 76), xycoords='axes points')
        axs[3].legend(loc = 'upper right')


        axs[4].set_title('Meridional Heat Transport vs Surface Zonal Wind at 40°S')
        axs[4].scatter(x_40,y_40, label = '40°S', color = 'blue')
        axs[4].plot(x_40, regression_line_40, label='Fitted line', color = 'black')
        axs[4].annotate('Correlation = {:.2f}'.format(corr_40), (0,76), xycoords='axes points')
        axs[4].annotate('R² = {:.2f}'.format(R2_40), (100, 76), xycoords='axes points')
        axs[4].legend(loc = 'upper right')


        axs[5].set_title('Meridional Heat Transport vs Surface Zonal Wind at 30°S')
        axs[5].scatter(x_30,y_30, label = '30°S', color = 'purple')
        axs[5].plot(x_30, regression_line_30, label='Fitted line', color = 'black')
        axs[5].annotate('Correlation = {:.2f}'.format(corr_30), (0,76), xycoords='axes points')
        axs[5].annotate('R² = {:.2f}'.format(R2_30), (100, 76), xycoords='axes points')
        axs[5].legend(loc = 'upper right')

        fig.tight_layout()
        fig.savefig('CESM_Zwind_Tgrad.jpg')

    elif which_figure == 2: # maximum value positions vs time
        fig1, axs1 = plt.subplots()
        axs1.plot(time_ax, max_wind[3], label='max wind position')
        axs1.plot(time_ax, non_zero_ice_line, label='sea ice edge', color='red')
        axs1.plot(time_ax, ice_line_half, label='sea ice edge', color='green')
        axs1.plot(time_ax, ice_line_three_q, label='sea ice edge', color='blue')
        axs1.plot(time_ax, max_T_grad, label='tgrad max', color='purple')
        axs1.plot(time_ax, max_WSC_SO, label='WSC max', color='orange')
        axs1.plot(time_ax, max_heat_transport, label='MHT', color='pink')
        axs1.set_xlim(0, 12 * 4)
        axs1.set_ylim(-90, -20)
        axs1.set_title("Maximum Zonal Wind Speed and Max Sea Ice Fraction vs time")
        axs1.set_xlabel('time (months)')
        axs1.legend()
        fig1.savefig('CESM_plot.jpg')

    elif which_figure == 3: # zonal wind and MHT profile

        fig2, axs2 = plt.subplots()
        axs2.plot(lat, np.mean(Z_wind, axis = 0), label = 'zonal wind')
        axs2_1 = axs2.twinx()
        axs2_1.plot(lat, np.mean(M_heat_trans, axis = 0), color = 'green', label = 'MHT')
        axs2.set_xlabel('Lattiude')
        axs2.set_ylabel('Z wind [m/s]')
        axs2_1.set_ylabel("MHT [Km/s]")
        axs2.set_title('CAM Annual Mean Surface MHT and Zonal Wind Speed vs. Latitude')

        fig2.legend(loc = (0.83,0.9))
        
        fig2.savefig('CESM_plot.jpg')

    elif which_figure == 4: # linear regression slope and intercepts v latitude (MHT and zonal wind)

        fig, axs = plt.subplots()

        axs.plot(lat, all_m, label = 'm')
        axs1 = axs.twinx()
        axs1.plot(lat, all_b, label = 'b', color = 'red')
        axs.set_ylabel('slope')
        axs1.set_ylabel('intercept')
        axs.set_xlabel('latitude')
        fig.legend(loc = (0.83,0.9))

        fig.tight_layout()

        fig.savefig('CESM_plot.jpg')

    elif which_figure == 5: # Hovmoller plot of MTG, Zonal Wind, and MHT
        
        fig, axs = plt.subplots(ncols= 3, figsize = (10,10))

        contour_1 = axs[0].contourf(time_ax[0:24],lat,T_grad[0:24,:].T,np.arange(np.min(T_grad),np.max(T_grad),0.01), extend = 'both',cmap=plt.get_cmap('bwr'))
        contour_2 = axs[1].contourf(time_ax[0:24],lat,Z_wind[0:24,:].T,np.arange(np.min(Z_wind),np.max(Z_wind),1), extend = 'both',cmap=plt.get_cmap('bwr'))
        contour_3 = axs[2].contourf(time_ax[0:24],lat,M_heat_trans[0:24,:].T,np.arange(np.min(M_heat_trans),np.max(M_heat_trans),1), extend = 'both',cmap=plt.get_cmap('bwr'))

        axs[0].set_title('Surface Temperature Gradient')
        axs[0].set_ylabel('latitude')
        axs[0].set_xlabel('time (months)')
        
        axs[1].set_title('Zonal Wind Velocity')
        axs[1].set_xlabel('time (months)')

        axs[2].set_title('Meridional Heat Transport')
        axs[2].set_xlabel('time (months)')

        
        plt.colorbar(contour_1,ax = axs[0])
        plt.colorbar(contour_2,ax = axs[1])
        plt.colorbar(contour_3,ax = axs[2])

        fig.tight_layout()


        fig.savefig('CESM_plot.jpg')

    elif which_figure == 6: # Hovmoller of southern Ocean Regression vs Data compairison

        fig, axs = plt.subplots(ncols = 4, figsize = (10,8))

        diff = np.array(Z_year_reg).T - np.array(Z_y_data).T
        fig.suptitle('Zonal Wind Regression from Surface Temperature Gradients in the Southern Ocean \n R² = {}'.format(R2_contour))
        
        axs[3].remove()
                
        contour_1 = axs[0].contourf(np.linspace(0,12,12),xdata_full[0],np.array(Z_year_reg).T,np.arange(13,50,1), extend = 'both',cmap=plt.get_cmap('Reds'))
        contour_2 = axs[1].contourf(np.linspace(0,12,12),xdata_full[0],np.array(Z_y_data).T,np.arange(13,50,1), extend = 'both',cmap=plt.get_cmap('Reds'))
        contour_3 = axs[2].contourf(np.linspace(0,12,12),xdata_full[0],diff, np.arange(-5,5,0.1), extend = 'both',cmap=plt.get_cmap('bwr'))

        
        axs[0].set_title('Regression Values')
        axs[1].set_title('Actual Values')
        axs[2].set_title('Difference')
        axs[0].set_ylabel('latitude')
        axs[0].set_xlabel('Month')
        axs[1].set_xlabel('Month')
        axs[2].set_xlabel('Month')
        plt.setp(axs[1].get_yticklabels(), visible=False)
        plt.setp(axs[2].get_yticklabels(), visible=False)

        #plt.colorbar(contour_1,ax = axs[0])
        plt.colorbar(contour_1, cax = fig.add_axes([0.8, 0.1, 0.05, 0.8]))
        plt.colorbar(contour_3, cax = fig.add_axes([0.9, 0.1, 0.05, 0.8]))

        fig.tight_layout()

        fig.savefig('CESM_plot.jpg')

    elif which_figure == 7:

        fig, axs = plt.subplots()

        axs.plot(np.linspace(0,12,12),normalized_diff, label = 'normalized difference')
        
        axs.set_xlabel('Month')
        axs.set_ylabel('Normalized Difference')
        axs.set_title("Normalized difference between optimized coefficient and coriolis coefficien PGRADt")
        
        axs.legend()

        fig.savefig('CESM_lineplot.jpg')

    elif which_figure == 8:

        fig, axs = plt.subplots(ncols = 3)


        contour_1 = axs[0].contourf(time_ax[0:12],lat,Z_wind.T[:,0:12],np.arange(-20,20,0.1), extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_2 = axs[1].contourf(time_ax[0:12],lat,Z_wind_reg.T[:,0:12],np.arange(-20,20,0.1), extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_3 = axs[2].contourf(time_ax[0:12],lat,T_grad.T[:,0:12],np.arange(-3,3,0.1), extend = 'both', cmap=plt.get_cmap('bwr'))
        #contour_4 = axs[3].contourf(time_ax[0:12],lat,Z_wind_reg.T[:,0:12],np.arange(-20,20,0.1), extend = 'both', cmap=plt.get_cmap('bwr'))
        #ice_contour = axs[0].plot(time_ax[0:12],max_ice[3][0:12], color = 'black', label = 'Sea Ice Extent')


        fig.savefig('CESM_plot.jpg')

    elif which_figure == 9:

        fig, axs = plt.subplots()

        pdf1, pdf2, pdf3, x_range, array_vals = pdf_analysis(T_grad, lat, non_zero_ice_line)
    
        # axs.plot(x_range,pdf1, color = 'blue', label = 'over ice')
        # axs.plot(x_range,pdf2, color = 'red', label = 'over water')
        axs.plot(x_range,pdf3, color = 'green',label = 'at margins')
        axs.set_xlabel('Surface Temperature Gradient')
        axs.set_ylabel('Probability Density')
        axs.set_title('PDF of Surface Temperature Gradients over Ice, Water, and at the Margins')
        breakpoint()
        axs.legend()
        axs.hist(array_vals[1],color = 'red', label = 'sub arctic tgrad', density = True, alpha = 0.1, histtype='stepfilled')
        axs.hist(array_vals[0],color = 'blue', label = 'over ice tgrad', density = True, alpha = 0.1, histtype='stepfilled')
        axs.hist(array_vals[2],color = 'green', label = 'around icelines tgrad', density = True, alpha = 0.1, histtype='stepfilled')


        # axs.hist(T_grad_val[1],color = 'red', label = 'sub arctic tgrad', density = True)
        # axs.hist(T_grad_val[0],color = 'blue', label = 'over ice tgrad', density = True)
        # axs.hist(T_grad_val[2],color = 'green', label = 'around icelines tgrad', density = True)
        # axs[1].plot(lat[0:30], np.mean(wind_stress[0:60],axis = 0)[0:30], label = "SO Summer")
        # axs[1].plot(lat[0:30], np.mean(WSC[0:60], axis = 0)[0:30], label = "SO Summer")

        # axs[1].plot(lat[0:30], np.mean(wind_stress[120:180],axis = 0)[0:30], label = "SO Winter")
        # axs[1].plot(lat[0:30], np.mean(WSC[120:180], axis = 0)[0:30], label = "SO Winter")

        # axs[1].plot(lat[0:30], np.mean(wind_stress[180:240],axis = 0)[0:30], label = "SO Srping")
        # axs[1].plot(lat[0:30], np.mean(WSC[180:240], axis = 0)[0:30], label = "SO Spring")
        # axs1 = axs[1].twinx()
        # axs1.plot(lat[0:30], np.mean(Z_wind[180:240], axis = 0)[0:30], label = "SO Spring")


        # axs[0].scatter(T_grad_val[4], T_grad_val[1], color = 'blue', label = 'sub arctic tgrad')
        # axs[1].scatter(T_grad_val[3], T_grad_val[0], color = 'red', label = 'over ice tgrad')
        # axs[2].scatter(T_grad_val[5], T_grad_val[2], color = 'green', label = 'around icelines tgrad')
        #axs[1].legend()

        fig.savefig('CESM_pdfanalysis.jpg')
    
    elif which_figure == 10: # MEBM, numpy, scipy and data comparison (linear zonally averaged regression analysis)

        fig, axs = plt.subplots(nrows = 4, figsize = (10,8))

        axs[0].plot(lat, np.mean(T_grad, axis = 0), color = 'red', label = 'GCM Tgrad')
        axs[0].plot(model_lat, np.mean(MEBM_Tgrad, axis = 0), color = 'blue', label = 'MEBM Tgrad')
        axs[0].set_ylabel('T grad')
        axs[0].set_title('MEBM v GCM Tgrad Comparison')
        axs[0].set_xlabel('latitude')
        axs[0].legend()

        axs[1].plot(lat, np.mean(Z_wind, axis = 0), color = 'green', label = 'GCM output')
        axs[1].plot(lat, np.mean(Z_wind_reg, axis = 0), color = 'red', label = 'Regression of GCM Tgrad', linestyle='dashed')
        axs[1].plot(model_lat, np.mean(Z_wind_MEBM_reg_scipy, axis = 0), color = 'orange', label = 'Regression on MEBM Tgrad using scipy interp')
        axs[1].plot(model_lat, np.mean(Z_wind_MEBM_reg_numpy, axis = 0), color = 'blue', label = 'Regression on MEBM Tgrad using numpy interp', linestyle='dashed')
        axs[1].set_ylabel('Wind Speed')
        axs[1].set_title('Zonal Wind of GCM output, Regression on GCM Tgrad, & Regression on MEBM Tgrad')
        axs[1].legend()

        axs[2].plot(lat, func_m(lat), color = 'green', label = 'scipy interpolated  m(lat)')
        axs[2].plot(lat, all_m, color = 'red', label = 'regression m values', linestyle='dashed')
        axs[2].plot(model_lat, m_interp, color = 'blue', label = 'numpy interpolation m values')
        axs[2].set_ylabel('m value')
        axs[2].set_title('m value for regression, numpy interpolation & scipy interpolation')
        axs[2].legend()

        axs[3].plot(lat, func_b(lat), color = 'green', label = 'scipy interpolated  b(lat)')
        axs[3].plot(lat, all_b, color = 'red', label = 'regression b values', linestyle='dashed')
        axs[3].plot(model_lat, b_interp, color = 'blue', label = 'numpy interpolation b values')
        axs[3].set_ylabel('b values')
        axs[3].set_title('b value for regression, numpy interpolation & scipy interpolation')
        axs[3].legend()

        fig.tight_layout()
        fig.savefig('CESM_plot.jpg')
    
    elif which_figure == 11:

        fig, axs = plt.subplots(nrows = 3)

        ##breakpoint()

        axs[0].plot(lat, np.mean(Z_year_reg, axis = 0), label = 'a')
        axs[0].plot(lat, np.mean(Z_wind, axis = 0), color = 'green', label = 'GCM output')
        axs[0].plot(lat, np.mean(U2d, axis = 0), color = 'red', label = 'U10')
        axs[1].plot(np.linspace(0,12,12), all_popts[:,1], label = 'b')
        axs[2].plot(np.linspace(0,12,12), all_popts[:,2] + all_popts[:,3], label = 'c+d')
       # axs[]
        #axs[3].plot(np.linspace(0,12,12), all_popts[:,3], label = 'd')

        fig.tight_layout()
        fig.savefig('CESM_plot.jpg')
    
    elif which_figure == 12: #contour plots of lon regressions

        fig, axs = plt.subplots(nrows = 2, ncols = 10, figsize = (10,6))

        contouring = np.arange(-15,15,0.1)

        
        contour_1 = axs[0,0].contourf(time_ax[0:12],lat,z_wind_at_each_lon[0][:][:].T[:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_2 = axs[0,1].contourf(time_ax[0:12],lat,z_wind_at_each_lon[14][:][:].T[:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_3 = axs[0,2].contourf(time_ax[0:12],lat,z_wind_at_each_lon[28][:][:].T[:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_4 = axs[0,3].contourf(time_ax[0:12],lat,z_wind_at_each_lon[42][:][:].T[:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_5 = axs[0,4].contourf(time_ax[0:12],lat,z_wind_at_each_lon[56][:][:].T[:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_6 = axs[0,5].contourf(time_ax[0:12],lat,z_wind_at_each_lon[70][:][:].T[:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_7 = axs[0,6].contourf(time_ax[0:12],lat,z_wind_at_each_lon[84][:][:].T[:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_8 = axs[0,7].contourf(time_ax[0:12],lat,z_wind_at_each_lon[98][:][:].T[:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_9 = axs[0,8].contourf(time_ax[0:12],lat,z_wind_at_each_lon[112][:][:].T[:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_10 = axs[0,9].contourf(time_ax[0:12],lat,z_wind_at_each_lon[126][:][:].T[:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))

        contour_1 = axs[1,0].contourf(time_ax[0:12],lat,Z_wind_lon[0][:][:][:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_2 = axs[1,1].contourf(time_ax[0:12],lat,Z_wind_lon[14][:][:][:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_3 = axs[1,2].contourf(time_ax[0:12],lat,Z_wind_lon[28][:][:][:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_4 = axs[1,3].contourf(time_ax[0:12],lat,Z_wind_lon[42][:][:][:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_5 = axs[1,4].contourf(time_ax[0:12],lat,Z_wind_lon[56][:][:][:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_6 = axs[1,5].contourf(time_ax[0:12],lat,Z_wind_lon[70][:][:][:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_7 = axs[1,6].contourf(time_ax[0:12],lat,Z_wind_lon[84][:][:][:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_8 = axs[1,7].contourf(time_ax[0:12],lat,Z_wind_lon[98][:][:][:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_9 = axs[1,8].contourf(time_ax[0:12],lat,Z_wind_lon[112][:][:][:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))
        contour_10 = axs[1,9].contourf(time_ax[0:12],lat,Z_wind_lon[126][:][:][:,0:12],contouring, extend = 'both', cmap=plt.get_cmap('bwr'))

        #plt.colorbar(contour_10, cax = fig.add_axes([0.8, 0.1, 0.05, 0.8]))

        fig.tight_layout()
        fig.savefig('CESM_lon_plot.jpg')

    elif which_figure == 13: # looking at lon regression m and bs

        fig, axs = plt.subplots(nrows = 11, ncols = 4, figsize = (10,12))

        z_wind_lon_reg_time_mean = np.mean(z_wind_at_each_lon, axis = 1)
        z_wind_lon_data_time_mean = np.mean(Z_wind_lon, axis = 0)

        fig.suptitle('Linear Regression Analysis of Sea Ice Fraction and Zonal Wind')
        
        axs[0,0].plot(lat, all_m, color = 'red')

        axs[1,0].plot(lat, m_for_each_lon[0,:])
        axs[2,0].plot(lat, m_for_each_lon[14,:])
        axs[3,0].plot(lat, m_for_each_lon[28,:])
        axs[4,0].plot(lat, m_for_each_lon[42,:])
        axs[5,0].plot(lat, m_for_each_lon[56,:])
        axs[6,0].plot(lat, m_for_each_lon[70,:])
        axs[7,0].plot(lat, m_for_each_lon[84,:])
        axs[8,0].plot(lat, m_for_each_lon[98,:])
        axs[9,0].plot(lat, m_for_each_lon[112,:])
        axs[10,0].plot(lat, m_for_each_lon[126,:])

        axs[0,0].set_title('m')

        axs[0,1].plot(lat, all_b, color = 'red')

        axs[1,1].plot(lat, b_for_each_lon[0,:])
        axs[2,1].plot(lat, b_for_each_lon[14,:])
        axs[3,1].plot(lat, b_for_each_lon[28,:])
        axs[4,1].plot(lat, b_for_each_lon[42,:])
        axs[5,1].plot(lat, b_for_each_lon[56,:])
        axs[6,1].plot(lat, b_for_each_lon[70,:])
        axs[7,1].plot(lat, b_for_each_lon[84,:])
        axs[8,1].plot(lat, b_for_each_lon[98,:])
        axs[9,1].plot(lat, b_for_each_lon[112,:])
        axs[10,1].plot(lat, b_for_each_lon[126,:])

        axs[0,1].set_title('b')


        axs[0,2].plot(lat, np.mean(Z_wind_reg, axis = 0), color = 'red')

        axs[1,2].plot(lat, z_wind_lon_reg_time_mean[0,:])
        axs[2,2].plot(lat, z_wind_lon_reg_time_mean[14,:])
        axs[3,2].plot(lat, z_wind_lon_reg_time_mean[28,:])
        axs[4,2].plot(lat, z_wind_lon_reg_time_mean[42,:])
        axs[5,2].plot(lat, z_wind_lon_reg_time_mean[56,:])
        axs[6,2].plot(lat, z_wind_lon_reg_time_mean[70,:])
        axs[7,2].plot(lat, z_wind_lon_reg_time_mean[84,:])
        axs[8,2].plot(lat, z_wind_lon_reg_time_mean[98,:])
        axs[9,2].plot(lat, z_wind_lon_reg_time_mean[112,:])
        axs[10,2].plot(lat, z_wind_lon_reg_time_mean[126,:])

        axs[0,2].set_title('time averaged regression of zonal wind')

        axs[0,3].plot(lat, np.mean(Z_wind, axis = 0), color = 'red')


        axs[0,3].set_title('GCM output of zonal wind')

        axs[1,3].plot(lat, z_wind_lon_data_time_mean[:,0])
        axs[2,3].plot(lat, z_wind_lon_data_time_mean[:,14])
        axs[3,3].plot(lat, z_wind_lon_data_time_mean[:,28])
        axs[4,3].plot(lat, z_wind_lon_data_time_mean[:,42])
        axs[5,3].plot(lat, z_wind_lon_data_time_mean[:,56])
        axs[6,3].plot(lat, z_wind_lon_data_time_mean[:,70])
        axs[7,3].plot(lat, z_wind_lon_data_time_mean[:,84])
        axs[8,3].plot(lat, z_wind_lon_data_time_mean[:,98])
        axs[9,3].plot(lat, z_wind_lon_data_time_mean[:,112])
        axs[10,3].plot(lat, z_wind_lon_data_time_mean[:,126])

        if True == True: #adds line thorugh y=0
            axs[0,0].axhline(y=0, color='black', linestyle='-')
            axs[1,0].axhline(y=0, color='black', linestyle='-')
            axs[2,0].axhline(y=0, color='black', linestyle='-')
            axs[3,0].axhline(y=0, color='black', linestyle='-')
            axs[4,0].axhline(y=0, color='black', linestyle='-')
            axs[5,0].axhline(y=0, color='black', linestyle='-')
            axs[6,0].axhline(y=0, color='black', linestyle='-')
            axs[7,0].axhline(y=0, color='black', linestyle='-')
            axs[8,0].axhline(y=0, color='black', linestyle='-')
            axs[9,0].axhline(y=0, color='black', linestyle='-')
            axs[10,0].axhline(y=0, color='black', linestyle='-')

            axs[0,0].set_ylabel('zonal average')
            axs[1,0].set_ylabel('0 longitude')
            axs[2,0].set_ylabel('35 longitude')
            axs[3,0].set_ylabel('70 longitude')
            axs[4,0].set_ylabel('105 longitude')
            axs[5,0].set_ylabel('140 longitude')
            axs[6,0].set_ylabel('175 longitude')
            axs[7,0].set_ylabel('210 longitude')
            axs[8,0].set_ylabel('245 longitude')
            axs[9,0].set_ylabel('280 longitude')
            axs[10,0].set_ylabel('315 longitude')


        


        fig.tight_layout()
        fig.savefig('CESM_lon_plot.jpg')

    elif which_figure == 14:

        fig, axs = plt.subplots(figsize = (8, 10))

        to_time = 12

        # contour_1 = axs.contourf(time_ax[0:to_time], lat[5:25], WSC.T[5:25,0:to_time] ,np.arange(-0.04,0.04,0.0001), cmap=plt.get_cmap('bwr'), extend = 'both')

        # axs.plot(time_ax[0:to_time], max_T_grad[0:to_time], color = 'red')
        # axs.plot(time_ax[0:to_time], non_zero_ice_line[0:to_time], color = 'white')
        # axs.plot(time_ax[0:to_time], ice_line_half[0:to_time], color = 'orange')
        # axs.scatter(time_ax[0:to_time], max_WSC_SO[0:to_time], color = 'purple')
        # axs.plot(time_ax[0:to_time], boundry_ice_line[0:to_time], color = 'black')

        contour_1 = axs.contourf(time_ax[0:to_time], lat[5:SO_upper], WSC_avg.T[5:SO_upper,0:to_time] ,np.arange(-0.04,0.04,0.0001), cmap=plt.get_cmap('bwr'), extend = 'both')

        axs.plot(time_ax[0:to_time], max_T_grad_avg[0:to_time], color = 'green', label = 'Tgrad Max', linewidth = 3)
        #axs.plot(time_ax[0:to_time], non_zero_ice_line_avg[0:to_time], color = 'white', label = 'Sea Ice Edge')
        axs.plot(time_ax[0:to_time], ice_line_half_avg[0:to_time], color = 'black', label = '50% Sea Ice Cover', linestyle = 'dashed', linewidth = 3)
        axs.plot(time_ax[0:to_time], max_WSC_SO_avg[0:to_time], color = 'yellow', label = 'WSC Max', linewidth = 3)
        axs.plot(time_ax[0:to_time], boundry_ice_line_avg[0:to_time], color = 'black', label = 'Contintental Boundry', linestyle = 'dotted')
        #axs.plot(time_ax[0:to_time], max_wind_speed_SO_avg[0:to_time], color = 'purple', label = 'maximum zonal wind speed')
        #axs.legend()

        #axs.set_title('Zonally Averaged Wind Stress Curl \n Over Southern Ocean')
        axs.set_xlabel('Months')
        axs.set_ylabel('Latitude (φ)')
        #plt.suptitle('Wind Stress Curl Map with Sea Ice and Temperature Gradients')


        plt.colorbar(contour_1)#, cax = fig.add_axes([0.8, 0.1, 0.05, 0.8]))


        fig.savefig('CESM_plot.pdf')

    elif which_figure == 15: #Atl, Pac, Ind wind stress, tgrad and sea ice contour map

        fig, axs = plt.subplots(ncols = 4, figsize = (10,7))

        to_time = 12

        days = np.linspace(0,365,12)


        #plt.suptitle('Wind Stress Curl Map with Sea Ice and Temperature Gradients')

        contour_1 = axs[0].contourf(days, lat[5:SO_upper], Pacific_WSC_avg.T[5:SO_upper,0:to_time] ,np.arange(-0.3,0.3,0.001), cmap=plt.get_cmap('bwr'), extend = 'both')
        contour_2 = axs[1].contourf(days, lat[5:SO_upper], Atlantic_WSC_avg.T[5:SO_upper,0:to_time] ,np.arange(-0.3,0.3,0.001), cmap=plt.get_cmap('bwr'), extend = 'both')
        contour_3 = axs[2].contourf(days, lat[5:SO_upper], Indian_WSC_avg.T[5:SO_upper,0:to_time] ,np.arange(-0.3,0.3,0.001), cmap=plt.get_cmap('bwr'), extend = 'both')

        axs[0].plot(days, Pacific_max_T_grad_avg[0:to_time], color = 'limegreen', label = 'Tgrad Max')
        axs[0].plot(days, Pacific_non_zero_ice_line_avg[0:to_time], color = 'white', label = 'Sea Ice Edge')
        axs[0].plot(days, Pacific_max_WSC_SO_avg[0:to_time], color = 'yellow', label = 'WSC Max')
        axs[0].plot(days, Pacific_ice_line_half_avg[0:to_time], color = 'black', label = '50% Sea Ice Cover', linestyle='dashed')
        axs[0].plot(days, Pacific_boundry_ice_line_avg[0:to_time], color = 'black', label = 'Contintental Boundry', linestyle='dotted')
        

        axs[1].plot(days, Atlantic_max_T_grad_avg[0:to_time], color = 'limegreen', label = 'Tgrad Max')
        axs[1].plot(days, Atlantic_non_zero_ice_line_avg[0:to_time], color = 'white', label = 'Sea Ice Edge')
        axs[1].plot(days, Atlantic_ice_line_half_avg[0:to_time], color = 'black', linestyle='dashed')
        axs[1].plot(days, Atlantic_max_WSC_SO_avg[0:to_time], color = 'yellow', label = 'WSC Max')
        axs[1].plot(days, Atlantic_boundry_ice_line_avg[0:to_time], color = 'black', linestyle='dotted')
        #axs[1].legend()

        axs[2].plot(days, Indian_max_T_grad_avg[0:to_time], color = 'limegreen', label = 'Tgrad Max')
        axs[2].plot(days, Indian_non_zero_ice_line_avg[0:to_time], color = 'white', label = 'Sea Ice Edge')
        axs[2].plot(days, Indian_ice_line_half_avg[0:to_time], color = 'black', linestyle='dashed', label = '50% Sea Ice Cover')
        axs[2].plot(days, Indian_max_WSC_SO_avg[0:to_time], color = 'yellow', label = 'WSC Max')
        axs[2].plot(days, Indian_boundry_ice_line_avg[0:to_time], color = 'black', linestyle='dotted', label = 'Contintental Boundry')
        axs[2].legend()

        axs[0].set_title('Pacific Ocean')
        axs[1].set_title('Atlantic Ocean')
        axs[2].set_title('Indian Ocean')

        axs[0].set_xlabel('Time (Days)')
        axs[1].set_xlabel('Time (Days)')
        axs[2].set_xlabel('Time (Days)')

        axs[1].set_yticklabels([])
        axs[2].set_yticklabels([])

        axs[0].set_ylabel('Latitude (φ)')

        axs[3].remove()

        plt.suptitle('Wind Stress Curl, Sea Ice and Meridional Temperature Gradients in CAM')

        #fig.legend()

        plt.colorbar(contour_1, cax = fig.add_axes([0.73, 0.1, 0.05, 0.8]))
        
        

        fig.savefig('CESM_plot_contour.png')

    return None

# Execute Sript
if __name__ == '__main__':
    generate_figures(6)
