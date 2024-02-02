import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import time
#from scipy import interpolation
import scipy.io as sio
import scipy.interpolate as interp
import mixedlayer
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from climlab.solar.orbital.long import OrbitalTable
import climlab
from climlab.solar.insolation import daily_insolation


years = np.linspace(-2000, 0, 2001)

# subset of orbital parameters for specified time
orb = OrbitalTable.interp(kyear=years)




# def local_avg():

#     #day = np.linspace(0,365,1000)
      
#     avg_lat = []
#     #avg_day = []

#     #breakpoint()
#     count = 0
#     counter = 0

#     for i in range(65,66,2):

#         avg_day = []
#         counter = counter+1

#         for f in range(151, 243):

#             count = count+1
            
#             day_at_lat = daily_insolation(orb,f,i)

#             day_at_lat = list(day_at_lat)

#             #breakpoint()

#             avg_day.append(day_at_lat)
        
#         #print("}}}}}}}}}}}}",avg_day) #avg_day should have shape (1000,2000)
        
#         #avg_day = np.array(avg_day)

#         #avg_day = np.mean(avg_day)

#         #avg_lat.append(avg_day)

#         avg_day = np.mean(avg_day, axis=0) # average insoaltion of all days at a single lattitude for 2Mya

#         avg_lat.append(avg_day)

#     #breakpoint()

#     avg_lat = np.array(avg_lat)

#     #a#vg_lat = avg_lat.T

#     avg_lat = np.mean(avg_lat, axis= (0))

#     #diff = avg_lat - mean_lat

#     #breakpoint()

#     return avg_lat

# #print(daily_insolation(orb,172,45))
# #print(local_avg())

#S65 = daily_insolation(lat=65, day=172, orb=orb)

def func():

    ins = []

    for i in range(151,243):
        output = daily_insolation(day = i,lat=65,orb=orb)

        ins.append(output)

    ins = np.mean(ins,axis = 0)

    return ins


def fig():
    fig,axs = plt.subplots(figsize = (14,5))


    axs.plot(-orb['kyear'],func())
    plt.savefig('clim_plot.jpg')

f = daily_insolation(day = 172,lat = 65,orb=orb)

# fig()
# # print(func())
# breakpoint()

import numpy as np
import matplotlib.pyplot as plt
import climlab

#  for convenience, set up a dictionary with our reference parameters
param = {'A':210, 'B':2, 'a0':0.3, 'a2':0.078, 'ai':0.62, 'Tf':-10.}
model1 = climlab.EBM_annual(name='Annual EBM with ice line', 
                            num_lat=180, D=0.55, **param )

def ebm_plot( model, figsize=(8,12), show=True ):
    '''This function makes a plot of the current state of the model,
    including temperature, energy budget, and heat transport.'''
    templimits = -30,35
    radlimits = -340, 340
    htlimits = -7,7
    latlimits = -90,90
    lat_ticks = np.arange(-90,90,30)
    
    fig, ax1 = plt.subplots(figsize=(8,4))
    
    Ts = np.mean(model.Ts, axis = 1)
    Ts = np.array(Ts)


    ax1.plot(model.lat,model.Ts, color = 'blue', label = "Temperature")
    ax_1 = ax1.twinx()
    ax_1.plot(model.lat,np.gradient(Ts), color = "red", label = "Temperature Gradient")
    ax_2 = ax1.twinx()
    ax_2.plot(model.lat,1-model.albedo, color = 'green', label = "co-albedo")
    ax1.set_ylabel('Temperature (deg C)') 
    ax_1.set_ylabel('Temperature Difference (deg C)') 
    ax1.set_xlabel("Lattitude")
    ax1.set_title("Climlab EBM_Annual Temp and Temp Gradient vs Latitude")
    ax1.legend(loc = (0.661,0.79))
    ax_1.legend()



#     #ax1 = fig.add_subplot(3,1,1)
#     ax1.plot(model.lat, model.Ts, label = "Temperature")
#     ax_1 = ax1.twinx()
#     ax_1.plot(model.lat, np.gradient(Ts), color = 'red', label = "Temperature Gradient")
#     #ax1.set_xlim(latlimits)
#    # ax1.set_ylim(templimits)
#     ax1.set_ylabel('Temperature (deg C)')
#     ax_1.set_ylabel('Temperature Difference (deg C)')
#     ax1.set_xlabel('Latitude')
#     ax1.set_title('Climlab EBM_Annual Temp and Temp Gradient vs Latitude')
#     #ax1.set_xticks( lat_ticks )
#     ax1.legend(loc = (0.661,0.79))
#     ax_1.legend()
#     ax1.grid()

    # Ts = np.mean(model.Ts, axis = 1)
    # Ts = np.array(Ts)


    # ax2 = fig.add_subplot(3,1,2)

    # ax2.plot(model.lat,np.gradient(Ts))
    # #ax2.plot(model.lat,model.net_radiation)
   
    # ax3 = fig.add_subplot(3,1,3)
    # ax3.plot(model.lat, 1-model.albedo)
    
    # ax3 = fig.add_subplot(3,1,3)
    # ax3.plot(model.lat_bounds, model.heat_transport)
    # ax3.set_xlim(latlimits)
    # ax3.set_ylim(htlimits)
    # ax3.set_ylabel('Heat transport (PW)')
    # ax3.set_xlabel('Latitude')
    # ax3.set_xticks( lat_ticks )
    # ax3.grid()

    plt.savefig('MEBMplots.jpg')

    return fig

model1.integrate_years(5)
f = ebm_plot(model1)


# print(model1.net_radiation)