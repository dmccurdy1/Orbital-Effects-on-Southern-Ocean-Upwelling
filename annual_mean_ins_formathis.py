from climlab.solar.orbital import OrbitalTable
from sea_ice_MEBM_ghost import Orbital_Insolation
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n = 120
dx = 2/n #grid box width
x = np.linspace(-1+dx/2,1-dx/2,n) #native grid
xb = np.linspace(-1+dx,1-dx,n-1) 
nt = 1000
dur= 1
dt = 1./nt

grid_dict = {'n': n, 'dx': dx, 'x': x, 'xb': xb, 'nt': nt, 'dur': dur, 'dt': dt}

#orbit = OrbitalTable

x = np.linspace(0,5000,5000)

y = np.load('5mya_AMI.npy')


#breakpoint()

# annual_mean_ins = []

# for i in range(1,5001):
    
#     x = np.mean(Orbital_Insolation().s_array(grid = grid_dict, kyear = i, lat_array = 'annual'))

#     annual_mean_ins.append(x)
fig, axs = plt.subplots()

axs.plot(x[0:3001],y[0:3001])
axs.set_xlabel('time [kyr BP]')
axs.set_ylabel('Annual Mean TOA Insolation [W/m²]')
axs.set_title('Annual Mean TOA Insolation vs Kyr')
axs.invert_xaxis()

plt.savefig('AMI_5mya.jpg')

data_list = [OrbitalTable.kyear.values[0:3001], y[0:3001]]


df = pd.DataFrame(data_list)
df = df.transpose()
df.columns = ['kyear', 'Annual Mean Insolation']

df.to_csv('kyear_v_Insolation.csv')

breakpoint()#