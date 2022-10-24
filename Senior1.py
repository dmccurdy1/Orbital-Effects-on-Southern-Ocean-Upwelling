import numpy as np
from scipy.special import logsumexp
import climlab


lat = np.linspace(-90,90,46)

print(lat)
s_hem_box = lat[:12]
mid_hem_box = lat[12:33]
n_hem_box = lat[34:]

# print('lat',lat)
# print('s',s_hem_box)
# print('m',mid_hem_box)
# print('n',n_hem_box)

s_hem_EBM = climlab.EBM(num_lat = 45)

### Variables ###

# s_hem_box = -45#lat[:46]
# mid_hem_box = lat[44:135] ###when i try to run this through EBM as an array it returns Nan for some values. ???
# n_hem_box = 45#lat[136:]

s_hem_lats = np.deg2rad(s_hem_box)
mid_hem_lats = np.deg2rad(mid_hem_box)
n_hem_lats = np.deg2rad(n_hem_box)


solar_constant = 1365.2 # W/m^2
alb = 0.3
epsilon = 1
steph_boltz = 5.67e-8 # W m^-2 K^-4 s^-1
euler_gamma_func = 0.608

#print(mid_hem_lats)
# print(s_hem_lats[3])
# print(np.cos(s_hem_lats[3]))

# print(s_hem_lats[4])
# print(np.cos(s_hem_lats[4]))


def s_EBM(solar_constant,alb,epsilon,steph_boltz,s_hem_lats):

    s_hem_Temp = abs(((1-alb)*solar_constant*(np.cos(s_hem_lats))) / (4*epsilon*steph_boltz))**(1/4)

    return(s_hem_Temp)

print('Southern Temp',s_EBM(solar_constant,alb,epsilon,steph_boltz,s_hem_box))

def mid_EBM(solar_constant,alb,epsilon,steph_boltz,mid_hem_lats):

    a = np.cos(mid_hem_lats)
    b = (1-alb)*solar_constant
    c = 4*epsilon*steph_boltz

    d = a*b
    
    mid_hem_Temp = (abs(d / c))**0.25
    
    #print('logsum',logsumexp(mid_hem_Temp))

    return(mid_hem_Temp)

print('mid Temp',mid_EBM(solar_constant,alb,epsilon,steph_boltz,mid_hem_box))



def n_EBM(solar_constant,alb,epsilon,steph_boltz,n_hem_lats):


    n_hem_Temp = (((1-alb)*solar_constant*(np.cos(n_hem_lats))) / (4*epsilon*steph_boltz))**(1/4)

    return(n_hem_Temp)

#print('Northern Temp', n_EBM(solar_constant,alb,epsilon,steph_boltz,n_hem_box))

