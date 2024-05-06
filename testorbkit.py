import numpy as np
import matplotlib.pyplot as plt
import orbkit as ok

#age = ok.age()

#ins = ok.insolation()

# ins_65_ky0 = ok.insolation(0, latitude=65)
# ins_65_ky20 = ok.insolation(20, latitude=65)
# ins_65_diff = ins_65_ky0 - ins_65_ky20

# prec = ok.precession((100,0))
# ecc = ok.eccentricity((100,0))
# obl = ok.obliquity((100,0))

# dobl_dkyr = np.gradient(obl)

# wave_max_obl, wave_min_obl = ok.wave_peaks('obliquity',(100,0))
# wave_max_prec, wave_min_rec = ok.wave_peaks('precession',(100,0))


# kyears = np.linspace(0,100,101)
# plt.plot(-kyears, obl)
# plt1 = plt.twinx()
# plt1.plot(-kyears, prec, color = 'orange')
# [plt.axvline(i, color = 'pink') for i in wave_max_obl]
# [plt.axvline(i, color = 'purple') for i in wave_min_obl]
# [plt.axvline(i, color = 'blue') for i in wave_max_prec]
# [plt.axvline(i, color = 'green') for i in wave_min_rec]
# plt.savefig('orbkit_testplot.png')


# clim_state = ok.climate()



# N65_all_kyear = []
# for i in range(0,2000):

#     N65_kyear_i = ok.insolation(i)
#     N65_kyear_i_max = np.mean(N65_kyear_i)

#     N65_all_kyear.append(N65_kyear_i_max)

# N65 = np.hstack(N65_all_kyear)

# plt.plot(-np.linspace(0,2000,len(N65)), N65)


# plt.savefig('orbkit_testplot.png')s

#(0.001,25,300)
# inso_m = ok.insolation(kyear = (0,0,0), latitude = [1,65], output_type = 'latitude mean')
# inso = ok.insolation(kyear = (0,0,0), latitude = [1,65])#, output_type = 'latitude mean')

# inso_tuple_1 = ok.insolation(kyear = None, latitude = [1,5,7], output_type = 'array')
# inso_tuple_1_lm = ok.insolation(kyear = None,  latitude = [1,5,7], output_type = 'latitude mean')
#inso_tuple_1_tm = ok.insolation(kyear = None, latitude = [10], output_type = 'time mean')

#ploting = ok.insolation(kyear= 2000, latitude = None, output_type = 'array', show_plot= 'On')

#clim_state = ok.climate(latitude = 1)

#inso = ok.insolation(kyear = (50,0), latitude = (0,90), output_type='array', show_plot='On')
inso = ok.insolation(kyear = (10,0),latitude = (0,30), show_plot = 'On', output_type = 'global annual mean')


breakpoint()


