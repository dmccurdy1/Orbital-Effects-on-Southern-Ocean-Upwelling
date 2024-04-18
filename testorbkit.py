import numpy as np
import matplotlib.pyplot as plt
import orbkit as ok

#age = ok.age()

#ins = ok.insolation()

ins_65_ky0 = ok.insolation(0, latitude=65)
ins_65_ky20 = ok.insolation(20, latitude=65)
ins_65_diff = ins_65_ky0 - ins_65_ky20

prec = ok.precession((100,0))
ecc = ok.eccentricity((100,0))
obl = ok.obliquity((100,0))

dobl_dkyr = np.gradient(obl)

wave_max_obl, wave_min_obl = ok.wave_peaks('obliquity',(100,0))
wave_max_prec, wave_min_rec = ok.wave_peaks('precession',(100,0))


kyears = np.linspace(0,100,101)
plt.plot(-kyears, obl)
plt1 = plt.twinx()
plt1.plot(-kyears, prec, color = 'orange')
#plt.plot(-kyears, dobl_dkyr, color = 'red')
#plt1 = plt.twinx()
#plt1.plot(-kyears, dobl_dkyr, color = 'green')
[plt.axvline(i, color = 'pink') for i in wave_max_obl]
[plt.axvline(i, color = 'purple') for i in wave_min_obl]
[plt.axvline(i, color = 'blue') for i in wave_max_prec]
[plt.axvline(i, color = 'green') for i in wave_min_rec]
plt.savefig('orbkit_testplot.png')

clim_state = ok.climate((0,20,180))

breakpoint()

