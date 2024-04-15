import numpy as np
import matplotlib.pyplot as plt
import orbkit as ok

#age = ok.age()

#ins = ok.insolation()

ins_65_ky0 = ok.insolation(0, latitude=65)
ins_65_ky20 = ok.insolation(20, latitude=65)
ins_65_diff = ins_65_ky0 - ins_65_ky20

obl = ok.obliquity((2000,0))

kyears = np.linspace(0,2000,2001)
plt.plot(-kyears, obl)
plt.savefig('orbkit_testplot.png')

GMI = ok.global_mean_insolation((109,76,9))

breakpoint()

