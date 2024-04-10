import numpy as np
import matplotlib.pyplot as plt
import orbkit as ok

#age = ok.age()

#ins = ok.insolation()

ins_65 = ok.insolation((2000,0), latitude=65)

obl = ok.obliquity((2000,0))

kyears = np.linspace(0,2000,2001)

plt.plot(-kyears, obl)

plt.savefig('orbkit_testplot.png')
#age = ok.age(80)

breakpoint()

