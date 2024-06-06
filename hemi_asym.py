import milutin as ml
import numpy as np
import matplotlib.pyplot as plt

def hemi_asym_for_contours(ecc, resolution = 100):
    obl_ax = []
    inso_no_ecc_S = []
    inso_no_ecc_N = []
    for i in range(resolution):
        ecc = ecc
        obl = (180/resolution)*i
        inso_row_S = []
        inso_row_N = []
        long_ax = []
        for f in range(resolution):
            long_peri = (360/resolution)*f
            inso_S = np.array(ml.insolation(kyear = (ecc,obl,long_peri),latitude=(-90,0),show_plot=False,output_type='latitude day mean'))
            inso_N = np.array(ml.insolation(kyear = (ecc,obl,long_peri),latitude=(0,90),show_plot=False,output_type='latitude day mean'))
            inso_row_S.append(inso_S)
            inso_row_N.append(inso_N)
            long_ax.append(long_peri)
        inso_no_ecc_S.append(inso_row_S)
        inso_no_ecc_N.append(inso_row_N)
        obl_ax.append(obl)
    inso_no_ecc_S = np.array(inso_no_ecc_S)
    inso_no_ecc_N = np.array(inso_no_ecc_N)
    diff = inso_no_ecc_S - inso_no_ecc_N
    return diff

diff_0 = hemi_asym_for_contours(0)
diff_0001 = hemi_asym_for_contours(0.001)
diff_001 = hemi_asym_for_contours(0.01)
diff_003 = hemi_asym_for_contours(0.03)
diff_006 = hemi_asym_for_contours(0.06)
diff_01 = hemi_asym_for_contours(0.1)

long_ax = np.linspace(0,360,100)
obl_ax = np.linspace(0,180,100)



fig, axs = plt.subplots(nrows = 2, ncols = 3)

contour = axs[0,0].contourf(long_ax,obl_ax,diff_0)
contour = axs[0,1].contourf(long_ax,obl_ax,diff_0001)
contour = axs[0,2].contourf(long_ax,obl_ax,diff_001)
contour = axs[1,0].contourf(long_ax,obl_ax,diff_003)
contour = axs[1,1].contourf(long_ax,obl_ax,diff_006)
contour = axs[1,2].contourf(long_ax,obl_ax,diff_01)

axs[0,0].set_title('0')
axs[0,1].set_title('0.001')
axs[0,2].set_title('0.01')
axs[1,0].set_title('0.03')
axs[1,1].set_title('0.06')
axs[1,2].set_title('0.1')
#plt.colorbar(contour, label = 'Insolation (W/m²)')
plt.tight_layout()
plt.savefig('testfig.png')