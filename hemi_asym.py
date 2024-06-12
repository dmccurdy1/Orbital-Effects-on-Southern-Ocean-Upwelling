import milutin as ml
import numpy as np
import matplotlib.pyplot as plt

res = 10

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
    return diff, obl_ax, long_ax

diff_0, obl_ax, long_ax = hemi_asym_for_contours(0,res)
diff_0001, obl_ax, long_ax = hemi_asym_for_contours(0.01,res)
diff_001, obl_ax, long_ax = hemi_asym_for_contours(0.1,res)
diff_003, obl_ax, long_ax = hemi_asym_for_contours(0.3,res)
diff_006, obl_ax, long_ax = hemi_asym_for_contours(0.6,res)
diff_01, obl_ax, long_ax = hemi_asym_for_contours(0.9,res)

all_diffs = [diff_0,diff_0001,diff_001,diff_003,diff_006,diff_01]

# long_ax = np.linspace(0,360,res)
# obl_ax = np.linspace(0,360,res)

rows = 2
cols = 3

fig, axs = plt.subplots(nrows = rows, ncols = cols,figsize =(10,8) )

levs = np.arange(np.min(all_diffs),np.max(all_diffs),0.1)

contour = axs[0,0].contourf(long_ax,obl_ax,diff_0,levels = levs, extend = 'both')
contour = axs[0,1].contourf(long_ax,obl_ax,diff_0001,levels = levs, extend = 'both')
contour = axs[0,2].contourf(long_ax,obl_ax,diff_001,levels = levs, extend = 'both')
contour = axs[1,0].contourf(long_ax,obl_ax,diff_003,levels = levs, extend = 'both')
contour = axs[1,1].contourf(long_ax,obl_ax,diff_006,levels = levs, extend = 'both')
contour = axs[1,2].contourf(long_ax,obl_ax,diff_01,levels = levs, extend = 'both')

axs[0,0].set_title('Eccentricity = 0')
axs[0,1].set_title('Eccentricity = 0.001')
axs[0,2].set_title('Eccentricity = 0.01')
axs[1,0].set_title('Eccentricity = 0.03')
axs[1,1].set_title('Eccentricity = 0.06')
axs[1,2].set_title('Eccentricity = 0.1')

for i in range(rows):
    axs[i,0].set_ylabel('Obliquity')
    for f in range(cols):
        axs[1,f].set_xlabel('Longitude of Perihelion')
        
cb_ax = fig.add_axes([.85,.124,.04,.754])
fig.colorbar(contour,label = 'Annual Mean Hemispheric TOA Insolation Difference (W/m²)',cax = cb_ax, anchor = (10,1))
fig.suptitle('Annual Mean Radiative Hemispheric Imbalance Under Various Obrital Conditions')

plt.tight_layout(rect=[0,0,.8,1])
plt.savefig('testfig.png')