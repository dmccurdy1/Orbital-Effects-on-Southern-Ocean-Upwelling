import numpy as np
import matplotlib.pyplot as plt
import milutin as ok

#age = ok.age()

#ins = ok.insolation()

# ins_65_ky0 = ok.insolation(0, latitude=65)
# ins_65_ky20 = ok.insolation(20, latitude=65)
# ins_65_diff = ins_65_ky0 - ins_65_ky20

# prec = ok.precession((100,0))
# ecc = ok.eccentricity((100,0))
# obl = ok.obliquity((100,0))

# # dobl_dkyr = np.gradient(obl)

# # wave_max_obl, wave_min_obl = ok.wave_peaks('obliquity',(100,0))
# # wave_max_prec, wave_min_rec = ok.wave_peaks('precession',(100,0))


# kyears = np.linspace(0,100,101)
# plt.plot(-kyears, obl)
# # plt1 = plt.twinx()
# # plt1.plot(-kyears, prec, color = 'orange')
# # [plt.axvline(i, color = 'pink') for i in wave_max_obl]
# # [plt.axvline(i, color = 'purple') for i in wave_min_obl]
# # [plt.axvline(i, color = 'blue') for i in wave_max_prec]
# # [plt.axvline(i, color = 'green') for i in wave_min_rec]
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

#inso = ok.insolation(kyear = (2000,0), latitude = 65, show_plot='On', output_type= 'global annual mean')
# inso = ok.insolation(kyear = (10,0), latitude = None, season = None, days = None, show_plot = 'On', output_type= 'array')


#inso = ok.insolation(kyear = [1,2,3,4,5], latitude = [1,2,3,4,5,6,7,8,9,10], show_plot= 'On', output_type='kyear mean')

#gmi = ok.insolation(kyear = (6000,0), output_type='global annual mean')


######### MATHIS TESTS #############
# def MathissTests():
#     myok = ok.insolation()
#     print(type(myok))
#     print(myok.size,myok.shape)
    
#     # method returns modern insolation by default, xarray directly drives plt contour
#     #plt.contour(myok)
#     #plt.show()
    
#     # built-in plot
    #myok2 = ok.insolation(show_plot = 'On',kyear=[21,0])#, latitude=66)
#     print(myok2.size,myok2.shape)
    
# if __name__=="__main__":
#     MathissTests()
#     print("Mathis is done testing")
# ######### MATHIS TESTS #############
    


#myok2 = ok.insolation(kyear=(21,0), latitude = (-55,-45), filename='test',show_plot=True)

#biglist = list(np.linspace(21,0,1000))

# arr = np.linspace(0,100,5)

# inso = ok.insolation(latitude = [-65,0,65], kyear = [20,10,0], output_type='kyear mean', season = 'DJF')
# breakpoint()

























#inso = ok.insolation(kyear=[0,1,2,3],output_type = 'kyear mean',latitude = [-10,-1,1,10])
inso = ok.insolation(latitude = [-45,45], kyear = (100,1),output_type='day mean')
breakpoint()
    


