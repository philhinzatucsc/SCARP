import numpy as np
import matplotlib.pyplot as plt

matrix = np.loadtxt('data/PerfvsActuators.txt',skiprows=1,delimiter=',')

mask = (matrix[:,7] == 0.32)
actuators = matrix[mask,11]
open_FWHM = matrix[mask,12]
av_FWHM = matrix[mask,13]
gain_032 = open_FWHM / av_FWHM 

mask_2 = (matrix[:,7] == 0.5)
actuators_2 = matrix[mask_2,11]
open_FWHM_2 = matrix[mask_2,12]
av_FWHM_2 = matrix[mask_2,13]
gain_040 = open_FWHM_2 / av_FWHM_2 



mask_3 = (matrix[:,7] == 0.8)
actuators_3 = matrix[mask_3,11]
open_FWHM_3 = matrix[mask_3,12]
av_FWHM_3 = matrix[mask_3,13]
gain_080 = open_FWHM_3 / av_FWHM_3 

plt.plot(actuators,gain_032, '*',label='0.32 um')
plt.plot(actuators_2,gain_040, '*',label='0.5 um')
plt.plot(actuators_3,gain_080, '*',label='0.8 um')

#plt.plot(actuators,av_FWHM, 'o',color='C0')
plt.ylim([1,3])
plt.xlabel('Number of Actuators')
plt.ylabel('Gain in Averaged FWHM over 9 arcmin square')
plt.legend()
plt.savefig('plots/PerfvsActuators.png')
#plt.show()