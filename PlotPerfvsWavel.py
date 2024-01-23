import numpy as np
import matplotlib.pyplot as plt

matrix = np.loadtxt('data/PerfvsWavelength.txt',skiprows=1,delimiter=',')

mask = (matrix[:,0] == 251)

wavel = matrix[mask,7]
open_FWHM = matrix[mask,12]
av_FWHM = matrix[mask,13]

plt.plot(wavel,open_FWHM, '*',color='C1')
plt.plot(wavel,av_FWHM, 'o',color='C0')
plt.ylim([0,0.8])
plt.xlabel('Observing Wavelength (um)')
plt.ylabel('Averaged FWHM over 9 arcmin square (arcesec)')
plt.savefig('plots/PerfvsWavelength.png')
#plt.show()