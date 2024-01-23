import numpy as np
import matplotlib.pyplot as plt

matrix = np.loadtxt('data/PerfvsWavelength-251.txt',skiprows=0,delimiter=',')

mask = (matrix[:,9] == 16)

wavel = matrix[mask,7]
open_FWHM = matrix[mask,12]
av_FWHM = matrix[mask,13]

na = matrix[1,11]
e = matrix[1,6]
gsd = matrix[1,9]
ngs = matrix[1,8]
hgs = matrix[1,10]



titlestring = 'actuators = {:.0f}, elevation = {:.0f}, \n GS diam.={:.1f}, N_GS={:d=.0f}, H_GS = {:.0f}'.format(na,e,gsd,ngs,hgs)


plt.plot(wavel,open_FWHM, '*',color='C1')
plt.plot(wavel,av_FWHM, 'o',color='C0')
plt.ylim([0,0.8])
plt.xlabel('Observing Wavelength (um)')
plt.ylabel('Averaged FWHM over 9 arcmin square (arcsec)')
plt.title(titlestring)
plt.savefig('plots/PerfvsWavelength-Vis.png')
#plt.show()