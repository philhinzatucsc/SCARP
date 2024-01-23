import numpy as np
import matplotlib.pyplot as plt

matrix = np.loadtxt('data/PerfvsWavelength-51.txt',skiprows=0,delimiter=',')

def ExtractWavel(wavel,matrix) :
    mask = (matrix[:,7] == wavel)

    #wavel = matrix[mask,7]
    open_FWHM = matrix[mask,12]
    av_FWHM = matrix[mask,13]
    gsd = matrix[mask,9]
    gain = open_FWHM / av_FWHM
    return gsd, gain
gsd2_2,gain2_2 =ExtractWavel(2.2,matrix)
gsd1_65,gain1_65 =ExtractWavel(1.65,matrix)
gsd1_25,gain1_25 =ExtractWavel(1.25,matrix)
gsd1_05,gain1_05 =ExtractWavel(1.05,matrix)


na = matrix[1,11]
e = matrix[1,6]
#wavel = matrix[1,7]
ngs = matrix[1,8]
hgs = matrix[1,10]




titlestring = 'actuators = {:.0f}, elevation = {:.0f}, \n N_GS={:d=.0f}, H_GS = {:.0f}'.format(na,e,ngs,hgs)


plt.plot(gsd1_05,gain1_05, '*',label='1.05 um')
plt.plot(gsd1_25,gain1_25, '*',label='1.25 um')
plt.plot(gsd1_65,gain1_65, '*',label='1.65 um')
plt.plot(gsd2_2,gain2_2, '*',label='2.2 um')
#plt.plot(gsd,av_FWHM, 'o',color='C0')
plt.ylim([1,8])
plt.xlabel('GS Diam. (arcmin)')
plt.ylabel('Averaged gain in FWHM ')
plt.title(titlestring)
plt.legend()
plt.savefig('plots/PerfvsGSDiam-NIR.png')
#plt.show()