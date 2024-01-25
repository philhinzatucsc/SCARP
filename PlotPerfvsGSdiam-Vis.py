import numpy as np
import matplotlib.pyplot as plt

matrix = np.loadtxt('data/PerfvsWavelength-251.txt',skiprows=0,delimiter=',')

def ExtractWavel(wavel,matrix) :
    mask = (matrix[:,7] == wavel)

    #wavel = matrix[mask,7]
    open_FWHM = matrix[mask,12]
    av_FWHM = matrix[mask,13]
    gsd = matrix[mask,9]
    gain = open_FWHM / av_FWHM
    return gsd, gain
gsd32,gain32 =ExtractWavel(0.32,matrix)
gsd40,gain40 =ExtractWavel(0.40,matrix)
gsd50,gain50 =ExtractWavel(0.50,matrix)
gsd60,gain60 =ExtractWavel(0.60,matrix)
gsd70,gain70 =ExtractWavel(0.70,matrix)
gsd80,gain80 =ExtractWavel(0.80,matrix)
gsd90,gain90 =ExtractWavel(0.90,matrix)
gsd100,gain100 =ExtractWavel(1.00,matrix)

na = matrix[1,11]
e = matrix[1,6]
#wavel = matrix[1,7]
ngs = matrix[1,8]
hgs = matrix[1,10]




titlestring = 'actuators = {:.0f}, elevation = {:.0f}, \n N_GS={:d=.0f}, H_GS = {:.0f}'.format(na,e,ngs,hgs)


plt.plot(gsd32,gain32, '*',label='0.32 um')
plt.plot(gsd40,gain40, '*',label='0.40 um')
plt.plot(gsd50,gain50, '*',label='0.50 um')
plt.plot(gsd60,gain60, '*',label='0.60 um')
plt.plot(gsd70,gain70, '*',label='0.70 um')
plt.plot(gsd80,gain80, '*',label='0.80 um')
plt.plot(gsd90,gain90, '*',label='0.90 um')
plt.plot(gsd100,gain100, '*',label='1.0 um')
#plt.plot(gsd,av_FWHM, 'o',color='C0')
plt.ylim([1,4])
plt.xlabel('GS Diam. (arcmin)')
plt.ylabel('Averaged gain in FWHM ')
plt.title(titlestring)
plt.legend()
plt.savefig('plots/PerfvsGSDiam-Vis.png')
#plt.show()