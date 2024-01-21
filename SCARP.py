import numpy as np
import matplotlib.pyplot as plt
import calcpsf as C
import multiprocessing
from joblib import Parallel, delayed


'''
SCARP is the Software-Calculated AO-Reconstituted PSF simulation 
It is based on the formalism from Tokovinin 2004
It is being developed to support performance predictions for GEO
Phil Hinz 2024
'''

#wavel = 0.7E-6              #Wavelength for PSF estimation
wavels = [0.32E-6,0.4E-6,0.5E-6,0.6E-6,0.7E-6,0.8E-6, 0.9E-6,1.0E-6]
elev = 60                   #elevation of observation
field_size = 9.0            #field size of observations (arcminutes)
num_field_points = 81       #number of points in field to sample.  Should be the square of an odd number (to get central value and fit on a grid)           
h_gs = 99000                #Height of guide stars  
gs_diam = 16                #diameter guide stars
num_gs = 6                  #Number of guide stars
low_perc = 1                #index for low layers percentile (0-2)  lower is better conditions
hi_perc = 1                 #index for high layers percentile (0-2)  lower is better conditions
L_0 = 30                    #outer scale of layers
num_act = 2000
#num_actuators = [200,500,1000,2000,4000]
#num_pix = 1001             #number of pixels on a side in images. Odd, so we have a central pixel.  Is this really necessary??  One reason to keep it this way is the Bessel function fix 
#num_pix = 51                #good for 1 um and longer
#num_pix = 101               # good for 0.7 um and longer
num_pix = 251               # good for 0.32 um and longer

#Write Column Headers to File
output_file_name = 'PerfvsWavelength.txt'
file = open(output_file_name,'w')
file.write("nn, field_size, field_points, lp, hp ,L_0, elevation, wavel, num_guide_stars, gsdiam, H, numactuators openFWHM, avFWHM, stdFWHM, maxFWHM, minFWHM \n")
file.close()

#https://medium.com/@mjschillawski/quick-and-easy-parallelization-in-python-32cb9027e490
#num_cores = multiprocessing.cpu_count()

if __name__ == "__main__":
    #results = Parallel(n_jobs=-1)(delayed(C.CalcPSF)(num_pix, field_size, num_field_points, low_perc, hi_perc, L_0, elev, wavel, num_gs, gs_diam, h_gs, num_act,output_file_name) for num_act in num_actuators)
    results = Parallel(n_jobs=-1)(delayed(C.CalcPSF)(num_pix, field_size, num_field_points, low_perc, hi_perc, L_0, elev, wavel, num_gs, gs_diam, h_gs, num_act,output_file_name) for wavel in wavels)     
    


