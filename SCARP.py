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

#wavel = 1.0E-6              #Wavelength for PSF estimation
#wavels = [0.32E-6,0.4E-6,0.5E-6,0.6E-6,0.7E-6,0.8E-6, 0.9E-6,1.0E-6]
#wavels = [1.05E-6, 1.25E-6, 1.65E-6, 2.2E-6]
#wavels = [0.32E-6,0.5E-6,1.0E-6,1.65E-6,2.2E-6]
wavels = [0.5E-6]
elev = 90                   #elevation of observation

num_field_points = 81       #number of points in field to sample.  Should be the square of an odd number (to get central value and fit on a grid)           
h_gs = 99000                #Height of guide stars  
gs_diam = 2                #diameter guide stars
field_size = gs_diam * 9 / 16           #field size of observations (arcminutes)
num_gs = 6                  #Number of guide stars
low_perc = 1                #index for low layers percentile (0-2)  lower is better conditions
hi_perc = 1                 #index for high layers percentile (0-2)  lower is better conditions
L_0 = 30 * 1E6                   #outer scale of layers
num_act = 2000
#num_actuators = [200,500,1000,2000, 3000,4000,5000,6000]
num_actuators = [2000]

#Write Column Headers to File
output_file_name = 'data/log.txt'
#file = open(output_file_name,'a')
#file.write("nn, field_size, field_points, lp, hp ,L_0, elevation, wavel, num_guide_stars, gsdiam, H, numactuators openFWHM, avFWHM, stdFWHM, maxFWHM, minFWHM \n")
#file.close()

for wavel in wavels :
    C.CalcPSF(field_size*8, num_field_points, low_perc, hi_perc, L_0, elev, wavel, num_gs, gs_diam*8, h_gs, num_act,output_file_name)         
    
    
'''
#https://medium.com/@mjschillawski/quick-and-easy-parallelization-in-python-32cb9027e490
num_cores = multiprocessing.cpu_count()
num_cores = 5       #set to 5 for macbook nn=251 for macbook to avoid running out of memory

if __name__ == "__main__":
    #results = Parallel(n_jobs=num_cores)(delayed(C.CalcPSF)(field_size, num_field_points, low_perc, hi_perc, L_0, elev, wavel, num_gs, gs_diam, h_gs, num_act,output_file_name) for num_act in num_actuators)
    #results = Parallel(n_jobs=num_cores)(delayed(C.CalcPSF)(field_size, num_field_points, low_perc, hi_perc, L_0, elev, wavel, num_gs, gs_diam, h_gs, num_act,output_file_name) for wavel in wavels)
    #results = Parallel(n_jobs=num_cores)(delayed(C.CalcPSF)(field_size*2, num_field_points, low_perc, hi_perc, L_0, elev, wavel, num_gs, gs_diam*2, h_gs, num_act,output_file_name) for wavel in wavels)
    #results = Parallel(n_jobs=num_cores)(delayed(C.CalcPSF)(field_size*4, num_field_points, low_perc, hi_perc, L_0, elev, wavel, num_gs, gs_diam*4, h_gs, num_act,output_file_name) for wavel in wavels) 
    results = Parallel(n_jobs=num_cores)(delayed(C.CalcPSF)(field_size*8, num_field_points, low_perc, hi_perc, L_0, elev, wavel, num_gs, gs_diam*8, h_gs, num_act,output_file_name) for wavel in wavels)         
    
'''

