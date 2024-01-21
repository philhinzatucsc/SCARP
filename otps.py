import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from scipy.special import jv

'''
Optical Turbulence Profiles for Mauna Kea
 used in SCARP
'''



# Define Optical Turbulence Profile
#Chun model
lowlayers = [ 1,15,30,45,60,75,90,105,120,160,240,320,400,480,560]
Jlowcoeff = np.array([[48.6, 115., 167.],
                   [53.0, 45.1, 79.9],
                   [19.7, 15.8,  14.0 ],
                   [6.53 , 11.0 , 12.3],
                   [0.135, 0.412 ,2.86 ],
                   [0.017 ,0.11 , 1.72],
                   [0.00 , 0.0258, 3.55],
                   [0.00 , 0.0089, 1.87],
                   [0.00 , 0.073, 2.35],
                   [2.4 , 5.55, 10.9],
                   [2.53 , 4.51, 9.37],
                   [1.06 , 1.81, 4.03],
                   [1.09 , 0.699, 1.57],
                   [0.307 , 0, 0],
                   [0.264 , 0.145, 0.248]])
highlayers = [500,700,1000,1500,2000,3000,4000,6000,8000,12000,16000,22000 ]
Jhighcoeff = np.array([[6.72 , 10.1, 15.5],
                   [2.54 , 4.54, 10.3],
                   [1.88 , 3.68, 9.04],
                   [1.32 , 2.74, 6.19],
                   [2.28 , 4.84, 10.2],
                   [2.51 , 6.33, 15.5],
                   [2.59 , 6.69, 19.1],
                   [4.01 , 10.3, 31.8],
                   [7.66 , 18.3, 38.2],
                   [9.43 , 14.8, 15.7],
                   [3.20 , 3.08, 3.87],
                   [2.64 , 3.89, 6.75]])
Jlow = 1 * Jlowcoeff *10**-15
Jhigh = 1 * Jhighcoeff *10**-15
numlayers = len(lowlayers)+len(highlayers)
