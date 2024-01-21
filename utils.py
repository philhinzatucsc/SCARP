import numpy as np
import matplotlib.pyplot as plt
#from imexam.imexamine import Imexamine
#from astropy.modeling.functional_models import Moffat2D
from astropy.modeling import models, fitting
from scipy.special import jv

'''
utility functions for SCARP
'''

#FT functions
def AptoI(img):
    """
    Create PSF and OTF from aperture
    """
    (imside1,imside2) =np.shape(img)
    F = np.fft.fft2(img, (imside1, imside2))
    H = F * np.conj(F)
    I = np.real(np.fft.fftshift(H))
    normI = I / np.max(I)           #normalized PSF
    h = np.fft.ifft2(H, (imside1, imside2))
    OTF = np.fft.ifftshift(h)
    return normI, OTF

def normFT(OTF) :
    """
    Create PSF from OTF
    """
    (imside1,imside2) =np.shape(OTF)
    PSF = np.fft.fft2(OTF, (imside1,imside2))
    PSF = np.fft.fftshift(PSF)
    PSF = np.real(np.sqrt(PSF * np.conj(PSF)))
    peak = np.max(PSF)
    PSF = PSF / peak
    return PSF

def fitMoffat(PSF) :
    """
    Fit Moffat profile to PSF image
    Returns FWHM in pixels
    """
    (imside1,imside2) =np.shape(PSF)
    x = np.arange(imside1)
    y = np.arange(imside2) 
    X, Y = np.meshgrid(x, y)
    PSFmodel = models.Moffat2D(amplitude=np.max(PSF), x_0=imside1/2,y_0=imside2/2,gamma=100, alpha=2)        
    fitter = fitting.LevMarLSQFitter()
    fitted_model = fitter(PSFmodel,X,Y,PSF)
    return fitted_model.fwhm, fitted_model.alpha.value

