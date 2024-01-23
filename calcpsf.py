import numpy as np
import matplotlib.pyplot as plt
import utils as U
import otps as O
from scipy.special import jv


def CalcPSF(nn, field_size, field_points, lp, hp, L_0, elevation, wavel, num_guide_stars, gsdiam, H, numactuators, outname) :
    '''
    Top level function to execute a calculation
    '''
    
    '''
    nn = inputs[0]
    field_size = inputs[1]
    field_points = inputs[2]
    lp = inputs[3]
    hp = inputs[4]
    L_0 = inputs[5]
    elevation = inputs[6]
    wavel = inputs[7]
    num_guide_stars = inputs[8]
    gsdiam = inputs[9]
    H = inputs[10]
    numactuators = inputs[11]
    '''

    #Setup Guide star asterism
    theta_0 = gsdiam/2                #set science field as guide star size
    ZA = 90 - elevation             #Zenith Angle
    ZA_rad = ZA * np.pi / 180.0

    #lowpercentile = 0
    #hipercentile = 2

    gridside = int(np.sqrt(field_points))
    #alpha_fp_arcmin = field_size  * np.array([1,0,-1,0,0])
    #beta_fp_arcmin = field_size  * np.array([0,1,0,-1,0])
                


    dim = [nn, nn]
    pos = [(nn+1)/2,(nn+1)/2]
    diam_m = 10.0
    radius_m = diam_m / 2.0
    pad_size = 1                              #pad_size = 2 makes aperture half the size of array.   ->  image pixels are wavel / ?? / D
    #lengthperpix = radius_m * 2 / nn        #Aperture goes to edge of array.  Do we want to add padding?
    #lengthperpix = radius_m * 4 / nn        #Aperture is half the size of the array
    #lengthperpix = radius_m * 8 / nn        #Aperture is quarter the size of the array
    lengthperpix = diam_m * pad_size / nn     
    radius = radius_m/lengthperpix
    rhole = radius*0/6.35
    #Calculate diffraction-limited values for OTF and PSF
    x = np.arange(dim[1]) + 1
    y = np.arange(dim[0]) + 1
    X, Y = np.meshgrid(x, y)            
    Xap = (X - pos[0]) * lengthperpix
    Yap = (Y - pos[1]) * lengthperpix
    Rap = np.sqrt( Xap**2 + Yap**2 )
    Tel_App = (Rap <= radius_m).astype(int)

    dact = 2 * radius_m / np.sqrt(numactuators)
    #print ("Actuator Spacing = {:6.4f} m".format(dact))



    G2 = np.zeros((field_points,nn,nn))
    D_total_s = np.zeros((nn,nn))                       #Atmosphere structure Function
    D_total_e = np.zeros((field_points,nn,nn))          #Residual Atmosphere Structure Function
    OTF_e = np.zeros((field_points,nn,nn))              #Residual OTF
    PSF_e = np.zeros((field_points,nn,nn))              #Residual PSF


    #Optical Transfer Function coordinates
    k_x = (x - pos[0] - 1) * lengthperpix    #units of m (Tok. uses 1/radians)
    k_y = (y - pos[0] - 1) * lengthperpix    #units of m  (variable k in Tok. 2004)
    k_X, k_Y = np.meshgrid(k_x, k_y)
    k_r = np.sqrt((k_X)**2 + (k_Y)**2)

    #Spatial Frequency coordinates
    f_x = (x - pos[0] - 1) / lengthperpix / nn   #units of m^-1
    f_y = (y - pos[0] - 1) / lengthperpix / nn   #units of m^-1 
    f_X, f_Y = np.meshgrid(f_x, f_y)
    f_r = np.sqrt((f_X)**2 + (f_Y)**2)

    #DM spatial filter
    fcutoff = 1/2/dact
    R = f_r < fcutoff


    ns = np.arange(num_guide_stars)
    ns_angle = ns /num_guide_stars * 2 * np.pi
    #define angles in arcminutes alpha = X , beta = Y
    alpha_arcmin = theta_0 * np.cos(ns_angle)
    beta_arcmin = theta_0 * np.sin(ns_angle) 
    arcminutes_to_radians = 1 / 3438
    alpha = alpha_arcmin * arcminutes_to_radians
    beta = beta_arcmin * arcminutes_to_radians
    weight = 1/ num_guide_stars * np.ones(num_guide_stars)      #equally weight all guide stars

    arcsec_pixel = wavel /pad_size /  (diam_m ) * 206265              
    #print(" spatial sampling is {:6.3f} m.  arcsec/pixel is {:6.4f}".format(lengthperpix,arcsec_pixel))
    #Create PSF and OTF from aperture
    PSF_diff,OTF_diff = U.AptoI(Tel_App)     #I= normalized  diff. PSF,  OTF_diff= diffraction limited OTF before normalization
    diff_fwhm_pixels,diff_alpha = U.fitMoffat(PSF_diff)
    diff_fwhm_arcsec = diff_fwhm_pixels * arcsec_pixel
    #print ("Diff FWHM = {:6.2f} pixels, FWHM = {:6.4f} arcsec., Diff. alpha = {:6.2f}".format(diff_fwhm_pixels,diff_fwhm_arcsec,diff_alpha))

    #setting range 1->2 is only median vaue
    D_total_s = np.zeros((nn,nn))                       #Atmosphere structure Function
    D_total_e = np.zeros((field_points,nn,nn))          #Residual Atmosphere Structure Function

    for layer in range (O.numlayers) : 
        #print("Atmospheric layer {:d}".format(layer))
        if layer < len(O.lowlayers) :
            J_layer = O.Jlow[layer,lp] / np.cos(ZA_rad)
            h = O.lowlayers[layer]
        else :
            J_layer = O.Jhigh[(layer - len(O.lowlayers)),hp] / np.cos(ZA_rad)
            h = O.highlayers[layer - len(O.lowlayers)]
        #print('J_layer {:d} is {:6.4e} m^(1/3) at a height {:6.0f} m'.format(layer, J_layer,h))
        #Calculate Aperture scaling and shift for each guide star
        #Not sure I actually use it in this form
        #h = 1                                   #Height of this particular layer
        #scale = h/H
        #shift_x = alpha_rad * h
        #shift_y = beta_rad * h 
        #layer_X = Xap * scale  + shift_x
        #layer_Y = Yap * scale + shift_y


        #******Set up Spatial filter in frequency domain ( |G(f)^2| )
        gamma = 1 - (h/H)           #scale if GS not at infinity
        A = 2 * jv(1,np.pi * f_r * radius_m*2  * h / H) / (np.pi * f_r * radius_m*2 * h / H )   #jv(1,..) - first order Bessel Function. eq. A6
        A[int((nn+1)/2),int((nn+1)/2)] = 1          #replace center pixel  (why is this needed?)

        field = np.zeros(field_points)
        for field_point in range(field_points) :
            sum_over_guide_stars = 0
            ii = np.mod(field_point, gridside)
            jj = np.floor_divide(field_point,gridside)
            
            cent = np.floor(gridside/2)
            alpha_fp_arcmin = (ii - cent) / cent * field_size / 2
            beta_fp_arcmin = (jj - cent) / cent * field_size / 2 
            alpha_fp = alpha_fp_arcmin * arcminutes_to_radians 
            beta_fp = beta_fp_arcmin * arcminutes_to_radians 
            for i in range (num_guide_stars) :
                dot_product = (f_X * (alpha[i] - alpha_fp)) + (f_Y * (beta[i]-beta_fp))
                #print(np.shape(dot_product))                                #_fp sets the field dependence
                sum_over_guide_stars += weight[i] * np.cos (2 * np.pi * h * dot_product )
            doublesum_over_guide_stars = 0
            for i in range (num_guide_stars) :
                for j in range (num_guide_stars) :
                    dot_product2 = f_X * (alpha[i]-alpha[j]) + f_Y * (beta[i] - beta[j])
                    doublesum_over_guide_stars += weight[i] * weight[j] * np.cos (2 * np.pi * h * dot_product2 )
            G2[field_point,:,:] = 1 - (2 * gamma  * R  * A * sum_over_guide_stars) + (gamma**2 * R**2 * doublesum_over_guide_stars)            #eq. A18 
        
        field_grid = field.reshape((gridside,gridside))
        #plt.imshow(field_grid)
        #plt.show()
        #r_0 = 0.15 

        #J_layer =  (wavel/ 2 /np.pi)**2 / 0.423 / (r_0/2) ** (5/3)           #definition after eq. 4.   Stray factor of two is included to match r_0 input


        ###########Construct Structure functions
        D_s = np.zeros((nn,nn))         #Atmosphere structure Function
        D_e = np.zeros((field_points,nn,nn))         #Residual Atmosphere Structure Function
        integrand2_e = np.zeros((field_points,nn,nn))   
        integrand_e = np.zeros((field_points,nn,nn))        
        dot_product = np.zeros((nn,nn,nn,nn))
        dfx = f_x[1] - f_x[0]           #steps in spatial frequency
        dfy = f_y[1] - f_y[0]           #steps in spatial frequency

        integrand2 = ((f_X**2 + f_Y**2) + (1/L_0**2))**(-11/6)          #second term in eq. 7
        for fp in range(field_points) :
            integrand2_e[fp,:,:] = ((f_X**2 + f_Y**2) + (1/L_0**2))**(-11/6)  * G2[fp,:,:]  #second term in eq. 7
        factor = 0.0229 * 0.423 * (2 * np.pi / wavel)**2 * J_layer      #terms outside integral for eq. 7 and eq. 6
        factor = factor * 3                                             #fudge factor to fit stats at visible wavelength         
        '''
        #Attempt to speed up computations in nested for loops
        outer1 = np.einsum('i,j->ij',f_X.ravel(),Xap.ravel())      #generalization of np.outer for 2D matrices
        dot_product1 = outer1.reshape((nn,nn,nn,nn))
        outer2 = np.einsum('i,j->ij',f_Y.ravel(),Yap.ravel())      #generalization of np.outer for 2D matrices
        dot_product2 = outer2.reshape((nn,nn,nn,nn))

        dot_product = dot_product1 + dot_product2
        '''

        #+ np.outer(f_Y,Yap) 
        #print(np.shape(f_X), np.shape(Xap), np.shape(dot_product))

        for i in range (nn) :                           #populate structure functions
            #if (i % 10 == 0) :
            #    print(layer,i,j)           
            for j in range (nn) :
                dot_product[i,j,:,:] = f_X * Xap[i,j] + f_Y * Yap[i,j]       

                integrand1 = dfx * dfy * (1 - np.cos(2 * np.pi * dot_product[i,j,:,:])) 
                integrand = integrand1 * integrand2
                integral =np.sum(integrand)
                D_s[i,j] = factor * integral
                for fp in range (field_points) :
                    integrand_e[fp,:,:] = integrand1 * integrand2_e[fp,:,:]
                    integral_e =np.sum(integrand_e[fp,:,:])
                    D_e[fp,i,j] = factor * integral_e


        D_total_s = D_s + D_total_s
        for fp in range (field_points) :
            D_total_e[fp,:,:] = D_e[fp,:,:] + D_total_e[fp,:,:]
        '''
        #DEBUG: Take a look at the Structure functions to check whether they look right
        plt.subplot(221)
        plt.imshow(f_X)
        plt.subplot(222)
        plt.imshow(f_Y)
        plt.subplot(223)
        plt.imshow(D_e)
        plt.subplot(224)
        plt.imshow(D_s)
        plt.show()
        '''

    #Original and residual OTF's
    OTF = OTF_diff * np.exp(-D_total_s / 2)         #eq. 1
    PSF = U.normFT(OTF)
    for fp in range(field_points) :
        OTF_e[fp,:,:] = OTF_diff * np.exp(-D_total_e[fp,:,:] / 2)
        PSF_e[fp,:,:] = U.normFT(OTF_e[fp,:,:])
    '''
    plt.subplot(121)
    plt.imshow(PSF)
    plt.subplot(122)
    plt.imshow(PSF_e)
    plt.show()
    '''

    open_fwhm_pixels,open_alpha = U.fitMoffat(PSF)
    open_fwhm_arcsec = open_fwhm_pixels * arcsec_pixel
    
    PSF_vector = np.zeros(field_points)
    for fp in range(field_points) :
        corrected_fwhm_pixels, corrected_alpha = U.fitMoffat(PSF_e[fp,:,:])
        corrected_fwhm_arcsec = corrected_fwhm_pixels * arcsec_pixel
        PSF_vector[fp] = corrected_fwhm_arcsec 
        '''
        print(" ")
        print("--------------------------------------------------")
        print(lp,hp,fp)
        print("Guide star diameter = {:6.2f} arcminutes, wavel = {:6.2f} um".format(gsdiam, wavel*1E6))
        print ("Original FWHM = {:6.2f}, Original alpha = {:6.2f}".format(open_fwhm_arcsec,open_alpha))
        print ("Corrected FWHM = {:6.2f}, Corrected alpha = {:6.2f}".format(corrected_fwhm_arcsec,corrected_alpha))
        print("--------------------------------------------------")
        '''
    FWHM_av = np.average(PSF_vector)
    FWHM_std = np.std(PSF_vector)
    FWHM_max = np.max(PSF_vector)
    FWHM_min = np.min(PSF_vector)
    #print("pixels = {:d} wavel = {:6.2f} num stars = {:d}".format(nn, wavel*1E6,num_guide_stars))
    #print("av = {:6.3f} std = {:6.3f} max = {:6.3f} min = {:6.3f}".format(FWHM_av,FWHM_std,FWHM_max,FWHM_min))
    PSF_grid = PSF_vector.reshape((gridside,gridside))
    fr = field_size /2 
    plt.imshow(PSF_grid * 1000, cmap = 'viridis_r', vmin = 200, vmax = 500, extent =[-fr,fr,-fr,fr])
    plt.colorbar(label = 'FWHM (mas)')
    filename = 'maps/{:.2f}um-{:.0f}arcmin.png'.format(wavel*1E6,gsdiam)
    plt.savefig(filename)
    #plt.show() 

    print("{:d}, {:.2f}, {:d}, {:d}, {:d},  {:.2f}, {:.2f}, {:.3f}, {:d}, {:.1f}, {:.0f}, {:d},  {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}"  
          .format(nn,field_size,field_points, lp,hp,L_0,elevation, wavel*1E6,num_guide_stars,gsdiam,H,numactuators,
                  open_fwhm_arcsec, FWHM_av, FWHM_std, FWHM_max,FWHM_min))
    
    file = open(outname,'a')
    file.write("{:d}, {:.2f}, {:d}, {:d}, {:d},  {:.2f}, {:.2f}, {:.3f}, {:d}, {:.1f}, {:.0f}, {:d},  {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f} \n"  
          .format(nn,field_size,field_points, lp,hp,L_0,elevation, wavel*1E6,num_guide_stars,gsdiam,H,numactuators,
                  open_fwhm_arcsec, FWHM_av, FWHM_std, FWHM_max,FWHM_min))
    file.close()
    return nn,field_size,field_points, lp,hp,L_0,elevation, wavel,num_guide_stars,gsdiam,H,numactuators, open_fwhm_arcsec, FWHM_av, FWHM_std, FWHM_max,FWHM_min
