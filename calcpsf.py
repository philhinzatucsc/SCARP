import numpy as np
import matplotlib.pyplot as plt
import utils as U
import otps as O
from scipy.special import jv
from astropy.io import fits
from scipy.interpolate import griddata


def CalcPSF(field_size, field_points, lp, hp, L_0, elevation, wavel, num_guide_stars, gsdiam, H, numactuators, outname) :
    '''
    Top level function to execute a calculation
    '''

    #Setup spatial sampling and angular scales
    n_pos = 201                 #201 with pad=1 at 0.5 um is +/-1"
    pos_cen = (n_pos+1)/2
    n_ang = 201
    ang_cen = (n_ang+1)/2
    pad_size = 1
    n_struct = 21
    struct_cen = (n_struct+1)/2

    print ('number of resolution elements = {:d}, pad size = {:0.1f}, wavel = {:.3f} um'.format(n_pos,pad_size,wavel*1E6))

    #Setup Guide star asterism
    theta_0 = gsdiam/2                #set science field as guide star size
    ZA = 90 - elevation             #Zenith Angle
    ZA_rad = ZA * np.pi / 180.0


    gridside = int(np.sqrt(field_points))
                

    diam_m = 10.0
    radius_m = diam_m / 2.0
    
    lengthperpix = diam_m * pad_size / n_pos     
    radius = radius_m/lengthperpix
    rhole = radius*0/6.35
    #Calculate diffraction-limited values for OTF and PSF
    x = np.arange(n_pos) + 1
    y = np.arange(n_pos) + 1
    X, Y = np.meshgrid(x, y)            
    Xap = (X - pos_cen) * lengthperpix
    Yap = (Y - pos_cen) * lengthperpix
    Rap = np.sqrt( Xap**2 + Yap**2 )
    Tel_App = (Rap <= radius_m).astype(int)

    #plt.imshow(Tel_App)
    #plt.show()

    dact = 2 * radius_m / np.sqrt(numactuators)
    #print ("Actuator Spacing = {:6.4f} m".format(dact))

    G2 = np.zeros((field_points,n_ang,n_ang))
    D_total_s = np.zeros((n_struct,n_struct))                       #Atmosphere structure Function
    D_total_e = np.zeros((field_points,n_struct,n_struct))          #Residual Atmosphere Structure Function
    OTF_e = np.zeros((field_points,n_pos,n_pos))              #Residual OTF
    PSF_e = np.zeros((field_points,n_pos,n_pos))              #Residual PSF
    PSF_fits = np.zeros((field_points+2,n_pos,n_pos))         #PSF cube for FITS , field points + open loop and diffraction

    #Structure Functions coordinates
    max_struct_value = diam_m            #I think we want to calculate all values seen by telescope here
    lengthperstructpix = 1 / n_struct * max_struct_value * 2
    x_s = np.arange(n_struct) + 1
    y_s = np.arange(n_struct) + 1
    X_s, Y_s = np.meshgrid(x_s, y_s)            
    X_struct = (X_s - struct_cen) * lengthperstructpix
    Y_struct = (Y_s - struct_cen) * lengthperstructpix

    R_struct = np.sqrt(X_struct**2 + Y_struct**2)

    #Spatial Frequency coordinates
    max_freq_value = 4            #Not sure how to set this.  At 4, 4000 actuators are still well-modeled.
    lengthperfreqpix = 1 / n_ang * max_freq_value * 2
    tempf_x = np.arange(n_ang) + 1
    tempf_y = np.arange(n_ang) + 1
    f_x = (tempf_x - ang_cen - 1) * lengthperfreqpix   #units of m^-1
    f_y = (tempf_y - ang_cen - 1) * lengthperfreqpix    #units of m^-1          
    f_X, f_Y = np.meshgrid(f_x, f_y)
    f_r = np.sqrt((f_X)**2 + (f_Y)**2)
    max_f = np.max(f_r) / np.sqrt(2)
    print('Maximum frequency along one axis is {:.2f}'.format(max_f))

    #DM spatial filter
    fcutoff = 1/2/dact
    R = f_r < fcutoff

    #plt.imshow(R)
    #plt.show()

    ns = np.arange(num_guide_stars)
    ns_angle = ns /num_guide_stars * 2 * np.pi
    #define angles in arcminutes alpha = X , beta = Y
    alpha_arcmin = theta_0 * np.cos(ns_angle)
    beta_arcmin = theta_0 * np.sin(ns_angle) 
    arcminutes_to_radians = 1 / 3438
    alpha = alpha_arcmin * arcminutes_to_radians
    beta = beta_arcmin * arcminutes_to_radians
    weight = 1/ num_guide_stars * np.ones(num_guide_stars)      #equally weight all guide stars

    arcsec_pixel = wavel / pad_size /  (diam_m ) * 206265              
    print(" spatial sampling is {:6.3f} m.  arcsec/pixel is {:6.4f}".format(lengthperpix,arcsec_pixel))
    #Create PSF and OTF from aperture
    PSF_diff,OTF_diff = U.AptoI(Tel_App)     #I= normalized  diff. PSF,  OTF_diff= diffraction limited OTF before normalization
    diff_fwhm_pixels,diff_alpha = U.fitMoffat(PSF_diff)
    diff_fwhm_arcsec = diff_fwhm_pixels * arcsec_pixel
    print ("Diff FWHM = {:6.2f} pixels, FWHM = {:6.4f} arcsec., Diff. alpha = {:6.2f}".format(diff_fwhm_pixels,diff_fwhm_arcsec,diff_alpha))

    D_total_s = np.zeros((n_struct,n_struct))                       #Total Atmosphere structure Function
    D_total_e = np.zeros((field_points,n_struct,n_struct))          #Total Residual Atmosphere Structure Function

    '''   Chun model '''
    num_layers = len(O.lowlayers) + len(O.highlayers) 
    for layer in range (num_layers) : 
        print("Atmospheric layer {:d}".format(layer))
        if layer < len(O.lowlayers) :
            J_layer = O.Jlow[layer,lp] / np.cos(ZA_rad) 
            h = O.lowlayers[layer] / np.cos(ZA_rad)
        else :
            J_layer = O.Jhigh[(layer - len(O.lowlayers)),hp] / np.cos(ZA_rad) 
            h = (O.highlayers[layer - len(O.lowlayers)] ) / np.cos(ZA_rad)
        #'''
            
        ''' Dekany model  
    num_layers = len(O.layers_dekany)
    column = (2-hp) + (2-lp)*3
    for layer in range (num_layers) :
        print("Atmospheric layer {:d}".format(layer))
        J_layer = O.J_dekany[layer,column] / np.cos(ZA_rad)
        h =  O.layers_dekany[layer] / np.cos(ZA_rad)
        '''

        Calc_GLAO_correction = False
    
        if (Calc_GLAO_correction == True) :
            #******Set up Spatial filter in frequency domain ( |G(f)^2| )
            gamma = 1 - (h/H)           #scale if GS not at infinity
            A = 2 * jv(1,np.pi * f_r * radius_m*2  * h / H) / (np.pi * f_r * radius_m*2 * h / H )   #jv(1,..) - first order Bessel Function. eq. A6
            A[int((n_ang+1)/2),int((n_ang+1)/2)] = 1          #replace center pixel  (why is this needed?)

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
            

        ###########Construct Structure functions
        D_s = np.zeros((n_struct,n_struct))                         #per layer Atmosphere structure Function
        D_e = np.zeros((field_points,n_struct,n_struct))            #per layer Residual Atmosphere Structure Function
        integrand2_e = np.zeros((field_points,n_struct,n_struct))   
        integrand_e = np.zeros((field_points,n_struct,n_struct))        
        dot_product = np.zeros((n_struct,n_struct,n_ang,n_ang))
        dfx = f_x[1] - f_x[0]           #steps in spatial frequency
        dfy = f_y[1] - f_y[0]           #steps in spatial frequency
        #terms outside integral for eq. 7 and eq. 6
        factor =  6.88 / 2   *  0.0229 * 0.423 * (2 * np.pi / wavel)**2 * J_layer   
        integrand2 = ((f_X**2 + f_Y**2) + (1/L_0**2))**(-11/6)          #second term in eq. 7
        if (Calc_GLAO_correction == True) :
            for fp in range(field_points) :
                integrand2_e[fp,:,:] = ((f_X**2 + f_Y**2) + (1/L_0**2))**(-11/6)  * G2[fp,:,:]  #second term in eq. 7
        for i in range (n_struct) :                           #populate structure functions
            #if (i % 10 == 0) :
            #    print(layer,i,j)           
            for j in range (n_struct) :
                dot_product[i,j,:,:] = f_X * X_struct[i,j] + f_Y * Y_struct[i,j]       
                integrand1 = dfx * dfy * (1 - np.cos(2 * np.pi * dot_product[i,j,:,:])) 
                integrand = integrand1 * integrand2
                integral =np.sum(integrand)
                D_s[i,j] = factor * integral
                if (Calc_GLAO_correction == True) :
                    for fp in range (field_points) :
                        integrand_e[fp,:,:] = integrand1 * integrand2_e[fp,:,:]
                        integral_e =np.sum(integrand_e[fp,:,:])
                        D_e[fp,i,j] = factor * integral_e
        D_total_s = D_s + D_total_s
        if (Calc_GLAO_correction == True) :
            for fp in range (field_points) :
                D_total_e[fp,:,:] = D_e[fp,:,:] + D_total_e[fp,:,:]

    # Create a finer grid
    finer_x = np.linspace(np.min(X_struct), np.max(X_struct), n_ang)
    finer_y = np.linspace(np.min(X_struct), np.max(X_struct), n_ang)

    # Create meshgrid for the finer grid
    finer_x_mesh, finer_y_mesh = np.meshgrid(finer_x, finer_y)

    # Flatten the original array and corresponding coordinates
    points = np.column_stack((X_struct.flatten(), Y_struct.flatten()))
    values = D_total_s.flatten()

    # Interpolate onto the finer grid using scipy.interpolate.griddata
    D_total_s_fine = griddata(points, values, (finer_x_mesh, finer_y_mesh), method='linear')


    '''
    plt.subplot(131)
    plt.imshow(np.real(OTF_diff))
    plt.subplot(132)
    plt.imshow(D_total_s)
    plt.subplot(133)
    plt.imshow(D_total_s_fine)
    plt.show()
    '''

    #Original and residual OTF's
    OTF = OTF_diff * np.exp(-D_total_s_fine / 2)         #eq. 1
    PSF = U.normFT(OTF)

    '''
    PSF_compare = U.normFT(OTF_diff)
    plt.subplot(121)
    plt.imshow(PSF_diff)
    plt.subplot(122)
    plt.imshow(PSF_compare)
    plt.show()
    '''
    rr = np.arange(n_pos) * lengthperpix
    r_0 = 0.168
    normfactor =   r_0**(5/3)
    D_theory = 6.88 * (rr/r_0)**(5/3) 
    plt.subplot(121)
    plt.imshow(D_total_s)
    plt.subplot(122)
    plt.plot(R_struct,D_total_s,'.')
    plt.plot(rr,D_theory)
    plt.xlim((0,10))
    plt.ylim((0,10000))
    plt.show()

    if (Calc_GLAO_correction == True) :
        for fp in range(field_points) :
            OTF_e[fp,:,:] = OTF_diff * np.exp(-D_total_e[fp,:,:] / 2)
            PSF_e[fp,:,:] = U.normFT(OTF_e[fp,:,:])
            PSF_fits[fp,:,:] = PSF_e[fp,:,:]
        PSF_fits[field_points,:,:] = PSF                        #add open loop to end of cube for FITS
        PSF_fits[field_points+1,:,:] = PSF_diff                 #add open loop to end of cube for FITS
    else :
        PSF_fits[field_points,:,:] = PSF                        #add open loop to end of cube for FITS
        PSF_fits[field_points+1,:,:] = PSF_diff                 #add open loop to end of cube for FITS
    
    ############ Create FITS file
    hdr = fits.Header()
    hdr['CDELT'] = arcsec_pixel
    hdr['COMMENT'] = "Cube is grid of field points over field size"
    hdr['COMMENT'] = "Last frame of cube is open loop"
    hdr['FIELDSZE'] = field_size
    hdr['FIELDPTS'] = field_points
    hdr['LOW_PERC'] = lp
    hdr['HI_PERC'] = hp
    hdr['L_0'] = L_0
    hdr['ELEV'] = elevation
    hdr['WAVEL'] = wavel*1E6
    hdr['N_GS'] = num_guide_stars
    hdr['GS_DIAM'] = gsdiam
    hdr['H_GS'] = H
    hdr['NUM_ACT'] = numactuators
    fitsoutname = filename = 'fits/{:.2f}um-{:.0f}arcmin-{:.0f}km-{:d}_GS-{:d}_act.fits'.format(wavel*1E6,gsdiam,H,num_guide_stars,numactuators)
    hdu = fits.PrimaryHDU(PSF_fits,header=hdr)
    hdu.writeto(fitsoutname, overwrite=True)

    

    open_fwhm_pixels,open_alpha = U.fitMoffat(PSF)
    open_fwhm_arcsec = open_fwhm_pixels * arcsec_pixel

    open_EnsqE50_pixels = U.calc_ensq_e(PSF,0.5)
    open_EnsqE50_arcsec = open_EnsqE50_pixels * arcsec_pixel
    
    #print("Made it here")
    if (Calc_GLAO_correction == True) :
        PSF_vector = np.zeros(field_points)
        alpha_vector = np.zeros(field_points)
        EsqE50_vector = np.zeros(field_points)
        for fp in range(field_points) :
            corrected_fwhm_pixels, corrected_alpha = U.fitMoffat(PSF_e[fp,:,:])
            corrected_fwhm_arcsec = corrected_fwhm_pixels * arcsec_pixel
            PSF_vector[fp] = corrected_fwhm_arcsec 
            alpha_vector[fp] = corrected_alpha

            corr_EnsqE50_pixels = U.calc_ensq_e(PSF_e[fp,:,:],0.5)
            corr_EnsqE50_arcsec = corr_EnsqE50_pixels * arcsec_pixel
            EsqE50_vector[fp] = corr_EnsqE50_arcsec
            '''
            print(" ")
            print("--------------------------------------------------")
            print(lp,hp,fp)
            print("Guide star diameter = {:6.2f} arcminutes, wavel = {:6.2f} um".format(gsdiam, wavel*1E6))
            print ("Original FWHM = {:6.2f}, Original alpha = {:6.2f}".format(open_fwhm_arcsec,open_alpha))
            print ("Corrected FWHM = {:6.2f}, Corrected alpha = {:6.2f}".format(corrected_fwhm_arcsec,corrected_alpha))
            print ("Original EnsqE50 = {:6.3f}, Corrected_ensq50 = {:6.2f}".format(open_EnsqE50_arcsec,corr_EnsqE50_arcsec))
            print("--------------------------------------------------")
            '''
        
        FWHM_av = np.average(PSF_vector)
        FWHM_std = np.std(PSF_vector)
        FWHM_max = np.max(PSF_vector)
        FWHM_min = np.min(PSF_vector)
        a_av = np.average(alpha_vector)
        a_std = np.std(alpha_vector)
        a_max = np.max(alpha_vector)
        a_min = np.min(alpha_vector)
        EE50_av = np.average(EsqE50_vector)
        EE50_std = np.std(EsqE50_vector)
        EE50_max = np.max(EsqE50_vector)
        EE50_min = np.min(EsqE50_vector)

        #print("pixels = {:d} wavel = {:6.2f} num stars = {:d}".format(nn, wavel*1E6,num_guide_stars))
        #print("av = {:6.3f} std = {:6.3f} max = {:6.3f} min = {:6.3f}".format(FWHM_av,FWHM_std,FWHM_max,FWHM_min))
        PSF_grid = PSF_vector.reshape((gridside,gridside))
        fr = field_size /2 
        plt.figure(1,figsize=(12,12))
        plt.imshow(PSF_grid * 1000, cmap = 'viridis_r', extent =[-fr,fr,-fr,fr])
        plt.colorbar(label = 'FWHM (mas)')
        mapoutname = filename = 'maps/{:.2f}um-{:.0f}arcmin-{:.0f}km-{:d}_GS-{:d}_act.png'.format(wavel*1E6,gsdiam,H,num_guide_stars,numactuators)
        hdu = fits.PrimaryHDU(PSF_e,header=hdr)
        #filename = 'maps/{:.2f}um-{:.0f}arcmin.png'.format(wavel*1E6,gsdiam)
        plt.savefig(mapoutname)
        #plt.show() 

        print("{:d}, {:.1f}, {:.2f}, {:d}, {:d}, {:d},  {:.2f}, {:.2f}, {:.3f}, {:d}, {:.1f}, {:.0f}, {:d},  {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}"  
            .format(n_pos,pad_size,field_size,field_points, lp,hp,L_0,elevation, wavel*1E6,num_guide_stars,gsdiam,H,numactuators,
                    open_fwhm_arcsec, FWHM_av, FWHM_std, FWHM_max,FWHM_min))
        
        file = open(outname,'a')
        file.write("{:d}, {:.1f}, {:.2f}, {:d}, {:d}, {:d},  {:.2f}, {:.2f}, {:.3f}, {:d}, {:.1f}, {:.0f}, {:d},  {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f},  {:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}\n"  
            .format(n_pos,pad_size,field_size,field_points, lp,hp,L_0,elevation, wavel*1E6,num_guide_stars,gsdiam,H,numactuators,
                    open_fwhm_arcsec, FWHM_av, FWHM_std, FWHM_max,FWHM_min,
                    open_alpha, a_av, a_std, a_max,a_min,
                    open_EnsqE50_arcsec, EE50_av, EE50_std, EE50_max,EE50_min ))
        file.close()
        return n_pos,pad_size,field_size,field_points, lp,hp,L_0,elevation, wavel,num_guide_stars,gsdiam,H,numactuators, open_fwhm_arcsec, FWHM_av, FWHM_std, FWHM_max,FWHM_min
    else :
        print("{:d}, {:.1f}, {:.2f}, {:d}, {:d}, {:d},  {:.2f}, {:.2f}, {:.3f}, {:d}, {:.1f}, {:.0f}, {:d},  {:.3f}\n"  
            .format(n_pos,pad_size,field_size,field_points, lp,hp,L_0,elevation, wavel*1E6,num_guide_stars,gsdiam,H,numactuators,
                    open_fwhm_arcsec ))
        file = open(outname,'a')
        file.write("{:d}, {:.1f}, {:.2f}, {:d}, {:d}, {:d},  {:.2f}, {:.2f}, {:.3f}, {:d}, {:.1f}, {:.0f}, {:d},  {:.3f}\n"  
            .format(n_pos,pad_size,field_size,field_points, lp,hp,L_0,elevation, wavel*1E6,num_guide_stars,gsdiam,H,numactuators,
                    open_fwhm_arcsec ))
        return n_pos,pad_size,field_size,field_points, lp,hp,L_0,elevation, wavel,num_guide_stars,gsdiam,H,numactuators, open_fwhm_arcsec
        
