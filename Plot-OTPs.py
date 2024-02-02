import numpy as np
import matplotlib.pyplot as plt
import otps as O


lp = 1
hp = 1

num_layers = len(O.lowlayers) + len(O.highlayers)
height_chun = np.zeros(num_layers)
J_chun =  np.zeros(num_layers)
for layer in range (num_layers) : 
    #print("Atmospheric layer {:d}".format(layer))
    if layer < len(O.lowlayers) :
        J_chun[layer] = O.Jlow[layer,lp] 
        height_chun[layer]= O.lowlayers[layer]
    else :
        J_chun[layer] = O.Jhigh[(layer - len(O.lowlayers)),hp]
        height_chun[layer] = O.highlayers[layer - len(O.lowlayers)]

J_d_sum = np.sum(O.J_dekany[:,4])
J_c_sum = np.sum(J_chun)
lambda_ref = 0.5E-6


r_0_d = ((lambda_ref / 2 / np.pi)**2 / 0.423  / J_d_sum) **(3/5)
r_0_c = ((lambda_ref / 2 / np.pi)**2 / 0.423  / J_c_sum) ** (3/5)

print ("Chun model r_0 = {:.3f} cm,  Dekany Model r_0 = {:.3f} cm ".format(r_0_c*100,r_0_d*100))


plt.loglog(height_chun,J_chun,'-*')
plt.loglog(O.layers_dekany,O.J_dekany[:,0],'-.',label ='75/75')
#plt.loglog(O.layers_dekany,O.J_dekany[:,1],'-.',label ='75/50')
#plt.loglog(O.layers_dekany,O.J_dekany[:,2],'-.',label ='75/25')
#plt.loglog(O.layers_dekany,O.J_dekany[:,3],'-.',label ='50/75')
plt.loglog(O.layers_dekany,O.J_dekany[:,4],'-.',label ='50/50')
#plt.loglog(O.layers_dekany,O.J_dekany[:,5],'-.',label ='50/25')
#plt.loglog(O.layers_dekany,O.J_dekany[:,6],'-.',label ='25/75')
#plt.loglog(O.layers_dekany,O.J_dekany[:,7],'-.',label ='25/50')
plt.loglog(O.layers_dekany,O.J_dekany[:,8],'-.',label ='25/25')

plt.xlabel('Height (m)')
plt.ylabel('Cn^2')
plt.legend()
plt.savefig('plots/OTPs.png')
#plt.show()