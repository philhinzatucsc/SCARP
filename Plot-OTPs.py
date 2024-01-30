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



plt.loglog(height_chun,J_chun,'-*')
plt.loglog(O.layers_dekany,O.J_dekany[:,4],'-.')

plt.xlabel('Height (m)')
plt.ylabel('Cn^2')
#plt.legend()
plt.savefig('plots/OTPs.png')
#plt.show()