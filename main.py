import numpy as np
from matplotlib import pyplot as plt

def The_Guss_func(X,mf):
    return np.exp(-1/2*((X-mf)/10)**2)

def Phi_func(X,Fl:list):
    Phi_basic = []
    for mf in Fl:
        Phi_basic.append(The_Guss_func(X, mf))
    # np.array(Phi_basic)
    Phi = np.r_[Phi_basic]
    return Phi/np.sum(Phi, axis=0)

sample = np.linspace(0, 100, 10000)
Fl_1 = [20, 35, 50, 65, 80]
Fl_2 = [20, 25, 50, 62, 80]

fig = plt.figure(figsize=[12,5])
plt.subplot(1,2,1)
plt.plot(sample,Phi_func(sample,Fl_1).T)
plt.xlim([0,100])
plt.ylim([0,1])
plt.subplot(1,2,2)
plt.plot(sample,Phi_func(sample,Fl_2).T)
plt.xlim([0,100])
plt.ylim([0,1])
plt.show()