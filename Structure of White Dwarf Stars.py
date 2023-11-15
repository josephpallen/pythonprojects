# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 21:29:30 2023

@author: joeal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#defining the constants

G=6.67259e-11 #gravitational constant
m_p=1.6726231e-27 #mass of a proton (kg)
m_e=9.1093897e-31 #mass of an electron (kg)
c=2.99792458e8 #speed of light (kg)
solar_mass=1.98e30 #unit of solar mass
solar_radius=6.95e8 #unit of solar radius
h_bar=1.05457266e-34 #planck's constant


#carbon Y_e=0.5, iron Y_e=0.46

def rho_0(Y_e):
    return (m_p*(m_e**3)*(c**3))/(3*(np.pi**2)*(h_bar**3)*Y_e)

def R_0(Y_e):
    return np.sqrt((3*Y_e*m_e*c**2)/(4*np.pi*rho_0(Y_e)*m_p*G))

def M_0(Y_e):
    return (4/3)*np.pi*(R_0(Y_e)**3)*(rho_0(Y_e))




#differential equations for the relativistic case
def rel_diff_eq(x,b):
    y=b[0]
    z=b[1]
    if z<= 0:
        return np.nan
    A=(4*np.pi*G*(R_0(0.5)**2)*rho_0(0.5)*m_p)/(3*0.5*m_e*c**2) #A is absorbed when we redefine dimensionless variables
    gamma=(z**(2/3))/(3*(1+z**(2/3))**(1/2))
    dzdx=-(A*y*z)/(gamma*x**2) 
    dydx=(3*z*x**2)
    return [dydx,dzdx] #setting up the differential equation as an array

def non_rel_diff_eq(x,b):
    y=b[0]
    z=b[1]
    if z<= 0:
        return np.nan
    B=(4*np.pi*G*(R_0(0.5)**2)*rho_0(0.5)*m_p)/(0.5*m_e*c**2) #B is absorbed when we redefine dimensionless variables
    dzdx=-(B*y*z**(1/3))/x**2
    dydx=(3*z*x**2)
    return [dydx,dzdx] #setting up the differential equation as an array

def ext_rel_diff_eq(x,b):
    y=b[0]
    z=b[1]
    if z<= 0:
        return np.nan
    B=(4*np.pi*G*(R_0(0.5)**2)*rho_0(0.5)*m_p)/(0.5*m_e*c**2) #B is absorbed when we redefine dimensionless variables
    dzdx=-(B*y*z**(2/3))/x**2
    dydx=(3*z*x**2)
    return [dydx,dzdx] #setting up the differential equation as an array





#solving diff_eq using RK45
start_val=[1/M_0(0.5),1] # initial conditions z(0)=1, y(0)=0
x=np.linspace(1/R_0(0.5),5,5000) #x is set such that that we avoid a zero division error
s=solve_ivp(rel_diff_eq,[x[0],x[-1]],start_val,t_eval=x)


#Plotting
x=s.t
z,y=s.y

#defining the solar radii
R_carbon=(x*R_0(0.5))/solar_radius
R_iron=(x*R_0(0.46))/solar_radius


# plotting rho/rho_0 against r/R_0
plt.plot(x,y)
plt.grid()
plt.xlabel("$r/R_0$")
plt.ylabel(r'$\rho$/$\rho_0$')

plt.figure()

# plotting m/M_0 against r/R_0
plt.plot(x,z)
plt.grid()
plt.xlabel("$r/R_0$")
plt.ylabel("$m/M_0$")

plt.figure()

#plotting rho/rho_0 against solar radius
plt.plot(R_carbon,y,label='Carbon-12')
plt.plot(R_iron,y,label='Iron-56')
plt.ylim(0,1)
plt.xlim(0,0.014)
plt.legend()
plt.grid()
plt.xlabel(r'$Radius (R_\odot)$')
plt.ylabel(r'$Density (\rho$/$\rho_0)$')
plt.show()

plt.figure()



time_int=np.linspace(0.0001,10000,num=1000000)

M_vals=[0,0]
R_vals=[0,0]

M_vals2=[0,0]
R_vals2=[0,0]

M_vals3=[0,0]
R_vals3=[0,0]
n = 0
for i in [0.5,0.46]: #looping over both values of Y_e
    M_vals[n]=[]
    R_vals[n]=[]
    
    M_vals2[n]=[]
    R_vals2[n]=[]
    
    M_vals3[n]=[]
    R_vals3[n]=[]
    rho_c_vals=np.logspace(6,15,num=250) #integrating over a range of central densities, 10^6 - 10^15
    for rho_c in rho_c_vals:
        start_val=[0.0000001,rho_c/rho_0(i)]
        sol=solve_ivp(rel_diff_eq,[time_int[0],time_int[-1]],start_val,t_eval=time_int)
        sol2=solve_ivp(non_rel_diff_eq,[time_int[0],time_int[-1]],start_val,t_eval=time_int)
        sol3=solve_ivp(ext_rel_diff_eq,[time_int[0],time_int[-1]],start_val,t_eval=time_int)
        ### Reintroduce Dimensionality
        r=R_0(i)*sol.t 
        m=M_0(i)*sol.y[0]
        
        r2=R_0(i)*sol2.t 
        m2=M_0(i)*sol2.y[0]
        
        r3=R_0(i)*sol3.t 
        m3=M_0(i)*sol3.y[0]
        plt.plot(r/solar_radius,m/solar_mass)
        ### find the limit:
        R_final=r[-1]/solar_radius #converting to solar mass/radius
        M_final=m[-1]/solar_mass
        
        R_final2=r2[-1]/solar_radius
        M_final2=m2[-1]/solar_mass
        
        R_final3=r3[-1]/solar_radius
        M_final3=m3[-1]/solar_mass
        #print(R_wd, M_wd)
        M_vals[n].append(M_final) #appending the final value of mass radius relationship for each rho_c
        R_vals[n].append(R_final)
        
        M_vals2[n].append(M_final2) 
        R_vals2[n].append(R_final2)
        
        M_vals3[n].append(M_final3) 
        R_vals3[n].append(R_final3)
    
    plt.title('Constructing the Mass against Radius Figure for C-12 and Fe-56')
    plt.grid()  
    plt.xlabel(r'$Radius (R_\odot)$')
    plt.ylabel(r'$Mass (M_\odot)$')
    
    plt.show()
    n = n +1
    
print('Maximum mass of a Carbon-12 white dwarf is',M_vals[0][-1],'solar masses')
print('=======================================================================')
print('Maximum mass of an Iron-56 white dwarf is',M_vals[1][-1],'solar masses')   

plt.figure()    
plt.plot(M_vals[0],R_vals[0],label='Relativistic Carbon-12')
plt.plot(M_vals[1],R_vals[1],label='Relativistic Iron-56')

plt.plot(M_vals2[0],R_vals2[0],label='Non-Relativistic Carbon-12')
plt.plot(M_vals2[1],R_vals2[1],label='Non-Relativistic Iron-56')

plt.plot(M_vals3[0],R_vals3[0],label='Extremely-Relativistic Carbon-12')
plt.plot(M_vals3[1],R_vals3[1],label='Extremely-Relativistic Iron-56')

plt.xlim(0.0,1.5)
plt.ylim(0.00,0.06)
plt.plot([1.404,1.4040001],[0,0.03],label='Chandrasekhar Limit',linestyle='dashed') #plotting the Chandrasekhar limit
plt.errorbar(1.053,0.0074,yerr=6e-4,xerr=0.028,label='Sirius B',marker='o') #plotting the known white dwarves
plt.errorbar(0.48,0.0124,yerr=5e-4,xerr=0.02,label='40 Eri B',marker='o')
plt.errorbar(0.5,0.0115,xerr=0.05,yerr=1.2e-3,label='Stein 2051',marker='o')

plt.errorbar(0.534,0.0140,yerr=0.0007,xerr=0.009,label='SDSS J0024+1745',marker='^',color='black') #plotting the SDSS data
plt.errorbar(0.415,0.0177,yerr=0.0002,xerr=0.004,label='SDSS J1028+0931',marker='^',color='black')
plt.errorbar(0.436,0.0157,xerr=0.002,yerr=0.0004,label='SDSS J1307+2156',marker='^',color='black')
plt.title('White Dwarf Mass-Radius Relationship')
plt.ylabel(r'$Radius (R_\odot)$')
plt.xlabel(r'$Mass (M_\odot)$')
plt.grid()
plt.legend()
plt.show()    
    


