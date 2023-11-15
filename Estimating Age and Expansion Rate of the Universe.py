# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 22:15:43 2021

@author: Joseph
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt

#STEP 1


def chisq(y, sig_y, y_m): #defining the chi-squared fucntion to be used throughout this script
    """takes model, data and error vectors and calculates the chi2"""
    chi2 = np.sum(((y-y_m)**2.0)/(sig_y**2.0))
    return chi2

def chisqxy(y, sig_y,sig_x, y_m,m): #defining the chi-squared function using x and y used throughout this script
    chi_2 = np.sum(((y-y_m)**2.0)/((sig_y**2.0)+((m**2.0)*sig_x**2.0)))
    return chi_2

#Loading in data for each of the 10 galaxies
galaxy = np.loadtxt('MW_Cepheids.dat',\
                    unpack=True, \
                    usecols=(0), \
                    dtype=str)

parallax, err_parallax, period, m, A, err_A  = np.loadtxt('MW_Cepheids.dat',\
                                                          unpack=True, \
                                                          usecols=(1,2,3,4,5,6), \
                                                          dtype=float)
#Calculating variables from data
d_pc=1000/parallax
logd_pc=np.log10(d_pc)
logperiod=np.log10(period)
abs_mag=m-5*logd_pc-A+5

#Degrees of freedom
dof=len(d_pc)-2

#Propagating errors for STEP 1
sig_d_pc=(1000*err_parallax)/(parallax**2)
sig_5logd_pc=(5*sig_d_pc)/(np.log(10)*d_pc)
sig_abs_mag=np.sqrt((sig_5logd_pc**2)+(err_A**2))

#Defining function for optimal curve fitting
def straight_line(slope, x, intercept):
    """calculates the model"""
    line = slope*x+intercept
    return line

#Starting the curve fit (Using LM Algorithm)
start_slope = 10 #this is my starting slope for the LM algorithm
start_intercept = -10 #this is my starting intercept for the LM algorithm
best_pars, covar = opt.curve_fit(f=straight_line, xdata=logperiod, ydata=abs_mag, 
                                       sigma=sig_abs_mag, p0=(start_slope, start_intercept),
                                       absolute_sigma=True)

#let's quickly work out the min chisq
best_slope = best_pars[0]
best_intercept = best_pars[1]
y_best = straight_line(best_slope, logperiod, best_intercept)
best_chi2 = chisq(abs_mag,sig_abs_mag,y_best)

#Get the 1 sigma errors in alpha and beta, (One variable at a time)
slope_err = np.sqrt(covar[0, 0])
intercept_err = np.sqrt(covar[1, 1])

#Convert covariance matrix into a correlation matrix

#next, we convert (co)variances to correlations
corr_ss = covar[0, 0]/(slope_err*slope_err)
corr_ii = covar[1, 1]/(intercept_err*intercept_err)
corr_si = covar[0, 1]/(slope_err*intercept_err)
corr_is = covar[1, 0]/(intercept_err*slope_err)

xzero=0.845115 #An x-shift is needed to combat the anti-correlation between the diagonal 1 sigma parameter errors.
xshift=logperiod-xzero

def straight_line_new(xshift, slope, intercept): #defining a new straight line model use the newly shifted logP values
    """calculates the model"""
    line_new = slope*xshift+intercept
    return line_new

#Starting the curve fit (Using LM Algorithm)
start_slope = 10 #this is my starting slope for the LM algorithm
start_intercept = -10 #this is my starting intercept for the LM algorithm
best_pars_new, covar_new = opt.curve_fit(f=straight_line_new, xdata=xshift, ydata=abs_mag, 
                                       sigma=sig_abs_mag, p0=(start_slope, start_intercept),
                                       absolute_sigma=True)

#let's quickly work out the min chisq
best_slope_new = best_pars_new[0]
best_intercept_new = best_pars_new[1]
y_best_new = straight_line_new(best_slope_new, xshift, best_intercept_new)
best_chi2_new = chisq(abs_mag,sig_abs_mag,y_best_new)

#Get the errors in alpha and beta, (One variable at a time)
slope_err_new = np.sqrt(covar_new[0, 0])
intercept_err_new = np.sqrt(covar_new[1, 1])

#Convert covariance matrix into a correlation matrix

#next, we convert (co)variances to correlations
corr_ss_new = covar_new[0, 0]/(slope_err_new*slope_err_new)
corr_ii_new = covar_new[1, 1]/(intercept_err_new*intercept_err_new)
corr_si_new = covar_new[0, 1]/(slope_err_new*intercept_err_new)
corr_is_new = covar_new[1, 0]/(intercept_err_new*slope_err_new)

#printing the values of best fit alpha and beta as well as their errors.
#Corresponding chi-squared and reduced chi-squared are printed as well as the correlation matrixs for the shifted and unshifted logP values.
print('STEP 1: The Cepheid Period-Luminosity Relationship')
print('==================================================')
print('curve_fit results (Unshifted):')
print('========================================')
print('Best-fitting alpha = ', best_slope)
print('Error in alpha = ', slope_err)
print('Best-fitting beta = ', best_intercept)
print('Error in beta = ', intercept_err)
print('Corresponding Chi^2 = ', best_chi2)
print('Corresponding Reduced Chi^2 =', best_chi2/dof)
print()
print('curve_fit results (Shifted) Shifting value is 0.845115:')
print('=====================================')
print('Best-fitting alpha = ', best_slope_new)
print('Error in alpha = ', slope_err_new)
print('Best-fitting beta = ', best_intercept_new)
print('Error in beta = ', intercept_err_new)
print('Corresponding Chi^2 = ', best_chi2_new)
print('Corresponding Reduced Chi^2 =', best_chi2_new/dof)
print()
print('Correlation Matrix:')
print('==============================') 
print('              Slope ', 'Intercept')
print('Slope        ',corr_ss, ' ',corr_si)
print('Intercept   ',corr_is,'  ',corr_ii)
print('======================================')
print('Shifted Correlation Matrix:')
print('==============================')
print('              Slope ', 'Intercept')
print('Slope        ',corr_ss_new, ' ',corr_si_new)
print('Intercept   ',corr_is_new,'  ',corr_ii_new)
print('=====================================')
print('=====================================')
print('STEP 2: Distances for a Set of Nearby Galaxies')
print('=====================================')


abs_mag_model=best_slope*logperiod+best_intercept 
#Plotting the graph of unshifted logperiod and M
plt.subplot(2,1,1)#subplots are used as to have both plots on the same graph window.
plt.xlabel('Period (logP)')
plt.ylabel('Absolute Magnitude M')
plt.scatter(logperiod, abs_mag, color='red', marker='o', s=20, linestyle="None")
plt.errorbar(logperiod, abs_mag, yerr=sig_abs_mag, color="red", linewidth=1, linestyle="None")
plt.plot(logperiod, abs_mag_model, color='blue', linewidth=2)
plt.axis([0,2,-5.5,-1.5])
plt.title('Unshifted Cepheid Period-Luminosity Relationship')
plt.show()

abs_mag_model_new=best_slope_new*xshift+best_intercept_new
#Plotting the graph of shifted logperiod and M
plt.subplot(2,1,2)
plt.xlabel('Period (logP)')
plt.ylabel('Absolute Magnitude M')
plt.scatter(xshift, abs_mag, color='green', marker='o', s=20, linestyle="None")
plt.errorbar(xshift, abs_mag, yerr=sig_abs_mag, color="green", linewidth=1, linestyle="None")
plt.plot(xshift, abs_mag_model_new, color='cyan', linewidth=2)
plt.axis([0,2,-5.5,-1.5])
plt.title('Shifted Cepheid Period-Luminosity Relationship')
plt.show()


#STEP 2
#Loading in the data for these 8 galaxies

step_2_galaxies = np.loadtxt('galaxy_data.dat',\
                             unpack=True, \
                             usecols=(0), \
                             dtype=str)

r_vels, err_ext  = np.loadtxt('galaxy_data.dat',\
                          unpack=True, \
                          usecols=(1,2), \
                          dtype=float)

cepheids_1 = np.loadtxt('hst_gal1_cepheids.dat',\
                        unpack=True, \
                        usecols=(0), \
                        dtype=str)

log_cepheid_period1, app_magnitude1  = np.loadtxt('hst_gal1_cepheids.dat',\
                                                  unpack=True, \
                                                  usecols=(1,2), \
                                                  dtype=float)


cepheids_2 = np.loadtxt('hst_gal2_cepheids.dat',\
                        unpack=True, \
                        usecols=(0), \
                        dtype=str)

log_cepheid_period2, app_magnitude2  = np.loadtxt('hst_gal2_cepheids.dat',\
                                                  unpack=True, \
                                                  usecols=(1,2), \
                                                  dtype=float)


cepheids_3 = np.loadtxt('hst_gal3_cepheids.dat',\
                        unpack=True, \
                        usecols=(0), \
                        dtype=str)

log_cepheid_period3, app_magnitude3  = np.loadtxt('hst_gal3_cepheids.dat',\
                                                  unpack=True, \
                                                  usecols=(1,2), \
                                                  dtype=float)


cepheids_4 = np.loadtxt('hst_gal4_cepheids.dat',\
                        unpack=True, \
                        usecols=(0), \
                        dtype=str)

log_cepheid_period4, app_magnitude4  = np.loadtxt('hst_gal4_cepheids.dat',\
                                                  unpack=True, \
                                                  usecols=(1,2), \
                                                  dtype=float)


cepheids_5 = np.loadtxt('hst_gal5_cepheids.dat',\
                        unpack=True, \
                        usecols=(0), \
                        dtype=str)

log_cepheid_period5, app_magnitude5  = np.loadtxt('hst_gal5_cepheids.dat',\
                                                  unpack=True, \
                                                  usecols=(1,2), \
                                                  dtype=float)


cepheids_6 = np.loadtxt('hst_gal6_cepheids.dat',\
                        unpack=True, \
                        usecols=(0), \
                        dtype=str)

log_cepheid_period6, app_magnitude6  = np.loadtxt('hst_gal6_cepheids.dat',\
                                                  unpack=True, \
                                                  usecols=(1,2), \
                                                  dtype=float)


cepheids_7 = np.loadtxt('hst_gal7_cepheids.dat',\
                        unpack=True, \
                        usecols=(0), \
                        dtype=str)

log_cepheid_period7, app_magnitude7  = np.loadtxt('hst_gal7_cepheids.dat',\
                                                  unpack=True, \
                                                  usecols=(1,2), \
                                                  dtype=float)


cepheids_8 = np.loadtxt('hst_gal8_cepheids.dat',\
                        unpack=True, \
                        usecols=(0), \
                        dtype=str)

log_cepheid_period8, app_magnitude8  = np.loadtxt('hst_gal8_cepheids.dat',\
                                                  unpack=True, \
                                                  usecols=(1,2), \
                                                  dtype=float)



alpha=best_slope_new
beta=best_intercept_new
#Defining the absolute magnitudes for the 8 galaxies
M1=alpha*(log_cepheid_period1-xzero)+beta
M2=alpha*(log_cepheid_period2-xzero)+beta
M3=alpha*(log_cepheid_period3-xzero)+beta
M4=alpha*(log_cepheid_period4-xzero)+beta
M5=alpha*(log_cepheid_period5-xzero)+beta
M6=alpha*(log_cepheid_period6-xzero)+beta
M7=alpha*(log_cepheid_period7-xzero)+beta
M8=alpha*(log_cepheid_period8-xzero)+beta

#Defining the log_dpc for each galaxy
log_dpc1=(app_magnitude1-M1+5-err_ext[0])/5
log_dpc2=(app_magnitude2-M2+5-err_ext[1])/5
log_dpc3=(app_magnitude3-M3+5-err_ext[2])/5
log_dpc4=(app_magnitude4-M4+5-err_ext[3])/5
log_dpc5=(app_magnitude5-M5+5-err_ext[4])/5
log_dpc6=(app_magnitude6-M6+5-err_ext[5])/5
log_dpc7=(app_magnitude7-M7+5-err_ext[6])/5
log_dpc8=(app_magnitude8-M8+5-err_ext[7])/5

#Propagating errors for STEP 2
err_M1=np.sqrt(((log_cepheid_period1-xzero)**2)*(slope_err_new**2)+(intercept_err_new)**2)
err_M2=np.sqrt(((log_cepheid_period2-xzero)**2)*(slope_err_new**2)+(intercept_err_new)**2)
err_M3=np.sqrt(((log_cepheid_period3-xzero)**2)*(slope_err_new**2)+(intercept_err_new)**2)
err_M4=np.sqrt(((log_cepheid_period4-xzero)**2)*(slope_err_new**2)+(intercept_err_new)**2)
err_M5=np.sqrt(((log_cepheid_period5-xzero)**2)*(slope_err_new**2)+(intercept_err_new)**2)
err_M6=np.sqrt(((log_cepheid_period6-xzero)**2)*(slope_err_new**2)+(intercept_err_new)**2)
err_M7=np.sqrt(((log_cepheid_period7-xzero)**2)*(slope_err_new**2)+(intercept_err_new)**2)
err_M8=np.sqrt(((log_cepheid_period8-xzero)**2)*(slope_err_new**2)+(intercept_err_new)**2)

err_log_dpc1=np.sqrt((err_M1/5)**2)#
err_log_dpc2=np.sqrt((err_M2/5)**2)#
err_log_dpc3=np.sqrt((err_M3/5)**2)#
err_log_dpc4=np.sqrt((err_M4/5)**2)## All these errors don't take into account the intrinsic dispersion later found
err_log_dpc5=np.sqrt((err_M5/5)**2)#
err_log_dpc6=np.sqrt((err_M6/5)**2)#
err_log_dpc7=np.sqrt((err_M7/5)**2)#
err_log_dpc8=np.sqrt((err_M8/5)**2)#

#Degrees of freedom, -1 as we are modelling the brute force gridding with a single constant
df_1=len(log_dpc1)-1
df_2=len(log_dpc2)-1
df_3=len(log_dpc3)-1
df_4=len(log_dpc4)-1
df_5=len(log_dpc5)-1
df_6=len(log_dpc6)-1
df_7=len(log_dpc7)-1
df_8=len(log_dpc8)-1

#Eliminating Outliers
list_with_outlier=log_dpc4.tolist()
del list_with_outlier[6]
log_dpc4=np.array(list_with_outlier)

list_with_outlier_err=err_log_dpc4.tolist()
del list_with_outlier_err[6]
err_log_dpc4=np.array(list_with_outlier_err)


#Finding minimum chi-squared of log_dpc1
x1 = np.arange(0,25,1) #Creating an arbitrary range of x values to be plotted against the log_dpc1 values
constants=np.arange(5.0,9.0,0.001)
chi2_1 = 1.e5 + constants*0.0
best_chi2_1 = 1.e5
i=0
for m in constants: #Finding the best fitting constant
    y_test=m+0.0*log_dpc1
    chi2_1[i]=chisq(log_dpc1, err_log_dpc1, y_test)
    if (chi2_1[i] < best_chi2_1):
        best_chi2_1 = chi2_1[i]
        best_constant1 = m
    i = i+1
    
y_best1 = x1*0+best_constant1
plt.clf()
plt.xlabel('Arbitrary x')
plt.ylabel('logd_pc1')
plt.scatter(x1, log_dpc1, color='red', marker='o', linestyle="None")
plt.plot(x1, y_best1, color='blue')
plt.errorbar(x1, log_dpc1, yerr=err_log_dpc1, color="red", linewidth=2, linestyle="None")
print('=========================================================')
print('Result Without Intrinsic Dispersion')
print('=========================================================')
print('Galaxy:','NGC3627')
print('Lower Reduced Chi Value:',1-1*np.sqrt(2/(df_1)))
print('Actual Reduced Chi Value:',(best_chi2_1)/df_1,)
print('Higher Reduced Chi Value:',1+1*np.sqrt(2/(df_1)))
if best_chi2_1/df_1<1-1*np.sqrt(2/(df_1)) or best_chi2_1/df_1>1+1*np.sqrt(2/(df_1)):
    print('Galaxy NGC3627 falls outside the range,','with a reduced chi-squared value of:',(best_chi2_1)/df_1)
print('=========================================================')
print('=========================================================')
print('Result Including Intrinsic Dispersion:')
print('=========================================================')   
#Calculating error in logd_pc for Galaxy 1 (NGC 3627)
h_low_1=best_constant1 + 100.0
h_high_1=best_constant1 -100.0
j=0
for m in constants:    
    if (chi2_1[j] <= (best_chi2_1 + 1.0)):#Finding the error in constant to be used later in the script
        if (m < h_low_1):
            h_low_1 = m
        if (m > h_high_1):
            h_high_1 = m
    j=j+1 
err_h1_1=h_high_1-best_constant1
err_h2_1=best_constant1-h_low_1


err_int_1=0.047 #adjusting the intrinsic dispersion until it is sufficiently big such that the galaxy falls within the 1-sigma limit.
err_log_dpc1_int=np.sqrt(((err_M1/5)**2)+(err_int_1)**2)
constants=np.arange(5.0,9.0,0.001)
chi2_1 = 1.e5 + constants*0.0
best_chi2_1 = 1.e5
i=0
for m in constants:
    y_test=m+0.0*log_dpc1
    chi2_1[i]=chisq(log_dpc1, err_log_dpc1_int, y_test)
    if (chi2_1[i] < best_chi2_1):
        best_chi2_1 = chi2_1[i]
        best_constant1 = m
    i=i+1    
if (best_chi2_1/df_1)<1+1*np.sqrt(2/(df_1)):
    print('Updated Shifted Galaxy:','NGC3627')
    print('Updated Actual Reduced Chi:',best_chi2_1/df_1)
    print('Intrinsic error required:',err_int_1)
    print('New Distance:',10**best_constant1,'parsecs')
    print('Distance error:',(err_h1_1)*(np.log(10)*(10**best_constant1)),'parsecs')
    print('=========================================================')    


#Finding minimum chi-squared of log_dpc2
x2 = np.arange(0,17,1) #Creating an arbitrary range of x values to be plotted against the log_dpc2 values
constants=np.arange(5.0,9.0,0.001)
chi2_2 = 1.e5 + constants*0.0
best_chi2_2 = 1.e5
i=0
for m in constants: #Finding the best fitting constant
    y_test=m+0.0*log_dpc2
    chi2_2[i]=chisq(log_dpc2, err_log_dpc2, y_test)
    if (chi2_2[i] < best_chi2_2):
        best_chi2_2 = chi2_2[i]
        best_constant2 = m
    i = i+1
    
y_best2 = x2*0+best_constant2
plt.clf()
plt.xlabel('Arbitrary x')
plt.ylabel('logd_pc2')
plt.scatter(x2, log_dpc2, color='red', marker='o', linestyle="None")
plt.plot(x2, y_best2, color='blue')
plt.errorbar(x2, log_dpc2, yerr=err_log_dpc2, color="red", linewidth=2, linestyle="None")
print('=========================================================')
print('Result Without Intrinsic Dispersion')
print('=========================================================')
print('Galaxy:','NGC3982')
print('Lower Reduced Chi Value:',1-1*np.sqrt(2/(df_2)))
print('Actual Reduced Chi Value:',(best_chi2_2)/df_2,)
print('Higher Reduced Chi Value:',1+1*np.sqrt(2/(df_2)))
if best_chi2_2/df_2<1-1*np.sqrt(2/(df_2)) or best_chi2_2/df_2>1+1*np.sqrt(2/(df_2)):
    print('Galaxy NGC3982 falls outside the range,','with a reduced chi-squared value of:',(best_chi2_2)/df_2)
print('=========================================================')
print('=========================================================')
print('Result Including Intrinsic Dispersion:')
print('=========================================================')   
#Calculating error in logd_pc for Galaxy 2 (NGC 3982)
h_low_2=best_constant1 + 100
h_high_2=best_constant1 -100
j=0
for m in constants:    
    if (chi2_2[j] <= (best_chi2_2 + 1.0)):#Finding the error in constant to be used later in the script
        if (m < h_low_2):
            h_low_2 = m
        if (m > h_high_2):
            h_high_2 = m
    j=j+1 
err_h1_2=h_high_2-best_constant2
err_h2_2=best_constant2-h_low_2


err_int_2=0.087 #adjusting the intrinsic dispersion until it is sufficiently big such that the galaxy falls within the 1-sigma limit.
err_log_dpc2_int=np.sqrt(((err_M2/5)**2)+(err_int_2)**2)
constants=np.arange(5.0,9.0,0.001)
chi2_2 = 1.e5 + constants*0.0
best_chi2_2 = 1.e5
i=0
for m in constants:
    y_test=m+0.0*log_dpc2
    chi2_2[i]=chisq(log_dpc2, err_log_dpc2_int, y_test)
    if (chi2_2[i] < best_chi2_2):
        best_chi2_2 = chi2_2[i]
        best_constant2 = m
    i=i+1    
if (best_chi2_2/df_2)<1+1*np.sqrt(2/(df_2)):
    print('Updated Shifted Galaxy:','NGC3982')
    print('Updated Actual Reduced Chi:',best_chi2_2/df_2)
    print('Intrinsic error required:',err_int_2)
    print('New Distance:',10**best_constant2,'parsecs')
    print('Distance error:',(err_h1_2)*(np.log(10)*(10**best_constant2)),'parsecs')
    print('=========================================================')


#Finding minimum chi-squared of log_dpc3
x3 = np.arange(0,43,1) #Creating an arbitrary range of x values to be plotted against the log_dpc3 values
constants=np.arange(5.0,9.0,0.001)
chi2_3 = 1.e5 + constants*0.0
best_chi2_3 = 1.e5
i=0
for m in constants: #Finding the best fitting constant
    y_test=m+0.0*log_dpc3
    chi2_3[i]=chisq(log_dpc3, err_log_dpc3, y_test)
    if (chi2_3[i] < best_chi2_3):
        best_chi2_3 = chi2_3[i]
        best_constant3 = m
    i = i+1
    
y_best3 = x3*0+best_constant3
plt.clf()
plt.xlabel('Arbitrary x')
plt.ylabel('logd_pc3')
plt.scatter(x3, log_dpc3, color='red', marker='o', linestyle="None")
plt.plot(x3, y_best3, color='blue')
plt.errorbar(x3, log_dpc3, yerr=err_log_dpc3, color="red", linewidth=2, linestyle="None")
print('=========================================================')
print('Result Without Intrinsic Dispersion')
print('=========================================================')
print('Galaxy:','NGC4496A')
print('Lower Reduced Chi Value:',1-1*np.sqrt(2/(df_3)))
print('Actual Reduced Chi Value:',(best_chi2_3)/df_3,)
print('Higher Reduced Chi Value:',1+1*np.sqrt(2/(df_3)))
if best_chi2_3/df_3<1-1*np.sqrt(2/(df_3)) or best_chi2_3/df_3>1+1*np.sqrt(2/(df_3)):
    print('Galaxy NGC4496A falls outside the range,','with a reduced chi-squared value of:',(best_chi2_3)/df_3)
print('=========================================================')
print('=========================================================')
print('Result Including Intrinsic Dispersion:')
print('=========================================================')   
#Calculating error in logd_pc for Galaxy 3 (NGC 4496A)
h_low_3=best_constant3 + 100
h_high_3=best_constant3 -100
j=0
for m in constants:    
    if (chi2_3[j] <= (best_chi2_3 + 1.0)):#Finding the error in constant to be used later in the script
        if (m < h_low_3):
            h_low_3 = m
        if (m > h_high_3):
            h_high_3 = m
    j=j+1 
err_h1_3=h_high_3-best_constant3
err_h2_3=best_constant1-h_low_3


err_int_3=0.044 #adjusting the intrinsic dispersion until it is sufficiently big such that the galaxy falls within the 1-sigma limit.
err_log_dpc3_int=np.sqrt(((err_M3/5)**2)+(err_int_3)**2)
constants=np.arange(5.0,9.0,0.001)
chi2_3 = 1.e5 + constants*0.0
best_chi2_3 = 1.e5
i=0
for m in constants:
    y_test=m+0.0*log_dpc3
    chi2_3[i]=chisq(log_dpc3, err_log_dpc3_int, y_test)
    if (chi2_3[i] < best_chi2_3):
        best_chi2_3 = chi2_3[i]
        best_constant3 = m
    i=i+1    
if (best_chi2_3/df_3)<1+1*np.sqrt(2/(df_3)):
    print('Updated Shifted Galaxy:','NGC4496A')
    print('Updated Actual Reduced Chi:',best_chi2_3/df_3)
    print('Intrinsic error required:',err_int_3)
    print('New Distance:',10**best_constant3,'parsecs')
    print('Distance error:',(err_h1_3)*(np.log(10)*(10**best_constant3)),'parsecs')
    print('=========================================================')


#Finding minimum chi-squared of log_dpc4
x4 = np.arange(0,21,1) #Creating an arbitrary range of x values to be plotted against the log_dpc4 values
constants=np.arange(5.0,9.0,0.001)
chi2_4 = 1.e5 + constants*0.0
best_chi2_4 = 1.e5
i=0
for m in constants: #Finding the best fitting constant
    y_test=m+0.0*log_dpc4
    chi2_4[i]=chisq(log_dpc4, err_log_dpc4, y_test)
    if (chi2_4[i] < best_chi2_4):
        best_chi2_4 = chi2_4[i]
        best_constant4 = m
    i = i+1
    
y_best4 = x4*0+best_constant4
plt.clf()
plt.xlabel('Arbitrary x')
plt.ylabel('logd_pc4')
plt.scatter(x4, log_dpc4, color='red', marker='o', linestyle="None")
plt.plot(x4, y_best4, color='blue')
plt.errorbar(x4, log_dpc4, yerr=err_log_dpc4, color="red", linewidth=2, linestyle="None")
print('=========================================================')
print('Result Without Intrinsic Dispersion')
print('=========================================================')
print('Galaxy:','NGC4527')
print('Lower Reduced Chi Value:',1-1*np.sqrt(2/(df_4)))
print('Actual Reduced Chi Value:',(best_chi2_4)/df_4,)
print('Higher Reduced Chi Value:',1+1*np.sqrt(2/(df_4)))
if best_chi2_4/df_4<1-1*np.sqrt(2/(df_4)) or best_chi2_4/df_4>1+1*np.sqrt(2/(df_4)):
    print('Galaxy NGC4527 falls outside the range,','with a reduced chi-squared value of:',(best_chi2_4)/df_4)
print('=========================================================')
print('=========================================================')
print('Result Including Intrinsic Dispersion:')
print('=========================================================')   
#Calculating error in logd_pc for Galaxy 4 (NGC 4527)
h_low_4=best_constant4 + 100
h_high_4=best_constant4 -100
j=0
for m in constants:    
    if (chi2_4[j] <= (best_chi2_4 + 1.0)):#Finding the error in constant to be used later in the script
        if (m < h_low_4):
            h_low_4 = m
        if (m > h_high_4):
            h_high_4 = m
    j=j+1 
err_h1_4=h_high_4-best_constant4
err_h2_4=best_constant4-h_low_4


err_int_4=0.016 #adjusting the intrinsic dispersion until it is sufficiently big such that the galaxy falls within the 1-sigma limit.
err_log_dpc4_int=np.sqrt(((err_M4/5)**2)+(err_int_4)**2) 
list_with_outlier_err4=err_log_dpc4_int.tolist()
del list_with_outlier_err4[6] #Removing th error corressponding with the earlier outlier spotted in log_dpc4
err_log_dpc4_int=np.array(list_with_outlier_err4)
constants=np.arange(5.0,9.0,0.001)
chi2_4 = 1.e5 + constants*0.0
best_chi2_4 = 1.e5
i=0
for m in constants:
    y_test=m+0.0*log_dpc4
    chi2_4[i]=chisq(log_dpc4, err_log_dpc4_int, y_test)
    if (chi2_4[i] < best_chi2_4):
        best_chi2_4 = chi2_4[i]
        best_constant4 = m
    i=i+1    
if (best_chi2_4/df_4)<1+1*np.sqrt(2/(df_4)):
    print('Updated Shifted Galaxy:','NGC4527')
    print('Updated Actual Reduced Chi:',best_chi2_4/df_4)
    print('Intrinsic error required:',err_int_4)
    print('New Distance:',10**best_constant4,'parsecs')
    print('Distance error:',(err_h1_4)*(np.log(10)*(10**best_constant4)),'parsecs')
    print('=========================================================')


#Finding minimum chi-squared of log_dpc5
x5 = np.arange(0,31,1) #Creating an arbitrary range of x values to be plotted against the log_dpc5 values
constants=np.arange(5.0,9.0,0.001)
chi2_5 = 1.e5 + constants*0.0
best_chi2_5 = 1.e5
i=0
for m in constants: #Finding the best fitting constant
    y_test=m+0.0*log_dpc5
    chi2_5[i]=chisq(log_dpc5, err_log_dpc5, y_test)
    if (chi2_5[i] < best_chi2_5):
        best_chi2_5 = chi2_5[i]
        best_constant5 = m
    i = i+1
    
y_best5 = x5*0+best_constant5
plt.clf()
plt.xlabel('Arbitrary x')
plt.ylabel('logd_pc5')
plt.scatter(x5, log_dpc5, color='red', marker='o', linestyle="None")
plt.plot(x5, y_best5, color='blue')
plt.errorbar(x5, log_dpc5, yerr=err_log_dpc5, color="red", linewidth=2, linestyle="None")
print('=========================================================')
print('Result Without Intrinsic Dispersion')
print('=========================================================')
print('Galaxy:','NGC4536')
print('Lower Reduced Chi Value:',1-1*np.sqrt(2/(df_5)))
print('Actual Reduced Chi Value:',(best_chi2_5)/df_5,)
print('Higher Reduced Chi Value:',1+1*np.sqrt(2/(df_5)))
if best_chi2_5/df_5<1-1*np.sqrt(2/(df_5)) or best_chi2_5/df_5>1+1*np.sqrt(2/(df_5)):
    print('Galaxy NGC4536 falls outside the range,','with a reduced chi-squared value of:',(best_chi2_5)/df_5)
print('=========================================================')
print('=========================================================')
print('Result Including Intrinsic Dispersion:')
print('=========================================================')   
#Calculating error in logd_pc for Galaxy 5 (NGC 4536)
h_low_5=best_constant5 + 100
h_high_5=best_constant5 -100
j=0
for m in constants:    
    if (chi2_5[j] <= (best_chi2_5 + 1.0)):#Finding the error in constant to be used later in the script
        if (m < h_low_5):
            h_low_5 = m
        if (m > h_high_5):
            h_high_5 = m
    j=j+1 
err_h1_5=h_high_5-best_constant5
err_h2_5=best_constant5-h_low_5


err_int_5=0.040 #adjusting the intrinsic dispersion until it is sufficiently big such that the galaxy falls within the 1-sigma limit.
err_log_dpc5_int=np.sqrt(((err_M5/5)**2)+(err_int_5)**2)
constants=np.arange(5.0,9.0,0.001)
chi2_5 = 1.e5 + constants*0.0
best_chi2_5 = 1.e5
i=0
for m in constants:
    y_test=m+0.0*log_dpc5
    chi2_5[i]=chisq(log_dpc5, err_log_dpc5_int, y_test)
    if (chi2_5[i] < best_chi2_5):
        best_chi2_5 = chi2_5[i]
        best_constant5 = m
    i=i+1    
if (best_chi2_5/df_5)<1+1*np.sqrt(2/(df_5)):
    print('Updated Shifted Galaxy:','NGC4536')
    print('Updated Actual Reduced Chi:',best_chi2_5/df_5)
    print('Intrinsic error required:',err_int_5)
    print('New Distance:',10**best_constant5,'parsecs')
    print('Distance error:',(err_h1_5)*(np.log(10)*(10**best_constant5)),'parsecs')
    print('=========================================================') 


#Finding minimum chi-squared of log_dpc6
x6 = np.arange(0,15,1) #Creating an arbitrary range of x values to be plotted against the log_dpc6 values
constants=np.arange(5.0,9.0,0.001)
chi2_6 = 1.e5 + constants*0.0
best_chi2_6 = 1.e5
i=0
for m in constants: #Finding the best fitting constant
    y_test=m+0.0*log_dpc6
    chi2_6[i]=chisq(log_dpc6, err_log_dpc6, y_test)
    if (chi2_6[i] < best_chi2_6):
        best_chi2_6 = chi2_6[i]
        best_constant6 = m
    i = i+1
    
y_best6 = x6*0+best_constant6
plt.clf()
plt.xlabel('Arbitrary x')
plt.ylabel('logd_pc6')
plt.scatter(x6, log_dpc6, color='red', marker='o', linestyle="None")
plt.plot(x6, y_best6, color='blue')
plt.errorbar(x6, log_dpc6, yerr=err_log_dpc6, color="red", linewidth=2, linestyle="None")
print('=========================================================')
print('Result Without Intrinsic Dispersion')
print('=========================================================')
print('Galaxy:','NGC4639')
print('Lower Reduced Chi Value:',1-1*np.sqrt(2/(df_6)))
print('Actual Reduced Chi Value:',(best_chi2_6)/df_6,)
print('Higher Reduced Chi Value:',1+1*np.sqrt(2/(df_6)))
if best_chi2_6/df_6<1-1*np.sqrt(2/(df_6)) or best_chi2_6/df_6>1+1*np.sqrt(2/(df_6)):
    print('Galaxy NGC4639 falls outside the range,','with a reduced chi-squared value of:',(best_chi2_6)/df_6)
print('=========================================================')
print('=========================================================')
print('Result Including Intrinsic Dispersion:')
print('=========================================================')   
#Calculating error in logd_pc for Galaxy 6 (NGC 4639)
h_low_6=best_constant6 + 100
h_high_6=best_constant6 -100
j=0
for m in constants:    
    if (chi2_6[j] <= (best_chi2_6 + 1.0)):#Finding the error in constant to be used later in the script
        if (m < h_low_6):
            h_low_6 = m
        if (m > h_high_6):
            h_high_6 = m
    j=j+1 
err_h1_6=h_high_6-best_constant6
err_h2_6=best_constant6-h_low_6


err_int_6=0.061 #adjusting the intrinsic dispersion until it is sufficiently big such that the galaxy falls within the 1-sigma limit.
err_log_dpc6_int=np.sqrt(((err_M6/5)**2)+(err_int_6)**2)
constants=np.arange(5.0,9.0,0.001)
chi2_6 = 1.e5 + constants*0.0
best_chi2_6 = 1.e5
i=0
for m in constants:
    y_test=m+0.0*log_dpc6
    chi2_6[i]=chisq(log_dpc6, err_log_dpc6_int, y_test)
    if (chi2_6[i] < best_chi2_6):
        best_chi2_6 = chi2_6[i]
        best_constant6 = m
    i=i+1    
if (best_chi2_6/df_6)<1+1*np.sqrt(2/(df_6)):
    print('Updated Shifted Galaxy:','NGC4639')
    print('Updated Actual Reduced Chi:',best_chi2_6/df_6)
    print('Intrinsic error required:',err_int_6)
    print('New Distance:',10**best_constant6,'parsecs')
    print('Distance error:',(err_h1_6)*(np.log(10)*(10**best_constant6)),'parsecs')
    print('=========================================================')


#Finding minimum chi-squared of log_dpc7
x7 = np.arange(0,12,1) #Creating an arbitrary range of x values to be plotted against the log_dpc7 values
constants=np.arange(5.0,9.0,0.001)
chi2_7 = 1.e5 + constants*0.0
best_chi2_7 = 1.e5
i=0
for m in constants: #Finding the best fitting constant
    y_test=m+0.0*log_dpc7
    chi2_7[i]=chisq(log_dpc7, err_log_dpc7, y_test)
    if (chi2_7[i] < best_chi2_7):
        best_chi2_7 = chi2_7[i]
        best_constant7 = m
    i = i+1
    
y_best7 = x7*0+best_constant7
plt.clf()
plt.xlabel('Arbitrary x')
plt.ylabel('logd_pc7')
plt.scatter(x7, log_dpc7, color='red', marker='o', linestyle="None")
plt.plot(x7, y_best7, color='blue')
plt.errorbar(x7, log_dpc7, yerr=err_log_dpc7, color="red", linewidth=2, linestyle="None")
print('=========================================================')
print('Result Without Intrinsic Dispersion')
print('=========================================================')
print('Galaxy:','NGC5253')
print('Lower Reduced Chi Value:',1-1*np.sqrt(2/(df_7)))
print('Actual Reduced Chi Value:',(best_chi2_7)/df_7,)
print('Higher Reduced Chi Value:',1+1*np.sqrt(2/(df_7)))
if best_chi2_7/df_7<1-1*np.sqrt(2/(df_7)) or best_chi2_7/df_7>1+1*np.sqrt(2/(df_7)):
    print('Galaxy NGC5253 falls outside the range,','with a reduced chi-squared value of:',(best_chi2_7)/df_7)
print('=========================================================')
print('=========================================================')
print('Result Including Intrinsic Dispersion:')
print('=========================================================')   
#Calculating error in logd_pc for Galaxy 7 (NGC 5253)
h_low_7=best_constant7 + 100
h_high_7=best_constant7 -100
j=0
for m in constants:    
    if (chi2_7[j] <= (best_chi2_7 + 1.0)):#Finding the error in constant to be used later in the script
        if (m < h_low_7):
            h_low_7 = m
        if (m > h_high_7):
            h_high_7 = m
    j=j+1 
err_h1_7=h_high_7-best_constant7
err_h2_7=best_constant7-h_low_7


err_int_7=0.044 #adjusting the intrinsic dispersion until it is sufficiently big such that the galaxy falls within the 1-sigma limit.
err_log_dpc7_int=np.sqrt(((err_M7/5)**2)+(err_int_7)**2)
constants=np.arange(5.0,9.0,0.001)
chi2_7 = 1.e5 + constants*0.0
best_chi2_7 = 1.e5
i=0
for m in constants:
    y_test=m+0.0*log_dpc7
    chi2_7[i]=chisq(log_dpc7, err_log_dpc7_int, y_test)
    if (chi2_7[i] < best_chi2_7):
        best_chi2_7 = chi2_7[i]
        best_constant7 = m
    i=i+1    
if (best_chi2_7/df_7)<1+1*np.sqrt(2/(df_7)):
    print('Updated Shifted Galaxy:','NGC5253')
    print('Updated Actual Reduced Chi:',best_chi2_7/df_7)
    print('Intrinsic error required:',err_int_7)
    print('New Distance:',10**best_constant7,'parsecs')
    print('Distance error:',(err_h1_7)*(np.log(10)*(10**best_constant7)),'parsecs')
    print('=========================================================')


#Finding minimum chi-squared of log_dpc8
x8 = np.arange(0,27,1) #Creating an arbitrary range of x values to be plotted against the log_dpc8 values
constants=np.arange(5.0,9.0,0.001)
chi2_8 = 1.e5 + constants*0.0
best_chi2_8 = 1.e5
i=0
for m in constants: #Finding the best fitting constant
    y_test=m+0.0*log_dpc8
    chi2_8[i]=chisq(log_dpc8, err_log_dpc8, y_test)
    if (chi2_8[i] < best_chi2_8):
        best_chi2_8 = chi2_8[i]
        best_constant8 = m
    i = i+1
    
y_best8 = x8*0+best_constant8
plt.clf()
plt.xlabel('Arbitrary x')
plt.ylabel('logd_pc8')
plt.scatter(x8, log_dpc8, color='red', marker='o', linestyle="None")
plt.plot(x8, y_best8, color='blue')
plt.errorbar(x8, log_dpc8, yerr=err_log_dpc8, color="red", linewidth=2, linestyle="None")
print('=========================================================')
print('Result Without Intrinsic Dispersion')
print('=========================================================')
print('Galaxy:','IC4182')
print('Lower Reduced Chi Value:',1-1*np.sqrt(2/(df_8)))
print('Actual Reduced Chi Value:',(best_chi2_8)/df_8,)
print('Higher Reduced Chi Value:',1+1*np.sqrt(2/(df_8)))
if best_chi2_8/df_8<1-1*np.sqrt(2/(df_8)) or best_chi2_8/df_8>1+1*np.sqrt(2/(df_8)):
    print('Galaxy IC4182 falls outside the range,','with a reduced chi-squared value of:',(best_chi2_8)/df_8)
print('=========================================================')
print('=========================================================')
print('Result Including Intrinsic Dispersion:')
print('=========================================================')   
#Calculating error in logd_pc for Galaxy 8 (IC 4182)
h_low_8=best_constant8 + 100
h_high_8=best_constant8 -100
j=0
for m in constants:    
    if (chi2_8[j] <= (best_chi2_8 + 1.0)):#Finding the error in constant to be used later in the script
        if (m < h_low_8):
            h_low_8 = m
        if (m > h_high_8):
            h_high_8 = m
    j=j+1 
err_h1_8=h_high_8-best_constant8
err_h2_8=best_constant8-h_low_8


err_int_8=0.046 #adjusting the intrinsic dispersion until it is sufficiently big such that the galaxy falls within the 1-sigma limit.
err_log_dpc8_int=np.sqrt(((err_M8/5)**2)+(err_int_8)**2)
constants=np.arange(5.0,9.0,0.001)
chi2_8 = 1.e5 + constants*0.0
best_chi2_8 = 1.e5
i=0
for m in constants:
    y_test=m+0.0*log_dpc8
    chi2_8[i]=chisq(log_dpc8, err_log_dpc8_int, y_test)
    if (chi2_8[i] < best_chi2_8):
        best_chi2_8 = chi2_8[i]
        best_constant8 = m
    i=i+1    
if (best_chi2_8/df_8)<1+1*np.sqrt(2/(df_8)):
    print('Updated Shifted Galaxy:','IC4182')
    print('Updated Actual Reduced Chi:',best_chi2_8/df_8)
    print('Intrinsic error required:',err_int_8)
    print('New Distance:',10**best_constant8,'parsecs')
    print('Distance error:',(err_h1_8)*(np.log(10)*(10**best_constant8)),'parsecs')
    print('=========================================================') 
    
#Defining the distance modulus for each galaxy
dist_mod1=5*best_constant1-5+err_ext[0]    
dist_mod2=5*best_constant2-5+err_ext[1]    
dist_mod3=5*best_constant3-5+err_ext[2]    
dist_mod4=5*best_constant4-5+err_ext[3]    
dist_mod5=5*best_constant5-5+err_ext[4]    
dist_mod6=5*best_constant6-5+err_ext[5]    
dist_mod7=5*best_constant7-5+err_ext[6]    
dist_mod8=5*best_constant8-5+err_ext[7]

dist_mod_values=[dist_mod1,dist_mod2,dist_mod3,dist_mod4,dist_mod5,dist_mod6,dist_mod7,dist_mod8]

#Defining the error in distance modulus for each galaxy
err_dist_mod1=5*err_h1_1  
err_dist_mod2=5*err_h1_2
err_dist_mod3=5*err_h1_3
err_dist_mod4=5*err_h1_4
err_dist_mod5=5*err_h1_5
err_dist_mod6=5*err_h1_6
err_dist_mod7=5*err_h1_7
err_dist_mod8=5*err_h1_8

err_dist_mod_values=[err_dist_mod1,err_dist_mod2,err_dist_mod3,err_dist_mod4,err_dist_mod5,err_dist_mod6,err_dist_mod7,err_dist_mod8]

#Plotting graph of distance modulus against an arbitrary x
plt.clf()
xvalues=[1,2,3,4,5,6,7,8]
plt.scatter(xvalues,dist_mod_values)
plt.errorbar(xvalues,dist_mod_values,yerr=err_dist_mod_values,ls='none', color='red')
plt.xlabel('Arbitrary x')
plt.ylabel('Distance Modulus')
plt.title('Distance Moduli')
plt.show()
print('================================================')
print('STEP 3: Expansion Rate of the Universe')
print('================================================')

slopes=np.arange(45.0,80.0,0.001)
err_r_vels=177 #This error will be varied to find the a sufficient intrinsic dispersion such that the reduced chi is close to 1, around 0.1 away 
distance_list=[(10**best_constant1)*10**(-6),(10**best_constant2)*10**(-6),(10**best_constant3)*10**(-6),(10**best_constant4)*10**(-6),(10**best_constant5)*10**(-6),(10**best_constant6)*10**(-6),(10**best_constant7)*10**(-6),(10**best_constant8)*10**(-6)]
distance_list_array=np.array(distance_list)
err_distance_list=[(err_h1_1)*(np.log(10))*(10**best_constant1)*10**(-6),(err_h1_2)*(np.log(10))*(10**best_constant2)*10**(-6),(err_h1_3)*(np.log(10))*(10**best_constant3)*10**(-6),(err_h1_4)*(np.log(10))*(10**best_constant4)*10**(-6),(err_h1_5)*(np.log(10))*(10**best_constant5)*10**(-6),(err_h1_6)*(np.log(10))*(10**best_constant6)*10**(-6),(err_h1_7)*(np.log(10))*(10**best_constant7)*10**(-6),(err_h1_8)*(np.log(10))*(10**best_constant8)*10**(-6)]
err_distance_list_array=np.array(err_distance_list)
df=7 #degrees of freedom is 7 because we are only modelling with one constant
chi2 = 1.e5+slopes*0.0
best_chi2 = 1.e5
i=0
for m in slopes:
    r_test=m*distance_list_array
    chi2[i]=chisqxy(r_vels, err_r_vels, err_distance_list_array, r_test, m) #finding the minimum chi2 value while using errors 
    if (chi2[i] < best_chi2):
        best_chi2 = chi2[i]
        best_slope=m
    i=i+1
               
        

h_low=best_slope + 100
h_high=best_slope -100
i=0
for m in slopes:    
    if (chi2[i] <= (best_chi2 + 1.0)):#Finding the error in h0
        if (m < h_low):
            h_low = m
        if (m > h_high):
            h_high = m
    i=i+1 
err_h1=h_high-best_slope
err_h2=best_slope-h_low
print('Best fit value of H0:',best_slope,'km/s/Mpc')
print('Reduced chi-squared value:',best_chi2/df)
print('Error on best fit H0:',err_h1,'km/s/Mpc') #Using the higher error 
print('Intrinsic error for r_vels:',err_r_vels,'km/s')
print('==============================================')
print('==============================================')
print('STEP 4: Age of the Universe')
print('==============================================')

time=31536000
one_over_h0=1/(best_slope/3.086e19) #Estimate for the age of the universe is simply 1 over the hubble constant
err_one_over_h0=(err_h1/3.086e19)*(one_over_h0)**2
print('Age of the Universe:',one_over_h0/(time*1.e9),'Billion years')
print('Error in Age of Universe:','+/-',err_one_over_h0/(time*1.e9),'Billion years')
plt.clf()
x_final=np.arange(0,25,1)
plt.plot(x_final,straight_line(best_slope,x_final,0),color='red')
plt.plot(x_final,straight_line(h_low,x_final,0),ls=':',color='blue',lw=1)
plt.plot(x_final,straight_line(h_high,x_final,0),ls=':',color='blue',lw=1)
plt.scatter(distance_list_array,r_vels,marker='o',color='yellow')
plt.errorbar(distance_list_array,r_vels, xerr=err_distance_list_array, yerr=err_r_vels, ls='none',color='green')
plt.xlabel('Distance of Galaxy (Mpc)')
plt.ylabel('Recession Velocity (km/s)')
plt.title('Expansion Rate Of The Universe')
plt.show()