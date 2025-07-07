### python program to plot all OLCI L2 water vapour and A865 products
#########################################################################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import glob as glob
from netCDF4 import Dataset
import pandas as pd
from pylab import *
import sys
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator as rgi
from mpl_toolkits.basemap import Basemap

def OC5_chl_gen(Rrs412,Rrs443,Rrs490,Rrs510,Rrs560,LUT):
#input variables as mentioned in LUT
    xmin = -2
    ymin = -0.2
    xmin560 = 0
    pasx = 0.02
    pasy = 0.0352
    pasx560 = 0.03
    nb = 200
    nlw_412 = Rrs412 * 171.4
    nlw_443 = Rrs443 * 187.7
    nlw_490 = Rrs490 * 192.9
    nlw_510 = Rrs510 * 192.7
    nlw_560 = Rrs560 * 180.0
    E412 = 171.4
    E443 = 187.7
    E490 = 192.9
    E510 = 192.7
    E560 = 180.0
    E665 = 153.1
#determine LUT indices
    Ecor560div510 = E560 / E510
    Ecor560div490 = E560 / E490
    Ecor560div443 = E560 / E443
    r443div560 = nlw_443 / nlw_560
    r490div560 = nlw_490 / nlw_560
    r510div560 = nlw_510 / nlw_560
    ind412 = (nlw_412 - xmin)/pasx
    ind560 = (nlw_560 - xmin560)/pasx560
    R_oc4 = np.max([r510div560 * Ecor560div510,r490div560 * Ecor560div490,r443div560 * Ecor560div443])
    indil = (R_oc4 - ymin)/pasy
#Extract CHL value from LUT by interpolation
    if ind560 > 0 and indil > 0 and ind412 <= 199 and ind560 <= 199 and indil <= 199:
        x = np.array(range(nb))
        y = np.array(range(nb))
        z = np.array(range(nb))
        fn = rgi((x,y,z),LUT)
        chl = fn([ind560,ind412,indil])
    else:
        chl = -999
    return chl



def apply_OC4(CHL_OC4, R12, R53, R560):
    iN = np.shape(CHL_OC4)[0]
    jN = np.shape(CHL_OC4)[1]
    RES = np.ones([iN, jN])  ### OC4 OK default
## High CHL
    RES[np.where(CHL_OC4 >= 10)] = 5 # high CHL
    c1 = np.where(RES == 5, 0, 1)
### High SPM
    c2 = np.where(np.log10(R560) > -2.588 +0.676*R53 -0.117*R53*R53  +0.4*0.205, 1, 0)
    RES[np.where(c1+c2 == 2)] = 2  ## high SPM
### High CDOM
    c3 = np.where(R12 < 1.043 - -0.226*R53 -0.056*R53*R53-0.3*0.212, 1, 0)
    RES[np.where(c1 + c3 == 2)] = 3 ## high CDOM
### high CDOM and high SPM
    RES[np.where(c1 + c2 + c3 == 3)] = 4 ## high SPM
### low CDOM
    c4 = np.where(R12  > 1.25, 1, 0)
    RES[np.where(c1+c4 == 2)] = 6
    return(RES)




def apply_OC5(CHL_OC5, R12, R53, R560):
    iN = np.shape(CHL_OC5)[0]
    jN = np.shape(CHL_OC5)[1]
    RES = np.ones([iN, jN])  ### OC4 OK default
## High CHL
    RES[np.where(CHL_OC5 >= 10)] = 6 # high CHL
    c1 = np.where(RES == 5, 0, 1)
### High SPM (a bit)
    c2 = np.where(np.log10(R560) > -2.624 +0.787*R53 -0.125*R53*R53  +1.3*0.239, 1, 0)
    RES[np.where(c1+c2 == 2)] = 2  ## a OC5 "large"
### High CDOM (a bit)
    c3 = np.where(R12 <  1.014 - 0.079*R53 -0.123*R53*R53-2.1*0.247, 1, 0)
    RES[np.where(c1 + c3 == 2)] = 2 ## OC5 "large"
### Very high SPM (a bit)
    c4 = np.where(np.log10(R560) > -2.624 +0.787*R53 -0.125*R53*R53  +2.5*0.239, 1, 0)
    RES[np.where(c1+c4 == 2)] = 3  ## very high SPM
### Very high CDOM (a bit)
    c5 = np.where(R12 <  1.014 - 0.079*R53 -0.123*R53*R53-3.2*0.247, 1, 0)
    RES[np.where(c1 + c5 == 2)] = 4 ## very high CDOM
### high CDOM and high SPM
    RES[np.where(c1 + c4 + c5 == 3)] = 5 ## high SPM
### low CDOM
    c4 = np.where(R12  > 1.25, 1, 0)
    RES[np.where(c1+c4 == 2)] = 7
    return(RES)



def apply_GONS(CHL_OC4, R665, CHL_GONS):
    iN = np.shape(CHL_OC4)[0]
    jN = np.shape(CHL_OC4)[1]
    RES = np.ones([iN, jN])*0  ### Don't apply GONS algorithm
## condition 1: chl_OC4 > 5
    c1 = np.where(CHL_OC4 >= 8.5, 1, 0)
### condition 2: R665 > 0.002
    c2 = np.where(R665 >= 0.0081, 1, 0)
### condition 4: CHL_GONS > 2
    c3 = np.where(CHL_GONS > 2, 1, 0)
    RES[np.where(c2 == 0)] = 3 ## low SPM
    RES[np.where(c1 + c3 < 2)] = 2 ## low CHL
    RES[np.where(c1 + c2 + c3 == 3)] = 1 ## Apply CHL_GONS algorithm
### high CDOM and high SPM
    return(RES)


def fGONS(R665, R709, R779):
    aw1 = 0.4
    aw2 = 0.7
    asp_665 = 0.0146
    p = 1.05
    bb = 1.61*(R779/1)/(0.082-0.6*(R779/1)) 
    CHL = (1/asp_665)*((R709/R665)*(aw2 + bb) - aw1 - bb**p) 
    return CHL


def fOC4(R443, R490, R510, R560):
    allBLUE = np.array([R443, R490, R510])
    BLUEmax = np.max(allBLUE, axis=0)
    R = np.log10(BLUEmax/R560)
    a0 = 0.4502748
    a1 = -3.259491
    a2 = 3.52271
    a3 = -3.359422
    a4 = 0.949586
    CHL = 10**(a0 + a1*R + a2*R**2 + a3*R**3 + a4*R**4)
    return CHL


############################### START SCRIPT  ##############################
############################################################################


### list available files

### Load LUT
LUTpath = '/home/heloise/Documents/algo_CHL/EUNOSAT/OC5/NEW_LUT_meris_2015_ext_processed.npy'
LUT = np.load(LUTpath)


datadir = "/home/heloise/Documents/algo_CHL/APPLY_OLCI_2/OLCI_F3/"

#olcidir = sys.argv[1]
olcidir=datadir+"S3A_OL_1_EFR____20170618T104144_20170618T104344_20171019T200615_0119_019_051______MR1_R_NT_002.SEN3.pol.nc"
print(olcidir)

patterns = olcidir[70:85]

### read data
################

### Lat and lon
#################
nc = Dataset(olcidir, mode='r')
lon = np.array(nc.variables['longitude'])
lat = np.array(nc.variables['latitude'])

### OLCI mask
##############
flag =  np.array(nc.variables['bitmask'])

### reflectances
###################
R1 = np.array(nc.variables['Rw412'])
R2 = np.array(nc.variables['Rw443'])
R3 = np.array(nc.variables['Rw490'])
R4 = np.array(nc.variables['Rw510'])
R5 = np.array(nc.variables['Rw560'])
R6 = np.array(nc.variables['Rw620'])
R7 = np.array(nc.variables['Rw665'])
R9 = np.array(nc.variables['Rw709'])
R12 = np.array(nc.variables['Rw779'])

#Application mask and QC
#R1[np.where(mask == 2)] = np.nan
R1[np.where(R1 < 0)] = np.nan
R1[np.where(R1 > 1)] = np.nan
#R2[np.where(mask == 2)] = np.nan
R2[np.where(R2 < 0)] = np.nan
R2[np.where(R2 > 1)] = np.nan
#R3[np.where(mask == 2)] = np.nan
R3[np.where(R3 < 0)] = np.nan
R3[np.where(R3 > 1)] = np.nan
#R4[np.where(mask == 2)] = np.nan
R4[np.where(R4 < 0)] = np.nan
R4[np.where(R4 > 1)] = np.nan
#R5[np.where(mask == 2)] = np.nan
R5[np.where(R5 < 0)] = np.nan
R5[np.where(R5 > 1)] = np.nan
#R6[np.where(mask == 2)] = np.nan
R6[np.where(R6 < 0)] = np.nan
R6[np.where(R6 > 1)] = np.nan
#R7[np.where(mask == 2)] = np.nan
R7[np.where(R7 < 0)] = np.nan
R7[np.where(R7 > 1)] = np.nan
#R9[np.where(mask == 2)] = np.nan
R9[np.where(R9 < 0)] = np.nan
R9[np.where(R9 > 1)] = np.nan
#R12[np.where(mask == 2)] = np.nan
R12[np.where(R12 < 0)] = np.nan
R12[np.where(R12 > 1)] = np.nan

### Application algorithms 

### CHL-OC4
CHL_OC4 = fOC4(R2, R3, R4, R5)
CHL_OC4[np.where(np.isnan(R1))] = np.nan

### CHL-OC5
iN = np.shape(CHL_OC4)[0]
jN = np.shape(CHL_OC4)[1]

CHL_OC5= np.ones([iN, jN])*-999
for i in range(iN):
    for j in range(jN):
        chlij = OC5_chl_gen(R1[i,j]/np.pi,R2[i,j]/np.pi,R3[i,j]/np.pi,R4[i,j]/np.pi,R5[i,j]/np.pi,LUT)
        CHL_OC5[i,j] = chlij
#CHL_OC5[np.where(mask == 2)] = np.nan
CHL_OC5[np.where(CHL_OC5 < 0)] = np.nan
CHL_OC5[np.where(np.isnan(R1))] = np.nan

APPOC4 = apply_OC4(CHL_OC4=CHL_OC4, R12=R1/R2, R53=R5/R3, R560=R5)
#APPOC4[np.where(mask == 2)] = np.nan
APPOC4[np.where(np.isnan(R1))] = np.nan

APPOC5 = apply_OC5(CHL_OC5=CHL_OC5, R12=R1/R2, R53=R5/R3, R560=R5)
#APPOC5[np.where(mask == 2)] = np.nan
APPOC5[np.where(np.isnan(R1))] = np.nan

CHL_GONS = fGONS(R665=R7, R709=R9, R779=R12)
#CHL_GONS[np.where(mask == 2)] = np.nan
CHL_GONS[np.where(np.isnan(R1))] = np.nan

APPGONS = apply_GONS(CHL_OC4=CHL_OC4, R665=R7, CHL_GONS=CHL_GONS)
#APPGONS[np.where(mask == 2)] = np.nan
APPGONS[np.where(np.isnan(R1))] = np.nan

WALGO = np.ones([iN, jN])  ### No algo apriori
WALGO[np.where(APPOC5 == 2)] = 3 ### OC5
WALGO[np.where(APPOC5 == 1)] = 3 ### OC5
WALGO[np.where(APPOC4 == 1)] = 2 ### OC4
WALGO[np.where(APPGONS == 1)] = 4 ### GONS
#WALGO[np.where(mask == 2)] = np.nan
WALGO[np.where(np.isnan(R1))] = np.nan



##### PLOTS  algo application  ###########################
##########################################################

## Plot CHL OC4
 
fig = plt.figure(figsize=(10, 10))


m = Basemap(projection='merc',llcrnrlat=np.min(lat),urcrnrlat=np.max(lat),\
              llcrnrlon=np.min(lon),urcrnrlon=np.max(lon),lat_ts=20,resolution='h')


p1=plt.subplot(2,2,1)  
cmap = cm.get_cmap('gist_rainbow', 6)
APPOC42 = np.ma.masked_where(np.isnan(APPOC4),APPOC4)
im1 = m.contourf(lon,lat,APPOC42,shading='flat', latlon=True, vmin=0.5, vmax=6.5, cmap=cmap, levels=[0.5,1.5,2.5, 3.5, 4.5, 5.5, 6.5] )
m.drawcoastlines()
m.fillcontinents(color='grey')
cb = m.colorbar(im1,"bottom", size="5%", pad="2%", ticks=[1,2,3,4,5,6])
cb.ax.set_xticklabels(labels=["OC4_OK", "high SPM", "high CDOM", "high CDOM&SPM", "high CHL", "low CDOM"], rotation=-45)
plt.title(patterns+" / OC4 / POLYMER")


#####  plot 2
####################

p2=plt.subplot(2,2,2)
cmap = cm.get_cmap('gist_rainbow', 7)
APPOC52 = np.ma.masked_where(np.isnan(APPOC5),APPOC5)
im1 = m.contourf(lon,lat,APPOC52,shading='flat', latlon=True, vmin=0.5, vmax=7.5, cmap=cmap, levels=[0.5,1.5,2.5, 3.5, 4.5, 5.5, 6.5, 7.5] )
m.drawcoastlines()
m.fillcontinents(color='grey')
cb = m.colorbar(im1,"bottom", size="5%", pad="2%", ticks=[1,2,3,4,5,6, 7])
cb.ax.set_xticklabels(labels=["OC5_OK", "OC5_still_OK", "high SPM", "high CDOM", "high CDOM&SPM", "high CHL", "low CDOM"], rotation=-45)
plt.title(patterns+" / OC5 / POLYMER")


#####  plot 3
####################

p3=plt.subplot(2,2,3)
cmap = cm.get_cmap('Spectral', 4)
APPGONS2 = np.ma.masked_where(np.isnan(APPGONS),APPGONS)
im1 = m.contourf(lon,lat,APPGONS2,shading='flat', latlon=True, vmin=0.5, vmax=4.5, cmap=cmap, levels=[0.5,1.5,2.5, 3.5, 4.5] )
m.drawcoastlines()
m.fillcontinents(color='grey')
cb = m.colorbar(im1,"bottom", size="5%", pad="2%", ticks=[1,2,3,4])
cb.ax.set_xticklabels(labels=["GONS_OK", "low CHL", "low SPM", "very high SPM"], rotation=-45)
plt.title(patterns+" / NIR-red / POLYMER")


#####  plot 4
####################

p3=plt.subplot(2,2,4)
cmap = cm.get_cmap('viridis', 4)
WALGO2 = np.ma.masked_where(np.isnan(WALGO),WALGO)
im1 = m.contourf(lon,lat,WALGO2,shading='flat', latlon=True, vmin=0.5, vmax=4.5, cmap=cmap, levels=[0.5,1.5,2.5, 3.5, 4.5] )
m.drawcoastlines()
m.fillcontinents(color='grey')
cb = m.colorbar(im1,"bottom", size="5%", pad="2%", ticks=[1,2,3,4])
cb.ax.set_xticklabels(labels=["No algo", "OC4", "OC5", "GONS"], rotation=-45)
plt.title(patterns+" / best algo / POLYMER")


plt.savefig("/home/heloise/Documents/algo_CHL/APPLY_OLCI_2/OLCI_F3_res_polymer_"+patterns+".png")




##### PLOTS chlorophyll-a ###########################
##########################################################

chlmax1 = np.nanpercentile(CHL_OC4, 90)
chlmax2 = np.nanpercentile(CHL_OC5, 90)
chlmax3 = np.nanpercentile(CHL_GONS, 90)
chlmax = np.percentile(np.array([chlmax1, chlmax2, chlmax3]), 100)
chlmin = 0


## Plot CHL OC4
 
fig = plt.figure(figsize=(10, 10))

p1=plt.subplot(2,2,1)
CHL_OC4[CHL_OC4 >= chlmax] = chlmax
CHL_OC4[CHL_OC4 <= chlmin] = chlmin

CHL_OC42 = np.ma.masked_where(np.isnan(CHL_OC4),CHL_OC4)
im1 = m.contourf(lon,lat,CHL_OC42,shading='flat', latlon=True, vmin=0, vmax=chlmax, cmap='jet')
m.drawcoastlines()
m.fillcontinents(color='grey')
cb = m.colorbar(im1,"bottom", size="5%", pad="2%")
cb.set_label("CHL_OC4 (mg $m^{-3}$)")
plt.title("OC4 CHL") #+ file[0][65:80]


#####  plot 2
####################

p2=plt.subplot(2,2,2)

CHL_OC5[CHL_OC5 >= chlmax] = chlmax
CHL_OC5[CHL_OC5 <= chlmin] = chlmin

CHL_OC52 = np.ma.masked_where(np.isnan(CHL_OC5),CHL_OC5)
im1 = m.contourf(lon,lat,CHL_OC52,shading='flat', latlon=True, vmin=0, vmax=chlmax, cmap='jet')
m.drawcoastlines()
m.fillcontinents(color='grey')
cb = m.colorbar(im1,"bottom", size="5%", pad="2%")
cb.set_label("CHL_OC5 (mg $m^{-3}$)")
plt.title("OC5 CHL") #+ file[0][65:80]


#####  plot 3
####################

p3=plt.subplot(2,2,3)

CHL_GONS[CHL_GONS >= chlmax] = chlmax
CHL_GONS[CHL_GONS <= chlmin] = chlmin

CHL_GONS2 = np.ma.masked_where(np.isnan(CHL_GONS),CHL_GONS)
im1 = m.contourf(lon,lat,CHL_GONS2,shading='flat', latlon=True, vmin=0, vmax=chlmax, cmap='jet')
m.drawcoastlines()
m.fillcontinents(color='grey')
cb = m.colorbar(im1,"bottom", size="5%", pad="2%")
cb.set_label("CHL_NIR-red (mg $m^{-3}$)")
plt.title("NIR-red band ratio CHL") #+ file[0][65:80]


#####  plot 4
####################

APPOC5[np.where(APPOC5 == 2)] = 1
CHL_OC4[np.where(APPOC4 != 1)] = np.nan
CHL_OC5[np.where(APPOC5 != 1)] = np.nan
CHL_GONS[np.where(APPGONS != 1)] = np.nan
CHL3D =  np.array([CHL_OC4, CHL_OC5, CHL_GONS])
CHL = np.nanmean(CHL3D, axis=0)

CHL[CHL >= chlmax] = chlmax
CHL[CHL <= chlmin] = chlmin

p4=plt.subplot(2,2,4)

CHL2 = np.ma.masked_where(np.isnan(CHL),CHL)
im1 = m.contourf(lon,lat,CHL2,shading='flat', latlon=True, vmin=0, vmax=chlmax, cmap='jet')
m.drawcoastlines()
m.fillcontinents(color='grey')
cb = m.colorbar(im1,"bottom", size="5%", pad="2%")
cb.set_label("merged Chl-a (mg $m^{-3}$)")
plt.title("Merged Chl-a product") #+ file[0][65:80]


plt.savefig("/home/heloise/Documents/algo_CHL/APPLY_OLCI_2/OLCI_F3_chl_polymer_"+patterns+".png")



