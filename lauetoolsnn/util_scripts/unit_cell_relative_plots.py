# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 23:21:19 2022

@author: PURUSHOT
"""
import LaueTools.CrystalParameters as CP
import LaueTools.dict_LaueTools as dictLT

#% Compute lattice params from strain
# Plot the relative lattice parameters from a reference value
import numpy as np
import matplotlib.pyplot as plt
import os
import _pickle as cPickle
from mpl_toolkits.axes_grid1 import make_axes_locatable

folder = os.getcwd()
with open("results.pickle", "rb") as input_file:
    best_match, \
    mat_global, rotation_matrix1, strain_matrix, strain_matrixs,\
        col, colx, coly, match_rate, files_treated,\
            lim_x, lim_y, spots_len, iR_pix, fR_pix, material_, \
                material1_, lattice, lattice1, symmetry0, symmetry1,\
                    crystal, crystal1 = cPickle.load(input_file)
match_tol = 0
fR_tol = 10000
matnumber = 1
rangemin = -0.1
rangemax = 0.1
bins = 100
rangeval = len(match_rate)
material_id = [material_, material1_]


latticeparams = dictLT.dict_Materials["ZnCuOCl"][1]

strain_free_parameters = ['a']
additional_expression = ['a==b']

a,b,c,alp,bet,gam = [],[],[],[],[],[]

constantlength = "a"
if ("a" in strain_free_parameters) and ("b" in strain_free_parameters) and ("c" in strain_free_parameters):
    constantlength = "a"    
elif ("b" not in strain_free_parameters) and additional_expression[0]=="none" and \
    "b" not in additional_expression[0]:
    constantlength = "b"
elif ("c" not in strain_free_parameters):
    constantlength = "c"
    
for irot in range(len(rotation_matrix1[0][0])):
    lattice_parameter_direct_strain = CP.computeLatticeParameters_from_UB(rotation_matrix1[0][0][irot,:,:], 
                                                                          material_, 
                                                                          constantlength, 
                                                                          dictmaterials=dictLT.dict_Materials)
    a.append(lattice_parameter_direct_strain[0])
    b.append(lattice_parameter_direct_strain[1])
    c.append(lattice_parameter_direct_strain[2])
    alp.append(lattice_parameter_direct_strain[3])
    bet.append(lattice_parameter_direct_strain[4])
    gam.append(lattice_parameter_direct_strain[5])

logdata = np.array(a) - latticeparams[0]
logdata = logdata[~np.isnan(logdata)]
rangemina, rangemaxa = np.min(logdata)-0.01e-2, np.max(logdata)+0.01e-2
logdata = np.array(b) - latticeparams[1]
logdata = logdata[~np.isnan(logdata)]
rangeminb, rangemaxb = np.min(logdata)-0.01e-2, np.max(logdata)+0.01e-2
logdata = np.array(c) - latticeparams[2]
logdata = logdata[~np.isnan(logdata)]
rangeminc, rangemaxc = np.min(logdata)-0.01e-2, np.max(logdata)+0.01e-2
logdata = np.array(alp) - latticeparams[3]
logdata = logdata[~np.isnan(logdata)]
rangeminal, rangemaxal = np.min(logdata)-0.01, np.max(logdata)+0.01
logdata = np.array(bet) - latticeparams[4]
logdata = logdata[~np.isnan(logdata)]
rangeminbe, rangemaxbe = np.min(logdata)-0.01, np.max(logdata)+0.01
logdata = np.array(gam) - latticeparams[5]
logdata = logdata[~np.isnan(logdata)]
rangeminga, rangemaxga = np.min(logdata)-0.01, np.max(logdata)+0.01

fig = plt.figure(figsize=(11.69,8.27), dpi=100)
bottom, top = 0.1, 0.9
left, right = 0.1, 0.8
fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=0.15, wspace=0.25)

vmin = rangemina
vmax = rangemaxa
axs = fig.subplots(2, 3)
axs[0, 0].set_title(r"$a$", loc='center', fontsize=8)
strain_matrix_plot = np.array(a) - latticeparams[0]
im=axs[0, 0].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
axs[0, 0].set_xticks([])
axs[0, 0].set_yticks([])
divider = make_axes_locatable(axs[0,0])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=8) 

vmin = rangeminb
vmax = rangemaxb
axs[0, 1].set_title(r"$b$", loc='center', fontsize=8)
strain_matrix_plot = np.array(b) - latticeparams[1]
im=axs[0, 1].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
divider = make_axes_locatable(axs[0,1])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=8) 

vmin = rangeminc
vmax = rangemaxc
axs[0, 2].set_title(r"$c$", loc='center', fontsize=8)
strain_matrix_plot = np.array(c) - latticeparams[2]
im=axs[0, 2].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
divider = make_axes_locatable(axs[0,2])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=8) 

vmin = rangeminal
vmax = rangemaxal
axs[1, 0].set_title(r"$\alpha$", loc='center', fontsize=8)
strain_matrix_plot = np.array(alp) - latticeparams[3]
im=axs[1, 0].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
axs[1, 0].set_xticks([])
axs[1, 0].set_yticks([])
divider = make_axes_locatable(axs[1,0])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=8) 

vmin = rangeminbe
vmax = rangemaxbe
axs[1, 1].set_title(r"$\beta$", loc='center', fontsize=8)
strain_matrix_plot = np.array(bet) - latticeparams[4]
im=axs[1, 1].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
axs[1, 1].set_xticks([])
divider = make_axes_locatable(axs[1,1])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.ax.tick_params(labelsize=8) 

vmin = rangeminga
vmax = rangemaxga
axs[1, 2].set_title(r"$\gamma$", loc='center', fontsize=8)
strain_matrix_plot = np.array(gam) - latticeparams[5]
im = axs[1, 2].imshow(strain_matrix_plot.reshape((lim_x, lim_y)), origin='lower', cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
axs[1, 2].set_xticks([]) 
divider = make_axes_locatable(axs[1,2])
cax = divider.append_axes('right', size='5%', pad=0.05)
cbar = fig.colorbar(im, cax=cax, orientation='vertical')
cbar.formatter.set_useOffset(False)
cbar.ax.tick_params(labelsize=8) 

for ax in axs.flat:
    ax.label_outer()
plt.savefig(folder+ "//"+'figure_unitcell_relative.png', bbox_inches='tight',format='png', dpi=1000) 
plt.close(fig)