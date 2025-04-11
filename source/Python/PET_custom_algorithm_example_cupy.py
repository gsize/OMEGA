# -*- coding: utf-8 -*-
"""
## Python codes for PET custom algorithm reconstruction
This example contains a simplified example for custom algorithm
reconstruction using sinogram PET data. Currently the support for
some of the additional features is limited. The default configuration
uses MLEM, but OSEM with subsets is also available.

Note that custom algorithm refers to your own algorithms and not the built-in 
algorithms. This example merely has the MLEM/OSEM algorithm shown as an example.
The forward and/or backward projections of OMEGA are utilized for the computation
of these algorithms.

This example uses PyTorch and CuPy and thus requires CUDA (and CuPy and PyTorch)!
"""
import sys
sys.path.append("./")

import numpy as np
import cupy as cp
from omegatomo.projector.indices import indexMakerSinoMuMap
from omegatomo.projector.detcoord import detectorCoordinates
from omegatomo import proj_PET
import matplotlib.pyplot as plt

#%%
options = proj_PET.projectorClass()

###########################################################################
###########################################################################
###########################################################################
########################### SCANNER PROPERTIES ############################
###########################################################################
###########################################################################
###########################################################################
 
### R-sectors/modules/blocks/buckets in transaxial direction
options.blocks_per_ring = (32)

### R-sectors/modules/blocks/buckets in axial direction (i.e. number of physical
### scanner/crystal rings) 
# Multiplying this with the below cryst_per_block should equal the total
# number of crystal rings. options.
options.linear_multip = (4)

### R-sectors/modules/blocks/buckets in transaxial direction
# Required only if larger than 1
options.transaxial_multip = 2

### Number of detectors on the side of R-sector/block/module (transaxial
### direction)
# (e.g. 13 if 13x13, 20 if 20x10)
options.cryst_per_block = (8)

### Number of detectors on the side of R-sector/block/module (axial
### direction)
# (e.g. 13 if 13x13, 10 if 20x10)
options.cryst_per_block_axial = 8

### Crystal pitch/size in x- and y-directions (transaxial) (mm)
options.cr_p = 4.2

### Crystal pitch/size in z-direction (axial) (mm)
options.cr_pz = 4.2

### Ring diameter (distance between perpendicular detectors) (mm)
options.diameter = 718

### Number of pseudo rings between physical rings (use 0 or [] if none)
options.pseudot = 0

### Number of detectors per crystal ring (without pseudo detectors)
options.det_per_ring = options.blocks_per_ring * options.cryst_per_block * options.transaxial_multip

### Number of detectors per crystal ring (with pseudo detectors)
# If your scanner has a single pseudo detector on each (transaxial) side of
# the crystal block then simply add +1 inside the parenthesis (or uncomment
# the one below).
options.det_w_pseudo = options.det_per_ring
#options.det_w_pseudo = options.blocks_per_ring*(options.cryst_per_block + 1)

### Number of crystal rings
options.rings = options.linear_multip * options.cryst_per_block_axial

### Total number of detectors
options.detectors = options.det_per_ring*options.rings

### Scanner name
# Used for naming purposes (measurement data)
options.machine_name = 'Cylindrical_PET_example'
 
###########################################################################
###########################################################################
###########################################################################
###########################################################################


###########################################################################
###########################################################################
###########################################################################
########################### IMAGE PROPERTIES ##############################
###########################################################################
###########################################################################
###########################################################################
 
### Reconstructed image pixel count (X-direction)
# NOTE: Non-square image sizes (X- and Y-direction) may not work
options.Nx = 256

### Y-direction
options.Ny = 256

### Z-direction (number of slices) (axial)
options.Nz = options.rings*2

### Flip the image (in vertical direction)?
options.flip_image = False

### How much is the image rotated?
# You need to run the precompute phase again if you modify this
# NOTE: The rotation is done in the detector space (before reconstruction).
# This current setting is for systems whose detector blocks start from the
# right hand side when viewing the device from front.
# Positive values perform the rotation in clockwise direction
options.offangle = options.det_w_pseudo * (3/4)

imgPixelSize = options.cr_p/2.0

### Transaxial FOV size (mm), this is the length of the x (horizontal) side
# of the FOV
options.FOVa_x =  options.Nx*imgPixelSize 

### Transaxial FOV size (mm), this is the length of the y (vertical) side
# of the FOV
options.FOVa_y = options.FOVa_x

### Axial FOV (mm)
options.axial_fov =  options.Nz*imgPixelSize

 
###########################################################################
###########################################################################
###########################################################################
###########################################################################
 
 
 

###########################################################################
###########################################################################
###########################################################################
########################### SINOGRAM PROPERTIES ###########################
###########################################################################
###########################################################################
###########################################################################
 
### Span factor/axial compression
options.span = 1

### Maximum ring difference
options.ring_difference = options.rings - 1

### Number of radial positions (views) in sinogram
# You should primarily use the same number as the device uses.
# However, if that information is not available you can use ndist_max
# function to determine potential values (see help ndist_max for usage).
options.Ndist = 273

### Number of angles (tangential positions) in sinogram 
# This is the final amount after possible mashing, maximum allowed is the
# number of detectors per ring/2.
options.Nang = options.det_per_ring//2

### Specify the amount of sinograms contained on each segment
# (this should total the total number of sinograms).
# Currently this is computed automatically, but you can also manually
# specify the segment sizes.
options.segment_table = np.concatenate((np.array(options.rings*2-1,ndmin=1), np.arange(options.rings*2-1 - (options.span + 1), max(options.Nz - options.ring_difference*2, options.rings - options.ring_difference), -options.span*2)))
options.segment_table = np.insert(np.repeat(options.segment_table[1:], 2), 0, options.segment_table[0])

### Total number of sinograms
options.TotSinos = np.sum(options.segment_table)

if options.span ==1:
    options.TotSinos = options.rings*options.rings 

### Number of sinograms used in reconstruction
# The first NSinos sinograms will be used for the image reconstruction.
options.NSinos = options.TotSinos
 
###########################################################################
###########################################################################
###########################################################################
###########################################################################


######################## Normalization correction #########################
### Apply normalization correction
# If set to True, normalization correction is applied to either data
# formation or in the image reconstruction by using precomputed 
# normalization coefficients. I.e. once you have computed the normalization
# coefficients, turn above compute_normalization to False and set this to
# True.
options.normalization_correction = False


######################### Attenuation correction ##########################
### Image-based attenuation correction
# Include attenuation correction from images (e.g. CT-images) (for this you
# need attenuation images of each slice correctly rotated and scaled for
# 511 keV). For CT-images you can use attenuationCT_to_511 or
# create_atten_matrix_CT functions.
options.attenuation_correction = False

options.rotateAttImage = 1

### Attenuation image data file
# Specify the path (if not in MATLAB path) and filename.
# NOTE: the attenuation data must be the only variable in the file and
# have the dimensions of the final reconstructed image.
# If no file is specified here, the user will be prompted to select one
options.attenuation_datafile = '/path/to/cylpet_example_atn1-MuMap.mhd'



###########################################################################
###########################################################################
###########################################################################
######################## RECONSTRUCTION PROPERTIES ########################
###########################################################################
###########################################################################
###########################################################################
 
############################### PROJECTOR #################################
### Type of projector to use for the geometric matrix
# 0 = Regular Siddon's algorithm (only available with implementation 1 and
# when precomputed_lor = False) NOT RECOMMENDED.
# 1 = Improved/accelerated Siddon's algorithm
# 2 = Orthogonal distance based ray tracer
# 3 = Volume of intersection based ray tracer
# See the doc for more information:
# https://omega-doc.readthedocs.io/en/latest/selectingprojector.html
options.projector_type = 1

### Use point spread function (PSF) blurring
# Applies PSF blurring through convolution to the image space. This is the
# same as multiplying the geometric matrix with an image blurring matrix.
options.use_psf = False

# FWHM of the Gaussian used in PSF blurring in all three dimensions
# options.FWHM = [options.cr_p options.cr_p options.cr_pz]
options.FWHM = np.array([options.cr_p, options.cr_p, options.cr_pz])

# Orthogonal ray tracer (projector_type = 2) only
### The 2D (XY) width of the "strip/tube" where the orthogonal distances are
# included. If tube_width_z below is non-zero, then this value is ignored.
options.tube_width_xy = options.cr_p

# Orthogonal ray tracer (projector_type = 2) only
### The 3D (Z) width of the "tube" where the orthogonal distances are
# included. If set to 0, then the 2D orthogonal ray tracer is used. If this
# value is non-zero then the above value is IGNORED.
options.tube_width_z = options.cr_pz

# Volume ray tracer (projector_type = 3) only
### Radius of the tube-of-response (cylinder)
# The radius of the cylinder that approximates the tube-of-response.
options.tube_radius = np.sqrt(2) * (options.cr_pz / 2)

# Volume ray tracer (projector_type = 3) only
### Relative size of the voxel (sphere)
# In volume ray tracer, the voxels are modeled as spheres. This value
# specifies the relative radius of the sphere such that with 1 the sphere
# is just large enoough to encompass an entire cubic voxel, i.e. the
# corners of the cubic voxel intersect with the sphere shell. Larger values
# create larger spheres, while smaller values create smaller spheres.
options.voxel_radius = 1

# Siddon (projector_type = 1) only
### Number of rays
# Number of rays used per detector if projector_type = 1 (i.e. Improved
# Siddon is used) and precompute_lor = False. I.e. when using precomputed
# LOR data, only 1 rays is always used.
# Number of rays in transaxial direction
options.n_rays_transaxial = 1
# Number of rays in axial direction
options.n_rays_axial = 1

# Interpolation length for projector_type 4
options.dL = 0.5
 
######################### RECONSTRUCTION SETTINGS #########################
### Number of iterations (all reconstruction methods)
options.Niter = 1

### Number of subsets (all excluding MLEM and subset_type = 6)
options.subsets = 32

### Subset type (n = subsets)
# 1 = Every nth (column) measurement is taken
# 2 = Every nth (row) measurement is taken (e.g. if subsets = 3, then
# first subset has measurements 1, 4, 7, etc., second 2, 5, 8, etc.) 
# 3 = Measurements are selected randomly
# 4 = (Sinogram only) Take every nth column in the sinogram
# 5 = (Sinogram only) Take every nth row in the sinogram
options.subsetType = 1

options.listmode =1
options.useIndexBasedReconstruction=1

# Assumes that PyTorch tensors are used as input to either forward or backward projections
options.useTorch = False

# Required for PyTorch
options.useCUDA = True

# Uses CuPy instead of PyCUDA (recommended)
options.useCuPy = True

# Compute forward projection with options * f
# Compute backprojection with options.T() * y



#%% make muMap
nX,nY,nZ = options.Nx,options.Ny,options.Nz
x=np.linspace(-(nX-1)*imgPixelSize/2,(nX-1)*imgPixelSize/2,nX)
y=x
z=np.linspace(-(nZ-1)*imgPixelSize/2,(nZ-1)*imgPixelSize/2,nZ)
X,Y,Z = np.meshgrid(x,y,z,indexing="ij")
muMap = np.zeros((nX,nY,nZ),dtype = np.float32)
#muMap[(X-60)**2+(Y-80)**2<=125**2] =1.0
muMap[(X)**2+(Y)**2<=(536/2.0)**2] =1.0

d_f = cp.asarray(muMap,dtype = cp.float32 ).ravel("F")

#plt.figure();plt.imshow(muMap[:,:,20])
#plt.figure();plt.scatter(detX,detY)

options.printInfor()
detX,detY =detectorCoordinates(options)
options.x = np.asfortranarray(np.vstack((detX,detY )))
z_length = float(options.rings) * options.cr_pz
options.z = np.linspace(-(z_length / 2 - options.cr_pz / 2), z_length / 2 - options.cr_pz / 2, options.rings,dtype=np.float32)
#plt.figure();plt.scatter(detX,detY)

indexMakerSinoMuMap(options)
options.projector_type =3
options.addProjector()
options.initProj()

#%% FP
#attSino =[]
#for k in range(options.subsets):
#    #k=1
#    print(k)
#    options.subset =k 
#    options.d_trIndex[0] = cp.asarray(options.trIndex[options.nMeas[k]*2:options.nMeas[k+1] *2])
#    options.d_axIndex[0] = cp.asarray(options.axIndex[options.nMeas[k]*2:options.nMeas[k+1] *2])
#
#    att = options.forwardProject(d_f, k)
#    mask  = np.isnan(att)
#    #att[mask]=0.0
#    attSino.append(cp.asnumpy(att))
#attSino = np.array(attSino).reshape(-1,options.Ndist,options.NSinos, order = 'C')
#attSino = np.transpose(attSino,(2,0,1))
##attSino.tofile(f"attSino_proj{options.projector_type}_Ndist{options.Nang}_Ndist{options.Ndist}_1.raw")
#
#plt.figure()
#plt.subplot(2,2,1)
#plt.imshow(attSino[0]);plt.colorbar()
#plt.title("sino_0")
#plt.subplot(2,2,2)
#plt.imshow(attSino[1]);plt.colorbar()
#plt.title("sino_1")
#plt.subplot(2,2,3)
#plt.imshow(attSino[-2]);plt.colorbar()
#plt.title("sino_-2")
#plt.subplot(2,2,4)
#plt.imshow(attSino[-1]);plt.colorbar()
#plt.title("sino_-1")
#plt.show()

imgBPs =[]
for k in range(options.subsets):
#for k in range(2):
    print(f"{k},",end="")
    options.subset =k 
    options.d_trIndex[0] = cp.asarray(options.trIndex[options.nMeas[k]*2:options.nMeas[k+1] *2])
    options.d_axIndex[0] = cp.asarray(options.axIndex[options.nMeas[k]*2:options.nMeas[k+1] *2])

    d_meas = cp.ones(options.nMeasSubset[k],dtype = np.float32)
    imgBP = options.backwardProject(d_meas, k)
    imgBPs.append(cp.asnumpy(imgBP))
print(f"end",end="\n")
lenBP =len(imgBPs)
Nx,Ny,Nz =  options.Nx[0].item(),options.Ny[0].item(),options.Nz[0].item()
imgBPs = np.array(imgBPs).reshape(-1,Nx,Ny,Nz, order = 'F')
imgBPs = np.transpose(imgBPs,(0,3,2,1))

imgBPs.tofile(f"BP_proj{options.projector_type}_{lenBP}X{Nz}X{Ny}X{Nx}.raw")

plt.figure()
plt.subplot(1,3,1)
plt.imshow(imgBPs[0][:,:,Nx//2]);plt.colorbar()
plt.title("img_0")
plt.subplot(1,3,2)
plt.imshow(imgBPs[0][:,Ny//2,:]);plt.colorbar()
plt.title("img_1")
plt.subplot(1,3,3)
plt.imshow(imgBPs[0][Nz//2,:,:]);plt.colorbar()
plt.title("img_3")
#plt.show()


"""
OSEM
"""
#d_m = [None] * options.subsets
#for k in range(options.subsets):
#    d_m[k] = torch.tensor(m[options.nTotMeas[k].item() : options.nTotMeas[k + 1].item()], device='cuda')
#for it in range(options.Niter):
#    for k in range(options.subsets):
#        # This is necessary when using subsets
#        # Alternative, call options.forwardProject(d_f, k) to use forward projection
#        # options.backwardProject(m, k) for backprojection
#        options.subset = k
#        fp = options * d_f
#        Sens = options.T() * torch.ones(d_m[k].numel(), dtype=torch.float32, device='cuda')
#        bp = options.T() * (d_m[k] / fp)
#        d_f = d_f / Sens * bp
#    
#f_np = d_f.cpu().numpy()
#f_np = np.reshape(f_np, (options.Nx[0].item(), options.Ny[0].item(), options.Nz[0].item()), order='F')
#plt.pyplot.imshow(f_np[:,:,20], vmin=0)
