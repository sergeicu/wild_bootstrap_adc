# Separate error calculation from bootstrap. 
# source: https://stats.stackexchange.com/questions/408651/intuitively-how-does-the-wild-bootstrap-work

import os
import argparse

import numpy as np
import nibabel as nb
import torch 

from geometric_averages import average

from computeLinearADC import computeLinearADC_torch_image_batch


def load_args():
    parser = argparse.ArgumentParser(description="Load arguments for processing 4D nifti files.")
    parser.add_argument("--nifti", help="4D nifti filename")
    parser.add_argument("--bvals", help="B-values filename")
    parser.add_argument("--seg", help="3D segmentation filename - adcs will only be computed over masked region")
    parser.add_argument("--savedir", help="Save directory")
    parser.add_argument("--iterations", type=int, help="Number of wild bootstrap iterations to perform. Total number of values will be numberOfAverages * numberOfIterations")

    args = parser.parse_args()

    return args
    
def load_bvals(bvalsfilename):
    
    assert os.path.exists(bvalsfilename)
    
    # Load bvals 
    fo = open( bvalsfilename,'r')
    lines = fo.readlines()
    fo.close()
    bvals = [ int(bb) for bb in lines[0].split(' ')]
    bvals = [round(bval/10)*10 for bval in bvals]
    
    return bvals

def predict_adc_signal(b0,adc,bvalues): 
    
    # make sure bvals are floats 
    bvalues = np.array(bvalues).astype(b0.dtype)
    
    # remove nans in both arrays
    b0_nans = np.isnan(b0)
    adc_nans = np.isnan(adc)
    nans=np.logical_or(b0_nans, adc_nans)
    b0[nans] = 0 
    adc[nans] = 0 
    
    # get signal estimate from IVIM parameter estimates
    b0 = b0[...,np.newaxis] #expand with one new axis so that b* parameter can be broadcast
    adc = adc[...,np.newaxis]
    x,y,z,t = b0.shape

    bvals = np.array(bvalues)
    bvals_ = np.tile(bvals,(x,y,z,1))
    
    signal_image = b0*(np.exp(-1*bvals_*adc))
    return signal_image
    


if __name__ == '__main__':
    
    
    args = load_args()
    niftipath = args.nifti
    bvalsfilename=args.bvals
    segfilename=args.seg
    savedir=args.savedir + "/"
    iterations = args.iterations
    

    ####################################
    # load data and params 
    ####################################
    
    # load bvals and bvecs    
    bvals = load_bvals(bvalsfilename)

    # Get params 
    numAllImages = len(bvals)
    numDirections = len(np.where(np.array(bvals)==np.unique(bvals)[1])[0])
    num2PermuteFrom = numDirections # total number of directions which we would be permuting from 
    nii_basename = os.path.basename(niftipath).replace(".nii.gz", "_") # assumes that files will be name as svr.nii.gz or vol.nii.gz

    # load nifti 
    assert os.path.exists(niftipath)
    imo = nb.load(niftipath)
    dwiData = imo.get_fdata()
    
    # load mask 
    assert os.path.exists(segfilename)
    seg = nb.load(segfilename).get_fdata()    
    assert dwiData.shape[:-1] == seg.shape

    
    ####################################
    # default prediction (no directions removed) 
    ####################################

    # compute averaged signal
    dwiData_averaged = average(dwiData,bvals)
    bvals_averaged = sorted(np.unique(bvals))   
        
    # init variables 
    x,y,z,t=dwiData_averaged.shape
    nz = np.nonzero(seg)
    L=len(nz[0])
    adc = np.zeros((x,y,z))
    b0 = np.zeros((x,y,z))

    # get segmented image 
    seg_4D = np.tile(np.expand_dims(seg,-1), (1,1,1,t))
    im_seg = np.zeros_like(dwiData_averaged)
    im_seg[seg_4D>0] = dwiData_averaged[seg_4D>0]
    im_seg_r = np.reshape(im_seg,(x*y,z,t))
    
    
    # compute default adc and b0
    adc_est, b0_est = computeLinearADC_torch_image_batch(bvals = bvals_averaged,signal = im_seg_r)
    adc = torch.reshape(adc_est,(x,y,z))
    b0 = torch.reshape(b0_est,(x,y,z))
    adc = adc.cpu()
    b0 = b0.cpu()
    b0_default,adc_default = b0.numpy(),adc.numpy()

    # predict DWI signal from estimated default parameters (no bootstrap)
    dwiPredict_default=predict_adc_signal(b0_default,adc_default,bvals_averaged)

    # calculate error for base case 
    dwiPredict = [dwiPredict_default]
    paramPredict = [np.array([b0_default, adc_default])]


    ####################################
    # WildBootstrap
    ####################################


    # find residuals 
    num_bvals = len(bvals_averaged)
    ids = list(range(0,len(bvals)))
    samples = []
    for direction in range(0,numDirections):
        myids = ids[direction::numDirections]
        samples.append(dwiData[:,:,:,myids])

    residuals = []
    for sample in samples:
        residuals.append(sample-dwiPredict_default)

    # random swap of residuals (with permutation)
    for i in range(iterations):
        print(i)
        for sample, residual in zip(samples,residuals):
            residual_ids = np.random.choice(len(residuals),num_bvals).tolist() # choose between residuals three times (for each bvalue)
            random_residuals = []
            for i in range(0,num_bvals):
                image=residuals[residual_ids[i]][:,:,:,i]
                random_residuals.append(image)
            random_residuals = np.moveaxis(np.array(random_residuals),0,-1)

            # equation 3 in https://stats.stackexchange.com/questions/408651/intuitively-how-does-the-wild-bootstrap-work
            #
            w=np.random.rand(x,y,z,num_bvals)[0,0,0,0]-1
            newsample = sample-residual+random_residuals*w

            # compute new adc 
            newsample_r = np.reshape(newsample,(x*y,z,t))
            adc_est, b0_est = computeLinearADC_torch_image_batch(bvals = bvals_averaged,signal = newsample_r)
            adc = torch.reshape(adc_est,(x,y,z))
            b0 = torch.reshape(b0_est,(x,y,z))
            adc = adc.cpu()
            b0 = b0.cpu()
            b0_i,adc_i = b0.numpy(),adc.numpy()
            paramPredict.append(np.array([b0_i, adc_i]))


    

    # save predicted nifti     
    print("Saving images")        
    assert os.path.exists(savedir)
    for i, paramPredict_i in enumerate(paramPredict):
        print(i)

        # unpack
        b0_i = paramPredict_i[0,:,:,:]
        adc_i = paramPredict_i[1,:,:,:]

        # predict DWI signal from estimated parameters
        dwiPredict_i=predict_adc_signal(b0_i,adc_i,bvals_averaged)                

        # create new nifti objects 
        imonew_dwi = nb.Nifti1Image(dwiPredict_i, affine=imo.affine,header=imo.header)
        imonew_b0 = nb.Nifti1Image(b0_i, affine=imo.affine,header=imo.header)
        imonew_adc = nb.Nifti1Image(adc_i, affine=imo.affine,header=imo.header)

        # create savenames
        if i==0:
            savename_dwi = savedir + nii_basename + "_nowildbootstrap.nii.gz"
            savename_b0 = savedir + "b0_" + nii_basename + "_nowildbootstrap.nii.gz"
            savename_adc = savedir + "adc_" + nii_basename + "_nowildbootstrap.nii.gz"
        else:
            savename_dwi = savedir + nii_basename + "_wildbootstrap_iter_" + str(i).zfill(2) + ".nii.gz"
            savename_b0 = savedir + "b0_" + nii_basename + "_wildbootstrap_iter_" + str(i).zfill(2) + ".nii.gz"
            savename_adc = savedir + "adc_" + nii_basename + "_wildbootstrap_iter_" + str(i).zfill(2) + ".nii.gz"
            
        # save to files
        nb.save(imonew_dwi,savename_dwi)
        nb.save(imonew_b0,savename_b0)
        nb.save(imonew_adc,savename_adc)
            
            
            
    # save original files 
    imo_og = nb.Nifti1Image(dwiData_averaged, affine=imo.affine,header=imo.header)
    savename_og = savedir + nii_basename + "_og_averaged.nii.gz"
    nb.save(imo_og,savename_og)


    imo_og = nb.Nifti1Image(dwiData, affine=imo.affine,header=imo.header)
    savename_og = savedir + nii_basename + "_og.nii.gz"
    nb.save(imo_og,savename_og)
            
    
    print("Done")
    
