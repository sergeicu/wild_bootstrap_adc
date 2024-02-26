"""Usage:
    python computeLinearADC_image_torch.py <4Dnifti> <bvalfile> <savedir> ["cuda"] [3Dsegmentation]

    [] - optional arguments

"""

import os 
import sys
import time

import nibabel as nb 
import numpy as np
from tqdm import tqdm

import torch

##############################
# Compute for a single vector 
##############################
def lsq_svd_solver_torch(A,b): 
    """Solving least squares problem with SVD"""
    # Compute the SVD of A
    U,S,V = torch.linalg.svd(A)
    V = torch.transpose(V,0,1)
    # find number of nonzero singular value = rank(A)
    r = len(torch.nonzero(torch.diag(S)))
    Uhat = U[:,:r]
    Shat = torch.diag(S)    
    z = torch.matmul(torch.linalg.solve(Shat,torch.transpose(Uhat,0,1)), b)
    z = torch.unsqueeze(z,1)
    c = torch.matmul(V,z)
    return c

def computeLinearADC_torch(bvals = None,signal = None): 

    # ****************** FOR 3 or more independent bvalues (siemens) *********************
    # linear model
    bvals = torch.tensor(-1)*torch.Tensor(bvals)
    A = torch.vstack([bvals,torch.ones((len(bvals)))])
    A = torch.moveaxis(A,0,-1)
    signal = torch.Tensor(signal)
    assert signal.ndim == 1
    b=torch.log(signal)
    x = lsq_svd_solver_torch(A,b)
    initial_adc = torch.max(x[0][0],torch.Tensor([0]))
    initial_b0 = torch.exp(torch.amin(torch.Tensor([x[1][0], torch.Tensor([6.5])])))
    return initial_adc,initial_b0

##############################
# Compute for a 2D image 
##############################

def lsq_svd_solver_torch_image(A,b): 
    """Solving least squares problem with SVD"""
    # Compute the SVD of A
    U,S,V = torch.linalg.svd(A)
    V = torch.transpose(V,0,1)
    # find number of nonzero singular value = rank(A)
    r = len(torch.nonzero(torch.diag(S)))
    Uhat = U[:,:r]
    Shat = torch.diag(S)    
    
    Shat_x_Uhat = torch.linalg.solve(Shat,torch.transpose(Uhat,0,1))
    bt=torch.moveaxis(b,0,1)
    zt = torch.matmul(Shat_x_Uhat, bt)
    z = torch.moveaxis(zt,0,1)
    c = torch.matmul(V,zt)
    return c

def computeLinearADC_torch_image(bvals = None,signal = None): 
    
    # ****************** FOR 3 or more independent bvalues (siemens) *********************
    # linear model
    bvals = torch.tensor(-1)*torch.Tensor(bvals)
    A = torch.vstack([bvals,torch.ones((len(bvals)))])
    A = torch.moveaxis(A,0,-1)
    signal = torch.Tensor(signal)
    assert signal.ndim == 2, f"First dim of signal should be batch axis"
    assert signal.shape[1] == bvals.shape[0], f"columns of signal should equal to rows of bvals"
    b=torch.log(signal)
    x = lsq_svd_solver_torch_image(A,b)
    # adc 
    zeros=torch.zeros(x.shape[1])
    adcs=x[0,:]
    initial_adc = torch.max(torch.vstack([adcs,zeros]),axis=0)
    # b0s 
    bthresh=torch.Tensor([6.5]*x.shape[1])
    b0s=x[1,:]
    initial_b0_log = torch.amin(torch.vstack([b0s,bthresh]),axis=0)
    initial_b0 = torch.exp(initial_b0_log)
    return initial_adc[0],initial_b0

##############################
# Compute for a batch of 2D images 
##############################

def lsq_svd_solver_torch_image_batch(A,b): 
    assert A.ndim == 2 
    assert b.ndim == 3 
    """Solving least squares problem with SVD"""
    # Compute the SVD of A
    U,S,V = torch.linalg.svd(A)
    V = torch.transpose(V,0,1)
    # find number of nonzero singular value = rank(A)
    r = len(torch.nonzero(torch.diag(S)))
    Uhat = U[:,:r]
    Shat = torch.diag(S)    
    
    Shat_x_Uhat = torch.linalg.solve(Shat,torch.transpose(Uhat,0,1))
    x,y,z=b.shape
    bt=torch.moveaxis(b.reshape(x*y,z),-1,0)
    zt = torch.matmul(Shat_x_Uhat, bt)
    c = torch.matmul(V,zt)
    c = c.reshape((-1,x,y))
    return c

def computeLinearADC_torch_image_batch(bvals = None,signal = None): 
    
    # ****************** FOR 3 or more independent bvalues (siemens) *********************
    # linear model
    bvals = torch.tensor(-1)*torch.Tensor(bvals)
    A = torch.vstack([bvals,torch.ones((len(bvals)))])
    A = torch.moveaxis(A,0,-1)
    signal = torch.Tensor(signal)
    assert signal.ndim == 3, f"First dim of signal should be batch axis"
    assert signal.shape[-1] == bvals.shape[0], f"columns of signal should equal to rows of bvals"
    b=torch.log(signal)
    x = lsq_svd_solver_torch_image_batch(A,b)

    # adc 
    _,xx,yy = x.shape
    zeros=torch.zeros((xx,yy))
    adcs=x[0,:,:]
    vstack = torch.vstack([adcs[None,:,:],zeros[None,:,:]])
    initial_adc = torch.max(vstack,axis=0)

    # b0s 
    bthresh = torch.tile(torch.Tensor([6.5]),(xx,yy))
    b0s=x[1,:,:]
    vstack2=torch.vstack([b0s[None,:,:],bthresh[None,:,:]])
    initial_b0_log = torch.amin(vstack2,axis=0)
    initial_b0 = torch.exp(initial_b0_log)
    return initial_adc[0],initial_b0

##############################
# Compute for a batch of 2D images, via cuda 
##############################

def lsq_svd_solver_torch_image_batch_cuda(A,b): 
    """Solving least squares problem with SVD"""
    # Compute the SVD of A
    U,S,V = torch.linalg.svd(A)
    V = torch.transpose(V,0,1)
    # find number of nonzero singular value = rank(A)
    r = len(torch.nonzero(torch.diag(S)))
    Uhat = U[:,:r]
    Shat = torch.diag(S)    
    
    Shat_x_Uhat = torch.linalg.solve(Shat,torch.transpose(Uhat,0,1))
    x,y,z=b.shape
    bt=torch.moveaxis(b.reshape(x*y,z),-1,0)
    zt = torch.matmul(Shat_x_Uhat, bt)
    c = torch.matmul(V,zt)
    c = c.reshape((-1,x,y))
    return c

def computeLinearADC_torch_image_batch_cuda(bvals = None,signal = None): 
    
    # ****************** FOR 3 or more independent bvalues (siemens) *********************
    # linear model
    #bvals = torch.tensor(-1).to("cuda")*torch.Tensor(bvals).to("cuda")
    bvals = torch.tensor(-1)*torch.Tensor(bvals)
    A = torch.vstack([bvals,torch.ones((len(bvals)))]).to("cuda")
    A = torch.moveaxis(A,0,-1)
    signal = torch.Tensor(signal).to("cuda")
    assert signal.ndim == 3, f"First dim of signal should be batch axis"
    assert signal.shape[-1] == bvals.shape[0], f"columns of signal should equal to rows of bvals"
    b=torch.log(signal)
    x = lsq_svd_solver_torch_image_batch(A,b)

    # adc 
    _,xx,yy = x.shape
    zeros=torch.zeros((xx,yy)).to("cuda")
    adcs=x[0,:,:]
    vstack = torch.vstack([adcs[None,:,:],zeros[None,:,:]])
    initial_adc = torch.max(vstack,axis=0)

    # b0s 
    bthresh = torch.tile(torch.Tensor([6.5]),(xx,yy)).to("cuda")
    b0s=x[1,:,:]
    vstack2=torch.vstack([b0s[None,:,:],bthresh[None,:,:]])
    initial_b0_log = torch.amin(vstack2,axis=0)
    initial_b0 = torch.exp(initial_b0_log)
    return initial_adc[0],initial_b0


    
def read_nifti(nifti4D):
    imo=nb.load(nifti4D)
    x,y,z,t=imo.shape
    im = imo.get_fdata()
    assert im.ndim == 4
    assert t > 2, f"We expect all b-values to be in the same place. Currently the 4th dimension is: {t}"
    
    return im,imo
    
    
def get_segmentation(im, segfile=None):    
    # load segmentation
    if segfile is not None:
        assert segfile.endswith('.nii.gz') or segfile.endswith('.nii')        
        sego=nb.load(segfile).get_fdata()    
    else:
        # get binary image
        x,y,z,t=im.shape
        seg=np.zeros((x,y,z))
        nz_ = np.nonzero(im)
        nz_ = (nz_[0], nz_[1], nz_[2])
        seg[nz_] = 1
        
    if seg.ndim == 3: 
        seg = np.tile(seg,(t,1,1,1))
        seg = np.moveaxis(seg,0,-1)
    assert seg.shape == im.shape, f"Segmentation and nifti4D do not match in shape. {seg.shape}:{im.shape}"

    im_seg = np.zeros_like(im)
    im_seg[seg>0] = im[seg>0]
    
    return im_seg 

def load_bvals(bvalsfilename):
    
    assert os.path.exists(bvalsfilename)
    
    # Load bvals 
    fo = open( bvalsfilename,'r')
    lines = fo.readlines()
    fo.close()
    bvals = [ int(bb) for bb in lines[0].split(' ')]
    bvals = [round(bval/10)*10 for bval in bvals]
    
    return np.array(bvals)


def read_bvals(bvalfile):
    # read bvalues
    with open(bvalfile,'r') as file:
        lines=file.readlines()    
    lines = lines[0][:-1]
    lines=lines.split(' ')
    bvals=sorted(np.unique([float(l) for l in lines]))
    print(f"Bvals are: {bvals}")

    return bvals 


##############################
# Main
##############################

if __name__ == '__main__':
    
    techniques = ['cuda', '2D_batch', '2D', '1D']
    nifti4D = sys.argv[1]
    bvalfile = sys.argv[2]
    savedir = sys.argv[3]
    technique = "cuda" if len(sys.argv)<5 else sys.argv[4]
    segfile = sys.argv[5] if len(sys.argv) > 5 else None
    assert technique in techniques, f"Available techniques are {techniques}"
    
    # asserts
    assert nifti4D.endswith('.nii.gz') or nifti4D.endswith('.nii')
    assert bvalfile.endswith('.bval') or 'bvals' in bvalfile
    assert os.path.exists(nifti4D), f"File does not eixst:{nifti4D}"
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)
        print(f"Creating new dir:{savedir}")
    savedir=savedir+'/'

    # read files 
    im,imo = read_nifti(nifti4D)    
    im_seg = get_segmentation(im, segfile=None)
    bvals = load_bvals(bvalfile)    
    assert list(bvals) == sorted(list(set(bvals))), f"\n\nPlease provide an nifti where each bvalue is averaged across repetitions, and a corresponding bvals file. \n i.e. bvals={sorted(list(set(bvals)))}"
    
    
    # init variables 
    nz = np.nonzero(im_seg)
    L=len(nz[0])
    x,y,z,t=im.shape
    adc = np.zeros((x,y,z))
    b0 = np.zeros((x,y,z))
    # cycle through nonzeros
    
    start=time.time() 



    # choose technique
    if technique == '1D':
        for c in tqdm(range(0,L)):
            #print(f"{c}/{L}")
            xx=nz[0][c]
            yy=nz[1][c]
            zz=nz[2][c]
            signali = im_seg[xx,yy,zz, :]
            [adc_est, b0_est] = computeLinearADC_torch(bvals = bvals,signal = signali)
            adc[xx,yy,zz] = adc_est.numpy()
            b0[xx,yy,zz] = b0_est.numpy()
    elif technique == '2D':
        #from IPython import embed; embed()
        signali = im_seg[nz[0],nz[1],nz[2], :]
        adc_est, b0_est = computeLinearADC_torch_image(bvals = bvals,signal = signali)
        adc[nz[0],nz[1],nz[2]] = adc_est
        b0[nz[0],nz[1],nz[2]] = b0_est
        
    elif technique == '2D_batch':
        
        im_seg_r = np.reshape(im_seg,(x*y,z,t))
        adc_est, b0_est = computeLinearADC_torch_image_batch(bvals = bvals,signal = im_seg_r)
        adc = torch.reshape(adc_est,(x,y,z))
        b0 = torch.reshape(b0_est,(x,y,z))

    elif technique == 'cuda':
        
        im_seg_r = np.reshape(im_seg,(x*y,z,t))
        adc_est, b0_est = computeLinearADC_torch_image_batch_cuda(bvals = bvals,signal = im_seg_r)
        adc = torch.reshape(adc_est,(x,y,z))
        b0 = torch.reshape(b0_est,(x,y,z))
        adc = adc.cpu()
        b0 = b0.cpu()
        
        
    timetaken=time.time()-start
    print(np.round(timetaken))
    # save image 
    adco=nb.Nifti1Image(adc,affine=imo.affine,header=imo.header)
    nb.save(adco, savedir+'adc.nii.gz')
    