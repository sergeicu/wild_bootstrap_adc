import sys 
import os 

import nibabel as nb 
import numpy as np 

def average(im,bvals, mode='geometric'):
    
    # pair up all bvals 
    assert len(bvals) == im.shape[-1]
    bvals_unique = np.unique(bvals)
    x,y,z,t = im.shape
    final = np.zeros((x,y,z,len(bvals_unique)))
    for count, bval in enumerate(sorted(bvals_unique)):
        
        # find index 
        bvals_arr = np.array(bvals)
        indices = np.where(bvals_arr==bval)
        # select all indices that equal to current bval 
        im_bval = im[:,:,:,indices]
        im_bval = im_bval[:,:,:,0,:]
        if mode =='geometric':
            # calculate geometric average 
            product = np.prod(im_bval,axis=-1)
            power=1/len(indices[0])
            result = product**power
        else:
            # calculate arithmetic average 
            result=np.mean(im_bval,axis=-1)
        # save 
        final[:,:,:,count] = result
    
    return final


if __name__ == '__main__':
    
    """Geometrically averages files
    
    Usage: python geometric_averages_python.py <filepath> [geometric/arithmetic]
    
    Require: bval file to be named in the same way as filepath
    
    """
    
    
    nifti=sys.argv[1]
    assert nifti.endswith(".nii.gz") or nifti.endswith(".nii")
    if nifti.endswith(".nii.gz"):
        bvals_f=nifti.replace(".nii.gz", ".bval")
    
    assert os.path.exists(nifti)
    assert os.path.exists(bvals_f)

    # type 
    if len(sys.argv)>2:
        mode=sys.argv[2]
        modes=['geometric', 'arithmetic']
        assert mode in modes, f"Available modes are: {modes}"
    else:
        mode='geometric'

    # savesuffix 
    if mode =='geometric':
        savesuffix= "_gaveraged.nii.gz"
    else:
        savesuffix= "_averaged.nii.gz"

    # read nifti 
    imo = nb.load(nifti)
    im = imo.get_fdata()
    
    # read bvals 
    with open(bvals_f,'r') as f:
        lines = f.readlines()
    lines = lines[0]
    if lines.endswith('\n'):
        lines = lines[:-1]
    lines = lines.split(' ')
    bvals = [float(l) for l in lines]
    bvals_unique = np.unique(bvals)

    # geometric average
    final = average(im,bvals, mode=mode)
    
    # save the result 
    print(sorted(bvals_unique))
    newimo=nb.Nifti1Image(final,header=imo.header, affine=imo.affine)
    if nifti.endswith(".nii.gz"):
        savename = nifti.replace(".nii.gz", savesuffix)
        bvals_f_new = savename.replace(".nii.gz", ".bval")
    else:
        savename = nifti.replace(".nii", savesuffix)
        bvals_f_new = savename.replace(".nii", ".bval")
    nb.save(newimo, savename)
    
    # write bvals to file 
    mystring=bvals_unique.tolist()
    mystring=[str(int(i)) for i in mystring]
    mystring=" ".join(mystring)
    with open(bvals_f_new, 'w') as f:
        f.writelines(mystring)

