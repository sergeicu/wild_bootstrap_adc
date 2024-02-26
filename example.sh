

# install (works on python 3.11, but may work on lower python versions too)
python -m venv venv 
source venv/bin/activate
pip install -r requirements.txt 

# run 
python wild_bootstrap_adc.py --nifti allb_mc.nii.gz --bvals allb_mc.bval --seg seg3.nii.gz --iterations 10 --savedir output