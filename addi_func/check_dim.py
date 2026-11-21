import SimpleITK as sitk
import nibabel as nib
from numpy.ma.core import size

# img = r"/home/mengao/Code/4Dsegment/data/subject_18/segs/seg_lvsa_ED.nii.gz"
img = r"/home/mengao/Code/4Dsegment/data/subject_18/sizes/lvsa_ED.nii.gz"

mask_img = nib.load(img)
print(mask_img.shape)
# mask = mask_img.get_fdata()   # (x, y, z, t)
# print(shape(mask))