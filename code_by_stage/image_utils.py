import os
import shutil
import json
import subprocess
import tempfile
import numpy as np
import nibabel as nib
from numpy.linalg import inv

def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l + 1e-6)
    return image2


def validate_nonnegative_image(path):
    """Check saved MRI intensities before legacy integer conversion or inference."""
    data = np.asarray(nib.load(path).dataobj)
    if not data.size or not np.all(np.isfinite(data)):
        raise ValueError('Empty or nonfinite MRI intensities: ' + str(path))
    minimum, maximum = float(data.min()), float(data.max())
    if minimum < 0:
        raise ValueError('Unexpected negative MRI intensities (minimum {0:g}): {1}'.format(minimum, path))
    return {'minimum': minimum, 'maximum': maximum}


def ensure_temporal_int16_range(path):
    """Rescale only an unsafe working copy, before temporalalign casts to int16."""
    limits = validate_nonnegative_image(path)
    report = {'rescaled': False, 'input_range': limits,
              'int16_maximum': int(np.iinfo(np.int16).max),
              'method': 'unchanged_safe_input'}
    if limits['maximum'] <= np.iinfo(np.int16).max:
        return report
    image = nib.load(path)
    data = np.asarray(image.dataobj).astype(np.float64, copy=False)
    foreground = data > 0
    positive = data[foreground]
    low, high = np.percentile(positive, (1.0, 99.0))
    if high <= low:
        low, high = positive.min(), positive.max()
    output = np.zeros(data.shape, dtype=np.int16)
    if high > low:
        scaled = 1.0 + 4094.0 * (np.clip(positive, low, high) - low) / (high - low)
        output[foreground] = np.rint(scaled).astype(np.int16)
    else:
        output[foreground] = 4095
    converted = nib.Nifti1Image(output, image.affine, header=image.header.copy())
    converted.set_qform(image.get_qform(), code=int(image.header['qform_code']))
    converted.set_sform(image.get_sform(), code=int(image.header['sform_code']))
    converted.header.set_data_dtype(np.int16)
    converted.header.set_slope_inter(1, 0)
    temporary = tempfile.mkdtemp(prefix='.int16-safe-', dir=os.path.dirname(os.path.abspath(path)))
    try:
        candidate = os.path.join(temporary, os.path.basename(path))
        nib.save(converted, candidate)
        saved = nib.load(candidate)
        if (not np.array_equal(np.asarray(saved.dataobj), output)
                or not np.array_equal(saved.affine, image.affine)):
            raise RuntimeError('Safe-int16 output changed image data or geometry unexpectedly.')
        os.rename(candidate, path)
    finally:
        shutil.rmtree(temporary)
    report.update({'rescaled': True, 'method': 'nonzero_percentile_1_99_to_int16_1_4095',
                   'percentile_values': [float(low), float(high)],
                   'output_range': validate_nonnegative_image(path),
                   'zero_background_preserved': bool(np.array_equal(output == 0, data == 0)),
                   'shape_and_affine_preserved': True})
    return report


def imagePreprocessing(originalNii, data_dir, atlas_dir, native_frame_phases=False):
    working = os.path.join(data_dir, 'lvsa_.nii.gz')
    if os.path.realpath(originalNii) == os.path.realpath(working):
        raise ValueError('The original image must differ from the preprocessing working copy.')
    temporary = tempfile.mkdtemp(prefix='.native-phases-', dir=data_dir) if native_frame_phases else None
    try:
        reset_input = originalNii
        metadata = None
        if native_frame_phases:
            from nifti_working_copy import consistent_image, normalization_report
            original = nib.load(originalNii)
            normalized = consistent_image(np.asarray(original.dataobj), original)
            metadata = normalization_report(original, normalized)
            reset_input = os.path.join(temporary, 'scaled_original.nii.gz')
            nib.save(normalized, reset_input)
        subprocess.check_call(['headertool', reset_input, working, '-reset'])
        if native_frame_phases:
            reset = nib.load(working)
            if (len(original.shape) != 4 or original.shape != reset.shape
                    or not np.allclose(np.asarray(normalized.dataobj), np.asarray(reset.dataobj),
                                       rtol=1e-6, atol=1e-6)):
                raise RuntimeError('Header reset changed the native voxel layout or intensities; '
                                   'cannot align native segmentation by voxel index.')
        report = ensure_temporal_int16_range(working)
        report['original_image'] = os.path.abspath(originalNii)
        report['working_image'] = os.path.abspath(working)
        if metadata is not None:
            report['input_metadata_normalization'] = metadata
        with open(os.path.join(data_dir, 'lvsa_intensity_preprocessing.json'), 'w') as stream:
            json.dump(report, stream, indent=2)
            stream.write('\n')
        print('  Int16 safety: {0}'.format(report['method']))
        if native_frame_phases:
            native_path = os.path.join(temporary, 'native_cine.nii.gz')
            shutil.copyfile(working, native_path)
        subprocess.check_call(['temporalalign', os.path.join(atlas_dir, 'temporal.gipl'), working, working,
                               '-St1', '0', '-St2', '0', '-Et1', '1', '-Et2', '1'])
        validate_nonnegative_image(working)
        subprocess.check_call(['autocontrast', working, working])
        validate_nonnegative_image(working)
        phases = dict((frame, os.path.join(data_dir, 'lvsa_' + frame + '.nii.gz')) for frame in ('ED', 'ES'))
        command = ['cardiacphasedetection', working, phases['ED'], phases['ES']]
        if native_frame_phases:
            output = subprocess.check_output(command).decode('utf-8', 'replace')
            print(output)
            from native_phase_frames import rebuild_native_phases
            rebuild_native_phases(originalNii, native_path, working, phases, output,
                                  atlas_dir, temporary, data_dir)
        else:
            subprocess.check_call(command)
        for phase in phases.values():
            validate_nonnegative_image(phase)
    finally:
        if temporary is not None:
            shutil.rmtree(temporary)
    print('  Image preprocessing is done ...')


def clearBaseManbrance(data_dir, output_name):

    nim = nib.load('{0}/{1}'.format(data_dir, output_name))
    seg = nim.get_data()
    if seg.ndim == 4:
        seg = np.squeeze(seg, axis=-1).astype(np.int16)
        
    # find the last slice that has RV component 
    rv = (seg == 4).astype(np.int16)   
    for i in reversed(range(seg.shape[2])):
        if np.sum(rv[:,:,i]) > 0: 
            break
    lastAppearSlice = i 
    
    # create a 2D mask that average the last 12 slice of RV which is robust to noise
    slicesUsed = 12
    tmp = np.zeros((seg.shape[0], seg.shape[1]), dtype=np.int16)
    for j in range(slicesUsed):
        tmp = tmp + rv[:,:,lastAppearSlice-j]
    tmp = (tmp > 0).astype(np.int16)   
    
    # create a volume mask used to remove the base manbrance caused by RV myocardium
    mask = np.ones((seg.shape[2], seg.shape[0], seg.shape[1]), dtype=np.int16)
    mask[lastAppearSlice-slicesUsed:,:,:] = 1 - tmp
    mask = np.transpose(mask, axes=(1, 2, 0))
    seg = np.multiply(seg, mask)
    
    # fill the gap/hole of RV 
    flat = False
    if flat:
        seg = np.transpose(seg, axes=(2, 0, 1))
        seg[lastAppearSlice-slicesUsed:lastAppearSlice-3,:,:] = seg[lastAppearSlice-slicesUsed:lastAppearSlice-3,:,:] + 4*tmp
        seg = np.transpose(seg, axes=(1, 2, 0))
    else:
        seg[:,:,lastAppearSlice-slicesUsed:] = seg[:,:,lastAppearSlice-slicesUsed:] + 4*rv[:,:,lastAppearSlice-slicesUsed:]
    
    ###############################################################################
    # find the last slice that has LV component 
    lv = (seg == 1).astype(np.int16)   
    for i in reversed(range(seg.shape[2])):
        if np.sum(lv[:,:,i]) > 0: 
            break
    lastAppearSlice = i 
    
    # create a 2D mask that average the last 5 slice of RV which is robust to noise
    slicesUsed = 10
    tmp = np.zeros((seg.shape[0], seg.shape[1]), dtype=np.int16)
    for j in range(slicesUsed):
        tmp = tmp + lv[:,:,lastAppearSlice-j]
    tmp = (tmp > 0).astype(np.int16)   
    
    # create a volume mask used to remove the manbrance caused by LV myocardium
    mask = np.ones((seg.shape[2], seg.shape[0], seg.shape[1]), dtype=np.int16)
    mask[lastAppearSlice-slicesUsed:,:,:] = 1 - tmp
    mask = np.transpose(mask, axes=(1, 2, 0))
    seg = np.multiply(seg, mask)
    
    # fill the gap/hole of LV 
    flat = True
    if flat:
        seg = np.transpose(seg, axes=(2, 0, 1))
        seg[lastAppearSlice-slicesUsed:lastAppearSlice-1,:,:] = seg[lastAppearSlice-slicesUsed:lastAppearSlice-1,:,:] + tmp
        seg = np.transpose(seg, axes=(1, 2, 0))
    else:
        seg[:,:,lastAppearSlice-slicesUsed:] = seg[:,:,lastAppearSlice-slicesUsed:] + lv[:,:,lastAppearSlice-slicesUsed:]

    # save the result
    nim2 = nib.Nifti1Image(seg, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    nib.save(nim2, '{0}/{1}'.format(data_dir, output_name))


def removeSegsAboveBase(data_dir, output_name):

    # Read segmentations
    nim = nib.load('{0}/{1}'.format(data_dir, output_name))
    seg = nim.get_data()
    if seg.ndim == 4:
        seg = np.squeeze(seg, axis=-1).astype(np.int16)
    os.system('vtk2txt {0}/landmarks.vtk {0}/landmarks.txt'.format(data_dir)) 
    
    # convert txt file into matrix
    file = open('{0}/landmarks.txt'.format(data_dir), 'r') 
    A = file.read()
    tmp = np.zeros(18)  
    i, c = 0, 0
    for p in range(len(A)):
        if A[p] == ' ' or A[p] == '\n':
            tmp[i] = np.float32(A[c:p])
            i = i + 1
            c = p + 1;
    tmp = np.reshape(tmp,(6,3))        
    landmarks = np.ones((6,4)) 
    landmarks[:,:-1] = tmp
    landmarks = np.transpose(landmarks).astype(np.float32)           
    os.system('rm {0}/landmarks.txt'.format(data_dir))    
       
    # map world coordinate system to image pixel position
    pixelsPositions = np.ceil(np.dot(inv(nim.affine),landmarks)).astype(np.int16) 
    pixelsPositions = np.delete(pixelsPositions, (-1), axis=0)
    pixelsPositions = np.transpose(pixelsPositions).astype(np.int16) 
    
    if output_name[9:11]=='ED':
        seg[:,:,pixelsPositions[5,2]+1:] = 0
    else:
        seg[:,:,pixelsPositions[5,2]:] = 0
    
    nim2 = nib.Nifti1Image(seg, nim.affine)
    nim2.header['pixdim'] = nim.header['pixdim']
    nib.save(nim2, '{0}/{1}'.format(data_dir, output_name))
    

def formHighResolutionImg(subject_dir, fr): 
    
    os.system('resample ' 
              '{0}/lvsa_{1}.nii.gz '
              '{0}/lvsa_SR_{1}.nii.gz '
              '-size 1.25 1.25 2'
              .format(subject_dir, fr))
        
#    os.system('enlarge_image '
#              '{0}/lvsa_SR_{1}.nii.gz '
#              '{0}/lvsa_SR_{1}.nii.gz '
#              '-z 20 '
#              '-value 0'
#              .format(subject_dir, fr))

    
def convertImageSegment(data_dir, fr):
   
    os.system('convert '
              '{0}/seg_lvsa_SR_{1}.nii.gz '
              '{0}/PHsegmentation_{1}.gipl'
              .format(data_dir, fr))
    
    os.system('cp '
              '{0}/lvsa_SR_{1}.nii.gz '
              '{0}/lvsa_{1}_enlarged_SR.nii.gz'
              .format(data_dir, fr))
    
    
def outputVolumes(subject_dir, data_dir, subject, fr):
     
    os.system('rm '
              '{0}/{1}_{2}.txt'
              .format(data_dir, subject, fr))
    
    os.system('cardiacvolumecount '
              '{0}/PHsegmentation_{3}.gipl 1 '
              '-output '
              '{1}/{2}_{3}.txt'
              .format(subject_dir, data_dir, subject, fr))
    
    os.system('cardiacvolumecount '
              '{0}/PHsegmentation_{3}.gipl 2 '
              '-output '
              '{1}/{2}_{3}.txt '
              '-scale 1.05'
              .format(subject_dir, data_dir, subject, fr))
    
    os.system('cardiacvolumecount '
              '{0}/PHsegmentation_{3}.gipl 4 '
              '-output '
              '{1}/{2}_{3}.txt'
              .format(subject_dir, data_dir, subject, fr))
    
    os.system('cardiacvolumecount '
              '{0}/PHsegmentation_{3}.gipl 3 '
              '-output '
              '{1}/{2}_{3}.txt '
              '-scale 1.05'
              .format(subject_dir, data_dir, subject, fr))
    
    
def moveVolumes(subject_dir, sizes_dir, fr, segmentation_input=None):
     
    os.system('cp '
              '{0}/lvsa_{2}.nii.gz '
              '{1}/lvsa_{2}.nii.gz'
              .format(subject_dir, sizes_dir, fr))
    
    os.system('rm '
              '{0}/lvsa_{1}.nii.gz '
              .format(subject_dir, fr))
    
    # Archive the actual CNN/manual input without removing it from segs/.
    source = segmentation_input or os.path.join(subject_dir, 'segs', 'seg_lvsa_{0}.nii.gz'.format(fr))
    shutil.copyfile(source, os.path.join(sizes_dir, '2D_segmentation_{0}.nii.gz'.format(fr)))
    
    os.system('cp '
              '{0}/lvsa_SR_{2}.nii.gz '
              '{1}/lvsa_{2}_SR.nii.gz'
              .format(subject_dir, sizes_dir, fr))
    
    os.system('rm '
              '{0}/lvsa_SR_{1}.nii.gz '
              .format(subject_dir, fr))
    
    os.system('cp '
              '{0}/seg_lvsa_SR_{2}.nii.gz '
              '{1}/3D_segmentation_{2}.nii.gz'
              .format(subject_dir, sizes_dir, fr))
    
    os.system('rm '
              '{0}/seg_lvsa_SR_{1}.nii.gz '
              .format(subject_dir, fr))

    
def refineFusionResults(data_dir, output_name, alfa): 
    
    ##########################################
    os.system('binarize '
              '{0}/{1} '
              '{0}/tmps/hrt.nii.gz '
              '1 4 255 0'
              .format(data_dir, output_name))
    
    os.system('blur '
              '{0}/tmps/hrt.nii.gz '
              '{0}/tmps/hrt.nii.gz '
              '{1}'
              .format(data_dir, alfa))
    
    os.system('threshold '
              '{0}/tmps/hrt.nii.gz '
              '{0}/tmps/hrt.nii.gz '
              '130'
              .format(data_dir))
    
    ##########################################
    os.system('binarize '
              '{0}/{1} '
              '{0}/tmps/rvendo.nii.gz '
              '4 4 255 0'
              .format(data_dir, output_name))
    
    os.system('blur '
              '{0}/tmps/rvendo.nii.gz '
              '{0}/tmps/rvendo.nii.gz '
              '{1}'
              .format(data_dir, alfa))
    
    os.system('threshold '
              '{0}/tmps/rvendo.nii.gz '
              '{0}/tmps/rvendo.nii.gz '
              '130'
              .format(data_dir))
    
    ##########################################
    os.system('binarize '
              '{0}/{1} '
              '{0}/tmps/lvepi.nii.gz '
              '1 2 255 0'
              .format(data_dir, output_name))
       
    os.system('blur '
              '{0}/tmps/lvepi.nii.gz '
              '{0}/tmps/lvepi.nii.gz '
              '{1}'
              .format(data_dir, alfa))
    
    os.system('threshold '
              '{0}/tmps/lvepi.nii.gz '
              '{0}/tmps/lvepi.nii.gz '
              '115'
              .format(data_dir))
    
    ##########################################
    os.system('binarize '
              '{0}/{1} '
              '{0}/tmps/lvendo.nii.gz '
              '1 1 255 0'
              .format(data_dir, output_name))
    
    os.system('blur '
              '{0}/tmps/lvendo.nii.gz '
              '{0}/tmps/lvendo.nii.gz '
              '{1}'
              .format(data_dir, alfa))
    
    os.system('threshold '
              '{0}/tmps/lvendo.nii.gz '
              '{0}/tmps/lvendo.nii.gz '
              '130'
              .format(data_dir))
       
    ##########################################
    os.system('padding '
              '{0}/tmps/hrt.nii.gz '
              '{0}/tmps/hrt.nii.gz '
              '{0}/tmps/hrt.nii.gz '
              '1 3'
              .format(data_dir))
    
    os.system('padding '
              '{0}/tmps/hrt.nii.gz '
              '{0}/tmps/rvendo.nii.gz '
              '{0}/tmps/rvendo.nii.gz '
              '1 4'
              .format(data_dir))
    
    os.system('padding '
              '{0}/tmps/rvendo.nii.gz '
              '{0}/tmps/lvepi.nii.gz '
              '{0}/tmps/lvepi.nii.gz '
              '1 2'
              .format(data_dir))
    
    os.system('padding '
              '{0}/tmps/lvepi.nii.gz '
              '{0}/tmps/lvendo.nii.gz '
              '{0}/{1} '
              '1 1'
              .format(data_dir, output_name))
    
    
def allAtlasShapeSelection(dataset_dir):
   
    atlases_list, landmarks_list = {}, {}
    
    for fr in ['ED', 'ES']:
            
        atlases_list[fr], landmarks_list[fr] = [], [] 
        i = 0
        for atlas in sorted(os.listdir(dataset_dir)): 
            
            atlas_dir = os.path.join(dataset_dir, atlas)
            
            if not os.path.isdir(atlas_dir):
                
                print('  {0} is not a valid atlas directory, Discard'.format(atlas_dir))
                
                continue 
            
            atlas_3D_shape = '{0}/PHsegmentation_{1}.nii.gz'.format(atlas_dir, fr)
           
            landmarks = '{0}/landmarks.vtk'.format(atlas_dir)
        
            if i < 400:
                if os.path.exists(atlas_3D_shape) or os.path.exists(landmarks):
                    
                    atlases_list[fr] += [atlas_3D_shape]
                
                    landmarks_list[fr] += [landmarks]
            else:
                print(atlases_list)
                break
            
            i = i + 1
                 
    return atlases_list, landmarks_list


def topSimilarAtlasShapeSelection(atlases, atlas_landmarks, subject_landmarks, tmps_dir, dofs_dir, DLSeg, param_dir, topSimilarAltasNo):            
    
    landmarks = False
    
    nmi = []
    
    topSimilarAtlases_list = []
    
    atlasNo = len(atlases)
    
    os.system('rm {0}/shapenmi*.txt'.format(tmps_dir))
    
    for i in range(atlasNo):
        
        if landmarks:
            
            os.system('pareg '
                      '{0} '
                      '{1} '
                      '-dofout {2}/shapelandmarks_{3}.dof.gz'
                      .format(subject_landmarks, atlas_landmarks[i], dofs_dir, i))
        else: 
            
            os.system('areg '
                      '{0} '
                      '{1} '
                      '-dofout {2}/shapelandmarks_{4}.dof.gz '
                      '-parin {3}/segareg.txt'
                      .format(DLSeg, atlases[i], dofs_dir, param_dir, i))
                      
        os.system('cardiacimageevaluation '
                  '{0} '
                  '{1} '
                  '-nbins_x 64 '
                  '-nbins_y 64 '
                  '-dofin {2}/shapelandmarks_{4}.dof.gz '
                  '-output {3}/shapenmi_{4}.txt'
                  .format(DLSeg, atlases[i], dofs_dir, tmps_dir, i))
                        
        if os.path.exists('{0}/shapenmi_{1}.txt'.format(tmps_dir, i)):

            similarities = np.genfromtxt('{0}/shapenmi_{1}.txt'.format(tmps_dir, i))
            nmi += [similarities[3]]
            
        else:
            nmi += [0]
    
    if topSimilarAltasNo < atlasNo:
        
        sortedIndexes = np.array(nmi).argsort()[::-1]  
        
        savedInd = np.zeros(topSimilarAltasNo, dtype=int)
        
        for i in range(topSimilarAltasNo):
            
            topSimilarAtlases_list += [atlases[sortedIndexes[i]]]
            
            savedInd[i] = sortedIndexes[i]
    else:
        
        savedInd = np.zeros(atlasNo, dtype=int)
        
        topSimilarAtlases_list = atlases
        
        savedInd = np.arange(atlasNo)
        
    return topSimilarAtlases_list, savedInd    
 
