#!/bin/sh

for subj in {1..8}; do
    echo "processing subj $subj"
    # extract trial ID list
    python utils/extract_image_list.py --subj $subj --type trial
    python utils/extract_image_list.py --subj $subj --type cocoId

    # prepare brain voxels for encoding models:
    #   - extract cortical mask;
    #   - mask volume metric data;
    #   - zscore data by runs

    python utils/extract_cortical_voxels.py --zscore_by_run --subj $subj

    # extract ROI mask to apply on cortical data
    python utils/extract_cortical_voxels.py --subj $subj --mask_only --roi prf-eccrois
    python utils/extract_cortical_voxels.py --subj $subj --mask_only --roi prf-visualrois
    python utils/extract_cortical_voxels.py --subj $subj --mask_only --roi floc-faces
    python utils/extract_cortical_voxels.py --subj $subj --mask_only --roi floc-words
    python utils/extract_cortical_voxels.py --subj $subj --mask_only --roi floc-places
    python utils/extract_cortical_voxels.py --subj $subj --mask_only --roi floc-bodies
    python utils/extract_cortical_voxels.py --subj $subj --mask_only --roi Kastner2015
    python utils/extract_cortical_voxels.py --subj $subj --mask_only --roi HCP_MMP1