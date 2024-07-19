To obtain the normalized correlation by noise ceiling.  

1. Run the `calculate_fmri_profile_corr.py` to obtain the raw profile correlation.  
2. Run the `calculate_fmri_profile_pattern.py` to obtain the raw pattern correlation.  

The normalized correlation can be obtained by dividing the raw correlation by noise ceiling (note that voxels whose noise ceiling below 99th percentile threshold should be excluded). See `normalized_profile_correlation.ipynb` as an example.  
