To calculate the raw correlation for conversion accuracy, use the following commands:

- For content loss-based converters:

  ```sh
  # pattern correlation
  python fmri_pattern_corr_content_loss.py
  
  # profile correlation
  python fmri_profile_corr_content_loss.py
  ```
  
- For brain loss-based converters:

  ```sh
  # pattern correlation
  python fmri_pattern_corr_brain_loss.py
  
  # profile correlation
  python fmri_profile_corr_brain_loss.py
  ```

To obtain the normalized correlation and plot the Figure 2E and 2F, use the following command.
```sh
python plot_figure.py
```
