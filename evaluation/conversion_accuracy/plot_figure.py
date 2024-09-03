import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def update_plot_params():
    """Update global plot parameters."""
    plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42})
    sns.set_style("ticks")

def load_and_concatenate_data(file1, file2):
    """Load and concatenate two CSV files."""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df = pd.concat([df1, df2]).reset_index(drop=True)
    return df

def load_nc_data_pattern(nc_file, nc_thd_file):
    """Load noise ceiling data and add identifiers for pattern analysis."""
    df_nc = pd.read_csv(nc_file)
    df_nc_thd = pd.read_csv(nc_thd_file)
    df_nc['Subject'] = df_nc['Subject'].str.replace('-', '')
    df_nc['Image index'] = df_nc['Image index'].astype(int)
    df_nc_thd['Image index'] = df_nc_thd['Image index'].astype(int)
    df_nc['identifier'] = df_nc['Subject'].values + df_nc['ROI'].values + df_nc['Image index'].values.astype(str)
    df_nc_thd['identifier'] = df_nc_thd['Subject'].values + df_nc_thd['ROI'].values + df_nc_thd['Image index'].values.astype(str)
    df_nc = df_nc.join(df_nc_thd.set_index('identifier'), rsuffix='_', on='identifier')
    return df_nc

def load_nc_data_profile(nc_file, nc_thd_file):
    """Load noise ceiling data and add identifiers for profile analysis."""
    df_nc = pd.read_csv(nc_file)
    df_nc_thd = pd.read_csv(nc_thd_file)
    df_nc['identifier'] = df_nc['Subject'].values + df_nc['ROI'].values + df_nc['Vox_idx'].values.astype(str)
    df_nc_thd['identifier'] = df_nc_thd['Subject'].values + df_nc_thd['ROI'].values + df_nc_thd['Vox_idx'].values.astype(str)
    df_nc = df_nc.join(df_nc_thd.set_index('identifier'), rsuffix='_', on='identifier')
    return df_nc

def merge_data_pattern(df, df_nc):
    """Merge the main dataset with noise ceiling data for pattern analysis."""
    df['identifier'] = df['Target'].values + df['ROI'].values + df['Image index'].values.astype(int).astype(str)
    df_plot = df.merge(df_nc, on='identifier', suffixes=('', '_nc'))
    return df_plot

def merge_data_profile(df, df_nc):
    """Merge the main dataset with noise ceiling data for profile analysis."""
    df['identifier'] = df['Target'].values + df['ROI'].values + df['Vox_idx'].values.astype(int).astype(str)
    df_plot = df.merge(df_nc, on='identifier', suffixes=('', '_nc'))
    return df_plot

def normalize_correlation(df):
    """Normalize correlation values and drop rows below the threshold."""
    df.drop(df.loc[df['Correlation_nc'] <= df['Threshold 1']].index, inplace=True)
    df['Normalized correlation'] = np.clip(np.nan_to_num(df['Correlation'].values / df['Correlation_nc'].values),
                                           a_min=-1, a_max=1)
    return df

def aggregate_results(df, result_label):
    """Aggregate results by ROI and Method."""
    result = df.groupby(['Source', 'Target', 'ROI', 'Method']).agg(
        **{result_label: ('Normalized correlation', 'mean')}).reset_index()
    mean_result = result.groupby(['ROI', 'Method']).agg(**{result_label: (result_label, 'mean')}).reset_index()
    return result, mean_result

def sort_and_order_results(result, mean_result, roi_order):
    """Sort and order results based on ROI and Method."""
    result['ROI'] = pd.Categorical(result['ROI'], categories=roi_order, ordered=True)
    result = result.sort_values(['ROI', 'Method']).reset_index(drop=True)

    mean_result['ROI'] = pd.Categorical(mean_result['ROI'], categories=roi_order, ordered=True)
    mean_result = mean_result.sort_values(['ROI', 'Method']).reset_index(drop=True)

    return result, mean_result

def plot_results(result, mean_result, output_dir, output_filename, ylabel, result_label, palette):
    """Plot the results and save the plot as a PDF."""
    sns.set()
    sns.set_theme(context="talk", style='ticks', font_scale=1.1)
    fig = plt.figure(figsize=(5.5, 3))

    g = sns.stripplot(data=result, x="ROI", y=result_label, size=3, hue='Method',
                      palette=palette, dodge=True, jitter=False, alpha=0)
    g.set(ylim=[0, 1])
    plt.xticks(rotation=0)
    g.legend_.remove()

    for collection in g.collections:
        for x, y in collection.get_offsets():
            g.plot(x, y, 'o', mec='grey', mew=1, mfc='none', ms=4)

    plt.ylabel(ylabel, fontsize=22)
    plt.vlines(x=0.5, ymin=0, ymax=6, color='grey', ls="-", alpha=0.5)

    unique_rois = mean_result['ROI'].unique()
    for index, roi in enumerate(unique_rois):
        method1_value = mean_result[(mean_result['ROI'] == roi) & (mean_result['Method'] == list(palette.keys())[0])][result_label].values[0]
        method2_value = mean_result[(mean_result['ROI'] == roi) & (mean_result['Method'] == list(palette.keys())[1])][result_label].values[0]
        plt.hlines(y=method1_value, xmin=index - 0.4, xmax=index, color='c')
        plt.hlines(y=method2_value, xmin=index, xmax=index + 0.4, color='lightcoral')

    sns.despine()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # plt.subplots_adjust(top=1.02, bottom=0.1, left=0.1, right=0.9)
    plt.savefig(os.path.join(output_dir, output_filename), bbox_inches='tight')


def run_analysis(df1_file, df2_file, nc_file, nc_thd_file, analysis_type, result_label, ylabel, output_dir, output_filename):
    """Run the analysis pipeline and generate plots."""
    # Load and concatenate data
    df = load_and_concatenate_data(df1_file, df2_file)

    # Extract unique method names from df1 and df2
    method1 = pd.read_csv(df1_file)['Method'].unique()[0]
    method2 = pd.read_csv(df2_file)['Method'].unique()[0]

    # Define the palette based on these method names
    palette = {method1: "grey", method2: "grey"}

    # Load noise ceiling data and merge based on analysis type
    if analysis_type == 'pattern':
        df_nc = load_nc_data_pattern(nc_file, nc_thd_file)
        df = merge_data_pattern(df, df_nc)
    elif analysis_type == 'profile':
        df_nc = load_nc_data_profile(nc_file, nc_thd_file)
        df = merge_data_profile(df, df_nc)

    # Normalize correlation
    df = normalize_correlation(df)

    # Aggregate and sort results
    result, mean_result = aggregate_results(df, result_label)
    roi_order = ['VC', 'V1', 'V2', 'V3', 'V4', 'HVC']
    result, mean_result = sort_and_order_results(result, mean_result, roi_order)

    # Plot and save results
    plot_results(result, mean_result, output_dir, output_filename, ylabel, result_label, palette)


def main():
    update_plot_params()

    # Pattern analysis (blue color for brain_loss, red color for content_loss)
    run_analysis(
        df1_file='./results/conversion_accuracy_pattern_brain_loss.csv',
        df2_file='./results/conversion_accuracy_pattern_content_loss.csv',
        nc_file='./results/pattern_noise_ceiling_single_trial.csv',
        nc_thd_file='./results/pattern_nc_threshold_single_trial.csv',
        analysis_type='pattern',
        result_label='pattern_mean',
        ylabel="Conversion accuracy\n(normalized pattern)",
        output_dir='./figure',
        output_filename='pattern.pdf'
    )

    # Profile analysis (blue color for brain_loss, red color for content_loss)
    run_analysis(
        df1_file='./results/conversion_accuracy_profile_brain_loss.csv',
        df2_file='./results/conversion_accuracy_profile_content_loss.csv',
        nc_file='./results/profile_noise_ceiling_single_trial.csv',
        nc_thd_file='./results/profile_nc_threshold_single_trial.csv',
        analysis_type='profile',
        result_label='profile_mean',
        ylabel="Conversion accuracy\n(normalized profile)",
        output_dir='./figure',
        output_filename='profile.pdf'
    )

if __name__ == "__main__":
    main()
