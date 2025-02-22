import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from joblib import Parallel, delayed


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
    df_nc_thd['identifier'] = df_nc_thd['Subject'].values + df_nc_thd['ROI'].values + df_nc_thd[
        'Image index'].values.astype(str)
    df_nc = df_nc.join(df_nc_thd.set_index('identifier'), rsuffix='_', on='identifier')
    return df_nc


def load_nc_data_profile(nc_file, nc_thd_file):
    """Load noise ceiling data and add identifiers for profile analysis."""
    df_nc = pd.read_csv(nc_file)
    df_nc_thd = pd.read_csv(nc_thd_file)
    df_nc['identifier'] = df_nc['Subject'].values + df_nc['ROI'].values + df_nc['Vox_idx'].values.astype(str)
    df_nc_thd['identifier'] = df_nc_thd['Subject'].values + df_nc_thd['ROI'].values + df_nc_thd[
        'Vox_idx'].values.astype(str)
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
    """Aggregate results by ROI and Method using bootstrap sampling."""
    result = df.groupby(['Source', 'Target', 'ROI', 'Method']).agg(
        **{result_label: ('Normalized correlation', 'mean')}
    ).reset_index()

    def bootstrap_sample(data, n_iterations, alpha, sample_size=20):
        """Bootstrap sampling"""
        means = []
        np.random.seed(42)

        unique_sources = data['Source'].unique()
        unique_targets = data['Target'].unique()

        for i in range(n_iterations):
            combined_sample = pd.DataFrame()

            while combined_sample.shape[0] < sample_size:
                temp_sources = []
                temp_targets = []

                while len(temp_sources) < sample_size:
                    source_sample = np.random.choice(unique_sources, 1, replace=True)[0]
                    target_sample = np.random.choice(unique_targets, 1, replace=True)[0]

                    if source_sample != target_sample:
                        temp_sources.append(source_sample)
                        temp_targets.append(target_sample)

                temp_sample = pd.DataFrame()
                for source, target in zip(temp_sources, temp_targets):
                    subset = data[(data['Source'] == source) & (data['Target'] == target)]
                    temp_sample = pd.concat([temp_sample, subset])
                combined_sample = pd.concat([combined_sample, temp_sample])
                combined_sample = combined_sample.head(sample_size)

            if not combined_sample.empty:
                mean_val = combined_sample[result_label].mean()
                means.append(mean_val)

        if means:
            lower = np.percentile(means, (alpha / 2) * 100)
            upper = np.percentile(means, (1 - alpha / 2) * 100)
            return np.mean(means), lower, upper
        else:
            return None, None, None

    def calculate_bootstrap_for_methods_rois(data, n_iterations=1000, alpha=0.05, sample_size=20, n_jobs=-1):
        """Apply bootstrap sampling to each ROI and Method combination using parallel processing."""
        grouped_data = list(data.groupby(['ROI', 'Method']))

        def process_group(roi, method, group):
            mean_val, lower, upper = bootstrap_sample(group, n_iterations, alpha, sample_size)
            if mean_val is not None:
                return {
                    'ROI': roi,
                    'Method': method,
                    'bootstrap_mean': mean_val,
                    'ci_lower': lower,
                    'ci_upper': upper,
                    'ci_width': upper - mean_val
                }
            else:
                return None

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_group)(roi, method, group) for (roi, method), group in grouped_data
        )
        results = [res for res in results if res is not None]
        return pd.DataFrame(results)

    bootstrap_results = calculate_bootstrap_for_methods_rois(result, n_iterations=1000, alpha=0.05, sample_size=20)
    return result, bootstrap_results


def sort_and_order_results(result, bootstrap_results, roi_order):
    """Sort and order results based on ROI and Method."""
    result['ROI'] = pd.Categorical(result['ROI'], categories=roi_order, ordered=True)
    result = result.sort_values(['ROI', 'Method']).reset_index(drop=True)

    bootstrap_results['ROI'] = pd.Categorical(bootstrap_results['ROI'], categories=roi_order, ordered=True)
    bootstrap_results = bootstrap_results.sort_values(['ROI', 'Method']).reset_index(drop=True)
    return result, bootstrap_results


def plot_results(result, bootstrap_results, output_dir, output_filename, ylabel, result_label, palette):
    """Plot the results and save the plot"""
    sns.set()
    sns.set_theme(context="talk", style='ticks', font_scale=1.1)
    fig = plt.figure(figsize=(5.5, 3))  # Specify figure name and canvas size

    g = sns.stripplot(data=result, x="ROI", y=result_label, size=3, hue='Method',
                      palette=palette, dodge=True, jitter=False, alpha=0)
    g.set(ylim=[0, 1])
    plt.xticks(rotation=0)
    g.legend_.remove()

    for collection in g.collections:
        for x, y in collection.get_offsets():
            g.plot(x, y, 'o', mec='grey', mew=1, mfc='none', ms=4, zorder=1)

    plt.vlines(x=0.5, ymin=0, ymax=1, color='grey', ls="-", alpha=0.5)

    # Dynamic extraction of method names from palette
    method1_name = list(palette.keys())[0]
    method2_name = list(palette.keys())[1]

    unique_rois = bootstrap_results['ROI'].unique()
    for index, roi in enumerate(unique_rois):
        method1_value = bootstrap_results[(bootstrap_results['ROI'] == roi) & (bootstrap_results['Method'] == method1_name)][
            'bootstrap_mean'].values[0]
        method1_ci_lower = bootstrap_results[(bootstrap_results['ROI'] == roi) & (bootstrap_results['Method'] == method1_name)][
            'ci_lower'].values[0]
        method1_ci_upper = bootstrap_results[(bootstrap_results['ROI'] == roi) & (bootstrap_results['Method'] == method1_name)][
            'ci_upper'].values[0]

        method2_value = bootstrap_results[(bootstrap_results['ROI'] == roi) & (bootstrap_results['Method'] == method2_name)][
            'bootstrap_mean'].values[0]
        method2_ci_lower = bootstrap_results[(bootstrap_results['ROI'] == roi) & (bootstrap_results['Method'] == method2_name)][
            'ci_lower'].values[0]
        method2_ci_upper = bootstrap_results[(bootstrap_results['ROI'] == roi) & (bootstrap_results['Method'] == method2_name)][
            'ci_upper'].values[0]

        plt.hlines(y=method1_value, xmin=index - 0.4, xmax=index, color='c')
        plt.hlines(y=method2_value, xmin=index, xmax=index + 0.4, color='lightcoral')

        plt.vlines(x=index - 0.2, ymin=method1_ci_lower, ymax=method1_ci_upper, color='c')
        plt.vlines(x=index + 0.2, ymin=method2_ci_lower, ymax=method2_ci_upper, color='lightcoral')

    plt.ylabel(ylabel, fontsize=22)

    sns.despine()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, output_filename), bbox_inches='tight')


def run_analysis(df1_file, df2_file, nc_file, nc_thd_file, analysis_type, result_label, ylabel, output_dir,
                 output_filename):
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
    result, bootstrap_results = aggregate_results(df, result_label)
    roi_order = ['VC', 'V1', 'V2', 'V3', 'V4', 'HVC']
    result, bootstrap_results = sort_and_order_results(result, bootstrap_results, roi_order)

    # Plot and save results
    plot_results(result, bootstrap_results, output_dir, output_filename, ylabel, result_label, palette)


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
        ylabel="Conversion accuracy\n(pattern, normalized)",
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
        ylabel="Conversion accuracy\n(profile, normalized )",
        output_dir='./figure',
        output_filename='profile.pdf'
    )


if __name__ == "__main__":
    main()
