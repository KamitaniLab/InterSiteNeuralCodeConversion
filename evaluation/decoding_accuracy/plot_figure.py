import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
import os
from multiprocessing import Pool


def load_and_sort_data(file_path):
    """Load data and sort by specified layer order"""
    layer_order = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2',
                   'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
                   'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'fc6', 'fc7', 'fc8']

    df = pd.read_csv(file_path)
    df['Layer'] = pd.Categorical(df['Layer'], categories=layer_order, ordered=True)
    return df.sort_values('Layer')


def bootstrap_iteration(layer_data, unique_sources, unique_targets, sample_size):
    """Perform a single bootstrap iteration"""
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
            subset = layer_data[(layer_data['Source'] == source) & (layer_data['Target'] == target)]
            temp_sample = pd.concat([temp_sample, subset])

        combined_sample = pd.concat([combined_sample, temp_sample])
        combined_sample = combined_sample.head(sample_size)

    profile_mean = combined_sample['Profile correlation'].mean()
    pattern_mean = combined_sample['Pattern correlation'].mean()

    return profile_mean, pattern_mean


def perform_bootstrap(layer_data, unique_sources, unique_targets, n_iterations, sample_size):
    """Perform multiple bootstrap calculations for the specified layer"""
    profile_means = []
    pattern_means = []

    with Pool() as pool:
        results = [pool.apply_async(bootstrap_iteration, args=(layer_data, unique_sources, unique_targets, sample_size))
                   for _ in range(n_iterations)]
        for r in results:
            profile_mean, pattern_mean = r.get()
            profile_means.append(profile_mean)
            pattern_means.append(pattern_mean)

    return profile_means, pattern_means


def calculate_bootstrap_confidence_intervals(data, n_iterations=1000, alpha=0.05, sample_size=20):
    """Calculate bootstrap confidence intervals for all layers"""
    results = {}

    unique_layers = data['Layer'].unique()

    for layer in unique_layers:
        layer_data = data[data['Layer'] == layer]
        unique_sources = layer_data['Source'].unique()
        unique_targets = layer_data['Target'].unique()

        profile_means, pattern_means = perform_bootstrap(layer_data, unique_sources, unique_targets, n_iterations,
                                                         sample_size)

        if profile_means and pattern_means:
            profile_lower = np.percentile(profile_means, (alpha / 2) * 100)
            profile_upper = np.percentile(profile_means, (1 - alpha / 2) * 100)

            pattern_lower = np.percentile(pattern_means, (alpha / 2) * 100)
            pattern_upper = np.percentile(pattern_means, (1 - alpha / 2) * 100)

            results[layer] = {
                'profile_mean': np.mean(profile_means),
                'profile_ci_lower': profile_lower,
                'profile_ci_upper': profile_upper,
                'pattern_mean': np.mean(pattern_means),
                'pattern_ci_lower': pattern_lower,
                'pattern_ci_upper': pattern_upper
            }
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index': 'Layer'}, inplace=True)
    return results_df


def add_errorbars(data, x, y, ci_lower, ci_upper, ax, color, err_kws):
    """Add error bars to the plot"""
    error_param = {
        'yerr': (
            data[y] - data[ci_lower],
            data[ci_upper] - data[y]
        )
    }
    ebars = ax.errorbar(
        data[x], data[y], **error_param,
        linestyle="", color=color, **err_kws
    )

    # Set the cap style of error bars
    for obj in ebars.get_children():
        if isinstance(obj, mpl.collections.LineCollection):
            obj.set_capstyle('round')


def plot_decoding_accuracy(df1, df2_CI, df3_CI, measure_type):
    """Plot decoding accuracy graphs with calculated bootstrap confidence intervals"""
    ylabel = f"Decoding accuracy\n({measure_type})"
    filename = measure_type

    sns.set_theme(context="poster", style='ticks', font_scale=1.4)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)

    sns.lineplot(data=df2_CI, x="Layer", y=f"{measure_type}_mean", linewidth=3, err_style=None, color='c', marker="o",
                 markersize=8)
    add_errorbars(df2_CI, x="Layer", y=f"{measure_type}_mean", ci_lower=f"{measure_type}_ci_lower",
                  ci_upper=f"{measure_type}_ci_upper", ax=ax, color='c',
                  err_kws={'elinewidth': 3, 'capsize': 0, 'capthick': 2})

    sns.lineplot(data=df3_CI, x="Layer", y=f"{measure_type}_mean", linewidth=3, err_style=None, color='lightcoral',
                 marker="s", markersize=8)
    add_errorbars(df3_CI, x="Layer", y=f"{measure_type}_mean", ci_lower=f"{measure_type}_ci_lower",
                  ci_upper=f"{measure_type}_ci_upper", ax=ax, color='lightcoral',
                  err_kws={'elinewidth': 3, 'capsize': 0, 'capthick': 2})

    sns.lineplot(data=df1[(df1["ROI"] == 'VC')], x="Layer", y=f"{measure_type.capitalize()} correlation", linewidth=3,
                 err_style="bars", color='m', marker="v", markersize=8)

    for line in ax.lines:
        line.set_markerfacecolor('none')
        line.set_markeredgewidth(2)
        line.set_markeredgecolor(line.get_color())

    custom_xticks = ['1', '2', '1', '2', '1', '2', '3', '4', '1', '2', '3', '4', '1', '2', '3', '4', '6', '7', '8']
    plt.xticks(range(len(custom_xticks)), custom_xticks)

    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5], fontsize=28)

    sns.despine()

    plt.ylabel(ylabel, fontsize=30)
    plt.xlabel("Layer", fontsize=30)

    if not os.path.exists('./figure'):
        os.makedirs('./figure')
    plt.savefig(f'./figure/{filename}.pdf', bbox_inches='tight')


def main():
    # Load and sort data
    df1 = load_and_sort_data(
        './results/decoding_accuracy_within.csv')
    df2 = load_and_sort_data(
        '/results/decoding_accuracy_brain_loss.csv')
    df3 = load_and_sort_data(
        './results/decoding_accuracy_content_loss.csv')

    # Calculate bootstrap confidence intervals only once
    df2_CI = calculate_bootstrap_confidence_intervals(df2)
    df3_CI = calculate_bootstrap_confidence_intervals(df3)

    # Plot graphs with bootstrap confidence intervals for each measure type
    for measure_type in ['profile', 'pattern']:
        plot_decoding_accuracy(df1, df2_CI, df3_CI, measure_type)

if __name__ == "__main__":
    main()

