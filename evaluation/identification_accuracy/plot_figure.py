import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib as mpl
import os
from multiprocessing import Pool


def load_and_prepare_data(paths):
    """Load datasets and perform initial preprocessing"""
    df1 = pd.read_csv(paths['df1'])
    df2 = pd.read_csv(paths['df2'])
    df3 = pd.read_csv(paths['df3'])
    df4 = pd.read_csv(paths['df4'])
    df5 = pd.read_csv(paths['df5'])
    df6 = pd.read_csv(paths['df6'])

    # Set 'Layer' column to 'pixel' for pixel dataframes
    for df in [df1, df2, df3]:
        df['Layer'] = 'pixel'

    # Remove 'relu6' and 'relu7' layers from DNN dataframes
    df4_norelu = df4.drop(df4[(df4["Layer"] == "relu6") | (df4["Layer"] == "relu7")].index)
    df5_norelu = df5.drop(df5[(df5["Layer"] == "relu6") | (df5["Layer"] == "relu7")].index)
    df6_norelu = df6.drop(df6[(df6["Layer"] == "relu6") | (df6["Layer"] == "relu7")].index)

    return df1, df2, df3, df4_norelu, df5_norelu, df6_norelu


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

    identification_mean = combined_sample['Identification accuracy'].mean()
    return identification_mean


def perform_bootstrap(layer_data, unique_sources, unique_targets, n_iterations, sample_size):
    """Perform multiple bootstrap calculations for the specified layer"""
    identification_means = []

    with Pool() as pool:
        results = [pool.apply_async(bootstrap_iteration, args=(layer_data, unique_sources, unique_targets, sample_size))
                   for _ in range(n_iterations)]
        for r in results:
            identification_mean = r.get()
            identification_means.append(identification_mean)

    return identification_means


def calculate_bootstrap_confidence_intervals(data, n_iterations=1000, alpha=0.05, sample_size=20):
    """Calculate bootstrap confidence intervals for all layers"""
    results = {}

    unique_layers = data['Layer'].unique()

    for layer in unique_layers:
        layer_data = data[data['Layer'] == layer]
        unique_sources = layer_data['Source'].unique()
        unique_targets = layer_data['Target'].unique()

        identification_means = perform_bootstrap(layer_data, unique_sources, unique_targets, n_iterations, sample_size)

        if identification_means:
            identification_lower = np.percentile(identification_means, (alpha / 2) * 100)
            identification_upper = np.percentile(identification_means, (1 - alpha / 2) * 100)

            results[layer] = {
                'identification_mean': np.mean(identification_means),
                'identification_ci_lower': identification_lower,
                'identification_ci_upper': identification_upper
            }
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index': 'Layer'}, inplace=True)
    return results_df


def adjust_layer_positions(df1, df2_CI, df3_CI, df4_norelu, df5_CI, df6_CI):
    """Adjust layer positions for plotting"""
    df1["Layer"] = 0.8
    df2_CI["Layer"] = 1.2
    df3_CI["Layer"] = 1

    layer_list = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    a = 0
    for layer in layer_list:
        a += 1
        df4_norelu.loc[(df4_norelu["Layer"] == layer), "Layer"] = 0.8 + a
        df5_CI.loc[(df5_CI["Layer"] == layer), "Layer"] = 1.2 + a
        df6_CI.loc[(df6_CI["Layer"] == layer), "Layer"] = 1 + a

    return df1, df2_CI, df3_CI, df4_norelu, df5_CI, df6_CI


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


def plot_identification_accuracy(df1, df2_CI, df3_CI, df4_norelu, df5_CI, df6_CI):
    """Plot identification accuracy with calculated bootstrap confidence intervals"""
    sns.set_theme(context="poster", style='ticks', font="Helvetica", font_scale=1.25, rc={"lines.linewidth": 4})

    number = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    layer = ['Pixel', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # Plot the data with error bars
    sns.lineplot(data=df1, x="Layer", y="Identification accuracy", ax=ax1, err_style="bars", color='m', marker="v",
                 markersize=10)
    sns.lineplot(data=df2_CI, x="Layer", y="identification_mean", ax=ax1, err_style=None, color='c', marker='o',
                 markersize=10)
    add_errorbars(df2_CI, x="Layer", y='identification_mean', ci_lower='identification_ci_lower',
                  ci_upper='identification_ci_upper', ax=ax1, color='c',
                  err_kws={'elinewidth': 4, 'capsize': 0, 'capthick': 2})

    sns.lineplot(data=df3_CI, x="Layer", y="identification_mean", ax=ax1, err_style=None, color='lightcoral',
                 marker="s", markersize=10)
    add_errorbars(df3_CI, x="Layer", y='identification_mean', ci_lower='identification_ci_lower',
                  ci_upper='identification_ci_upper', ax=ax1, color='lightcoral',
                  err_kws={'elinewidth': 4, 'capsize': 0, 'capthick': 2})

    sns.lineplot(data=df4_norelu, x="Layer", y="Identification accuracy", ax=ax1, err_style="bars", color='m',
                 marker="v", markersize=10)
    sns.lineplot(data=df5_CI, x="Layer", y="identification_mean", ax=ax1, err_style=None, color='c', marker='o',
                 markersize=10)
    add_errorbars(df5_CI, x="Layer", y='identification_mean', ci_lower='identification_ci_lower',
                  ci_upper='identification_ci_upper', ax=ax1, color='c',
                  err_kws={'elinewidth': 4, 'capsize': 0, 'capthick': 2})

    sns.lineplot(data=df6_CI, x="Layer", y="identification_mean", ax=ax1, err_style=None, color='lightcoral',
                 marker="s", markersize=10)
    add_errorbars(df6_CI, x="Layer", y='identification_mean', ci_lower='identification_ci_lower',
                  ci_upper='identification_ci_upper', ax=ax1, color='lightcoral',
                  err_kws={'elinewidth': 4, 'capsize': 0, 'capthick': 2})

    ax1.set_xlabel('Feature', fontsize=34)
    ax1.set_ylabel('Identification accuracy', fontsize=34)

    for line in ax1.lines:
        line.set_markerfacecolor('none')
        line.set_markeredgewidth(4)
        line.set_markeredgecolor(line.get_color())

    ax1.axhline(0.5, c="grey", ls="--", alpha=0.5)
    ax1.axvline(1.5, c="grey", ls="-", alpha=0.5)
    plt.xticks(number, layer, fontsize=32)
    plt.ylim([0.4, 1.05])
    plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=32)
    plt.xticks(rotation=60)
    sns.despine()

    if not os.path.exists('./figure'):
        os.makedirs('./figure')
    plt.savefig('./figure/result.pdf', bbox_inches='tight')


def main():
    """Main function to execute the data loading, processing, and plotting"""
    # Define file paths
    paths = {
        'df1': './results/pixel_identification_within.csv',
        'df2': './results/pixel_identification_brain_loss.csv',
        'df3': './results/pixel_identification_content_loss.csv',
        'df4': './results/dnn_identification_within.csv',
        'df5': './results/dnn_identification_brain_loss.csv',
        'df6': './results/dnn_identification_content_loss.csv'
    }

    # Load and prepare data
    df1, df2, df3, df4, df5, df6 = load_and_prepare_data(paths)

    # Calculate bootstrap confidence intervals
    df2_CI = calculate_bootstrap_confidence_intervals(df2)
    df3_CI = calculate_bootstrap_confidence_intervals(df3)
    df5_CI = calculate_bootstrap_confidence_intervals(df5)
    df6_CI = calculate_bootstrap_confidence_intervals(df6)

    # Adjust layer positions for plotting
    df1, df2_CI, df3_CI, df4, df5_CI, df6_CI = adjust_layer_positions(df1, df2_CI, df3_CI, df4, df5_CI,
                                                                             df6_CI)

    # Plot identification accuracy
    plot_identification_accuracy(df1, df2_CI, df3_CI, df4, df5_CI, df6_CI)


if __name__ == "__main__":
    main()
