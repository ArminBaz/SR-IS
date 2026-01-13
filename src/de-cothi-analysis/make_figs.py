import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def create_bar_plot(means, colors, ylabel, xlabels, y_lim=None, std=None, title=None, save_path=None):
    """
    Bar plot for NHB Plotting

    Args:
        means (array) : mean of each bar to plot
        colors (list) : idx of color palette color to use
        ylabel (string) : label for the y-axis
        xlabels (list of strings) : labels for the x-axis
        std (array, optional) : std error of each bar
        title (string, optional) : title of the plot
        save_path (string, optional) : where to save the figure
    """

    color_palette = sns.color_palette("tab10")
    color_list = []
    for color in colors:
        color_list.append(color_palette[color])

    plt.rcParams['font.family'] = 'serif'
    
    fig, ax = plt.subplots(figsize=(4, 4))
    x = np.arange(len(means)) * 0.25
    
    plot_means = np.array(means, dtype=float).copy()
    min_visible_height = 0.2
    plot_means[plot_means < 1] = min_visible_height
    
    bars = ax.bar(x, plot_means, color=color_list, edgecolor='black', linewidth=1, width=0.14, alpha=0.8)
    
    if std is not None:
        ax.errorbar(x, means, yerr=std, fmt='none', color='black', capsize=0)
    
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=22) if title else None
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, rotation=0, ha='center', fontsize=18)
    
    # ax.set_yticks([0, y_lim[1]])
    ax.set_ylim(0, y_lim[1])
    ax.set_yticks(np.arange(0, y_lim[1], 2))
    
    # for spine in ['left', 'right', 'bottom', 'top']:
    for spine in ['right', 'top']:
        # ax.spines[spine].set_color('black')
        ax.spines[spine].set_linewidth(0)
    
    ax.grid(False)
    
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

mean_rats = [0, 0, 1, 8]
mean_humans = [0, 0, 5, 13]

    

if __name__ == '__main__':
    save_path_rats = os.path.join('figures/') + 'model_comparison_rats.png'
    save_path_humans = os.path.join('figures/') + 'model_comparison_humans.png'
    
    create_bar_plot(means=mean_rats, colors=[1, 3, 0, 2],
                    ylabel=None, xlabels=['MF', 'MB', 'SR', 'SR-IS'], 
                    y_lim=[0, 10], title="Rats", save_path=save_path_rats)
    create_bar_plot(means=mean_humans, colors=[1, 3, 0, 2],
                    ylabel='Num Participants', xlabels=['MF', 'MB', 'SR', 'SR-IS'], 
                    y_lim=[0, 15], title="Humans", save_path=save_path_humans)