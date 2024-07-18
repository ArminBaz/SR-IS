import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.animation as manimation
import random

from utils import get_map


def render_maze(agent, state=None, locs=None, colors=None, ax=None, save_path=None, wall=None):
    """
    Renders the maze

    Args:
        agent (LinearRL class) : The agent
        state (tuple/array, Optional) : The state to draw the agent in, if None will use starting location
        locs (List of states, Optional) : Color specific locations (states)
        colors (List of color idxs, Optional) : The specific idx of colors to use from the colorblind color palette
        wall (list, Optional) : List containing two sublists for wall coordinates [[row1, row2], [col1, col2]]
    """
    if ax is None:
        fig, ax = plt.subplots()
    m = get_map(agent)

    if state is None:
        state = agent.start_loc

    # Define color palette
    color_palette = sns.color_palette("colorblind")
    
    # Display maze
    ax.imshow(m, origin='upper', cmap='gray_r')

    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(m), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(m), 1), minor=True)

    ax.grid(which="minor", color='black', linewidth=2, alpha=0.5)
    # Display agent
    agent_loc = patches.Circle((state[1],state[0]), radius=0.4, fill=True, color='blue', alpha=0.7)
    ax.add_patch(agent_loc)

    # Display Reward
    for i, target_loc in enumerate(agent.target_locs):
        # reward = patches.Circle((target_loc[1], target_loc[0]), radius=0.4, fill=True, color='green')
        reward = patches.Rectangle((target_loc[1] - 0.5, target_loc[0] - 0.5), 1.0, 1.0, fill=True, color='green', alpha=0.7)
        ax.text(target_loc[1], target_loc[0], f'r{i+1}', color='white', fontsize=10, ha='center', va='center')
        ax.add_patch(reward)

    # Color specific maze locations using Rectangle patches
    if locs is not None:
        for loc, color in zip(locs, colors):
            rect = patches.Rectangle((loc[1] - 0.5, loc[0] - 0.5), 1.0, 1.0, fill=True, color=color_palette[color])
            ax.text(loc[1], loc[0], f's{locs.index(loc) + 1}', color='white', fontsize=10, ha='center', va='center')
            ax.add_patch(rect)

    if wall is not None:
        print(f"Attempting to draw wall: {wall}")  # Debugging print
        [row, col], [direction] = wall
        if direction == 'h':  # Horizontal wall
            ax.plot([col - 0.5, col + 0.5], [row - 0.5, row - 0.5], color='red', linewidth=4, zorder=10)
        elif direction == 'v':  # Vertical wall
            ax.plot([col - 0.5, col - 0.5], [row - 0.5, row + 0.5], color='red', linewidth=4, zorder=10)

    # Hide tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Hide tick marks
    ax.tick_params(which='both', size=0)

    # Save the image
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_decision_prob(probs_train, probs_test, colors, leg_loc=None, save_path=None, title=None):
    """
    Plots the decision probability of going towards a terminal state

    Args:
        probs_train (array) : Probability of heading towards each terminal state before policy revaluation
        probs_test (array) : Probability of heading towrads each terminal state after policy revaluation
        colors (array) : idx of color pallette color to use 
        leg_loc (string, Optional) : Location to place the legend
        save_path (string, Optional) : File path to save the image to
    """
    color_palette = sns.color_palette("colorblind")
    color_list = []
    for color in colors:
        color_list.append(color_palette[color])

    bar_positions_training = np.arange(len(probs_train)) * 0.4
    bar_positions_test = np.arange(len(probs_train)) * 0.4 + 1.5

    plt.bar(bar_positions_training, probs_train, width=0.3, color=color_list, edgecolor='black')
    plt.bar(bar_positions_test, probs_test, width=0.3, color=color_list, edgecolor='black')

    handles = [plt.Rectangle((0,0),1,1, facecolor=color_list[i], edgecolor='black') for i in range(len(probs_train))]

    if leg_loc is not None:
        plt.legend(handles, [f'State {i+1}' for i in range(len(probs_train))], title='States', loc=leg_loc)
    else:
        plt.legend(handles, [f'State {i+1}' for i in range(len(probs_train))], title='States', loc='upper right')
    
    plt.ylabel('Probabilities')
    plt.xticks([0.2, 1.7], ['Training', 'Test'])

    # Set custom y-axis ticks
    max_prob = max(max(probs_train), max(probs_test))
    y_ticks = np.arange(0, max_prob + 0.1, 0.1)
    plt.yticks(y_ticks)

    plt.rcParams['font.family'] = 'serif'

    if title is not None:
        plt.title(title)

    # Save the image
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

def render_DR(agent, state, ax=None):
    state_idx = agent.mapping[(state[0], state[1])]
    ax.imshow(agent.DR[state_idx].reshape(agent.height, agent.width), 
              origin='upper', cmap='plasma')
    ax.set_title("DR(%d, %d)" % (state[0], state[1]))
    ax.set_axis_off()

def render_V(values, agent, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    min_value = np.min(values[~np.isinf(values)])
    max_value = np.max(values)

    cmap = plt.cm.Greys_r
    cmap.set_bad('black', 1.0)

    ax.imshow(values.reshape(agent.height, agent.width),
                origin='upper',
                cmap=cmap, vmin=min_value, vmax=max_value)
    ax.set_title("$Values$")
    ax.set_axis_off()

def render_V_log(values, agent, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    min_value = np.min(values[~np.isinf(values)])
    max_value = np.max(values)

    # Logarithmic transformation
    values_log = np.log(values - min_value + 1)  # Add 1 to avoid log(0)

    # Normalize the transformed values
    min_value_log = np.min(values_log)
    max_value_log = np.max(values_log)
    values_scaled = (values_log - min_value_log) / (max_value_log - min_value_log)

    cmap = plt.cm.Greys_r
    cmap.set_bad('black', 1.0)

    ax.imshow(values_scaled.reshape(agent.height, agent.width),
                origin='upper',
                cmap=cmap, vmin=0, vmax=1)  # Scale between 0 and 1
    ax.set_title("$Values$")
    ax.set_axis_off()

def make_plots(agent, values, state=None):
    # Adjust DR at terminal state
    for target_loc in agent.target_locs:
        idx = agent.mapping[target_loc[0], target_loc[1]]
    agent.DR[idx, :] = 0
    agent.DR[idx, idx] = 1

    if state is None:
        state = agent.start_loc
        # state = (0,0)
        
    fig, axs = plt.subplots(1, 3, dpi=144)
    render_maze(agent, state, ax=axs[0])
    render_DR(agent, state, ax=axs[1])
    render_V(agent, ax=axs[2])
    
    plt.show()

def record_trials(agent, title="recorded_trials", n_trial_per_loc=1,
                    start_locs=None, max_steps=100):
    metadata = dict(title=title, artist='JG')
    writer = manimation.FFMpegFileWriter(fps=10, metadata=metadata)
    fig, axs = plt.subplots(1, 3, figsize=(7, 3))
    fig.tight_layout()

    with writer.saving(fig, "./out/%s.mp4" % title, 144):
        for sl in start_locs:
            for trial in range(n_trial_per_loc):
                agent.env.reset()
                done = False
                steps = 0
                state = sl
                
                # set the start and agent location
                agent.env.unwrapped.start_loc, agent.env.unwrapped.agent_loc = state, state
                # Render starting state
                render_maze(agent, state, ax=axs[0])
                render_DR(agent, state, ax=axs[1])
                render_V(agent, ax=axs[2])
                writer.grab_frame()
                for ax in axs:
                        ax.clear()

                # Act greedily and record each state as well
                while not done and steps < max_steps:
                    action = agent.select_action(state)
                    obs, _, done, _, _ = agent.env.step(action)

                    render_maze(agent, state, ax=axs[0])
                    render_DR(agent, state, ax=axs[1])
                    render_V(agent, ax=axs[2])
                    writer.grab_frame()

                    steps += 1

                    state = obs["agent"]

                    for ax in axs:
                        ax.clear()

def record_trajectory(agent, traj, save_path=None):
    fig, ax = plt.subplots()

    m = get_map(agent)
    
    # Display maze
    ax.imshow(m, origin='upper', cmap='gray_r')

    # Minor ticks
    ax.set_xticks(np.arange(-.5, len(m), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(m), 1), minor=True)

    ax.grid(which="minor", color='black', linewidth=2, alpha=0.5)
    # Display agent
    agent_loc = patches.Circle((agent.start_loc[1],agent.start_loc[0]), radius=0.4, fill=True, color='blue', alpha=0.7)
    ax.add_patch(agent_loc)

    # Display Reward
    for i, target_loc in enumerate(agent.target_locs):
        # reward = patches.Circle((target_loc[1], target_loc[0]), radius=0.4, fill=True, color='green')
        reward = patches.Rectangle((target_loc[1] - 0.5, target_loc[0] - 0.5), 1.0, 1.0, fill=True, color='green', alpha=0.7)
        ax.text(target_loc[1], target_loc[0], f'r{i+1}', color='white', fontsize=10, ha='center', va='center')
        ax.add_patch(reward)
    
    # loop through trajectory and add arrows
    for i in range(1, len(traj)):  # Start from the second position
        # Calculate the direction of movement
        diff = traj[i] - traj[i-1]
        # Start the arrow slightly behind the current position
        start_point = traj[i-1] - 0.2 * diff
        arrow = patches.FancyArrow(x=start_point[1], y=start_point[0], dx=0.5*diff[1], dy=0.5*diff[0], width=0.06, length_includes_head=True, color="red")
        ax.add_patch(arrow)

    # Hide tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Hide tick marks
    ax.tick_params(which='both', size=0)

    # Save the image
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)