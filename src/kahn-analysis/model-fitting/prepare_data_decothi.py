import os
import pickle
import numpy as np
from scipy.io import loadmat

TERM_IDX= 33

def extract_subject_data_decothi(data, maze_idx, subject_id):
    """
    Extract state sequences for one subject in one maze configuration.

    Parameters:
    -----------
    data : numpy array
        Full De-Cothi dataset (subjects × mazes × trials)
    maze_idx : int
        Maze configuration index
    subject_id : int
        Subject ID

    Returns:
    --------
    subject_dict : dict with keys:
        'subject_id': subject ID
        'maze': maze index
        'states': list of state sequences (one per trial/starting point)
    """
    states = []
    for seq in data[subject_id, maze_idx, :]:
        if len(seq) == 0:
            continue
        states.append(seq[0])

    return {
        'subject_id': subject_id,
        'maze': maze_idx,
        'states': states
    }


def extract_all_subjects_decothi(species=None):
    """
    Extract data for all subjects in the format needed for model fitting.

    Parameters:
    -----------
    species : str
        Either 'humans' or 'rats'

    Returns:
    --------
    all_subjects : list of lists
        Each element is a list of maze configuration dictionaries for one subject.
        Format: [subject_1_configurations, subject_2_configurations, ...]
        where subject_i_configurations = [
            {'maze': int, 'states': [...]},
            {'maze': int, 'states': [...]},
            ...
        ]
    """
    path = 'data/de-cothi/'

    if species == 'humans':
        data = loadmat(path + 'humans.mat')['humans']
    elif species == 'rats':
        data = loadmat(path + 'rat.mat')['rat']
        # Fix rat data (known issue in original dataset)
        data[0, 11, 4] -= 10
    else:
        print("Species not specified or not one of [humans, rats]")
        return None

    mazes = loadmat(path + 'mazes.mat')['mazes'][0]
    all_subjects = []

    print(f"Extracting data for {len(data)} subjects...")
    for subject_id in range(len(data)):
        subject_list = []
        for maze_idx in range(len(mazes)):
            # Get the subject's data for this maze
            subject_dict = extract_subject_data_decothi(data, maze_idx, subject_id)
            subject_list.append(subject_dict)

        all_subjects.append(subject_list)

    print(f"Done! Data ready for {len(all_subjects)} subjects")
    return all_subjects


def create_transition_matrix_open(N=10):
    """Create a transition matrix for a random walk on an NxN grid."""
    total_states = N * N
    transition_matrix = np.zeros((total_states, total_states))
    
    def coord_to_index(i, j):
        return i * N + j
    
    def index_to_coord(idx):
        return idx // N, idx % N
    
    # For each state (position on the grid)
    for idx in range(total_states):
        i, j = index_to_coord(idx)
        
        # Find valid neighbors (up, down, left, right)
        neighbors = []
        
        # Up
        if i > 0:
            neighbors.append(coord_to_index(i-1, j))
        
        # Down
        if i < N - 1:
            neighbors.append(coord_to_index(i+1, j))
        
        # Left
        if j > 0:
            neighbors.append(coord_to_index(i, j-1))
        
        # Right
        if j < N - 1:
            neighbors.append(coord_to_index(i, j+1))
        
        # Uniform transition probability to each valid neighbor
        if neighbors:
            prob = 1.0 / len(neighbors)
            for neighbor_idx in neighbors:
                transition_matrix[idx, neighbor_idx] = prob
    
    transition_matrix[TERM_IDX, :] = 0
    transition_matrix[TERM_IDX, TERM_IDX] = 1
    
    return transition_matrix


if __name__ == "__main__":
    # Process both species
    for species in ['humans', 'rats']:
        output_file = f'data/de-cothi/all_subjects_data_{species}.pkl'
        print(f"\nProcessing {species} data...")

        all_subjects_data = extract_all_subjects_decothi(species)

        if all_subjects_data is not None:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Save for future use
            with open(output_file, 'wb') as f:
                pickle.dump(all_subjects_data, f)

            print(f"Saved preprocessed data to {output_file}")
            print(f"  - {len(all_subjects_data)} subjects")
            print(f"  - {len(all_subjects_data[0])} maze configurations per subject")
