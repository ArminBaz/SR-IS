import os
import pickle
import pandas as pd
import numpy as np

def extract_subject_data_kahn(df, subject_id):
    """
    Extract states, actions, and rewards for one subject.

    Parameters:
    -----------
    df : pandas DataFrame
        Full dataset from load_blockwise_data()
    subject_id : int
        Subject ID (1-100)

    Returns:
    --------
    subject_dict : dict with keys:
        'states': list of lists - state sequences for each trial
        'actions': list of lists - action sequences for each trial
        'rewards': numpy array - reward received on each trial (0 or 1)
        'reward_probs': numpy array (n_trials × 4) - reward probs for each boat
        'trial_types': numpy array - trial type for each trial
                       (0=boat-only, 1=full-traversal, 2=island-only)

    Example:
    --------
    Trial with full traversal:
        states = [1, 3, 6]     # start → right island → boat 3
        actions = [3, 6]       # chose island 3, chose boat 6
        reward = 1.0

    Trial with boat-only (non-traversal):
        states = [4]           # shown boat 1 directly
        actions = []           # no actions (just observation)
        reward = 0.0
    """
    # Get data for this subject
    subject_df = df[df['sub'] == subject_id].copy().reset_index(drop=True)
    n_trials = len(subject_df)

    # Initialize outputs
    states = []
    actions = []
    rewards = np.zeros(n_trials)
    reward_probs = subject_df[['r1', 'r2', 'r3', 'r4']].values
    trial_types = np.zeros(n_trials, dtype=int)

    # Process each trial
    for i, row in subject_df.iterrows():
        # Extract state sequence
        state_seq = [int(row['state1'])]
        if row['state2'] > 0:  # Not missing
            state_seq.append(int(row['state2']))
        if row['state3'] > 0:  # Not missing
            state_seq.append(int(row['state3']))
        states.append(state_seq)

        # Extract action sequence
        action_seq = []
        if row['state2'] > 0:
            action_seq.append(int(row['state2']))  # Island choice
        if row['state3'] > 0:
            action_seq.append(int(row['state3']))  # Boat choice
        actions.append(action_seq)

        # Extract reward
        rewards[i] = row['reward']

        # Determine trial type
        if row['state1'] in [4, 5, 6, 7]:
            trial_types[i] = 0  # Boat-only (non-traversal)
        elif row['state1'] == 1 and row['state3'] > 3:
            trial_types[i] = 1  # Full traversal
        elif row['state1'] == 1 and row['state3'] == -1:
            trial_types[i] = 2  # Island-only

    return {
        'subject_id': subject_id,
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'reward_probs': reward_probs,
        'trial_types': trial_types,
        'n_trials': n_trials
    }

def extract_all_subjects_kahn(df):
    """
    Extract data for all subjects in the format needed for model fitting.
    
    Now includes auxiliary terminal states (8,9,10,11) where rewards are housed:
    - State 4 -> auxiliary state 8
    - State 5 -> auxiliary state 9
    - State 6 -> auxiliary state 10
    - State 7 -> auxiliary state 11

    Returns:
    --------
    all_subjects : list of lists
        Each element is a list of trial dictionaries for one subject.
        Format: [subject_1_trials, subject_2_trials, ...]
        where subject_i_trials = [
            {'states': [...], 'reward': ...},
            {'states': [...], 'reward': ...},
            ...
        ]
    """
    all_subjects = []
    subject_ids = sorted(df['sub'].unique())

    print(f"Extracting data for {len(subject_ids)} subjects...")
    for subject_id in subject_ids:
        # Get the subject's data in the old format
        subject_dict = extract_subject_data_kahn(df, subject_id)
        
        # Convert to list of trial dictionaries
        subject_trials = []
        for i in range(subject_dict['n_trials']):
            states = subject_dict['states'][i].copy()  # Make a copy
            
            # Add auxiliary terminal state
            last_state = states[-1]
            if last_state in [4, 5, 6, 7]:
                # Map to auxiliary state: 4->8, 5->9, 6->10, 7->11
                auxiliary_state = last_state + 4
                states.append(auxiliary_state)
            
            trial = {
                'states': states,
                'reward': subject_dict['rewards'][i]
            }
            subject_trials.append(trial)
        
        all_subjects.append(subject_trials)

    print(f"Done! Data ready for {len(all_subjects)} subjects")
    return all_subjects

if __name__ == "__main__":
    all_subjects_data_file = 'data/kahn/all_subjects_data.pkl'
    print("Processing subject data...")
    df = pd.read_pickle(os.path.join(os.getcwd(), 'data/kahn/blockwise.pkl'))
    all_subjects_data = extract_all_subjects_kahn(df)
    # Save for future use
    with open(all_subjects_data_file, 'wb') as f:
        pickle.dump(all_subjects_data, f)
    print(f"Saved preprocessed data to {all_subjects_data_file}")