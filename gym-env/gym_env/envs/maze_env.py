import os
import time

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, maze_file, render_mode=None):
        # Read maze_file
        self.maze = self._read_maze_file(maze_file=maze_file)

        # Check if render mode is valid and set render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Get important positions
        start = np.where(self.maze == 'S')
        target = np.where(self.maze == 'G')

        # self.start_loc = np.array([start[0][0], start[1][0]])
        self.start_loc = np.argwhere(self.maze == 'S')[0]
        # self.target_loc = np.array([target[0][0], target[1][0]])
        self.target_locs = np.argwhere(self.maze == 'G')
        self.agent_loc = self.start_loc

        # Size of maze and pygame window
        self.num_rows, self.num_cols = self.maze.shape
        self.window_size = 512

        # 4 possible actions: 0=up, 1=down, 2=right, 3=left
        self.action_space = spaces.Discrete(4)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=np.array([0,0]), high=np.array([self.num_rows-1, self.num_cols-1]), shape=(2,), dtype=int),
                "targets": spaces.Box(low=np.tile(np.array([0, 0]), (len(self.target_locs), 1)), high=np.tile(np.array([self.num_rows-1, self.num_cols-1]), (len(self.target_locs), 1)), shape=(len(self.target_locs), 2), dtype=int),
            }
        )

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        # Only used for human-rendering
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.agent_loc = self.start_loc

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        """
        One step in our environment given the action
        """
        # Get direction
        direction = self._action_to_direction[action]

        new_loc = np.copy(self.agent_loc)
        new_loc += direction
        # Check if the new position is valid
        if self._is_valid_position(new_loc):
            self.agent_loc = new_loc

        # Check if terminated
        # terminated = np.array_equal(self.agent_loc, self.target_loc)
        terminated = self._is_at_target(self.agent_loc)
        reward = 1 if terminated else 0  # Binary sparse rewards
        
        if self.render_mode == "human":
            self._render_frame()
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
    
    def random_action(self):
        """
        Returns a random action from the environment
        """
        available_actions = []
        for action in np.arange(self.action_space.n, dtype=int):
            direction = self._action_to_direction[action]
            new_loc = np.copy(self.agent_loc)
            new_loc += direction
            if self._is_valid_position(new_loc):
                available_actions.append(action)
        
        return np.random.choice(available_actions)
    
    def get_available_actions(self, state):
        """
        Returns available actions at specified state
        """
        available_actions = []
        for action in np.arange(self.action_space.n, dtype=int):
            direction = self._action_to_direction[action]
            new_loc = np.copy(state)
            new_loc += direction
            if self._is_valid_position(new_loc):
                available_actions.append(action)
        return available_actions
    
    def get_successor_states(self, state):
        """
        Returns a list of successor states and if they are terminal states
        """
        next_states = []
        for action in np.arange(self.action_space.n, dtype=int):
            direction = self._action_to_direction[action]
            new_loc = np.copy(state)
            new_loc += direction
            if self._is_valid_position(new_loc):
                terminated = self._is_at_target(new_loc)
                # terminated = np.array_equal(new_loc, self.target_loc)
                next_states.append((new_loc, terminated))
        
        return next_states
    
    def get_walls(self):
        """
        Returns a list of all the walls (blocked squares)
        """
        walls = []
        # Loop through the maze
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                # If square is blocked
                if self.maze[row, col] == '1':
                    walls.append((row,col))
        
        return walls
    
    def _is_at_target(self, current_loc):
        """
        Check to see if current_loc is at one of the target_locs
        """
        return np.any(np.all(self.target_locs == current_loc, axis=1))
    
    def _read_maze_file(self, maze_file):
        dir_path = os.path.dirname(os.path.abspath(__file__))
        rel_path = os.path.join(dir_path, "maze_files", maze_file)

        return np.load(file=rel_path)

    def _get_obs(self):
        """
        Observation, returns the agent and target positions
        """
        return {"agent": self.agent_loc, "targets":self.target_locs}
    
    def _get_info(self):
        """
        Information, returns the manhattan (L1) distance between agent and target.
        """
        return {
            "distances": np.linalg.norm(self.target_locs - self.agent_loc, ord=1, axis=1)
        }

    def _is_valid_position(self, pos):
        """
        Checks if position is in bounds or if obstacle is hit
        """
        row, col = pos
   
        # If agent goes out of the grid
        if row < 0 or col < 0 or row >= self.num_rows or col >= self.num_cols:
            return False

        # If the agent hits an obstacle
        if self.maze[row, col] == '1':
            return False
        
        return True

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "pyplot":
            return self._render_pyplot()
    
    def _render_pyplot(self):
        """
        Renders current frame using pyplot
        """
        
        raise NotImplementedError

    def _render_frame(self):
        """
        Renders a frame in pygame
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.num_cols
        )  # The size of a single grid square in pixels

        # Draw targets
        for target_loc in self.target_locs:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    target_loc[1] * pix_square_size,
                    target_loc[0] * pix_square_size,
                    pix_square_size,
                    pix_square_size
                )
            )
        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self.target_loc,
        #         (pix_square_size, pix_square_size),
        #     ),
        # )
        # Draw start
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self.start_loc,
                (pix_square_size, pix_square_size),
            ),
        )
        # Draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self.agent_loc + 0.5) * pix_square_size,
            pix_square_size / 3,
        )
        # Draw horizontal lines
        for y in range(self.num_rows + 1):
            pygame.draw.line(canvas, 0, (0, y * pix_square_size),
                             (self.window_size, y * pix_square_size))

        # Draw vertical lines
        for x in range(self.num_cols + 1):
            pygame.draw.line(canvas, 0, (x * pix_square_size, 0),
                             (x * pix_square_size, self.window_size))
        
        # Draw obstacles
        obs = np.where(self.maze == '1')
        for i in range(obs[0].size):
            row = obs[0][i]
            col = obs[1][i]

            cell_left = col * pix_square_size
            cell_top = row * pix_square_size

            # Draw Obstacle
            if self.maze[row, col] == '1':  # Obstacle
                pygame.draw.rect(
                    canvas,
                    (0, 0, 0),
                    pygame.Rect(
                        (cell_left, cell_top),
                        (pix_square_size, pix_square_size),
                    ),
                )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    

class MazeEnv5x5(MazeEnv):
    def __init__(self):
        super(MazeEnv5x5, self).__init__(maze_file="maze2d_5x5.npy")

class MazeEnv5x5_2G(MazeEnv):
    def __init__(self):
        super(MazeEnv5x5_2G, self).__init__(maze_file="maze_5x5_2g.npy")

class MazeEnv7x7_2G(MazeEnv):
    def __init__(self):
        super(MazeEnv7x7_2G, self).__init__(maze_file="maze_7x7_2g.npy")

class MazeEnv10x10_2G(MazeEnv):
    def __init__(self):
        super(MazeEnv10x10_2G, self).__init__(maze_file="maze_10x10_2g.npy")

class MazeEnvHairpin(MazeEnv):
    def __init__(self):
        super(MazeEnvHairpin, self).__init__(maze_file="hairpin_14x14.npy")

class MazeEnvTolmanNB(MazeEnv):
    def __init__(self):
        super(MazeEnvTolmanNB, self).__init__(maze_file="tolman_9x9_v0.npy")

class MazeEnvTolmanB(MazeEnv):
    def __init__(self):
        super(MazeEnvTolmanB, self).__init__(maze_file="tolman_9x9_v1.npy")

class MazeEnv15x15(MazeEnv):
    def __init__(self):
        super(MazeEnv15x15, self).__init__(maze_file="maze_15x15.npy")

class MazeEnv15x15NewGoal(MazeEnv):
    def __init__(self):
        super(MazeEnv15x15NewGoal, self).__init__(maze_file="maze_15x15_new_goal.npy")

class MazeEnvTolmanLatentOG(MazeEnv):
    def __init__(self):
        super(MazeEnvTolmanLatentOG, self).__init__(maze_file="tolman_latent.npy")

class MazeEnvTolmanLatent(MazeEnv):
    def __init__(self):
        super(MazeEnvTolmanLatent, self).__init__(maze_file="tolman_10x10_latent.npy")

class MazeEnvTolmanLatentNewGoal(MazeEnv):
    def __init__(self):
        super(MazeEnvTolmanLatentNewGoal, self).__init__(maze_file="tolman_10x10_latent_new_goal.npy")

class MazeEnv4RoomTR(MazeEnv):
    def __init__(self):
        super(MazeEnv4RoomTR, self).__init__(maze_file="four_room_tr.npy")

class MazeEnv4RoomBR(MazeEnv):
    def __init__(self):
        super(MazeEnv4RoomBR, self).__init__(maze_file="four_room_br.npy")

class MazeEnv4RoomNG(MazeEnv):
    def __init__(self):
        super(MazeEnv4RoomNG, self).__init__(maze_file="four_room_ng.npy")

class MazeEnv4RoomSG(MazeEnv):
    def __init__(self):
        super(MazeEnv4RoomSG, self).__init__(maze_file="four_room_sg.npy")

class MazeEnv15x15_G0(MazeEnv):
    def __init__(self):
        super(MazeEnv15x15_G0, self).__init__(maze_file="maze_15x15_G0.npy")

class MazeEnv15x15_G1(MazeEnv):
    def __init__(self):
        super(MazeEnv15x15_G1, self).__init__(maze_file="maze_15x15_G1.npy")

class MazeEnv10x10_G0(MazeEnv):
    def __init__(self):
        super(MazeEnv10x10_G0, self).__init__(maze_file="maze_10x10_G0.npy")

class MazeEnv10x10_G1(MazeEnv):
    def __init__(self):
        super(MazeEnv10x10_G1, self).__init__(maze_file="maze_10x10_G1.npy")

class MazeEnv10x10_G2(MazeEnv):
    def __init__(self):
        super(MazeEnv10x10_G2, self).__init__(maze_file="maze_10x10_G2.npy")

class MazeEnv10x10_G3(MazeEnv):
    def __init__(self):
        super(MazeEnv10x10_G3, self).__init__(maze_file="maze_10x10_G3.npy")

class MazeEnvOpenFieldNoGoal(MazeEnv):
    def __init__(self):
        super(MazeEnvOpenFieldNoGoal, self).__init__(maze_file="open_field_no_goal.npy")

class MazeEnvOpenFieldCenterGoal(MazeEnv):
    def __init__(self):
        super(MazeEnvOpenFieldCenterGoal, self).__init__(maze_file="open_field_center_goal.npy")

class MazeEnvOpenFieldNoGoalLarge(MazeEnv):
    def __init__(self):
        super(MazeEnvOpenFieldNoGoalLarge, self).__init__(maze_file="open_field_no_goal_large.npy")

class MazeEnvOpenFieldCenterGoalLarge(MazeEnv):
    def __init__(self):
        super(MazeEnvOpenFieldCenterGoalLarge, self).__init__(maze_file="open_field_center_goal_large.npy")

class MazeEnvCarpenter(MazeEnv):
    def __init__(self):
        super(MazeEnvCarpenter, self).__init__(maze_file="carpenter_maze.npy")


if __name__ == '__main__':
    # Test it out
    env = MazeEnv(maze_file="tolman_latent.npy", render_mode='human')
    # env = MazeEnv(maze_file="hairpin_14x14.npy")
    print(f"env: {env}")
    print(f"start loc: {env.start_loc}, target locs: {env.target_locs}")
    obs, info = env.reset()
    print(f"Post reset obs: {obs}, info: {info}")
    rand_action = env.action_space.sample()
    print(f"Random action: {rand_action}")
    obs, reward, term, _, info = env.step(rand_action)
    print(f"Post step obs: {obs}, reward: {reward}, terminated: {term}, info: {info}")
    print("Printing maze:")
    print(env.maze)
    print("Printing wall locations:")
    print(env.get_walls())
    # env.render()
    # time.sleep(20)