"""
This package contains all code related to the generation and
interaction with the environment r8-env
"""

# Import packages
import numpy as np
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from IPython.display import clear_output
from matplotlib import colors

# These colours are needed for the dots
colours = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

colour_names = list(colours.keys())
colour_names.pop(64)

colour_names = ['#{}'.format(i) for i in ['332288', '88CCEE', '44AA99', '117733', '999933', 'DDCC77', 'CC6677', '882255', 'AA4499']]

class R8_env:
    """
    This is the most basic version of r8-env
    """

    def __init__(self, grid_dim, num_pairs, num_obstacles, staterep, reward_struc, radius):

        # Grid dimention must be input as a two dimensional structure
        # Will change this to allow a single number for square
        try:

            self.grid_dim = tuple(grid_dim)
            self.grid_size = np.prod(self.grid_dim)
            self.grid_rows, self.grid_cols = self.grid_dim

        except ValueError:

            print('Grid dimension must be a two-dimensional data structure')
            
        # Ingrave the number of pairs of points and obstacles in each instance of the environment
        self.num_pairs = num_pairs
        self.num_obstacles = num_obstacles
        
        # Initialise an order to connect the dots in randomly and hence set start position
        self.order = np.arange(num_pairs)
        
        # Choosing the colours of the points
        self.colours = np.append('grey', np.random.choice(colour_names, self.num_pairs, replace=False))
        
        # Create a new random board
        self.new_board()
        
        # Save an action dictionary for moves
        self.action_dict = {'n' : [-1, 0],
                            'ne' : [-1, 1],
                            'e' : [0, 1],
                            'se' : [1, 1],
                            's' : [1, 0],
                            'sw' : [1, -1],
                            'w' : [0, -1],
                            'nw' : [-1, -1]}
        
        # Having a different step function depending on state representation, for efficiency
        if staterep == 'naive':
            self.get_obs = self.get_obs_naive
        elif staterep == 'lidar1':
            self.get_obs = self.get_obs_lidar1
        elif staterep == 'lidar2':
            self.get_obs = self.get_obs_lidar2
        elif staterep == 'radius':
            self.get_obs = self.get_obs_radius
            self.radius = radius
            
        if reward_struc == 'v1':
            self.get_reward_done = self.get_reward_done_version1
            
        # Saved for efficiency later
        self.xvector = np.array([1, 0])
        self.flags = (-1, 1)
        
    def new_board(self):
        """ Generates a new layout of the environment """
        
        # Generate empty grid
        self.clean_board = np.zeros(self.grid_dim)
        
        # Chooses which points on the grid should be pairs
        self.pairs = self.choose_pairs()
        
        # Tempory variables
        t1 = self.pairs // self.grid_cols
        t2 = self.pairs % self.grid_cols
        
        # Convert into indexes
        self.pair_coords = np.concatenate(([t1[:, 0]], [t2[:, 0]], [t1[:, 1]], [t2[:, 1]]), axis=0).T.reshape((self.num_pairs, 2, 2))
        
        # Choose which squares should be obstacles
        self.obstacles = self.choose_obstacles()
        
        # Convert into indexes
        self.obstacle_coords = np.concatenate(([self.obstacles // self.grid_cols], [self.obstacles % self.grid_cols]), axis=0).T
        
        # Need to figure out how to do this as vectors
        # Adding the obstacles to the environment
        for coord in self.obstacle_coords:
            self.clean_board[coord[0], coord[1]] = 1
            
        # This is the environment with just empty spaces and obstacles, for rendering
        self.env_mask = np.copy(self.clean_board)
        
        # Sets the pairs on the grid
        for i, pair in enumerate(self.pair_coords):
            
            self.clean_board[pair[0][0], pair[0][1]] = i + 2
            self.clean_board[pair[1][0], pair[1][1]] = i + 2

    def reset(self):
        """ Resets a current layout of the environment and start position etc, returns current state"""
        
        # Reset position to start position based on saved order
        self.current_pair = self.order[0]
        self.position = self.pair_coords[self.current_pair][0]
        
        # Here keeping track of pairs of moves for traces
        self.traces = []
        
        # Going to keep track of the move for the rendering
        self.moves_history = {i: [] for i in self.order}
        self.moves_history[self.current_pair].append(self.position)
        
        # Reset the environment to a clean board, this variable will hold the most information 
        # but is mainly used internally.
        self.env = np.copy(self.clean_board)
        
        return self.get_obs()
        
    def choose_pairs(self, method='naive'):
        """ 
        Chooses pairs of points in the grid world based on a choice method 
        """
        
        if method == 'naive':
            # Initial approach, unsophisticated and can be changed for large env which can't be checked by eye
            return np.random.choice(np.arange(self.grid_size), 2 * self.num_pairs, replace=False).reshape((self.num_pairs, 2))
        
    def choose_obstacles(self, method='naive'):
        """ 
        Chooses obstacles based on what has already been filled by points 
        """
        
        if method == 'naive':
            
            # Initial unsophisticated approach which is random
            # Make this set stuff better as not very smart at the moment
            return np.random.choice(list(set(np.arange(self.grid_size)) - set(self.pairs.flatten())), self.num_obstacles, replace=False)
    
    def step(self, action):
        """ 
        This function prdouces one step in the environment returning a state based on a chosen representation 
        """
        # Need to decide if a trace should be point specific or a new number?
        
        # For rewards, do I give a negative for hitting other traces or obstacles? 
        # Will the negative reward for movement solve this?
        
        # Convert string to action vector
        action_vector = self.action_dict[action]
        
        # Find new possible position, clipped to remain on the board
        candidate_position_coord = np.clip(self.position + action_vector, [0, 0], [self.grid_rows - 1, self.grid_cols - 1])
        
        # Discovers if the new position is legal
        flag = self.legal_move(action, candidate_position_coord)
        
        # Set the new position and change environment accordingly
        # complete is wether all pairs have been connected
        complete = self.set_new_position(flag, candidate_position_coord)
        
        # Generate observation
        obs = self.get_obs()
        
        # Generate the reward for the previous step
        reward, done = self.get_reward_done(complete)
        
        # If done is true then obs might look funny but shouldn't matter as should not be involved in updates.
            
        return obs, reward, done
    
    def legal_move(self, action, candidate_position_coord):
        """
        Decides if an action in a given state is legal or not
        """
        
        # Get position as single number
        candidate_position = candidate_position_coord[0] * self.grid_cols + candidate_position_coord[1]
        position_index = self.position[0] * self.grid_cols + self.position[1]
        candidate_position_value = self.env[candidate_position_coord[0], candidate_position_coord[1]]
        
        if candidate_position_value == 1:
            # Now we check whether the agent has hit an obstacle
            return 0
        
        elif candidate_position_value == -1:
            # Checks whether the agent hits a trace
            return 0
        
        elif (candidate_position_value - 2) in self.order[self.order != self.current_pair]:
            # Checks whether we have hit another point
            return 0
        
        elif np.all(candidate_position_coord == self.pair_coords[self.current_pair][0]):
            # Checks whether we have hit the start of the pair
            return 0
        
        elif action in ['ne', 'se', 'sw', 'nw'] and set((candidate_position_coord - self.position) 
                                                        * [self.grid_cols, 1] 
                                                        + position_index) in self.traces:
            # Checks whether the agent hits a diagonal trace
            return 0
        
        elif np.all(candidate_position_coord == self.pair_coords[self.current_pair][1]):
            # Checks whether the agent has completed a paring
            
            return 1
        
        else:
            # This is just a normal move
            return 2
        
    def set_new_position(self, flag, candidate_position_coord):
        """
        Given whether a move is legal, this function moves the agent to the correct new position
        """
        
        # We assume that the agent has not connected the pair initially
        # complete indicates whether pairs have been connected
        complete = False
        
        if flag == 0:
            # This is where a move has been unsuccessful, so nothing changes
            new_position = self.position
            
        elif flag == 1:
            # This is where a pairing has been complete
            
            # The position of the end of the pairing
            new_position = candidate_position_coord
            
            # If a move is successful then update it as a trace
            self.traces.append(set((self.position[0] * self.grid_cols + self.position[1], 
                                   new_position[0] * self.grid_cols + new_position[1])))
            
            # Saved for rendering
            self.moves_history[self.current_pair].append(new_position)
            
            # Now we need to move the agent to the true new position, the start of the next pair
            self.current_pair += 1
            
            true_pair = self.current_pair
            
            self.current_pair = np.clip(self.current_pair, 0, self.num_pairs - 1)
            
            if true_pair < self.num_pairs:
            
                new_position = self.pair_coords[self.current_pair][0]
                self.moves_history[self.current_pair].append(new_position)
                
            else:
                
                new_position = self.pair_coords[self.current_pair][1]
            
            # Indication that pair has been connected
            complete = True
            
        elif flag == 2:
            # This is a normal move
            new_position = candidate_position_coord
            self.env[new_position[0], new_position[1]] = -1
            self.moves_history[self.current_pair].append(new_position)
            
            # If a move is successful then update it as a trace
            self.traces.append(set((self.position[0] * self.grid_cols + self.position[1], 
                                   new_position[0] * self.grid_cols + new_position[1])))
            
        # Sets the position as the new position
        self.position = new_position
        
        return complete
        
    def get_obs_lidar1(self):
        """ 
        Generates an observation based on the distance to objects in each direction and goal location.
        Only looks in north east south and west direction.

        The observation is distance to nearest object and a binary flag of whether the object is the goal,
        followed by the continuous direction the goal is from the agent.
        """
                
        # North first
        # Everything north of agent
        all_north = self.env[:, self.position[1]][:self.position[0]]
        locs = (all_north != 0).nonzero()[0]
        if len(locs) == 0:
            north = [len(all_north) + 1, 0]
        else:
            north = [self.position[0] - locs[-1], np.all([locs[-1], self.position[1]] == self.pair_coords[self.current_pair][1]) * 1]
        
        # East
        all_east = self.env[self.position[0]][self.position[1] + 1:]
        locs = (all_east != 0).nonzero()[0]
        if len(locs) == 0:
            east = [len(all_east) + 1, 0]
        else:
            east = [locs[0] + 1, np.all([self.position[0], locs[0] + self.position[1] + 1] == self.pair_coords[self.current_pair][1]) * 1]
        
        # South
        all_south = self.env[:, self.position[1]][self.position[0] + 1:]
        locs = (all_south != 0).nonzero()[0]
        if len(locs) == 0:
            south = [len(all_south) + 1, 0]
        else:
            south = [locs[0] + 1, np.all([locs[0] + self.position[0] + 1, self.position[1]] == self.pair_coords[self.current_pair][1]) * 1]
        
        # West 
        all_west = self.env[self.position[0]][:self.position[1]]
        locs = (all_west != 0).nonzero()[0]
        if len(locs) == 0:
            west = [len(all_west) + 1, 0]
        else:
            west = [self.position[1] - locs[-1], np.all([self.position[0], locs[-1]] == self.pair_coords[self.current_pair][1]) * 1]
            
        # This is the continuous direction from the agent to the goal in clockwise angle from origin
        vector_location_to_goal = self.pair_coords[self.current_pair][1] - self.position
        if np.all(vector_location_to_goal == 0):
            # Case where we have found the goal, filler value
            goal_location_direc = -1
        else:
            goal_location_direc = (vector_location_to_goal @ self.xvector) / np.linalg.norm(vector_location_to_goal)
                
        return np.array([north, east, south, west, [goal_location_direc, 0]]).flatten()[:-1]
    
    def get_obs_lidar2(self):
        """ 
        This is very similar to original lidar but more compact state space.
        The sign on the distance is the flag for whether an object is the goal for not.
        Negative means it is an obstacle and positive is a goal.
        """
        
        # North first
        # Everything north of agent
        all_north = self.env[:, self.position[1]][:self.position[0]]
        locs = (all_north != 0).nonzero()[0]
        if len(locs) == 0:
            north = -len(all_north) - 1
        else:
            obj_index = np.all([locs[-1], self.position[1]] == self.pair_coords[self.current_pair][1]) * 1
            north = (self.position[0] - locs[-1]) * self.flags[obj_index]
        
        # East
        all_east = self.env[self.position[0]][self.position[1] + 1:]
        locs = (all_east != 0).nonzero()[0]
        if len(locs) == 0:
            east = -len(all_east) - 1
        else:
            obj_index = np.all([self.position[0], locs[0] + self.position[1] + 1] == self.pair_coords[self.current_pair][1]) * 1
            east = (locs[0] + 1) * self.flags[obj_index]
        
        # South
        all_south = self.env[:, self.position[1]][self.position[0] + 1:]
        locs = (all_south != 0).nonzero()[0]
        if len(locs) == 0:
            south = -len(all_south) - 1
        else:
            obj_index = np.all([locs[0] + self.position[0] + 1, self.position[1]] == self.pair_coords[self.current_pair][1]) * 1
            south = (locs[0] + 1) * self.flags[obj_index]
        
        # West 
        all_west = self.env[self.position[0]][:self.position[1]]
        locs = (all_west != 0).nonzero()[0]
        if len(locs) == 0:
            west = -len(all_west) - 1
        else:
            obj_index = np.all([self.position[0], locs[-1]] == self.pair_coords[self.current_pair][1]) * 1
            west = (self.position[1] - locs[-1]) * self.flags[obj_index]
            
        # This is the continuous direction from the agent to the goal in clockwise angle from origin
        vector_location_to_goal = self.pair_coords[self.current_pair][1] - self.position
        if np.all(vector_location_to_goal == 0):
            # Case where we have found the goal, filler value
            goal_location_direc = -1
        else:
            goal_location_direc = (vector_location_to_goal @ self.xvector) / np.linalg.norm(vector_location_to_goal)
                
        return [north, east, south, west, goal_location_direc]
    
    def get_obs_naive(self):
        """ 
        This fetches and returns an observation which is the whole board. 
        In this representation the agent is a number 1, the goal a number 2,
        blank squares are 0 and obstacles are -1
        """
        # Is it a bad idea to have blank squares as 0s for neural networks?
        
        obs = np.copy(self.env)
        
        obs[obs != 0] = -1
        obs[self.pair_coords[self.current_pair][1][0], self.pair_coords[self.current_pair][1][1]] = 2
        obs[self.position[0], self.position[1]] = 1
        
        return obs
    
    def get_obs_radius(self):
        """
        This generates an observation which is a radius of information around the agent and
        a direction the goal is in. Only does odd numbered nxn squares.
        """
        
        whole_board = np.copy(self.get_obs_naive())
        
        border = self.radius // 2
        
        obs = whole_board[np.clip(self.position[0] - border, 0, self.position[0]) : self.position[0] + border + 1, 
                          np.clip(self.position[1] - border, 0, self.position[1]) : self.position[1] + border + 1]
        
        # This is the continuous direction from the agent to the goal in clockwise angle from origin
        vector_location_to_goal = self.pair_coords[self.current_pair][1] - self.position
        if np.all(vector_location_to_goal == 0):
            # Case where we have found the goal, filler value
            goal_location_direc = -1
        else:
            goal_location_direc = (vector_location_to_goal @ self.xvector) / np.linalg.norm(vector_location_to_goal)
        
        return [obs, goal_location_direc]
    
    def find_neighbours(self):
        """
        This returns the immediate neighbours of the agent.
        Used to find dead ends etc.
        """
        
        whole_board = np.copy(self.get_obs_naive())
        
        border = 1
        
        neighbours = whole_board[np.clip(self.position[0] - border, 0, self.position[0]) : self.position[0] + border + 1, 
                                 np.clip(self.position[1] - border, 0, self.position[1]) : self.position[1] + border + 1]
        
        return neighbours
    
    def get_reward_done_version1(self, complete=False):
        """ 
        This calculates the reward of each step and whether an episode has terminated.
        It is a first incarnation of the reward structure, can be viewed as the simplest/naive.
        """
        
        # Boolean to denote whether episode has ended
        done = np.all(self.position == self.pair_coords[-1][1])
        
        # Negative reward for every step
        reward = -1
        
        if complete:
            # If points have been connected then positive reward
            reward += 10
        
        # If trapped then episode should end and large negative reward is given
        agent_trapped = self.trapped()
        if agent_trapped:
            reward -= 100
            done = True
        
        return reward, done
    
    def trapped(self):
        """ 
        Decides whether the agent has trapped itself in a deadend
        """
        
        # Collect radius around the agent
        neighbours = self.find_neighbours()
        
        index = np.where(neighbours == 1)
        
        # Collect all the agency values to the agent
        row = neighbours[index[0][0], index[1][0] - 1 : index[1][0] + 2]
        col = neighbours[index[0][0] - 1 : index[0][0] + 2, index[1][0]]
        adjacent_values = np.append(row, col)
        
        
        if np.any((adjacent_values == 0) | (adjacent_values == 2)):
            # First check the cross around the agent
            return False
            
        else:
            # This checks diagonal traces or obstacles in diagonal squares
            
            # First check if there are traces in each diagonal
            position_index = self.position[0] * self.grid_cols + self.position[1]
            
            # Coordinates of each adjacent square
            adjacent_positions = np.array([(position_index) * ((self.position[1] - 1) >= 0),
                                           (position_index - self.grid_cols + 1) * ((self.position[0] - 1) >= 0),
                                           (position_index + 2) * ((self.position[1] + 1) < self.grid_cols),
                                           (position_index + self.grid_cols + 1) * ((self.position[0] + 1) < self.grid_rows)])
       
            adjacent_positions = adjacent_positions[adjacent_positions != 0] - 1
            
            N = len(adjacent_positions)
            
            # This is a boolean vector denoting whether a diagonal has a trace blocking it
            trace_index = np.array([pair in self.traces for pair in [set((adjacent_positions[i], adjacent_positions[(i + 1) % N])) 
                                                                for i in range(N)]])
            
            # Need to make this more efficient, by using a dictionary maybe?
            
            # Treating each case of the size of the neighbourhood differently
            if neighbours.shape == (3, 3):
                trace_indexindex = [0, 1, 2, 3]
            elif neighbours.shape == (2, 3):
                if index[0][0] == 0:
                    trace_indexindex = [1, 2]
                elif index[0][0] == 1:
                    trace_indexindex = [0, 1]
            elif neighbours.shape == (3, 2):
                if index[1][0] == 0:
                    trace_indexindex = [0, 1]
                elif index[1][0] == 1:
                    trace_indexindex = [0, 2]
            elif neighbours.shape == (2, 2):
                trace_indexindex = [0]
            
            trace_index = trace_index[trace_indexindex]
            
            # Now a vector which denotes whether a diagonal is blocked by an obstacle
            # For this we need the diagonals so will make all the other values different
            # to abstract the small ones
            
            neighbours[index[0][0], index[1][0] - 1 : index[1][0] + 2] = -2
            neighbours[index[0][0] - 1 : index[0][0] + 2, index[1][0]] = -2
            
            row = neighbours[index[0][0], index[1][0] - 1 : index[1][0] + 2]
            col = neighbours[index[0][0] - 1 : index[0][0] + 2, index[1][0]]
            adjacent_values = np.append(row, col)
            
            # Need the neighbours fliped in this way because of how numpy vectors work
            flipped_neighbours = np.append(neighbours[:-1], [np.flip(neighbours[-1])], axis=0)
            
            # Generates a boolean vector deonting whether the agent can enter into a diagonal square
            diagonal_values = flipped_neighbours[np.isin(flipped_neighbours, adjacent_values, invert=True)]
            obstacle_index = (diagonal_values != 0) & (diagonal_values != 2)
        
            if np.all(obstacle_index | trace_index):
                # If the agent can't pass into any diagonal square then return true
                return True
            
            else:
                # This clause is where the agent can pass into a neighbouring square
                return False
        
    def render(self):
        """ 
        This renders the gridworld envrionment, adapted from Joshua's code 
        """
        
        # Turn interactive mode on.
        plt.ion()
        fig = plt.figure(num = "env_render", figsize=self.grid_dim)
        ax = plt.gca()
        ax.clear()
        clear_output(wait = True)

        # Prepare the environment
        env_plot = np.copy(self.env_mask).astype(int)
        
        colours = ['k', 'grey']
        
        # Plot the gridworld.
        cmap = colors.ListedColormap(colours)
#         bounds = list(range(self.num_pairs + 1))
        bounds = [0, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N - 1)
        ax.imshow(env_plot, cmap = cmap, norm = norm, zorder = 0)
        
        # Plot the points and traces
        for i, (pair, colour) in enumerate(zip(self.pair_coords, self.colours)):
            ax.scatter(pair[:, 1], pair[:, 0], color=colour, linewidth = 5, label='{}'.format(i))
            if len(self.moves_history[i]) != 0:
                moves = np.array(self.moves_history[i])
                ax.plot(moves[:, 1], moves[:, 0], color=colour, linewidth = 2)
                
        # Temp so know where I start
        ax.scatter(self.moves_history[self.current_pair][-1][1], self.moves_history[self.current_pair][-1][0], 
                   color='w', linewidth = 5, zorder=10)
        
        # Set up axes.
        ax.grid(which = 'major', axis = 'both', linestyle = '-', color = 'grey', linewidth = 2, zorder = 1)
        ax.set_xticks(np.arange(-0.5, self.env_mask.shape[1] , 1));
        ax.set_xticklabels([])
        ax.set_yticks(np.arange(-0.5, self.env_mask.shape[0], 1));
        ax.set_yticklabels([])
        
        plt.legend(bbox_to_anchor=(1.25, 1))
        plt.show()
        
def make(version='Classic', grid_dim=(25, 10), num_pairs=9, 
         num_obstacles=20, staterep='naive', reward_struc = 'v1', radius=3):
    """ Generates a specific version of R8, capped at 9 as the moment """
    
    if version == 'Classic':
        
        return R8_env(grid_dim, num_pairs, num_obstacles, staterep, reward_struc, radius)