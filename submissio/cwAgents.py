

from pacman import Directions
from game import Agent
import api as api
import random
import game
import util
import sys
from collections import deque

import math
import heapq
import Queue

class MyBellmanUpdateAgent(Agent):

    def __init__(self):
        self.utilities = None
        self.rewards = None
        self.height = 0
        self.width = 0
        self.turn = 0

        self.threshold = 0.01
        self.discount_factor = 0.9

        self.ghost_reward = -200  # Penalty for ghosts
        self.food_reward = 10     # Reward for food
        self.default_reward = -1  # Default reward for empty cells

        # For ghost proximity
        self.max_ghost_distance = 5
        self.decay_rate_danger = 0.5 # Higher decay = 

        # For efficient sweep
        self.sweep_bonus = 10 # Added to the self.food_reward
        # self.decay_rate_food = 0.9
        # self.max_food_distance = 100
        self.weak_path_reward = 1

        # For ghost hunting
        self.capsules = []  # Initialize a list to store capsules
        self.in_chase_mode = False
        self.chase_time_remaining = 0
        self.ghost_path_reward = 15
        self.total_chase_time = 0
        self.total_chase_distance = 0
        self.initialized = False
        self.initial_food_locations = set()
        self.initial_setup_done = False
        self.initial_good_locations = set()
        self.edible_ghost_reward = 0

        # Aesthetics
        self.print_all = False # Add this line
        self.warning_counter = 0  # Counter for the warning messages
        self.turn = 0  # Counter for the turns taken

    def print_game_grid(self, state, pacman):
        # if self.print_all:
        pacman = api.whereAmI(state)
        ghost_states, walls, food, capsules = api.ghostStates(state), api.walls(state), api.food(state), api.capsules(state)
        grid = [[' ' for _ in range(self.height)] for _ in range(self.width)]

        # Place walls, Pacman, food, capsules, and ghosts on grid
        for wall in walls: grid[wall[0]][wall[1]] = '#'
        grid[pacman[0]][pacman[1]] = '@'
        for food in food: grid[food[0]][food[1]] = '*'
        for capsule in capsules: grid[capsule[0]][capsule[1]] = '~'

        # Place ghosts on grid, distinguishing between edible and non-edible
        for ghost_state in ghost_states:
            ghost_position, is_edible = ghost_state
            ghost_char = 'E' if is_edible else 'G'
            grid[int(ghost_position[0])][int(ghost_position[1])] = ghost_char

        # Print grid rotated 90 degrees and flipped vertically so that it is correctly oriented
        for col in reversed(range(self.height)):
            for row in range(self.width):
                print grid[row][col],
            print
        print

    def print_utilities_grid(self, state):
        # if self.print_all:
        print "Utilities Grid:"
        pacman = api.whereAmI(state)
        for y in reversed(range(self.height)):
            for x in range(self.width):
                if (x, y) == pacman:
                    print "   x  ",
                else:
                    print "      " if self.utilities[y][x] is None else "{:6.2f}".format(self.utilities[y][x]),
            print
        print

    def print_rewards_grid(self):
        # if self.print_all:
        print "Rewards Grid:"
        for y in reversed(range(self.height)):
            for x in range(self.width):
                reward = self.rewards[y][x]
                print "      " if reward is None else "{:6.2f}".format(reward),
            print
        print






    ### Helper functions for 1, 2.

    def get_grid_dimensions(self, state):
        walls = api.walls(state)
        max_x = max(wall[0] for wall in walls) + 1
        max_y = max(wall[1] for wall in walls) + 1
        return max_y, max_x  # Height, Width
    
    def exponential_fall_off(self, distance, decay_rate):
        # decay_rate is positive to avoid increase in penalty with distance
        return math.exp(-decay_rate * distance)
    
    def gaussian_fall_off(self, distance):
        # Gaussian function for fall-off
        return math.exp(- (distance ** 2) / (2 * self.decay_rate_food ** 2))

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
    def get_neighbors(self, node, walls):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        result = []
        for dx, dy in directions:
            next = (node[0] + dx, node[1] + dy)
            if next not in walls:
                result.append(next)
        return result

    def a_star_search(self, start, goal, walls):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == goal:
                break

            for next in self.get_neighbors(current, walls):
                new_cost = cost_so_far[current] + 1  # Cost between neighbors is 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current

        return cost_so_far.get(goal, float('inf'))
    
    def calculate_food_clusters(self, food, walls):
        food_clusters = {}  # Dictionary to hold the cluster sizes for each food coordinate
        visited = set()  # Set to keep track of visited coordinates
        cluster_sizes = {}  # Temporary dictionary to hold sizes before assigning to all members

        def get_food_neighbors(coord, food, walls):
            # Given a coordinate, return the neighboring coordinates that have food
            x, y = coord
            possible_moves = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
            neighbors = [move for move in possible_moves if move in food and move not in walls]
            return neighbors

        for food_coord in food:
            if food_coord not in visited:
                # BFS to find all connected foods
                cluster_size = 0
                q = Queue.Queue()
                q.put(food_coord)
                visited.add(food_coord)
                cluster_members = []

                while not q.empty():
                    current = q.get()
                    cluster_size += 1
                    cluster_members.append(current)

                    for neighbor in get_food_neighbors(current, food, walls):
                        if neighbor not in visited:
                            q.put(neighbor)
                            visited.add(neighbor)

                # Store the size in the temporary dictionary
                for member in cluster_members:
                    cluster_sizes[member] = cluster_size

        # Now that we've calculated the sizes for each cluster, assign them
        for food_coord, size in cluster_sizes.items():
            food_clusters[food_coord] = size

        return food_clusters

    def apply_food_mask(self, food_positions):
        # Apply a Gaussian mask to the rewards grid based on food positions
        for y in range(self.height):
            for x in range(self.width):
                if self.rewards[y][x] is None:
                    continue  # Skip walls

                # Initialize the additional reward for this cell due to food
                food_effect = 0

                # Sum the effects of all food positions
                for food_x, food_y in food_positions:
                    distance = math.sqrt((food_x - x) ** 2 + (food_y - y) ** 2)
                    food_effect += self.gaussian_fall_off(distance)

                # Update the reward for the cell
                self.rewards[y][x] += food_effect


    def coordinate_chase_value(self, state, coordinate_location):
        ghost_states_with_times = api.ghostStatesWithTimes(state)

        # Extracting ghost positions as tuples of integers
        ghost_positions = []
        for ghost_state_with_time in ghost_states_with_times:
            ghost_position = ghost_state_with_time[0]  # The position tuple
            x, y = ghost_position  # Unpacking the position tuple
            ghost_positions.append((int(x), int(y)))

        total_distance = 0

        # Calculate the sum of Manhattan distances from the coordinate_location to each ghost
        for ghost in ghost_positions:
            distance_to_ghost = self.manhattan_distance(coordinate_location, ghost)
            total_distance += distance_to_ghost

        path_found = total_distance > 0
        return None, total_distance, path_found

    def manhattan_distance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])



    def coordinate_chase_value(self, state, coordinate_location):
        ghost_states_with_times = api.ghostStatesWithTimes(state)

        # Extracting ghost positions as tuples of integers
        ghost_positions = []
        for ghost_state_with_time in ghost_states_with_times:
            ghost_position = ghost_state_with_time[0]  # The position tuple
            x, y = ghost_position  # Unpacking the position tuple
            ghost_positions.append((int(x), int(y)))

        # ghost_positions = [(int(ghost_state[0][0]), int(ghost_state[0][1])) for ghost_state, _ in ghost_states_with_times]

        walls = api.walls(state)
        shortest_paths = {}
        # print "coordinate_location:", coordinate_location
        shortest_total_distance = float('inf')
        shortest_total_path = None  # Initialize with a default non-beneficial path

        # Calculate shortest path from spot to each ghost and between ghosts
        paths_to_ghosts = {}
        for ghost in ghost_positions:
            distance_to_ghost, path_to_ghost = self.a_star_search_with_path(coordinate_location, ghost, walls)
            paths_to_ghosts[ghost] = (distance_to_ghost, path_to_ghost)

        # Calculate shortest paths between ghosts and find the best combination
        for ghost1 in ghost_positions:
            for ghost2 in ghost_positions:
                if ghost1 != ghost2:
                    distance_between_ghosts, path_between_ghosts = self.a_star_search_with_path(ghost1, ghost2, walls)
                    total_distance = paths_to_ghosts[ghost1][0] + distance_between_ghosts

                    if total_distance < shortest_total_distance:
                        shortest_total_distance = total_distance
                        shortest_total_path = paths_to_ghosts[ghost1][1] + path_between_ghosts

        # print "Shortest total path:", shortest_total_path
        # print "Length:", len(shortest_total_path)

        path_found = shortest_total_path is not None
        return shortest_total_path, len(shortest_total_path) if path_found else 0, path_found




    def ghost_hunt(self, state):
        pacman = api.whereAmI(state)
        ghost_states_with_times = api.ghostStatesWithTimes(state)
        walls = api.walls(state)

        food_locations = api.food(state) 
        # Prioritize food if less than 30 pellets remain
        if len(food_locations) < 30:
            print "Prioritizing food over ghosts."
            return False, -1, []

        min_distance = float('inf')
        min_time_remaining = 0
        path_to_closest_ghost = []

        for index, (ghost_state, time) in enumerate(ghost_states_with_times):
            if time > 0:  # Check if the ghost is edible
                ghost_position = (int(ghost_state[0]), int(ghost_state[1]))
                distance, path = self.a_star_search_with_path(pacman, ghost_position, walls)
                if distance < min_distance:
                    min_distance = distance
                    closest_ghost_index = index
                    min_time_remaining = time
                    path_to_closest_ghost = path

        if self.print_all:
            print "Closest Ghost:", closest_ghost_index
            print "Min_distance:", min_distance
            print "Min time remaining:", min_time_remaining
        
        # Check if Pacman should chase the ghost
        if closest_ghost_index >= 0 and min_distance < min_time_remaining:
            ghost_position = ghost_states_with_times[closest_ghost_index][0]
            rounded_ghost_positions = (int(ghost_position[0]), int(ghost_position[1]))

            # print "Ghost position", rounded_ghost_positions
            if rounded_ghost_positions in self.initial_food_locations:
                # print "DEN:", self.initial_food_locations
                return True, closest_ghost_index, path_to_closest_ghost
            else:
                print "Ghost in den, ignoring."
                return False, -1, []
        else:
            print "Ghost outside of max area!"
            return False, -1, []

        # # Check if Pacman should chase the ghost
        # if closest_ghost and min_distance < min_time_remaining:
        #     return True, path_to_closest_ghost
        # else:
            # return False, []

    def a_star_search_with_path(self, start, goal, walls):
        frontier = []
        heapq.heappush(frontier, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}

        while frontier:
            _, current = heapq.heappop(frontier)

            if current == goal:
                break

            for next in self.get_neighbors(current, walls):
                new_cost = cost_so_far[current] + 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(goal, next)
                    heapq.heappush(frontier, (priority, next))
                    came_from[next] = current

        # Reconstruct the path
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()

        return cost_so_far.get(goal, float('inf')), path

    #### Main functions for 1.

    # Ghost proximity (Danger)
    def apply_ghost_mask_with_astar(self, rounded_ghost_positions, walls, state):
        ghost_states = api.ghostStates(state)

        for y in range(self.height):
            for x in range(self.width):
                if self.rewards[y][x] is None:
                    continue  # Skip walls

                # Initialize the reward/penalty for this cell
                ghost_effect = 0

                for ghost_state in ghost_states:
                    # print "Ghost state:", ghost_state
                    ghost_position, ghost_is_edible = ghost_state
                    ghost_x, ghost_y = map(int, ghost_position)

                    # Calculate the distance to the ghost
                    distance = self.a_star_search((x, y), (ghost_x, ghost_y), walls)

                    # Apply different effects based on ghost state
                    if distance <= self.max_ghost_distance:
                        falloff = self.exponential_fall_off(distance, self.decay_rate_danger)

                        if ghost_is_edible:
                            # Increase reward for proximity to edible ghost
                            ghost_effect += self.edible_ghost_reward * falloff
                            # print "Ghost is edible."
                        else:
                            # Apply penalty for proximity to dangerous ghost
                            ghost_effect += self.ghost_reward * falloff
                            # print "Ghost is not edible."

                # Update the reward for the cell
                self.rewards[y][x] += ghost_effect


    def find_closest_food_path(self, pacman_position, food_positions, walls):
        closest_food_distance = float('inf')
        closest_food_path = []
        for food_position in food_positions:
            distance, path = self.a_star_search_with_path(pacman_position, food_position, walls)
            if distance < closest_food_distance:
                closest_food_distance = distance
                closest_food_path = path
        return closest_food_distance, closest_food_path

    # Efficient sweep
    def modify_food_values(self, state):
        food = api.food(state)
        walls = api.walls(state)
        food_clusters = self.calculate_food_clusters(food, walls)

        for food_coord, cluster_size in food_clusters.items():
            x, y = food_coord
            self.rewards[y][x] += self.sweep_bonus / (cluster_size ** 2)

    ### 1. Model Rewards Main Function

    def model_rewards(self, state):
        food = api.food(state)
        # Store initial food locations the first time this method is called
        if not self.initialized:
            self.initial_food_locations = set(food)
            self.initialized = True

        ghost_positions = api.ghosts(state)
        walls = api.walls(state)
        new_capsules = api.capsules(state)
        rounded_ghost_positions = [(int(ghost[0]), int(ghost[1])) for ghost in ghost_positions]

        # Populate initial good locations with food and capsules
        self.initial_good_locations = set(food + new_capsules)
        
        # Initialize the rewards grid with None for walls and -1 for all other cells
        # self.rewards = [[None if (x, y) in walls else -1 for x in range(self.width)] for y in range(self.height)]
        # Initialize the rewards grid
        self.rewards = [
            [None if (x, y) in walls else -20 if (x, y) not in self.initial_good_locations else -1
            for x in range(self.width)] for y in range(self.height)
        ]
        # Mark initial setup as done
        self.initial_setup_done = True

        # Check if chase mode should be deactivated
        ghost_states_with_times = api.ghostStatesWithTimes(state)
        any_ghosts_edible = any(time > 0 for _, time in ghost_states_with_times)
        if self.in_chase_mode and (self.chase_time_remaining <= 0 or not any_ghosts_edible):
            self.in_chase_mode = False
            # print "Chase mode over."
            print "Total Chase Time:", self.total_chase_time, "turns"
            # Reset metrics after printing
            self.total_chase_time = 0
            self.total_chase_distance = 0

        # print "Before food mask"
        # self.print_rewards_grid()

        pacman = api.whereAmI(state)
        # self.apply_food_mask(food)
        # Use the existing A* search function to find the closest food
        closest_food_distance, closest_food_path = self.find_closest_food_path(pacman, food, walls)
        for position in closest_food_path:
            x, y = position
            # Assign a weak but positive reward to guide Pacman
            self.rewards[y][x] = self.weak_path_reward  # define this value appropriately

        # print "After food mask"
        # self.print_rewards_grid()

        # Populate model with food rewards
        for y in range(self.height):
            for x in range(self.width):
                # if (x, y) in rounded_ghost_positions:
                #     self.rewards[y][x] = self.ghost_reward  # Penalty for ghosts
                if (x, y) in food:
                    self.rewards[y][x] = self.food_reward # Reward for food

        # Make pacman run an efficient sweep (less backtracking)
        food_clusters = self.calculate_food_clusters(food, walls)
        for food_coord, cluster_size in food_clusters.items():
            x, y = food_coord
            self.rewards[y][x] += self.sweep_bonus / (cluster_size ** 2)

        # Populate model with capsule rewards
        for capsule in new_capsules:
            capsule_value = 15
            chase_path, path_length, path_found = self.coordinate_chase_value(state, capsule)
            print "Path length:", path_length
            # print "Chase Path:", chase_path
            if path_found:
                capsule_value = 300 / path_length -10
                print "Capsule value:", capsule_value
            else:
                capsule_value = 8 # Set to -20 if no beneficial path is found
            x, y = capsule
            self.rewards[y][x] = capsule_value
            # print "Capsule", capsule, "- value:", capsule_value

        # Ghost hunting

        if self.in_chase_mode:
            self.chase_time_remaining -= 1
            self.total_chase_time += 1

            if self.print_all:
                print "Chase time remaining:", self.chase_time_remaining
            # should_chase, path = self.ghost_hunt(state)
            should_chase, ghost_index, path = self.ghost_hunt(state)
            # If Pacman can't get to both, it will stop

            if should_chase:
                print "Chasing ghost" , ghost_index, "/ Path:", path
                for cell in path:
                    x, y = cell
                    self.rewards[y][x] = self.ghost_path_reward

            if self.chase_time_remaining <= 0:
                self.in_chase_mode = False
                print "Chase time ended."

        for capsule in self.capsules:
            if self.print_all:
                print "Capsule:", capsule
                print "new_capsules", new_capsules
            if capsule not in new_capsules:
                self.in_chase_mode = True
                self.chase_time_remaining = 39  # Reset chase time
                if self.print_all:
                    print "Capsule collected at", capsule, "Starting chase time."
                    print "Chase time remaining:", self.chase_time_remaining
                # should_chase, path = self.ghost_hunt(state)
                should_chase, ghost_index, path = self.ghost_hunt(state)

                if should_chase:
                    print "Chasing ghost" , ghost_index, "/ Path:", path

                    for cell in path:
                        x, y = cell
                        self.rewards[y][x] = self.ghost_path_reward

        # Update the capsules list
        self.capsules = new_capsules

        # Apply Gaussian mask to rewards based on ghost proximity
        self.apply_ghost_mask_with_astar(rounded_ghost_positions, walls, state)









    ### 2. Value Iteration Function 

    def value_iteration(self, state, discount_factor):
        # Retrieve the dimensions of the grid based on the wall locations
        self.height, self.width = self.get_grid_dimensions(state)
        # Initialise the rewards for each cell in the grid
        self.model_rewards(state)
        # Initialise the utilities for each cell in the grid to zero.
        self.utilities = [[0 if self.rewards[y][x] is not None else None for x in range(self.width)] for y in range(self.height)]

        iteration_count = 0

        self.print_rewards_grid()

        while True:
            delta = 0
            new_utilities = [row[:] for row in self.utilities]  # Create a new copy for updated utilities

            for y in range(self.height):
                for x in range(self.width):
                    if self.utilities[y][x] is None:
                        continue  # Skip walls

                    # Initialize a dictionary to store the utilities of moving in all directions
                    move_utilities = {
                        'Up': 0.0,
                        'Left': 0.0,
                        'Down': 0.0,
                        'Right': 0.0
                    }

                    # Get utility of staying in the same place (in case of running into a wall)
                    stay_utility = self.utilities[y][x]

                    # Compute utilities for all possible movements
                    for move, (dy, dx) in [('Up', (-1, 0)), ('Down', (1, 0)), ('Left', (0, -1)), ('Right', (0, 1))]:
                        new_y, new_x = y + dy, x + dx
                        if 0 <= new_x < self.width and 0 <= new_y < self.height and self.utilities[new_y][new_x] is not None:
                            move_utilities[move] += 0.8 * new_utilities[new_y][new_x]
                        else:
                            move_utilities[move] += 0.8 * stay_utility

                        # Add probabilities for perpendicular directions
                        perpendicular_directions = [('Left', (0, -1)), ('Right', (0, 1))] if move in ['Up', 'Down'] else [('Up', (-1, 0)), ('Down', (1, 0))]
                        for perp_move, (p_dy, p_dx) in perpendicular_directions:
                            p_new_y, p_new_x = y + p_dy, x + p_dx
                            if 0 <= p_new_x < self.width and 0 <= p_new_y < self.height and self.utilities[p_new_y][p_new_x] is not None:
                                move_utilities[move] += 0.1 * new_utilities[p_new_y][p_new_x]
                            else:
                                move_utilities[move] += 0.1 * stay_utility

                    # Select the best utility out of all possible moves
                    best_move_utility = max(move_utilities.values())
                    adjusted_reward = self.rewards[y][x]
                    self.utilities[y][x] = adjusted_reward + discount_factor * best_move_utility

                    # Update delta for convergence check
                    delta = max(delta, abs(self.utilities[y][x] - new_utilities[y][x]))

            iteration_count += 1

            # print 
            # print "iteration %d" % (iteration_count)
            # self.print_utilities_grid(state)
            
            # Check for convergence
            if delta < self.threshold:
                if self.print_all:
                    print "Convergence achieved in %d iterations." % (iteration_count)
                break

    ### Helper Function for 0. - this can be dropped, I have a similar function above.

    def get_new_position(self, current_position, move):
        x, y = current_position
        if move == Directions.NORTH:
            new_pos = (x, y + 1)
        elif move == Directions.SOUTH:
            new_pos = (x, y - 1)
        elif move == Directions.EAST:
            new_pos = (x + 1, y)
        elif move == Directions.WEST:
            new_pos = (x - 1, y)
        else:
            return current_position

        # Check if the new position is a wall
        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= self.width or new_pos[1] >= self.height or self.utilities[new_pos[1]][new_pos[0]] is None:
            return current_position  # Return original position if the new position is a wall or out of bounds
        return new_pos
    
    ### 0. Main Function

    def getAction(self, state):
        self.turn += 1
        if self.print_all:
            print "Next Turn:"
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        self.value_iteration(state, self.discount_factor)
        # self.print_utilities_grid(state)

        pacman = api.whereAmI(state)
        best_move, best_utility = self.find_best_move(legal, pacman)

        final_move = api.makeMove(best_move, legal)
        if self.print_all:
            self.print_decision_info(state, pacman, legal, best_move, final_move)

        self.handle_non_determinism(best_move, final_move, state, pacman)
            
        return final_move

    def find_best_move(self, legal_moves, pacman_position):
        best_move = None
        best_utility = -float('inf')
        for move in legal_moves:
            new_pos = self.get_new_position(pacman_position, move)
            if new_pos and self.utilities[new_pos[1]][new_pos[0]] is not None:
                utility = self.utilities[new_pos[1]][new_pos[0]]
                if utility > best_utility:
                    best_utility = utility
                    best_move = move
        return best_move, best_utility

    def print_decision_info(self, legal_moves, best_move, final_move):
        print "\nCurrent Game State:"
        print "Legal moves: ", legal_moves
        print "Best move: " + best_move
        print "Move being made: ", final_move

    def handle_non_determinism(self, best_move, final_move, state, pacman):
        if final_move != best_move:
            self.warning_counter += 1
            self.print_game_grid(state, pacman)

            warning_rate = float(self.warning_counter) / self.turn
            print "Warning: Non-determinism detected!"
            print "Warning #{} (of batch): Best move: {} vs. Final move: {}".format(self.warning_counter, best_move, final_move)
            print "Non-determinism rate: {:.2f}%".format(warning_rate * 100)


    # def value_iteration(self, state, discount_factor):
    #     # Retrieve the dimensions of the grid based on the wall locations
    #     self.height, self.width = self.get_grid_dimensions(state)
    #     # Initialise the rewards for each cell in the grid
    #     self.model_rewards(state)
    #     # Initialise the utilities for each cell in the grid to zero.
    #     self.utilities = [[0 if self.rewards[y][x] is not None else None for x in range(self.width)] for y in range(self.height)]

    #     iteration_count = 0

    #     print "Greedy utility"
    #     self.print_rewards_grid()
        
    #     while True:
    #         delta = 0
    #         utilities_copy = [row[:] for row in self.utilities]

    #         for y in range(self.height):
    #             for x in range(self.width):
    #                 if self.utilities[y][x] is None:
    #                     continue  # Skip walls

    #                 utility_values = []
    #                 for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Neighboring cells
    #                     new_y, new_x = y + dy, x + dx
    #                     if 0 <= new_x < self.width and 0 <= new_y < self.height and utilities_copy[new_y][new_x] is not None:
    #                         utility_values.append(utilities_copy[new_y][new_x])

    #                 if not utility_values:
    #                     continue

    #                 max_expected_utility = max(utility_values)
    #                 adjusted_reward = self.rewards[y][x]

    #                 self.utilities[y][x] = adjusted_reward + discount_factor * max_expected_utility

    #                 delta = max(delta, abs(self.utilities[y][x] - utilities_copy[y][x]))
            
    #         iteration_count += 1

    #         # print 
    #         # print "iteration %d" % (iteration_count)
    #         # self.print_utilities_grid(state)
            
    #         # Check for convergence
    #         if delta < self.threshold:
    #             # print "ghost_utility_adjustment" + str(ghost_utility_adjustment)
    #             print "Convergence achieved in %d iterations." % (iteration_count)

    #             break








# class MyBellmanUpdateAgent(Agent):

#     def __init__(self):
#         self.utilities = None
#         self.rewards = None
#         self.height = 0
#         self.width = 0
#         self.turn = 0

#     # Print Grid
#     def print_game_grid(self, state, pacman):

#         legal = api.legalActions(state)
#         if Directions.STOP in legal:
#             legal.remove(Directions.STOP)
#             # Prevents stopping

#         pacman = api.whereAmI(state)
#         theGhosts = api.ghosts(state)
#         walls = api.walls(state)
#         food_list = api.food(state)
#         capsule_list = api.capsules(state)

#         width = self.width
#         height = self.height
#         grid_size = (width, height)

#         grid = [[' ' for _ in range(grid_size[1])] for _ in range(grid_size[0])]

#         # Place walls
#         for wall in walls:
#             grid[wall[0]][wall[1]] = '#'

#         # Place Pacman
#         grid[pacman[0]][pacman[1]] = '@'

#         for food in food_list:
#             grid[food[0]][food[1]] = '*'

#         for capsule in capsule_list:
#             grid[capsule[0]][capsule[1]] = '~'

#         # Place ghosts
#         for ghost in theGhosts:
#             # Convert ghost position to integers if necessary
#             ghost_pos = (int(ghost[0]), int(ghost[1]))
#             grid[ghost_pos[0]][ghost_pos[1]] = 'G'

#         # Rotate 90 degrees and then flip vertically
#         for col in reversed(range(grid_size[1])):
#             for row in range(grid_size[0]):
#                 print grid[row][col],
#             print
#         print

#     def print_utilities_grid(self, state):
#         print "Utilities Grid:"
#         pacman_position = api.whereAmI(state)
#         for y in reversed(range(self.height)):
#             for x in range(self.width):
#                 # print pacman_position
#                 if (x, y) == pacman_position:
#                     print "   x  ",
            
#                 else:
#                     utility = self.utilities[y][x]
#                     if utility is None:
#                         print "      ",
#                     else:
#                         print "{:6.2f}".format(utility),
#             print
#         print

#     def print_rewards_grid(self):
#         print "Rewards Grid:"
#         for y in reversed(range(self.height)):  # Print from top to bottom
#             for x in range(self.width):
#                 reward = self.rewards[y][x]
#                 # Check if the cell is a wall represented by None
#                 if reward is None:
#                     print "      ",  # Six spaces to align with the "{:6.2f}" format
#                 else:
#                     # Format the reward for printing with two decimal places and a fixed width
#                     print "{:6.2f}".format(reward),
#             print  # Newline at the end of each row
#         print  # Extra newline for spacing







#     def initialise_rewards(self, state):
#         food = api.food(state)
#         ghost_positions = api.ghosts(state)
#         walls = api.walls(state)
#         rounded_ghost_positions = [(int(ghost[0]), int(ghost[1])) for ghost in ghost_positions]
        
#         # Initialize the rewards grid with None for walls and -1 for all other cells
#         self.rewards = [[None if (x, y) in walls else -1 for x in range(self.width)] for y in range(self.height)]

#         for y in range(self.height):
#             for x in range(self.width):
#                 if (x, y) in rounded_ghost_positions:
#                     self.rewards[y][x] = -200  # Penalty for ghosts
#                 elif (x, y) in food:
#                     self.rewards[y][x] = 10    # Reward for food

#     def get_grid_dimensions(self, state):
#         walls = api.walls(state)
#         max_x = max(wall[0] for wall in walls) + 1
#         max_y = max(wall[1] for wall in walls) + 1
#         return max_y, max_x  # Height, Width

#     # 1. Building utility grid
#     def initialise_utilities(self, state):

#         walls = api.walls(state)
#         self.height, self.width = self.get_grid_dimensions(state)
#         self.initialise_rewards(state)
#         self.utilities = [[self.rewards[y][x] if (x, y) not in walls else None for x in range(self.width)] for y in range(self.height)]
#         # self.print_utilities_grid()

#     def get_valid_actions(self, current_position):
#         valid_actions = []
#         x, y = current_position

#         # Check each direction and add to valid actions if it's not a wall or out of bounds
#         if y + 1 < self.height and self.utilities[y + 1][x] is not None:  # North
#             valid_actions.append(Directions.NORTH)
#         if y - 1 >= 0 and self.utilities[y - 1][x] is not None:  # South
#             valid_actions.append(Directions.SOUTH)
#         if x + 1 < self.width and self.utilities[y][x + 1] is not None:  # East
#             valid_actions.append(Directions.EAST)
#         if x - 1 >= 0 and self.utilities[y][x - 1] is not None:  # West
#             valid_actions.append(Directions.WEST)

#         return valid_actions
    
#     def get_the_position(self, current_position, move):
#         x, y = current_position

#         # Calculate the new position based on the action
#         if move == Directions.NORTH:
#             new_pos = (x, y + 1)
#         elif move == Directions.SOUTH:
#             new_pos = (x, y - 1)
#         elif move == Directions.EAST:
#             new_pos = (x + 1, y)
#         elif move == Directions.WEST:
#             new_pos = (x - 1, y)
#         else:
#             return None  # Invalid move

#         # Check if the new position is within the grid and not a wall
#         if 0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height:
#             if self.utilities[new_pos[1]][new_pos[0]] is not None:
#                 return new_pos  # Valid move
#         return None  # Move leads to a wall or out of bounds



#     def value_iteration(self, state, discount_factor, threshold=0.01):
#         # Retrieve the dimensions of the grid based on the wall locations
#         self.height, self.width = self.get_grid_dimensions(state)
#         # Initialise the rewards for each cell in the grid
#         self.initialise_rewards(state)
#         # Initialise the utilities for each cell in the grid to zero.
#         self.utilities = [[0 if self.rewards[y][x] is not None else None for x in range(self.width)] for y in range(self.height)]

#         print
#         self.print_rewards_grid()
        
#         # Repeat until utilities converge
#         while True:
#             delta = 0
#             # Create a copy of the utilities to compare after updates.
#             utilities_copy = [row[:] for row in self.utilities]

#             # Update utilities for each state based on the Bellman equation.
#             for y in range(self.height):
#                 for x in range(self.width):
#                     # Skip the wall cells as they are not part of the state space for the MDP.
#                     if self.rewards[y][x] is None:
#                         continue  # Skip walls and any cells initialized to None




#                     # Calculate the expected utility of the current state based on neighboring states
#                     utility_values = []
#                     for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Neighboring cells (N, S, W, E)
#                         new_y, new_x = y + dy, x + dx
#                         if 0 <= new_x < self.width and 0 <= new_y < self.height and utilities_copy[new_y][new_x] is not None:
#                             utility_values.append(utilities_copy[new_y][new_x])

#                     # If there are no valid neighboring states, skip the update
#                     if not utility_values:
#                         continue

#                     # Bellman update using the maximum expected utility of neighboring states
#                     max_expected_utility = max(utility_values)
#                     self.utilities[y][x] = self.rewards[y][x] + discount_factor * max_expected_utility

#                     # Calculate the change in utility for this state
#                     delta = max(delta, abs(self.utilities[y][x] - utilities_copy[y][x]))

#             # Check for convergence
#             if delta < threshold:
#                 break





#     def get_neighbors(self, x, y, utilities):
#         neighbors = []
#         if x > 0 and utilities[y][x - 1] is not None: neighbors.append((x - 1, y))  # West
#         if x < self.width - 1 and utilities[y][x + 1] is not None: neighbors.append((x + 1, y))  # East
#         if y > 0 and utilities[y - 1][x] is not None: neighbors.append((x, y - 1))  # North
#         if y < self.height - 1 and utilities[y + 1][x] is not None: neighbors.append((x, y + 1))  # South
#         return neighbors


#     def get_new_position(self, current_position, move):
#         x, y = current_position
#         if move == Directions.NORTH:
#             new_pos = (x, y + 1)
#         elif move == Directions.SOUTH:
#             new_pos = (x, y - 1)
#         elif move == Directions.EAST:
#             new_pos = (x + 1, y)
#         elif move == Directions.WEST:
#             new_pos = (x - 1, y)
#         else:
#             return current_position

#         # Check if the new position is a wall
#         if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= self.width or new_pos[1] >= self.height or self.utilities[new_pos[1]][new_pos[0]] is None:
#             return current_position  # Return original position if the new position is a wall or out of bounds
#         return new_pos

#     def getAction(self, state):

#         discount_factor = 0.9 # Higher values makes ghosts more dangerous

#         print "Next Turn:"
        
#         self.value_iteration(state, discount_factor)
#         print "Values converged"
#         print
#         self.print_utilities_grid(state)

#         legal = api.legalActions(state)
#         if Directions.STOP in legal:
#             legal.remove(Directions.STOP)

#         pacman = api.whereAmI(state)
#         best_move = None
#         best_utility = -float('inf')

#         for move in legal:
#             new_pos = self.get_new_position(pacman, move)
#             if new_pos is not None and self.utilities[new_pos[1]][new_pos[0]] is not None:
#                 utility = self.utilities[new_pos[1]][new_pos[0]]
#                 print "Move:", move, "Leads to:", new_pos, "Utility:", utility
#                 if utility > best_utility:
#                     best_utility = utility
#                     best_move = move

#         print "Best move: " + best_move
#         print
#         self.print_game_grid(state, pacman)

#         return api.makeMove(best_move, legal)









class MyDecentUtilityAgent(Agent):
    # Need to get the value iteration process down to a T

    # At the moment, the utilities grid isn't making much sense while values converge; the walls are instantly considered spaces worth moving to.

    # I need to ensure the bellman equation runs, but only for the available spaces. 

    # Wait - it seems the utility grid isn't configuring correctly either?

    def __init__(self):
        self.utilities = None
        self.rewards = None
        self.height = 0
        self.width = 0
        self.turn = 0

    # Print Grid
    def print_game_grid(self, state, pacman):

        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
            # Prevents stopping

        pacman = api.whereAmI(state)
        theGhosts = api.ghosts(state)
        walls = api.walls(state)
        food_list = api.food(state)
        capsule_list = api.capsules(state)

        width = self.width
        height = self.height
        grid_size = (width, height)

        grid = [[' ' for _ in range(grid_size[1])] for _ in range(grid_size[0])]

        # Place walls
        for wall in walls:
            grid[wall[0]][wall[1]] = '#'

        # Place Pacman
        grid[pacman[0]][pacman[1]] = '@'

        for food in food_list:
            grid[food[0]][food[1]] = '*'

        for capsule in capsule_list:
            grid[capsule[0]][capsule[1]] = '~'

        # Place ghosts
        for ghost in theGhosts:
            # Convert ghost position to integers if necessary
            ghost_pos = (int(ghost[0]), int(ghost[1]))
            grid[ghost_pos[0]][ghost_pos[1]] = 'G'

        # Rotate 90 degrees and then flip vertically
        for col in reversed(range(grid_size[1])):
            for row in range(grid_size[0]):
                print grid[row][col],
            print
        print

    def print_utilities_grid(self, state):
        print "Utilities Grid:"
        pacman_position = api.whereAmI(state)
        for y in reversed(range(self.height)):
            for x in range(self.width):
                # print pacman_position
                if (x, y) == pacman_position:
                    print "   x  ",
            
                else:
                    utility = self.utilities[y][x]
                    if utility is None:
                        print "      ",
                    else:
                        print "{:6.2f}".format(utility),
            print
        print

    def initialise_rewards(self, state):
        food = api.food(state)
        ghost_positions = api.ghosts(state)
        rounded_ghost_positions = [(int(ghost[0]), int(ghost[1])) for ghost in ghost_positions]
        print ghost_positions
        print rounded_ghost_positions
        self.rewards = [[-1 for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in rounded_ghost_positions:
                    self.rewards[y][x] = -400  # Penalty for ghosts
                elif (x, y) in food:
                    self.rewards[y][x] = 1  # Reward for food
                else:
                    self.rewards[y][x] = -1  # Small penalty to encourage exploration

    def get_grid_dimensions(self, state):
        walls = api.walls(state)
        max_x = max(wall[0] for wall in walls) + 1
        max_y = max(wall[1] for wall in walls) + 1
        return max_y, max_x  # Height, Width

    # 1. Building utility grid
    def initialise_utilities(self, state):

        walls = api.walls(state)
        self.height, self.width = self.get_grid_dimensions(state)
        self.initialise_rewards(state)
        self.utilities = [[self.rewards[y][x] if (x, y) not in walls else None for x in range(self.width)] for y in range(self.height)]
        # self.print_utilities_grid()

    def get_valid_actions(self, current_position):
        valid_actions = []
        x, y = current_position

        # Check each direction and add to valid actions if it's not a wall or out of bounds
        if y + 1 < self.height and self.utilities[y + 1][x] is not None:  # North
            valid_actions.append(Directions.NORTH)
        if y - 1 >= 0 and self.utilities[y - 1][x] is not None:  # South
            valid_actions.append(Directions.SOUTH)
        if x + 1 < self.width and self.utilities[y][x + 1] is not None:  # East
            valid_actions.append(Directions.EAST)
        if x - 1 >= 0 and self.utilities[y][x - 1] is not None:  # West
            valid_actions.append(Directions.WEST)

        return valid_actions
    
    def get_the_position(self, current_position, move):
        x, y = current_position

        # Calculate the new position based on the action
        if move == Directions.NORTH:
            new_pos = (x, y + 1)
        elif move == Directions.SOUTH:
            new_pos = (x, y - 1)
        elif move == Directions.EAST:
            new_pos = (x + 1, y)
        elif move == Directions.WEST:
            new_pos = (x - 1, y)
        else:
            return None  # Invalid move

        # Check if the new position is within the grid and not a wall
        if 0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height:
            if self.utilities[new_pos[1]][new_pos[0]] is not None:
                return new_pos  # Valid move
        return None  # Move leads to a wall or out of bounds




    def value_iteration(self, state, discount_factor=0.9, threshold=0.01):
        # Update grid dimensions and walls each iteration
        self.height, self.width = self.get_grid_dimensions(state)
        walls = api.walls(state)
        
        utilities = [[0 for _ in range(self.width)] for _ in range(self.height)]
        iteration_count = 0

        def get_neighbors(x, y):
            neighbors = []
            if x > 0 and utilities[y][x - 1] is not None: neighbors.append((x - 1, y))  # West
            if x < self.width - 1 and utilities[y][x + 1] is not None: neighbors.append((x + 1, y))  # East
            if y > 0 and utilities[y - 1][x] is not None: neighbors.append((x, y - 1))  # North
            if y < self.height - 1 and utilities[y + 1][x] is not None: neighbors.append((x, y + 1))  # South
            return neighbors

        while True:
            # Reinitialize rewards based on the current state
            food = api.food(state)
            ghost_positions = api.ghosts(state)
            rewards = [[-1 for _ in range(self.width)] for _ in range(self.height)]
            for y in range(self.height):
                for x in range(self.width):
                    if (x, y) in walls:
                        utilities[y][x] = None
                    elif (x, y) in ghost_positions:
                        rewards[y][x] = -400
                    elif (x, y) in food:
                        rewards[y][x] = 1

            delta = 0
            new_utilities = [[None if utilities[y][x] is None else 0 for x in range(self.width)] for y in range(self.height)]

            for y in range(self.height):
                for x in range(self.width):
                    if utilities[y][x] is None:  # Skip updating utility if it's a wall
                        continue

                    max_utility = -float('inf')
                    for nx, ny in get_neighbors(x, y):
                        utility = utilities[ny][nx]
                        if utility is not None and utility > max_utility:
                            max_utility = utility

                    new_utility = rewards[y][x] + discount_factor * max_utility
                    new_utilities[y][x] = new_utility
                    delta = max(delta, abs(new_utility - utilities[y][x]))

            utilities = new_utilities
            iteration_count += 1

            if delta < threshold:
                break

        self.utilities = utilities
        print "Convergence achieved in %d iterations." % (iteration_count)

    # def value_iteration(self, state, discount_factor=0.9, threshold=0.01):

    #     self.initialise_utilities(state)
    #     iteration_count = 0
    #     utilities = [[self.utilities[y][x] for x in range(self.width)] for y in range(self.height)]

    #     def get_neighbors(x, y):
    #         neighbors = []
    #         if x > 0 and utilities[y][x - 1] is not None: neighbors.append((x - 1, y))  # West
    #         if x < self.width - 1 and utilities[y][x + 1] is not None: neighbors.append((x + 1, y))  # East
    #         if y > 0 and utilities[y - 1][x] is not None: neighbors.append((x, y - 1))  # North
    #         if y < self.height - 1 and utilities[y + 1][x] is not None: neighbors.append((x, y + 1))  # South
    #         return neighbors

    #     while True:
    #         delta = 0
    #         new_utilities = [[None if utilities[y][x] is None else 0 for x in range(self.width)] for y in range(self.height)]

    #         for y in range(self.height):
    #             for x in range(self.width):
    #                 # Skip updating utility if it's a wall
    #                 if utilities[y][x] is None:
    #                     continue
                    
    #                 max_utility = -float('inf')

    #                 # Iterate over all neighboring cells of the current cell (x, y).
    #                 # get_neighbors(x, y) returns a list of coordinates for all adjacent cells.
    #                 for nx, ny in get_neighbors(x, y):
    #                     utility = utilities[ny][nx]
                        
    #                     # Retrieve the utility value of the neighboring cell.
    #                     # utilities[ny][nx] accesses the utility of the cell at coordinates (ny, nx).
    #                     if utility is not None and utility > max_utility:
    #                         max_utility = utility

    #                 # Updating the utility of the current cell
    #                 new_utility = self.rewards[y][x] + discount_factor * max_utility
    #                 new_utilities[y][x] = new_utility
    #                 delta = max(delta, abs(new_utility - utilities[y][x]))

    #         utilities = new_utilities
    #         if delta < threshold:
    #             break

    #     self.utilities = utilities
    #     print "Convergence achieved in %d iterations." % (iteration_count)




    # def value_iteration(self, state, discount_factor=0.9, threshold=0.001):
    #     # Here we iteratively calculate the utility of each state until the utilities converge to a stable set of values (delta<0.001) 

    #     self.initialise_utilities(state)

    #     print
    #     self.print_utilities_grid(state)

    #     iteration_count = 0

    #     while True:
    #         # Initialize new_utilities with None for walls and 0 for other cells
    #         new_utilities = [[None if self.utilities[y][x] is None else 0 for x in range(self.width)] for y in range(self.height)]
    #         delta = 0

    #         for y in range(self.height):
    #             for x in range(self.width):
    #                 if self.utilities[y][x] is None:  # Skip updating utility if it's a wall
    #                     continue

    #                 utility_values = []
    #                 for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
    #                     new_position = self.get_the_position((x, y), action)
                        
    #                     # Skip the action if it leads into a wall or out of bounds
    #                     if new_position is None:
    #                         continue
                        
    #                     new_x, new_y = new_position
    #                     utility_values.append(self.utilities[new_y][new_x])

    #                     # # For the new position, find the maximum utility and update it -- check this
    #                     # new_x, new_y = new_position
    #                     # utility = self.utilities[new_y][new_x]
    #                     # max_utility = max(max_utility, utility)

    #                 # Only update utility if valid actions were found
    #                 if utility_values:
    #                     max_utility = max(utility_values)
                        
    #                     temp_utility = self.rewards[y][x] + discount_factor * max_utility
    #                     new_utilities[y][x] = temp_utility
    #                     delta = max(delta, abs(temp_utility - self.utilities[y][x]))

    #                 # if max_utility != -float('inf'):
    #                 #     temp_utility = self.rewards[y][x] + discount_factor * max_utility
    #                 #     new_utilities[y][x] = temp_utility
    #                 #     delta = max(delta, abs(temp_utility - self.utilities[y][x]))

    #         self.utilities = new_utilities
    #         iteration_count += 1
    #         # print "Iteration: %d" % (iteration_count)

    #         if delta < threshold:
    #             print "Convergence achieved in %d iterations with delta = %.6f" % (iteration_count, delta)
    #             break
    
class MyBellmanAgent(Agent):
    # This agent uses Bellman equation to determine best move

    # Reward is currently using manhattan - switch to Pathfind

    def __init__(self):
        self.utility_grid = None  # Initialize utility grid

    def initialize_agent(self, state):
        self.utility_grid = self.calibrate_utility_grid(state)

    # 
    def calibrate_utility_grid(self, state):
        food_list = api.food(state)
        ghost_positions = api.ghosts(state)
        walls = api.walls(state)
        wallGrid = state.getWalls()
        width, height = wallGrid.width, wallGrid.height

        utility_grid = [[0 for _ in range(height)] for _ in range(width)]

        # Set utilities based on food distance
        for x in range(width):
            for y in range(height):
                if (x, y) not in walls:

                    food_distances = [util.manhattanDistance((x, y), food) for food in food_list]
                    closest_food_distance = min(food_distances) if food_distances else float('inf')

                    ghost_distances = [util.manhattanDistance((x, y), ghost) for ghost in ghost_positions]
                    closest_ghost_distance = min(ghost_distances) if ghost_distances else float('inf')

                    # value of food
                    food_utility = 1.0 / (closest_food_distance + 1)

                    # value of ghosts
                    ghost_penalty = -2.0 / (closest_ghost_distance + 1)

                    utility_grid[x][y] = food_utility + ghost_penalty

                    # Check for proximity to ghosts and set a negative utility
                    # if any(util.manhattanDistance((x, y), ghost) <= 2 for ghost in ghost_positions):
                    #     utility_grid[x][y] = -100  # Large negative utility for ghost proximity

        return utility_grid

    def value_iteration(self, state, utility_grid, discount_factor=0.9, threshold=0.001):
        wallGrid = state.getWalls()
        width, height = wallGrid.width, wallGrid.height

        while True:
            new_utility_grid = [[0 for _ in range(height)] for _ in range(width)]
            delta = 0

            for x in range(width):
                for y in range(height):
                    if not wallGrid[x][y]:
                        legal_actions = api.legalActions(state)

                        # Calculate updated utility using Bellman equation
                        reward = self.get_reward(state, (x, y))

                        # Bellman equation part

                        max_utility = -float('inf')

                        # Iterate over possible actions and calculate expected utility
                        for action in legal_actions:    
                            # Calculate expected utility   
                            expected_utility = self.calculate_expected_utility(state, (x, y), action, utility_grid)
                            max_utility = max(max_utility, expected_utility)

                        # Update utility using the Bellman equation
                        new_utility = reward + discount_factor * max_utility
                        delta = max(delta, abs(new_utility - utility_grid[x][y]))
                        new_utility_grid[x][y] = new_utility
                    else:
                        new_utility_grid[x][y] = utility_grid[x][y]

            utility_grid = new_utility_grid
            if delta > threshold:
                break
        
        return new_utility_grid
    def get_reward(self, state, position):

        food_list = api.food(state)
        ghost_positions = api.ghosts(state)

        # Naive reward structure - change to A* pathfind instead
        food_reward = sum(10.0 / (util.manhattanDistance(position, food) + 1) for food in food_list)

        ghost_penalty = sum(-20.0 / (util.manhattanDistance(position, ghost) + 1) for ghost in ghost_positions)

        reward = food_reward + ghost_penalty

        return reward

    def choose_action(self, state, utility_grid):
        pacman = api.whereAmI(state)
        legal = api.legalActions(state)

        best_move = Directions.STOP
        best_utility = -float('inf')

        for action in legal:
            # Calculate expected utility for each action
            expected_utility = self.calculate_expected_utility(state, pacman, action, utility_grid)
            if expected_utility > best_utility:
                best_utility = expected_utility
                best_move = action

        return best_move, best_utility
    def calculate_expected_utility(self, state, position, action, utility_grid):
            wallGrid = state.getWalls()
            width, height = wallGrid.width, wallGrid.height
            x, y = position

            # Probabilities for different outcomes
            intended_prob = 1.0  # Probability Pac-Man moves in the intended direction
            other_prob = 0.0  # Probability for each other direction (left/right)

            # Calculate utility for the intended direction
            new_position = self.get_new_position(x, y, action, wallGrid)
            if new_position and 0 <= new_position[0] < width and 0 <= new_position[1] < height:
                expected_utility = intended_prob * utility_grid[new_position[0]][new_position[1]]
            else:
                expected_utility = 0  # If the move is not legal, utility is zero

            # Calculate utility for other possible directions
            for alt_action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                if alt_action != action:
                    alt_position = self.get_new_position(x, y, alt_action, wallGrid)
                    if alt_position and 0 <= alt_position[0] < width and 0 <= alt_position[1] < height:
                        expected_utility += other_prob * utility_grid[alt_position[0]][alt_position[1]]

            return expected_utility

    def get_new_position(self, x, y, action, wallGrid):
        if action == Directions.NORTH and y + 1 < wallGrid.height and not wallGrid[x][y+1]:
            return (x, y+1)
        elif action == Directions.SOUTH and y - 1 >= 0 and not wallGrid[x][y-1]:
            return (x, y-1)
        elif action == Directions.EAST and x + 1 < wallGrid.width and not wallGrid[x+1][y]:
            return (x+1, y)
        elif action == Directions.WEST and x - 1 >= 0 and not wallGrid[x-1][y]:
            return (x-1, y)
        return None  # Return None if the move is not legal

    def getAction(self, state):

        # 1. Utility determination
        if self.utility_grid is None:
            # Initialise the agent with strong utilities, to make value iteration faster
            self.initialize_agent(state)
        else:
            # Run value_iteration on the grid
            self.utility_grid = self.value_iteration(state, self.utility_grid)

        # 2. Make move
        # Make the agent move where utility is highest
        best_move, best_utility = self.choose_action(state, self.utility_grid)

        # Call the function to print the game grid
        self.print_utility_grid(state, self.utility_grid)
        self.print_game_grid(state)
        print ''

        print 'best_move: ' + best_move
        print 'best_utility: ' + str(best_utility)

        return api.makeMove(best_move, api.legalActions(state))





    # # Method to print the utility grid
    # def print_utility_grid(self, state, utility_grid):
    #     print("Utility Grid:")
    #     transposed_grid = [list(x) for x in zip(*utility_grid)]

    #     flipped_grid = transposed_grid[::-1]
    #     for row in flipped_grid:
    #         print ' '.join("{:.2f}".format(cell) for cell in row)
    #     print()

    def print_utility_grid(self, state, utility_grid):
        print "Utility Grid:"
        wallGrid = state.getWalls()
        width, height = wallGrid.width, wallGrid.height

        for y in range(height - 1, -1, -1):
            row_str = ""
            for x in range(width):
                if wallGrid[x][y]:
                    row_str += "# "
                else:
                    row_str += "{:.2f} ".format(utility_grid[x][y])
            print row_str
        print
        
    # Print Grid
    def print_game_grid(self, state):

        pacman = api.whereAmI(state)
        theGhosts = api.ghosts(state)
        walls = api.walls(state)
        food_list = api.food(state)
        capsule_list = api.capsules(state)

        wallGrid = state.getWalls()
        width = wallGrid.width
        height = wallGrid.height
        grid_size = (width, height)

        grid = [[' ' for _ in range(grid_size[1])] for _ in range(grid_size[0])]

        # Place walls
        for wall in walls:
            grid[wall[0]][wall[1]] = '#'

        # Place Pacman
        grid[pacman[0]][pacman[1]] = '@'

        for food in food_list:
            grid[food[0]][food[1]] = '*'

        for capsule in capsule_list:
            grid[capsule[0]][capsule[1]] = '~'

        # Place ghosts
        for ghost in theGhosts:
            # Convert ghost position to integers if necessary
            ghost_pos = (int(ghost[0]), int(ghost[1]))
            grid[ghost_pos[0]][ghost_pos[1]] = 'G'

        # Rotate 90 degrees and then flip vertically
        for col in reversed(range(grid_size[1])):
            for row in range(grid_size[0]):
                print grid[row][col],
            print




    def get_new_position(self, current_position, move):
        x, y = current_position
        if move == Directions.NORTH:
            new_pos = (x, y + 1)
        elif move == Directions.SOUTH:
            new_pos = (x, y - 1)
        elif move == Directions.EAST:
            new_pos = (x + 1, y)
        elif move == Directions.WEST:
            new_pos = (x - 1, y)
        else:
            return current_position

        # Check if the new position is a wall
        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= self.width or new_pos[1] >= self.height or self.utilities[new_pos[1]][new_pos[0]] is None:
            return current_position  # Return original position if the new position is a wall or out of bounds
        return new_pos

    def getAction(self, state):
        # self.turn =+ 1
        # print
        print "Next Turn:"
        # print
        # self.print_utilities_grid(state)

        self.value_iteration(state)
        print "Values converged"
        print
        self.print_utilities_grid(state)
        # print

        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        pacman = api.whereAmI(state)
        best_move = None
        best_utility = -float('inf')

        for move in legal:
            new_pos = self.get_new_position(pacman, move)
            utility = self.utilities[new_pos[1]][new_pos[0]]
            if utility > best_utility:
                best_utility = utility
                best_move = move
                print "Best move:" + best_move

        self.print_game_grid(state, pacman)

        return api.makeMove(best_move, legal)











class MyBadUtilityAgent(Agent):
    # Bellman equation hasn't been implemented properly here. I need to go back and have a look at how best to apply it. 

    def __init__(self):
        self.utilities = None
        self.rewards = None
        self.height = 0
        self.width = 0
        self.turn = 0

    # Print Grid
    def print_game_grid(self, state, pacman):

        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
            # Prevents stopping

        pacman = api.whereAmI(state)
        theGhosts = api.ghosts(state)
        walls = api.walls(state)
        food_list = api.food(state)
        capsule_list = api.capsules(state)

        width = self.width
        height = self.height
        grid_size = (width, height)

        grid = [[' ' for _ in range(grid_size[1])] for _ in range(grid_size[0])]

        # Place walls
        for wall in walls:
            grid[wall[0]][wall[1]] = '#'

        # Place Pacman
        grid[pacman[0]][pacman[1]] = '@'

        for food in food_list:
            grid[food[0]][food[1]] = '*'

        for capsule in capsule_list:
            grid[capsule[0]][capsule[1]] = '~'

        # Place ghosts
        for ghost in theGhosts:
            # Convert ghost position to integers if necessary
            ghost_pos = (int(ghost[0]), int(ghost[1]))
            grid[ghost_pos[0]][ghost_pos[1]] = 'G'

        # Rotate 90 degrees and then flip vertically
        for col in reversed(range(grid_size[1])):
            for row in range(grid_size[0]):
                print grid[row][col],
            print
        print
        
        # Get Pac-Man's current position
        pacman_x, pacman_y = pacman

        # Define the range for the 5x5 grid centered on Pac-Man
        grid_range = 2  # Grid range of 2 cells in each direction from Pac-Man

        # for dy in range(grid_range, -grid_range - 1, -1):
        #     for dx in range(-grid_range, grid_range + 1):
        #         x = pacman_x + dx
        #         y = pacman_y + dy

        #         # Check if the position is within the grid bounds
        #         if 0 <= x < self.width and 0 <= y < self.height:
        #             utility = self.utilities[y][x]
        #             if utility is not None:
        #                 print "{:6.2f}".format(utility),
        #             else:
        #                 print "  Wall",
        #         else:
        #             print "     ",
        #     print  # Newline for the next row
        # print

    def print_utilities_grid(self):
        print "Utilities Grid:"
        for y in reversed(range(self.height)):
            for x in range(self.width):
                utility = self.utilities[y][x]
                if utility is None:
                    print "      ",
                else:
                    print "{:6.2f}".format(utility),
            print
        print

    def initialise_rewards(self, state):
        food = api.food(state)
        ghosts = api.ghosts(state)
        self.rewards = [[-1 for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in food:
                    self.rewards[y][x] = 10  # Reward for food
                elif (x, y) in ghosts:
                    self.rewards[y][x] = -200  # Penalty for ghosts
                else:
                    self.rewards[y][x] = -1  # Small penalty to encourage exploration

    def get_grid_dimensions(self, state):
        walls = api.walls(state)
        max_x = max(wall[0] for wall in walls) + 1
        max_y = max(wall[1] for wall in walls) + 1
        return max_y, max_x  # Height, Width

    # 1. Building utility grid
    def initialise_utilities(self, state):

        walls = api.walls(state)
        self.height, self.width = self.get_grid_dimensions(state)
        self.initialise_rewards(state)
        self.utilities = [[self.rewards[y][x] if (x, y) not in walls else None for x in range(self.width)] for y in range(self.height)]
        # self.print_utilities_grid()

    def get_new_position(self, current_position, move):
        x, y = current_position
        if move == Directions.NORTH:
            new_pos = (x, y + 1)
        elif move == Directions.SOUTH:
            new_pos = (x, y - 1)
        elif move == Directions.EAST:
            new_pos = (x + 1, y)
        elif move == Directions.WEST:
            new_pos = (x - 1, y)
        else:
            return current_position

        # Check if the new position is a wall
        if new_pos[0] < 0 or new_pos[1] < 0 or new_pos[0] >= self.width or new_pos[1] >= self.height or self.utilities[new_pos[1]][new_pos[0]] is None:
            return current_position  # Return original position if the new position is a wall or out of bounds
        return new_pos

    # def calculate_expected_utility(self, position, action):
    #     new_position = self.get_new_position(position, action)
    #     x, y = new_position
    #     if self.utilities[y][x] is None:
    #         return float('-inf')  # Return a very low utility for invalid or wall positions
    #     return self.utilities[y][x]

    def get_valid_actions(self, current_position):
        valid_actions = []
        x, y = current_position

        # Check each direction and add to valid actions if it's not a wall or out of bounds
        if y + 1 < self.height and self.utilities[y + 1][x] is not None:  # North
            valid_actions.append(Directions.NORTH)
        if y - 1 >= 0 and self.utilities[y - 1][x] is not None:  # South
            valid_actions.append(Directions.SOUTH)
        if x + 1 < self.width and self.utilities[y][x + 1] is not None:  # East
            valid_actions.append(Directions.EAST)
        if x - 1 >= 0 and self.utilities[y][x - 1] is not None:  # West
            valid_actions.append(Directions.WEST)

        return valid_actions

    def value_iteration(self, state, discount_factor=0.9, threshold=0.001):
        # Here we iteratively calculate the utility of each state until the utilities converge to a stable set of values (delta<0.001) 

        self.initialise_utilities(state)

        iteration_count = 0

        while True:
            new_utilities = [[0 for _ in range(self.width)] for _ in range(self.height)]
            delta = 0

            # self.print_utilities_grid(state)

            # for each cell in the grid that isn't a wall...
            for y in range(self.height):
                for x in range(self.width):
                    if self.utilities[y][x] is None:
                        continue
                    # print "For (x, y): %d, %d" % (x, y)

                    # Consider each legal action and calculate the utility of moving there, then return max utility
                    # max_utility = -float('inf')
                    # for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
                    #     utility = self.calculate_expected_utility((x, y), action)
                    #     max_utility = max(max_utility, utility)
                    valid_actions = self.get_valid_actions((x, y))
                    max_utility = -float('inf')

                    for action in valid_actions:
                        # Calculate new position based on the action
                        if action == Directions.NORTH:
                            new_pos = (x, y + 1)
                        elif action == Directions.SOUTH:
                            new_pos = (x, y - 1)
                        elif action == Directions.EAST:
                            new_pos = (x + 1, y)
                        elif action == Directions.WEST:
                            new_pos = (x - 1, y)

                        # Get utility of the new position
                        utility = self.utilities[new_pos[1]][new_pos[0]]
                        max_utility = max(max_utility, utility)

                    # We use the best route move out of that square which gives 'temp_utility', adding to new_utilities
                    temp_utility = self.rewards[y][x] + discount_factor * max_utility # U(s) = R(s) + discount_rate * max(U(s'))
                    new_utilities[y][x] = temp_utility

                    # If delta falls below threshold, then things have stabilised
                    delta = max(delta, abs(temp_utility - self.utilities[y][x]))
            
            self.utilities = new_utilities
            iteration_count += 1
            print "Iteration: %d" % (iteration_count)
            if delta < threshold:
                print "Convergence achieved in %d iterations with delta = %.6f" % (iteration_count, delta)
                break

    def getAction(self, state):
        # self.turn =+ 1
        # print
        # print "Turn %d" % (self.turn)
        # print

        self.value_iteration(state)
        print "Values converged"
        print
        self.print_utilities_grid(state)
        print

        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        pacman = api.whereAmI(state)
        best_move = None
        best_utility = -float('inf')

        for move in legal:
            new_pos = self.get_new_position(pacman, move)
            utility = self.utilities[new_pos[1]][new_pos[0]]
            if utility > best_utility:
                best_utility = utility
                best_move = move
                print "Best move:" + best_move

        self.print_game_grid(state, pacman)

        return api.makeMove(best_move, legal)











# Next: Take My Greedy Agent and add Bellman equation (lab 5 -> 6)

class MyGreedyAgent(Agent):
    # This agent has a naive method for checking where food is (Manhattan distance) - so gets stuck
    # It avoids ghost at the last minute
    # It generates a map

    # 1. For each legal position
    def get_new_position(self, current_position, move):
        x, y = current_position
        if move == Directions.NORTH:
            return (x, y+1)
        elif move == Directions.SOUTH:
            return (x, y-1)
        elif move == Directions.EAST:
            return (x+1, y)
        elif move == Directions.WEST:
            return (x-1, y)
        return current_position

    # 2. Calculate utility
    def get_closest_food_distance(self, position, food_list):
        if not food_list:  # Check if the list is empty
            return float('inf')  # No food means distance is infinite
        distances = [util.manhattanDistance(position, food) for food in food_list]
        return min(distances)
    
    # 3. Check for ghost
    def is_ghost_too_close(self, position, ghost_positions, threshold=2):
        return any(util.manhattanDistance(position, ghost) <= threshold for ghost in ghost_positions)

    # Print Grid
    def print_game_grid(self, pacman, ghosts, walls, food_list, grid_size, capsule_list):
        grid = [[' ' for _ in range(grid_size[1])] for _ in range(grid_size[0])]

        # Place walls
        for wall in walls:
            grid[wall[0]][wall[1]] = '#'

        # Place Pacman
        grid[pacman[0]][pacman[1]] = '@'

        for food in food_list:
            grid[food[0]][food[1]] = '*'

        for capsule in capsule_list:
            grid[capsule[0]][capsule[1]] = '~'

        # Place ghosts
        for ghost in ghosts:
            # Convert ghost position to integers if necessary
            ghost_pos = (int(ghost[0]), int(ghost[1]))
            grid[ghost_pos[0]][ghost_pos[1]] = 'G'

        # Rotate 90 degrees and then flip vertically
        for col in reversed(range(grid_size[1])):
            for row in range(grid_size[0]):
                print grid[row][col],
            print

    def getAction(self, state):
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
            # Prevents stopping

        pacman = api.whereAmI(state)
        theGhosts = api.ghosts(state)
        walls = api.walls(state)
        food_list = api.food(state)
        capsule_list = api.capsules(state)

        wallGrid = state.getWalls()
        width = wallGrid.width
        height = wallGrid.height
        grid_size = (width, height)
        
        best_score = -float('inf')
        best_move = Directions.STOP

        # 1. For each position
        # 2. Calculate which is closest to food
        # 3. Check for ghost
        # Print layout

        for move in legal:

            new_pacman = self.get_new_position(pacman, move)
            score = -self.get_closest_food_distance(new_pacman, food_list)

            if self.is_ghost_too_close(new_pacman, theGhosts):
                score -= 100

            # Update best score and move until each possible choice is analysed
            if score > best_score:
                best_score = score
                best_move = move

        # Call the function to print the game grid
        self.print_game_grid(pacman, theGhosts, walls, food_list, grid_size, capsule_list)
        print ''

        return api.makeMove(best_move, legal)








# This Agent considers food and ghosts when moving - it is one larger state. 
# It uses BFS assigning values to a map. Considers food and ghosts.
# It should work on many APIs.
class ScaredAgent(Agent):

    def getAction(self, state):
        
        legal = api.legalActions(state)
        # print "Legal moves: ", legal

                # Where is Pacman?
        pacman = api.whereAmI(state)
        # print "Pacman position: ", pacman

        # Where are the ghosts?
        # print "Ghost positions:"
        theGhosts = api.ghosts(state)
        # for i in range(len(theGhosts)):
            # print theGhosts[i]

        # Where are the capsules?
        # print "Capsule locations:"
        # print api.capsules(state)
        
        # Where is the food?
        # print "Food locations: "
        # print api.food(state)

        # Where are the walls?
        # print "Wall locations: "
        walls = api.walls(state)
        # print walls

        wallGrid = state.getWalls()
        width = wallGrid.width
        height = wallGrid.height
        grid_size = (width, height)

        # How far away are the ghosts?
        # print "Manhattan distance to ghosts:"
        for i in range(len(theGhosts)):
            runway = util.manhattanDistance(pacman,theGhosts[i])
            # print 'Ghost',i+1,'is:', runway, 'squares away'

        # Prepare grid for BFS
        grid = [[' ' for _ in range(grid_size[1])] for _ in range(grid_size[0])]
        for wall in walls:
            grid[wall[0]][wall[1]] = '#'

        # Calculate and print path lengths to each ghost
        # print "Path length to ghosts:"
        for i, ghost in enumerate(theGhosts):
            path_length = self.bfs_path(grid, pacman, (int(ghost[0]), int(ghost[1])))
            # print 'Ghost', i+1, 'is:', path_length, 'squares away'

        # Locate the closest food
        food_list = api.food(state)
        closest_food = None
        closest_food_distance = float('inf')

        for food in food_list:
            distance = self.bfs_path(grid, pacman, food)
            if distance < closest_food_distance:
                closest_food_distance = distance
                closest_food = food

        # Evaluate each legal move
        best_move = Directions.STOP
        best_score = -float('inf')
        
        for move in legal:
            if move == Directions.STOP:
                continue

            new_pacman = self.get_new_position(pacman, move)

            # Safety check
            if any(util.manhattanDistance(new_pacman, ghost) <= 1 for ghost in theGhosts):
                continue

            # Scoring
            min_path_length_to_ghosts = min([self.bfs_path(grid, new_pacman, (int(ghost[0]), int(ghost[1]))) for ghost in theGhosts])
            new_legal_moves = self.get_legal_moves(new_pacman, walls, grid_size)
            num_options = len(new_legal_moves)

            # Food seeking
            food_score = 0
            if closest_food:
                food_distance = self.bfs_path(grid, new_pacman, closest_food)
                food_score = -food_distance * 20  # Increase weight for food proximity

            score = min_path_length_to_ghosts * 10 + num_options + food_score

            if score > best_score:
                best_score = score
                best_move = move

        # Call the function to print the game grid
        self.print_game_grid(pacman, theGhosts, walls, grid_size)
        return api.makeMove(best_move, legal)
    


    def print_game_grid(self, pacman, ghosts, walls, grid_size):
        grid = [[' ' for _ in range(grid_size[1])] for _ in range(grid_size[0])]

        # Place walls
        for wall in walls:
            grid[wall[0]][wall[1]] = '#'

        # Place Pacman
        grid[pacman[0]][pacman[1]] = '@'

        # Place ghosts
        for ghost in ghosts:
            # Convert ghost position to integers if necessary
            ghost_pos = (int(ghost[0]), int(ghost[1]))
            grid[ghost_pos[0]][ghost_pos[1]] = 'G'

        # Rotate 90 degrees and then flip vertically
        for col in reversed(range(grid_size[1])):
            for row in range(grid_size[0]):
                print grid[row][col],
            print



    def bfs_path(self, grid, start, goal):
        queue = deque([[start]])
        seen = set([start])
        while queue:
            path = queue.popleft()
            x, y = path[-1]
            if (x, y) == goal:
                return len(path) - 1  # Subtract 1 because start counts as a step
            for x2, y2 in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:  # Neighbors
                if 0 <= x2 < len(grid) and 0 <= y2 < len(grid[0]) and grid[x2][y2] != '#' and (x2, y2) not in seen:
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))
                    # print('Seen:', seen)

        return None
    

    def get_new_position(self, current_position, move):
        x, y = current_position
        if move == Directions.NORTH:
            return (x, y+1)
        elif move == Directions.SOUTH:
            return (x, y-1)
        elif move == Directions.EAST:
            return (x+1, y)
        elif move == Directions.WEST:
            return (x-1, y)
        return current_position
    
    def get_legal_moves(self, position, walls, grid_size):
        legal_moves = []
        for move in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            new_position = self.get_new_position(position, move)
            if new_position not in walls and 0 <= new_position[0] < grid_size[0] and 0 <= new_position[1] < grid_size[1]:
                legal_moves.append(move)
        return legal_moves