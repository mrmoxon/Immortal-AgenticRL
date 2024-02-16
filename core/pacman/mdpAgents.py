# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py

from core.pacman.main import Directions
from game import Agent
import agent.api as api
import random
import game
import util

import math
import heapq
import Queue

class MDPAgent(Agent):

    def __init__(self):
        self.utilities = None
        self.rewards = None
        self.height = 0
        self.width = 0
        self.turn = 0

        self.threshold = 0.01
        self.discount_factor = 0.9

        self.ghost_reward = -300  # Penalty for ghosts
        self.food_reward = 10     # Reward for food
        self.default_reward = -1  # Default reward for empty cells

        # For ghost proximity
        self.max_ghost_distance = 7
        self.decay_rate_danger = 0.5

        # For efficient sweep
        self.sweep_bonus = 10
        self.weak_path_reward = 1

        # For ghost hunting
        self.capsules = []
        self.in_chase_mode = False
        self.chase_time_remaining = 0
        self.ghost_path_reward = 15
        self.total_chase_time = 0
        self.total_chase_distance = 0

        # Capsules
        self.step_up = 11
        self.initialised = False
        self.initial_good_locations = set()
        self.edible_ghost_reward = 0

        # Aesthetics
        self.print_all = False 
        self.warning_counter = 0 
        self.turn = 0

        self.moves = [('Up', (-1, 0)), ('Down', (1, 0)), ('Left', (0, -1)), ('Right', (0, 1))]

    # Functions for getAction (3)

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
        return best_move
    
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
    
    def handle_non_determinism(self, best_move, final_move, state, pacman):
        if self.print_all:
            if final_move != best_move:
                self.warning_counter += 1
                warning_rate = float(self.warning_counter) / self.turn
                print "Warning: Non-determinism detected!"
                print "Warning #{} (of batch): Best move: {} vs. Final move: {}".format(self.warning_counter, best_move, final_move)
                print "Non-determinism rate: {:.2f}%".format(warning_rate * 100)
                print
            
    ### Helper functions for Model (1) and Value Iterate (2)

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
    
    def manhattan_distance(self, point1, point2):
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
        
    def get_neighbors(self, node, walls):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        result = []
        for dx, dy in directions:
            next = (node[0] + dx, node[1] + dy)
            if next not in walls:
                result.append(next)
        return result
        
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

                # Store the size in temporary dictionary
                for member in cluster_members:
                    cluster_sizes[member] = cluster_size

        # We've calculated the sizes for each cluster, now assign them
        for food_coord, size in cluster_sizes.items():
            food_clusters[food_coord] = size

        return food_clusters
    
    def find_closest_food_path(self, pacman_position, food_positions, walls):
        closest_food_distance = float('inf')
        closest_food_path = []
        for food_position in food_positions:
            distance, path = self.a_star_search_with_path(pacman_position, food_position, walls)
            if distance < closest_food_distance:
                closest_food_distance = distance
                closest_food_path = path
        return closest_food_path

    def ghost_hunt(self, state):
        pacman = api.whereAmI(state)
        ghost_states_with_times = api.ghostStatesWithTimes(state)
        walls = api.walls(state)
        food_locations = api.food(state) 

        # Prioritize food if there is high risk
        if len(self.initial_good_locations) <= 60 and len(food_locations) < 50:
            # print "Prioritising food over ghosts."
            return False, []

        min_distance = float('inf')
        min_time_remaining = 0
        path_to_closest_ghost = []

        # Calculate route to ghosts
        for index, (ghost_state, time) in enumerate(ghost_states_with_times):
            if time > 0:  # Check if the ghost is edible
                ghost_position = (int(ghost_state[0]), int(ghost_state[1]))
                distance, path = self.a_star_search_with_path(pacman, ghost_position, walls)
                if distance < min_distance:
                    min_distance = distance
                    closest_ghost_index = index
                    min_time_remaining = time
                    path_to_closest_ghost = path
        
        # Check if Pacman should chase the ghost
        if closest_ghost_index >= 0 and min_distance < min_time_remaining:
            ghost_position = ghost_states_with_times[closest_ghost_index][0]
            rounded_ghost_positions = (int(ghost_position[0]), int(ghost_position[1]))

            # print "Ghost position", rounded_ghost_positions
            if rounded_ghost_positions in self.initial_good_locations:
                return True, path_to_closest_ghost
            else:
                # print "Ghost in den, ignoring."
                return False, []
        else:
            # print "Ghost outside of max area!"
            return False, []
        
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

    #### Main functions for 1.

    def calibrate_food(self, pacman, food, walls):
        # Use the existing A* search function to find the closest food
        closest_food_path = self.find_closest_food_path(pacman, food, walls)
        for position in closest_food_path:
            x, y = position
            self.rewards[y][x] = self.weak_path_reward

        # Populate model with food rewards
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in food:
                    self.rewards[y][x] = self.food_reward # Reward for food

        # Make pacman run an efficient sweep (less backtracking)
        food_clusters = self.calculate_food_clusters(food, walls)
        for food_coord, cluster_size in food_clusters.items():
            x, y = food_coord
            self.rewards[y][x] += self.sweep_bonus / (cluster_size ** 2.5)

    def update_hunt(self, state):

        should_chase, path = self.ghost_hunt(state)

        if should_chase:
            for x, y in path:
                self.rewards[y][x] = self.ghost_path_reward

    def calibrate_capsules(self, state, new_capsules):
        for capsule in new_capsules:
            capsule_value = 10
            chase_path, path_length, path_found = self.coordinate_chase_value(state, capsule)
            if path_found:
                if path_length > self.step_up:
                    capsule_value = 5  # Set value around 15 for longer paths
                else:
                    capsule_value = 50 + (300 / path_length) # Increase value for shorter paths
            else:
                capsule_value = 2  # Set to -20 if no beneficial path is found
            x, y = capsule
            self.rewards[y][x] = capsule_value

        # Update the capsules list
        self.capsules = new_capsules

    # Ghost proximity (Danger)
    def apply_ghost_mask_with_astar(self, walls, state):
        ghost_states = api.ghostStates(state)

        for y in range(self.height):
            for x in range(self.width):
                if self.rewards[y][x] is None:
                    continue  # Skip walls

                ghost_effect = 0
                for ghost_state in ghost_states:
                    ghost_position, ghost_is_edible = ghost_state
                    ghost_x, ghost_y = map(int, ghost_position)

                    manhattan_distance = abs(x - ghost_x) + abs(y - ghost_y)

                    if manhattan_distance <= self.max_ghost_distance:
                        # Calculate the precise distance to the ghost using A*
                        distance, path = self.a_star_search_with_path((x, y), (ghost_x, ghost_y), walls)

                        # Apply different effects based on ghost state
                        if distance <= self.max_ghost_distance:
                            falloff = self.exponential_fall_off(distance, self.decay_rate_danger)

                            if ghost_is_edible:
                                ghost_effect += self.edible_ghost_reward * falloff # Increase reward for proximity to edible ghost
                            else:
                                ghost_effect += self.ghost_reward * falloff # Apply penalty for proximity to dangerous ghost

                # Update the reward for the cell
                self.rewards[y][x] += ghost_effect

    ### 1. Model Rewards Main Function

    def model_rewards(self, state):
        # print "Hello"
        pacman = api.whereAmI(state)
        food = api.food(state)
        ghost_positions = api.ghosts(state)
        walls = api.walls(state)
        new_capsules = api.capsules(state)
        ghost_states_with_times = api.ghostStatesWithTimes(state)

        if not self.initialised:
            self.initial_good_locations = set(food + new_capsules)
            self.initialised = True
            print "Game Start"

        # Initialise baseline rewards
        self.rewards = [
            [None if (x, y) in walls else -20 if (x, y) not in self.initial_good_locations else -1
            for x in range(self.width)] for y in range(self.height)]
        
        # If chase mode should be deactivated
        any_ghosts_edible = any(time > 0 for _, time in ghost_states_with_times)
        if self.in_chase_mode and (self.chase_time_remaining <= 0 or not any_ghosts_edible):
            self.in_chase_mode, self.total_chase_time, self.total_chase_distance = False, 0, 0

        # Food
        self.calibrate_food(pacman, food, walls)

        # Ghost hunt update
        if self.in_chase_mode: 
            self.chase_time_remaining -= 1
            self.total_chase_time += 1

            self.update_hunt(state)

            if self.chase_time_remaining <= 0: # If out of time, stop chasing
                self.in_chase_mode = False

        # Activate Ghost Hunt
        for capsule in self.capsules: # If capsule is eaten, enter chase mode
            if capsule not in new_capsules:
                self.in_chase_mode, self.chase_time_remaining = True, 39

                self.update_hunt(state)

        # Capsules
        self.calibrate_capsules(state, new_capsules)

        # Apply mask to spaces based on ghost proximity
        self.apply_ghost_mask_with_astar(walls, state)

    ### 2. Value Iteration Function 

    def value_iteration(self, state, discount_factor):
        self.height, self.width = self.get_grid_dimensions(state)

        # Initialise rewards
        self.model_rewards(state)

        iteration_count = 0
        self.utilities = [[0 if self.rewards[y][x] is not None else None for x in range(self.width)] for y in range(self.height)]
        delta = float('inf')

        while delta >= self.threshold:
            delta = 0
            new_utilities = [row[:] for row in self.utilities] # Copy for updated utilities

            for y in range(self.height):
                for x in range(self.width):
                    if self.utilities[y][x] is None:
                        continue  # Skip walls

                    # Initialize a dictionary to store the utilities of moving in all directions
                    move_utilities = {'Up': 0.0, 'Left': 0.0, 'Down': 0.0, 'Right': 0.0}
                    stay_utility = self.utilities[y][x] # Get utility of staying in the same place (in case of running into a wall)

                    # Compute utilities for all possible movements
                    for move, (dy, dx) in self.moves:
                        new_y, new_x = y + dy, x + dx

                        if 0 <= new_x < self.width and 0 <= new_y < self.height and self.utilities[new_y][new_x] is not None:
                            move_utilities[move] += 0.8 * new_utilities[new_y][new_x]
                        else:
                            move_utilities[move] += 0.8 * stay_utility

                        perpendicular_directions = [('Left', (0, -1)), ('Right', (0, 1))] if move in ['Up', 'Down'] else [('Up', (-1, 0)), ('Down', (1, 0))]
                        for perp_move, (p_dy, p_dx) in perpendicular_directions:
                            p_new_y, p_new_x = y + p_dy, x + p_dx
                            if 0 <= p_new_x < self.width and 0 <= p_new_y < self.height and self.utilities[p_new_y][p_new_x] is not None:
                                move_utilities[move] += 0.1 * new_utilities[p_new_y][p_new_x]
                            else:
                                move_utilities[move] += 0.1 * stay_utility

                    best_move_utility = max(move_utilities.values()) # Select the best utility out of all possible moves
                    adjusted_reward = self.rewards[y][x]
                    self.utilities[y][x] = adjusted_reward + discount_factor * best_move_utility

                    delta = max(delta, abs(self.utilities[y][x] - new_utilities[y][x])) # Update delta for convergence check
                    
            iteration_count += 1
    
    ### 0. Main Function

    def getAction(self, state):
        self.turn += 1
        pacman = api.whereAmI(state)
        legal = api.legalActions(state)

        self.value_iteration(state, self.discount_factor)
        # self.print_utilities_grid(state)

        best_move = self.find_best_move(legal, pacman)
        final_move = api.makeMove(best_move, legal)

        self.handle_non_determinism(best_move, final_move, state, pacman)

        if self.print_all:
            self.print_game_grid(state, pacman)
            self.print_grid('rewards', pacman)
            self.print_grid('U', pacman)
            
        return final_move





    def print_game_grid(self, state, pacman):
        print self.turn
        print
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

    def print_grid(self, grid_type, pacman):
        grid_name = "Rewards Grid:" if grid_type == 'rewards' else "Utilities Grid:"
        print grid_name

        for y in reversed(range(self.height)):
            for x in range(self.width):
                if (x, y) == pacman:
                    print "   x  ",
                else:
                    value = self.rewards[y][x] if grid_type == 'rewards' else self.utilities[y][x]
                    print "      " if value is None else "{:6.2f}".format(value),
            print
