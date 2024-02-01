# -*- coding: utf-8 -*-


# sampleAgents.py
# parsons/07-oct-2017
#
# Version 1.1
#
# Some simple agents to work with the PacMan AI projects from:
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

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
from collections import deque
import math

# RandomAgent
#
# A very simple agent. Just makes a random pick every time that it is
# asked for an action.
class RandomAgent(Agent):

    def getAction(self, state):
        # Get the actions we can try, and remove "STOP" if that is one of them to avoid stopping
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # Random choice between the legal options.
        return api.makeMove(random.choice(legal), legal)





# Initialize an empty grid
initial_grid = [[' ' for _ in range(20)] for _ in range(11)]
last_pacman_position = None
last_ghost_positions = set()

def visualise_grid(width, height, food_positions, wall_positions, ghost_positions, grid, pacman):

    global last_pacman_position, last_ghost_positions

    # print("Ghost positions: ", ghost_positions)
    ghost_positions = [(int(math.floor(x)), int(math.floor(y))) for x, y in ghost_positions]
    # print("Last ghost positions: ", last_ghost_positions)

    if last_pacman_position:
        last_x, last_y = last_pacman_position
        grid[last_y][last_x] = ' '

    # Clear last Ghost positions
    for (x, y) in last_ghost_positions:
        x, y = map(int, (x, y))  # convert to int
        grid[y][x] = ' '

        # grid = grid[int(y)][int(x)]
        # grid[math.floor(y)][math.floor(x)] = ' '

    for y in reversed(range(height)):
        for x in range(width):
            if (x, y) == pacman:
                grid[y][x] = 'O'
                last_pacman_position = (x, y)
            elif (x, y) in wall_positions:
                grid[y][x] = '@'
            elif (x, y) in ghost_positions:
                grid[y][x] = '!'
            elif (x, y) in food_positions:
                grid[y][x] = '.'
    
    # Update last ghost positions
    last_ghost_positions = set(ghost_positions)

    for row in reversed(grid):
        print(''.join(row))





def propagate_utility(start_positions, wall_positions, width, height, initial_utility, decay_factor):
    utility_map = [[0 for _ in range(width)] for _ in range(height)]
    visited = set()

    # Initialize BFS queue with start positions and their initial utilities
    queue = deque([(pos, initial_utility) for pos in start_positions])

    while queue:
        (x, y), utility = queue.popleft()

        if (x, y) in visited:
            continue

        visited.add((x, y))

        # Update utility_map with the maximum utility
        # utility_map[y][x] = max(utility_map[y][x], utility)
        utility_map[y][x] += utility  # Sum up utilities

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x, next_y = x + dx, y + dy
            next_pos = (next_x, next_y)

            if 0 <= next_x < width and 0 <= next_y < height:
                if next_pos not in wall_positions and next_pos not in visited:
                    # Decay the utility value as it propagates
                    next_utility = utility * decay_factor

                    if abs(next_utility) > 0.001:  # Stop propagating if utility is too low
                        queue.append((next_pos, next_utility))

    return utility_map





def find_utility(state, food_positions):

    def food_utility_map(state, food_positions, wall_positions):
        
        # print("Start positions for food:", food_positions)  # Debug print statement
        food_utility_map = propagate_utility(food_positions, wall_positions, 20, 11, 10, 0.9)
        return food_utility_map

    def ghost_utility_map(state, ghost_positions, wall_positions):

        # print("Start positions for ghosts:", ghost_positions)  # Debug print statement
        ghost_utility_map = propagate_utility(ghost_positions, wall_positions, 20, 11, -25, 0.9)
        return ghost_utility_map
    


    ghost_positions = [(int(x), int(y)) for x, y in api.ghosts(state)]  # Convert to integers
    wall_positions = api.walls(state)

    food_utility_map = food_utility_map(state, food_positions, wall_positions)
    ghost_utility_map = ghost_utility_map(state, ghost_positions, wall_positions)

    # Combine the utilities
    combined_utility_map = [[0 for _ in range(20)] for _ in range(11)]
    for y in range(11):
        for x in range(20):
            combined_utility_map[y][x] = food_utility_map[y][x] + ghost_utility_map[y][x]

    # Define possible moves for agent in current position
    agent_x, agent_y = api.whereAmI(state)
    moves = {'North': (0, 1), 'East': (1, 0), 'South': (0, -1), 'West': (-1, 0)}

    print("Utility in each direction:")
    for direction, (dx, dy) in moves.items():
        next_x, next_y = agent_x + dx, agent_y + dy
        if 0 <= next_x < 20 and 0 <= next_y < 11:  # Replace with actual grid size
            if (next_x, next_y) not in wall_positions:
                utility = combined_utility_map[next_y][next_x]
                print("{}: {}".format(direction, utility))

    max_utility = float('-inf')
    best_direction = None

    # Map from your 'moves' directions to the actions that your Agent class understands
    direction_to_action = {'North': 'North', 'East': 'East', 'South': 'South', 'West': 'West'}

    legal = api.legalActions(state)

    for direction, (dx, dy) in moves.items():
        next_x, next_y = agent_x + dx, agent_y + dy
        if 0 <= next_x < 20 and 0 <= next_y < 11:  # Replace with actual grid size
            if (next_x, next_y) not in wall_positions:
                utility = combined_utility_map[next_y][next_x]

                # Check if the direction is also a legal move
                if direction_to_action[direction] in legal:
                    if utility > max_utility:
                        max_utility = utility
                        best_direction = direction_to_action[direction]

    return best_direction




class AgentUtility(Agent):

    def __init__(self):
        self.grid = [[' ' for _ in range(20)] for _ in range(11)]
        self.global_food_positions = set()
        self.all_positions_ever = set()
        self.all_positions_remaining = set()
        self.eaten_food_positions = set()
        self.last_agent_position = None  # Track last position of the agent

        self.path = []

    def is_looping(self, N=8):
        if len(self.path) < 2 * N:  # Not enough data to check for loops
            return False
    
        last_N_positions = self.path[-N:]
        prev_N_positions = self.path[-(2 * N):-N]
        
        if last_N_positions == prev_N_positions:
            return True
        
        return False

    def explore(self, state):
        return api.makeMove(random.choice(api.legalActions(state)), api.legalActions(state))

    def getAction(self, state):

        # Update global food positions with newly visible food
        visible_food_positions = set(api.food(state))

        # Update global positions with all positions ever seen
        self.all_positions_ever.update(visible_food_positions)

        # Find the food that has been eaten
        current_agent_position = api.whereAmI(state)
        if self.last_agent_position in self.all_positions_ever and self.last_agent_position not in visible_food_positions:
            # print("Food eaten at position: ", self.last_agent_position)
            self.eaten_food_positions = self.eaten_food_positions.union({self.last_agent_position})
        self.last_agent_position = current_agent_position

        # Update set of all positions with food that has been eaten
        self.all_positions_remaining = self.all_positions_ever - self.eaten_food_positions

        # Set goal foods
        self.global_food_positions = self.all_positions_remaining

        # print("Visible food positions: ", api.food(state))
        # print("All positions ever:", self.all_positions_ever)
        # print("Eaten food positions:", self.eaten_food_positions)
        # print("All positions remaining:", self.all_positions_remaining)

        # Now use self.global_food_positions to compute utility
        pacman = api.whereAmI(state)
        ghost_positions = api.ghosts(state)
        visualise_grid(20, 11, self.global_food_positions, api.walls(state), ghost_positions, self.grid, pacman)

        # Call find_utility to print the utility in each direction
        best_direction = find_utility(state, self.global_food_positions)

        self.path.append(current_agent_position)

        if best_direction:
            if not self.is_looping():
                print("Best direction: ", best_direction)
                return api.makeMove(best_direction, api.legalActions(state))
            else: 
                print("Loop detected")
                return best_direction == self.explore(state)
        else:
            print("No best direction found (No utility)")
            return self.explore(state)











        # legal = api.legalActions(state)

        # print("Food: ", api.food(state))
        # print("Walls: ", api.walls(state))

        # Find closest food using A* search
        # my_pos = api.whereAmI(state)
        # food = api.food(state)

        # Use min(A, manhattanDistance to B), considering walls
        # closest_food = min(food, key=lambda x: util.manhattanDistance(my_pos, x))
        # print("Closest food: ", closest_food)

        # Make move
        # make_move = api.makeMove(random.choice(legal), legal)






        # Define food that was eaten
        food_eaten = []
        for i in range(len(food)): # Look at previous food stack
            if food[i] not in api.food(state): # if food is not in current food stack, it was eaten
                food_eaten.append(food[i])
        print("Food eaten: ", food_eaten)

        return make_move

# RandomishAgent
#
# A tiny bit more sophisticated. Having picked a direction, keep going
# until that direction is no longer possible. Then make a random
# choice.
class RandomishAgent(Agent):

    # Constructor
    #
    # Create a variable to hold the last action
    def __init__(self):
         self.last = Directions.STOP
    
    def getAction(self, state):
        # Get the actions we can try, and remove "STOP" if that is one of them.
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # If we can repeat the last action, do it. Otherwise make a
        # random choice.
        if self.last in legal:
            return api.makeMove(self.last, legal)
        else:
            pick = random.choice(legal)
            # Since we changed action, record what we did
            self.last = pick
            return api.makeMove(pick, legal)

# SensingAgent
#
# Doesn't move, but reports sensory data available to Pacman
class SensingAgent(Agent):

    def getAction(self, state):

        # Demonstrates the information that Pacman can access about the state
        # of the game.

        # What are the current moves available
        legal = api.legalActions(state)
        print "Legal moves: ", legal

        # Where is Pacman?
        pacman = api.whereAmI(state)
        print "Pacman position: ", pacman

        # Where are the ghosts?
        print "Ghost positions:"
        theGhosts = api.ghosts(state)
        for i in range(len(theGhosts)):
            print theGhosts[i]

        # How far away are the ghosts?
        print "Distance to ghosts:"
        for i in range(len(theGhosts)):
            print util.manhattanDistance(pacman,theGhosts[i])

        # Where are the capsules?
        print "Capsule locations:"
        print api.capsules(state)
        
        # Where is the food?
        print "Food locations: "
        print api.food(state)

        # Where are the walls?
        print "Wall locations: "
        print api.walls(state)
        
        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are.
        return api.makeMove(Directions.STOP, legal)




from heapq import heappop, heappush  # Don't forget to import these

class AStarAgent(Agent):

    def __init__(self):
        # Make plan list that will store the sequence of moves
        self.plan = []
        self.first_move = True

    def heuristic(self, pos, goal):
        # Manhattan distance as the heuristic
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def a_star_search(self, start, goal, walls):

        # A* search algorithm

        # Frontier = start state and its heursitic value
        frontier = [(self.heuristic(start, goal), start)]
        came_from = {} # To keep track of the path
        cost_so_far = {} # To keep track of the cost
        came_from[start] = None # No parenting for starting point
        cost_so_far[start] = 0 # Cost from start to self is zero

        # Main loop for A* algorithm
        while frontier:
            # Get node with minimum heuristic value by popping node with lowest probability (i.e. lowest heuristic value)
            _, current = heappop(frontier)

            # Goal Test: If the current node is the goal, we're done
            if current == goal:
                break # Reached the goal, stop searching
            
            # Explore neighbors by finding their coordinates (manhattan distance)
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0)]:
                next_move = (current[0] + dx, current[1] + dy)

                # Skip walls by ignoring them
                if next_move in walls:
                    continue
                
                # Compute the new cost to reach each neighbour
                new_cost = cost_so_far[current] + 1

                # Update the cost and frontier if this path is better 
                if next_move not in cost_so_far or new_cost < cost_so_far[next_move]:
                    cost_so_far[next_move] = new_cost
                    priority = new_cost + self.heuristic(goal, next_move) # Calculate priority for heap
                    heappush(frontier, (priority, next_move)) # Push new node to frontier with its priority
                    came_from[next_move] = current # Update parent for this node

        # Reconstruct the path from goal to start if goal has been reached
        if current != goal:
            return []

        # Path starts empty and is filled with the path from goal to start
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()

        return path # Return the computed path

    def getAction(self, state):
        # Get available actions
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Get current position
        my_pos = api.whereAmI(state)

        print("my_pos: ", my_pos)

        # If it's the first move, go right first
        if self.first_move and Directions.EAST in legal:
            return api.makeMove(Directions.EAST, legal)
        
        self.first_move = False

        # If self.plan is empty, go right first (only once)
        # if not self.plan and Directions.EAST in legal:
        #     return api.makeMove(Directions.EAST, legal)

        # Plan path if there is no existing plan
        if not self.plan:
            food = api.food(state)
            print("food: ", food)
            print("api.food(state):", api.food(state))
            
            walls = api.walls(state)

            if food:
                closest_food = min(food, key=lambda x: self.heuristic(my_pos, x))
                self.plan = self.a_star_search(my_pos, closest_food, walls)

        # Execute the next action from the plan
        if self.plan:
            next_move = self.plan[0]
            self.plan = self.plan[1:]

            # Determine direction to move
            if next_move[0] > my_pos[0]:
                return api.makeMove(Directions.EAST, legal)
            if next_move[0] < my_pos[0]:
                return api.makeMove(Directions.WEST, legal)
            if next_move[1] > my_pos[1]:
                return api.makeMove(Directions.NORTH, legal)
            if next_move[1] < my_pos[1]:
                return api.makeMove(Directions.SOUTH, legal)

        # Fallback to random action if no plan exists
        return api.makeMove(random.choice(legal), legal)








class QLearningAgent(Agent):

    def __init__(self, alpha=0.1, epsilon=0.1, gamma=0.9):
        self.q_values = {}  # Q-values
        self.alpha = alpha  # Learning rate
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma  # Discount factor
        self.last_state = None
        self.last_action = None

    def getAction(self, state):
        legal_actions = api.legalActions(state)
        if Directions.STOP in legal_actions:
            legal_actions.remove(Directions.STOP)

        # Exploration vs exploitation
        if random.random() < self.epsilon:
            action = random.choice(legal_actions)
        else:
            action = max(legal_actions, key=lambda x: self.get_q_value(state, x))

        # Update Q-value of the last state-action pair
        if self.last_state is not None:
            reward = self.get_reward(state)  # Define your own reward function
            self.update_q_value(self.last_state, self.last_action, reward, state)

        self.last_state = state
        self.last_action = action

        return api.makeMove(action, legal_actions)

    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = max(api.legalActions(next_state), key=lambda x: self.get_q_value(next_state, x))
        td_target = reward + self.gamma * self.get_q_value(next_state, best_next_action)
        td_error = td_target - self.get_q_value(state, action)
        new_value = self.get_q_value(state, action) + self.alpha * td_error
        self.q_values[(state, action)] = new_value

