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

# RandomAgent
#
# A very simple agent. Just makes a random pick every time that it is
# asked for an action.
class RandomAgent(Agent):

    def getAction(self, state):
        # Get the actions we can try, and remove "STOP" if that is one of them.
        legal = api.legalActions(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        # Random choice between the legal options.
        return api.makeMove(random.choice(legal), legal)

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

