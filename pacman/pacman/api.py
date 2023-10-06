# api.py
# parsons/08-oct-2017
#
# Version 2.1
#
# With acknowledgements to Jiaming Ke, who was the first to report the
# bug in corners.
#
# An API for use with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# This provides a simple way of controlling the way that Pacman moves
# and senses its world, to permit exercises with limited sensing
# ability and nondeterminism in sensing and action.
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

# The code here was written by Simon Parsons, based on examples from
# the PacMan AI projects.

import util

distanceLimit = 5

def distanceLimited(objects, state):
    # When passed a list of objects, tests how far they are from
    # Pacman, and only returns the ones that are within the distance
    # limit.

    pacman = state.getPacmanPosition()
    nearObjects = []
    
    for i in range(len(objects)):
        if util.manhattanDistance(pacman,objects[i]) <= distanceLimit:
            nearObjects.append(objects[i])

    return nearObjects
#
# Sensing
#
def whereAmI(state):
    # Returns an (x, y) pair of Pacman's position.
    #
    # This version says exactly where Pacman is.
    # In later version this may be obfusticated.

    return state.getPacmanPosition()

def legalActions(state):
    # Returns the legal set of actions
    #
    # Just pulls this data out of the state. Functin included so that
    # all interactions are through this API.
    
    return state.getLegalPacmanActions()

def ghosts(state):
    # Returns a list of (x, y) pairs of ghost positions.
    #
    # This version just returns the ghost positions from the state data
    # In later versions this will be more restricted, and include some
    # uncertainty.
            
    return distanceLimited(state.getGhostPositions(),state)

def capsules(state):
    # Returns a list of (x, y) pairs of capsule positions.
    #
    # This version returns the capsule positions if they are within
    # the distance limit.

    return distanceLimited(state.getCapsules(), state)

def food(state):
    # Returns a list of (x, y) pairs of food positions
    #
    # This version returns all the current food locations that are
    # within the distance limit.
    
    foodList= []
    foodGrid = state.getFood()
    width = foodGrid.width
    height = foodGrid.height
    for i in range(width):
        for j in range(height):
            if foodGrid[i][j] == True:
                foodList.append((i, j))            
    return distanceLimited(foodList, state)

def walls(state):
    # Returns a list of (x, y) pairs of wall positions
    #
    # This version just returns all the current wall locations
    # extracted from the state data.  In later versions, this will be
    # restricted by distance, and include some uncertainty.
    
    wallList= []
    wallGrid = state.getWalls()
    width = wallGrid.width
    height = wallGrid.height
    for i in range(width):
        for j in range(height):
            if wallGrid[i][j] == True:
                wallList.append((i, j))            
    return wallList

def corners(state):
    # Returns the coordinates of the four corners of the state space.
    #
    # For harder exploration we could obfusticate this information.

    corners=[]
    wallGrid = state.getWalls()
    width = wallGrid.width
    height = wallGrid.height
    corners.append((0, 0))
    corners.append((width-1, 0))
    corners.append((0, height-1))
    corners.append((width-1, height-1))
    return corners

#
# Acting
#
def makeMove(direction, legal):
    # This version is simple, just return the direction that was picked.
    # In later versions, this will be more complex
    
    return direction