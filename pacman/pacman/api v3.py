# api.py
# parsons/15-oct-2017
#
# Version 3
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

from pacman import Directions
import util

sideLimit = 1
hearingLimit = 2
visibilityLimit = 5

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
            
    return union(visible(state.getGhostPositions(),state), audible(state.getGhostPositions(),state))

def capsules(state):
    # Returns a list of (x, y) pairs of capsule positions.
    #
    # This version returns the capsule positions if they are within
    # the distance limit.
    #
    # Capsules are visible if:
    #
    # 1) Pacman is moving and the capsule is in front of Pacman and
    # within the visibilityLimit, or to the side of Pacman and within
    # the sideLimit.
    #
    # 2) Pacman is not moving, and the capsule is within the visibilityLimit.
    #
    # In both cases, walls block the view.
    
    return visible(state.getCapsules(), state)

def food(state):
    # Returns a list of (x, y) pairs of food positions
    #
    # This version returns all the current food locations that are
    # visible.
    #
    # Food is visible if:
    #
    # 1) Pacman is moving and the food is in front of Pacman and
    # within the visibilityLimit, or to the side of Pacman and within
    # the sideLimit.
    #
    # 2) Pacman is not moving, and the food is within the visibilityLimit.
    #
    # In both cases, walls block the view.
    
    foodList= []
    foodGrid = state.getFood()
    width = foodGrid.width
    height = foodGrid.height
    for i in range(width):
        for j in range(height):
            if foodGrid[i][j] == True:
                foodList.append((i, j))
            
    # Return list of food that is visible
    return visible(foodList, state)

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

#
# Details that you don't need to look at if you don't want to.
#

def distanceLimited(objects, state, limit):
    # When passed a list of object locations, tests how far they are
    # from Pacman, and only returns the ones that are within "limit".

    pacman = state.getPacmanPosition()
    nearObjects = []
    
    for i in range(len(objects)):
        if util.manhattanDistance(pacman,objects[i]) <= limit:
            nearObjects.append(objects[i])

    return nearObjects

def inFront(object, facing, state):
    # Returns true if the object is along the corridor in the
    # direction of the parameter "facing" before a wall gets in the
    # way.
    
    pacman = state.getPacmanPosition()
    pacman_x = pacman[0]
    pacman_y = pacman[1]
    wallList = walls(state)

    # If Pacman is facing North
    if facing == Directions.NORTH:
        # Check if the object is anywhere due North of Pacman before a
        # wall intervenes.
        next = (pacman_x, pacman_y + 1)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (pacman_x, next[1] + 1)
        return False

    # If Pacman is facing South
    if facing == Directions.SOUTH:
        # Check if the object is anywhere due North of Pacman before a
        # wall intervenes.
        next = (pacman_x, pacman_y - 1)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (pacman_x, next[1] - 1)
        return False

    # If Pacman is facing East
    if facing == Directions.EAST:
        # Check if the object is anywhere due East of Pacman before a
        # wall intervenes.
        next = (pacman_x + 1, pacman_y)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (next[0] + 1, pacman_y)
        return False
    
    # If Pacman is facing West
    if facing == Directions.WEST:
        # Check if the object is anywhere due West of Pacman before a
        # wall intervenes.
        next = (pacman_x - 1, pacman_y)
        while not next in wallList:
            if next == object:
                return True
            else:
                next = (next[0] - 1, pacman_y)
        return False

def atSide(object, facing, state):
    # Returns true if the object is in a side corridor perpendicular
    # to the direction that Pacman is travelling.
    
    pacman = state.getPacmanPosition()

    # If Pacman is facing North or Sout, then objects to the side are to the
    # East and West.
    #
    # These are objects that Pacman would see if it were facing East
    # or West.
    
    if facing == Directions.NORTH or facing == Directions.SOUTH:
        # Check if the object is anywhere due North of Pacman before a
        # wall intervenes.
       if inFront(object, Directions.WEST, state) or inFront(object, Directions.EAST, state):
                return True
       else:
                return False
            
    # Similarly for other directions
    if facing == Directions.WEST or facing == Directions.EAST:
        # Check if the object is anywhere due North of Pacman before a
        # wall intervenes.
       if inFront(object, Directions.NORTH, state) or inFront(object, Directions.SOUTH, state):
                return True
       else:
                return False

    else:
        return False
    
def visible(objects, state):
    # When passed a list of objects, returns those that are visible to
    # Pacman.

    facing = state.getPacmanState().configuration.direction
    visibleObjects = []
    sideObjects = []
    
    if facing != Directions.STOP:
    
        # If Pacman is moving, visible objects are those in front of,
        # and to the side (if there are any side corridors).

        # Objects in front. Visible up to "visibilityLimit"
        for i in range(len(objects)):
            if inFront(objects[i], facing, state):
                visibleObjects.append(objects[i])
        visibleObjects = distanceLimited(visibleObjects, state, visibilityLimit)
        
        # Objects to the side. Visible up to "sideLimit"
        for i in range(len(objects)):
            if atSide(objects[i], facing, state):
                sideObjects.append(objects[i])
        sideObjects = distanceLimited(sideObjects, state, sideLimit)

        # Combine lists.
        visibleObjects = visibleObjects + sideObjects
        
    else:

        # If Pacman is not moving, they can see in all directions.

        for i in range(len(objects)):
            if inFront(objects[i], Directions.NORTH, state):
                visibleObjects.append(objects[i])
            if inFront(objects[i], Directions.SOUTH, state):
                visibleObjects.append(objects[i])
            if inFront(objects[i], Directions.EAST, state):
                visibleObjects.append(objects[i])
            if inFront(objects[i], Directions.WEST, state):
                visibleObjects.append(objects[i])
        visibleObjects = distanceLimited(visibleObjects, state, visibilityLimit)
      
    return visibleObjects

def audible(ghosts, state):
    # A ghost is audible if it is any direction and less than
    # "hearingLimit" away.

    return distanceLimited(ghosts, state, hearingLimit)  
    
def union(a, b):
    # return the union of two lists 
    #
    # From https://www.saltycrane.com/blog/2008/01/how-to-find-intersection-and-union-of/
    #
    return list(set(a) | set(b))
