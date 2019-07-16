# multiAgents.py
# --------------
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        foodPos = newFood.asList()
        #foodPos = sorted(foodsPos, key = lambda pos: manhattanDistance(newPos, pos))
        closestFoodDist = 0

        if successorGameState.isWin():
            return float("inf")

        for ghostState in newGhostStates:
            if util.manhattanDistance(ghostState.getPosition(), newPos) < 2:
                return float("-inf")
        
        foodDist = []
        for food in foodPos:
            foodDist.append(util.manhattanDistance(food, newPos))
        
        foodSuccessor = 0
        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            foodSuccessor = 100
    
        return successorGameState.getScore() - 5 * min(foodDist) + foodSuccessor

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        maxValue, nextAction = self.minimaxValue(gameState, self.index, 0)
        return nextAction

    def minimaxValue(self, state, agent, depth):
        numAgents = state.getNumAgents()

        if depth == self.depth and agent % numAgents == 0:
            return self.evaluationFunction(state), None

        if agent % numAgents == 0:
            return self.maximizeValue(state, agent % numAgents, depth)

        return self.minimizeValue(state, agent % numAgents, depth)


    def maximizeValue(self, state, agent, depth):
        value = float("-inf")
        valueAction = None
        successors = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]

        if len(successors) == 0:
            return self.evaluationFunction(state), None

        nextAgent = agent + 1
        nextDepth = depth + 1
        for successor, action in successors:
            nextValue, nextAction = self.minimaxValue(successor, nextAgent, nextDepth)
            if nextValue > value:
                value = nextValue
                valueAction = action

        return value, valueAction


    def minimizeValue(self, state, agent, depth):
        value = float("inf")
        valueAction = None       
        successors = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]

        if len(successors) == 0:
            return self.evaluationFunction(state), None

        nextAgent = agent + 1
        for successor, action in successors:
            nextValue, nextAction = self.minimaxValue(successor, nextAgent, depth)
            if nextValue < value:
                value = nextValue
                valueAction = action

        return value, valueAction

        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        alpha = float("-inf")
        beta = float("inf")
        maxValue, nextAction = self.minimaxAlphaBeta(gameState, self.index, 0, alpha, beta)
        return nextAction

    def minimaxAlphaBeta(self, state, agent, depth, alpha, beta):
        numAgents = state.getNumAgents()

        if depth == self.depth and agent % numAgents == 0:
            return self.evaluationFunction(state), None

        if agent % numAgents == 0:
            return self.maximizeValue(state, agent % numAgents, depth, alpha, beta)

        return self.minimizeValue(state, agent % numAgents, depth, alpha, beta)


    def maximizeValue(self, state, agent, depth, alpha, beta):
        value = float("-inf")
        valueAction = None
        legalActions = state.getLegalActions(agent)

        if len(legalActions) == 0:
            return self.evaluationFunction(state), None

        nextAgent = agent + 1
        nextDepth = depth + 1
        for action in legalActions:
            successor = state.generateSuccessor(agent, action)
            nextValue, nextAction = self.minimaxAlphaBeta(successor, nextAgent, nextDepth, alpha, beta)
            if nextValue > value:
                value = nextValue
                valueAction = action
            if value > beta:
                return value, valueAction
            alpha = max(alpha, value)

        return value, valueAction


    def minimizeValue(self, state, agent, depth, alpha, beta):
        value = float("inf")
        valueAction = None       
        legalActions = state.getLegalActions(agent)

        if len(legalActions) == 0:
            return self.evaluationFunction(state), None

        nextAgent = agent + 1
        for action in legalActions:
            successor = state.generateSuccessor(agent, action)
            nextValue, nextAction = self.minimaxAlphaBeta(successor, nextAgent, depth, alpha, beta)
            if nextValue < value:
                value = nextValue
                valueAction = action
            if value < alpha:
                return value, valueAction
            beta = min(beta, value)

        return value, valueAction



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        maxValue, nextAction = self.expectimaxValue(gameState, self.index, 0)
        return nextAction

    def expectimaxValue(self, state, agent, depth):
        numAgents = state.getNumAgents()

        if depth == self.depth and agent % numAgents == 0:
            return self.evaluationFunction(state), None

        if agent % numAgents == 0:
            return self.maximizeValue(state, agent % numAgents, depth)

        return self.probValue(state, agent % numAgents, depth)


    def maximizeValue(self, state, agent, depth):
        value = float("-inf")
        valueAction = None
        successors = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]

        if len(successors) == 0:
            return self.evaluationFunction(state), None

        nextAgent = agent + 1
        nextDepth = depth + 1
        for successor, action in successors:
            nextValue, nextAction = self.expectimaxValue(successor, nextAgent, nextDepth)
            if nextValue > value:
                value = nextValue
                valueAction = action

        return value, valueAction

    def probValue(self, state, agent, depth):
        value = 0
        value_action = None        
        successors = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]

        if len(successors) == 0:
            return self.evaluationFunction(state), None

        nextAgent = agent + 1
        events = 0
        for successor, action in successors:
            nextValue, nextAction = self.expectimaxValue(successor, nextAgent, depth)
            value += nextValue
            events += 1

        return float(value) / events, value_action


def betterEvaluationFunction(currentGameState):

    score = scoreEvaluationFunction(currentGameState)
    newFood = currentGameState.getFood()
    newPos = currentGameState.getPacmanPosition()

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    ghostDist = []
    for i in range(1, currentGameState.getNumAgents()):
        ghostDist.append(util.manhattanDistance(currentGameState.getGhostPosition(i), newPos))
    if min(ghostDist) < 2:
        return float("-inf")

    foodDist = []
    for food in list(newFood.asList()):
        foodDist.append(util.manhattanDistance(food, newPos))

    return score - 2*min(foodDist) - max(foodDist) - 8*currentGameState.getNumFood() + 1.5*min(ghostDist) + max(ghostDist)  

# Abbreviation
better = betterEvaluationFunction
