# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

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
    some Directions.X for some X in the set {North, South, West, East, Stop}
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
    score = successorGameState.getScore()
    ghostDistance = 0.01
    surrounding = 0
    ghostList = successorGameState.getGhostPositions()
    for ghost in ghostList:
      d = util.manhattanDistance(newPos, ghost)
      if d <= 1:
        surrounding += 1
      ghostDistance += d

    foodList = newFood.asList()
    import sys
    closestFood = sys.maxsize
    for food in foodList:
      d = util.manhattanDistance(newPos, food)
      if d <= closestFood:
        closestFood = d

    return score - (1/float(ghostDistance)) - surrounding + (1/float(closestFood))

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
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    def max_value(state, depth, index):
      if depth == 0 or state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      v = float("-infinity")
      for action in state.getLegalActions(index):
        successor = state.generateSuccessor(index, action)
        res = min_value(successor, depth, state.getNumAgents()-1)
        if res > v:
          v = res
      return v

    def min_value(state, depth, index):
      if depth == 0 or state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      v = float("infinity")
      for action in state.getLegalActions(index):
        successor = state.generateSuccessor(index, action)
        if index < 2:
          res = max_value(successor, depth-1, 0)
          if res < v:
            v = res
        else:
          res = min_value(successor, depth, index-1)
          if res < v:
            v = res
      return v

    val = float("-infinity")
    optAction = Directions.STOP
    agents = gameState.getNumAgents()
    for action in gameState.getLegalActions():
      successor = gameState.generateSuccessor(0, action)
      currVal = min_value(successor, self.depth, agents-1)
      if currVal > val:
        val = currVal
        optAction = action
    return optAction

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    def max_value(state, depth, index, a, b):
      if depth == 0 or state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      v = float("-infinity")
      for action in state.getLegalActions(index):
        successor = state.generateSuccessor(index, action)
        res = min_value(successor, depth, state.getNumAgents()-1, a, b)
        if res > v:
          v = res
        if v > b:
          return v
        a = max(a, v)
      return v

    def min_value(state, depth, index, a, b):
      if depth == 0 or state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      v = float("infinity")
      for action in state.getLegalActions(index):
        successor = state.generateSuccessor(index, action)
        if index < 2:
          res = max_value(successor, depth-1, 0, a, b)
          if res < v:
            v = res
        else:
          res = min_value(successor, depth, index-1, a, b)
          if res < v:
            v = res
        if v < a:
          return v
        b = min(b, v)
      return v

    val = float("-infinity")
    optAction = Directions.STOP
    agents = gameState.getNumAgents()
    alpha = float("-infinity")
    beta = float("infinity")
    for action in gameState.getLegalActions():
      successor = gameState.generateSuccessor(0, action)
      currVal = min_value(successor, self.depth, agents-1, alpha, beta)
      if currVal > val:
        val = currVal
        optAction = action
      if currVal > beta:
        return optAction
      alpha = max(alpha, currVal)
    return optAction

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    def max_value(state, depth, index):
      if depth == 0 or state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      v = float("-infinity")
      for action in state.getLegalActions(index):
        successor = state.generateSuccessor(index, action)
        res = min_value(successor, depth, state.getNumAgents()-1)
        if res > v:
          v = res
      return v

    def min_value(state, depth, index):
      if depth == 0 or state.isWin() or state.isLose():
        return self.evaluationFunction(state)
      sum = 0
      v = float("infinity")
      for action in state.getLegalActions(index):
        successor = state.generateSuccessor(index, action)
        if index < 2:
          res = max_value(successor, depth-1, 0)
          sum = sum + res
        else:
          res = min_value(successor, depth, index-1)
          sum = sum + res
      average = sum / len(state.getLegalActions(index))
      return average

    val = float("-infinity")
    optAction = Directions.STOP
    agents = gameState.getNumAgents()
    for action in gameState.getLegalActions():
      successor = gameState.generateSuccessor(0, action)
      currVal = min_value(successor, self.depth, agents-1)
      if currVal > val:
        val = currVal
        optAction = action
    return optAction

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
      Get the distance to the closest food pellet
      Get the total distance to all the ghosts
      If a ghost is near you, increment surrounding
      If the ghost near you is scared, then increment scared
      - (1/float(ghostDistance)) => you do not want to get closer to ghosts
      - surrounding => you do not want to collide with a ghost near
      + (1/float(closestFood)) => you want to go towards the closest food
      + scared => if a surrounding ghost is scared, the you want to eat it
  """
  "*** YOUR CODE HERE ***"
  score = currentGameState.getScore()
  pos = currentGameState.getPacmanPosition()
  foodList = currentGameState.getFood().asList()
  import sys
  closestFood = sys.maxsize
  for food in foodList:
    d = util.manhattanDistance(pos, food)
    if d <= closestFood:
      closestFood = d

  ghostDistance = 0.0001
  surrounding = 0
  scared = 0
  ghostStates = currentGameState.getGhostStates()
  for ghost in ghostStates:
    d = util.manhattanDistance(pos, ghost.getPosition())
    if d < 2:
      surrounding += 1
      if ghost.scaredTimer > 1:
        scared = scared + 1
    ghostDistance += d

  return score - (1/float(ghostDistance)) - surrounding + (1/float(closestFood)) + scared

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

