# multiAgents.py
# --------------
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
        #print(gameState)
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
        inf = 9999999
        # calculate the distance to each ghost. If the min value is 0, then return a low value to avoid this step.
        ghost_distance = []
        for ghost  in newGhostStates:
            if ghost.scaredTimer == 0:
                ghost_distance.append(manhattanDistance(newPos, ghost.configuration.pos))
        min_ghost = min(ghost_distance,default=1)
        if min_ghost == 0:
            return -1*inf

        # If next step is food, return a high value to choose this step.
        # Use the grid of current state because we consider the next move base on current food position
        now_food = currentGameState.getFood()
        if now_food[newPos[0]][newPos[1]] == True:
            return inf
        # If next step is not food, the return the distance to the cloest food
        else:
            food_distance = []
            for x in range(now_food.width):
                for y in range(now_food.height):
                    if now_food[x][y] == True:
                        food_distance.append(manhattanDistance(newPos, (x,y)))
            min_food = min(food_distance,default=1)

        # return a lower score if ghost is close and return a high score if a food is close
        return -1/min_ghost + 1/min_food

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

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # set up the numebr of ghost for recursion. -1 because pacman is counted as an agent
        ghost_number = gameState.getNumAgents() - 1
        inf = float("inf")

        # def the max function  for pacman
        def maximizer(curr_state, curr_depth, index):
            # terminate the recursion if the games ends or reaches the depth
            if curr_depth == 0 or curr_state.isLose() or curr_state.isWin():
                return self.evaluationFunction(curr_state)
            # set up initial score and explore potential actions
            score = -inf
            potential_moves = curr_state.getLegalActions(index)
            # start recursion bu calling the minimizer function
            for move in potential_moves:
                # check the ghost starting from index 1
                opponent = minimizer(curr_state.generateSuccessor(index, move), curr_depth, index + 1)
                score = max(opponent , score)
            return score

        # def the min function for ghost
        def minimizer(curr_state, curr_depth, index):
            # terminate the recursion if the games ends or reaches the depth
            if curr_depth == 0 or curr_state.isLose() or curr_state.isWin():
                return self.evaluationFunction(curr_state)
            # set up initial score and explore potential actions
            score = inf
            potential_moves = curr_state.getLegalActions(index)
            for move in potential_moves:
                # This is the thing to do if not all ghosts are evaluated. Find the minimum value from all ghosts
                if index != ghost_number:
                    companion = minimizer(curr_state.generateSuccessor(index, move), curr_depth, index + 1)
                    score = min(score, companion)
                # If all the ghosts are searched, the turn goes back to pacman.
                else:
                    # Go back to pacman -> index = 0. Go to next level -> depth - 1
                    opponent = maximizer(curr_state.generateSuccessor(index, move), curr_depth - 1, 0)
                    score = min(score, opponent)
            return score

        # set up the parameters for beginning
        best_score = -inf
        best_move = None
        # Explore the possible moves of pacman and evaluate the opponent's move (minimizer)
        potential_moves = gameState.getLegalActions(0)
        for move in potential_moves:
            curr_score = minimizer(gameState.generateSuccessor(0,move), self.depth, 1)
            if best_score < curr_score:
                best_score = curr_score
                best_move = move
        return best_move


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        inf = float("inf")
        alpha = -inf
        beta = inf
        # set up the numebr of ghost for recursion. -1 because pacman is counted as an agent
        ghost_number = gameState.getNumAgents() - 1

        # def the max function  for pacman
        def maximizer(curr_state, curr_depth, alpha, beta, index):
            # terminate the recursion if the games ends or reaches the depth
            if curr_depth == 0 or curr_state.isLose() or curr_state.isWin():
                return self.evaluationFunction(curr_state)
            # follow the pseudo code
            v = -inf
            potential_moves = curr_state.getLegalActions(index)
            for move in potential_moves:
                opponent = minimizer(curr_state.generateSuccessor(index, move), curr_depth, alpha, beta, index + 1)
                v = max(v, opponent)
                if beta < v:
                    return v
                alpha = max(alpha ,v)
            return v

        # def the min function for ghost
        def minimizer(curr_state, curr_depth, alpha, beta, index):
            # terminate the recursion if the games ends or reaches the depth
            if curr_depth == 0 or curr_state.isLose() or curr_state.isWin():
                return self.evaluationFunction(curr_state)
            # follow the pseudo code
            v = inf
            potential_moves = curr_state.getLegalActions(index)
            for move in potential_moves:
                if index != ghost_number:
                    companion = minimizer(curr_state.generateSuccessor(index, move), curr_depth, alpha, beta, index + 1)
                    v = min(v, companion)
                else:
                    # If all the ghosts are searched, the turn goes back to pacman.
                    # Go back to pacman -> index = 0. Go to next level -> depth - 1
                    opponent = maximizer(curr_state.generateSuccessor(index, move), curr_depth - 1, alpha, beta, 0)
                    v = min(v, opponent)
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        best_score = -inf
        best_move = None
        # Explore the possible moves of pacman and evaluate the opponent's move (minimizer)
        potential_moves = gameState.getLegalActions(0)
        for move in potential_moves:
            curr_score = minimizer(gameState.generateSuccessor(0,move), self.depth, alpha, beta, 1)
            if best_score < curr_score:
                best_score = curr_score
                best_move = move
            alpha = max(alpha, best_score)
        return best_move

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
        inf = float("inf")
        # set up the numebr of ghost for recursion. -1 because pacman is counted as an agent
        ghost_number = gameState.getNumAgents() - 1

        # def a function for ghosts in expectimax search
        def expecti_ghost(curr_state, curr_depth, index):
            # terminate the recursion if the games ends or reaches the depth
            if curr_depth == 0 or curr_state.isLose() or curr_state.isWin():
                return self.evaluationFunction(curr_state)
            score = []
            potential_moves = curr_state.getLegalActions(index)
            for move in potential_moves:
                if index != ghost_number:
                    score.append(expecti_ghost(curr_state.generateSuccessor(index, move), curr_depth, index + 1))
                else:
                    score.append(maximizer(curr_state.generateSuccessor(index, move), curr_depth-1, 0))
            return sum(score)/len(score)

        # def the max function  for pacman
        def maximizer(curr_state, curr_depth, index):
            # terminate the recursion if the games ends or reaches the depth
            if curr_depth == 0 or curr_state.isLose() or curr_state.isWin():
                return self.evaluationFunction(curr_state)
            # set up initial score and explore potential actions
            score = -inf
            potential_moves = curr_state.getLegalActions(index)
            # start recursion bu calling the expecti_ghost function
            for move in potential_moves:
                # check the ghost starting from index 1
                opponent = expecti_ghost(curr_state.generateSuccessor(index, move), curr_depth, index + 1)
                score = max(opponent , score)
            return score

        best_score = -inf
        best_move = None
        # Explore the possible moves of pacman and evaluate the opponent's move (expecti_ghost)
        potential_moves = gameState.getLegalActions(0)
        for move in potential_moves:
            curr_score = expecti_ghost(gameState.generateSuccessor(0,move), self.depth, 1)
            if best_score < curr_score:
                best_score = curr_score
                best_move = move
        return best_move

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
    ghost_penalty:
        Compute the distance between each ghost and pacman.
        If the ghost is scared, let pacman approach the ghost.
        If the ghost is not scared, do not approach.
        This value could be positive or negative.
    avg_food_distance:
        Compute the average distance between pacman and all the foods.
        Let pacman approach more foods.
    avg_capsule_distance:
        Compute the average distance between pacman and all the capsule.
        Let pacman eat capsule if it is nearby.
    """
    "*** YOUR CODE HERE ***"
    pac_pos = currentGameState.getPacmanPosition()
    food_pos = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghost.scaredTimer for ghost in ghostStates]

    ghost_penalty = 0
    for ghost in ghostStates:
        distance = manhattanDistance(pac_pos, ghost.getPosition())
        if ghost.scaredTimer > 0:
            if distance:
                ghost_penalty -= 1 / distance
            else:
                ghost_penalty -= 1
        else:
            if distance:
                ghost_penalty += 1 / distance
            else:
                ghost_penalty += 1

    food_distance = sum([manhattanDistance(pac_pos, food) for food in food_pos])
    if food_pos:
        avg_food_distance = food_distance / len(food_pos)
    else:
        avg_food_distance = 0

    capsule_distance = sum([manhattanDistance(pac_pos, cap) for cap in currentGameState.getCapsules()])
    if currentGameState.getCapsules():
        avg_capsule_distance = capsule_distance / len(currentGameState.getCapsules())
    else:
        avg_capsule_distance = 0

    return (currentGameState.getScore()
            - 1.5 * avg_food_distance
            - 2 * avg_capsule_distance
            + 50 * ghost_penalty)

# Abbreviation
better = betterEvaluationFunction
