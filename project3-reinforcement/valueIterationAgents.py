# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            # In each iteration, get the information of states first
            state = self.mdp.getStates()
            # Use counter to record the value of states in each iteration
            counter = util.Counter()
            # Calculate the v value for all the states
            for position in state:
                max_value = float("-inf")
                # Computer the Q value on each direction and keep the max one
                for action in self.mdp.getPossibleActions(position):
                    q_value = self.computeQValueFromValues(position, action)
                    max_value = max(q_value, max_value)
                    # Record the max Q value
                    counter[position] = max_value
            self.values = counter


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Get the transition function first
        transition = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0
        # Calculate the Q value
        for state_prime, p in transition:
            r = self.mdp.getReward(state, action, state_prime)
            q_value += p * (r + self.discount * self.getValue(state_prime))
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        max_q_value = float("-inf")
        optimal_move = None
        for a in self.mdp.getPossibleActions(state):
            value = self.computeQValueFromValues(state, a)
            if value > max_q_value:
                max_q_value = value
                optimal_move = a
        return optimal_move

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Get all the states
        state = self.mdp.getStates()
        num = len(state)
        for i in range(self.iterations):
            # Use "idx" to make a cyclic turn
            idx = i % num
            position = state[idx]
            if self.mdp.isTerminal(position) == False:
                q_value = float("-inf")
                # Check all the directions and save the max q_value in counter
                for direction in self.mdp.getPossibleActions(position):
                    value = self.getQValue(position, direction)
                    q_value = max(q_value, value)
                self.values[position] = q_value

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # Initialize an empty priority queue.
        pq = util.PriorityQueue()
        # Compute predecessors of all states.
        predecessors = self.compute_predecessors()

        # In the order returned by self.mdp.getStates()
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s) == False:
                # Find the absolute diff between s in self.values and max q_value
                max_q_value = max(self.getQValue(s, action)
                                  for action in self.mdp.getPossibleActions(s))
                diff = abs(max_q_value - self.values[s])
                # Push s into the priority queue with priority -diff
                pq.update(s, -diff)

        # Update state values
        for _ in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if pq.isEmpty():
                return
            # Pop a state s off the priority queue.
            s = pq.pop()
            # Update the value of s (if it is not a terminal state) in self.values.
            if self.mdp.isTerminal(s) == False:
                self.values[s] = max(self.getQValue(s, action)
                                         for action in self.mdp.getPossibleActions(s))
            self.update_predecessors(pq, predecessors[s])

    # Go through all states and their possible actions
    #  to find and record the predecessors
    def compute_predecessors(self):
        predecessors = {}
        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s) == False:
                for action in self.mdp.getPossibleActions(s):
                    for s_prime, _ in self.mdp.getTransitionStatesAndProbs(s, action):
                        if s_prime not in predecessors:
                            predecessors[s_prime] = set()
                        predecessors[s_prime].add(s)
        return predecessors

    # If diff > theta, push p into the priority queue with priority -diff
    def update_predecessors(self, pq, predecessors):
        for p in predecessors:
            if self.mdp.isTerminal(p) == False:
                max_q_value = max(self.getQValue(p, action)
                                  for action in self.mdp.getPossibleActions(p))
                diff = abs(max_q_value - self.values[p])
                if diff > self.theta:
                    pq.update(p, -diff)
