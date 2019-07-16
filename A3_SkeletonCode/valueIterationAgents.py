# valueIterationAgents.py
# -----------------------

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
        gamma = self.discount

        possibleActions = lambda state: self.mdp.getPossibleActions(state)
        nextStates = lambda state: lambda action: self.mdp.getTransitionStatesAndProbs(state, action)
        reward = lambda state: lambda action: lambda nextState: self.mdp.getReward(state, action, nextState)

        for i in range(self.iterations):
            copy = self.values.copy()
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    rewards = []
                    nextActions = possibleActions(state)
                    for action in nextActions:
                        currentSum = 0
                        for newState, prob in nextStates(state)(action):
                            currentReward = prob * (reward(state)(action)(newState) + self.discount * copy[newState])
                            currentSum += currentReward
                        rewards.append(currentSum)
                    self.values[state] = max(rewards) if rewards else 0


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
        qVal = 0

        nextStates = self.mdp.getTransitionStatesAndProbs(state, action)
        reward = lambda state: lambda action: lambda nextState: \
            self.mdp.getReward(state, action, nextState)

        for newState, prob in nextStates:
            q = prob * (reward(state)(action)(newState)
                            + self.discount * self.values[newState])
            qVal += q
        return qVal


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None

        possibleActions = self.mdp.getPossibleActions(state)
        qVals = []
        for action in possibleActions:
            q = self.computeQValueFromValues(state, action)
            qVals.append((q, action))
        max_qVal = float('-inf')
        policy = None
        for qVal, action in qVals:
            if qVal > max_qVal:
                max_qVal = qVal
                policy = action
        return policy

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

        possibleActions = lambda state: self.mdp.getPossibleActions(state)
        nextStates = lambda state: lambda action: self.mdp.getTransitionStatesAndProbs(state, action)
        reward = lambda state: lambda action: lambda nextState: self.mdp.getReward(state, action, nextState)

        for i in range(self.iterations):
            state = self.mdp.getStates()[i % len(self.mdp.getStates())]
            if not self.mdp.isTerminal(state):
                rewards = []
                nextActions = possibleActions(state)
                for action in nextActions:
                    currentSum = 0
                    for newState, prob in nextStates(state)(action):
                        currentReward = prob * (reward(state)(action)(newState)
                                                 + self.discount * self.values[newState])
                        currentSum += currentReward
                    rewards.append(currentSum)
                self.values[state] = max(rewards) if rewards else 0


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
        theta = self.theta

        nextStates = lambda state: lambda action: self.mdp.getTransitionStatesAndProbs(state, action)
        reward = lambda state: lambda action: lambda nextState: self.mdp.getReward(state, action, nextState)
        states = self.mdp.getStates()

        predecessors = {}
        for state in states:
            for action in possibleActions(state):
                for newState, prob in nextStates(state)(action):
                    if prob > 0:
                        if newState not in predecessors:
                            predecessors[newState] = set()
                        predecessors[newState].add(state)

        fringe = util.PriorityQueue()

        for state in states:
            if self.mdp.isTerminal(state):
                continue
            currentVal = self.values[state]
            diff = abs(max([ValueIterationAgent.computeQValueFromValues(self, state, action)
                            for action in possibleActions(state)]) - currentVal)
            fringe.update(state, -diff)

        for i in range(self.mdp.getPossibleActions(state)):
            if fringe.isEmpty():
                return
            current_state = fringe.pop()
            if not self.mdp.isTerminal(current_state):
                sums = [ValueIterationAgent.computeQValueFromValues(self, current_state, action) for action in possibleActions(current_state)]
                self.values[current_state] = max(sums) if sums else 0
            for p in predecessors[current_state]:
                currentVal = self.values[p]
                diff = abs(max([ValueIterationAgent.computeQValueFromValues(self, p, action) for action in
                                possibleActions(p)]) - currentVal)
                if diff > theta:
                    fringe.update(p, -diff)