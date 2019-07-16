import unittest
import util
import random


## For the sake of brevity...
T, F = True, False

## From class:
def P(var, value, evidence={}):
    '''The probability distribution for P(var | evidence),
    when all parent variables are known (in evidence)'''
    if len(var.parents)==1:
        # only one parent
        row = evidence[var.parents[0]]
    else:
        # multiple parents
        row = tuple(evidence[parent] for parent in var.parents)
    return var.cpt[row] if value else 1-var.cpt[row]

## Also from class:
class BayesNode:

    def __init__(self, name, parents, values, cpt):
        if isinstance(parents, str):
            parents = parents.split()

        if len(parents)==0:
            # if no parents, empty dict key for cpt
            cpt = {(): cpt}
        elif isinstance(cpt, dict):
            # if there is only one parent, only one tuple argument
            if cpt and isinstance(list(cpt.keys())[0], bool):
                cpt = {(v): p for v, p in cpt.items()}

        self.variable = name
        self.parents = parents
        self.cpt = cpt
        self.values = values


    def __repr__(self):
        return repr((self.variable, ' '.join(self.parents)))

class BayesNet:
    '''Bayesian network containing only boolean-variable nodes.'''

    def __init__(self, nodes):
        '''Initialize the Bayes net by adding each of the nodes,
        which should be a list BayesNode class objects ordered
        from parents to children (`top` to `bottom`, from causes
        to effects)'''

        # your code goes here...


    def add(self, node):
        '''Add a new BayesNode to the BayesNet. The parents should all
        already be in the net, and the variable itself should not be'''
        util.raiseNotDefined()


    def find_node(self, var):
        '''Find and return the BayesNode in the net with name `var`'''
        # your code goes here...
        util.raiseNotDefined()


    def find_values(self, var):
        '''Return the set of possible values for variable `var`'''

        # your code goes here...
        util.raiseNotDefined()


    def __repr__(self):
        return 'BayesNet({})'.format(self.nodes)


def normalize(prob_distr):
    '''This is here to help'''
    total = sum(prob_distr)
    if total != 0:
        return map(lambda a: a / total, prob_distr)
    else:
        return prob_distr

def get_prob(Q, e, bn):
    '''Return probability distribution Q given evidence e in BayesNet bn
     e.g. P(Q|e). You may want to make helper functions here!'''

    """Your Code Goes here"""
    util.raiseNotDefined()



def make_Prediction(Q,e, bn):
    '''Return most likely value for variable Q given evidence e in BayesNet bn
     '''
    """Your Code Goes here"""
    util.raiseNotDefined()


def prior_sample_n(bn, n):
    '''Return a list of samples from the BayesNet bn, where each sample is a dictionary
    Use Prior sampling (no evidence) to generate your samples, you will need
    to sample in the correct order '''

    """Your Code Goes here"""
    util.raiseNotDefined()
