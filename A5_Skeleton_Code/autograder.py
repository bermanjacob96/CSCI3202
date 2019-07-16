import unittest
from BayesNets import *

T, F = True, False
values_tf = (T,F)

Sm_cpt = .2
Sm = BayesNode("Sm",[],values_tf,Sm_cpt)

ME_cpt = .5
ME = BayesNode("ME",[],values_tf,ME_cpt)

HBP_cpt = {(T,T) : .6,
           (T,F) : .72,
           (F,T) : .33,
           (F,F) : .51}
HBP = BayesNode("HBP",["Sm","ME"],values_tf,HBP_cpt)

Ath_cpt = .53
Ath = BayesNode("Ath",[],values_tf,Ath_cpt)

FH_cpt = .15
FH = BayesNode("FH",[],values_tf,FH_cpt)

HD_cpt = {(T,T,T):.92,
          (T,T,F):.91,
          (T,F,T):.81,
          (T,F,F):.77,
          (F,T,T):.75,
          (F,T,F):.69,
          (F,F,T):.38,
          (F,F,F):.23}
HD = BayesNode("HD",["HBP","Ath","FH"],values_tf,HD_cpt)

Ang_cpt = {(T):.85,
           (F):.40}
Ang = BayesNode("Ang",["HD"],values_tf,Ang_cpt)

Rapid_cpt = {(T):.99,
             (F):.30}
Rapid = BayesNode("Rapid",["HD"],values_tf,Rapid_cpt)
nodes = [Sm,ME,HBP,Ath,FH,HD,Ang,Rapid]
BN = BayesNet(nodes)

class Tests_Q1(unittest.TestCase):
    def setUp(self):
        self.p1 = BayesNode('p1', '', [T,F], 0.3)
        self.p2 = BayesNode('p2', '', [T,F], 0.6)
        self.c  = BayesNode('c', ['p1', 'p2'], [T,F], {(T,T):0.1, (T,F):0.2, (F,T):0.3, (F,F):0.4})
    def test_onenode(self):
        self.assertEqual(P(self.p1, T), 0.3)
    def test_twonode(self):
        self.assertEqual(P(self.c, F, {'p1':T, 'p2':F}), 0.8)
    def checkFindNode(self):
        self.assertEqual(BN.find_node("HD"), HD)
    def checkParents(self):
        self.assertEqual(BN.find_node("HD").parents, ["HBP","Ath","FH"])
    def checkFindValue(self):
        self.assertEqual(BN.find_values("HD"), values_tf)

class Tests_Q2(unittest.TestCase):

    def test_NoParents(self):
        x = get_prob(Sm,{}, BN)
        self.assertEqual(x[0], 0.2)

    def test_NoParents_2(self):
        x = get_prob(FH,{},BN)
        self.assertEqual(x[0], 0.15)

    def test_Parents(self):
        x = get_prob(HD,{},BN)
        self.assertEqual(x[0], 0.65700256)

    def test_Normalised(self):
        x = get_prob(HD,{},BN)
        self.assertEqual(sum(x), 1)

    def test_evidence(self):
        x = get_prob(Ang,{"HD" : T}, BN)
        self.assertEqual(round(x[1],3), round(0.1500000000000001,3))

    def test_evidence_2(self):
        x = get_prob(HBP,{"HD":T,"FH":T},BN)
        self.assertEqual(round(x[0],3), round(0.570056292379206,3))

class Tests_Q3(unittest.TestCase):
    def test_NoParents(self):
        x = make_Prediction(Sm,{}, BN)
        self.assertEqual(x, F)

    def test_NoParents_2(self):
        x = make_Prediction(FH,{},BN)
        self.assertEqual(x, F)

    def test_Parents(self):
        x = make_Prediction(HD,{},BN)
        self.assertEqual(x, T)

    def test_evidence(self):
        x = make_Prediction(Ang,{"HD" : T}, BN)
        self.assertEqual(x, T)

    def test_evidence_2(self):
        x = make_Prediction(HBP,{"HD":T,"FH":T},BN)
        self.assertEqual(x, T)

class Tests_Q4(unittest.TestCase):
    def setUp(self):
        self.samples = prior_sample_n(BN, 1000)
    def test_N_Samples(self):
        self.assertEqual(len(self.samples), 1000)
    def test_NoParents(self):
        total = 0
        for s in self.samples:
            if s["Sm"]:
                total = total + 1
        x = total/len(self.samples)
        print(x)
        self.assertTrue(0.18 <= x <= 0.22)

    def test_Parents(self):
        total = 0
        for s in self.samples:
            if s["HD"]:
                total = total + 1
        x = total/len(self.samples)
        print(x)
        self.assertTrue(0.62 <= x <= 0.68)






print ("Grading Solution")



tests_q1 = unittest.TestSuite()
tests_q1.addTest(Tests_Q1("test_onenode"))
tests_q1.addTest(Tests_Q1("test_twonode"))
tests_q1.addTest(Tests_Q1("checkFindNode"))
tests_q1.addTest(Tests_Q1("checkParents"))
tests_q1.addTest(Tests_Q1("checkFindValue"))

tests_q2 = unittest.TestSuite()
tests_q2.addTest(Tests_Q2("test_NoParents"))
tests_q2.addTest(Tests_Q2("test_NoParents_2"))
tests_q2.addTest(Tests_Q2("test_Parents"))
tests_q2.addTest(Tests_Q2("test_Normalised"))
tests_q2.addTest(Tests_Q2("test_evidence"))
tests_q2.addTest(Tests_Q2("test_evidence_2"))

tests_q3 = unittest.TestSuite()
tests_q3.addTest(Tests_Q3("test_NoParents"))
tests_q3.addTest(Tests_Q3("test_NoParents_2"))
tests_q3.addTest(Tests_Q3("test_Parents"))
tests_q3.addTest(Tests_Q3("test_evidence"))
tests_q3.addTest(Tests_Q3("test_evidence_2"))

tests_q4 = unittest.TestSuite()
tests_q4.addTest(Tests_Q4("test_N_Samples"))
tests_q4.addTest(Tests_Q4("test_NoParents"))
tests_q4.addTest(Tests_Q4("test_Parents"))


#Actually run the tests. If you want to only test certain things you
#can comment out here

print("**** Question 1 ****")
unittest.TextTestRunner().run(tests_q1)

print("**** Question 2 ****")
unittest.TextTestRunner().run(tests_q2)

print("**** Question 3 ****")
unittest.TextTestRunner().run(tests_q3)

print("**** Question 4 ****")
unittest.TextTestRunner().run(tests_q4)

print("You must upload your code to the autograder in Moodle in order to get credit. The autograder will be available Monday April 1")
