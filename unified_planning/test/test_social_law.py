# Copyright 2022 Technion project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unified_planning as up
from unified_planning.shortcuts import *
from unified_planning.test import TestCase, main
from unified_planning.test.examples.social_law import get_example_problems, get_intersection_problem
from unified_planning.social_law.single_agent_projection import SingleAgentProjection
from unified_planning.social_law.robustness_verification import RobustnessVerifier, SimpleInstantaneousActionRobustnessVerifier, WaitingActionRobustnessVerifier
from unified_planning.social_law.robustness_checker import SocialLawRobustnessChecker, SocialLawRobustnessStatus
from unified_planning.social_law.social_law import SocialLaw
from unified_planning.social_law.waitfor_specification import WaitforSpecification
from unified_planning.social_law.ma_problem_waitfor import MultiAgentProblemWithWaitfor
from unified_planning.model.multi_agent.ma_centralizer import MultiAgentProblemCentralizer
from unified_planning.model.multi_agent.sa_to_ma_converter import SingleAgentToMultiAgentConverter
from unified_planning.social_law.synthesis import SocialLawGenerator, SocialLawGeneratorSearch, get_gbfs_social_law_generator
from unified_planning.model.multi_agent import *
from unified_planning.io import PDDLWriter, PDDLReader
from unified_planning.engines import PlanGenerationResultStatus
from collections import namedtuple
import random
import os

POSITIVE_OUTCOMES = frozenset(
    [
        PlanGenerationResultStatus.SOLVED_SATISFICING,
        PlanGenerationResultStatus.SOLVED_OPTIMALLY,
    ]
)

UNSOLVABLE_OUTCOMES = frozenset(
    [
        PlanGenerationResultStatus.UNSOLVABLE_INCOMPLETELY,
        PlanGenerationResultStatus.UNSOLVABLE_PROVEN,
    ]
)

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
PDDL_DOMAINS_PATH = os.path.join(FILE_PATH, "pddl")


class RobustnessTestCase:
    def __init__(self, name, 
                    expected_outcome : SocialLawRobustnessStatus, 
                    cars = ["car-north", "car-south", "car-east", "car-west"], 
                    yields_list = [], 
                    wait_drive = True):
        self.name = name
        self.cars = cars
        self.yields_list = yields_list
        self.expected_outcome = expected_outcome
        self.wait_drive = wait_drive        


class TestProblem(TestCase):
    def setUp(self):
        TestCase.setUp(self)        
        self.test_cases = [         
            RobustnessTestCase("4cars_crash", SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL, yields_list=[], wait_drive=False),   
            RobustnessTestCase("4cars_deadlock", SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK, yields_list=[]),
            RobustnessTestCase("4cars_yield_deadlock", SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK, yields_list=[("south-ent", "east-ent"),("east-ent", "north-ent"),("north-ent", "west-ent"),("west-ent", "south-ent")]),
            RobustnessTestCase("4cars_robust", SocialLawRobustnessStatus.ROBUST_RATIONAL, yields_list=[("south-ent", "cross-ne"),("north-ent", "cross-sw"),("east-ent", "cross-nw"),("west-ent", "cross-se")]),
            RobustnessTestCase("2cars_crash", SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL, cars=["car-north", "car-east"], yields_list=[], wait_drive=False),   
            RobustnessTestCase("2cars_robust", SocialLawRobustnessStatus.ROBUST_RATIONAL, cars=["car-north", "car-south"], yields_list=[], wait_drive=False)            
        ]

    def test_synthesis(self):
        problem = MultiAgentProblemWithWaitfor()
        
        loc = UserType("loc")
    
        # Environment     
        connected = Fluent('connected', BoolType(), l1=loc, l2=loc)        
        problem.ma_environment.add_fluent(connected, default_initial_value=False)

        free = Fluent('free', BoolType(), l=loc)
        problem.ma_environment.add_fluent(free, default_initial_value=True)

        nw, ne, sw, se = Object("nw", loc), Object("ne", loc), Object("sw", loc), Object("se", loc)        
        problem.add_objects([nw, ne, sw, se])
        problem.set_initial_value(connected(nw, ne), True)
        problem.set_initial_value(connected(nw, sw), True)
        problem.set_initial_value(connected(ne, nw), True)
        problem.set_initial_value(connected(ne, se), True)
        problem.set_initial_value(connected(sw, se), True)
        problem.set_initial_value(connected(sw, nw), True)
        problem.set_initial_value(connected(se, sw), True)
        problem.set_initial_value(connected(se, ne), True)


        at = Fluent('at', BoolType(), l1=loc)

        move = InstantaneousAction('move', l1=loc, l2=loc)
        l1 = move.parameter('l1')
        l2 = move.parameter('l2')
        move.add_precondition(at(l1))
        move.add_precondition(free(l2))
        move.add_precondition(connected(l1,l2))
        move.add_effect(at(l2),True)
        move.add_effect(free(l2), False)
        move.add_effect(at(l1), False)
        move.add_effect(free(l1), True)    

        agent1 = Agent("a1", problem)
        problem.add_agent(agent1)
        agent1.add_fluent(at, default_initial_value=False)
        agent1.add_action(move)
        problem.waitfor.annotate_as_waitfor(agent1.name, move.name, free(l2))

        agent2 = Agent("a2", problem)
        problem.add_agent(agent2)
        agent2.add_fluent(at, default_initial_value=False)
        agent2.add_action(move)
        problem.waitfor.annotate_as_waitfor(agent2.name, move.name, free(l2))

        problem.set_initial_value(Dot(agent1, at(nw)), True)
        problem.set_initial_value(Dot(agent2, at(se)), True)
        problem.set_initial_value(free(nw), False)
        problem.set_initial_value(free(se), False)

        problem.add_goal(Dot(agent1, at(sw)))
        problem.add_goal(Dot(agent2, at(ne)))


        slrc = SocialLawRobustnessChecker(
            planner_name="fast-downward",
            robustness_verifier_name="SimpleInstantaneousActionRobustnessVerifier",
            save_pddl=True
            )
        l = SocialLaw()
        l.disallow_action("a1", "move", ("nw","ne"))
        l.disallow_action("a1", "move", ("sw","se"))
        l.disallow_action("a2", "move", ("ne","nw"))
        l.disallow_action("a2", "move", ("se","sw"))
        pr = l.compile(problem).problem
        # prr = slrc.is_robust(pr)
        # self.assertEqual(prr.status,SocialLawRobustnessStatus.ROBUST_RATIONAL)

        l2 = SocialLaw()
        l.disallow_action("a1", "move", ("nw","ne"))
        l.disallow_action("a2", "move", ("ne","nw"))

        self.assertTrue(l.is_stricter_than(l2))
        self.assertFalse(l2.is_stricter_than(l))

        # g1 = SocialLawGenerator(SocialLawGeneratorSearch.BFS)
        # rprob1 = g1.generate_social_law(problem)
        # self.assertIsNotNone(rprob1)

        # g2 = SocialLawGenerator(SocialLawGeneratorSearch.DFS)
        # rprob2 = g2.generate_social_law(problem)
        # self.assertIsNotNone(rprob2)

        g3 = get_gbfs_social_law_generator()
        rprob3 = g3.generate_social_law(problem)
        self.assertIsNotNone(rprob3)
        

    def test_social_law(self):
        slrc = SocialLawRobustnessChecker(
            planner_name="fast-downward",
            robustness_verifier_name="SimpleInstantaneousActionRobustnessVerifier"
            )
        p_4cars_crash = get_intersection_problem(wait_drive=False).problem
        l = SocialLaw()
        for agent in p_4cars_crash.agents:
            l.add_waitfor_annotation(agent.name, "drive", "free", ("l2",)  )
        
        res = l.compile(p_4cars_crash)
        p_4cars_deadlock = res.problem
        self.assertEqual(len(p_4cars_crash.waitfor.waitfor_map), 0)
        self.assertEqual(len(p_4cars_deadlock.waitfor.waitfor_map), 4)

        r_result = slrc.is_robust(p_4cars_deadlock)
        self.assertEqual(r_result.status, SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_DEADLOCK)
        r_result = slrc.is_robust(p_4cars_crash)
        self.assertEqual(r_result.status, SocialLawRobustnessStatus.NON_ROBUST_MULTI_AGENT_FAIL)        

        l2 = SocialLaw()
        l2.disallow_action("car-north", "drive", ("south-ent", "cross-se", "north") )
        res = l2.compile(p_4cars_crash)
        p_nosap = res.problem

        r_result = slrc.is_robust(p_nosap)
        self.assertEqual(r_result.status, SocialLawRobustnessStatus.NON_ROBUST_SINGLE_AGENT)

        l3 = SocialLaw()        
        l3.add_new_fluent(None, "yieldsto", (("l1","loc"), ("l2","loc")), False)
        l3.add_new_object("dummy_loc", "loc")
        for loc1,loc2 in [("south-ent", "cross-ne"),("north-ent", "cross-sw"),("east-ent", "cross-nw"),("west-ent", "cross-se")]:
            l3.set_initial_value_for_new_fluent(None, "yieldsto", (loc1, loc2), True)
        for loc in p_4cars_crash.objects(p_4cars_crash.user_type("loc")):
            if loc.name not in ["south-ent", "north-ent", "east-ent", "west-ent"]:
                l3.set_initial_value_for_new_fluent(None, "yieldsto", (loc.name, "dummy_loc"), True)
        for agent in p_4cars_crash.agents:
            l3.add_parameter_to_action(agent.name, "drive", "ly", "loc")            
            l3.add_precondition_to_action(agent.name, "drive", "yieldsto", ("l1", "ly") )            
            l3.add_precondition_to_action(agent.name, "drive", "free", ("ly",) )
            l3.add_waitfor_annotation(agent.name, "drive", "free", ("ly",) )
        res = l3.compile(p_4cars_deadlock)
        p_robust = res.problem
        r_result = slrc.is_robust(p_robust)
        self.assertEqual(r_result.status, SocialLawRobustnessStatus.ROBUST_RATIONAL)
        self.assertEqual(len(p_robust.ma_environment.fluents), len(p_4cars_deadlock.ma_environment.fluents) + 1)

    def test_all_cases(self):
        for t in self.test_cases:
            problem = get_intersection_problem(t.cars, t.yields_list, t.wait_drive, durative=False).problem
            slrc = SocialLawRobustnessChecker(
                planner_name="fast-downward",
                robustness_verifier_name="SimpleInstantaneousActionRobustnessVerifier"
                )
            r_result = slrc.is_robust(problem)
            self.assertEqual(r_result.status, t.expected_outcome, t.name)
            if t.expected_outcome == SocialLawRobustnessStatus.ROBUST_RATIONAL:
                presult = slrc.solve(problem)
                self.assertIn(presult.status, POSITIVE_OUTCOMES, t.name)

    # def test_all_cases_waiting(self):
    #     for t in self.test_cases:
    #         problem = get_intersection_problem(t.cars, t.yields_list, t.wait_drive, durative=False).problem
    #         slrc = SocialLawRobustnessChecker(
    #             planner_name="fast-downward",
    #             robustness_verifier_name="WaitingActionRobustnessVerifier"
    #             )
    #         r_result = slrc.is_robust(problem)
    #         self.assertEqual(r_result.status, t.expected_outcome, t.name)
    #         if t.expected_outcome == SocialLawRobustnessStatus.ROBUST_RATIONAL:
    #             presult = slrc.solve(problem)
    #             self.assertIn(presult.status, POSITIVE_OUTCOMES, t.name)


    def test_centralizer(self):
        for t in self.test_cases:
            for durative in [False]:# True]:
                problem = get_intersection_problem(t.cars, t.yields_list, t.wait_drive, durative=durative).problem
                mac = MultiAgentProblemCentralizer()
                cresult = mac.compile(problem)
                with OneshotPlanner(problem_kind=cresult.problem.kind) as planner:
                    presult = planner.solve(cresult.problem)
                    self.assertIn(presult.status, POSITIVE_OUTCOMES, t.name)

        
    def test_all_cases_durative(self):
        for t in self.test_cases:
            problem = get_intersection_problem(t.cars, t.yields_list, t.wait_drive, durative=True).problem
            with open("waitfor.json", "w") as f:
                f.write(str(problem.waitfor))

            slrc = SocialLawRobustnessChecker(                                
                save_pddl=True,                
                )
            self.assertEqual(slrc.is_robust(problem).status, t.expected_outcome, t.name)

    def test_sa_ma_converter(self):
        reader = PDDLReader()
        random.seed(2023)
        
        domain_filename = os.path.join(PDDL_DOMAINS_PATH, "transport", "domain.pddl")
        problem_filename = os.path.join(PDDL_DOMAINS_PATH, "transport", "task10.pddl")
        problem = reader.parse_problem(domain_filename, problem_filename)

        samac = SingleAgentToMultiAgentConverter(["vehicle"])

        result = samac.compile(problem)

        print(result.problem)
        

            
