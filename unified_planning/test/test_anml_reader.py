# Copyright 2021 AIPlan4EU project
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


from typing import cast
from unified_planning.shortcuts import *
from unified_planning.test import TestCase, main, skipIfEngineNotAvailable
from unified_planning.test.examples import get_example_problems
from unified_planning.io import ANMLReader
import tempfile
import os


FILE_PATH = os.path.dirname(os.path.abspath(__file__))
ANML_FILES_PATH = os.path.join(FILE_PATH, "anml")


class TestANMLReader(TestCase):
    def setUp(self):
        TestCase.setUp(self)
        self.start_timing = StartTiming()
        self.start_interval = TimePointInterval(self.start_timing)
        self.end_timing = EndTiming()
        self.global_start_timing = GlobalStartTiming()
        self.global_end_timing = GlobalEndTiming()
        self.all_interval = ClosedTimeInterval(self.start_timing, self.end_timing)

    def test_basic(self):
        reader = ANMLReader()

        problem_filename = os.path.join(ANML_FILES_PATH, "basic.anml")
        problem = reader.parse_problem(problem_filename)
        em = problem.env.expression_manager

        self.assertIsNotNone(problem)
        self.assertEqual(len(problem.fluents), 1)
        self.assertEqual(len(problem.actions), 1)
        self.assertEqual(len(problem.goals), 1)
        self.assertEqual(len(problem.timed_goals), 0)
        self.assertEqual(len(problem.timed_effects), 0)
        a = problem.action("a")
        self.assertEqual(a.duration, FixedDuration(em.Int(6)))
        self.assertEqual(len(a.conditions), 0)
        for timing, effect_list in a.effects.items():
            if timing == self.end_timing:
                self.assertEqual(len(effect_list), 1)
            else:
                self.assertTrue(False)

    def test_match_reader(self):
        reader = ANMLReader()

        problem_filename = os.path.join(ANML_FILES_PATH, "match.anml")
        problem = reader.parse_problem(problem_filename)
        em = problem.env.expression_manager

        self.assertIsNotNone(problem)
        self.assertEqual(len(problem.fluents), 4)
        self.assertEqual(len(problem.actions), 2)
        self.assertEqual(len(list(problem.objects(problem.user_type("Match")))), 3)
        self.assertEqual(len(list(problem.objects(problem.user_type("Fuse")))), 3)
        self.assertEqual(len(problem.goals), 3)
        self.assertEqual(len(problem.timed_goals), 0)
        self.assertEqual(len(problem.timed_effects), 0)

        light_match = problem.action("light_match")
        self.assertEqual(light_match.duration, FixedDuration(em.Int(6)))
        for interval, cond_list in light_match.conditions.items():
            self.assertEqual(interval, self.start_interval)
            self.assertEqual(len(cond_list), 1)
        for timing, effect_list in light_match.effects.items():
            if timing == self.start_timing:
                self.assertEqual(len(effect_list), 2)
            elif timing == self.end_timing:
                self.assertEqual(len(effect_list), 1)
            else:
                self.assertTrue(False)

        mend_fuse = problem.action("mend_fuse")
        self.assertEqual(mend_fuse.duration, FixedDuration(em.Int(5)))
        for interval, cond_list in mend_fuse.conditions.items():
            if interval in (self.start_interval, self.all_interval):
                self.assertEqual(len(cond_list), 1)
            else:
                self.assertTrue(False)
        for timing, effect_list in mend_fuse.effects.items():
            if timing == self.start_timing:
                self.assertEqual(len(effect_list), 1)
            elif timing == self.end_timing:
                self.assertEqual(len(effect_list), 2)
            else:
                self.assertTrue(False)

    def test_connected_locations_reader(self):
        reader = ANMLReader()

        problem_filename = os.path.join(ANML_FILES_PATH, "connected_locations.anml")
        problem = reader.parse_problem(problem_filename)

        self.assertIsNotNone(problem)
        self.assertEqual(len(problem.fluents), 2)
        self.assertEqual(len(problem.actions), 1)
        self.assertEqual(len(list(problem.objects(problem.user_type("Location")))), 3)
        self.assertEqual(len(problem.goals), 1)
        self.assertEqual(len(problem.timed_goals), 0)
        self.assertEqual(len(problem.timed_effects), 0)

        move = problem.action("move")
        for interval, cond_list in move.conditions.items():
            self.assertEqual(interval, self.start_interval)
            self.assertEqual(len(cond_list), 2)
        for timing, effect_list in move.effects.items():
            if timing == self.start_timing:
                self.assertEqual(len(effect_list), 2)
            else:
                self.assertTrue(False)

    def test_constants_no_variable_duration_reader(self):
        reader = ANMLReader()

        problem_filename = os.path.join(
            ANML_FILES_PATH, "constants_no_variable_duration.anml"
        )
        problem = reader.parse_problem(problem_filename)
        em = problem.env.expression_manager

        self.assertIsNotNone(problem)
        self.assertEqual(len(problem.fluents), 4)
        self.assertEqual(len(problem.actions), 1)
        self.assertEqual(len(list(problem.objects(problem.user_type("Location")))), 5)
        self.assertEqual(len(problem.goals), 1)
        self.assertEqual(len(problem.timed_goals), 0)
        self.assertEqual(len(problem.timed_effects), 0)

        move = problem.action("move")
        self.assertEqual(move.duration, FixedDuration(em.Int(4)))
        for interval, cond_list in move.conditions.items():
            self.assertEqual(interval, self.start_interval)
            self.assertEqual(len(cond_list), 3)
        for timing, effect_list in move.effects.items():
            if timing == self.start_timing:
                self.assertEqual(len(effect_list), 1)
            elif timing == self.end_timing:
                self.assertEqual(len(effect_list), 2)
            else:
                self.assertTrue(False)

    def test_durative_goals_reader(self):
        reader = ANMLReader()

        problem_filename = os.path.join(ANML_FILES_PATH, "durative_goals.anml")
        problem = reader.parse_problem(problem_filename)
        em = problem.env.expression_manager

        self.assertIsNotNone(problem)
        self.assertEqual(len(problem.fluents), 2)
        self.assertEqual(len(problem.actions), 1)
        self.assertEqual(len(problem.all_objects), 0)
        self.assertEqual(len(problem.goals), 1)
        self.assertEqual(len(problem.timed_goals), 1)
        self.assertEqual(len(problem.timed_effects), 1)

        a = problem.action("a")
        self.assertEqual(a.duration, FixedDuration(em.Int(1)))
        for interval, cond_list in a.conditions.items():
            self.assertEqual(interval, self.all_interval)
            self.assertEqual(len(cond_list), 1)
        for timing, effect_list in a.effects.items():
            if timing == self.end_timing:
                self.assertEqual(len(effect_list), 1)
            else:
                self.assertTrue(False)

    def test_forall_reader(self):
        reader = ANMLReader()

        problem_filename = os.path.join(ANML_FILES_PATH, "forall.anml")
        pass  # TODO

    def test_hydrone_reader(self):
        reader = ANMLReader()

        problem_filename = os.path.join(ANML_FILES_PATH, "hydrone.anml")
        problem = reader.parse_problem(problem_filename)
        em = problem.env.expression_manager

        self.assertIsNotNone(problem)
        self.assertEqual(len(problem.fluents), 5)
        self.assertEqual(len(problem.actions), 1)
        self.assertEqual(len(list(problem.objects(problem.user_type("Location")))), 9)
        self.assertEqual(len(problem.goals), 3)
        self.assertEqual(len(problem.timed_goals), 0)
        self.assertEqual(len(problem.timed_effects), 0)

        move = problem.action("move")
        distance_fluent = problem.fluent("distance")
        from_parameter = move.parameter("from")
        destination_parameter = move.parameter("destination")
        self.assertEqual(
            move.duration,
            FixedDuration(distance_fluent(from_parameter, destination_parameter)),
        )
        for interval, cond_list in move.conditions.items():
            self.assertEqual(interval, self.start_interval)
            self.assertEqual(len(cond_list), 5)
        for timing, effect_list in move.effects.items():
            if timing == self.start_timing:
                self.assertEqual(len(effect_list), 2)
            elif timing == self.end_timing:
                self.assertEqual(len(effect_list), 4)
            else:
                self.assertTrue(False)

    def test_match_int_id_reader(self):
        reader = ANMLReader()

        problem_filename = os.path.join(ANML_FILES_PATH, "match_int_id.anml")
        problem = reader.parse_problem(problem_filename)
        em = problem.env.expression_manager

        self.assertIsNotNone(problem)
        self.assertEqual(len(problem.fluents), 4)
        self.assertEqual(len(problem.actions), 2)
        self.assertEqual(len(problem.all_objects), 0)
        self.assertEqual(len(problem.goals), 3)
        self.assertEqual(len(problem.timed_goals), 0)
        self.assertEqual(len(problem.timed_effects), 0)

        light_match = problem.action("light_match")
        self.assertEqual(light_match.duration, FixedDuration(em.Int(6)))
        for interval, cond_list in light_match.conditions.items():
            self.assertEqual(interval, self.start_interval)
            self.assertEqual(len(cond_list), 1)
        for timing, effect_list in light_match.effects.items():
            if timing == self.start_timing:
                self.assertEqual(len(effect_list), 2)
            elif timing == self.end_timing:
                self.assertEqual(len(effect_list), 1)
            else:
                self.assertTrue(False)

        mend_fuse = problem.action("mend_fuse")
        self.assertEqual(mend_fuse.duration, FixedDuration(em.Int(5)))
        for interval, cond_list in mend_fuse.conditions.items():
            if interval in (self.start_interval, self.all_interval):
                self.assertEqual(len(cond_list), 1)
            else:
                self.assertTrue(False)
        for timing, effect_list in mend_fuse.effects.items():
            if timing == self.start_timing:
                self.assertEqual(len(effect_list), 1)
            elif timing == self.end_timing:
                self.assertEqual(len(effect_list), 2)
            else:
                self.assertTrue(False)

    def test_simple_mais_reader(self):
        reader = ANMLReader()

        problem_filename = os.path.join(ANML_FILES_PATH, "simple_mais.anml")
        problem = reader.parse_problem(problem_filename)
        em = problem.env.expression_manager

        self.assertIsNotNone(problem)
        self.assertEqual(len(problem.fluents), 5)
        self.assertEqual(len(problem.actions), 6)
        self.assertEqual(len(problem.all_objects), 0)
        self.assertEqual(len(problem.goals), 1)
        self.assertEqual(len(problem.timed_goals), 0)
        self.assertEqual(len(problem.timed_effects), 0)

        recipe = problem.action("recipe")
        self.assertEqual(recipe.duration, FixedDuration(em.Int(160)))
        possible_delays = [
            10,
            20,
            25,
            35,
            40,
            50,
            55,
            65,
            70,
            80,
            85,
            95,
            100,
            110,
            115,
            125,
            130,
            140,
            145,
            155,
        ]
        possible_timepoint_intervals = [
            TimePointInterval(StartTiming(d)) for d in possible_delays
        ]
        for interval, cond_list in recipe.conditions.items():
            if interval in possible_timepoint_intervals:
                self.assertEqual(len(cond_list), 1)
            else:
                self.assertTrue(False)
        for timing, effect_list in recipe.effects.items():
            if timing == self.start_timing:
                self.assertEqual(len(effect_list), 1)
            elif timing == self.end_timing:
                self.assertEqual(len(effect_list), 1)
            else:
                self.assertTrue(False)

        prepare_bar = problem.action("prepare_bar")
        self.assertEqual(prepare_bar.duration, FixedDuration(em.Int(6)))
        for interval, cond_list in prepare_bar.conditions.items():
            if interval == self.start_interval:
                self.assertEqual(len(cond_list), 1)
            else:
                self.assertTrue(False)
        for timing, effect_list in prepare_bar.effects.items():
            if timing == self.end_timing:
                self.assertEqual(len(effect_list), 1)
            else:
                self.assertTrue(False)

        finish_bar = problem.action("finish_bar")
        self.assertEqual(finish_bar.duration, FixedDuration(em.Int(6)))
        for interval, cond_list in finish_bar.conditions.items():
            if interval == self.start_interval:
                self.assertEqual(len(cond_list), 1)
            else:
                self.assertTrue(False)
        for timing, effect_list in finish_bar.effects.items():
            if timing == self.end_timing:
                self.assertEqual(len(effect_list), 1)
            else:
                self.assertTrue(False)

        load = problem.action("load")
        self.assertEqual(load.duration, FixedDuration(em.Int(1)))
        for interval, cond_list in load.conditions.items():
            if interval == self.start_interval:
                self.assertEqual(len(cond_list), 1)
            else:
                self.assertTrue(False)
        for timing, effect_list in load.effects.items():
            if timing == self.end_timing:
                self.assertEqual(len(effect_list), 1)
            else:
                self.assertTrue(False)

        unload = problem.action("unload")
        self.assertEqual(unload.duration, FixedDuration(em.Int(1)))
        for interval, cond_list in unload.conditions.items():
            if interval == self.start_interval:
                self.assertEqual(len(cond_list), 1)
            else:
                self.assertTrue(False)
        for timing, effect_list in unload.effects.items():
            if timing == self.end_timing:
                self.assertEqual(len(effect_list), 1)
            else:
                self.assertTrue(False)

        move_hoist = problem.action("move_hoist")
        self.assertEqual(move_hoist.duration, FixedDuration(em.Int(1)))
        for interval, cond_list in move_hoist.conditions.items():
            if interval == self.start_interval:
                self.assertEqual(len(cond_list), 1)
            else:
                self.assertTrue(False)
        for timing, effect_list in move_hoist.effects.items():
            if timing == self.end_timing:
                self.assertEqual(len(effect_list), 1)
            else:
                self.assertTrue(False)

    def test_tils_reader(self):
        reader = ANMLReader()

        problem_filename = os.path.join(ANML_FILES_PATH, "tils.anml")
        problem = reader.parse_problem(problem_filename)
        em = problem.env.expression_manager

        self.assertIsNotNone(problem)
        self.assertEqual(len(problem.fluents), 2)
        self.assertEqual(len(problem.actions), 1)
        self.assertEqual(len(problem.all_objects), 0)
        self.assertEqual(len(problem.goals), 1)
        self.assertEqual(len(problem.timed_goals), 0)
        self.assertEqual(len(problem.timed_effects), 2)

        a = problem.action("a")
        self.assertEqual(a.duration, FixedDuration(em.Int(1)))
        for interval, cond_list in a.conditions.items():
            self.assertEqual(interval, self.all_interval)
            self.assertEqual(len(cond_list), 1)
        for timing, effect_list in a.effects.items():
            if timing == self.end_timing:
                self.assertEqual(len(effect_list), 1)
            else:
                self.assertTrue(False)

    def test_a_hierarchical_blocks_world_reader(self):
        reader = ANMLReader()

        problem_filename = os.path.join(
            ANML_FILES_PATH, "hierarchical_blocks_world.anml"
        )
        problem = reader.parse_problem(problem_filename)
        em = problem.env.expression_manager

        self.assertIsNotNone(problem)
        self.assertEqual(len(problem.fluents), 2)
        self.assertEqual(len(problem.actions), 1)
        self.assertEqual(len(problem.goals), 3)
        self.assertEqual(len(problem.timed_goals), 0)
        self.assertEqual(len(problem.timed_effects), 0)
        types_with_6_objects = ("Entity", "Location")
        for ut in problem.user_types:
            if cast(up.model.types._UserType, ut).name in types_with_6_objects:
                self.assertEqual(len(list(problem.objects(ut))), 6)
            else:
                self.assertEqual(len(list(problem.objects(ut))), 3)

        move = problem.action("move")
        self.assertEqual(move.duration, FixedDuration(em.Int(0)))
        for interval, cond_list in move.conditions.items():
            self.assertEqual(interval, self.start_interval)
            self.assertEqual(len(cond_list), 3)
        for timing, effect_list in move.effects.items():
            if timing == self.start_timing:
                self.assertEqual(len(effect_list), 4)
            else:
                self.assertTrue(False)
