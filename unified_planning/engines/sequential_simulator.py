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
#

from typing import Dict, Iterator, List, Optional, Set, Tuple, Union, cast
import unified_planning as up
from unified_planning.engines.compilers import Grounder
from unified_planning.engines.engine import Engine
from unified_planning.engines.mixins.simulator import Event, SimulatorMixin
from unified_planning.exceptions import UPUsageError, UPConflictingEffectsException
from unified_planning.plans import ActionInstance
from unified_planning.model.walkers import StateEvaluator


class InstantaneousEvent(Event):
    """Implements the Event class for an Instantaneous Action."""

    def __init__(
        self,
        conditions: List["up.model.FNode"],
        effects: List["up.model.Effect"],
        simulated_effect: Optional["up.model.SimulatedEffect"] = None,
    ):
        self._conditions = conditions
        self._effects = effects
        self._simulated_effect = simulated_effect

    @property
    def conditions(self) -> List["up.model.FNode"]:
        return self._conditions

    @property
    def effects(self) -> List["up.model.Effect"]:
        return self._effects

    @property
    def simulated_effect(self) -> Optional["up.model.SimulatedEffect"]:
        return self._simulated_effect


class SequentialSimulator(Engine, SimulatorMixin):
    """
    Sequential SimulatorMixin implementation.
    """

    def __init__(self, problem: "up.model.Problem"):
        SimulatorMixin.__init__(self, problem)
        pk = problem.kind
        assert Grounder.supports(pk)
        assert isinstance(self._problem, up.model.Problem)
        grounder = Grounder()
        self._grounding_result = grounder.compile(
            self._problem, up.engines.CompilationKind.GROUNDING
        )

        grounded_problem: "up.model.Problem" = cast(
            up.model.Problem, self._grounding_result.problem
        )
        lift_map = self._grounding_result.map_back_action_instance
        assert lift_map is not None
        self._events: Dict[
            Tuple["up.model.Action", Tuple["up.model.FNode", ...]], List[Event]
        ] = {}
        for grounded_action in grounded_problem.actions:
            if isinstance(grounded_action, up.model.InstantaneousAction):
                lifted_ai = lift_map(ActionInstance(grounded_action))
                assert lifted_ai is not None
                event_list = self._events.setdefault(
                    (lifted_ai.action, lifted_ai.actual_parameters), []
                )
                event_list.append(
                    InstantaneousEvent(
                        grounded_action.preconditions,
                        grounded_action.effects,
                        grounded_action.simulated_effect,
                    )
                )
            else:
                raise NotImplementedError
        self._se = StateEvaluator(self._problem)

    def _get_unsatisfied_conditions(
        self, event: "Event", state: "up.model.ROState", early_termination: bool = False
    ) -> List["up.model.FNode"]:
        """
        Returns the list of unsatisfied event conditions evaluated in the given state.
        If the flag "early_termination" is set, the method ends and returns at the first unsatisfied condition.
        :param state: The State in which the event conditions are evaluated.
        :param early_termination: Flag deciding if the method ends and returns at the first unsatisfied condition.
        :return: The list of all the event conditions that evaluated to False or the list containing the first condition evaluated to False if the flag "early_termination" is set.
        """
        # Evaluate every condition and if the condition is False or the condition is not simplified as a
        # boolean constant in the given state, return False. Return True otherwise
        unsatisfied_conditions = []
        for c in event.conditions:
            evaluated_cond = self._se.evaluate(c, state)
            if (
                not evaluated_cond.is_bool_constant()
                or not evaluated_cond.bool_constant_value()
            ):
                unsatisfied_conditions.append(c)
                if early_termination:
                    break
        return unsatisfied_conditions

    def _apply(
        self, event: "Event", state: "up.model.COWState"
    ) -> Optional["up.model.COWState"]:
        """
        Returns None if the event is not applicable in the given state, otherwise returns a new COWState,
        which is a copy of the given state but the applicable effects of the event are applied; therefore
        some fluent values are updated.
        :param state: the state where the event formulas are calculated.
        :param event: the event that has the information about the conditions to check and the effects to apply.
        :return: None if the event is not applicable in the given state, a new COWState with some updated values
         if the event is applicable.
        """
        if not self.is_applicable(event, state):
            return None
        else:
            return self.apply_unsafe(event, state)

    def _apply_unsafe(
        self, event: "Event", state: "up.model.COWState"
    ) -> "up.model.COWState":
        """
        Returns a new COWState, which is a copy of the given state but the applicable effects of the event are applied; therefore
        some fluent values are updated.
        IMPORTANT NOTE: Assumes that self.is_applicable(state, event) returns True
        :param state: the state where the event formulas are evaluated.
        :param event: the event that has the information about the effects to apply.
        :return: A new COWState with some updated values.
        """
        updated_values: Dict["up.model.FNode", "up.model.FNode"] = {}
        assigned_fluent: Set["up.model.FNode"] = set()
        em = self._problem.env.expression_manager
        for e in event.effects:
            evaluated_args = tuple(self._se.evaluate(a, state) for a in e.fluent.args)
            fluent = self._problem.env.expression_manager.FluentExp(
                e.fluent.fluent(), evaluated_args
            )
            if (not e.is_conditional()) or self._se.evaluate(
                e.condition, state
            ).is_true():
                if e.is_assignment():
                    if fluent in updated_values:
                        raise UPConflictingEffectsException(
                            f"The fluent {fluent} is modified by 2 assignments in the same event."
                        )
                    updated_values[fluent] = self._se.evaluate(e.value, state)
                    assigned_fluent.add(fluent)
                else:
                    if fluent in assigned_fluent:
                        raise UPConflictingEffectsException(
                            f"The fluent {fluent} is modified by an assignment and an increase/decrease in the same event."
                        )
                    # If the fluent is in updated_values, we take his modified value, (which was modified by another increase or deacrease)
                    # otherwisee we take it's evaluation in the state as it's value.
                    f_eval = updated_values.get(
                        fluent, self._se.evaluate(fluent, state)
                    )
                    v_eval = self._se.evaluate(e.value, state)
                    if e.is_increase():
                        updated_values[fluent] = em.auto_promote(
                            f_eval.constant_value() + v_eval.constant_value()
                        )[0]
                    elif e.is_decrease():
                        updated_values[fluent] = em.auto_promote(
                            f_eval.constant_value() - v_eval.constant_value()
                        )[0]
                    else:
                        raise NotImplementedError
        if event.simulated_effect is not None:
            for f, v in zip(
                event.simulated_effect.fluents,
                event.simulated_effect.function(self._problem, state, {}),
            ):
                if f in updated_values:
                    raise UPConflictingEffectsException(
                        f"The fluent {f} is modified twice in the same event."
                    )
                updated_values[f] = v
        return state.make_child(updated_values)

    def _get_applicable_events(self, state: "up.model.ROState") -> Iterator["Event"]:
        """
        Returns a view over all the events that are applicable in the given State;
        an Event is considered applicable in a given State, when all the Event condition
        simplify as True when evaluated in the State.
        :param state: the state where the formulas are evaluated.
        :return: an Iterator of applicable Events.
        """
        for events in self._events.values():
            for event in events:
                if self.is_applicable(event, state):
                    yield event

    def _get_events(
        self,
        action: "up.model.Action",
        parameters: Union[
            Tuple["up.model.Expression", ...], List["up.model.Expression"]
        ],
    ) -> List["Event"]:
        """
        Returns a list containing all the events derived from the given action, grounded with the given parameters.
        :param action: the action containing the information to create the event.
        :param parameters: the parameters needed to ground the action
        :return: the List of Events derived from this action with these parameters.
        """
        if action not in cast(up.model.Problem, self._problem).actions:
            raise UPUsageError(
                "The action given as parameter does not belong to the problem given to the SequentialSimulator."
            )
        params_exp = tuple(
            self._problem.env.expression_manager.auto_promote(parameters)
        )
        return self._events.get((action, tuple(params_exp)), [])

    def _get_unsatisfied_goals(
        self, state: "up.model.ROState", early_termination: bool = False
    ) -> List["up.model.FNode"]:
        """
        Returns the list of unsatisfied goals evaluated in the given state.
        If the flag "early_termination" is set, the method ends and returns at the first unsatisfied goal.
        :param state: The State in which the problem goals are evaluated.
        :param early_termination: Flag deciding if the method ends and returns at the first unsatisfied goal.
        :return: The list of all the goals that evaluated to False or the list containing the first goal evaluated to False if the flag "early_termination" is set.
        """
        unsatisfied_goals = []
        for g in cast(up.model.Problem, self._problem).goals:
            g_eval = self._se.evaluate(g, state).bool_constant_value()
            if not g_eval:
                unsatisfied_goals.append(g)
                if early_termination:
                    break
        return unsatisfied_goals

    @property
    def name(self) -> str:
        return "sequential_simulator"

    @staticmethod
    def supported_kind() -> "up.model.ProblemKind":
        supported_kind = up.model.ProblemKind()
        supported_kind.set_problem_class("ACTION_BASED")
        supported_kind.set_typing("FLAT_TYPING")
        supported_kind.set_typing("HIERARCHICAL_TYPING")
        supported_kind.set_numbers("CONTINUOUS_NUMBERS")
        supported_kind.set_numbers("DISCRETE_NUMBERS")
        supported_kind.set_problem_type("SIMPLE_NUMERIC_PLANNING")
        supported_kind.set_problem_type("GENERAL_NUMERIC_PLANNING")
        supported_kind.set_fluents_type("NUMERIC_FLUENTS")
        supported_kind.set_fluents_type("OBJECT_FLUENTS")
        supported_kind.set_conditions_kind("NEGATIVE_CONDITIONS")
        supported_kind.set_conditions_kind("DISJUNCTIVE_CONDITIONS")
        supported_kind.set_conditions_kind("EQUALITY")
        supported_kind.set_conditions_kind("EXISTENTIAL_CONDITIONS")
        supported_kind.set_conditions_kind("UNIVERSAL_CONDITIONS")
        supported_kind.set_effects_kind("CONDITIONAL_EFFECTS")
        supported_kind.set_effects_kind("INCREASE_EFFECTS")
        supported_kind.set_effects_kind("DECREASE_EFFECTS")
        supported_kind.set_simulated_entities("SIMULATED_EFFECTS")
        return supported_kind

    @staticmethod
    def supports(problem_kind):
        return problem_kind <= SequentialSimulator.supported_kind()
