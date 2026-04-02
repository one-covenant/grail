"""Tests for the Universal Regulatory Compliance Reasoning environment.

Covers:
- Scenario generation determinism and distribution
- Parser robustness (valid JSON, malformed input, edge cases)
- Reward function correctness (verdict, rule accuracy, precision, recall)
- Environment lifecycle (reset/step contract)
- Multi-framework coverage
"""

from __future__ import annotations

import json

import pytest

from grail.environments.compliance_env import (
    ComplianceEnv,
    ComplianceParser,
    ComplianceScenario,
    ComplianceTaskSource,
    RegulatoryFramework,
    _build_compliance_prompt,
    _create_compliance_reward_vector,
    _ensure_frameworks,
    _generate_scenario,
    _rule_accuracy_reward,
    _rule_coverage_reward,
    _severity_weighted_accuracy,
    _verdict_reward,
    _violation_precision_reward,
    _violation_recall_reward,
)
from grail.environments.core import ChatMessage

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def frameworks() -> dict[str, RegulatoryFramework]:
    return _ensure_frameworks()


@pytest.fixture
def gdpr_scenario() -> ComplianceScenario:
    """A GDPR scenario with known violations."""
    return _generate_scenario(seed=42, framework_id="gdpr")


@pytest.fixture
def compliant_scenario() -> ComplianceScenario:
    """Find a fully compliant scenario by scanning seeds."""
    for seed in range(1000):
        s = _generate_scenario(seed, framework_id="gdpr")
        if s.expected_verdict:
            return s
    pytest.skip("Could not find a compliant scenario in 1000 seeds")


@pytest.fixture
def noncompliant_scenario() -> ComplianceScenario:
    """Find a non-compliant scenario by scanning seeds."""
    for seed in range(1000):
        s = _generate_scenario(seed, framework_id="osha")
        if not s.expected_verdict:
            return s
    pytest.skip("Could not find a non-compliant scenario in 1000 seeds")


@pytest.fixture
def parser() -> ComplianceParser:
    return ComplianceParser()


# =============================================================================
# FRAMEWORK REGISTRY
# =============================================================================


class TestFrameworkRegistry:
    def test_all_frameworks_registered(self, frameworks: dict[str, RegulatoryFramework]) -> None:
        expected = {"gdpr", "osha", "sox", "fda_21cfr11", "epa"}
        assert set(frameworks.keys()) == expected

    def test_each_framework_has_rules(self, frameworks: dict[str, RegulatoryFramework]) -> None:
        for fw_id, fw in frameworks.items():
            assert len(fw.rules) >= 5, f"{fw_id} should have at least 5 rules"

    def test_rule_ids_unique_within_framework(
        self, frameworks: dict[str, RegulatoryFramework]
    ) -> None:
        for fw_id, fw in frameworks.items():
            ids = [r.rule_id for r in fw.rules]
            assert len(ids) == len(set(ids)), f"Duplicate rule IDs in {fw_id}"

    def test_predicate_keys_unique_within_framework(
        self, frameworks: dict[str, RegulatoryFramework]
    ) -> None:
        for fw_id, fw in frameworks.items():
            keys = [r.predicate_key for r in fw.rules]
            assert len(keys) == len(set(keys)), f"Duplicate predicate keys in {fw_id}"

    def test_severity_values_valid(self, frameworks: dict[str, RegulatoryFramework]) -> None:
        valid = {"critical", "major", "minor"}
        for fw_id, fw in frameworks.items():
            for rule in fw.rules:
                assert rule.severity in valid, (
                    f"Rule {rule.rule_id} in {fw_id} has invalid severity: {rule.severity}"
                )


# =============================================================================
# SCENARIO GENERATION
# =============================================================================


class TestScenarioGeneration:
    def test_deterministic(self) -> None:
        """Same seed produces identical scenarios."""
        s1 = _generate_scenario(seed=123)
        s2 = _generate_scenario(seed=123)
        assert s1.scenario_id == s2.scenario_id
        assert s1.framework.framework_id == s2.framework.framework_id
        assert s1.facts == s2.facts
        assert s1.expected_verdict == s2.expected_verdict
        assert s1.expected_rule_results == s2.expected_rule_results

    def test_different_seeds_differ(self) -> None:
        """Different seeds produce different scenarios (with high probability)."""
        scenarios = [_generate_scenario(seed=i) for i in range(20)]
        ids = [s.scenario_id for s in scenarios]
        # At least most should be unique
        assert len(set(ids)) > 10

    def test_framework_filtering(self) -> None:
        """framework_id restricts generation to specified framework."""
        for _ in range(20):
            import random

            seed = random.randint(0, 10000)
            s = _generate_scenario(seed, framework_id="sox")
            assert s.framework.framework_id == "sox"

    def test_invalid_framework_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown framework"):
            _generate_scenario(seed=1, framework_id="nonexistent")

    def test_facts_match_framework_rules(self) -> None:
        """Every rule's predicate_key should appear in the facts dict."""
        s = _generate_scenario(seed=42, framework_id="gdpr")
        for rule in s.framework.rules:
            assert rule.predicate_key in s.facts, (
                f"Rule {rule.rule_id} predicate_key '{rule.predicate_key}' not in scenario facts"
            )

    def test_expected_rule_results_match_rules(self) -> None:
        """Every framework rule should have an expected result."""
        s = _generate_scenario(seed=42, framework_id="osha")
        for rule in s.framework.rules:
            assert rule.rule_id in s.expected_rule_results

    def test_verdict_consistency(self) -> None:
        """Verdict should be True iff num_violations == 0."""
        for seed in range(50):
            s = _generate_scenario(seed)
            assert s.expected_verdict == (s.num_violations == 0), (
                f"Seed {seed}: verdict={s.expected_verdict} but num_violations={s.num_violations}"
            )

    def test_violation_count_matches_rule_results(self) -> None:
        """num_violations should match count of False in expected_rule_results."""
        for seed in range(50):
            s = _generate_scenario(seed)
            actual_violations = sum(1 for v in s.expected_rule_results.values() if not v)
            assert s.num_violations == actual_violations

    def test_violation_distribution(self) -> None:
        """Over many seeds, we should see a mix of compliant and non-compliant."""
        compliant_count = 0
        total = 200
        for seed in range(total):
            s = _generate_scenario(seed)
            if s.expected_verdict:
                compliant_count += 1

        # Expect roughly 30-50% compliant (target is 40%)
        ratio = compliant_count / total
        assert 0.20 < ratio < 0.60, f"Compliant ratio {ratio:.2f} outside expected range"

    def test_narrative_not_empty(self) -> None:
        """Narrative should contain meaningful text."""
        for fw_id in ["gdpr", "osha", "sox", "fda_21cfr11", "epa"]:
            s = _generate_scenario(seed=42, framework_id=fw_id)
            assert len(s.narrative) > 100, f"Narrative for {fw_id} too short"
            assert "\n" in s.narrative, "Narrative should be multi-line"


# =============================================================================
# PROMPT CONSTRUCTION
# =============================================================================


class TestPromptConstruction:
    def test_prompt_contains_framework_name(self, gdpr_scenario: ComplianceScenario) -> None:
        prompt = _build_compliance_prompt(gdpr_scenario)
        assert "General Data Protection Regulation" in prompt

    def test_prompt_contains_all_rule_ids(self, gdpr_scenario: ComplianceScenario) -> None:
        prompt = _build_compliance_prompt(gdpr_scenario)
        for rule in gdpr_scenario.framework.rules:
            assert rule.rule_id in prompt

    def test_prompt_contains_scenario_narrative(self, gdpr_scenario: ComplianceScenario) -> None:
        prompt = _build_compliance_prompt(gdpr_scenario)
        # Narrative should appear in the prompt
        assert "SCENARIO:" in prompt

    def test_prompt_contains_solution_tags_instruction(
        self, gdpr_scenario: ComplianceScenario
    ) -> None:
        prompt = _build_compliance_prompt(gdpr_scenario)
        assert "<SOLUTION>" in prompt
        assert "</SOLUTION>" in prompt

    def test_prompt_contains_json_format_instruction(
        self, gdpr_scenario: ComplianceScenario
    ) -> None:
        prompt = _build_compliance_prompt(gdpr_scenario)
        assert "verdict" in prompt
        assert "rule_results" in prompt
        assert "violations" in prompt


# =============================================================================
# PARSER
# =============================================================================


class TestComplianceParser:
    def test_parse_valid_compliant(
        self, parser: ComplianceParser, compliant_scenario: ComplianceScenario
    ) -> None:
        answer = json.dumps(
            {
                "verdict": "COMPLIANT",
                "rule_results": {
                    r.rule_id: "COMPLIANT" for r in compliant_scenario.framework.rules
                },
                "violations": [],
            }
        )
        completion = f"<think>Analysis...</think>\n<SOLUTION>{answer}</SOLUTION>"
        parsed = parser.parse(completion, compliant_scenario)

        assert parsed["verdict"] == "COMPLIANT"
        assert parsed["parse_success"] is True
        assert parsed["has_thinking"] is True
        assert parsed["has_answer"] is True
        assert len(parsed["violations"]) == 0

    def test_parse_valid_noncompliant(
        self, parser: ComplianceParser, noncompliant_scenario: ComplianceScenario
    ) -> None:
        violations = [
            r_id for r_id, comp in noncompliant_scenario.expected_rule_results.items() if not comp
        ]
        rule_results = {}
        for rule in noncompliant_scenario.framework.rules:
            rule_results[rule.rule_id] = (
                "NON-COMPLIANT" if rule.rule_id in violations else "COMPLIANT"
            )

        answer = json.dumps(
            {
                "verdict": "NON-COMPLIANT",
                "rule_results": rule_results,
                "violations": violations,
            }
        )
        completion = f"<think>Analysis...</think>\n<SOLUTION>{answer}</SOLUTION>"
        parsed = parser.parse(completion, noncompliant_scenario)

        assert parsed["verdict"] == "NON-COMPLIANT"
        assert parsed["parse_success"] is True
        assert set(parsed["violations"]) == set(violations)

    def test_parse_empty_completion(
        self, parser: ComplianceParser, gdpr_scenario: ComplianceScenario
    ) -> None:
        parsed = parser.parse("", gdpr_scenario)
        assert parsed["verdict"] is None
        assert parsed["parse_success"] is False
        assert parsed["has_answer"] is False

    def test_parse_no_solution_tags(
        self, parser: ComplianceParser, gdpr_scenario: ComplianceScenario
    ) -> None:
        parsed = parser.parse("The scenario is compliant.", gdpr_scenario)
        assert parsed["verdict"] is None
        assert parsed["has_answer"] is False

    def test_parse_malformed_json(
        self, parser: ComplianceParser, gdpr_scenario: ComplianceScenario
    ) -> None:
        completion = "<SOLUTION>{not valid json}</SOLUTION>"
        parsed = parser.parse(completion, gdpr_scenario)
        assert parsed["parse_success"] is False

    def test_parse_json_with_extra_text(
        self, parser: ComplianceParser, gdpr_scenario: ComplianceScenario
    ) -> None:
        answer = json.dumps({"verdict": "COMPLIANT", "rule_results": {}, "violations": []})
        completion = f"<SOLUTION>Here is my answer: {answer}</SOLUTION>"
        parsed = parser.parse(completion, gdpr_scenario)
        # Should still find JSON within the content
        assert parsed["parse_success"] is True
        assert parsed["verdict"] == "COMPLIANT"

    def test_parse_case_insensitive_verdict(
        self, parser: ComplianceParser, gdpr_scenario: ComplianceScenario
    ) -> None:
        answer = json.dumps({"verdict": "compliant", "rule_results": {}, "violations": []})
        completion = f"<SOLUTION>{answer}</SOLUTION>"
        parsed = parser.parse(completion, gdpr_scenario)
        assert parsed["verdict"] == "COMPLIANT"

    def test_trailing_text_counted(
        self, parser: ComplianceParser, gdpr_scenario: ComplianceScenario
    ) -> None:
        answer = json.dumps({"verdict": "COMPLIANT", "rule_results": {}, "violations": []})
        trailing = "Some extra text"
        completion = f"<SOLUTION>{answer}</SOLUTION>{trailing}"
        parsed = parser.parse(completion, gdpr_scenario)
        assert parsed["trailing_after_answer"] == len(trailing)


# =============================================================================
# REWARD FUNCTIONS
# =============================================================================


class TestVerdictReward:
    def test_correct_compliant(self, compliant_scenario: ComplianceScenario) -> None:
        parsed = {"verdict": "COMPLIANT"}
        assert _verdict_reward(parsed, compliant_scenario) == 1.0

    def test_correct_noncompliant(self, noncompliant_scenario: ComplianceScenario) -> None:
        parsed = {"verdict": "NON-COMPLIANT"}
        assert _verdict_reward(parsed, noncompliant_scenario) == 1.0

    def test_wrong_verdict(self, compliant_scenario: ComplianceScenario) -> None:
        parsed = {"verdict": "NON-COMPLIANT"}
        assert _verdict_reward(parsed, compliant_scenario) == 0.0

    def test_no_verdict(self, compliant_scenario: ComplianceScenario) -> None:
        parsed = {"verdict": None}
        assert _verdict_reward(parsed, compliant_scenario) == -0.2

    def test_non_dict_input(self, compliant_scenario: ComplianceScenario) -> None:
        assert _verdict_reward("not a dict", compliant_scenario) == -0.2


class TestRuleAccuracyReward:
    def test_all_correct(self, gdpr_scenario: ComplianceScenario) -> None:
        rule_results = {}
        for rule_id, compliant in gdpr_scenario.expected_rule_results.items():
            rule_results[rule_id] = "COMPLIANT" if compliant else "NON-COMPLIANT"
        parsed = {"rule_results": rule_results}
        assert _rule_accuracy_reward(parsed, gdpr_scenario) == pytest.approx(1.0)

    def test_all_wrong(self, gdpr_scenario: ComplianceScenario) -> None:
        rule_results = {}
        for rule_id, compliant in gdpr_scenario.expected_rule_results.items():
            # Invert expected
            rule_results[rule_id] = "NON-COMPLIANT" if compliant else "COMPLIANT"
        parsed = {"rule_results": rule_results}
        assert _rule_accuracy_reward(parsed, gdpr_scenario) == pytest.approx(0.0)

    def test_empty_results(self, gdpr_scenario: ComplianceScenario) -> None:
        parsed = {"rule_results": {}}
        assert _rule_accuracy_reward(parsed, gdpr_scenario) == 0.0

    def test_partial_results(self, gdpr_scenario: ComplianceScenario) -> None:
        """Partial results yield partial credit."""
        rules = list(gdpr_scenario.expected_rule_results.items())
        half = len(rules) // 2
        rule_results = {}
        for rule_id, compliant in rules[:half]:
            rule_results[rule_id] = "COMPLIANT" if compliant else "NON-COMPLIANT"
        parsed = {"rule_results": rule_results}
        reward = _rule_accuracy_reward(parsed, gdpr_scenario)
        expected = half / len(rules)
        assert reward == pytest.approx(expected)


class TestViolationPrecisionReward:
    def test_perfect_precision(self, noncompliant_scenario: ComplianceScenario) -> None:
        real_violations = [
            r_id for r_id, comp in noncompliant_scenario.expected_rule_results.items() if not comp
        ]
        parsed = {"violations": real_violations}
        assert _violation_precision_reward(parsed, noncompliant_scenario) == pytest.approx(1.0)

    def test_false_positive(self, compliant_scenario: ComplianceScenario) -> None:
        # Claiming violations when there are none
        parsed = {"violations": ["FAKE-RULE-1"]}
        assert _violation_precision_reward(parsed, compliant_scenario) == pytest.approx(0.0)

    def test_no_violations_claimed_compliant(self, compliant_scenario: ComplianceScenario) -> None:
        parsed = {"violations": []}
        assert _violation_precision_reward(parsed, compliant_scenario) == pytest.approx(0.5)


class TestViolationRecallReward:
    def test_perfect_recall(self, noncompliant_scenario: ComplianceScenario) -> None:
        real_violations = [
            r_id for r_id, comp in noncompliant_scenario.expected_rule_results.items() if not comp
        ]
        parsed = {"violations": real_violations}
        assert _violation_recall_reward(parsed, noncompliant_scenario) == pytest.approx(1.0)

    def test_no_violations_found(self, noncompliant_scenario: ComplianceScenario) -> None:
        parsed = {"violations": []}
        assert _violation_recall_reward(parsed, noncompliant_scenario) == pytest.approx(0.0)

    def test_compliant_scenario_returns_half(self, compliant_scenario: ComplianceScenario) -> None:
        parsed = {"violations": []}
        assert _violation_recall_reward(parsed, compliant_scenario) == pytest.approx(0.5)


class TestRuleCoverageReward:
    def test_full_coverage(self, gdpr_scenario: ComplianceScenario) -> None:
        rule_results = {r.rule_id: "COMPLIANT" for r in gdpr_scenario.framework.rules}
        parsed = {"rule_results": rule_results}
        assert _rule_coverage_reward(parsed, gdpr_scenario) == pytest.approx(1.0)

    def test_no_coverage(self, gdpr_scenario: ComplianceScenario) -> None:
        parsed = {"rule_results": {}}
        assert _rule_coverage_reward(parsed, gdpr_scenario) == pytest.approx(0.0)

    def test_partial_coverage(self, gdpr_scenario: ComplianceScenario) -> None:
        rules = gdpr_scenario.framework.rules
        half = len(rules) // 2
        rule_results = {r.rule_id: "COMPLIANT" for r in rules[:half]}
        parsed = {"rule_results": rule_results}
        expected = half / len(rules)
        assert _rule_coverage_reward(parsed, gdpr_scenario) == pytest.approx(expected)


class TestSeverityWeightedAccuracy:
    def test_all_correct(self, gdpr_scenario: ComplianceScenario) -> None:
        rule_results = {}
        for rule_id, compliant in gdpr_scenario.expected_rule_results.items():
            rule_results[rule_id] = "COMPLIANT" if compliant else "NON-COMPLIANT"
        parsed = {"rule_results": rule_results}
        assert _severity_weighted_accuracy(parsed, gdpr_scenario) == pytest.approx(1.0)

    def test_empty_results(self, gdpr_scenario: ComplianceScenario) -> None:
        parsed = {"rule_results": {}}
        assert _severity_weighted_accuracy(parsed, gdpr_scenario) == pytest.approx(0.0)


# =============================================================================
# REWARD VECTOR
# =============================================================================


class TestRewardVector:
    def test_reward_vector_weights_sum_to_one(self) -> None:
        rv = _create_compliance_reward_vector()
        assert sum(rv.weights) == pytest.approx(1.0)

    def test_reward_vector_has_bounds(self) -> None:
        rv = _create_compliance_reward_vector()
        assert rv.has_bounds()
        lo, hi = rv.reward_bounds()
        assert lo < hi
        assert lo < 0  # verdict penalty can go negative

    def test_perfect_score_achievable(self, compliant_scenario: ComplianceScenario) -> None:
        """A perfect response should achieve a high reward."""
        rule_results = {}
        for rule_id, compliant in compliant_scenario.expected_rule_results.items():
            rule_results[rule_id] = "COMPLIANT" if compliant else "NON-COMPLIANT"

        answer = json.dumps(
            {
                "verdict": "COMPLIANT",
                "rule_results": rule_results,
                "violations": [],
            }
        )
        completion = f"<think>Thorough analysis...</think>\n<SOLUTION>{answer}</SOLUTION>"

        rv = _create_compliance_reward_vector()
        reward = rv.compute_reward(completion, compliant_scenario)
        # Should be high (close to max)
        assert reward > 0.7, f"Perfect response reward {reward:.3f} too low"


# =============================================================================
# ENVIRONMENT LIFECYCLE
# =============================================================================


class TestComplianceEnv:
    def test_reset_returns_observation(self) -> None:
        env = ComplianceEnv()
        obs = env.reset(seed=42)
        assert obs.turn_index == 0
        assert len(obs.messages) == 1
        assert obs.messages[0].role == "user"
        assert "REGULATORY COMPLIANCE ANALYSIS" in obs.messages[0].content

    def test_step_returns_reward(self) -> None:
        env = ComplianceEnv()
        obs = env.reset(seed=42)
        answer = json.dumps(
            {
                "verdict": "COMPLIANT",
                "rule_results": {},
                "violations": [],
            }
        )
        completion = f"<SOLUTION>{answer}</SOLUTION>"
        obs, reward, terminated, truncated, info = env.step(
            ChatMessage(role="assistant", content=completion)
        )
        assert isinstance(reward, float)
        assert terminated is True  # SingleTurnEnv forces termination
        assert "success" in info
        assert "rules_correct" in info
        assert "rules_total" in info

    def test_step_without_reset_raises(self) -> None:
        env = ComplianceEnv()
        with pytest.raises(RuntimeError, match="Must call reset"):
            env.step(ChatMessage(role="assistant", content="test"))

    def test_framework_filtering(self) -> None:
        env = ComplianceEnv(framework_id="sox")
        obs = env.reset(seed=42)
        assert "Sarbanes-Oxley" in obs.messages[0].content

    def test_deterministic_across_resets(self) -> None:
        env = ComplianceEnv()
        obs1 = env.reset(seed=99)
        obs2 = env.reset(seed=99)
        assert obs1.messages[0].content == obs2.messages[0].content

    def test_info_contains_diagnostics(self) -> None:
        env = ComplianceEnv(framework_id="gdpr")
        _obs = env.reset(seed=42)

        answer = json.dumps(
            {
                "verdict": "COMPLIANT",
                "rule_results": {},
                "violations": [],
            }
        )
        _, _, _, _, info = env.step(
            ChatMessage(role="assistant", content=f"<SOLUTION>{answer}</SOLUTION>")
        )

        assert "framework_id" in info
        assert info["framework_id"] == "gdpr"
        assert "num_violations" in info
        assert "rules_accuracy" in info
        assert "reward_components" in info

    def test_perfect_answer_high_reward(self) -> None:
        """A perfect answer for a known scenario should get high reward."""
        env = ComplianceEnv(framework_id="gdpr")
        obs = env.reset(seed=42)  # noqa: F841

        # Get the scenario from env internals
        scenario = env._scenario
        assert scenario is not None

        rule_results = {}
        violations = []
        for rule_id, compliant in scenario.expected_rule_results.items():
            rule_results[rule_id] = "COMPLIANT" if compliant else "NON-COMPLIANT"
            if not compliant:
                violations.append(rule_id)

        expected_verdict = "COMPLIANT" if scenario.expected_verdict else "NON-COMPLIANT"
        answer = json.dumps(
            {
                "verdict": expected_verdict,
                "rule_results": rule_results,
                "violations": violations,
            }
        )
        completion = f"<think>Analyzing each rule...</think>\n<SOLUTION>{answer}</SOLUTION>"

        _, reward, _, _, info = env.step(ChatMessage(role="assistant", content=completion))

        assert info["success"] is True
        assert info["rules_correct"] == info["rules_total"]
        assert reward > 0.7


class TestTaskSource:
    def test_deterministic(self) -> None:
        source = ComplianceTaskSource()
        t1 = source.next(seed=42)
        t2 = source.next(seed=42)
        assert t1.id == t2.id

    def test_framework_filtering(self) -> None:
        source = ComplianceTaskSource(framework_id="epa")
        for seed in range(10):
            task = source.next(seed=seed)
            assert task.metadata["framework_id"] == "epa"

    def test_task_id_fallback(self) -> None:
        source = ComplianceTaskSource()
        task = source.next(task_id="test-scenario-1")
        assert task.id is not None

    def test_metadata_fields(self) -> None:
        source = ComplianceTaskSource()
        task = source.next(seed=42)
        assert "framework_id" in task.metadata
        assert "num_rules" in task.metadata
        assert "num_violations" in task.metadata
        assert "expected_verdict" in task.metadata
