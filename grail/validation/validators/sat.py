"""SAT-specific validators for problem, prompt, and solution verification."""

from __future__ import annotations

import logging
from typing import Any

from ...environments import generate_sat_problem
from ...environments.sat import SATParser, create_sat_prompt
from ...mining.rollout_generator import REASONING_START, SYSTEM_PROMPT
from ..base import Validator
from ..context import ValidationContext

logger = logging.getLogger(__name__)


class SATProblemValidator(Validator):
    """Verifies SAT problem regenerates deterministically from seed."""

    check_name = "sat_problem_valid"

    def validate(self, ctx: ValidationContext) -> bool:
        """Regenerate SAT problem and verify it matches commit."""
        sat_data = ctx.commit.get("sat_problem")
        if not sat_data:
            logger.debug("Missing SAT problem data in commit")
            ctx.checks[self.check_name] = False
            return False

        try:
            seed = sat_data["seed"]
            difficulty = sat_data.get("difficulty", 0.5)
            problem = generate_sat_problem(seed, difficulty)

            # Verify structure matches
            if problem.num_vars != sat_data.get("num_vars"):
                logger.debug(
                    f"SAT num_vars mismatch: expected {problem.num_vars}, "
                    f"got {sat_data.get('num_vars')}"
                )
                ctx.checks[self.check_name] = False
                return False

            if problem.clauses != sat_data.get("clauses"):
                logger.debug("SAT clauses mismatch")
                ctx.checks[self.check_name] = False
                return False

            # Cache for downstream validators
            ctx.verified_problem = problem
            ctx.checks[self.check_name] = True
            return True

        except Exception as e:
            logger.debug(f"SAT problem validation error: {e}")
            ctx.checks[self.check_name] = False
            return False


class SATPromptValidator(Validator):
    """Verifies canonical prompt prefix matches exactly."""

    check_name = "prompt_valid"

    def validate(self, ctx: ValidationContext) -> bool:
        """Verify commit tokens start with canonical SAT prompt."""
        problem = ctx.verified_problem
        if problem is None:
            logger.debug("No verified problem for prompt validation")
            ctx.checks[self.check_name] = False
            return False

        try:
            # Build canonical prompt
            user_prompt = create_sat_prompt(problem)
            messages = [{"role": "user", "content": user_prompt}]

            # Render through chat template
            try:
                rendered = ctx.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback template
                eos = ctx.tokenizer.eos_token or ""
                rendered = f"{SYSTEM_PROMPT}{eos}{user_prompt}{REASONING_START}"

            # Tokenize canonical prompt
            canonical_ids = (
                ctx.tokenizer(rendered, return_tensors="pt", return_attention_mask=False)
                .input_ids[0]
                .tolist()
            )

            # Extract commit data
            tokens = ctx.commit.get("tokens", [])
            rollout = ctx.commit.get("rollout", {})
            prompt_len = int(rollout.get("prompt_length", 0))
            completion_len = int(rollout.get("completion_length", 0))

            # Verify lengths
            if prompt_len != len(canonical_ids):
                logger.debug(f"Prompt length mismatch: {prompt_len} != {len(canonical_ids)}")
                ctx.checks[self.check_name] = False
                return False

            if prompt_len + completion_len != len(tokens):
                logger.debug(
                    f"Token count mismatch: {prompt_len}+{completion_len} != {len(tokens)}"
                )
                ctx.checks[self.check_name] = False
                return False

            # Verify prefix matches
            if tokens[:prompt_len] != canonical_ids:
                logger.debug("Prompt prefix mismatch")
                ctx.checks[self.check_name] = False
                return False

            ctx.checks[self.check_name] = True
            return True

        except Exception as e:
            logger.debug(f"Prompt validation error: {e}")
            ctx.checks[self.check_name] = False
            return False


class SATSolutionValidator(Validator):
    """Verifies assignment solves SAT problem if success claimed."""

    check_name = "solution_valid"

    def validate(self, ctx: ValidationContext) -> bool:
        """Verify solution assignment if success is claimed."""
        rollout = ctx.commit.get("rollout", {})

        # If not claiming success, check is vacuously true
        if not rollout.get("success"):
            ctx.checks[self.check_name] = True
            return True

        problem = ctx.verified_problem
        if problem is None:
            logger.debug("No verified problem for solution validation")
            ctx.checks[self.check_name] = False
            return False

        try:
            # Parse assignment from completion text
            assignment = self._extract_assignment(ctx, problem)
            if assignment is None:
                logger.debug("Failed to parse assignment from completion")
                ctx.checks[self.check_name] = False
                return False

            # Verify assignment in rollout matches parsed
            claimed = rollout.get("assignment", [])
            if not isinstance(claimed, list) or len(claimed) != problem.num_vars:
                logger.debug(f"Invalid claimed assignment structure: len={len(claimed)}")
                ctx.checks[self.check_name] = False
                return False

            if assignment != claimed:
                logger.debug("Assignment mismatch: parsed != claimed")
                ctx.checks[self.check_name] = False
                return False

            # Verify assignment solves problem
            if not problem.check_solution(assignment):
                logger.debug("Assignment does not solve SAT problem")
                ctx.checks[self.check_name] = False
                return False

            ctx.checks[self.check_name] = True
            return True

        except Exception as e:
            logger.debug(f"Solution validation error: {e}")
            ctx.checks[self.check_name] = False
            return False

    def _extract_assignment(self, ctx: ValidationContext, problem: Any) -> list[bool] | None:
        """Extract boolean assignment from completion tokens."""
        try:
            # Decode completion
            tokens = ctx.commit.get("tokens", [])
            rollout = ctx.commit.get("rollout", {})
            prompt_len = int(rollout.get("prompt_length", 0))
            completion_len = int(rollout.get("completion_length", 0))

            if completion_len > 0:
                completion_ids = tokens[prompt_len : prompt_len + completion_len]
            else:
                completion_ids = tokens[prompt_len:]

            if not completion_ids:
                return None

            text = ctx.tokenizer.decode(completion_ids, skip_special_tokens=False)

            # Parse with SATParser
            parser = SATParser()
            parsed = parser.parse(text, problem)
            if not isinstance(parsed, dict):
                return None

            values = parsed.get("assignment", [])
            return [bool(v) for v in values[: problem.num_vars]]

        except Exception as e:
            logger.debug(f"Assignment extraction failed: {e}")
            return None
