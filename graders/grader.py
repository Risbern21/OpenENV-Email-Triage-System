import math
import random
from typing import Callable, Dict, List

import numpy as np

from environment import EmailTriageEnvironment
from models import EmailTriageAction

# ---------------------------------------------------------------------------
# Constants derived directly from your environment's reward structure:
#   classification correct  → +0.3
#   intent correct          → +0.3
#   reply correct           → +0.4  (substring match)
#   reply acceptable        → +0.2  (fallback, no substring match)
#   max per episode         → 1.0   (perfect triage, all 3 stages correct)
# ---------------------------------------------------------------------------

VALID_CLASSIFICATIONS = ["important", "support", "spam"]
VALID_INTENTS = ["pricing inquiry", "complaint", "booking", "promotion"]
RANDOM_REPLY = "xyz"  # guaranteed to never substring-match any true reply


def compute_analytical_ceiling(n_episodes: int) -> float:
    """
    Theoretical maximum reward across n_episodes.

    Perfect agent: classification (+0.3) + intent (+0.3) + reply (+0.4) = 1.0
    plus a +0.2 bonus headroom per episode (mirrors reference grader's
    frequency-control bonus).

    ceiling = n_episodes * 1.2

    NOTE: The +0.2 bonus is structurally inaccessible on ambiguous emails where
    a substring reply match is unlikely. The effective ceiling on such batches
    is closer to n_episodes * 1.0. Scores stay comparable across agents on the
    same email batch — the ceiling just compresses the achievable range.
    """
    return n_episodes * 1.2


# ---------------------------------------------------------------------------
# Score clamping — identical to reference grader.
# Validator requires scores strictly in the open interval (0, 1).
# ---------------------------------------------------------------------------

_SCORE_EPSILON = 0.02
_SCORE_MIN = _SCORE_EPSILON  # 0.02
_SCORE_MAX = 1.0 - _SCORE_EPSILON  # 0.98


def _safe_float(x: float) -> float:
    """Plain Python float; NaN/Inf → 0.5 (safe midpoint inside (0,1))."""
    v = float(x)
    return 0.5 if not math.isfinite(v) else v


def _clamp_score(score: float) -> float:
    """
    Clamp to open interval (0, 1) as a plain Python float.
    Truncates (not rounds) to 4 decimal places so downstream
    rounding can never push the value to exactly 0.0 or 1.0.
    """
    score = _safe_float(score)
    score = max(_SCORE_MIN, min(_SCORE_MAX, score))
    score = math.floor(score * 10000) / 10000
    score = max(_SCORE_MIN, min(_SCORE_MAX, score))
    return score


# ---------------------------------------------------------------------------
# Shared normalizer — used by both grade() and RobustnessGrader
# ---------------------------------------------------------------------------


def normalize_score(
    cumulative_reward: float,
    reward_floor: float,
    reward_ceiling: float,
    zero_miss_rate: float = 1.0,
) -> float:
    """
    Map raw cumulative reward → open interval (0, 1).

    Args:
        cumulative_reward : total reward across all evaluated episodes
        reward_floor      : empirical worst-case (random policy, seeded RNG)
        reward_ceiling    : analytical upper bound (perfect triage every step)
        zero_miss_rate    : fraction of episodes where classification was correct
                            adds up to +10% bonus (mirrors N-1 survival bonus)

    Scores are clamped to [0.02, 0.98] — never exactly 0 or 1.
    """
    raw_range = _safe_float(reward_ceiling) - _safe_float(reward_floor)
    if raw_range < 1.0:
        raw_range = 1.0  # guard against near-zero division

    normalized = (
        _safe_float(cumulative_reward) - _safe_float(reward_floor)
    ) / raw_range

    # Zero-miss bonus: reward episodes where classification was never wrong
    # Analogous to N-1 survival bonus in the reference grader
    zero_miss_bonus = float(zero_miss_rate) * 0.1
    score = normalized + zero_miss_bonus

    return _clamp_score(score)


# ---------------------------------------------------------------------------
# Floor policy — deliberately bad, seeded for reproducibility
# ---------------------------------------------------------------------------


def _random_triage_policy(obs, rng: np.random.Generator) -> EmailTriageAction:
    """
    Worst-case baseline: random answers at every stage.

    - classification : random pick from valid labels
    - intent         : random pick from valid intents
    - reply          : fixed nonsense string (never substring-matches any true reply)

    Uses an explicit seeded RNG (not global random) so floor estimates are
    identical across process lifetimes — mirrors _random_thrash_policy.
    """
    stage = obs.current_stage

    if stage == "classification":
        content = str(rng.choice(VALID_CLASSIFICATIONS))
    elif stage == "intent":
        content = str(rng.choice(VALID_INTENTS))
    else:
        content = RANDOM_REPLY  # reply stage — guaranteed wrong

    return EmailTriageAction(action_type=stage, content=content)


# ---------------------------------------------------------------------------
# RobustnessGrader — mirrors the reference class exactly
# ---------------------------------------------------------------------------


class RobustnessGrader:
    """
    Evaluates an email triage policy using the same pipeline as the
    reference OpenGrid RobustnessGrader.

    Scoring pipeline:
      1. Floor   — empirical: random policy, n_samples=10, seeded RNG
                   result = mean − std (conservatively low)
      2. Ceiling — analytical: n_episodes * 1.2
      3. Score   — normalize(avg_reward, floor, ceiling) + zero-miss bonus (≤10%)
      4. Clamp   — final score always in (0.02, 0.98)
    """

    def __init__(self):
        self.reward_floor = None
        self.reward_ceiling = None

    # ------------------------------------------------------------------
    # Bound estimation
    # ------------------------------------------------------------------

    def _estimate_bounds(self, task_id: str = "task_easy", n_samples: int = 10):
        """
        Compute floor (empirical) and ceiling (analytical).

        Floor  = mean − std over n_samples random-policy episodes.
                 Seeded RNG + varied env seeds → stable, reproducible.
        Ceiling = analytical upper bound.
        """
        thrash_rng = np.random.default_rng(seed=12345)
        floors: List[float] = []

        for i in range(n_samples):
            # Vary seed per episode so floor reflects email variety,
            # not just n_samples random actions on one fixed email.
            env = EmailTriageEnvironment()
            env.task_id = task_id
            obs = env.reset(seed=42 + i)
            ep_reward = 0.0
            done = False

            while not done:
                action = _random_triage_policy(obs, rng=thrash_rng)
                obs = env.step(action)
                ep_reward += obs.reward
                done = obs.done

            floors.append(ep_reward)

        self.reward_floor = float(np.mean(floors) - np.std(floors))

        # Ceiling: scale to n_samples episodes for a fair comparison
        self.reward_ceiling = compute_analytical_ceiling(n_samples)

        # Ensure minimum spread so normalization is meaningful
        if self.reward_ceiling - self.reward_floor < 1.0:
            self.reward_ceiling = self.reward_floor + 5.0

    def get_bounds(self) -> Dict[str, float]:
        """Return floor and ceiling, computing them on first call."""
        if self.reward_floor is None:
            self._estimate_bounds()
        return {
            "reward_floor": round(self.reward_floor, 4),
            "reward_ceiling": round(self.reward_ceiling, 4),
        }

    # ------------------------------------------------------------------
    # Policy evaluation
    # ------------------------------------------------------------------

    def evaluate_policy(
        self,
        policy_fn: Callable,
        n_episodes: int = 3,
    ) -> Dict:
        """
        Run policy_fn for n_episodes and return a normalized score dict.

        policy_fn signature:
            policy_fn(obs: EmailTriageObservation) -> EmailTriageAction

        Returns:
            {
                "avg_raw_reward" : float,   # mean cumulative reward per episode
                "zero_miss_rate" : float,   # fraction of episodes: classification correct
                "reward_floor"   : float,
                "reward_ceiling" : float,
                "score"          : float,   # final score in (0.02, 0.98)
            }
        """
        if self.reward_floor is None:
            self._estimate_bounds()

        # Re-compute ceiling to match the actual n_episodes being evaluated
        reward_ceiling = compute_analytical_ceiling(n_episodes)
        if reward_ceiling - self.reward_floor < 1.0:
            reward_ceiling = self.reward_floor + 5.0

        rewards: List[float] = []
        zero_miss_count = 0

        for ep in range(n_episodes):
            env = EmailTriageEnvironment()
            # Seeds are distinct from floor seeds (42+i) to avoid overlap
            obs = env.reset(seed=100 + ep)
            ep_reward = 0.0
            done = False
            classification_correct = False

            while not done:
                prev_stage = obs.current_stage
                action = policy_fn(obs)
                obs = env.step(action)
                ep_reward += obs.reward
                done = obs.done

                # +0.3 is only awarded on correct classification
                if prev_stage == "classification" and obs.reward == 0.3:
                    classification_correct = True

            rewards.append(ep_reward)
            if classification_correct:
                zero_miss_count += 1

        avg_reward = float(np.mean(rewards))
        zero_miss_rate = zero_miss_count / n_episodes

        final_score = normalize_score(
            cumulative_reward=avg_reward,
            reward_floor=self.reward_floor,
            reward_ceiling=reward_ceiling,
            zero_miss_rate=zero_miss_rate,
        )

        return {
            "avg_raw_reward": round(avg_reward, 4),
            "zero_miss_rate": round(zero_miss_rate, 4),
            "reward_floor": round(self.reward_floor, 4),
            "reward_ceiling": round(reward_ceiling, 4),
            "score": final_score,
        }


# ---------------------------------------------------------------------------
# grade() — called directly by the FastAPI /grader endpoint
# Mirrors EnvClient.grade() → GET /grader?session_id=...
# ---------------------------------------------------------------------------


def grade(episode_id: str, task_id: str = "task_easy") -> Dict:
    """
    Grade a completed episode by episode_id.

    Reads cumulative reward directly from global episode state (no re-running),
    then normalizes it to (0.02, 0.98) using the same floor/ceiling/bonus
    pipeline as RobustnessGrader.evaluate_policy().

    This keeps /grader endpoint results consistent with standalone grader runs.

    Args:
        episode_id : the UUID from _episodes / _histories globals in environment.py

    Returns:
        {
            "episode_id"     : str,
            "avg_raw_reward" : float,   # total reward for this single episode
            "zero_miss_rate" : float,   # 1.0 if classification correct, else 0.0
            "reward_floor"   : float,
            "reward_ceiling" : float,
            "score"          : float,   # in (0.02, 0.98)
        }
    """
    from environment import (_episodes,  # access global episode state
                             _histories)

    if episode_id not in _episodes:
        return {
            "error": f"Episode '{episode_id}' not found.",
            "score": _SCORE_MIN,
        }

    state = _episodes[episode_id]
    history = _histories.get(episode_id, [])

    # Recompute reward from history using the exact same rules as env.step()
    total_reward = 0.0
    classification_correct = False

    for entry in history:
        stage = entry["stage"]
        output = entry["output"]

        if stage == "classification":
            if output == state.true_classification:
                total_reward += 0.3
                classification_correct = True
            # else: +0.0

        elif stage == "intent":
            if output == state.true_intent:
                total_reward += 0.3
            # else: +0.0

        elif stage == "reply":
            if output.lower() in state.true_reply.lower():
                total_reward += 0.4
            else:
                total_reward += 0.2  # acceptable reply fallback

    # Bounds: single episode scale
    reward_ceiling = compute_analytical_ceiling(n_episodes=1)  # 1.2

    # Floor: empirical estimate (10 random-policy episodes, seeded)
    # Averaged to single-episode scale by dividing by n_samples
    _grader = RobustnessGrader()
    _grader._estimate_bounds(task_id, n_samples=10)
    reward_floor = _grader.reward_floor / 10.0  # rescale: floor was over 10 eps

    zero_miss_rate = 1.0 if classification_correct else 0.0

    score = normalize_score(
        cumulative_reward=total_reward,
        reward_floor=reward_floor,
        reward_ceiling=reward_ceiling,
        zero_miss_rate=zero_miss_rate,
    )

    return {
        "episode_id": episode_id,
        "task_id": task_id,
        "avg_raw_reward": round(total_reward, 4),
        "zero_miss_rate": round(zero_miss_rate, 4),
        "reward_floor": round(reward_floor, 4),
        "reward_ceiling": round(reward_ceiling, 4),
        "score": score,
    }
