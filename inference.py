import asyncio
import math
import os
import re
import textwrap
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

from environment import EmailTriageAction, EmailTriageEnvironment
from graders.grader import grade

load_dotenv()


# ===== ENV CONFIG =====
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "openenv-email-triage"

MAX_STEPS = 8
TEMPERATURE = 0.1
MAX_TOKENS = 100
SUCCESS_THRESHOLD = 0.1

TASKS = ["task1_easy", "task2_medium", "task3_hard"]

# ===== SYSTEM PROMPT =====
SYSTEM_PROMPT = textwrap.dedent("""
You are an Email Triage AI agent.

Follow rules carefully.
Return ONLY required format.
Do not include explanations.

Stages:
classification → intent → reply

Output EXACTLY:
action_type:<type>;content:<value>
""").strip()


# ===== LOGGING =====
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error=None):
    done_val = str(done).lower()
    error_val = str(error) if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list):
    clamped = clamp_score(score)
    success_val = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_val} steps={steps} score={clamped:.2f} rewards={rewards_str}",
        flush=True,
    )


# ===== PROMPT BUILDER =====
def build_prompt(obs):
    history_text = (
        "\n".join([f"{h['stage']} -> {h['output']}" for h in obs.history])
        if obs.history
        else "None"
    )

    return f"""
Email: {obs.email_text}

Current Stage: {obs.current_stage}

IMPORTANT:
- Only act for CURRENT STAGE
- Do NOT repeat previous stages
- Use keyword matching (simple logic)

History:
{history_text}

Output EXACTLY:
action_type:<type>;content:<value>
"""


def parse_action(text: str):
    try:
        match = re.search(
            r"action_type\s*:\s*(\w+)\s*;\s*content\s*:\s*(.+)",
            text.strip(),
        )
        if match:
            return match.group(1).strip(), match.group(2).strip()
    except:
        pass

    return "classification", "important"


def clamp_score(s: float) -> float:
    """Ensure score is strictly in (0, 1). Mirrors grader._clamp_score."""
    try:
        s = float(s)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(s):
        return 0.5
    s = max(0.02, min(0.98, s))
    s = math.floor(s * 10000) / 10000
    return max(0.02, min(0.98, s))


def run_task(client: OpenAI, env: EmailTriageEnvironment, task_id: str):
    """Run one task and return results."""
    history_msgs = []
    rewards = []
    steps_taken = 0
    score = 0.05
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            # ===== CALL LLM =====
            user_prompt = build_prompt(obs)

            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )

            response_text = completion.choices[0].message.content.strip()

            action_type, content = parse_action(response_text)

            # ===== ENFORCE STAGE =====
            if action_type != obs.current_stage:
                action_type = obs.current_stage

            action = EmailTriageAction(action_type=action_type, content=content)

            try:
                obs = env.step(action)
                reward = obs.reward or 0.0
                done = obs.done
                error = None
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            steps_taken = step

        grade_result = grade(task_id)
        score = clamp_score(grade_result.get("score", 0.5))
        success = score >= SUCCESS_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
        score = 0.05
        success = False

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_id, "score": score, "steps": steps_taken, "success": success}


# ===== MAIN =====
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = EmailTriageEnvironment()

    all_results = []

    try:
        for task_id in TASKS:
            print(f"\n{'='*60}", flush=True)
            print(f"Running task: {task_id}", flush=True)
            print(f"{'='*60}", flush=True)

            result = run_task(client, env, task_id)
            all_results.append(result)

    finally:
        env.close()

    # Summary
    print(f"\n{'='*60}", flush=True)
    print("FINAL RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    for r in all_results:
        status = "✓ PASS" if r["success"] else "✗ FAIL"
        print(
            f"  {r['task']}: score={r['score']:.4f}  steps={r['steps']}  [{status}]",
            flush=True,
        )

    avg_score = (
        sum(r["score"] for r in all_results) / len(all_results) if all_results else 0
    )
    print(f"\n  Average Score: {avg_score:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
