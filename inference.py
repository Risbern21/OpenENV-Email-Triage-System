import asyncio
import os
import textwrap
import re
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from environment import EmailTriageEnvironment, EmailTriageAction

load_dotenv()



# ===== ENV CONFIG =====
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

TASK_NAME = "email-triage"
BENCHMARK = "openenv-email-triage"

MAX_STEPS = 8
TEMPERATURE = 0.1   
MAX_TOKENS = 100
SUCCESS_THRESHOLD = 0.1

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
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ===== PROMPT BUILDER =====
def build_prompt(obs):
    history_text = "\n".join(
        [f"{h['stage']} -> {h['output']}" for h in obs.history]
    ) if obs.history else "None"

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


# ===== MAIN =====
async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = EmailTriageEnvironment()

    rewards: List[float] = []
    steps_taken = 0

    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)

    try:
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            # ===== CALL LLM =====
            user_prompt = build_prompt(obs)

            try:
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

                

            except Exception as e:
                response_text = "action_type:classification;content:important"

            action_type, content = parse_action(response_text)

            # ===== ENFORCE STAGE =====
            if action_type != obs.current_stage:
                action_type = obs.current_stage

            action = EmailTriageAction(
                action_type=action_type,
                content=content
            )

            # ===== STEP =====
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

            action_str = f"{action_type}:{content}"
            log_step(step, action_str, reward, done, error)

            if done:
                break

        total_reward = sum(rewards)
        success = total_reward >= SUCCESS_THRESHOLD

    finally:
        try:
            env.close()
        except:
            pass

        score = min(max(sum(rewards), 0.0), 1.0)
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())