import random
from environment import EmailTriageEnvironment
from models import EmailTriageAction, EmailTriageObservation


class RandomEmailPolicy:
    """Random decisions for all stages"""
    name = "Random Policy"

    classifications = ["spam", "important", "support"]
    intents = ["pricing inquiry", "complaint", "booking", "general question"]

    def select_action(self, obs: EmailTriageObservation) -> EmailTriageAction:
        stage = obs.current_stage

        if stage == "classification":
            return EmailTriageAction(
                action_type="classification",
                content=random.choice(self.classifications)
            )

        elif stage == "intent":
            return EmailTriageAction(
                action_type="intent",
                content=random.choice(self.intents)
            )

        elif stage == "reply":
            return EmailTriageAction(
                action_type="reply",
                content="Thank you for reaching out. We will get back to you soon."
            )
        


class RuleBasedEmailPolicy:
    name = "Rule-Based Policy"

    def select_action(self, obs: EmailTriageObservation) -> EmailTriageAction:
        email = obs.email_text.lower()
        stage = obs.current_stage

        # --- Classification ---
        if stage == "classification":
            if any(word in email for word in ["free", "click", "offer", "win"]):
                return EmailTriageAction(action_type="classification", content="spam")
            elif any(word in email for word in ["not working", "issue", "problem", "help", "delay"]):
                return EmailTriageAction(action_type="classification", content="support")
            else:
                return EmailTriageAction(action_type="classification", content="important")

        # --- Intent ---
        elif stage == "intent":
            if any(word in email for word in ["price", "cost", "emi", "subsidy"]):
                return EmailTriageAction(action_type="intent", content="pricing inquiry")
            elif any(word in email for word in ["not working", "problem", "issue", "delay"]):
                return EmailTriageAction(action_type="intent", content="complaint")
            elif any(word in email for word in ["book", "schedule", "demo", "visit"]):
                return EmailTriageAction(action_type="intent", content="booking")
            else:
                return EmailTriageAction(action_type="intent", content="general question")

        # --- Reply ---
        elif stage == "reply":
            if any(word in email for word in ["price", "cost", "emi", "subsidy"]):
                return EmailTriageAction(
                    action_type="reply",
                    content="We will share pricing and subsidy details shortly."
                )
            elif any(word in email for word in ["not working", "issue", "problem"]):
                return EmailTriageAction(
                    action_type="reply",
                    content="We’re sorry for the issue. Our support team will contact you soon."
                )
            elif any(word in email for word in ["book", "schedule", "demo"]):
                return EmailTriageAction(
                    action_type="reply",
                    content="We will schedule your request and confirm shortly."
                )
            else:
                return EmailTriageAction(
                    action_type="reply",
                    content="Thank you for reaching out. We will get back to you soon."
                )


def evaluate(env, policy, episodes=20):
    
    total_score = 0
    total_steps = 0

    for _ in range(episodes):
        obs = env.reset()
        episode_score = 0

        while not obs.done:
            action = policy.select_action(obs)
            obs = env.step(action)

            if obs.reward is not None:
                episode_score += obs.reward   

        total_score += episode_score
        total_steps += env.state.step_count

    avg_score = total_score / episodes
    avg_steps = total_steps / episodes

    return avg_score, avg_steps




env = EmailTriageEnvironment()

for policy in [RandomEmailPolicy(), RuleBasedEmailPolicy()]:
    score, steps = evaluate(env, policy)
    print(f"{policy.name:20s} → Avg Score: {score:.2f}, Avg Steps: {steps}")