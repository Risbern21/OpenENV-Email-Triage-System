---
title: Email Triage System
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_file: server/app.py
pinned: false
tags:
  - openenv
  - ai-agent
  - reinforcement-learning
  - automation
---

# 📧 OpenEnv: Email Triage System

[![OpenEnv Spec Compliance](https://img.shields.io/badge/OpenEnv-1.0-green.svg)](https://github.com/openenv/spec)
[![Platform](https://img.shields.io/badge/Platform-HuggingFace%20Spaces-orange.svg)](https://huggingface.co/spaces)

A complete, real-world OpenEnv environment for training and evaluating AI agents on the complex task of professional email triage.

## 🌟 Overview

The **Email Triage System** simulates the workflow of a digital assistant managing a high-volume inbox. Agents must process incoming emails through a multi-stage pipeline:
1.  **🔍 Classification**: Determine the nature of the email (Spam, Important, Support, etc.).
2.  **🎯 Intent Detection**: Extract the specific user need (Complaint, Pricing, Booking, etc.).
3.  **✍️ Reply Generation**: Draft a contextually accurate and professional response.

This environment provides a structured, reproducible benchmark for evaluating an agent's ability to maintain state, follow business logic, and generate high-quality outputs.

---

## 🚀 Key Features

-   **Full OpenEnv Compliance**: Implements the complete `step()`, `reset()`, and `state()` API.
-   **Typed Pydantic Models**: Strictly enforced schemas for Observations, Actions, and Rewards.
-   **Multi-Stage Trajectory**: Episodes involve sequential decision-making, moving from classification to drafting.
-   **Sophisticated Graders**: Deterministic reward function with partial progress signals and reasoning quality checks.
-   **Baseline Included**: Reproducible inference script using OpenAI-compatible APIs.

---

## 📊 Task & Difficulty Levels

| Task ID | Name | Difficulty | Description |
| :--- | :--- | :--- | :--- |
| `task_easy` | **Email Classification** | Easy | Simple spam vs. ham detection with clear triggers. |
| `task_medium` | **Intent Detection** | Medium | Understanding customer issues like card activation or billing. |
| `task_hard` | **Drafting Reply** | Hard | Generating professional emails that resolve user queries. |

---

## 🛠️ Environment Specification

### Action Space (`EmailTriageAction`)
-   `action_type`: Current stage (`classification`, `intent`, `reply`).
-   `content`: The agent's decision or text output.
-   `reasoning`: String explaining the logic behind the action.
-   `confidence`: Float [0.0 - 1.0] representing agent's certainty.

### Observation Space (`EmailTriageObservation`)
-   `email_text`: The content of the email being triaged.
-   `current_stage`: Which stage the environment is in.
-   `history`: Detailed log of previous actions and their fine-grained scores.
-   `reward`: Scalar reward for the current step.
-   `message`: Natural language feedback from the grader.

---

## 🏆 Reward Function

The environment provides a dense reward signal decomposed into three components:
-   **Label Correctness (60%)**: Accuracy relative to ground truth.
-   **Reasoning Quality (30%)**: Evaluation of the `reasoning` field for keyword relevance and length.
-   **Formatting & Metadata (10%)**: Proper usage of `action_type` and `confidence`.

Difficulty-based ceilings are applied to the final score:
-   **Easy**: max 0.90
-   **Medium**: max 0.80
-   **Hard**: max 0.70

---

## 📦 Installation & Setup

### 1. Prerequisite
Ensure you have Python 3.10+ and `docker` installed.

### 2. Local Setup
```bash
# Clone the repository
git clone https://github.com/Risbern21/OpenENV-Email-Triage-System.git
cd OpenENV-Email-Triage-System

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

### 3. Start the Environment
```bash
# Run via Uvicorn
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 4. Running Validation
```bash
# Run the built-in validator
./validate.sh http://localhost:7860
```

---

## 🤖 Baseline Inference

To run the baseline agent against the environment:
1. Set your `OPENAI_API_KEY` (or `HF_TOKEN`) in `.env`.
2. Run:
```bash
python inference.py
```

---

## 📄 License
MIT License. See [LICENSE](LICENSE) for details.
