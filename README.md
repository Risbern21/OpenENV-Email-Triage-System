---
title: OpenENV-Email-Triage-System
emoji: ♿
colorFrom: yellow
colorTo: green
sdk: docker
app_file: server/app.py
pinned: false
---

# Email Triage Environment (OpenEnv)

## Overview

This project implements a real-world email triage environment using the OpenEnv framework.

The agent is required to:
1. Classify incoming emails
2. Extract user intent
3. Generate an appropriate reply

---

## Motivation
 
Email triage is a practical, multi-step reasoning task that requires an agent to:
- Understand natural language in varied email contexts
- Make structured decisions across sequential stages
- Generate contextually appropriate responses
 
This environment provides a reproducible, sandboxed benchmark for evaluating language model agents on classification, intent recognition, and reply generation — three core skills in real-world business automation.
 
---

## Tasks

### 1. Classification (Easy)
Classify the email into one of:
- spam
- important
- support

### 2. Intent Extraction (Medium)
Identify the user's intent:
- pricing inquiry
- complaint
- booking
- general question

### 3. Reply Generation (Hard)
Generate a suitable response to the email.

---

## Environment Design

### Action Space

EmailTriageAction:
- action_type: "classification" | "intent" | "reply"
- content: string

---

### Observation Space

EmailTriageObservation:
- email_text: str
- current_stage: str
- history: list
- reward: float
- done: bool

---

### State

EmailTriageState:
- email_text
- true_classification
- true_intent
- true_reply
- current_stage

---

### Reward System

- Classification → 0.3  
- Intent → 0.3  
- Reply → 0.4  

Partial credit is given for acceptable replies.

---

## Running the Environment

### 1. Create and activate a virtual environment

```bash
python3.14 -m venv <venv-name>

source <venv-name>/bin/activate
```

### 2. Install dependencies

```bash
pip install fastapi uvicorn openenv-core
```

### 3. Run the environment
```bash
uvicorn server.app:main --host 0.0.0.0 --port 7860
```
---
## Running inference and validation
```bash
python inference.py #inference

./validate.sh https://risbern2121-openenv-email-triage-system.hf.space . #validate
```
