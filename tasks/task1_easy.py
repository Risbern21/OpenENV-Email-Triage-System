EASY_TASK = {
    "task_id": "task_easy",
    "difficulty": "easy",
    "description": (
        "Perform email classification, intent detection, and generate an appropriate reply. "
        "Emails are simple and have clear intent (spam, inquiry, or support)."
    ),
    "data_corpus": [
        {
            "id": "easy_001",
            "email_text": "Want 2 get laid tonight? Want real Dogging locations sent direct 2 ur mob? Join the UK's largest Dogging Network bt Txting GRAVEL to 69888!",
            "expected_classification": "spam",
            "expected_intent": "adult_scam",
            "expected_reply": "Ignore this message and mark it as spam.",
        },
        {
            "id": "easy_002",
            "email_text": "You are a winner! You have been specially selected to receive £1000 or a 4* holiday. Call now to claim.",
            "expected_classification": "spam",
            "expected_intent": "prize_scam",
            "expected_reply": "Do not respond. Mark this message as spam.",
        },
        {
            "id": "easy_003",
            "email_text": "URGENT! Your mobile number has won a £2000 prize. Call now to claim.",
            "expected_classification": "spam",
            "expected_intent": "prize_scam",
            "expected_reply": "Ignore and report this as spam.",
        },
        {
            "id": "easy_004",
            "email_text": "Free entry into our £250 weekly competition. Just text WIN to 80086 NOW.",
            "expected_classification": "spam",
            "expected_intent": "promotion_scam",
            "expected_reply": "Do not engage. Mark as spam.",
        },
        {
            "id": "easy_005",
            "email_text": "You have been specially selected to receive a £2000 award. Call before the lines close.",
            "expected_classification": "spam",
            "expected_intent": "prize_scam",
            "expected_reply": "Ignore this message and mark it as spam.",
        },
        {
            "id": "easy_006",
            "email_text": "Congratulations! Nokia phone or £500 prize waiting. Text COLLECT now!",
            "expected_classification": "spam",
            "expected_intent": "prize_scam",
            "expected_reply": "Do not respond. Mark this message as spam.",
        },
        {
            "id": "easy_007",
            "email_text": "FreeMsg: Hey there, I’m a local girl looking for fun. Reply to chat.",
            "expected_classification": "spam",
            "expected_intent": "adult_scam",
            "expected_reply": "Ignore and block this sender.",
        },
        {
            "id": "easy_008",
            "email_text": "Your account has 800 unredeemed points. Call now to claim your reward.",
            "expected_classification": "spam",
            "expected_intent": "phishing",
            "expected_reply": "Do not call. Mark as spam.",
        },
        {
            "id": "easy_009",
            "email_text": "You have an important customer service announcement. Call FREEPHONE now.",
            "expected_classification": "spam",
            "expected_intent": "phishing",
            "expected_reply": "Ignore and mark as spam.",
        },
        {
            "id": "easy_010",
            "email_text": "Win £200 this week in our quiz. Text PLAY now!",
            "expected_classification": "spam",
            "expected_intent": "promotion_scam",
            "expected_reply": "Do not engage. Mark as spam.",
        },
        {
            "id": "easy_011",
            "email_text": "Your mobile number has won £5000. Call now to claim.",
            "expected_classification": "spam",
            "expected_intent": "prize_scam",
            "expected_reply": "Ignore this message and mark it as spam.",
        },
        {
            "id": "easy_012",
            "email_text": "URGENT! We are trying to contact you. You have won a £900 prize. Call now.",
            "expected_classification": "spam",
            "expected_intent": "prize_scam",
            "expected_reply": "Do not respond. Report as spam.",
        },
    ],
}

