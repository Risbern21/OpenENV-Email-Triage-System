MEDIUM_TASK = {
    "task_id": "task_medium",
    "difficulty": "medium",
    "description": (
        "Perform email classification, intent detection, and generate an appropriate reply. "
        "Emails contain moderate complexity and require understanding of user intent."
    ),
    "data_corpus": [
        {
            "id": "medium_001",
            "email_text": "Subject: Card Activation Issue\n\nHi,\nI recently received my card but I am unable to activate it. Could you please help me with the process?\n\nThanks.",
            "expected_classification": "support",
            "expected_intent": "card_activation_issue",
            "expected_reply": "Guide the user through the card activation steps and ask for any required details.",
        },
        {
            "id": "medium_002",
            "email_text": "Subject: Unable to Activate Card\n\nHello,\nI tried activating my card multiple times but it doesn't seem to work. Please assist.\n\nRegards.",
            "expected_classification": "support",
            "expected_intent": "card_activation_issue",
            "expected_reply": "Ask for error details and guide the user through activation troubleshooting.",
        },
        {
            "id": "medium_003",
            "email_text": "Subject: Card Usage Help\n\nHi,\nI just received my new card and would like to start using it. Could you tell me how to activate it?\n\nThanks.",
            "expected_classification": "inquiry",
            "expected_intent": "card_activation_request",
            "expected_reply": "Provide steps to activate the card.",
        },
        {
            "id": "medium_004",
            "email_text": "Subject: Card Verification\n\nHello,\nCan you please guide me on how to verify and activate my new card?\n\nRegards.",
            "expected_classification": "inquiry",
            "expected_intent": "card_verification",
            "expected_reply": "Explain verification and activation process clearly.",
        },
        {
            "id": "medium_005",
            "email_text": "Subject: Activation Failure\n\nHi,\nI tried activating my card but it didn’t work. Not sure what went wrong. Can you help?\n\nThanks.",
            "expected_classification": "support",
            "expected_intent": "activation_failure",
            "expected_reply": "Ask for details and guide troubleshooting steps.",
        },
        {
            "id": "medium_006",
            "email_text": "Subject: Need Help Activating Card\n\nHello,\nHow do I activate my card? I’m not sure about the process.\n\nThanks.",
            "expected_classification": "inquiry",
            "expected_intent": "card_activation_request",
            "expected_reply": "Provide clear activation instructions.",
        },
        {
            "id": "medium_007",
            "email_text": "Subject: Assistance Required\n\nHi,\nCan someone assist me with activating my new card?\n\nRegards.",
            "expected_classification": "support",
            "expected_intent": "card_activation_request",
            "expected_reply": "Offer assistance and provide activation steps.",
        },
        {
            "id": "medium_008",
            "email_text": "Subject: Card Not Activated\n\nHello,\nMy card has not been activated yet. What should I do?\n\nThanks.",
            "expected_classification": "support",
            "expected_intent": "card_not_activated",
            "expected_reply": "Guide user on activation steps and verification.",
        },
        {
            "id": "medium_009",
            "email_text": "Subject: Activation Issue\n\nHi,\nI was unable to activate my card. Please help me resolve this.\n\nRegards.",
            "expected_classification": "support",
            "expected_intent": "activation_failure",
            "expected_reply": "Ask for details and provide troubleshooting.",
        },
        {
            "id": "medium_010",
            "email_text": "Subject: Card Status\n\nHello,\nIs my card already activated or do I need to activate it? Please guide me.\n\nThanks.",
            "expected_classification": "inquiry",
            "expected_intent": "activation_status",
            "expected_reply": "Explain how to check activation status and next steps.",
        },
        {
            "id": "medium_011",
            "email_text": "Subject: New Card Activation\n\nHi,\nI just received my new card. Can you help me activate it?\n\nThanks.",
            "expected_classification": "support",
            "expected_intent": "card_activation_request",
            "expected_reply": "Provide activation steps.",
        },
        {
            "id": "medium_012",
            "email_text": "Subject: ID Requirement for Activation\n\nHello,\nDo I need any ID proof to activate my new card?\n\nRegards.",
            "expected_classification": "inquiry",
            "expected_intent": "activation_requirements",
            "expected_reply": "Explain ID requirements if any.",
        },
    ],
}

