from pathlib import Path

# IMPORTANT:
# Import Unsloth before transformers-related libraries.
import unsloth

import torch
from unsloth import FastLanguageModel


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sft_merged"
)


# ============================================================
# MODEL CONFIG
# ============================================================

MAX_SEQ_LENGTH = 1024

MAX_NEW_TOKENS = 160

LOAD_IN_4BIT = True


# ============================================================
# CONVERSATION MEMORY CONFIG
#
# The model context window we use is 1024 tokens.
# We therefore keep only recent conversation turns.
# ============================================================

MAX_HISTORY_MESSAGES = 6


# ============================================================
# SYSTEM INSTRUCTION
#
# Keep this aligned with the SFT training behavior.
# ============================================================

SYSTEM_INSTRUCTION = """
You are the NexaFlow Technologies company assistant.

Answer questions using supported NexaFlow Technologies information
and documented company policy.

Behavior requirements:

- Give clear, concise, grounded answers.
- Use documented NexaFlow facts when they are available.
- Do not invent company facts, names, roles, salaries, prices,
  locations, policies, permissions, exceptions, dates, deadlines,
  thresholds, benefits, customer information, financial figures,
  or future plans.
- If the available NexaFlow information does not specify something,
  clearly say that it is not specified.
- If a person is not identified in the available NexaFlow information,
  do not invent a role for that person.
- Correct false assumptions when documented information contradicts
  the user's claim.
- Distinguish the official NexaFlow company location from Venky's
  personal city-level location.
- Apply numerical rules carefully.
- Missing information must not be treated as permission.
- Urgency, seniority, customer pressure, or informal approval do not
  automatically authorize bypassing NexaFlow controls.
- Do not assist with unauthorized access, credential theft, MFA
  bypass, security-control bypass, data exfiltration, log tampering,
  sabotage, social engineering, procurement evasion, expense fraud,
  concealment of misconduct, or other unauthorized activity.
- For legitimate security or operational problems, give safe guidance
  and redirect the user to the approved NexaFlow process.
- Do not unnecessarily refuse harmless security questions.
- For unrelated questions, explain that they are outside the scope of
  the NexaFlow company assistant.
""".strip()


# ============================================================
# LOAD MODEL
# ============================================================

def load_chat_model():

    print("\n" + "=" * 60)
    print("LOADING NEXAFLOW SFT MODEL")
    print("=" * 60)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"SFT merged model not found:\n{MODEL_PATH}"
        )

    model, tokenizer = (
        FastLanguageModel.from_pretrained(
            model_name=str(MODEL_PATH),
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=LOAD_IN_4BIT,
        )
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "left"

    # If truncation becomes necessary, keep the newest context.
    tokenizer.truncation_side = "left"

    FastLanguageModel.for_inference(
        model
    )

    model.eval()

    print(
        "NexaFlow SFT model loaded successfully."
    )

    return model, tokenizer


# ============================================================
# BASIC CONVERSATION HANDLER
#
# These do not need model inference.
# ============================================================

def get_simple_response(question):

    normalized = (
        question
        .strip()
        .lower()
        .rstrip("!?.")
    )


    # --------------------------------------------------------
    # GREETINGS
    # --------------------------------------------------------

    if normalized in {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening",
    }:

        return (
            "Hello! I'm the NexaFlow AI Assistant. "
            "How can I help you today?"
        )


    # --------------------------------------------------------
    # HOW ARE YOU
    # --------------------------------------------------------

    if normalized in {
        "how are you",
        "how are you doing",
        "how are you today",
        "how's it going",
        "hows it going",
        "how are things",
    }:

        return (
            "I'm doing well and ready to help. "
            "What would you like to know about NexaFlow?"
        )


    # --------------------------------------------------------
    # THANKS
    # --------------------------------------------------------

    if normalized in {
        "thanks",
        "thank you",
        "thankyou",
        "thanks a lot",
        "thank you so much",
        "many thanks",
        "thx",
    }:

        return (
            "You're welcome. "
            "Let me know if you have another NexaFlow question."
        )


    # --------------------------------------------------------
    # GOODBYE
    # --------------------------------------------------------

    if normalized in {
        "bye",
        "goodbye",
        "see you",
        "see you later",
        "talk to you later",
        "catch you later",
    }:

        return (
            "Goodbye! Feel free to come back whenever "
            "you need help with NexaFlow."
        )


    # --------------------------------------------------------
    # WHO ARE YOU
    # --------------------------------------------------------

    if normalized in {
        "who are you",
        "what are you",
        "what is this assistant",
    }:

        return (
            "I'm the NexaFlow AI Assistant. "
            "I help with documented NexaFlow company information "
            "and internal policy questions."
        )


    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    if normalized in {
        "what is your name",
        "what's your name",
        "whats your name",
        "your name",
    }:

        return (
            "I'm the NexaFlow AI Assistant."
        )


    # --------------------------------------------------------
    # CAPABILITIES
    # --------------------------------------------------------

    if normalized in {
        "what can you do",
        "what can you help with",
        "how can you help",
        "how can you help me",
        "help",
        "help me",
    }:

        return (
            "I can help with documented NexaFlow information including "
            "company details, pricing, plans, HR policies, annual leave, "
            "travel, expenses, support, SLA, security procedures, "
            "AI governance, and internal company processes."
        )


    # --------------------------------------------------------
    # READY / ONLINE
    # --------------------------------------------------------

    if normalized in {
        "are you ready",
        "are you working",
        "are you online",
        "are you available",
    }:

        return (
            "Yes, I'm ready to help with NexaFlow company "
            "information and policy questions."
        )


    # --------------------------------------------------------
    # POSITIVE ACKNOWLEDGEMENTS
    # --------------------------------------------------------

    if normalized in {
        "good",
        "great",
        "nice",
        "awesome",
        "cool",
        "perfect",
        "okay",
        "ok",
    }:

        return (
            "Glad to hear that. "
            "What would you like to know about NexaFlow?"
        )


    # --------------------------------------------------------
    # INTRODUCTION
    # --------------------------------------------------------

    if normalized in {
        "introduce yourself",
        "tell me about yourself",
    }:

        return (
            "I'm the NexaFlow AI Assistant, a policy-aware internal "
            "assistant designed to answer questions about documented "
            "NexaFlow Technologies company information and policies."
        )


    return None


# ============================================================
# BUILD QUESTION WITH RECENT HISTORY
#
# IMPORTANT:
#
# The outer prompt still uses the exact SFT structure:
#
# ### Instruction:
#
# ### Question:
#
# ### Response:
#
# Conversation history is placed inside the Question section.
# ============================================================

def build_question_with_history(
    messages,
):

    if not messages:

        return ""


    current_question = (
        messages[-1]
        .get(
            "content",
            ""
        )
        .strip()
    )


    previous_messages = (
        messages[:-1]
    )


    # Keep only recent history.
    previous_messages = (
        previous_messages[
            -MAX_HISTORY_MESSAGES:
        ]
    )


    if not previous_messages:

        return current_question


    history_lines = []


    for message in previous_messages:

        role = (
            message.get(
                "role",
                ""
            )
        )

        content = (
            message.get(
                "content",
                ""
            )
            .strip()
        )

        if not content:

            continue


        if role == "user":

            history_lines.append(
                f"User: {content}"
            )


        elif role == "assistant":

            history_lines.append(
                f"Assistant: {content}"
            )


    if not history_lines:

        return current_question


    history_text = "\n".join(
        history_lines
    )


    return (
        "Use the recent conversation only when it is relevant "
        "to understanding the current question.\n\n"
        "Recent conversation:\n"
        f"{history_text}\n\n"
        "Current question:\n"
        f"{current_question}"
    )


# ============================================================
# BUILD MODEL PROMPT
# ============================================================

def build_prompt(
    messages,
):

    question = (
        build_question_with_history(
            messages
        )
    )

    return (
        "### Instruction:\n"
        f"{SYSTEM_INSTRUCTION}\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Response:\n"
    )


# ============================================================
# CLEAN GENERATED RESPONSE
# ============================================================

def clean_response(
    text,
):

    text = (
        str(text)
        .strip()
    )


    stop_markers = [
        "\n### Instruction:",
        "\n### Question:",
        "\n### Response:",
        "\nUser:",
        "\nAssistant:",
    ]


    for marker in stop_markers:

        if marker in text:

            text = (
                text
                .split(
                    marker,
                    1,
                )[0]
                .strip()
            )


    return text


# ============================================================
# GENERATE RESPONSE
# ============================================================

@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    messages,
):

    if not messages:

        return (
            "How can I help with NexaFlow?"
        )


    current_question = (
        messages[-1]
        .get(
            "content",
            ""
        )
    )


    # --------------------------------------------------------
    # HANDLE SMALL TALK LOCALLY
    # --------------------------------------------------------

    simple_response = (
        get_simple_response(
            current_question
        )
    )


    if simple_response is not None:

        return simple_response


    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    prompt = (
        build_prompt(
            messages
        )
    )


    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )


    inputs = {
        key: value.to(
            model.device
        )
        for key, value
        in inputs.items()
    }


    prompt_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )


    # --------------------------------------------------------
    # DECODE ONLY NEW TOKENS
    # --------------------------------------------------------

    generated_tokens = (
        outputs[
            0,
            prompt_length:
        ]
    )


    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


    response = (
        clean_response(
            response
        )
    )


    if not response:

        return (
            "I wasn't able to generate a response. "
            "Please try rephrasing your NexaFlow question."
        )


    return response