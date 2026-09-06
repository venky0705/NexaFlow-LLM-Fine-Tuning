from pathlib import Path
import re

import unsloth
import torch
from unsloth import FastLanguageModel


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sft_merged"
)


# ============================================================
# MODEL CONFIG
# ============================================================

MAX_SEQ_LENGTH = 1024
MAX_NEW_TOKENS = 220
LOAD_IN_4BIT = True


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are the NexaFlow Technologies company assistant.

Answer questions using supported NexaFlow Technologies information
and documented company policy.

Behavior requirements:

- Give clear, grounded, professional answers.
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
- Distinguish NexaFlow's official company location from Venky's
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
- For unrelated questions, explain that they are outside the scope
  of the NexaFlow company assistant.

Answer-style requirements:

- For simple factual questions, answer briefly and directly.
- If the user asks to explain, describe, summarize, give details,
  asks about rules, or asks about a broad policy, provide a more
  detailed structured answer.
- For broad policy explanations, use concise bullet points when
  multiple supported rules are relevant.
- Do not add unsupported details only to make an answer longer.
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

    config_file = MODEL_PATH / "config.json"

    if not config_file.exists():
        raise FileNotFoundError(
            f"Missing config.json:\n{config_file}"
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
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    FastLanguageModel.for_inference(model)

    model.eval()

    print("NexaFlow SFT model loaded successfully.")

    return model, tokenizer


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    text = str(text).strip().lower()

    text = re.sub(
        r"[!?.,;:]+$",
        "",
        text,
    )

    text = " ".join(text.split())

    return text


# ============================================================
# SMALL TALK
# ============================================================

def get_simple_response(question):

    normalized = normalize_text(question)

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

    if normalized in {
        "how are you",
        "how are you doing",
        "how are you today",
        "how's it going",
        "hows it going",
    }:
        return (
            "I'm doing well and ready to help. "
            "What would you like to know about NexaFlow?"
        )

    if normalized in {
        "who are you",
        "what are you",
        "what is this assistant",
        "who is this",
    }:
        return (
            "I'm the NexaFlow AI Assistant. "
            "I help with documented NexaFlow company information "
            "and internal policy questions."
        )

    if normalized in {
        "what is your name",
        "what's your name",
        "whats your name",
        "your name",
        "tell me your name",
    }:
        return "I'm the NexaFlow AI Assistant."

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

    if normalized in {
        "introduce yourself",
        "tell me about yourself",
    }:
        return (
            "I'm the NexaFlow AI Assistant, a policy-aware internal "
            "assistant designed to answer questions about documented "
            "NexaFlow Technologies company information and policies."
        )

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

    if normalized in {
        "good",
        "great",
        "nice",
        "awesome",
        "cool",
        "perfect",
        "okay",
        "ok",
        "got it",
        "sounds good",
    }:
        return (
            "Glad to hear that. "
            "What would you like to know about NexaFlow?"
        )

    return None


# ============================================================
# DETECT DETAILED REQUEST
# ============================================================

def wants_detailed_answer(question):

    text = normalize_text(question)

    detailed_markers = [
        "explain",
        "describe",
        "summarize",
        "give details",
        "in detail",
        "tell me the rules",
        "what are the rules",
        "what rules",
        "policy",
        "policies",
        "rules",
        "new joinee",
        "new joiner",
        "new employee",
        "what should i know",
        "what do i need to know",
        "walk me through",
    ]

    return any(
        marker in text
        for marker in detailed_markers
    )


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(question):

    question = str(question).strip()

    if wants_detailed_answer(question):

        detail_instruction = (
            "\n\nFor this question, provide a structured explanation "
            "with multiple relevant supported points. Use concise bullet "
            "points when appropriate. Do not invent missing rules."
        )

    else:

        detail_instruction = (
            "\n\nFor this question, answer directly and concisely."
        )

    return (
        "### Instruction:\n"
        f"{SYSTEM_INSTRUCTION}"
        f"{detail_instruction}\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Response:\n"
    )


# ============================================================
# CLEAN RESPONSE
# ============================================================

def clean_response(text):

    text = str(text).strip()

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
                .split(marker, 1)[0]
                .strip()
            )

    return text


# ============================================================
# GENERATE RESPONSE
#
# UI can keep full history, but only latest user question
# goes to the model.
# ============================================================

@torch.inference_mode()
def generate_response(
    model,
    tokenizer,
    messages,
):

    if not messages:
        return "How can I help with NexaFlow?"

    current_question = (
        messages[-1]
        .get("content", "")
        .strip()
    )

    if not current_question:
        return "Please enter a question."

    # --------------------------------------------------------
    # SMALL TALK FIRST
    # --------------------------------------------------------

    simple_response = get_simple_response(
        current_question
    )

    if simple_response is not None:
        return simple_response

    # --------------------------------------------------------
    # MODEL PROMPT
    # --------------------------------------------------------

    prompt = build_prompt(
        current_question
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    prompt_length = (
        inputs["input_ids"].shape[1]
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    generated_tokens = outputs[
        0,
        prompt_length:
    ]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    response = clean_response(
        response
    )

    if not response:
        return (
            "I wasn't able to generate a response. "
            "Please try rephrasing your NexaFlow question. "
        )

    return response