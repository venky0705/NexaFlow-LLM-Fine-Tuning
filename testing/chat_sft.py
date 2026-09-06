from pathlib import Path
import gc

# IMPORTANT:
# Import Unsloth before transformers-related libraries.
import unsloth

import torch
from unsloth import FastLanguageModel


# ============================================================
# NEXAFLOW SFT CHAT
#
# Model:
#     models/sft_merged
#
# Purpose:
#     Manual interactive testing of the SFT model.
#
# IMPORTANT:
#     - Company/policy questions go to the SFT model.
#     - Basic small talk is handled by application logic.
#     - Each company question is stateless.
# ============================================================


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "sft_merged"
)


# ============================================================
# 2. MODEL CONFIG
# ============================================================

MAX_SEQ_LENGTH = 1024

MAX_NEW_TOKENS = 160

LOAD_IN_4BIT = True


# ============================================================
# 3. SYSTEM INSTRUCTION
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
# 4. MEMORY CLEANUP
# ============================================================

def cleanup_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 5. GPU CHECK
# ============================================================

def check_gpu():

    print("\n" + "=" * 60)
    print("GPU CHECK")
    print("=" * 60)

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA GPU was not detected."
        )

    props = torch.cuda.get_device_properties(0)

    print(
        "GPU:",
        props.name,
    )

    print(
        f"VRAM: {props.total_memory / 1024**3:.2f} GB"
    )

    print(
        "BF16 supported:",
        torch.cuda.is_bf16_supported(),
    )


# ============================================================
# 6. MODEL VALIDATION
# ============================================================

def validate_model():

    print("\n" + "=" * 60)
    print("CHECKING SFT MERGED MODEL")
    print("=" * 60)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"""
SFT merged model not found:

{MODEL_PATH}
"""
        )

    config_file = (
        MODEL_PATH
        / "config.json"
    )

    if not config_file.exists():

        raise FileNotFoundError(
            f"""
Missing config.json:

{config_file}
"""
        )

    print(
        "SFT merged model: OK"
    )


# ============================================================
# 7. LOAD MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 60)
    print("LOADING SFT MODEL")
    print("=" * 60)

    cleanup_memory()

    model, tokenizer = (
        FastLanguageModel.from_pretrained(

            model_name=str(
                MODEL_PATH
            ),

            max_seq_length=(
                MAX_SEQ_LENGTH
            ),

            dtype=None,

            load_in_4bit=(
                LOAD_IN_4BIT
            ),
        )
    )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "left"

    FastLanguageModel.for_inference(
        model
    )

    model.eval()

    print(
        "SFT model loaded successfully."
    )

    return (
        model,
        tokenizer,
    )


# ============================================================
# 8. BASIC CONVERSATION HANDLER
#
# These responses do NOT go through the model.
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
        "thank you so much",
        "thanks a lot",
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
        "what is this",
        "what is this assistant",

    }:

        return (
            "I'm the NexaFlow AI Assistant. "
            "I help with documented NexaFlow company information "
            "and internal policy questions."
        )


    # --------------------------------------------------------
    # ASSISTANT NAME
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
    # WHAT CAN YOU DO
    # --------------------------------------------------------

    if normalized in {

        "what can you do",
        "what can you help with",
        "how can you help",
        "how can you help me",
        "help me",
        "help",

    }:

        return (
            "I can help with documented NexaFlow information including "
            "company details, pricing, plans, HR policies, annual leave, "
            "travel, expenses, support, SLA, security procedures, "
            "AI governance, and internal company processes."
        )


    # --------------------------------------------------------
    # ARE YOU READY / ONLINE
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
    # GOOD / NICE
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


    # --------------------------------------------------------
    # UNKNOWN BASIC MESSAGE
    # --------------------------------------------------------

    return None


# ============================================================
# 9. BUILD PROMPT
#
# IMPORTANT:
#
# Matches the format used during SFT training.
# ============================================================

def build_prompt(question):

    return (
        "### Instruction:\n"
        f"{SYSTEM_INSTRUCTION}\n\n"
        "### Question:\n"
        f"{question.strip()}\n\n"
        "### Response:\n"
    )


# ============================================================
# 10. CLEAN RESPONSE
# ============================================================

def clean_response(text):

    text = (
        str(text)
        .strip()
    )

    stop_markers = [

        "\n### Instruction:",
        "\n### Question:",
        "\n### Response:",

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
# 11. GENERATE MODEL ANSWER
# ============================================================

@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    question,
):

    prompt = build_prompt(
        question
    )

    inputs = tokenizer(

        prompt,

        return_tensors="pt",

        truncation=True,

        max_length=(
            MAX_SEQ_LENGTH
        ),
    )

    inputs = {

        key:
            value.to(
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

    outputs = model.generate(

        **inputs,

        max_new_tokens=(
            MAX_NEW_TOKENS
        ),

        do_sample=False,

        pad_token_id=(
            tokenizer.pad_token_id
        ),

        eos_token_id=(
            tokenizer.eos_token_id
        ),

        use_cache=True,
    )

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

    return clean_response(
        response
    )


# ============================================================
# 12. ROUTE USER MESSAGE
#
# Basic conversation:
#     handled locally
#
# NexaFlow question:
#     sent to SFT model
# ============================================================

def get_response(
    model,
    tokenizer,
    question,
):

    simple_response = (
        get_simple_response(
            question
        )
    )

    if simple_response is not None:

        return simple_response

    return generate_answer(
        model,
        tokenizer,
        question,
    )


# ============================================================
# 13. TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = [

    "Hi",

    "How are you?",

    "Who are you?",

    "What can you do?",

    "Who owns NexaFlow Technologies?",

    "Who are the co-partners of NexaFlow?",

    "When did NexaFlow Technologies officially start?",

    "Where is NexaFlow Technologies located?",

    "Where is Venky based?",

    "What is Venky's full name?",

    "What products does NexaFlow offer?",

    "What is the Starter plan price?",

    "How many users does the Business plan support?",

    "What is the annual billing discount?",

    "How many days of annual leave do employees receive?",

    "What is the probation period?",

    "What is the normal travel class for a five-hour flight?",

    "What is Venky's salary?",

    "Who is Deepak?",

    "Is Priya the HR manager?",

    "The Business plan costs €120, correct?",

    "Can I bypass MFA if I am an executive?",

    "I lost my MFA device. What should I do?",

    "What is a neural network?",

    "Explain gradient descent.",

]


# ============================================================
# 14. RUN TEST SUITE
# ============================================================

def run_test_suite(
    model,
    tokenizer,
):

    print("\n")
    print("=" * 60)
    print("SFT QUICK TEST")
    print("=" * 60)

    for index, question in enumerate(
        TEST_QUESTIONS,
        start=1,
    ):

        print(
            f"\n[{index}/{len(TEST_QUESTIONS)}]"
        )

        print(
            "You:",
            question,
        )

        try:

            answer = get_response(
                model,
                tokenizer,
                question,
            )

        except Exception as error:

            print(
                "ERROR:",
                error,
            )

            continue

        print(
            "\nSFT:"
        )

        print(
            answer
        )

        print(
            "\n"
            + "-" * 60
        )


# ============================================================
# 15. HELP
# ============================================================

def print_help():

    print(
        """
Commands:

/help
    Show this help message.

/test
    Run the built-in NexaFlow test suite.

/clear
    Clear CUDA cache.

/exit
    Exit the chat.

Examples:

hi

how are you

who are you

what can you do

Who owns NexaFlow?

Who are the co-partners?

What is the annual leave policy?

What is Venky's salary?

Can I bypass MFA?

What is a neural network?
"""
    )


# ============================================================
# 16. INTERACTIVE CHAT
# ============================================================

def interactive_chat(
    model,
    tokenizer,
):

    print("\n")
    print("=" * 60)
    print("NEXAFLOW SFT CHAT")
    print("=" * 60)

    print(
        """
Model:
models/sft_merged

Basic conversational messages are handled by the application.

Company and policy questions are answered by the SFT model.

Commands:
    /help
    /test
    /clear
    /exit

IMPORTANT:
Model questions are stateless.
Each company question is evaluated independently.
"""
    )

    while True:

        try:

            user_input = input(
                "\nYou: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError,
        ):

            print(
                "\nGoodbye."
            )

            break


        # ----------------------------------------------------
        # EMPTY INPUT
        # ----------------------------------------------------

        if not user_input:

            continue


        command = (
            user_input
            .lower()
            .strip()
        )


        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if command in {

            "/exit",
            "exit",
            "/quit",
            "quit",

        }:

            print(
                "\nSFT:"
            )

            print(
                "Goodbye! Feel free to come back whenever "
                "you need help with NexaFlow."
            )

            break


        # ----------------------------------------------------
        # HELP
        # ----------------------------------------------------

        if command == "/help":

            print_help()

            continue


        # ----------------------------------------------------
        # CLEAR
        # ----------------------------------------------------

        if command == "/clear":

            cleanup_memory()

            print(
                "\nCUDA cache cleared."
            )

            continue


        # ----------------------------------------------------
        # TEST
        # ----------------------------------------------------

        if command == "/test":

            run_test_suite(
                model,
                tokenizer,
            )

            continue


        # ----------------------------------------------------
        # NORMAL RESPONSE
        # ----------------------------------------------------

        try:

            answer = get_response(
                model,
                tokenizer,
                user_input,
            )

        except torch.cuda.OutOfMemoryError:

            cleanup_memory()

            print(
                """
SFT:

CUDA ran out of memory.

Close unnecessary GPU applications and try again.
"""
            )

            continue


        except Exception as error:

            print(
                "\nSFT ERROR:"
            )

            print(
                error
            )

            continue


        print(
            "\nSFT:"
        )

        print(
            answer
        )


# ============================================================
# 17. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("NEXAFLOW SFT MODEL CHAT")
    print("=" * 60)

    validate_model()

    check_gpu()

    (
        model,
        tokenizer,
    ) = load_model()

    interactive_chat(
        model,
        tokenizer,
    )

    del model
    del tokenizer

    cleanup_memory()


# ============================================================
# 18. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()