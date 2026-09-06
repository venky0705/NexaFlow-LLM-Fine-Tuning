from pathlib import Path
import gc
import json
import math
import random
import shutil
import sys
import time

import numpy as np
import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel


# ============================================================
# NEXAFLOW TECHNOLOGIES
# SUPERVISED FINE-TUNING (SFT)
#
# Starting model:
#
#     models/cpt_merged
#
# Training:
#
#     4-bit QLoRA
#     Fresh SFT LoRA adapter
#
# Data:
#
#     Data/03_sft/train.jsonl
#     Data/03_sft/validation.jsonl
#     Data/03_sft/test.jsonl
#
# Output:
#
#     models/sft_checkpoints/
#     models/sft_adapter/
#     results/sft/
#
# IMPORTANT:
#
# CPT LoRA is NOT reused.
#
# models/cpt_merged already contains:
#
#     Base model + CPT knowledge
#
# We attach a completely FRESH LoRA for SFT.
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 2. STARTING MODEL
# ============================================================

CPT_MERGED_DIR = (
    PROJECT_ROOT
    / "models"
    / "cpt_merged"
)


# ============================================================
# 3. DATA PATHS
# ============================================================

SFT_DATA_DIR = (
    PROJECT_ROOT
    / "Data"
    / "03_sft"
)

TRAIN_FILE = (
    SFT_DATA_DIR
    / "train.jsonl"
)

VALIDATION_FILE = (
    SFT_DATA_DIR
    / "validation.jsonl"
)

TEST_FILE = (
    SFT_DATA_DIR
    / "test.jsonl"
)


# ============================================================
# 4. OUTPUT PATHS
# ============================================================

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

CHECKPOINT_DIR = (
    MODELS_DIR
    / "sft_checkpoints"
)

ADAPTER_DIR = (
    MODELS_DIR
    / "sft_adapter"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "sft"
)

METRICS_FILE = (
    RESULTS_DIR
    / "sft_training_metrics.json"
)


# ============================================================
# 5. MODEL CONFIG
# ============================================================

MAX_SEQ_LENGTH = 1024

LOAD_IN_4BIT = True


# ============================================================
# 6. SFT LORA CONFIG
#
# Fresh adapter.
#
# r = 16
# alpha = 32
#
# Good balance for Qwen2.5-1.5B + 8 GB VRAM.
# ============================================================

LORA_RANK = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.0

LORA_TARGET_MODULES = [

    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",

    "gate_proj",
    "up_proj",
    "down_proj",
]


# ============================================================
# 7. TRAINING CONFIG
# ============================================================

NUM_EPOCHS = 3

TRAIN_BATCH_SIZE = 1

EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.05

MAX_GRAD_NORM = 1.0

OPTIMIZER = "adamw_8bit"

LR_SCHEDULER = "cosine"

SEED = 42


# ============================================================
# 8. LOGGING / SAVING
# ============================================================

LOGGING_STEPS = 10

SAVE_TOTAL_LIMIT = 3


# ============================================================
# 9. RESUME CONFIG
#
# False = fresh SFT run
#
# True = if training crashed, automatically find the latest
#        sft checkpoint and resume.
# ============================================================

AUTO_RESUME = False


# ============================================================
# 10. OLD OUTPUT HANDLING
#
# Keep False normally.
#
# If intentionally restarting SFT completely, either manually
# delete:
#
# models/sft_checkpoints
# models/sft_adapter
# results/sft
#
# OR temporarily set this True.
# ============================================================

CLEAR_OLD_SFT_OUTPUT = False


# ============================================================
# 11. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


set_seed(
    SEED
)


# ============================================================
# 12. MEMORY HELPERS
# ============================================================

def cleanup_memory():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


def gb(value):

    return (
        value
        / (1024 ** 3)
    )


def print_cuda_memory(title):

    if not torch.cuda.is_available():

        return

    allocated = (
        torch.cuda.memory_allocated(0)
    )

    reserved = (
        torch.cuda.memory_reserved(0)
    )

    peak = (
        torch.cuda.max_memory_allocated(0)
    )

    print(
        "\n"
        + "-" * 60
    )

    print(
        title
    )

    print(
        "-" * 60
    )

    print(
        f"Allocated     : "
        f"{gb(allocated):.2f} GB"
    )

    print(
        f"Reserved      : "
        f"{gb(reserved):.2f} GB"
    )

    print(
        f"Peak allocated: "
        f"{gb(peak):.2f} GB"
    )


# ============================================================
# 13. SAFE PERPLEXITY
# ============================================================

def safe_perplexity(loss):

    try:

        return float(
            math.exp(loss)
        )

    except OverflowError:

        return float("inf")


# ============================================================
# 14. CHECK STARTING MODEL
# ============================================================

def validate_starting_model():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CHECKING CPT MERGED MODEL"
    )

    print(
        "=" * 60
    )

    if not CPT_MERGED_DIR.exists():

        raise FileNotFoundError(
            f"""
CPT merged model was not found:

{CPT_MERGED_DIR}

You must run CPT merge before SFT.

Expected:

Qwen2.5-1.5B Base
        +
CPT adapter
        ↓
models/cpt_merged
"""
        )

    config_file = (
        CPT_MERGED_DIR
        / "config.json"
    )

    if not config_file.exists():

        raise FileNotFoundError(
            f"""
config.json was not found in:

{CPT_MERGED_DIR}

The CPT merge may be incomplete.
"""
        )

    weight_files = list(
        CPT_MERGED_DIR.glob(
            "*.safetensors"
        )
    )

    weight_indexes = list(
        CPT_MERGED_DIR.glob(
            "*.index.json"
        )
    )

    if (
        not weight_files
        and not weight_indexes
    ):

        raise RuntimeError(
            f"""
No model weight files were found in:

{CPT_MERGED_DIR}
"""
        )

    print(
        "CPT merged model:",
        CPT_MERGED_DIR,
    )

    print(
        "Model config: OK"
    )

    print(
        "Model weights: OK"
    )


# ============================================================
# 15. CHECK SFT DATA FILES
# ============================================================

def validate_data_files():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CHECKING SFT DATA"
    )

    print(
        "=" * 60
    )

    required_files = [

        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"""
Required SFT data file does not exist:

{path}

Run:

uv run python "Data preparation/prepare_sft_data.py"
"""
            )

        if path.stat().st_size == 0:

            raise RuntimeError(
                f"""
SFT dataset file is empty:

{path}
"""
            )

        print(
            "OK:",
            path
        )


# ============================================================
# 16. PREPARE OUTPUT DIRECTORIES
# ============================================================

def prepare_output_directories():

    if CLEAR_OLD_SFT_OUTPUT:

        print(
            "\nCLEAR_OLD_SFT_OUTPUT=True"
        )

        for path in [

            CHECKPOINT_DIR,
            ADAPTER_DIR,
            RESULTS_DIR,

        ]:

            if path.exists():

                print(
                    "Removing:",
                    path
                )

                shutil.rmtree(
                    path
                )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ADAPTER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 17. GPU CHECK
# ============================================================

def check_hardware():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "HARDWARE CHECK"
    )

    print(
        "=" * 60
    )

    print(
        "Python:",
        sys.version.split()[0],
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    if not torch.cuda.is_available():

        raise RuntimeError(
            """
CUDA GPU was not detected.

This SFT trainer requires CUDA for the configured
4-bit QLoRA run.
"""
        )

    device_index = (
        torch.cuda.current_device()
    )

    properties = (
        torch.cuda.get_device_properties(
            device_index
        )
    )

    print(
        "CUDA device:",
        device_index,
    )

    print(
        "GPU:",
        properties.name,
    )

    print(
        f"VRAM: "
        f"{gb(properties.total_memory):.2f} GB"
    )

    print(
        "BF16 supported:",
        torch.cuda.is_bf16_supported(),
    )


# ============================================================
# 18. LOAD RAW SFT DATA
# ============================================================

def load_sft_dataset():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "LOADING SFT DATASET"
    )

    print(
        "=" * 60
    )

    dataset = load_dataset(

        "json",

        data_files={

            "train":
                str(
                    TRAIN_FILE
                ),

            "validation":
                str(
                    VALIDATION_FILE
                ),

            "test":
                str(
                    TEST_FILE
                ),
        },
    )

    train_dataset = (
        dataset[
            "train"
        ]
    )

    validation_dataset = (
        dataset[
            "validation"
        ]
    )

    test_dataset = (
        dataset[
            "test"
        ]
    )

    print(
        "Train examples:",
        len(
            train_dataset
        ),
    )

    print(
        "Validation examples:",
        len(
            validation_dataset
        ),
    )

    print(
        "Held-out test examples:",
        len(
            test_dataset
        ),
    )

    if len(train_dataset) == 0:

        raise RuntimeError(
            "SFT train dataset is empty."
        )

    if len(validation_dataset) == 0:

        raise RuntimeError(
            "SFT validation dataset is empty."
        )

    if len(test_dataset) == 0:

        raise RuntimeError(
            "SFT test dataset is empty."
        )

    required_columns = {

        "instruction",
        "input",
        "output",
    }

    for split_name, split in [

        (
            "train",
            train_dataset
        ),

        (
            "validation",
            validation_dataset
        ),

        (
            "test",
            test_dataset
        ),

    ]:

        missing_columns = (
            required_columns
            - set(
                split.column_names
            )
        )

        if missing_columns:

            raise RuntimeError(
                f"""
SFT {split_name} dataset is missing columns:

{sorted(missing_columns)}

Columns found:

{split.column_names}
"""
            )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


# ============================================================
# 19. CHECK FOR EMPTY VALUES
# ============================================================

def check_empty_values(
    dataset,
    split_name,
):

    required_columns = [

        "instruction",
        "input",
        "output",
    ]

    for column in required_columns:

        empty_count = sum(

            1

            for value
            in dataset[
                column
            ]

            if (
                value is None
                or not str(value).strip()
            )
        )

        if empty_count > 0:

            raise RuntimeError(
                f"""
Found {empty_count} empty '{column}' values
in SFT {split_name}.
"""
            )

    print(
        f"{split_name} empty-field check: PASSED"
    )


# ============================================================
# 20. LOAD CPT MERGED MODEL IN 4-BIT
#
# The CPT model was saved merged in 16-bit.
#
# For SFT training, load it back in 4-bit.
# ============================================================

def load_model():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "LOADING CPT MERGED MODEL FOR SFT"
    )

    print(
        "=" * 60
    )

    cleanup_memory()

    try:

        model, tokenizer = (
            FastLanguageModel.from_pretrained(

                model_name=str(
                    CPT_MERGED_DIR
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

    except torch.cuda.OutOfMemoryError as error:

        cleanup_memory()

        raise RuntimeError(
            """
CUDA out-of-memory while loading models/cpt_merged.

Try:

1. Stop any old Python training process.
2. Close unnecessary GPU applications.
3. Confirm load_in_4bit=True.
4. Restart the SFT command.

Original CUDA error:

"""
            + str(error)
        ) from error

    if tokenizer.eos_token_id is None:

        raise RuntimeError(
            "Tokenizer does not have an EOS token."
        )

    if tokenizer.pad_token_id is None:

        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = (
        "right"
    )

    print(
        "Model loaded successfully."
    )

    print(
        "EOS token:",
        repr(
            tokenizer.eos_token
        ),
    )

    print(
        "EOS ID:",
        tokenizer.eos_token_id,
    )

    print(
        "PAD token:",
        repr(
            tokenizer.pad_token
        ),
    )

    print_cuda_memory(
        "MEMORY AFTER CPT MODEL LOAD"
    )

    return (
        model,
        tokenizer,
    )


# ============================================================
# 21. FORMAT DATA AS PROMPT + COMPLETION
#
# IMPORTANT:
#
# The SFT model sees:
#
# ### Instruction:
# ...
#
# ### Question:
# ...
#
# ### Response:
#
#
# and we calculate training loss ONLY on:
#
# the correct response.
#
# ============================================================

def prepare_prompt_completion_dataset(
    dataset,
    tokenizer,
):

    eos_token = (
        tokenizer.eos_token
    )

    def convert_example(example):

        instruction = (
            str(
                example[
                    "instruction"
                ]
            ).strip()
        )

        question = (
            str(
                example[
                    "input"
                ]
            ).strip()
        )

        answer = (
            str(
                example[
                    "output"
                ]
            ).strip()
        )

        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Question:\n"
            f"{question}\n\n"
            "### Response:\n"
        )

        completion = (
            answer
            + eos_token
        )

        return {

            "prompt":
                prompt,

            "completion":
                completion,
        }

    formatted_dataset = (
        dataset.map(
            convert_example
        )
    )

    return formatted_dataset


# ============================================================
# 22. VERIFY PROMPT / COMPLETION
# ============================================================

def verify_formatted_dataset(
    dataset,
    split_name,
):

    required = {

        "prompt",
        "completion",
    }

    missing = (
        required
        - set(
            dataset.column_names
        )
    )

    if missing:

        raise RuntimeError(
            f"""
Formatted {split_name} dataset missing:

{sorted(missing)}
"""
        )

    for index in range(
        len(dataset)
    ):

        prompt = (
            dataset[index][
                "prompt"
            ]
        )

        completion = (
            dataset[index][
                "completion"
            ]
        )

        if not prompt.strip():

            raise RuntimeError(
                f"Empty prompt at {split_name}[{index}]"
            )

        if not completion.strip():

            raise RuntimeError(
                f"Empty completion at {split_name}[{index}]"
            )

    print(
        f"{split_name} prompt/completion check: PASSED"
    )


# ============================================================
# 23. CHECK TOKEN LENGTHS
#
# We check prompt + completion together.
# ============================================================

def check_token_lengths(
    dataset,
    tokenizer,
    split_name,
):

    print(
        f"\nChecking token lengths: "
        f"{split_name}"
    )

    max_tokens = 0

    over_limit = []

    total_tokens = 0

    for index in range(
        len(dataset)
    ):

        text = (
            dataset[index][
                "prompt"
            ]
            + dataset[index][
                "completion"
            ]
        )

        token_count = len(

            tokenizer(

                text,

                add_special_tokens=False,

                truncation=False,

            )[
                "input_ids"
            ]
        )

        total_tokens += (
            token_count
        )

        max_tokens = max(
            max_tokens,
            token_count,
        )

        if token_count > MAX_SEQ_LENGTH:

            over_limit.append(
                (
                    index,
                    token_count,
                )
            )

    average_tokens = (
        total_tokens
        / len(dataset)
    )

    print(
        f"Average tokens: "
        f"{average_tokens:.2f}"
    )

    print(
        f"Max tokens: "
        f"{max_tokens}"
    )

    print(
        f"Examples > {MAX_SEQ_LENGTH}: "
        f"{len(over_limit)}"
    )

    if over_limit:

        print(
            "WARNING:"
        )

        print(
            f"{len(over_limit)} examples exceed "
            f"{MAX_SEQ_LENGTH} tokens."
        )

        print(
            "First few:"
        )

        print(
            over_limit[:10]
        )

        print(
            "TRL will truncate these examples."
        )

    return {

        "average_tokens":
            average_tokens,

        "max_tokens":
            max_tokens,

        "over_limit":
            len(
                over_limit
            ),
    }


# ============================================================
# 24. ATTACH FRESH SFT LORA
# ============================================================

def attach_fresh_sft_lora(
    model,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "ATTACHING FRESH SFT LORA"
    )

    print(
        "=" * 60
    )

    model = (
        FastLanguageModel.get_peft_model(

            model,

            r=(
                LORA_RANK
            ),

            target_modules=(
                LORA_TARGET_MODULES
            ),

            lora_alpha=(
                LORA_ALPHA
            ),

            lora_dropout=(
                LORA_DROPOUT
            ),

            bias="none",

            use_gradient_checkpointing=(
                "unsloth"
            ),

            random_state=(
                SEED
            ),

            use_rslora=False,

            loftq_config=None,
        )
    )

    print(
        "LoRA rank:",
        LORA_RANK,
    )

    print(
        "LoRA alpha:",
        LORA_ALPHA,
    )

    print(
        "LoRA dropout:",
        LORA_DROPOUT,
    )

    print(
        "Target modules:",
        LORA_TARGET_MODULES,
    )

    print(
        "\nTrainable parameters:"
    )

    model.print_trainable_parameters()

    print_cuda_memory(
        "MEMORY AFTER FRESH SFT LORA"
    )

    return model


# ============================================================
# 25. TRAINING PLAN
# ============================================================

def print_training_plan(
    train_dataset,
    validation_dataset,
    test_dataset,
):

    effective_batch = (
        TRAIN_BATCH_SIZE
        * GRADIENT_ACCUMULATION_STEPS
    )

    steps_per_epoch = (
        math.ceil(

            len(
                train_dataset
            )

            / effective_batch
        )
    )

    total_steps = (
        steps_per_epoch
        * NUM_EPOCHS
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SFT TRAINING PLAN"
    )

    print(
        "=" * 60
    )

    print(
        "Starting model:"
    )

    print(
        CPT_MERGED_DIR
    )

    print(
        "\nTrain examples:",
        len(
            train_dataset
        ),
    )

    print(
        "Validation examples:",
        len(
            validation_dataset
        ),
    )

    print(
        "Held-out test examples:",
        len(
            test_dataset
        ),
    )

    print(
        "\nMax sequence length:",
        MAX_SEQ_LENGTH,
    )

    print(
        "Epochs:",
        NUM_EPOCHS,
    )

    print(
        "Train batch size:",
        TRAIN_BATCH_SIZE,
    )

    print(
        "Gradient accumulation:",
        GRADIENT_ACCUMULATION_STEPS,
    )

    print(
        "Effective batch size:",
        effective_batch,
    )

    print(
        "Estimated optimizer steps / epoch:",
        steps_per_epoch,
    )

    print(
        "Estimated total optimizer steps:",
        total_steps,
    )

    print(
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "Optimizer:",
        OPTIMIZER,
    )

    print(
        "Scheduler:",
        LR_SCHEDULER,
    )

    print(
        "Warmup ratio:",
        WARMUP_RATIO,
    )

    print(
        "Completion-only loss:",
        True,
    )

    return {

        "effective_batch_size":
            effective_batch,

        "steps_per_epoch":
            steps_per_epoch,

        "estimated_total_steps":
            total_steps,
    }


# ============================================================
# 26. CREATE SFT CONFIG
#
# IMPORTANT:
#
# completion_only_loss=True
#
# Therefore loss is calculated on the desired assistant
# response, not on the system instruction/question.
# ============================================================

def create_training_args():

    use_bf16 = (
        torch.cuda.is_bf16_supported()
    )

    use_fp16 = (
        not use_bf16
    )

    args = SFTConfig(

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_dir=str(
            CHECKPOINT_DIR
        ),


        # ----------------------------------------------------
        # SFT sequence setup
        # ----------------------------------------------------

        max_length=(
            MAX_SEQ_LENGTH
        ),

        packing=False,

        completion_only_loss=True,


        # ----------------------------------------------------
        # Batch
        # ----------------------------------------------------

        per_device_train_batch_size=(
            TRAIN_BATCH_SIZE
        ),

        per_device_eval_batch_size=(
            EVAL_BATCH_SIZE
        ),

        gradient_accumulation_steps=(
            GRADIENT_ACCUMULATION_STEPS
        ),


        # ----------------------------------------------------
        # Training duration
        # ----------------------------------------------------

        num_train_epochs=(
            NUM_EPOCHS
        ),


        # ----------------------------------------------------
        # Optimization
        # ----------------------------------------------------

        learning_rate=(
            LEARNING_RATE
        ),

        weight_decay=(
            WEIGHT_DECAY
        ),

        optim=(
            OPTIMIZER
        ),

        lr_scheduler_type=(
            LR_SCHEDULER
        ),

        warmup_ratio=(
            WARMUP_RATIO
        ),

        max_grad_norm=(
            MAX_GRAD_NORM
        ),


        # ----------------------------------------------------
        # Precision
        # ----------------------------------------------------

        bf16=(
            use_bf16
        ),

        fp16=(
            use_fp16
        ),


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        eval_strategy="epoch",

        prediction_loss_only=True,


        # ----------------------------------------------------
        # Checkpoints
        # ----------------------------------------------------

        save_strategy="epoch",

        save_total_limit=(
            SAVE_TOTAL_LIMIT
        ),

        load_best_model_at_end=True,

        metric_for_best_model=(
            "eval_loss"
        ),

        greater_is_better=False,


        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        logging_strategy="steps",

        logging_steps=(
            LOGGING_STEPS
        ),

        logging_first_step=True,

        report_to="none",


        # ----------------------------------------------------
        # Memory / stability
        # ----------------------------------------------------

        gradient_checkpointing=True,

        use_cache=False,

        dataloader_num_workers=0,

        dataloader_pin_memory=True,


        # ----------------------------------------------------
        # Reproducibility
        # ----------------------------------------------------

        seed=(
            SEED
        ),

        data_seed=(
            SEED
        ),
    )

    print(
        "\nPrecision:"
    )

    print(
        "BF16:",
        use_bf16,
    )

    print(
        "FP16:",
        use_fp16,
    )

    return args


# ============================================================
# 27. FIND LATEST CHECKPOINT
# ============================================================

def find_latest_checkpoint():

    if not CHECKPOINT_DIR.exists():

        return None

    checkpoints = []

    for path in CHECKPOINT_DIR.glob(
        "checkpoint-*"
    ):

        if not path.is_dir():

            continue

        try:

            step = int(
                path.name.split(
                    "-"
                )[-1]
            )

            checkpoints.append(
                (
                    step,
                    path,
                )
            )

        except ValueError:

            continue

    if not checkpoints:

        return None

    checkpoints.sort(
        key=lambda item:
        item[0]
    )

    return checkpoints[-1][1]


# ============================================================
# 28. SAVE JSON
# ============================================================

def save_json(
    data,
    path,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
        )


# ============================================================
# 29. MAIN
# ============================================================

def main():

    total_start_time = (
        time.time()
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "NEXAFLOW SFT TRAINING"
    )

    print(
        "CPT-MERGED QWEN2.5-1.5B + FRESH QLORA"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # PRE-FLIGHT CHECKS
    # ========================================================

    validate_starting_model()

    validate_data_files()

    prepare_output_directories()

    check_hardware()


    # ========================================================
    # LOAD RAW DATA
    # ========================================================

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = load_sft_dataset()


    check_empty_values(
        train_dataset,
        "train",
    )

    check_empty_values(
        validation_dataset,
        "validation",
    )

    check_empty_values(
        test_dataset,
        "test",
    )


    # ========================================================
    # LOAD CPT MODEL
    # ========================================================

    (
        model,
        tokenizer,
    ) = load_model()


    # ========================================================
    # FORMAT DATA
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FORMATTING SFT DATA"
    )

    print(
        "=" * 60
    )


    train_dataset = (
        prepare_prompt_completion_dataset(
            train_dataset,
            tokenizer,
        )
    )


    validation_dataset = (
        prepare_prompt_completion_dataset(
            validation_dataset,
            tokenizer,
        )
    )


    test_dataset = (
        prepare_prompt_completion_dataset(
            test_dataset,
            tokenizer,
        )
    )


    verify_formatted_dataset(
        train_dataset,
        "train",
    )

    verify_formatted_dataset(
        validation_dataset,
        "validation",
    )

    verify_formatted_dataset(
        test_dataset,
        "test",
    )


    # ========================================================
    # PRINT ONE EXAMPLE
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SAMPLE SFT RECORD"
    )

    print(
        "=" * 60
    )

    print(
        "\nPROMPT:\n"
    )

    print(
        train_dataset[0][
            "prompt"
        ]
    )

    print(
        "\nCOMPLETION:\n"
    )

    print(
        train_dataset[0][
            "completion"
        ]
    )


    # ========================================================
    # TOKEN LENGTH CHECK
    # ========================================================

    token_stats = {

        "train":
            check_token_lengths(
                train_dataset,
                tokenizer,
                "train",
            ),

        "validation":
            check_token_lengths(
                validation_dataset,
                tokenizer,
                "validation",
            ),

        "test":
            check_token_lengths(
                test_dataset,
                tokenizer,
                "test",
            ),
    }


    # ========================================================
    # FRESH SFT LORA
    # ========================================================

    model = attach_fresh_sft_lora(
        model
    )


    # ========================================================
    # TRAINING PLAN
    # ========================================================

    training_plan = (
        print_training_plan(
            train_dataset,
            validation_dataset,
            test_dataset,
        )
    )


    # ========================================================
    # TRAINING CONFIG
    # ========================================================

    training_args = (
        create_training_args()
    )


    # ========================================================
    # TRAINER
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CREATING SFT TRAINER"
    )

    print(
        "=" * 60
    )


    trainer = SFTTrainer(

        model=model,

        args=training_args,

        train_dataset=(
            train_dataset
        ),

        eval_dataset=(
            validation_dataset
        ),

        processing_class=(
            tokenizer
        ),
    )


    # ========================================================
    # BASELINE VALIDATION
    #
    # This tells us how the CPT model performs on SFT
    # instructions BEFORE SFT updates.
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "BASELINE VALIDATION BEFORE SFT"
    )

    print(
        "=" * 60
    )


    cleanup_memory()


    baseline_start = (
        time.time()
    )


    try:

        baseline_metrics = (
            trainer.evaluate()
        )

    except torch.cuda.OutOfMemoryError as error:

        cleanup_memory()

        raise RuntimeError(
            """
CUDA OOM during baseline SFT validation.

Try:

1. Close unnecessary GPU applications.
2. Restart the Python process.
3. Confirm eval batch size = 1.
4. Keep load_in_4bit=True.
5. If necessary reduce MAX_SEQ_LENGTH from 1024 to 768.

Original CUDA error:

"""
            + str(error)
        ) from error


    baseline_time = (
        time.time()
        - baseline_start
    )


    baseline_loss = float(
        baseline_metrics[
            "eval_loss"
        ]
    )


    baseline_perplexity = (
        safe_perplexity(
            baseline_loss
        )
    )


    print(
        f"Baseline validation loss: "
        f"{baseline_loss:.6f}"
    )

    print(
        f"Baseline perplexity: "
        f"{baseline_perplexity:.4f}"
    )

    print(
        f"Baseline evaluation time: "
        f"{baseline_time:.2f}s"
    )


    print_cuda_memory(
        "MEMORY AFTER BASELINE VALIDATION"
    )


    # ========================================================
    # RESUME
    # ========================================================

    resume_checkpoint = None


    if AUTO_RESUME:

        resume_checkpoint = (
            find_latest_checkpoint()
        )


    if resume_checkpoint:

        print(
            "\n"
            + "=" * 60
        )

        print(
            "RESUMING SFT"
        )

        print(
            "=" * 60
        )

        print(
            "Checkpoint:",
            resume_checkpoint,
        )

    else:

        print(
            "\nStarting fresh SFT training."
        )


    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "STARTING SFT TRAINING"
    )

    print(
        "=" * 60
    )


    train_start = (
        time.time()
    )


    try:

        train_result = trainer.train(

            resume_from_checkpoint=(

                str(
                    resume_checkpoint
                )

                if resume_checkpoint

                else None
            )
        )

    except torch.cuda.OutOfMemoryError as error:

        cleanup_memory()

        raise RuntimeError(
            """
CUDA OUT OF MEMORY DURING SFT.

Try in this order:

1. Close other GPU applications.
2. Restart the Python process.
3. Keep batch size = 1.
4. Keep gradient accumulation = 8.
5. Keep 4-bit loading enabled.
6. If still necessary:
       MAX_SEQ_LENGTH = 768

Do not immediately reduce LoRA rank.
r=16 is already a reasonable setting.

Original CUDA error:

"""
            + str(error)
        ) from error


    training_time = (
        time.time()
        - train_start
    )


    # ========================================================
    # BEST CHECKPOINT
    # ========================================================

    best_checkpoint = (
        trainer.state.best_model_checkpoint
    )

    best_validation_loss = (
        trainer.state.best_metric
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "BEST SFT CHECKPOINT"
    )

    print(
        "=" * 60
    )

    print(
        "Best checkpoint:",
        best_checkpoint,
    )

    print(
        "Best validation loss:",
        best_validation_loss,
    )


    # ========================================================
    # FINAL VALIDATION
    #
    # load_best_model_at_end=True restores the best epoch.
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL VALIDATION USING BEST SFT MODEL"
    )

    print(
        "=" * 60
    )


    cleanup_memory()


    final_start = (
        time.time()
    )


    final_metrics = (
        trainer.evaluate()
    )


    final_eval_time = (
        time.time()
        - final_start
    )


    final_loss = float(
        final_metrics[
            "eval_loss"
        ]
    )


    final_perplexity = (
        safe_perplexity(
            final_loss
        )
    )


    loss_reduction = (
        baseline_loss
        - final_loss
    )


    if baseline_loss != 0:

        improvement_percent = (
            loss_reduction
            / baseline_loss
            * 100
        )

    else:

        improvement_percent = (
            None
        )


    print(
        f"Before SFT loss : "
        f"{baseline_loss:.6f}"
    )

    print(
        f"After SFT loss  : "
        f"{final_loss:.6f}"
    )

    print(
        f"Final perplexity: "
        f"{final_perplexity:.4f}"
    )

    print(
        f"Loss reduction  : "
        f"{loss_reduction:.6f}"
    )


    if improvement_percent is not None:

        print(
            f"Improvement     : "
            f"{improvement_percent:.2f}%"
        )


    # ========================================================
    # IMPORTANT:
    #
    # Do NOT evaluate on the SFT test set here.
    #
    # Validation chooses the checkpoint.
    #
    # Test remains held out from model selection.
    # ========================================================

    print(
        "\nSFT test split remains held out."
    )


    # ========================================================
    # SAVE BEST SFT ADAPTER
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SAVING BEST SFT ADAPTER"
    )

    print(
        "=" * 60
    )


    ADAPTER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    trainer.model.save_pretrained(
        str(
            ADAPTER_DIR
        )
    )


    tokenizer.save_pretrained(
        str(
            ADAPTER_DIR
        )
    )


    # ========================================================
    # VERIFY ADAPTER
    # ========================================================

    adapter_config = (
        ADAPTER_DIR
        / "adapter_config.json"
    )


    if not adapter_config.exists():

        raise RuntimeError(
            f"""
SFT completed but adapter_config.json was not found:

{adapter_config}
"""
        )


    print(
        "SFT adapter saved successfully:"
    )

    print(
        ADAPTER_DIR
    )


    # ========================================================
    # METRICS
    # ========================================================

    total_time = (
        time.time()
        - total_start_time
    )


    metrics_payload = {

        "stage":
            "supervised_fine_tuning",

        "starting_model":
            str(
                CPT_MERGED_DIR
            ),

        "base_family":
            "Qwen/Qwen2.5-1.5B",

        "training_method":
            "4-bit QLoRA SFT",

        "fresh_lora":
            True,

        "completion_only_loss":
            True,

        "max_sequence_length":
            MAX_SEQ_LENGTH,


        "dataset": {

            "train_examples":
                len(
                    train_dataset
                ),

            "validation_examples":
                len(
                    validation_dataset
                ),

            "held_out_test_examples":
                len(
                    test_dataset
                ),

            "token_statistics":
                token_stats,
        },


        "lora": {

            "rank":
                LORA_RANK,

            "alpha":
                LORA_ALPHA,

            "dropout":
                LORA_DROPOUT,

            "target_modules":
                LORA_TARGET_MODULES,
        },


        "training": {

            "epochs":
                NUM_EPOCHS,

            "train_batch_size":
                TRAIN_BATCH_SIZE,

            "eval_batch_size":
                EVAL_BATCH_SIZE,

            "gradient_accumulation_steps":
                GRADIENT_ACCUMULATION_STEPS,

            "effective_batch_size":
                training_plan[
                    "effective_batch_size"
                ],

            "learning_rate":
                LEARNING_RATE,

            "weight_decay":
                WEIGHT_DECAY,

            "warmup_ratio":
                WARMUP_RATIO,

            "max_grad_norm":
                MAX_GRAD_NORM,

            "optimizer":
                OPTIMIZER,

            "scheduler":
                LR_SCHEDULER,

            "estimated_steps_per_epoch":
                training_plan[
                    "steps_per_epoch"
                ],

            "estimated_total_steps":
                training_plan[
                    "estimated_total_steps"
                ],

            "actual_global_steps":
                trainer.state.global_step,
        },


        "validation": {

            "baseline_loss":
                baseline_loss,

            "baseline_perplexity":
                baseline_perplexity,

            "best_validation_loss":
                best_validation_loss,

            "final_loss":
                final_loss,

            "final_perplexity":
                final_perplexity,

            "loss_reduction":
                loss_reduction,

            "improvement_percent":
                improvement_percent,
        },


        "checkpointing": {

            "best_checkpoint":
                best_checkpoint,

            "load_best_model_at_end":
                True,

            "test_used_for_model_selection":
                False,
        },


        "runtime_seconds": {

            "baseline_evaluation":
                baseline_time,

            "training":
                training_time,

            "final_evaluation":
                final_eval_time,

            "overall":
                total_time,
        },


        "trainer_train_metrics":
            dict(
                train_result.metrics
            ),

        "trainer_history":
            trainer.state.log_history,
    }


    save_json(
        metrics_payload,
        METRICS_FILE,
    )


    # ========================================================
    # FINAL MEMORY REPORT
    # ========================================================

    print_cuda_memory(
        "FINAL CUDA MEMORY"
    )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        "\n\n"
        + "=" * 60
    )

    print(
        "SFT TRAINING COMPLETE"
    )

    print(
        "=" * 60
    )


    print(
        "\nStarting CPT model:"
    )

    print(
        CPT_MERGED_DIR
    )


    print(
        "\nBest SFT checkpoint:"
    )

    print(
        best_checkpoint
    )


    print(
        "\nSFT adapter:"
    )

    print(
        ADAPTER_DIR
    )


    print(
        "\nMetrics:"
    )

    print(
        METRICS_FILE
    )


    print(
        f"\nBaseline validation loss: "
        f"{baseline_loss:.6f}"
    )


    print(
        f"Final validation loss   : "
        f"{final_loss:.6f}"
    )


    print(
        f"Final validation PPL    : "
        f"{final_perplexity:.4f}"
    )


    print(
        f"Training runtime        : "
        f"{training_time / 60:.2f} minutes"
    )


    print(
        f"Overall runtime         : "
        f"{total_time / 60:.2f} minutes"
    )


    print(
        """

CURRENT PIPELINE:

Qwen2.5-1.5B Base
        ↓
CPT + QLoRA
        ↓
CPT merge
        ↓
models/cpt_merged
        ↓
Fresh SFT QLoRA
        ↓
models/sft_adapter


NEXT STEP:

models/cpt_merged
        +
models/sft_adapter
        ↓
MERGE
        ↓
models/sft_merged


DO NOT start DPO yet.

First:

1. Merge SFT
2. Load models/sft_merged
3. Test it aggressively
4. Check hallucination / unknown-person behavior
5. Only then continue to DPO
"""
    )


if __name__ == "__main__":

    main()