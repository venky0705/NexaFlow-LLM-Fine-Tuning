from pathlib import Path
import gc
import json
import math
import random
import shutil
import sys
import time

import unsloth

import numpy as np
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import DPOConfig, DPOTrainer


# ============================================================
# NEXAFLOW TECHNOLOGIES
# DIRECT PREFERENCE OPTIMIZATION (DPO)
#
# Starting model:
#
#     models/sft_merged
#
# Training:
#
#     4-bit QLoRA
#     FRESH DPO LoRA
#
# Data:
#
#     Data/04_dpo/train.jsonl
#     Data/04_dpo/validation.jsonl
#     Data/04_dpo/test.jsonl
#
# Output:
#
#     models/dpo_checkpoints/
#     models/dpo_adapter/
#     results/dpo/
#
# IMPORTANT:
#
# The SFT adapter is NOT reused.
#
# models/sft_merged already contains:
#
#     Base
#       +
#     CPT
#       +
#     SFT
#
# DPO now attaches a completely fresh LoRA adapter.
# ============================================================


# ============================================================
# 1. PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# 2. STARTING MODEL
# ============================================================

SFT_MERGED_DIR = (
    PROJECT_ROOT
    / "models"
    / "sft_merged"
)


# ============================================================
# 3. DPO DATA
# ============================================================

DPO_DATA_DIR = (
    PROJECT_ROOT
    / "Data"
    / "04_dpo"
)

TRAIN_FILE = (
    DPO_DATA_DIR
    / "train.jsonl"
)

VALIDATION_FILE = (
    DPO_DATA_DIR
    / "validation.jsonl"
)

TEST_FILE = (
    DPO_DATA_DIR
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
    / "dpo_checkpoints"
)

ADAPTER_DIR = (
    MODELS_DIR
    / "dpo_adapter"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "dpo"
)

METRICS_FILE = (
    RESULTS_DIR
    / "dpo_training_metrics.json"
)


# ============================================================
# 5. MODEL CONFIG
# ============================================================

MAX_SEQ_LENGTH = 1024

LOAD_IN_4BIT = True


# ============================================================
# 6. DPO LORA CONFIG
#
# Fresh LoRA on top of SFT merged model.
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
# 7. DPO TRAINING CONFIG
#
# DPO should normally use a MUCH smaller learning rate than
# CPT or SFT.
#
# CPT = 2e-4
# SFT = 1e-4
# DPO = 5e-6
#
# The SFT model already has good behavior. We only want to
# adjust preference boundaries.
# ============================================================

NUM_EPOCHS = 2

TRAIN_BATCH_SIZE = 1

EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 5e-6

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.10

MAX_GRAD_NORM = 1.0

OPTIMIZER = "adamw_8bit"

LR_SCHEDULER = "cosine"


# ============================================================
# 8. DPO-SPECIFIC CONFIG
#
# beta controls how strongly DPO moves away from the
# reference policy.
#
# 0.1 is a standard conservative starting value.
# ============================================================

DPO_BETA = 0.1

DPO_LOSS_TYPE = "sigmoid"


# ============================================================
# 9. REPRODUCIBILITY
# ============================================================

SEED = 42


# ============================================================
# 10. LOGGING / SAVING
# ============================================================

LOGGING_STEPS = 5

SAVE_TOTAL_LIMIT = 2


# ============================================================
# 11. RESUME
#
# False:
#     start a fresh DPO run.
#
# True:
#     resume from latest DPO checkpoint if one exists.
# ============================================================

AUTO_RESUME = False


# ============================================================
# 12. CLEAR OLD OUTPUT
#
# Keep False normally.
#
# Set True only when intentionally restarting DPO completely.
# ============================================================

CLEAR_OLD_DPO_OUTPUT = False


# ============================================================
# 13. RANDOM SEEDS
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
# 14. MEMORY HELPERS
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
# 15. SAFE FLOAT
# ============================================================

def safe_float(value):

    if value is None:

        return None

    try:

        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return None


# ============================================================
# 16. NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    return " ".join(
        str(
            text
        )
        .lower()
        .split()
    )


# ============================================================
# 17. CHECK STARTING MODEL
# ============================================================

def validate_starting_model():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CHECKING SFT MERGED MODEL"
    )

    print(
        "=" * 60
    )

    if not SFT_MERGED_DIR.exists():

        raise FileNotFoundError(
            f"""
SFT merged model does not exist:

{SFT_MERGED_DIR}

Before DPO you must have:

Base
  ↓
CPT
  ↓
CPT merge
  ↓
SFT
  ↓
SFT merge
  ↓
models/sft_merged
"""
        )

    config_file = (
        SFT_MERGED_DIR
        / "config.json"
    )

    if not config_file.exists():

        raise FileNotFoundError(
            f"""
Missing model config:

{config_file}

The SFT merge may be incomplete.
"""
        )

    weight_files = list(
        SFT_MERGED_DIR.glob(
            "*.safetensors"
        )
    )

    index_files = list(
        SFT_MERGED_DIR.glob(
            "*.index.json"
        )
    )

    if (
        not weight_files
        and not index_files
    ):

        raise RuntimeError(
            f"""
No model weights found inside:

{SFT_MERGED_DIR}
"""
        )

    print(
        "SFT merged model:"
    )

    print(
        SFT_MERGED_DIR
    )

    print(
        "\nSFT merged model check: PASSED"
    )


# ============================================================
# 18. CHECK DPO DATA FILES
# ============================================================

def validate_data_files():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CHECKING DPO DATA FILES"
    )

    print(
        "=" * 60
    )

    for path in [

        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE,

    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"""
Required DPO file does not exist:

{path}

Run:

uv run python "Data preparation/prepare_dpo_data.py"
"""
            )

        if path.stat().st_size == 0:

            raise RuntimeError(
                f"""
DPO data file is empty:

{path}
"""
            )

        print(
            "OK:",
            path
        )


# ============================================================
# 19. OUTPUT DIRECTORIES
# ============================================================

def prepare_output_directories():

    if CLEAR_OLD_DPO_OUTPUT:

        print(
            "\nCLEAR_OLD_DPO_OUTPUT=True"
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
# 20. HARDWARE CHECK
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

This DPO script is configured for CUDA + 4-bit QLoRA.
"""
        )

    properties = (
        torch.cuda.get_device_properties(0)
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
# 21. LOAD DPO DATASET
# ============================================================

def load_dpo_dataset():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "LOADING DPO DATASET"
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
            "DPO train dataset is empty."
        )

    if len(validation_dataset) == 0:

        raise RuntimeError(
            "DPO validation dataset is empty."
        )

    if len(test_dataset) == 0:

        raise RuntimeError(
            "DPO test dataset is empty."
        )

    required_columns = {

        "prompt",
        "chosen",
        "rejected",
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

        missing = (
            required_columns
            - set(
                split.column_names
            )
        )

        if missing:

            raise RuntimeError(
                f"""
DPO {split_name} dataset is missing columns:

{sorted(missing)}

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
# 22. REDUCE DATASET TO REQUIRED DPO COLUMNS
#
# Extra metadata is useful in the source JSONL, but the
# trainer only needs:
#
# prompt
# chosen
# rejected
# ============================================================

def select_dpo_columns(dataset):

    return dataset.select_columns(
        [
            "prompt",
            "chosen",
            "rejected",
        ]
    )


# ============================================================
# 23. DATA INTEGRITY CHECK
# ============================================================

def verify_dataset(
    dataset,
    split_name,
):

    print(
        f"\nChecking DPO split: "
        f"{split_name}"
    )

    prompt_keys = set()

    duplicate_prompts = 0

    for index in range(
        len(dataset)
    ):

        record = (
            dataset[index]
        )

        prompt = str(
            record[
                "prompt"
            ]
        ).strip()

        chosen = str(
            record[
                "chosen"
            ]
        ).strip()

        rejected = str(
            record[
                "rejected"
            ]
        ).strip()

        if not prompt:

            raise RuntimeError(
                f"""
Empty prompt detected:

split={split_name}
index={index}
"""
            )

        if not chosen:

            raise RuntimeError(
                f"""
Empty chosen response detected:

split={split_name}
index={index}
"""
            )

        if not rejected:

            raise RuntimeError(
                f"""
Empty rejected response detected:

split={split_name}
index={index}
"""
            )

        if (
            normalize_text(
                chosen
            )
            ==
            normalize_text(
                rejected
            )
        ):

            raise RuntimeError(
                f"""
Chosen and rejected are identical.

Split:
{split_name}

Index:
{index}

Prompt:
{prompt}
"""
            )

        prompt_key = (
            normalize_text(
                prompt
            )
        )

        if prompt_key in prompt_keys:

            duplicate_prompts += 1

        prompt_keys.add(
            prompt_key
        )

    print(
        "Examples:",
        len(
            dataset
        ),
    )

    print(
        "Unique prompts:",
        len(
            prompt_keys
        ),
    )

    print(
        "Duplicate prompts:",
        duplicate_prompts,
    )

    print(
        f"{split_name} integrity check: PASSED"
    )


# ============================================================
# 24. CROSS-SPLIT PROMPT LEAKAGE CHECK
# ============================================================

def verify_cross_split_overlap(
    train_dataset,
    validation_dataset,
    test_dataset,
):

    def prompts(dataset):

        return {

            normalize_text(
                row[
                    "prompt"
                ]
            )

            for row
            in dataset
        }

    train_prompts = prompts(
        train_dataset
    )

    validation_prompts = prompts(
        validation_dataset
    )

    test_prompts = prompts(
        test_dataset
    )

    train_validation = (
        train_prompts
        & validation_prompts
    )

    train_test = (
        train_prompts
        & test_prompts
    )

    validation_test = (
        validation_prompts
        & test_prompts
    )

    if (
        train_validation
        or train_test
        or validation_test
    ):

        raise RuntimeError(
            f"""
DPO CROSS-SPLIT PROMPT OVERLAP DETECTED.

Train / validation:
{len(train_validation)}

Train / test:
{len(train_test)}

Validation / test:
{len(validation_test)}
"""
        )

    print(
        "\nDPO cross-split exact prompt overlap: PASSED"
    )


# ============================================================
# 25. LOAD SFT MERGED MODEL
# ============================================================

def load_model():

    print(
        "\n"
        + "=" * 60
    )

    print(
        "LOADING SFT MERGED MODEL"
    )

    print(
        "=" * 60
    )

    cleanup_memory()

    try:

        model, tokenizer = (
            FastLanguageModel.from_pretrained(

                model_name=str(
                    SFT_MERGED_DIR
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
CUDA OUT OF MEMORY WHILE LOADING SFT MODEL.

Try:

1. Stop old Python processes.
2. Close unnecessary GPU applications.
3. Close unnecessary browser tabs.
4. Confirm load_in_4bit=True.
5. Restart this command.

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
        "left"
    )

    # --------------------------------------------------------
    # Training mode / cache
    # --------------------------------------------------------

    if hasattr(
        model,
        "config"
    ):

        model.config.use_cache = (
            False
        )

    print(
        "SFT merged model loaded."
    )

    print(
        "EOS token:",
        repr(
            tokenizer.eos_token
        ),
    )

    print(
        "PAD token:",
        repr(
            tokenizer.pad_token
        ),
    )

    print_cuda_memory(
        "MEMORY AFTER SFT MODEL LOAD"
    )

    return (
        model,
        tokenizer,
    )


# ============================================================
# 26. TOKEN LENGTH CHECK
#
# DPO processes:
#
# prompt + chosen
# prompt + rejected
#
# We inspect both paths before starting training.
# ============================================================

def check_token_lengths(
    dataset,
    tokenizer,
    split_name,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        f"TOKEN LENGTH CHECK: "
        f"{split_name.upper()}"
    )

    print(
        "=" * 60
    )

    max_prompt = 0

    max_chosen_total = 0

    max_rejected_total = 0

    chosen_over_limit = 0

    rejected_over_limit = 0


    for record in dataset:

        prompt = (
            record[
                "prompt"
            ]
        )

        chosen = (
            record[
                "chosen"
            ]
        )

        rejected = (
            record[
                "rejected"
            ]
        )


        prompt_tokens = len(

            tokenizer(

                prompt,

                add_special_tokens=False,

                truncation=False,

            )[
                "input_ids"
            ]
        )


        chosen_tokens = len(

            tokenizer(

                prompt
                + chosen,

                add_special_tokens=False,

                truncation=False,

            )[
                "input_ids"
            ]
        )


        rejected_tokens = len(

            tokenizer(

                prompt
                + rejected,

                add_special_tokens=False,

                truncation=False,

            )[
                "input_ids"
            ]
        )


        max_prompt = max(

            max_prompt,

            prompt_tokens,
        )


        max_chosen_total = max(

            max_chosen_total,

            chosen_tokens,
        )


        max_rejected_total = max(

            max_rejected_total,

            rejected_tokens,
        )


        if (
            chosen_tokens
            > MAX_SEQ_LENGTH
        ):

            chosen_over_limit += 1


        if (
            rejected_tokens
            > MAX_SEQ_LENGTH
        ):

            rejected_over_limit += 1


    print(
        "Maximum prompt tokens:",
        max_prompt,
    )

    print(
        "Maximum prompt + chosen:",
        max_chosen_total,
    )

    print(
        "Maximum prompt + rejected:",
        max_rejected_total,
    )

    print(
        "Chosen sequences over limit:",
        chosen_over_limit,
    )

    print(
        "Rejected sequences over limit:",
        rejected_over_limit,
    )


    if (
        chosen_over_limit > 0
        or rejected_over_limit > 0
    ):

        print(
            "\nWARNING:"
        )

        print(
            "Some DPO examples exceed "
            f"{MAX_SEQ_LENGTH} tokens."
        )

        print(
            "They may be truncated during DPO."
        )


    return {

        "max_prompt_tokens":
            max_prompt,

        "max_prompt_chosen_tokens":
            max_chosen_total,

        "max_prompt_rejected_tokens":
            max_rejected_total,

        "chosen_over_limit":
            chosen_over_limit,

        "rejected_over_limit":
            rejected_over_limit,
    }


# ============================================================
# 27. ATTACH FRESH DPO LORA
#
# This is NOT the SFT adapter.
#
# SFT is already merged into the model.
#
# We now create a brand-new preference-training adapter.
# ============================================================

def attach_fresh_dpo_lora(
    model,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        "ATTACHING FRESH DPO LORA"
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
        "MEMORY AFTER DPO LORA"
    )

    return model


# ============================================================
# 28. TRAINING PLAN
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
        "DPO TRAINING PLAN"
    )

    print(
        "=" * 60
    )


    print(
        "Starting model:"
    )

    print(
        SFT_MERGED_DIR
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
        "\nEpochs:",
        NUM_EPOCHS,
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
        "Learning rate:",
        LEARNING_RATE,
    )

    print(
        "DPO beta:",
        DPO_BETA,
    )

    print(
        "DPO loss:",
        DPO_LOSS_TYPE,
    )

    print(
        "Batch size:",
        TRAIN_BATCH_SIZE,
    )

    print(
        "Gradient accumulation:",
        GRADIENT_ACCUMULATION_STEPS,
    )

    print(
        "Effective batch:",
        effective_batch,
    )

    print(
        "Optimizer steps / epoch:",
        steps_per_epoch,
    )

    print(
        "Estimated total optimizer steps:",
        total_steps,
    )

    print(
        "Reference policy:"
    )

    print(
        "Initial SFT policy / LoRA-disabled reference"
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
# 29. CREATE DPO CONFIG
# ============================================================

def create_training_args():

    use_bf16 = (
        torch.cuda.is_bf16_supported()
    )

    use_fp16 = (
        not use_bf16
    )


    args = DPOConfig(

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        output_dir=str(
            CHECKPOINT_DIR
        ),


        # ----------------------------------------------------
        # Sequence
        # ----------------------------------------------------

        max_length=(
            MAX_SEQ_LENGTH
        ),

        truncation_mode=(
            "keep_start"
        ),


        # ----------------------------------------------------
        # DPO
        # ----------------------------------------------------

        beta=(
            DPO_BETA
        ),

        loss_type=(
            DPO_LOSS_TYPE
        ),

        label_smoothing=0.0,


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
        # Epochs
        # ----------------------------------------------------

        num_train_epochs=(
            NUM_EPOCHS
        ),


        # ----------------------------------------------------
        # Optimizer
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
        # Evaluation
        # ----------------------------------------------------

        eval_strategy="epoch",

        prediction_loss_only=False,


        # ----------------------------------------------------
        # Checkpointing
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
        # Memory
        # ----------------------------------------------------

        gradient_checkpointing=True,

        use_cache=False,

        dataloader_num_workers=0,

        dataloader_pin_memory=True,


        # ----------------------------------------------------
        # Reference model handling
        #
        # False:
        # calculate reference log-probs using the initial
        # policy / adapter-disabled path during training.
        # ----------------------------------------------------

        precompute_ref_log_probs=False,


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
# 30. FIND LATEST CHECKPOINT
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

        except ValueError:

            continue

        checkpoints.append(
            (
                step,
                path,
            )
        )


    if not checkpoints:

        return None


    checkpoints.sort(
        key=lambda item:
        item[0]
    )

    return checkpoints[-1][1]


# ============================================================
# 31. SAVE JSON
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
# 32. GET METRIC
# ============================================================

def get_metric(
    metrics,
    key,
):

    if (
        metrics is None
        or key not in metrics
    ):

        return None

    return safe_float(
        metrics[
            key
        ]
    )


# ============================================================
# 33. PRINT DPO METRICS
# ============================================================

def print_dpo_metrics(
    title,
    metrics,
):

    print(
        "\n"
        + "=" * 60
    )

    print(
        title
    )

    print(
        "=" * 60
    )

    keys = [

        "eval_loss",

        "eval_rewards/chosen",

        "eval_rewards/rejected",

        "eval_rewards/accuracies",

        "eval_rewards/margins",

        "eval_logps/chosen",

        "eval_logps/rejected",
    ]


    found_any = False


    for key in keys:

        if key in metrics:

            found_any = True

            print(
                f"{key}: "
                f"{metrics[key]}"
            )


    if not found_any:

        print(
            metrics
        )


# ============================================================
# 34. MAIN
# ============================================================

def main():

    total_start = (
        time.time()
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "NEXAFLOW DPO TRAINING"
    )

    print(
        "SFT-MERGED QWEN2.5-1.5B + FRESH QLORA"
    )

    print(
        "=" * 60
    )


    # ========================================================
    # PRE-FLIGHT
    # ========================================================

    validate_starting_model()

    validate_data_files()

    prepare_output_directories()

    check_hardware()


    # ========================================================
    # DATASET
    # ========================================================

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = load_dpo_dataset()


    # ========================================================
    # VERIFY RAW DATA
    # ========================================================

    verify_dataset(
        train_dataset,
        "train",
    )

    verify_dataset(
        validation_dataset,
        "validation",
    )

    verify_dataset(
        test_dataset,
        "test",
    )


    verify_cross_split_overlap(

        train_dataset,

        validation_dataset,

        test_dataset,
    )


    # ========================================================
    # KEEP ONLY DPO COLUMNS
    # ========================================================

    train_dataset = (
        select_dpo_columns(
            train_dataset
        )
    )

    validation_dataset = (
        select_dpo_columns(
            validation_dataset
        )
    )

    test_dataset = (
        select_dpo_columns(
            test_dataset
        )
    )


    # ========================================================
    # MODEL
    # ========================================================

    (
        model,
        tokenizer,
    ) = load_model()


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
    # FRESH DPO LORA
    # ========================================================

    model = attach_fresh_dpo_lora(
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
    # CONFIG
    # ========================================================

    training_args = (
        create_training_args()
    )


    # ========================================================
    # CREATE DPO TRAINER
    #
    # ref_model=None is intentional.
    #
    # With PEFT/LoRA, the initial SFT policy acts as the
    # reference while the fresh DPO adapter is optimized.
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "CREATING DPO TRAINER"
    )

    print(
        "=" * 60
    )


    try:

        trainer = DPOTrainer(

            model=model,

            ref_model=None,

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

    except Exception as error:

        cleanup_memory()

        raise RuntimeError(
            """
Failed to create DPOTrainer.

Check:

- TRL version
- Unsloth version
- PEFT version
- DPO dataset format

Required DPO columns:

prompt
chosen
rejected

Original error:

"""
            + str(error)
        ) from error


    print_cuda_memory(
        "MEMORY AFTER DPO TRAINER CREATION"
    )


    # ========================================================
    # BASELINE VALIDATION
    #
    # Evaluate preference behavior before DPO updates.
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "BASELINE DPO VALIDATION"
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
CUDA OUT OF MEMORY DURING DPO VALIDATION.

DPO uses more memory than standard SFT because it compares
chosen and rejected responses and also computes reference
log probabilities.

Try:

1. Close other GPU applications.
2. Stop old Python processes.
3. Restart this command.
4. Keep batch size = 1.
5. Keep 4-bit model loading.
6. If necessary reduce MAX_SEQ_LENGTH from 1024 to 768.

Do NOT immediately reduce LoRA rank.

Original CUDA error:

"""
            + str(error)
        ) from error


    baseline_time = (
        time.time()
        - baseline_start
    )


    print_dpo_metrics(

        "DPO BASELINE METRICS",

        baseline_metrics,
    )


    print_cuda_memory(
        "MEMORY AFTER BASELINE DPO EVALUATION"
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
            "RESUMING DPO"
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
            "\nStarting fresh DPO training."
        )


    # ========================================================
    # TRAIN
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "STARTING DPO TRAINING"
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
CUDA OUT OF MEMORY DURING DPO TRAINING.

Recommended recovery order:

1. Stop other Python processes.
2. Close unnecessary GPU applications.
3. Restart the command.
4. Keep:
       batch size = 1
       gradient accumulation = 8
       load_in_4bit = True
5. If it still fails, change:
       MAX_SEQ_LENGTH = 768

Your LoRA rank of 16 is already reasonable.

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

    best_metric = (
        trainer.state.best_metric
    )


    print(
        "\n"
        + "=" * 60
    )

    print(
        "BEST DPO CHECKPOINT"
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
        best_metric,
    )


    # ========================================================
    # FINAL VALIDATION
    #
    # load_best_model_at_end=True should restore the best
    # DPO checkpoint.
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "FINAL DPO VALIDATION USING BEST MODEL"
    )

    print(
        "=" * 60
    )


    cleanup_memory()

    final_eval_start = (
        time.time()
    )


    final_metrics = (
        trainer.evaluate()
    )


    final_eval_time = (
        time.time()
        - final_eval_start
    )


    print_dpo_metrics(

        "FINAL DPO METRICS",

        final_metrics,
    )


    # ========================================================
    # TEST SPLIT
    #
    # IMPORTANT:
    #
    # We do NOT use the DPO test split for checkpoint
    # selection or hyperparameter tuning here.
    # ========================================================

    print(
        "\nDPO test split remains held out."
    )


    # ========================================================
    # SAVE BEST DPO ADAPTER
    # ========================================================

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SAVING BEST DPO ADAPTER"
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
DPO training completed but adapter_config.json
was not found:

{adapter_config}
"""
        )


    print(
        "DPO adapter saved successfully:"
    )

    print(
        ADAPTER_DIR
    )


    # ========================================================
    # METRICS
    # ========================================================

    total_time = (
        time.time()
        - total_start
    )


    metrics_payload = {

        "stage":
            "direct_preference_optimization",

        "starting_model":
            str(
                SFT_MERGED_DIR
            ),

        "model_family":
            "Qwen/Qwen2.5-1.5B",

        "training_method":
            "4-bit QLoRA DPO",

        "fresh_dpo_lora":
            True,

        "reference_model":
            (
                "Initial SFT policy via "
                "adapter-disabled PEFT reference"
            ),

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


        "dpo": {

            "beta":
                DPO_BETA,

            "loss_type":
                DPO_LOSS_TYPE,

            "label_smoothing":
                0.0,

            "precompute_ref_log_probs":
                False,
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

            "steps_per_epoch":
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


        "baseline_validation":
            baseline_metrics,


        "final_validation":
            final_metrics,


        "checkpointing": {

            "best_checkpoint":
                best_checkpoint,

            "best_metric":
                best_metric,

            "load_best_model_at_end":
                True,

            "test_used_for_selection":
                False,
        },


        "runtime_seconds": {

            "baseline_validation":
                baseline_time,

            "training":
                training_time,

            "final_validation":
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
    # FINAL MEMORY
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
        "DPO TRAINING COMPLETE"
    )

    print(
        "=" * 60
    )


    print(
        "\nStarting SFT model:"
    )

    print(
        SFT_MERGED_DIR
    )


    print(
        "\nBest DPO checkpoint:"
    )

    print(
        best_checkpoint
    )


    print(
        "\nDPO adapter:"
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
        f"\nTraining runtime: "
        f"{training_time / 60:.2f} minutes"
    )

    print(
        f"Overall runtime: "
        f"{total_time / 60:.2f} minutes"
    )


    print(
        """

CURRENT PIPELINE:

Qwen/Qwen2.5-1.5B Base
        ↓
CPT + QLoRA
        ↓
CPT merge
        ↓
models/cpt_merged
        ↓
SFT + fresh QLoRA
        ↓
SFT merge
        ↓
models/sft_merged
        ↓
DPO + fresh QLoRA
        ↓
models/dpo_adapter


NEXT STEP:

models/sft_merged
        +
models/dpo_adapter
        ↓
MERGE
        ↓
models/dpo_merged


AFTER THAT:

Use the SAME frozen evaluation benchmark:

Data/05_evaluation/benchmark.jsonl

to compare:

Base
vs
CPT
vs
SFT
vs
DPO
"""
    )


if __name__ == "__main__":

    main()