from pathlib import Path
import json
import math

import torch
from datasets import load_dataset

from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CPT_MERGED_MODEL = (
    PROJECT_ROOT
    / "models"
    / "cpt_merged"
)

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
# 2. CONFIG
# ============================================================

MAX_SEQ_LENGTH = 1024

LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0

NUM_EPOCHS = 3

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

SEED = 42


# ============================================================
# 3. CREATE OUTPUT DIRECTORIES
# ============================================================

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
# 4. CHECK REQUIRED FILES
# ============================================================

if not CPT_MERGED_MODEL.exists():

    raise FileNotFoundError(
        f"""
CPT merged model not found:

{CPT_MERGED_MODEL}

Run the CPT merge script first.
"""
    )


for file_path in [
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"""
Required SFT data file not found:

{file_path}

Run:

uv run python "Data preparation/prepare_sft_data.py"
"""
        )


# ============================================================
# 5. GPU CHECK
# ============================================================

print("\n" + "=" * 60)
print("DEVICE")
print("=" * 60)

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU was not detected."
    )

gpu_name = (
    torch.cuda.get_device_name(0)
)

gpu_memory = (
    torch.cuda.get_device_properties(0)
    .total_memory
    / 1024**3
)

print(
    "GPU:",
    gpu_name,
)

print(
    f"GPU memory: {gpu_memory:.2f} GB"
)

print(
    "BF16 supported:",
    torch.cuda.is_bf16_supported(),
)


# ============================================================
# 6. LOAD SFT DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING SFT DATA")
print("=" * 60)

dataset = load_dataset(
    "json",
    data_files={
        "train":
            str(TRAIN_FILE),

        "validation":
            str(VALIDATION_FILE),

        "test":
            str(TEST_FILE),
    },
)

train_dataset = dataset["train"]

validation_dataset = dataset[
    "validation"
]

test_dataset = dataset["test"]

print(
    "Train examples:",
    len(train_dataset),
)

print(
    "Validation examples:",
    len(validation_dataset),
)

print(
    "Test examples:",
    len(test_dataset),
)


# ============================================================
# 7. VERIFY DATA FORMAT
# ============================================================

if len(train_dataset) == 0:

    raise RuntimeError(
        "SFT training dataset is empty."
    )


required_fields = {
    "instruction",
    "input",
    "output",
}


sample = train_dataset[0]

missing_fields = (
    required_fields
    - set(sample.keys())
)


if missing_fields:

    raise RuntimeError(
        f"""
SFT dataset is missing required fields:

{missing_fields}

Expected fields:

instruction
input
output
"""
    )


# ============================================================
# 8. LOAD CPT MERGED MODEL IN 4-BIT
# ============================================================

print("\n" + "=" * 60)
print("LOADING CPT MERGED MODEL")
print("=" * 60)

print(
    "Starting model:"
)

print(
    CPT_MERGED_MODEL
)


model, tokenizer = (
    FastLanguageModel.from_pretrained(
        model_name=str(
            CPT_MERGED_MODEL
        ),
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
)


# ============================================================
# 9. TOKENIZER SETUP
# ============================================================

if tokenizer.pad_token_id is None:

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


if tokenizer.eos_token is None:

    raise RuntimeError(
        "Tokenizer does not have an EOS token."
    )


# ============================================================
# 10. ATTACH FRESH SFT LORA
# ============================================================

print("\n" + "=" * 60)
print("ADDING FRESH SFT LORA")
print("=" * 60)

model = FastLanguageModel.get_peft_model(
    model,

    r=LORA_RANK,

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],

    lora_alpha=LORA_ALPHA,

    lora_dropout=LORA_DROPOUT,

    bias="none",

    use_gradient_checkpointing="unsloth",

    random_state=SEED,

    use_rslora=False,

    loftq_config=None,
)


model.print_trainable_parameters()


# ============================================================
# 11. FORMAT SFT DATA
#
# IMPORTANT:
#
# This matches the format you will use later for testing:
#
# ### Instruction:
# ...
#
# ### Question:
# ...
#
# ### Response:
# correct answer
#
# ============================================================

def format_example(example):

    instruction = (
        example["instruction"]
        .strip()
    )

    question = (
        example["input"]
        .strip()
    )

    answer = (
        example["output"]
        .strip()
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
        + tokenizer.eos_token
    )


    return {
        "prompt":
            prompt,

        "completion":
            completion,
    }


print("\n" + "=" * 60)
print("FORMATTING SFT DATA")
print("=" * 60)

train_dataset = train_dataset.map(
    format_example,
    remove_columns=(
        train_dataset.column_names
    ),
)


validation_dataset = (
    validation_dataset.map(
        format_example,
        remove_columns=(
            validation_dataset.column_names
        ),
    )
)


test_dataset = test_dataset.map(
    format_example,
    remove_columns=(
        test_dataset.column_names
    ),
)


# ============================================================
# 12. SHOW FORMATTED SAMPLE
# ============================================================

print("\n" + "=" * 60)
print("FORMATTED TRAINING SAMPLE")
print("=" * 60)

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


# ============================================================
# 13. TRAINING PLAN
# ============================================================

effective_batch_size = (
    TRAIN_BATCH_SIZE
    * GRADIENT_ACCUMULATION_STEPS
)


steps_per_epoch = math.ceil(
    len(train_dataset)
    / effective_batch_size
)


expected_total_steps = (
    steps_per_epoch
    * NUM_EPOCHS
)


warmup_steps = max(
    1,
    int(
        expected_total_steps
        * 0.05
    )
)


print("\n" + "=" * 60)
print("SFT TRAINING PLAN")
print("=" * 60)

print(
    "Train examples:",
    len(train_dataset),
)

print(
    "Validation examples:",
    len(validation_dataset),
)

print(
    "Test examples:",
    len(test_dataset),
)

print(
    "Epochs:",
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
    "Batch size:",
    TRAIN_BATCH_SIZE,
)

print(
    "Gradient accumulation:",
    GRADIENT_ACCUMULATION_STEPS,
)

print(
    "Effective batch size:",
    effective_batch_size,
)

print(
    "Steps per epoch:",
    steps_per_epoch,
)

print(
    "Expected total steps:",
    expected_total_steps,
)

print(
    "Warmup steps:",
    warmup_steps,
)


# ============================================================
# 14. TRAINING CONFIG
# ============================================================

training_args = SFTConfig(

    output_dir=str(
        CHECKPOINT_DIR
    ),

    max_length=MAX_SEQ_LENGTH,

    packing=False,

    completion_only_loss=True,


    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    per_device_train_batch_size=(
        TRAIN_BATCH_SIZE
    ),

    per_device_eval_batch_size=(
        EVAL_BATCH_SIZE
    ),

    gradient_accumulation_steps=(
        GRADIENT_ACCUMULATION_STEPS
    ),


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    num_train_epochs=(
        NUM_EPOCHS
    ),

    learning_rate=(
        LEARNING_RATE
    ),

    weight_decay=(
        WEIGHT_DECAY
    ),

    optim="adamw_8bit",

    lr_scheduler_type="cosine",

    warmup_steps=(
        warmup_steps
    ),


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    eval_strategy="epoch",


    # --------------------------------------------------------
    # Checkpoints
    # --------------------------------------------------------

    save_strategy="epoch",

    save_total_limit=3,

    load_best_model_at_end=True,

    metric_for_best_model=(
        "eval_loss"
    ),

    greater_is_better=False,


    # --------------------------------------------------------
    # Precision
    # --------------------------------------------------------

    bf16=(
        torch.cuda.is_bf16_supported()
    ),

    fp16=(
        not torch.cuda.is_bf16_supported()
    ),


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_steps=5,

    logging_first_step=True,

    report_to="none",


    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed=SEED,
)


# ============================================================
# 15. CREATE TRAINER
# ============================================================

trainer = SFTTrainer(
    model=model,

    args=training_args,

    train_dataset=(
        train_dataset
    ),

    eval_dataset=(
        validation_dataset
    ),

    processing_class=tokenizer,
)


# ============================================================
# 16. BASELINE VALIDATION
#
# This measures CPT + fresh LoRA BEFORE SFT updates.
# ============================================================

print("\n" + "=" * 60)
print("BASELINE VALIDATION BEFORE SFT")
print("=" * 60)

baseline_metrics = (
    trainer.evaluate()
)


baseline_validation_loss = (
    baseline_metrics[
        "eval_loss"
    ]
)


try:

    baseline_validation_perplexity = (
        math.exp(
            baseline_validation_loss
        )
    )

except OverflowError:

    baseline_validation_perplexity = (
        float("inf")
    )


print(
    f"Baseline validation loss: "
    f"{baseline_validation_loss:.4f}"
)

print(
    f"Baseline validation perplexity: "
    f"{baseline_validation_perplexity:.4f}"
)


# ============================================================
# 17. TRAIN
# ============================================================

print("\n" + "=" * 60)
print("STARTING SFT TRAINING")
print("=" * 60)

train_output = (
    trainer.train()
)


# ============================================================
# 18. BEST CHECKPOINT
# ============================================================

best_checkpoint = (
    trainer.state.best_model_checkpoint
)


best_validation_loss = (
    trainer.state.best_metric
)


print("\n" + "=" * 60)
print("BEST SFT CHECKPOINT")
print("=" * 60)

print(
    "Best checkpoint:",
    best_checkpoint,
)

print(
    "Best validation loss:",
    best_validation_loss,
)


# ============================================================
# 19. FINAL VALIDATION
#
# load_best_model_at_end=True means the trainer has restored
# the best validation checkpoint.
# ============================================================

print("\n" + "=" * 60)
print("FINAL SFT VALIDATION")
print("=" * 60)

final_validation_metrics = (
    trainer.evaluate()
)


final_validation_loss = (
    final_validation_metrics[
        "eval_loss"
    ]
)


try:

    final_validation_perplexity = (
        math.exp(
            final_validation_loss
        )
    )

except OverflowError:

    final_validation_perplexity = (
        float("inf")
    )


validation_loss_reduction = (
    baseline_validation_loss
    - final_validation_loss
)


if baseline_validation_loss > 0:

    validation_improvement_percent = (
        validation_loss_reduction
        / baseline_validation_loss
        * 100
    )

else:

    validation_improvement_percent = None


print(
    f"Before SFT loss : "
    f"{baseline_validation_loss:.4f}"
)

print(
    f"After SFT loss  : "
    f"{final_validation_loss:.4f}"
)

print(
    f"Final perplexity: "
    f"{final_validation_perplexity:.4f}"
)


if validation_improvement_percent is not None:

    print(
        f"Validation improvement: "
        f"{validation_improvement_percent:.2f}%"
    )


# ============================================================
# 20. SAVE SFT ADAPTER
#
# This saves ONLY the fresh SFT LoRA adapter.
# ============================================================

print("\n" + "=" * 60)
print("SAVING SFT ADAPTER")
print("=" * 60)

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


print(
    "SFT adapter saved to:"
)

print(
    ADAPTER_DIR
)


# ============================================================
# 21. SAVE METRICS
# ============================================================

METRICS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metrics = {

    "stage":
        "SFT",

    "starting_model":
        str(
            CPT_MERGED_MODEL
        ),

    "training_method":
        "QLoRA",

    "dataset_type":
        "prompt_completion",

    "completion_only_loss":
        True,

    "quantization":
        "4bit",

    "max_sequence_length":
        MAX_SEQ_LENGTH,

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

    "epochs":
        NUM_EPOCHS,

    "lora_rank":
        LORA_RANK,

    "lora_alpha":
        LORA_ALPHA,

    "lora_dropout":
        LORA_DROPOUT,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "train_batch_size":
        TRAIN_BATCH_SIZE,

    "gradient_accumulation_steps":
        GRADIENT_ACCUMULATION_STEPS,

    "effective_batch_size":
        effective_batch_size,

    "steps_per_epoch":
        steps_per_epoch,

    "expected_total_steps":
        expected_total_steps,

    "actual_global_steps":
        train_output.global_step,

    "best_checkpoint":
        best_checkpoint,

    "best_validation_loss":
        best_validation_loss,

    "baseline_validation_loss":
        baseline_validation_loss,

    "baseline_validation_perplexity":
        baseline_validation_perplexity,

    "final_validation_loss":
        final_validation_loss,

    "final_validation_perplexity":
        final_validation_perplexity,

    "validation_loss_reduction":
        validation_loss_reduction,

    "validation_improvement_percent":
        validation_improvement_percent,

    "training_history":
        trainer.state.log_history,
}


with METRICS_FILE.open(
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        metrics,
        file,
        indent=4,
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# 22. FINAL REPORT
# ============================================================

print("\n")
print("=" * 60)
print("SFT TRAINING COMPLETE")
print("=" * 60)

print(
    "\nStarting model:"
)

print(
    CPT_MERGED_MODEL
)

print(
    "\nBest checkpoint:"
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
    """
The SFT test split was NOT used for checkpoint selection.

Validation selected the best checkpoint.

Your current model lineage is:

Qwen2.5-0.5B
        ↓
CPT
        ↓
models/cpt_merged
        ↓
fresh SFT LoRA
        ↓
models/sft_adapter

The adapter is NOT merged yet.

Next step:

models/cpt_merged
        +
models/sft_adapter
        ↓
models/sft_merged
"""
)