from pathlib import Path
import json
import math

import torch
from datasets import load_dataset

from unsloth import FastLanguageModel

from trl import (
    SFTTrainer,
    SFTConfig,
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "Data"
    / "02_continued_pretraining"
)

TRAIN_FILE = (
    DATA_DIR
    / "train.jsonl"
)

VALIDATION_FILE = (
    DATA_DIR
    / "validation.jsonl"
)

TEST_FILE = (
    DATA_DIR
    / "test.jsonl"
)


MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

CHECKPOINT_DIR = (
    MODELS_DIR
    / "cpt_checkpoints"
)

ADAPTER_DIR = (
    MODELS_DIR
    / "cpt_adapter"
)


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "cpt"
)

METRICS_FILE = (
    RESULTS_DIR
    / "cpt_training_metrics.json"
)


# ============================================================
# 2. CREATE REQUIRED FOLDERS
# ============================================================

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 3. CONFIG
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-0.5B"

MAX_SEQ_LENGTH = 1024


# ------------------------------------------------------------
# LoRA
# ------------------------------------------------------------

LORA_RANK = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

NUM_EPOCHS = 3

TRAIN_BATCH_SIZE = 1

EVAL_BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 2e-4

WEIGHT_DECAY = 0.01

SEED = 42


# ============================================================
# 4. CHECK DATA FILES
# ============================================================

for file_path in [
    TRAIN_FILE,
    VALIDATION_FILE,
    TEST_FILE,
]:

    if not file_path.exists():

        raise FileNotFoundError(
            f"""
Required CPT data file not found:

{file_path}

Run the CPT data preparation script first.
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
# 6. LOAD CPT DATA
# ============================================================

print("\n" + "=" * 60)
print("LOADING CPT DATA")
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


train_dataset = dataset[
    "train"
]

validation_dataset = dataset[
    "validation"
]

test_dataset = dataset[
    "test"
]


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
        "Training dataset is empty."
    )


sample = train_dataset[0]


if "text" not in sample:

    raise RuntimeError(
        """
CPT data requires a field named:

text

Example:

{
    "text": "NexaFlow company policy text..."
}
"""
    )


# ============================================================
# 8. LOAD BASE MODEL IN 4-BIT
# ============================================================

print("\n" + "=" * 60)
print("LOADING BASE MODEL")
print("=" * 60)


model, tokenizer = (
    FastLanguageModel.from_pretrained(

        model_name=BASE_MODEL,

        max_seq_length=MAX_SEQ_LENGTH,

        dtype=None,

        load_in_4bit=True,
    )
)


# ============================================================
# 9. ATTACH FRESH LORA
# ============================================================

print("\n" + "=" * 60)
print("ADDING CPT LORA")
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
# 10. TRAINING PLAN
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


print("\n" + "=" * 60)
print("TRAINING PLAN")
print("=" * 60)


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
    "Train batch size:",
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


# ============================================================
# 11. SFT CONFIG
#
# We use SFTTrainer here for raw-text causal LM training.
# ============================================================

training_args = SFTConfig(

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_text_field="text",

    max_length=MAX_SEQ_LENGTH,

    packing=False,


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

    num_train_epochs=NUM_EPOCHS,

    learning_rate=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY,

    optim="adamw_8bit",

    lr_scheduler_type="cosine",


    # --------------------------------------------------------
    # Warmup
    #
    # Using explicit warmup_steps avoids the warning you saw
    # before about warmup_ratio.
    # --------------------------------------------------------

    warmup_steps=max(
        1,
        int(
            expected_total_steps
            * 0.05
        )
    ),


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    logging_steps=5,

    logging_first_step=True,

    report_to="none",


    # --------------------------------------------------------
    # Validation
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
    # Reproducibility
    # --------------------------------------------------------

    seed=SEED,


    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir=str(
        CHECKPOINT_DIR
    ),
)


# ============================================================
# 12. CREATE TRAINER
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
# 13. BASELINE VALIDATION BEFORE CPT
# ============================================================

print("\n" + "=" * 60)
print("BASELINE VALIDATION BEFORE CPT")
print("=" * 60)


baseline_metrics = (
    trainer.evaluate()
)


baseline_validation_loss = (
    baseline_metrics[
        "eval_loss"
    ]
)


baseline_validation_perplexity = (
    math.exp(
        baseline_validation_loss
    )
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
# 14. TRAIN CPT
# ============================================================

print("\n" + "=" * 60)
print("STARTING CPT TRAINING")
print("=" * 60)


train_output = (
    trainer.train()
)


# ============================================================
# 15. BEST CHECKPOINT
# ============================================================

best_checkpoint = (
    trainer.state.best_model_checkpoint
)

best_validation_loss = (
    trainer.state.best_metric
)


print("\n" + "=" * 60)
print("BEST CHECKPOINT")
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
# 16. FINAL VALIDATION
#
# Because load_best_model_at_end=True,
# trainer.model is now the best checkpoint.
# ============================================================

print("\n" + "=" * 60)
print("FINAL VALIDATION")
print("=" * 60)


final_validation_metrics = (
    trainer.evaluate()
)


final_validation_loss = (
    final_validation_metrics[
        "eval_loss"
    ]
)


final_validation_perplexity = (
    math.exp(
        final_validation_loss
    )
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
    f"Before CPT loss : "
    f"{baseline_validation_loss:.4f}"
)

print(
    f"After CPT loss  : "
    f"{final_validation_loss:.4f}"
)

print(
    f"Final perplexity: "
    f"{final_validation_perplexity:.4f}"
)


if (
    validation_improvement_percent
    is not None
):

    print(
        f"Validation improvement: "
        f"{validation_improvement_percent:.2f}%"
    )


# ============================================================
# 17. SAVE CPT ADAPTER
# ============================================================

print("\n" + "=" * 60)
print("SAVING CPT ADAPTER")
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
    "Adapter saved to:"
)

print(
    ADAPTER_DIR
)


# ============================================================
# 18. SAVE TRAINING METRICS
#
# IMPORTANT:
# Ensure folder still exists even if it was deleted during
# training.
# ============================================================

METRICS_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


metrics = {

    "stage":
        "CPT",

    "base_model":
        BASE_MODEL,

    "training_method":
        "QLoRA",

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

    "global_steps":
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
# 19. FINAL REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CPT TRAINING COMPLETE")
print("=" * 60)


print(
    "\nBest checkpoint:"
)

print(
    best_checkpoint
)


print(
    "\nCPT adapter:"
)

print(
    ADAPTER_DIR
)


print(
    "\nTraining metrics:"
)

print(
    METRICS_FILE
)


print(
    """
The CPT test set was NOT used for checkpoint selection.

Validation data selected the best checkpoint.

The held-out test and final evaluation benchmark should remain
untouched until model comparison.
"""
) 