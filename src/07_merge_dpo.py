from pathlib import Path
import gc
import shutil

import torch

# IMPORTANT:
# Import Unsloth before transformers / PEFT / TRL-related imports.
import unsloth
from unsloth import FastLanguageModel


# ============================================================
# NEXAFLOW TECHNOLOGIES
# DPO MERGE SCRIPT
#
# Input:
#
#     models/sft_merged
#     +
#     models/dpo_adapter
#
# Output:
#
#     models/dpo_merged
#
# Meaning:
#
# Base
#   +
# CPT
#   +
# SFT
#   +
# DPO
#
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

SFT_MERGED_DIR = (
    MODELS_DIR
    / "sft_merged"
)

DPO_ADAPTER_DIR = (
    MODELS_DIR
    / "dpo_adapter"
)

DPO_MERGED_DIR = (
    MODELS_DIR
    / "dpo_merged"
)


# ============================================================
# 2. MODEL CONFIG
# ============================================================

MAX_SEQ_LENGTH = 1024


# ============================================================
# 3. OUTPUT CONFIG
#
# True:
#     replace existing dpo_merged
#
# False:
#     fail if dpo_merged already exists
# ============================================================

OVERWRITE_EXISTING_MERGED_MODEL = True


# ============================================================
# 4. MEMORY HELPERS
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

    print("\n" + "-" * 60)
    print(title)
    print("-" * 60)

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
# 5. GPU CHECK
# ============================================================

def check_gpu():

    print("\n" + "=" * 60)
    print("GPU CHECK")
    print("=" * 60)

    if not torch.cuda.is_available():

        raise RuntimeError(
            """
CUDA GPU was not detected.

Run this script inside the same CUDA-enabled environment
used for DPO training.
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
# 6. CHECK SFT MERGED MODEL
#
# DPO was trained ON TOP OF this model.
# ============================================================

def validate_sft_merged():

    print("\n" + "=" * 60)
    print("CHECKING SFT MERGED MODEL")
    print("=" * 60)

    if not SFT_MERGED_DIR.exists():

        raise FileNotFoundError(
            f"""
SFT merged model does not exist:

{SFT_MERGED_DIR}

DPO adapter must be merged back into the same model
that DPO training started from.
"""
        )

    config_file = (
        SFT_MERGED_DIR
        / "config.json"
    )

    if not config_file.exists():

        raise FileNotFoundError(
            f"""
Missing config.json:

{config_file}
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
No SFT merged model weights found in:

{SFT_MERGED_DIR}
"""
        )

    print(
        "SFT merged model: OK"
    )


# ============================================================
# 7. CHECK DPO ADAPTER
# ============================================================

def validate_dpo_adapter():

    print("\n" + "=" * 60)
    print("CHECKING DPO ADAPTER")
    print("=" * 60)

    if not DPO_ADAPTER_DIR.exists():

        raise FileNotFoundError(
            f"""
DPO adapter directory does not exist:

{DPO_ADAPTER_DIR}

Finish DPO training first.
"""
        )

    adapter_config = (
        DPO_ADAPTER_DIR
        / "adapter_config.json"
    )

    if not adapter_config.exists():

        raise FileNotFoundError(
            f"""
adapter_config.json was not found:

{adapter_config}

The DPO adapter may not have been saved correctly.
"""
        )

    adapter_weights = list(
        DPO_ADAPTER_DIR.glob(
            "*.safetensors"
        )
    )

    print(
        "DPO adapter config: OK"
    )

    print(
        "Adapter safetensor files:",
        len(
            adapter_weights
        ),
    )


# ============================================================
# 8. PREPARE OUTPUT DIRECTORY
# ============================================================

def prepare_output_directory():

    print("\n" + "=" * 60)
    print("PREPARING DPO MERGED OUTPUT")
    print("=" * 60)

    if DPO_MERGED_DIR.exists():

        if OVERWRITE_EXISTING_MERGED_MODEL:

            print(
                "Removing existing merged model:"
            )

            print(
                DPO_MERGED_DIR
            )

            shutil.rmtree(
                DPO_MERGED_DIR
            )

        else:

            raise RuntimeError(
                f"""
Output directory already exists:

{DPO_MERGED_DIR}

Set:

OVERWRITE_EXISTING_MERGED_MODEL = True

if you intentionally want to overwrite it.
"""
            )

    DPO_MERGED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 9. LOAD DPO ADAPTER MODEL
#
# Important:
#
# We load FROM:
#
# models/dpo_adapter
#
# adapter_config.json should identify the model used during
# DPO training, which should ultimately be models/sft_merged.
#
# load_in_4bit=False because the merge output should be a
# normal 16-bit merged model.
# ============================================================

def load_dpo_adapter():

    print("\n" + "=" * 60)
    print("LOADING DPO ADAPTER MODEL")
    print("=" * 60)

    cleanup_memory()

    try:

        model, tokenizer = (
            FastLanguageModel.from_pretrained(

                model_name=str(
                    DPO_ADAPTER_DIR
                ),

                max_seq_length=(
                    MAX_SEQ_LENGTH
                ),

                dtype=None,

                load_in_4bit=False,
            )
        )

    except torch.cuda.OutOfMemoryError as error:

        cleanup_memory()

        raise RuntimeError(
            """
CUDA out-of-memory while loading the DPO adapter for merge.

Try:

1. Make sure DPO training has completely stopped.
2. Close unnecessary GPU applications.
3. Close unnecessary browser tabs.
4. Restart this merge script.

Original CUDA error:

"""
            + str(error)
        ) from error

    except Exception as error:

        raise RuntimeError(
            """
Failed to load models/dpo_adapter.

Check that:

models/dpo_adapter/adapter_config.json

exists and that the adapter was saved successfully after DPO.

Original error:

"""
            + str(error)
        ) from error

    print(
        "DPO adapter loaded successfully."
    )

    print_cuda_memory(
        "MEMORY AFTER DPO ADAPTER LOAD"
    )

    return (
        model,
        tokenizer,
    )


# ============================================================
# 10. MERGE + SAVE
#
# This performs:
#
# SFT merged weights
#       +
# DPO LoRA delta
#       =
# final DPO merged model
# ============================================================

def merge_and_save(
    model,
    tokenizer,
):

    print("\n" + "=" * 60)
    print("MERGING DPO ADAPTER")
    print("=" * 60)

    cleanup_memory()

    try:

        model.save_pretrained_merged(

            str(
                DPO_MERGED_DIR
            ),

            tokenizer,

            save_method="merged_16bit",
        )

    except torch.cuda.OutOfMemoryError as error:

        cleanup_memory()

        raise RuntimeError(
            """
CUDA out-of-memory during DPO merge.

Try:

1. Stop any old Python processes.
2. Close GPU-heavy applications.
3. Restart the merge script.

Original CUDA error:

"""
            + str(error)
        ) from error

    print(
        "\nDPO merged model saved successfully."
    )


# ============================================================
# 11. VERIFY SAVED FILES
# ============================================================

def verify_output_files():

    print("\n" + "=" * 60)
    print("VERIFYING DPO MERGED MODEL")
    print("=" * 60)

    config_file = (
        DPO_MERGED_DIR
        / "config.json"
    )

    if not config_file.exists():

        raise RuntimeError(
            f"""
Merged DPO model is missing config.json:

{config_file}
"""
        )

    tokenizer_candidates = [

        DPO_MERGED_DIR
        / "tokenizer.json",

        DPO_MERGED_DIR
        / "tokenizer_config.json",
    ]

    if not any(
        path.exists()
        for path in tokenizer_candidates
    ):

        raise RuntimeError(
            """
No tokenizer files were found in the merged DPO model.
"""
        )

    safetensors = list(
        DPO_MERGED_DIR.glob(
            "*.safetensors"
        )
    )

    indexes = list(
        DPO_MERGED_DIR.glob(
            "*.index.json"
        )
    )

    if (
        not safetensors
        and not indexes
    ):

        raise RuntimeError(
            f"""
No merged model weights were found in:

{DPO_MERGED_DIR}
"""
        )

    print(
        "config.json: OK"
    )

    print(
        "Tokenizer files: OK"
    )

    print(
        "Safetensor files:",
        len(
            safetensors
        ),
    )

    print(
        "Index files:",
        len(
            indexes
        ),
    )

    print(
        "\nDPO merged file verification: PASSED"
    )


# ============================================================
# 12. RELOAD TEST
#
# Reload final DPO merged model in 4-bit.
#
# This confirms:
#
# - weights are readable
# - tokenizer is available
# - Unsloth can load it
#
# This is also the mode we will use for chat/evaluation.
# ============================================================

def test_reload():

    print("\n" + "=" * 60)
    print("TESTING DPO MERGED MODEL RELOAD")
    print("=" * 60)

    cleanup_memory()

    try:

        test_model, test_tokenizer = (
            FastLanguageModel.from_pretrained(

                model_name=str(
                    DPO_MERGED_DIR
                ),

                max_seq_length=(
                    MAX_SEQ_LENGTH
                ),

                dtype=None,

                load_in_4bit=True,
            )
        )

    except torch.cuda.OutOfMemoryError as error:

        cleanup_memory()

        raise RuntimeError(
            """
The merged model was written, but the 4-bit reload test
ran out of GPU memory.

Close other GPU applications and retry.

Original CUDA error:

"""
            + str(error)
        ) from error

    except Exception as error:

        cleanup_memory()

        raise RuntimeError(
            """
The DPO merged model was saved,
but reloading it failed.

Original error:

"""
            + str(error)
        ) from error

    print(
        "DPO merged model reload: PASSED"
    )

    print_cuda_memory(
        "MEMORY DURING DPO RELOAD TEST"
    )

    del test_model
    del test_tokenizer

    cleanup_memory()


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("NEXAFLOW DPO MERGE")
    print("=" * 60)

    print(
        "\nSFT merged starting model:"
    )

    print(
        SFT_MERGED_DIR
    )

    print(
        "\nDPO adapter:"
    )

    print(
        DPO_ADAPTER_DIR
    )

    print(
        "\nFinal DPO merged output:"
    )

    print(
        DPO_MERGED_DIR
    )


    # --------------------------------------------------------
    # Pre-flight
    # --------------------------------------------------------

    check_gpu()

    validate_sft_merged()

    validate_dpo_adapter()

    prepare_output_directory()


    # --------------------------------------------------------
    # Load DPO adapter
    # --------------------------------------------------------

    (
        model,
        tokenizer,
    ) = load_dpo_adapter()


    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merge_and_save(
        model,
        tokenizer,
    )


    # --------------------------------------------------------
    # Free model before reload
    # --------------------------------------------------------

    del model
    del tokenizer

    cleanup_memory()


    # --------------------------------------------------------
    # Verify output files
    # --------------------------------------------------------

    verify_output_files()


    # --------------------------------------------------------
    # Reload final model
    # --------------------------------------------------------

    test_reload()


    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("DPO MERGE COMPLETE")
    print("=" * 60)

    print(
        "\nFinal model:"
    )

    print(
        DPO_MERGED_DIR
    )

    print(
        """

FINAL TRAINING PIPELINE:

Qwen/Qwen2.5-1.5B Base
        ↓
CPT + QLoRA
        ↓
models/cpt_adapter
        ↓
MERGE
        ↓
models/cpt_merged
        ↓
SFT + fresh QLoRA
        ↓
models/sft_adapter
        ↓
MERGE
        ↓
models/sft_merged
        ↓
DPO + fresh QLoRA
        ↓
models/dpo_adapter
        ↓
MERGE
        ↓
models/dpo_merged


FINAL MODEL:

models/dpo_merged


NEXT STEP:

Do NOT train this model again yet.

Next we should:

1. Chat-test models/dpo_merged
2. Compare SFT vs DPO behavior
3. Run the frozen Data/05_evaluation benchmark
4. Compare:
       Base
       CPT
       SFT
       DPO
5. Calculate hallucination / refusal / policy accuracy metrics
"""
    )


if __name__ == "__main__":

    main()