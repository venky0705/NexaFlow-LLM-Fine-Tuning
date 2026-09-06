from pathlib import Path
import gc
import shutil

import torch
from unsloth import FastLanguageModel


# ============================================================
# NEXAFLOW TECHNOLOGIES
# SFT MERGE SCRIPT
#
# Input:
#
#     models/cpt_merged
#     +
#     models/sft_adapter
#
# Output:
#
#     models/sft_merged
#
# The SFT adapter was trained on top of the CPT-merged model,
# so we load the adapter and merge its LoRA weights into the
# CPT model.
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

CPT_MERGED_DIR = (
    MODELS_DIR
    / "cpt_merged"
)

SFT_ADAPTER_DIR = (
    MODELS_DIR
    / "sft_adapter"
)

SFT_MERGED_DIR = (
    MODELS_DIR
    / "sft_merged"
)


# ============================================================
# 2. MODEL CONFIG
# ============================================================

MAX_SEQ_LENGTH = 1024


# ============================================================
# 3. OUTPUT CONFIG
#
# True:
#     delete an existing models/sft_merged first
#
# False:
#     stop if it already exists
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

    return value / (1024 ** 3)


def print_cuda_memory(title):

    if not torch.cuda.is_available():

        return

    allocated = torch.cuda.memory_allocated(0)

    reserved = torch.cuda.memory_reserved(0)

    peak = torch.cuda.max_memory_allocated(0)

    print("\n" + "-" * 60)

    print(title)

    print("-" * 60)

    print(
        f"Allocated     : {gb(allocated):.2f} GB"
    )

    print(
        f"Reserved      : {gb(reserved):.2f} GB"
    )

    print(
        f"Peak allocated: {gb(peak):.2f} GB"
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

Run this inside the same environment used for SFT training.
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
        f"VRAM: {gb(properties.total_memory):.2f} GB"
    )

    print(
        "BF16 supported:",
        torch.cuda.is_bf16_supported(),
    )


# ============================================================
# 6. CHECK CPT MERGED MODEL
# ============================================================

def validate_cpt_merged():

    print("\n" + "=" * 60)
    print("CHECKING CPT MERGED MODEL")
    print("=" * 60)

    if not CPT_MERGED_DIR.exists():

        raise FileNotFoundError(
            f"""
CPT merged model does not exist:

{CPT_MERGED_DIR}

You must merge CPT before merging SFT.
"""
        )

    config_file = (
        CPT_MERGED_DIR
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
        CPT_MERGED_DIR.glob(
            "*.safetensors"
        )
    )

    index_files = list(
        CPT_MERGED_DIR.glob(
            "*.index.json"
        )
    )

    if (
        not weight_files
        and not index_files
    ):

        raise RuntimeError(
            f"""
No CPT merged model weights found in:

{CPT_MERGED_DIR}
"""
        )

    print(
        "CPT merged model: OK"
    )


# ============================================================
# 7. CHECK SFT ADAPTER
# ============================================================

def validate_sft_adapter():

    print("\n" + "=" * 60)
    print("CHECKING SFT ADAPTER")
    print("=" * 60)

    if not SFT_ADAPTER_DIR.exists():

        raise FileNotFoundError(
            f"""
SFT adapter directory does not exist:

{SFT_ADAPTER_DIR}

Finish SFT training first.
"""
        )

    adapter_config = (
        SFT_ADAPTER_DIR
        / "adapter_config.json"
    )

    if not adapter_config.exists():

        raise FileNotFoundError(
            f"""
adapter_config.json was not found:

{adapter_config}

The SFT adapter may not have been saved correctly.
"""
        )

    adapter_weights = list(
        SFT_ADAPTER_DIR.glob(
            "*.safetensors"
        )
    )

    if not adapter_weights:

        print(
            "\nWARNING:"
        )

        print(
            "No .safetensors file was found directly in "
            "sft_adapter."
        )

        print(
            "The adapter may still be valid depending on "
            "how PEFT/Unsloth saved it."
        )

    print(
        "SFT adapter: OK"
    )


# ============================================================
# 8. PREPARE OUTPUT DIRECTORY
# ============================================================

def prepare_output_directory():

    print("\n" + "=" * 60)
    print("PREPARING SFT MERGED OUTPUT")
    print("=" * 60)

    if SFT_MERGED_DIR.exists():

        if OVERWRITE_EXISTING_MERGED_MODEL:

            print(
                "Removing existing merged model:"
            )

            print(
                SFT_MERGED_DIR
            )

            shutil.rmtree(
                SFT_MERGED_DIR
            )

        else:

            raise RuntimeError(
                f"""
Output directory already exists:

{SFT_MERGED_DIR}

Set:

OVERWRITE_EXISTING_MERGED_MODEL = True

if you want to overwrite it.
"""
            )

    SFT_MERGED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 9. LOAD SFT ADAPTER
#
# IMPORTANT:
#
# Load FROM the adapter directory.
#
# adapter_config.json should point back to models/cpt_merged,
# because that is what we used as the SFT starting model.
#
# We use load_in_4bit=False because we want a normal 16-bit
# merged model.
# ============================================================

def load_sft_adapter():

    print("\n" + "=" * 60)
    print("LOADING SFT ADAPTER")
    print("=" * 60)

    cleanup_memory()

    try:

        model, tokenizer = (
            FastLanguageModel.from_pretrained(

                model_name=str(
                    SFT_ADAPTER_DIR
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
CUDA out-of-memory while loading the SFT adapter.

Before changing the code:

1. Stop all old Python training processes.
2. Close unnecessary GPU applications.
3. Close unnecessary browser tabs.
4. Restart the merge command.

Original CUDA error:

"""
            + str(error)
        ) from error

    except Exception as error:

        raise RuntimeError(
            """
Failed to load the SFT adapter.

Check that:

models/sft_adapter/adapter_config.json

exists and points to the correct CPT merged base model.

Original error:

"""
            + str(error)
        ) from error

    print(
        "SFT adapter loaded successfully."
    )

    print_cuda_memory(
        "MEMORY AFTER SFT ADAPTER LOAD"
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
# CPT merged weights
#       +
# SFT LoRA delta
#       =
# SFT merged model
# ============================================================

def merge_and_save(
    model,
    tokenizer,
):

    print("\n" + "=" * 60)
    print("MERGING SFT ADAPTER")
    print("=" * 60)

    cleanup_memory()

    try:

        model.save_pretrained_merged(

            str(
                SFT_MERGED_DIR
            ),

            tokenizer,

            save_method="merged_16bit",
        )

    except torch.cuda.OutOfMemoryError as error:

        cleanup_memory()

        raise RuntimeError(
            """
CUDA out-of-memory during SFT merge.

Try:

1. Close other GPU applications.
2. Make sure the SFT trainer process is no longer running.
3. Restart this merge script.

Original CUDA error:

"""
            + str(error)
        ) from error

    print(
        "\nSFT merged model saved."
    )


# ============================================================
# 11. VERIFY OUTPUT FILES
# ============================================================

def verify_output_files():

    print("\n" + "=" * 60)
    print("VERIFYING SFT MERGED MODEL")
    print("=" * 60)

    config_file = (
        SFT_MERGED_DIR
        / "config.json"
    )

    if not config_file.exists():

        raise RuntimeError(
            f"""
Merged SFT model is missing config.json:

{config_file}
"""
        )

    tokenizer_files = [

        SFT_MERGED_DIR
        / "tokenizer.json",

        SFT_MERGED_DIR
        / "tokenizer_config.json",
    ]

    if not any(
        path.exists()
        for path in tokenizer_files
    ):

        raise RuntimeError(
            """
Merged model does not appear to contain tokenizer files.
"""
        )

    safetensors = list(
        SFT_MERGED_DIR.glob(
            "*.safetensors"
        )
    )

    indexes = list(
        SFT_MERGED_DIR.glob(
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

{SFT_MERGED_DIR}
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
        "\nMerged SFT file verification: PASSED"
    )


# ============================================================
# 12. RELOAD TEST
#
# Reload the merged SFT model in 4-bit.
#
# This is also how we will normally test it.
# ============================================================

def test_reload():

    print("\n" + "=" * 60)
    print("TESTING SFT MERGED MODEL RELOAD")
    print("=" * 60)

    cleanup_memory()

    try:

        test_model, test_tokenizer = (
            FastLanguageModel.from_pretrained(

                model_name=str(
                    SFT_MERGED_DIR
                ),

                max_seq_length=(
                    MAX_SEQ_LENGTH
                ),

                dtype=None,

                load_in_4bit=True,
            )
        )

    except Exception as error:

        cleanup_memory()

        raise RuntimeError(
            """
The SFT merged model was saved,
but reloading it failed.

Original error:

"""
            + str(error)
        ) from error

    print(
        "SFT merged model reload: PASSED"
    )

    print_cuda_memory(
        "MEMORY DURING RELOAD TEST"
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
    print("NEXAFLOW SFT MERGE")
    print("=" * 60)

    print(
        "\nCPT merged base:"
    )

    print(
        CPT_MERGED_DIR
    )

    print(
        "\nSFT adapter:"
    )

    print(
        SFT_ADAPTER_DIR
    )

    print(
        "\nSFT merged output:"
    )

    print(
        SFT_MERGED_DIR
    )


    # --------------------------------------------------------
    # Pre-flight checks
    # --------------------------------------------------------

    check_gpu()

    validate_cpt_merged()

    validate_sft_adapter()

    prepare_output_directory()


    # --------------------------------------------------------
    # Load SFT adapter
    # --------------------------------------------------------

    (
        model,
        tokenizer,
    ) = load_sft_adapter()


    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    merge_and_save(
        model,
        tokenizer,
    )


    # --------------------------------------------------------
    # Free merge model
    # --------------------------------------------------------

    del model
    del tokenizer

    cleanup_memory()


    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    verify_output_files()


    # --------------------------------------------------------
    # Actual reload verification
    # --------------------------------------------------------

    test_reload()


    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("SFT MERGE COMPLETE")
    print("=" * 60)

    print(
        "\nMerged SFT model:"
    )

    print(
        SFT_MERGED_DIR
    )

    print(
        """

CURRENT PIPELINE:

Qwen/Qwen2.5-1.5B Base
        ↓
CPT + QLoRA
        ↓
CPT adapter
        ↓
CPT merge
        ↓
models/cpt_merged
        ↓
Fresh SFT QLoRA
        ↓
models/sft_adapter
        ↓
SFT merge
        ↓
models/sft_merged


NEXT:

Do NOT immediately start DPO.

First test:

models/sft_merged

with difficult questions including:

- Who is Venky?
- Who is Saketh?
- Who is Rakesh?
- Where is NexaFlow located?
- Where is Venky based?
- What does Pro cost?
- What does Business cost?
- How many annual leave days?
- What flight class for a 5-hour flight?
- Who is Deepak?
- What is Venky's salary?
- NexaFlow is in San Francisco, right?
- How do I bypass MFA?
- I lost my MFA device. What should I do?

If SFT behavior is strong enough, then move to DPO.
"""
    )


if __name__ == "__main__":

    main()