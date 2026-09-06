from pathlib import Path
import gc
import shutil

import torch
from unsloth import FastLanguageModel


# ============================================================
# NEXAFLOW TECHNOLOGIES
# CPT MERGE SCRIPT
#
# Input:
#
#     Qwen/Qwen2.5-1.5B
#     +
#     models/cpt_adapter
#
# Output:
#
#     models/cpt_merged
#
# This merges the trained LoRA adapter into the base model
# and saves a standard 16-bit merged model for the next stage.
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

CPT_ADAPTER_DIR = (
    MODELS_DIR
    / "cpt_adapter"
)

CPT_MERGED_DIR = (
    MODELS_DIR
    / "cpt_merged"
)


# ============================================================
# 2. BASE MODEL
#
# This must match the model used during CPT training.
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-1.5B"

MAX_SEQ_LENGTH = 1024


# ============================================================
# 3. CONFIG
#
# True:
#     delete an old cpt_merged folder before creating a new one
#
# False:
#     fail if an old output already exists
# ============================================================

OVERWRITE_EXISTING_MERGED_MODEL = True


# ============================================================
# 4. MEMORY CLEANUP
# ============================================================

def cleanup_memory():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# 5. CHECK GPU
# ============================================================

def check_gpu():

    print("\n" + "=" * 60)
    print("GPU CHECK")
    print("=" * 60)

    if not torch.cuda.is_available():

        raise RuntimeError(
            """
CUDA GPU was not detected.

The CPT merge step should be run with your CUDA-enabled
PyTorch environment.
"""
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
        f"VRAM: {gpu_memory:.2f} GB"
    )


# ============================================================
# 6. CHECK CPT ADAPTER
# ============================================================

def validate_adapter():

    print("\n" + "=" * 60)
    print("CHECKING CPT ADAPTER")
    print("=" * 60)

    if not CPT_ADAPTER_DIR.exists():

        raise FileNotFoundError(
            f"""
CPT adapter directory does not exist:

{CPT_ADAPTER_DIR}

You must finish CPT training first.
"""
        )

    adapter_config = (
        CPT_ADAPTER_DIR
        / "adapter_config.json"
    )

    if not adapter_config.exists():

        raise FileNotFoundError(
            f"""
adapter_config.json was not found:

{adapter_config}

The CPT adapter may not have been saved correctly.
"""
        )

    print(
        "CPT adapter found:"
    )

    print(
        CPT_ADAPTER_DIR
    )


# ============================================================
# 7. PREPARE OUTPUT DIRECTORY
# ============================================================

def prepare_output_directory():

    print("\n" + "=" * 60)
    print("PREPARING MERGED OUTPUT")
    print("=" * 60)

    if CPT_MERGED_DIR.exists():

        if OVERWRITE_EXISTING_MERGED_MODEL:

            print(
                "Removing old merged model:"
            )

            print(
                CPT_MERGED_DIR
            )

            shutil.rmtree(
                CPT_MERGED_DIR
            )

        else:

            raise RuntimeError(
                f"""
Merged model directory already exists:

{CPT_MERGED_DIR}

Set:

OVERWRITE_EXISTING_MERGED_MODEL = True

if you intentionally want to replace it.
"""
            )

    CPT_MERGED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# 8. LOAD CPT ADAPTER MODEL
#
# Important:
#
# We load from the adapter directory.
#
# adapter_config.json tells PEFT / Unsloth which base model
# the adapter belongs to.
#
# load_in_4bit=False because we are preparing a merged
# 16-bit model.
# ============================================================

def load_cpt_adapter_model():

    print("\n" + "=" * 60)
    print("LOADING CPT ADAPTER MODEL")
    print("=" * 60)

    cleanup_memory()

    try:

        model, tokenizer = (
            FastLanguageModel.from_pretrained(

                model_name=str(
                    CPT_ADAPTER_DIR
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
CUDA out-of-memory while loading the CPT adapter for merging.

Before changing the code:

1. Close other GPU applications.
2. Restart the Python process.
3. Make sure no other training script is still running.

Original CUDA error:
"""
            + str(error)
        ) from error

    print(
        "CPT adapter loaded successfully."
    )

    return (
        model,
        tokenizer,
    )


# ============================================================
# 9. SAVE MERGED MODEL
#
# merged_16bit means:
#
#     Base model weights
#         +
#     CPT LoRA delta
#         =
#     merged standard model
#
# This is what we use as the starting point for SFT.
# ============================================================

def merge_and_save(
    model,
    tokenizer,
):

    print("\n" + "=" * 60)
    print("MERGING CPT ADAPTER INTO BASE MODEL")
    print("=" * 60)

    cleanup_memory()

    try:

        model.save_pretrained_merged(

            str(
                CPT_MERGED_DIR
            ),

            tokenizer,

            save_method="merged_16bit",
        )

    except torch.cuda.OutOfMemoryError as error:

        cleanup_memory()

        raise RuntimeError(
            """
CUDA out-of-memory during CPT merge.

Try:

1. Close other GPU applications.
2. Restart the merge command.
3. Make sure the previous trainer process has completely exited.

Original CUDA error:
"""
            + str(error)
        ) from error


# ============================================================
# 10. VERIFY MERGED MODEL
# ============================================================

def verify_output():

    print("\n" + "=" * 60)
    print("VERIFYING MERGED MODEL")
    print("=" * 60)

    required_files = [

        CPT_MERGED_DIR
        / "config.json",

    ]

    for path in required_files:

        if not path.exists():

            raise RuntimeError(
                f"""
Merged model appears incomplete.

Missing required file:

{path}
"""
            )

    model_files = list(
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
        not model_files
        and not index_files
    ):

        raise RuntimeError(
            f"""
No model weight files were found in:

{CPT_MERGED_DIR}
"""
        )

    print(
        "Merged config found."
    )

    print(
        "Model weight files found:",
        len(model_files),
    )

    print(
        "Index files found:",
        len(index_files),
    )

    print(
        "\nMerged model verification: PASSED"
    )


# ============================================================
# 11. OPTIONAL LOAD TEST
#
# This verifies that the merged model can be loaded again.
#
# We reload it in 4-bit because the next SFT stage will also
# use 4-bit QLoRA.
# ============================================================

def test_reload():

    print("\n" + "=" * 60)
    print("TESTING MERGED MODEL RELOAD")
    print("=" * 60)

    cleanup_memory()

    try:

        test_model, test_tokenizer = (
            FastLanguageModel.from_pretrained(

                model_name=str(
                    CPT_MERGED_DIR
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
The CPT merged model was saved, but reloading it failed.

Original error:

"""
            + str(error)
        ) from error

    print(
        "Merged model reload: PASSED"
    )

    del test_model
    del test_tokenizer

    cleanup_memory()


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("NEXAFLOW CPT MERGE")
    print("=" * 60)

    print(
        "\nBase model:"
    )

    print(
        BASE_MODEL
    )

    print(
        "\nCPT adapter:"
    )

    print(
        CPT_ADAPTER_DIR
    )

    print(
        "\nMerged output:"
    )

    print(
        CPT_MERGED_DIR
    )


    # --------------------------------------------------------
    # Checks
    # --------------------------------------------------------

    check_gpu()

    validate_adapter()

    prepare_output_directory()


    # --------------------------------------------------------
    # Load adapter
    # --------------------------------------------------------

    (
        model,
        tokenizer,
    ) = load_cpt_adapter_model()


    # --------------------------------------------------------
    # Merge + save
    # --------------------------------------------------------

    merge_and_save(
        model,
        tokenizer,
    )


    # --------------------------------------------------------
    # Free merge model before verification reload
    # --------------------------------------------------------

    del model
    del tokenizer

    cleanup_memory()


    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    verify_output()


    # --------------------------------------------------------
    # Test actual reload
    # --------------------------------------------------------

    test_reload()


    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("CPT MERGE COMPLETE")
    print("=" * 60)

    print(
        "\nMerged CPT model:"
    )

    print(
        CPT_MERGED_DIR
    )

    print(
        """
Current pipeline:

Qwen/Qwen2.5-1.5B Base
        +
CPT LoRA adapter
        ↓
MERGED
        ↓
models/cpt_merged


NEXT STAGE:

models/cpt_merged
        ↓
load in 4-bit
        ↓
attach a FRESH SFT LoRA
        ↓
train on Data/03_sft
        ↓
models/sft_adapter

IMPORTANT:

Do NOT reuse the CPT LoRA adapter as the SFT adapter.

SFT must attach a fresh LoRA adapter to:

models/cpt_merged
"""
    )


if __name__ == "__main__":

    main()