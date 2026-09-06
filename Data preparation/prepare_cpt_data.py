from pathlib import Path
import json
import random
import re
from collections import Counter

from transformers import AutoTokenizer


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_ROOT
    / "Data"
    / "01_source_documents"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "Data"
    / "02_continued_pretraining"
)

TRAIN_FILE = (
    OUTPUT_DIR
    / "train.jsonl"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "validation.jsonl"
)

TEST_FILE = (
    OUTPUT_DIR
    / "test.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "dataset_summary.json"
)


# ============================================================
# 2. MODEL TOKENIZER
#
# IMPORTANT:
#
# We are now using Qwen2.5-1.5B Base for training.
#
# For data preparation, we only use its TOKENIZER.
#
# We are NOT loading the full 1.5B neural-network model here.
# ============================================================

TOKENIZER_MODEL = "Qwen/Qwen2.5-1.5B"


# ============================================================
# 3. DATA CONFIG
# ============================================================

SEED = 42

TRAIN_RATIO = 0.80

VALIDATION_RATIO = 0.10

TEST_RATIO = 0.10


# ============================================================
# 4. TOKEN LIMITS
#
# Training max sequence length will be:
#
# 1024 tokens
#
# We use chunks of about 900 tokens so there is some safety
# margin.
# ============================================================

MAX_SEQUENCE_LENGTH = 1024

TARGET_CHUNK_TOKENS = 900

MIN_LAST_CHUNK_TOKENS = 100


# ============================================================
# 5. CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 6. LOAD TOKENIZER
# ============================================================

print("\n" + "=" * 60)
print("LOADING TOKENIZER")
print("=" * 60)

print(
    "Tokenizer:",
    TOKENIZER_MODEL,
)

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_MODEL,
    use_fast=True,
)

print(
    "Tokenizer loaded successfully."
)


# ============================================================
# 7. BASIC TEXT NORMALIZATION
# ============================================================

def clean_whitespace(text):

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
    )

    # Remove excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    # Remove trailing spaces
    lines = [
        line.rstrip()
        for line in text.splitlines()
    ]

    text = "\n".join(
        lines
    )

    return text.strip()


# ============================================================
# 8. REMOVE TECHNICAL METADATA
#
# Source documents may have internal document metadata.
#
# We keep useful company/policy content but remove obvious
# technical dataset metadata.
# ============================================================

TECHNICAL_METADATA_FIELDS = [

    "Document ID",
    "Document type",
    "Document Type",
    "Category",
    "Version",
    "Status",
    "Created",
    "Updated",
    "Owner",
    "Synthetic",
    "Dataset",
]


def remove_front_metadata(text):

    lines = text.splitlines()

    cleaned_lines = []

    metadata_phase = True


    for line in lines:

        stripped = line.strip()


        # ----------------------------------------------------
        # Keep Markdown heading
        # ----------------------------------------------------

        if stripped.startswith("# "):

            cleaned_lines.append(
                line
            )

            continue


        if metadata_phase:

            is_metadata = False


            for field in (
                TECHNICAL_METADATA_FIELDS
            ):

                if re.match(
                    rf"^{re.escape(field)}\s*:",
                    stripped,
                    flags=re.IGNORECASE,
                ):

                    is_metadata = True

                    break


            if is_metadata:

                continue


            # Empty lines near metadata can be skipped
            if stripped == "":

                continue


            metadata_phase = False


        cleaned_lines.append(
            line
        )


    return "\n".join(
        cleaned_lines
    ).strip()


# ============================================================
# 9. REMOVE DATASET NOTICE SECTION
#
# We don't want repetitive text like:
#
# "This is synthetic data..."
#
# appearing in every training document.
# ============================================================

def remove_dataset_notice(text):

    pattern = (
        r"\n## Dataset notice"
        r".*?"
        r"(?=\n## |\Z)"
    )

    text = re.sub(
        pattern,
        "",
        text,
        flags=(
            re.DOTALL
            | re.IGNORECASE
        ),
    )

    return text.strip()


# ============================================================
# 10. FINAL DOCUMENT CLEANING
# ============================================================

def clean_document(text):

    text = clean_whitespace(
        text
    )

    text = remove_front_metadata(
        text
    )

    text = remove_dataset_notice(
        text
    )

    text = clean_whitespace(
        text
    )

    return text


# ============================================================
# 11. TOKEN COUNT
# ============================================================

def count_tokens(text):

    tokens = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]

    return len(tokens)


# ============================================================
# 12. TOKEN-AWARE CHUNKING
#
# We chunk directly by tokenizer IDs.
#
# This guarantees chunk sizes are based on the actual
# Qwen2.5-1.5B tokenizer.
# ============================================================

def chunk_text_by_tokens(text):

    token_ids = tokenizer(
        text,
        add_special_tokens=False,
    )["input_ids"]


    if len(token_ids) <= TARGET_CHUNK_TOKENS:

        return [
            text
        ]


    chunks = []


    for start in range(
        0,
        len(token_ids),
        TARGET_CHUNK_TOKENS,
    ):

        chunk_ids = token_ids[
            start:
            start + TARGET_CHUNK_TOKENS
        ]


        chunk_text = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True,
        ).strip()


        if chunk_text:

            chunks.append(
                chunk_text
            )


    # ========================================================
    # If last chunk is tiny, merge it with previous chunk
    # ========================================================

    if len(chunks) >= 2:

        last_chunk_tokens = count_tokens(
            chunks[-1]
        )


        if (
            last_chunk_tokens
            < MIN_LAST_CHUNK_TOKENS
        ):

            merged = (
                chunks[-2]
                + "\n\n"
                + chunks[-1]
            )


            merged_tokens = count_tokens(
                merged
            )


            if (
                merged_tokens
                <= MAX_SEQUENCE_LENGTH
            ):

                chunks[-2] = merged

                chunks.pop()


    return chunks


# ============================================================
# 13. CATEGORY FROM DIRECTORY
#
# Example:
#
# Data/01_source_documents/security/file.md
#
# category = security
# ============================================================

def get_category(path):

    relative_path = path.relative_to(
        SOURCE_DIR
    )


    if len(relative_path.parts) > 1:

        return relative_path.parts[0]


    return "general"


# ============================================================
# 14. DOCUMENT ID
#
# Try to read:
#
# Document ID: CORP-001
#
# If not available, use filename stem.
# ============================================================

def get_document_id(
    raw_text,
    path,
):

    match = re.search(
        r"^Document ID\s*:\s*(.+)$",
        raw_text,
        flags=(
            re.MULTILINE
            | re.IGNORECASE
        ),
    )


    if match:

        return (
            match
            .group(1)
            .strip()
        )


    return path.stem


# ============================================================
# 15. LOAD SOURCE DOCUMENTS
# ============================================================

def load_documents():

    print("\n" + "=" * 60)
    print("LOADING SOURCE DOCUMENTS")
    print("=" * 60)


    if not SOURCE_DIR.exists():

        raise FileNotFoundError(
            f"""
Source directory was not found:

{SOURCE_DIR}
"""
        )


    files = sorted(
        SOURCE_DIR.rglob(
            "*.md"
        )
    )


    if not files:

        raise RuntimeError(
            f"""
No Markdown source documents found inside:

{SOURCE_DIR}
"""
        )


    print(
        "Source documents found:",
        len(files),
    )


    documents = []


    for path in files:

        raw_text = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )


        document_id = get_document_id(
            raw_text,
            path,
        )


        category = get_category(
            path
        )


        cleaned_text = clean_document(
            raw_text
        )


        if not cleaned_text:

            print(
                f"Skipping empty document: "
                f"{path.name}"
            )

            continue


        token_count = count_tokens(
            cleaned_text
        )


        documents.append(
            {

                "document_id":
                    document_id,

                "category":
                    category,

                "source_file":
                    str(
                        path.relative_to(
                            SOURCE_DIR
                        )
                    ),

                "text":
                    cleaned_text,

                "num_tokens":
                    token_count,
            }
        )


    print(
        "Usable documents:",
        len(documents),
    )


    return documents


# ============================================================
# 16. SPLIT DOCUMENTS
#
# IMPORTANT:
#
# We split WHOLE DOCUMENTS first.
#
# Then chunk each split.
#
# This prevents chunks from the same document appearing in
# both training and validation/test.
# ============================================================

def split_documents(
    documents,
):

    rng = random.Random(
        SEED
    )


    documents = (
        documents.copy()
    )


    rng.shuffle(
        documents
    )


    total_documents = len(
        documents
    )


    train_count = int(
        total_documents
        * TRAIN_RATIO
    )


    validation_count = int(
        total_documents
        * VALIDATION_RATIO
    )


    train_documents = (
        documents[
            :train_count
        ]
    )


    validation_documents = (
        documents[
            train_count:
            train_count
            + validation_count
        ]
    )


    test_documents = (
        documents[
            train_count
            + validation_count:
        ]
    )


    return (
        train_documents,
        validation_documents,
        test_documents,
    )


# ============================================================
# 17. CHUNK DOCUMENT SPLIT
# ============================================================

def create_chunk_records(
    documents,
    split_name,
):

    records = []


    for document in documents:

        chunks = chunk_text_by_tokens(
            document["text"]
        )


        for chunk_index, chunk in enumerate(
            chunks,
            start=1,
        ):

            num_tokens = count_tokens(
                chunk
            )


            if (
                num_tokens
                > MAX_SEQUENCE_LENGTH
            ):

                raise RuntimeError(
                    f"""
Chunk exceeds max sequence length.

Document:
{document['document_id']}

Chunk:
{chunk_index}

Tokens:
{num_tokens}
"""
                )


            records.append(
                {

                    "document_id":
                        document[
                            "document_id"
                        ],

                    "category":
                        document[
                            "category"
                        ],

                    "source_file":
                        document[
                            "source_file"
                        ],

                    "chunk_id":
                        (
                            f"{document['document_id']}"
                            f"-C{chunk_index:03d}"
                        ),

                    "split":
                        split_name,

                    "text":
                        chunk,

                    "num_tokens":
                        num_tokens,
                }
            )


    return records


# ============================================================
# 18. DOCUMENT LEAKAGE CHECK
# ============================================================

def check_document_leakage(
    train_records,
    validation_records,
    test_records,
):

    train_docs = {
        record[
            "document_id"
        ]
        for record
        in train_records
    }


    validation_docs = {
        record[
            "document_id"
        ]
        for record
        in validation_records
    }


    test_docs = {
        record[
            "document_id"
        ]
        for record
        in test_records
    }


    train_validation_overlap = (
        train_docs
        & validation_docs
    )


    train_test_overlap = (
        train_docs
        & test_docs
    )


    validation_test_overlap = (
        validation_docs
        & test_docs
    )


    if train_validation_overlap:

        raise RuntimeError(
            f"""
Document leakage between train and validation:

{train_validation_overlap}
"""
        )


    if train_test_overlap:

        raise RuntimeError(
            f"""
Document leakage between train and test:

{train_test_overlap}
"""
        )


    if validation_test_overlap:

        raise RuntimeError(
            f"""
Document leakage between validation and test:

{validation_test_overlap}
"""
        )


    print(
        "\nDocument leakage check: PASSED"
    )


# ============================================================
# 19. WRITE JSONL
# ============================================================

def write_jsonl(
    path,
    records,
):

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# 20. CATEGORY COUNTER
# ============================================================

def category_counts(
    records,
):

    return dict(
        Counter(
            record[
                "category"
            ]
            for record
            in records
        )
    )


# ============================================================
# 21. TOKEN STATISTICS
# ============================================================

def token_stats(
    records,
):

    token_counts = [
        record[
            "num_tokens"
        ]
        for record
        in records
    ]


    if not token_counts:

        return {
            "total_tokens": 0,
            "min_tokens": 0,
            "max_tokens": 0,
            "average_tokens": 0,
        }


    return {

        "total_tokens":
            sum(
                token_counts
            ),

        "min_tokens":
            min(
                token_counts
            ),

        "max_tokens":
            max(
                token_counts
            ),

        "average_tokens":
            round(
                sum(token_counts)
                / len(token_counts),
                2,
            ),
    }


# ============================================================
# 22. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("NEXAFLOW CPT DATA PREPARATION")
    print("QWEN2.5-1.5B")
    print("=" * 60)


    # --------------------------------------------------------
    # Load source documents
    # --------------------------------------------------------

    documents = load_documents()


    # --------------------------------------------------------
    # Split WHOLE documents
    # --------------------------------------------------------

    (
        train_documents,
        validation_documents,
        test_documents,
    ) = split_documents(
        documents
    )


    print("\n" + "=" * 60)
    print("DOCUMENT SPLIT")
    print("=" * 60)


    print(
        "Train documents:",
        len(train_documents),
    )


    print(
        "Validation documents:",
        len(validation_documents),
    )


    print(
        "Test documents:",
        len(test_documents),
    )


    # --------------------------------------------------------
    # Chunk each split
    # --------------------------------------------------------

    print("\nChunking train documents...")

    train_records = (
        create_chunk_records(
            train_documents,
            "train",
        )
    )


    print(
        "Chunking validation documents..."
    )

    validation_records = (
        create_chunk_records(
            validation_documents,
            "validation",
        )
    )


    print(
        "Chunking test documents..."
    )

    test_records = (
        create_chunk_records(
            test_documents,
            "test",
        )
    )


    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    check_document_leakage(
        train_records,
        validation_records,
        test_records,
    )


    # --------------------------------------------------------
    # Save JSONL files
    # --------------------------------------------------------

    write_jsonl(
        TRAIN_FILE,
        train_records,
    )


    write_jsonl(
        VALIDATION_FILE,
        validation_records,
    )


    write_jsonl(
        TEST_FILE,
        test_records,
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    train_token_stats = (
        token_stats(
            train_records
        )
    )


    validation_token_stats = (
        token_stats(
            validation_records
        )
    )


    test_token_stats = (
        token_stats(
            test_records
        )
    )


    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------

    summary = {

        "dataset_name":
            "NexaFlow Continued Pretraining Dataset",

        "tokenizer_model":
            TOKENIZER_MODEL,

        "seed":
            SEED,

        "max_sequence_length":
            MAX_SEQUENCE_LENGTH,

        "target_chunk_tokens":
            TARGET_CHUNK_TOKENS,

        "min_last_chunk_tokens":
            MIN_LAST_CHUNK_TOKENS,

        "source_documents":
            len(
                documents
            ),

        "document_split": {

            "train":
                len(
                    train_documents
                ),

            "validation":
                len(
                    validation_documents
                ),

            "test":
                len(
                    test_documents
                ),
        },

        "chunk_split": {

            "train":
                len(
                    train_records
                ),

            "validation":
                len(
                    validation_records
                ),

            "test":
                len(
                    test_records
                ),
        },

        "token_statistics": {

            "train":
                train_token_stats,

            "validation":
                validation_token_stats,

            "test":
                test_token_stats,
        },

        "category_counts": {

            "train":
                category_counts(
                    train_records
                ),

            "validation":
                category_counts(
                    validation_records
                ),

            "test":
                category_counts(
                    test_records
                ),
        },

        "document_level_split":
            True,

        "document_leakage":
            False,
    }


    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )


    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n")
    print("=" * 60)
    print("CPT DATA PREPARATION COMPLETE")
    print("=" * 60)


    print(
        f"\nSource documents : "
        f"{len(documents)}"
    )


    print(
        "\nDOCUMENT SPLIT"
    )


    print(
        f"Train documents      : "
        f"{len(train_documents)}"
    )


    print(
        f"Validation documents : "
        f"{len(validation_documents)}"
    )


    print(
        f"Test documents       : "
        f"{len(test_documents)}"
    )


    print(
        "\nCHUNK SPLIT"
    )


    print(
        f"Train chunks      : "
        f"{len(train_records)}"
    )


    print(
        f"Validation chunks : "
        f"{len(validation_records)}"
    )


    print(
        f"Test chunks       : "
        f"{len(test_records)}"
    )


    print(
        "\nTRAIN TOKEN STATS"
    )


    print(
        json.dumps(
            train_token_stats,
            indent=4,
        )
    )


    print(
        "\nVALIDATION TOKEN STATS"
    )


    print(
        json.dumps(
            validation_token_stats,
            indent=4,
        )
    )


    print(
        "\nTEST TOKEN STATS"
    )


    print(
        json.dumps(
            test_token_stats,
            indent=4,
        )
    )


    print(
        "\nOutput files:"
    )


    print(
        TRAIN_FILE
    )


    print(
        VALIDATION_FILE
    )


    print(
        TEST_FILE
    )


    print(
        SUMMARY_FILE
    )


    print("\n")
    print("=" * 60)
    print("IMPORTANT NEXT CHECK")
    print("=" * 60)


    print(
        """
Before CPT training, check whether the important NexaFlow
identity/company facts appear in TRAIN.

Run in PowerShell:

Select-String -Path "Data\\02_continued_pretraining\\train.jsonl" `
    -Pattern "Venky|Vemala Venkatesh|Saketh|Rakesh|Erlangen|Nuremberg|07 May 2026"

If these facts are absent from the TRAIN split, do not start CPT
training yet.

The tokenizer is Qwen2.5-1.5B, but this script does NOT train
or load the full 1.5B model.
"""
    )


if __name__ == "__main__":

    main()