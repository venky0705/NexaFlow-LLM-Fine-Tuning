from pathlib import Path
import json
import random
import re


# ============================================================
# 1. CONFIGURATION
# ============================================================

SOURCE_DIR = Path(
    r"C:\Users\ASUS\Desktop\Company_ChatBot_Finetuning\Data\01_source_documents"
)

OUTPUT_DIR = Path(
    r"C:\Users\ASUS\Desktop\Company_ChatBot_Finetuning\Data\02_continued_pretraining"
)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"

VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42


# ============================================================
# 2. SECTIONS TO REMOVE
# ============================================================

REMOVE_SECTIONS = {
    "dataset notice",
}


# ============================================================
# 3. CLEAN DOCUMENT
# ============================================================

def clean_document(text: str) -> str:

    lines = text.splitlines()

    cleaned_lines = []

    skip_section = False


    for line in lines:

        stripped = line.strip()


        # ----------------------------------------------------
        # Detect Markdown section headings
        # ----------------------------------------------------

        if stripped.startswith("## "):

            section_name = (
                stripped[3:]
                .strip()
                .lower()
            )


            if section_name in REMOVE_SECTIONS:

                skip_section = True

                continue

            else:

                skip_section = False


        # ----------------------------------------------------
        # Skip unwanted section content
        # ----------------------------------------------------

        if skip_section:

            continue


        # ----------------------------------------------------
        # Remove document metadata
        # ----------------------------------------------------

        metadata_prefixes = (
            "Document ID:",
            "Organization:",
            "Status:",
            "Version:",
            "Effective date:",
            "Category:",
            "Owner:",
        )


        if stripped.startswith(
            metadata_prefixes
        ):

            continue


        cleaned_lines.append(
            line
        )


    # --------------------------------------------------------
    # Join cleaned text
    # --------------------------------------------------------

    text = "\n".join(
        cleaned_lines
    )


    # --------------------------------------------------------
    # Remove excessive blank lines
    # --------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    return text.strip()


# ============================================================
# 4. EXTRACT DOCUMENT ID
# ============================================================

def extract_document_id(
    file_path: Path
) -> str:

    # Example:
    #
    # HR-002_annual_leave.md
    #
    # becomes:
    #
    # HR-002

    return file_path.stem.split("_")[0]


# ============================================================
# 5. LOAD ALL SOURCE DOCUMENTS
# ============================================================

def load_documents():

    documents = []


    markdown_files = sorted(
        SOURCE_DIR.rglob("*.md")
    )


    print(
        f"Found {len(markdown_files)} source documents."
    )


    for file_path in markdown_files:

        # ----------------------------------------------------
        # Read source file
        # ----------------------------------------------------

        raw_text = file_path.read_text(
            encoding="utf-8"
        )


        # ----------------------------------------------------
        # Clean source file
        # ----------------------------------------------------

        cleaned_text = clean_document(
            raw_text
        )


        # ----------------------------------------------------
        # Skip empty documents
        # ----------------------------------------------------

        if not cleaned_text:

            continue


        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        document_id = extract_document_id(
            file_path
        )

        category = file_path.parent.name


        # ----------------------------------------------------
        # Create CPT record
        # ----------------------------------------------------

        record = {

            "document_id": document_id,

            "category": category,

            "text": cleaned_text
        }


        documents.append(
            record
        )


    return documents


# ============================================================
# 6. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_documents(
    documents
):

    random.seed(
        RANDOM_SEED
    )


    # Make a copy so original list is not modified
    documents = documents.copy()


    random.shuffle(
        documents
    )


    total_documents = len(
        documents
    )


    test_size = max(
        1,
        int(
            total_documents
            * TEST_RATIO
        )
    )


    validation_size = max(
        1,
        int(
            total_documents
            * VALIDATION_RATIO
        )
    )


    # --------------------------------------------------------
    # Test documents
    # --------------------------------------------------------

    test_documents = documents[
        :test_size
    ]


    # --------------------------------------------------------
    # Validation documents
    # --------------------------------------------------------

    validation_documents = documents[
        test_size:
        test_size + validation_size
    ]


    # --------------------------------------------------------
    # Training documents
    # --------------------------------------------------------

    train_documents = documents[
        test_size + validation_size:
    ]


    return (
        train_documents,
        validation_documents,
        test_documents
    )


# ============================================================
# 7. SAVE JSONL
# ============================================================

def save_jsonl(
    data,
    path
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with path.open(
        "w",
        encoding="utf-8"
    ) as file:

        for record in data:

            json_record = json.dumps(
                record,
                ensure_ascii=False
            )


            file.write(
                json_record + "\n"
            )


# ============================================================
# 8. COUNT WORDS
# ============================================================

def count_words(
    documents
):

    return sum(

        len(
            document["text"].split()
        )

        for document in documents
    )


# ============================================================
# 9. CHECK DOCUMENT LEAKAGE
# ============================================================

def check_document_leakage(
    train_documents,
    validation_documents,
    test_documents
):

    train_ids = {

        document["document_id"]

        for document in train_documents
    }


    validation_ids = {

        document["document_id"]

        for document in validation_documents
    }


    test_ids = {

        document["document_id"]

        for document in test_documents
    }


    # --------------------------------------------------------
    # Make sure documents are not repeated across splits
    # --------------------------------------------------------

    assert train_ids.isdisjoint(
        validation_ids
    )

    assert train_ids.isdisjoint(
        test_ids
    )

    assert validation_ids.isdisjoint(
        test_ids
    )


    print(
        "\nDocument leakage check: PASSED"
    )


# ============================================================
# 10. PRINT STATISTICS
# ============================================================

def print_statistics(
    train_documents,
    validation_documents,
    test_documents
):

    train_words = count_words(
        train_documents
    )

    validation_words = count_words(
        validation_documents
    )

    test_words = count_words(
        test_documents
    )


    total_words = (
        train_words
        + validation_words
        + test_words
    )


    total_documents = (
        len(train_documents)
        + len(validation_documents)
        + len(test_documents)
    )


    print(
        "\n================================="
    )

    print(
        "CPT DATASET SUMMARY"
    )

    print(
        "================================="
    )


    print(
        f"Total documents      : "
        f"{total_documents}"
    )


    print(
        "\nDocument split"
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
        "\nWord count"
    )


    print(
        f"Train words          : "
        f"{train_words:,}"
    )

    print(
        f"Validation words     : "
        f"{validation_words:,}"
    )

    print(
        f"Test words           : "
        f"{test_words:,}"
    )

    print(
        f"Total words          : "
        f"{total_words:,}"
    )


# ============================================================
# 11. SHOW SAMPLE
# ============================================================

def show_sample(
    documents
):

    if not documents:

        return


    sample = documents[0]


    print(
        "\n================================="
    )

    print(
        "SAMPLE CPT RECORD"
    )

    print(
        "================================="
    )


    print(
        "\nDocument ID:"
    )

    print(
        sample["document_id"]
    )


    print(
        "\nCategory:"
    )

    print(
        sample["category"]
    )


    print(
        "\nText preview:"
    )

    print(
        sample["text"][:700]
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    documents = load_documents()


    if len(documents) < 3:

        raise ValueError(
            "At least 3 source documents are required "
            "for train, validation and test splits."
        )


    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    (
        train_documents,
        validation_documents,
        test_documents

    ) = split_documents(
        documents
    )


    # --------------------------------------------------------
    # Check leakage
    # --------------------------------------------------------

    check_document_leakage(

        train_documents,
        validation_documents,
        test_documents
    )


    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    save_jsonl(
        train_documents,
        TRAIN_FILE
    )


    save_jsonl(
        validation_documents,
        VALIDATION_FILE
    )


    save_jsonl(
        test_documents,
        TEST_FILE
    )


    # --------------------------------------------------------
    # Dataset statistics
    # --------------------------------------------------------

    print_statistics(

        train_documents,
        validation_documents,
        test_documents
    )


    # --------------------------------------------------------
    # Show sample
    # --------------------------------------------------------

    show_sample(
        train_documents
    )


    print(
        "\nFiles created:"
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


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()