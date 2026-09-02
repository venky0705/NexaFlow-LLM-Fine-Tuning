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
    r"C:\Users\ASUS\Desktop\Company_ChatBot_Finetuning\Data\03_sft"
)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"


# ------------------------------------------------------------
# We want approximately:
#
# 30 rules/documents × 5 examples = 150 examples
# ------------------------------------------------------------

NUMBER_OF_SOURCE_RULES = 30
EXAMPLES_PER_RULE = 5

RANDOM_SEED = 42


# ============================================================
# 2. SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = (
    "Answer using NexaFlow Technologies company policy. "
    "Provide only information supported by NexaFlow policy. "
    "Do not invent policy details. "
    "If the available policy does not provide the requested "
    "information, clearly state that the policy does not specify it."
)


# ============================================================
# 3. EXTRACT DOCUMENT TITLE
# ============================================================

def extract_title(text: str) -> str:

    for line in text.splitlines():

        line = line.strip()

        if line.startswith("# ") and not line.startswith("## "):

            return line[2:].strip()

    return "NexaFlow Policy"


# ============================================================
# 4. EXTRACT DOCUMENT ID
# ============================================================

def extract_document_id(file_path: Path) -> str:

    # Example:
    #
    # HR-002_annual_leave.md
    #
    # becomes:
    #
    # HR-002

    return file_path.stem.split("_")[0]


# ============================================================
# 5. EXTRACT POLICY RULES
# ============================================================

def extract_policy_rules(text: str):

    rules = []

    inside_policy_section = False

    for line in text.splitlines():

        stripped = line.strip()

        # ----------------------------------------------------
        # Start policy section
        # ----------------------------------------------------

        if stripped.lower() == "## policy and operating information":

            inside_policy_section = True

            continue

        # ----------------------------------------------------
        # Stop when next Markdown section begins
        # ----------------------------------------------------

        if (
            inside_policy_section
            and stripped.startswith("## ")
        ):

            break

        if not inside_policy_section:

            continue

        # ----------------------------------------------------
        # Extract numbered rules
        #
        # 1. ...
        # 2. ...
        # ----------------------------------------------------

        match = re.match(
            r"^\d+\.\s+(.+)$",
            stripped
        )

        if match:

            rule = match.group(1).strip()

            if rule:

                rules.append(rule)

    return rules


# ============================================================
# 6. CREATE SHORT TOPIC FROM RULE
# ============================================================

def create_topic(rule: str) -> str:

    """
    Extract a short phrase from a policy rule.

    This is only used for generating question variations.
    """

    cleaned = rule.strip()

    cleaned = re.sub(
        r"[.,;:]$",
        "",
        cleaned
    )

    words = cleaned.split()

    # Do not make the topic extremely long
    if len(words) > 12:

        cleaned = " ".join(
            words[:12]
        )

    return cleaned


# ============================================================
# 7. GENERATE 5 SFT VARIANTS
# ============================================================

def generate_variants(
    document_title: str,
    rule: str
):

    topic = create_topic(rule)

    examples = []


    # ========================================================
    # TYPE 1 — DIRECT POLICY QUESTION
    # ========================================================

    examples.append(
        {
            "input": (
                f"What requirement does NexaFlow have regarding "
                f"{document_title.lower()}?"
            ),

            "output": rule,

            "example_type": "direct"
        }
    )


    # ========================================================
    # TYPE 2 — EMPLOYEE QUESTION
    # ========================================================

    examples.append(
        {
            "input": (
                f"I need to understand the NexaFlow policy on "
                f"{document_title.lower()}. "
                f"What rule should I follow?"
            ),

            "output": rule,

            "example_type": "employee_question"
        }
    )


    # ========================================================
    # TYPE 3 — POLICY EXPLANATION
    # ========================================================

    examples.append(
        {
            "input": (
                f"Can you explain the NexaFlow requirement "
                f"related to this {document_title.lower()} rule: "
                f"{topic}?"
            ),

            "output": (
                f"According to NexaFlow policy, {rule}"
            ),

            "example_type": "policy_explanation"
        }
    )


    # ========================================================
    # TYPE 4 — POLICY VERIFICATION
    # ========================================================

    examples.append(
        {
            "input": (
                f"Is the following statement consistent with "
                f"NexaFlow's {document_title} policy: "
                f"'{rule}'?"
            ),

            "output": (
                f"Yes. This is consistent with NexaFlow policy. "
                f"{rule}"
            ),

            "example_type": "policy_verification"
        }
    )


    # ========================================================
    # TYPE 5 — PRACTICAL GUIDANCE
    # ========================================================

    examples.append(
        {
            "input": (
                f"What should an employee or authorized user "
                f"keep in mind when dealing with "
                f"{document_title.lower()}?"
            ),

            "output": rule,

            "example_type": "practical_guidance"
        }
    )


    return examples


# ============================================================
# 8. LOAD SOURCE DOCUMENTS
# ============================================================

def load_candidate_rules():

    candidates = []

    markdown_files = list(
        SOURCE_DIR.rglob("*.md")
    )

    print(
        f"Found {len(markdown_files)} source documents."
    )


    for file_path in markdown_files:

        raw_text = file_path.read_text(
            encoding="utf-8"
        )

        document_id = extract_document_id(
            file_path
        )

        document_title = extract_title(
            raw_text
        )

        category = file_path.parent.name

        rules = extract_policy_rules(
            raw_text
        )


        # ----------------------------------------------------
        # For this first SFT dataset we take ONE rule
        # from each document.
        #
        # This gives us more domain diversity.
        # ----------------------------------------------------

        if len(rules) > 0:

            selected_rule = rules[0]

            candidates.append(
                {
                    "document_id": document_id,
                    "document_title": document_title,
                    "category": category,
                    "rule_id": f"{document_id}-R01",
                    "rule": selected_rule
                }
            )


    return candidates


# ============================================================
# 9. SELECT BALANCED SOURCE RULES
# ============================================================

def select_source_rules(candidates):

    random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Group candidates by category
    # --------------------------------------------------------

    category_groups = {}

    for candidate in candidates:

        category = candidate["category"]

        category_groups.setdefault(
            category,
            []
        ).append(candidate)


    # Shuffle every category independently
    for category in category_groups:

        random.shuffle(
            category_groups[category]
        )


    selected = []


    # --------------------------------------------------------
    # Round-robin selection
    #
    # This prevents selecting all 30 examples from HR,
    # Security, etc.
    # --------------------------------------------------------

    categories = list(
        category_groups.keys()
    )

    while (
        len(selected)
        < NUMBER_OF_SOURCE_RULES
    ):

        added_something = False

        for category in categories:

            if len(selected) >= NUMBER_OF_SOURCE_RULES:

                break

            if category_groups[category]:

                selected.append(
                    category_groups[category].pop()
                )

                added_something = True


        if not added_something:

            break


    return selected


# ============================================================
# 10. GENERATE SFT DATA
# ============================================================

def build_sft_dataset(selected_rules):

    all_examples = []

    for source in selected_rules:

        variants = generate_variants(

            document_title=source[
                "document_title"
            ],

            rule=source[
                "rule"
            ]
        )


        for variant_number, example in enumerate(
            variants,
            start=1
        ):

            example_id = (
                f"SFT-"
                f"{source['rule_id']}"
                f"-V{variant_number}"
            )


            record = {

                "id": example_id,

                "instruction": SYSTEM_INSTRUCTION,

                "input": example["input"],

                "output": example["output"],

                "source_document": source[
                    "document_id"
                ],

                "source_rule": source[
                    "rule_id"
                ],

                "category": source[
                    "category"
                ],

                "example_type": example[
                    "example_type"
                ]
            }


            all_examples.append(
                record
            )


    return all_examples


# ============================================================
# 11. SPLIT BY SOURCE DOCUMENT
# ============================================================

def split_dataset(
    selected_rules,
    all_examples
):

    document_ids = [

        item["document_id"]

        for item in selected_rules
    ]


    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        document_ids
    )


    # ========================================================
    # 30 source documents
    #
    # Train = 24 documents
    # Validation = 3 documents
    # Test = 3 documents
    #
    # Each document has 5 SFT examples.
    #
    # Train = 120
    # Validation = 15
    # Test = 15
    # ========================================================


    test_documents = set(
        document_ids[:3]
    )

    validation_documents = set(
        document_ids[3:6]
    )

    train_documents = set(
        document_ids[6:]
    )


    train_data = []

    validation_data = []

    test_data = []


    for example in all_examples:

        doc_id = example[
            "source_document"
        ]


        if doc_id in test_documents:

            test_data.append(
                example
            )


        elif doc_id in validation_documents:

            validation_data.append(
                example
            )


        else:

            train_data.append(
                example
            )


    return (
        train_data,
        validation_data,
        test_data
    )


# ============================================================
# 12. SAVE JSONL
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
# 13. CHECK FOR DOCUMENT LEAKAGE
# ============================================================

def check_leakage(
    train_data,
    validation_data,
    test_data
):

    train_docs = {

        x["source_document"]

        for x in train_data
    }


    validation_docs = {

        x["source_document"]

        for x in validation_data
    }


    test_docs = {

        x["source_document"]

        for x in test_data
    }


    assert train_docs.isdisjoint(
        validation_docs
    )

    assert train_docs.isdisjoint(
        test_docs
    )

    assert validation_docs.isdisjoint(
        test_docs
    )


    print(
        "\nDocument leakage check: PASSED"
    )


# ============================================================
# 14. PRINT STATISTICS
# ============================================================

def print_statistics(
    selected_rules,
    all_examples,
    train_data,
    validation_data,
    test_data
):

    print(
        "\n================================="
    )

    print(
        "SFT DATASET SUMMARY"
    )

    print(
        "================================="
    )


    print(
        f"Selected source rules : "
        f"{len(selected_rules)}"
    )

    print(
        f"Examples per rule     : "
        f"{EXAMPLES_PER_RULE}"
    )

    print(
        f"Total SFT examples    : "
        f"{len(all_examples)}"
    )


    print(
        "\nDataset split"
    )


    print(
        f"Train      : "
        f"{len(train_data)}"
    )

    print(
        f"Validation : "
        f"{len(validation_data)}"
    )

    print(
        f"Test       : "
        f"{len(test_data)}"
    )


    # --------------------------------------------------------
    # Category distribution
    # --------------------------------------------------------

    categories = {}

    for example in all_examples:

        category = example[
            "category"
        ]

        categories[
            category
        ] = categories.get(
            category,
            0
        ) + 1


    print(
        "\nCategory distribution"
    )


    for category, count in sorted(
        categories.items()
    ):

        print(
            f"{category:<15} : {count}"
        )


# ============================================================
# 15. SHOW SAMPLE RECORDS
# ============================================================

def show_samples(
    data,
    number=5
):

    print(
        "\n================================="
    )

    print(
        "SAMPLE SFT RECORDS"
    )

    print(
        "================================="
    )


    for example in data[:number]:

        print(
            "\nID:"
        )

        print(
            example["id"]
        )


        print(
            "\nQUESTION:"
        )

        print(
            example["input"]
        )


        print(
            "\nANSWER:"
        )

        print(
            example["output"]
        )


        print(
            "\nTYPE:"
        )

        print(
            example["example_type"]
        )


        print(
            "\n---------------------------------"
        )


# ============================================================
# 16. MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load source rules
    # --------------------------------------------------------

    candidates = load_candidate_rules()


    print(
        f"Candidate source rules: "
        f"{len(candidates)}"
    )


    # --------------------------------------------------------
    # Select 30 source rules
    # --------------------------------------------------------

    selected_rules = select_source_rules(
        candidates
    )


    if (
        len(selected_rules)
        < NUMBER_OF_SOURCE_RULES
    ):

        raise ValueError(

            f"Only {len(selected_rules)} usable source rules "
            f"were found. Need at least "
            f"{NUMBER_OF_SOURCE_RULES}."
        )


    # --------------------------------------------------------
    # Generate 150 examples
    # --------------------------------------------------------

    all_examples = build_sft_dataset(
        selected_rules
    )


    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_data,
        validation_data,
        test_data

    ) = split_dataset(

        selected_rules,
        all_examples
    )


    # --------------------------------------------------------
    # Check leakage
    # --------------------------------------------------------

    check_leakage(

        train_data,
        validation_data,
        test_data
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_jsonl(
        train_data,
        TRAIN_FILE
    )

    save_jsonl(
        validation_data,
        VALIDATION_FILE
    )

    save_jsonl(
        test_data,
        TEST_FILE
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print_statistics(

        selected_rules,
        all_examples,
        train_data,
        validation_data,
        test_data
    )


    # --------------------------------------------------------
    # Preview
    # --------------------------------------------------------

    show_samples(
        train_data
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