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
    r"C:\Users\ASUS\Desktop\Company_ChatBot_Finetuning\Data\04_dpo"
)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"


# 20 source rules × 3 preference variants = 60 DPO pairs

NUMBER_OF_SOURCE_RULES = 20
PAIRS_PER_RULE = 3

RANDOM_SEED = 42


# ============================================================
# 2. EXTRACT DOCUMENT TITLE
# ============================================================

def extract_title(text: str) -> str:

    for line in text.splitlines():

        line = line.strip()

        if line.startswith("# ") and not line.startswith("## "):

            return line[2:].strip()

    return "NexaFlow Policy"


# ============================================================
# 3. EXTRACT DOCUMENT ID
# ============================================================

def extract_document_id(file_path: Path) -> str:

    return file_path.stem.split("_")[0]


# ============================================================
# 4. EXTRACT POLICY RULES
# ============================================================

def extract_policy_rules(text: str):

    rules = []

    inside_policy_section = False

    for line in text.splitlines():

        stripped = line.strip()

        if stripped.lower() == "## policy and operating information":

            inside_policy_section = True
            continue

        if (
            inside_policy_section
            and stripped.startswith("## ")
        ):

            break

        if not inside_policy_section:

            continue

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
# 5. CREATE A SHORT RULE TOPIC
# ============================================================

def create_topic(rule: str) -> str:

    cleaned = rule.strip()

    cleaned = re.sub(
        r"[.,;:]$",
        "",
        cleaned
    )

    words = cleaned.split()

    if len(words) > 10:

        cleaned = " ".join(
            words[:10]
        )

    return cleaned


# ============================================================
# 6. CREATE REJECTED ANSWER
# ============================================================

def create_wrong_answer(
    rule: str,
    document_title: str,
    variant_number: int
) -> str:

    """
    Creates intentionally worse answers.

    These are not meant to be factually correct.
    They teach DPO that grounded answers are preferred.
    """

    if variant_number == 1:

        return (
            f"NexaFlow does not have any specific requirement "
            f"for {document_title.lower()}, so employees may "
            f"decide what works best for them."
        )

    elif variant_number == 2:

        return (
            f"The policy allows exceptions whenever an employee "
            f"or customer prefers a different approach, even "
            f"without documented approval."
        )

    else:

        return (
            f"This requirement is optional and does not need "
            f"to be followed unless a manager personally asks "
            f"for it."
        )


# ============================================================
# 7. GENERATE 3 DPO VARIANTS PER RULE
# ============================================================

def generate_dpo_pairs(
    document_title: str,
    rule: str
):

    topic = create_topic(rule)

    pairs = []


    # ========================================================
    # VARIANT 1 — DIRECT POLICY QUESTION
    # ========================================================

    pairs.append(
        {
            "prompt": (
                f"What does NexaFlow policy require regarding "
                f"{document_title.lower()}?"
            ),

            "chosen": (
                f"According to NexaFlow policy, {rule}"
            ),

            "rejected": create_wrong_answer(
                rule,
                document_title,
                1
            ),

            "preference_type": "grounded_vs_unsupported"
        }
    )


    # ========================================================
    # VARIANT 2 — POLICY VERIFICATION
    # ========================================================

    pairs.append(
        {
            "prompt": (
                f"I need guidance about this NexaFlow rule: "
                f"{topic}. What should I follow?"
            ),

            "chosen": rule,

            "rejected": create_wrong_answer(
                rule,
                document_title,
                2
            ),

            "preference_type": "correct_policy_vs_false_exception"
        }
    )


    # ========================================================
    # VARIANT 3 — PRACTICAL GUIDANCE
    # ========================================================

    pairs.append(
        {
            "prompt": (
                f"How should an employee handle a situation "
                f"covered by NexaFlow's "
                f"{document_title.lower()} policy?"
            ),

            "chosen": (
                f"The employee should follow the documented "
                f"NexaFlow rule: {rule}"
            ),

            "rejected": create_wrong_answer(
                rule,
                document_title,
                3
            ),

            "preference_type": "policy_following_vs_optional"
        }
    )


    return pairs


# ============================================================
# 8. LOAD CANDIDATE RULES
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
        # Take one rule from each document
        # to maximize domain diversity.
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
# 9. BALANCED RULE SELECTION
# ============================================================

def select_source_rules(candidates):

    random.seed(
        RANDOM_SEED
    )

    category_groups = {}


    for candidate in candidates:

        category = candidate["category"]

        category_groups.setdefault(
            category,
            []
        ).append(candidate)


    for category in category_groups:

        random.shuffle(
            category_groups[category]
        )


    selected = []

    categories = list(
        category_groups.keys()
    )


    # --------------------------------------------------------
    # Round-robin across categories
    # --------------------------------------------------------

    while (
        len(selected)
        < NUMBER_OF_SOURCE_RULES
    ):

        added = False

        for category in categories:

            if (
                len(selected)
                >= NUMBER_OF_SOURCE_RULES
            ):

                break


            if category_groups[category]:

                selected.append(
                    category_groups[category].pop()
                )

                added = True


        if not added:

            break


    return selected


# ============================================================
# 10. BUILD DPO DATASET
# ============================================================

def build_dpo_dataset(
    selected_rules
):

    all_pairs = []


    for source in selected_rules:

        generated_pairs = generate_dpo_pairs(

            document_title=source[
                "document_title"
            ],

            rule=source[
                "rule"
            ]
        )


        for pair_number, pair in enumerate(
            generated_pairs,
            start=1
        ):

            record_id = (
                f"DPO-"
                f"{source['rule_id']}"
                f"-V{pair_number}"
            )


            all_pairs.append(
                {
                    "id": record_id,

                    "prompt": pair["prompt"],

                    "chosen": pair["chosen"],

                    "rejected": pair["rejected"],

                    "source_document": source[
                        "document_id"
                    ],

                    "source_rule": source[
                        "rule_id"
                    ],

                    "category": source[
                        "category"
                    ],

                    "preference_type": pair[
                        "preference_type"
                    ]
                }
            )


    return all_pairs


# ============================================================
# 11. SPLIT BY SOURCE DOCUMENT
# ============================================================

def split_dataset(
    selected_rules,
    all_pairs
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


    # --------------------------------------------------------
    # 20 documents × 3 pairs = 60 total
    #
    # 16 docs train      = 48
    # 2 docs validation  = 6
    # 2 docs test        = 6
    # --------------------------------------------------------

    test_documents = set(
        document_ids[:2]
    )

    validation_documents = set(
        document_ids[2:4]
    )

    train_documents = set(
        document_ids[4:]
    )


    train_data = []

    validation_data = []

    test_data = []


    for pair in all_pairs:

        document_id = pair[
            "source_document"
        ]


        if document_id in test_documents:

            test_data.append(
                pair
            )

        elif document_id in validation_documents:

            validation_data.append(
                pair
            )

        else:

            train_data.append(
                pair
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

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# 13. LEAKAGE CHECK
# ============================================================

def check_document_leakage(
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
# 14. STATISTICS
# ============================================================

def print_statistics(
    selected_rules,
    all_pairs,
    train_data,
    validation_data,
    test_data
):

    print(
        "\n================================="
    )

    print(
        "DPO DATASET SUMMARY"
    )

    print(
        "================================="
    )


    print(
        f"Selected source rules : "
        f"{len(selected_rules)}"
    )

    print(
        f"Pairs per rule        : "
        f"{PAIRS_PER_RULE}"
    )

    print(
        f"Total DPO pairs       : "
        f"{len(all_pairs)}"
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


    categories = {}


    for pair in all_pairs:

        category = pair[
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
# 15. SHOW SAMPLE PAIRS
# ============================================================

def show_samples(
    data,
    number=5
):

    print(
        "\n================================="
    )

    print(
        "SAMPLE DPO PAIRS"
    )

    print(
        "================================="
    )


    for example in data[:number]:

        print(
            "\nPROMPT:"
        )

        print(
            example["prompt"]
        )


        print(
            "\nCHOSEN:"
        )

        print(
            example["chosen"]
        )


        print(
            "\nREJECTED:"
        )

        print(
            example["rejected"]
        )


        print(
            "\nTYPE:"
        )

        print(
            example["preference_type"]
        )


        print(
            "\n---------------------------------"
        )


# ============================================================
# 16. MAIN
# ============================================================

def main():

    candidates = load_candidate_rules()


    print(
        f"Candidate source rules: "
        f"{len(candidates)}"
    )


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
    # Generate DPO preference pairs
    # --------------------------------------------------------

    all_pairs = build_dpo_dataset(
        selected_rules
    )


    # --------------------------------------------------------
    # Split dataset
    # --------------------------------------------------------

    (
        train_data,
        validation_data,
        test_data

    ) = split_dataset(

        selected_rules,
        all_pairs
    )


    # --------------------------------------------------------
    # Check leakage
    # --------------------------------------------------------

    check_document_leakage(

        train_data,
        validation_data,
        test_data
    )


    # --------------------------------------------------------
    # Save JSONL
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
        all_pairs,
        train_data,
        validation_data,
        test_data
    )


    # --------------------------------------------------------
    # Show sample data
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