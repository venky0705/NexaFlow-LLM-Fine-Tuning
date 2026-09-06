from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "evaluation"
    / "evaluation_summary.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "visualizations"
)


# ============================================================
# MODEL ORDER
# ============================================================

MODEL_ORDER = [
    "base",
    "cpt",
    "sft",
    "dpo",
]


MODEL_LABELS = {
    "base": "Base",
    "cpt": "CPT",
    "sft": "SFT",
    "dpo": "DPO",
}


# ============================================================
# CHECK INPUT FILE
# ============================================================

def validate_input_file():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"""
Evaluation summary was not found:

{INPUT_FILE}

Run your final evaluation first.
"""
        )


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "model",
        "overall_behavior_accuracy",
        "mean_token_f1",
        "canonical_accuracy",
        "policy_accuracy",
        "numerical_accuracy",
        "scenario_accuracy",
        "unsupported_handling_accuracy",
        "unknown_people_accuracy",
        "false_premise_accuracy",
        "harmful_refusal_accuracy",
        "benign_security_accuracy",
        "out_of_scope_accuracy",
        "hallucination_rate",
        "false_refusal_rate",
        "runtime_seconds",
        "peak_vram_gb",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(
                missing_columns
            )
        )

    # --------------------------------------------------------
    # Normalize model names
    # --------------------------------------------------------

    df["model"] = (
        df["model"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    # --------------------------------------------------------
    # Put models in correct research order
    #
    # Base -> CPT -> SFT -> DPO
    # --------------------------------------------------------

    df["model"] = pd.Categorical(
        df["model"],
        categories=MODEL_ORDER,
        ordered=True,
    )

    df = (
        df
        .sort_values("model")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# MODEL LABELS
# ============================================================

def get_model_labels(df):

    return [
        MODEL_LABELS.get(
            str(model),
            str(model).upper(),
        )
        for model in df["model"]
        .astype(str)
    ]


# ============================================================
# PERCENTAGE CONVERSION
# ============================================================

def to_percent(series):

    return series * 100


# ============================================================
# SAVE FIGURE
# ============================================================

def save_figure(
    filename,
):

    path = (
        OUTPUT_DIR
        / filename
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    print(
        f"Saved: {path}"
    )

    plt.close()


# ============================================================
# GRAPH 1
#
# OVERALL MODEL COMPARISON
# ============================================================

def plot_overall_comparison(df):

    models = get_model_labels(
        df
    )

    overall = to_percent(
        df[
            "overall_behavior_accuracy"
        ]
    )

    token_f1 = to_percent(
        df[
            "mean_token_f1"
        ]
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        models,
        overall,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Overall behavior accuracy",
    )


    plt.plot(
        models,
        token_f1,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="Mean token F1",
    )


    plt.title(
        "NexaFlow Model Performance Across Fine-Tuning Stages",
        fontsize=15,
        fontweight="bold",
    )


    plt.xlabel(
        "Model Stage"
    )

    plt.ylabel(
        "Score (%)"
    )


    plt.ylim(
        0,
        100,
    )


    plt.grid(
        alpha=0.25
    )


    plt.legend()


    # --------------------------------------------------------
    # Put values over points
    # --------------------------------------------------------

    for index, value in enumerate(
        overall
    ):

        plt.annotate(
            f"{value:.1f}%",
            (
                index,
                value,
            ),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )


    save_figure(
        "overall_model_comparison.png"
    )


# ============================================================
# GRAPH 2
#
# CATEGORY COMPARISON
# ============================================================

def plot_category_comparison(df):

    models = get_model_labels(
        df
    )


    metrics = {

        "Canonical":
            "canonical_accuracy",

        "Policy":
            "policy_accuracy",

        "Numerical":
            "numerical_accuracy",

        "Scenario":
            "scenario_accuracy",

        "Unsupported":
            "unsupported_handling_accuracy",

        "Unknown People":
            "unknown_people_accuracy",

        "False Premise":
            "false_premise_accuracy",

        "Harmful":
            "harmful_refusal_accuracy",

        "Benign Security":
            "benign_security_accuracy",

        "Out of Scope":
            "out_of_scope_accuracy",
    }


    plt.figure(
        figsize=(14, 8)
    )


    for label, column in metrics.items():

        values = (
            df[column]
            * 100
        )

        plt.plot(
            models,
            values,
            marker="o",
            linewidth=1.7,
            markersize=5,
            label=label,
        )


    plt.title(
        "Behavior Category Performance by Model Stage",
        fontsize=15,
        fontweight="bold",
    )


    plt.xlabel(
        "Model Stage"
    )

    plt.ylabel(
        "Accuracy (%)"
    )


    plt.ylim(
        0,
        105,
    )


    plt.grid(
        alpha=0.2
    )


    plt.legend(
        bbox_to_anchor=(
            1.02,
            1,
        ),
        loc="upper left",
    )


    save_figure(
        "category_comparison.png"
    )


# ============================================================
# GRAPH 3
#
# HALLUCINATION / ABSTENTION FAILURE
# ============================================================

def plot_hallucination_comparison(df):

    models = get_model_labels(
        df
    )


    hallucination = (
        df[
            "hallucination_rate"
        ]
        * 100
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        models,
        hallucination,
        marker="o",
        linewidth=2.5,
        markersize=8,
    )


    plt.title(
        "Unsupported / Unknown-Person Abstention Failure",
        fontsize=15,
        fontweight="bold",
    )


    plt.xlabel(
        "Model Stage"
    )

    plt.ylabel(
        "Failure Rate (%)"
    )


    plt.ylim(
        0,
        100,
    )


    plt.grid(
        alpha=0.25
    )


    for index, value in enumerate(
        hallucination
    ):

        plt.annotate(
            f"{value:.1f}%",
            (
                index,
                value,
            ),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )


    save_figure(
        "hallucination_comparison.png"
    )


# ============================================================
# GRAPH 4
#
# IMPORTANT BEHAVIOR METRICS
# ============================================================

def plot_key_behavior_metrics(df):

    models = get_model_labels(
        df
    )


    important_metrics = {

        "Overall":
            "overall_behavior_accuracy",

        "Policy":
            "policy_accuracy",

        "Unsupported Handling":
            "unsupported_handling_accuracy",

        "False Premise":
            "false_premise_accuracy",

        "Harmful Refusal":
            "harmful_refusal_accuracy",

        "Benign Security":
            "benign_security_accuracy",
    }


    plt.figure(
        figsize=(12, 7)
    )


    for label, column in important_metrics.items():

        values = (
            df[column]
            * 100
        )


        plt.plot(
            models,
            values,
            marker="o",
            linewidth=2,
            markersize=6,
            label=label,
        )


    plt.title(
        "Key NexaFlow Assistant Behaviors",
        fontsize=15,
        fontweight="bold",
    )


    plt.xlabel(
        "Model Stage"
    )

    plt.ylabel(
        "Accuracy (%)"
    )


    plt.ylim(
        0,
        105,
    )


    plt.grid(
        alpha=0.2
    )


    plt.legend()


    save_figure(
        "model_metrics_comparison.png"
    )


# ============================================================
# GRAPH 5
#
# RUNTIME COMPARISON
# ============================================================

def plot_runtime(df):

    models = get_model_labels(
        df
    )


    runtime = (
        df[
            "runtime_seconds"
        ]
    )


    plt.figure(
        figsize=(10, 6)
    )


    plt.bar(
        models,
        runtime,
    )


    plt.title(
        "Evaluation Runtime by Model",
        fontsize=15,
        fontweight="bold",
    )


    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Runtime (seconds)"
    )


    plt.grid(
        axis="y",
        alpha=0.2,
    )


    for index, value in enumerate(
        runtime
    ):

        plt.text(
            index,
            value,
            f"{value:.1f}s",
            ha="center",
            va="bottom",
        )


    save_figure(
        "runtime_comparison.png"
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(df):

    print("\n")
    print("=" * 70)
    print("NEXAFLOW MODEL COMPARISON")
    print("=" * 70)


    for _, row in df.iterrows():

        model = str(
            row["model"]
        ).upper()


        overall = (
            row[
                "overall_behavior_accuracy"
            ]
            * 100
        )


        hallucination = (
            row[
                "hallucination_rate"
            ]
            * 100
        )


        print(
            f"{model:<6}"
            f" | Overall: {overall:>6.2f}%"
            f" | Abstention failure: {hallucination:>6.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("NEXAFLOW EVALUATION VISUALIZATION")
    print("=" * 70)


    validate_input_file()


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    df = load_results()


    print_summary(
        df
    )


    print("\nGenerating graphs...\n")


    plot_overall_comparison(
        df
    )


    plot_category_comparison(
        df
    )


    plot_hallucination_comparison(
        df
    )


    plot_key_behavior_metrics(
        df
    )


    plot_runtime(
        df
    )


    print("\n")
    print("=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70)

    print(
        f"\nGraphs saved to:\n{OUTPUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()