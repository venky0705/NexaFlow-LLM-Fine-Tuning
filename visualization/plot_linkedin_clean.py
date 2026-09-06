from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# PATHS
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
    / "linkedin_clean"
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
# LOAD DATA
# ============================================================

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation summary not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    df["model"] = (
        df["model"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

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
# LABELS
# ============================================================

def get_labels(df):

    return [
        MODEL_LABELS.get(
            str(model),
            str(model).upper(),
        )
        for model in df["model"].astype(str)
    ]


# ============================================================
# SAVE
# ============================================================

def save_chart(filename):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    plt.savefig(
        output_path,
        dpi=450,
        bbox_inches="tight",
        facecolor="white",
    )

    print(
        f"Saved: {output_path}"
    )

    plt.close()


# ============================================================
# CHART 1
#
# OVERALL PERFORMANCE
# ============================================================

def plot_overall_performance(df):

    labels = get_labels(df)

    values = (
        df[
            "overall_behavior_accuracy"
        ]
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(12, 7.2)
    )

    # --------------------------------------------------------
    # LINE
    # --------------------------------------------------------

    ax.plot(
        labels,
        values,
        marker="o",
        linewidth=3.2,
        markersize=10,
    )

    # --------------------------------------------------------
    # MAIN TITLE
    # --------------------------------------------------------

    fig.suptitle(
        "How Fine-Tuning Changed Model Behavior",
        fontsize=23,
        fontweight="bold",
        y=0.97,
    )

    # --------------------------------------------------------
    # SUBTITLE
    #
    # Deliberately separated from title
    # --------------------------------------------------------

    ax.set_title(
        "NexaFlow Technologies · Qwen2.5-1.5B",
        fontsize=13,
        pad=18,
    )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    ax.set_xlabel(
        "Training Stage",
        fontsize=13,
        labelpad=10,
    )

    ax.set_ylabel(
        "Automatic Behavior Accuracy (%)",
        fontsize=13,
        labelpad=10,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.tick_params(
        axis="both",
        labelsize=12,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    # --------------------------------------------------------
    # VALUE LABELS
    # --------------------------------------------------------

    for index, value in enumerate(values):

        ax.annotate(
            f"{value:.1f}%",
            xy=(
                index,
                value,
            ),
            xytext=(
                0,
                14,
            ),
            textcoords="offset points",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )

    # --------------------------------------------------------
    # STORY NOTE
    # --------------------------------------------------------

    fig.text(
        0.5,
        0.02,
        (
            "SFT produced the largest improvement; "
            "DPO preserved most of the gain but did not improve the overall score."
        ),
        ha="center",
        fontsize=11,
    )

    # --------------------------------------------------------
    # LEAVE SPACE FOR TITLE + FOOTER
    # --------------------------------------------------------

    fig.subplots_adjust(
        top=0.82,
        bottom=0.16,
        left=0.11,
        right=0.97,
    )

    save_chart(
        "01_linkedin_overall_performance.png"
    )


# ============================================================
# CHART 2
#
# CAPABILITY COMPARISON
# ============================================================

def plot_capability_comparison(df):

    labels = get_labels(df)

    metrics = {
        "Policy":
            "policy_accuracy",

        "Scenario":
            "scenario_accuracy",

        "Unsupported Handling":
            "unsupported_handling_accuracy",

        "False Premise":
            "false_premise_accuracy",

        "Safety":
            "harmful_refusal_accuracy",
    }

    fig, ax = plt.subplots(
        figsize=(12, 7.5)
    )

    # --------------------------------------------------------
    # LINES
    # --------------------------------------------------------

    for label, column in metrics.items():

        values = (
            df[column]
            * 100
        )

        ax.plot(
            labels,
            values,
            marker="o",
            linewidth=2.4,
            markersize=7,
            label=label,
        )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    fig.suptitle(
        "Capability Changes Across Fine-Tuning Stages",
        fontsize=22,
        fontweight="bold",
        y=0.97,
    )

    # --------------------------------------------------------
    # SUBTITLE
    # --------------------------------------------------------

    ax.set_title(
        "NexaFlow Technologies · Automatic evaluation",
        fontsize=13,
        pad=18,
    )

    # --------------------------------------------------------
    # AXES
    # --------------------------------------------------------

    ax.set_xlabel(
        "Training Stage",
        fontsize=13,
        labelpad=10,
    )

    ax.set_ylabel(
        "Accuracy (%)",
        fontsize=13,
        labelpad=10,
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.tick_params(
        axis="both",
        labelsize=12,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    # --------------------------------------------------------
    # LEGEND BELOW GRAPH
    # --------------------------------------------------------

    ax.legend(
        frameon=False,
        fontsize=11,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            -0.12,
        ),
    )

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    fig.subplots_adjust(
        top=0.82,
        bottom=0.22,
        left=0.11,
        right=0.97,
    )

    save_chart(
        "02_linkedin_capability_comparison.png"
    )


# ============================================================
# CHART 3
#
# ABSTENTION FAILURE
# ============================================================

def plot_abstention_failure(df):

    labels = get_labels(df)

    values = (
        df[
            "hallucination_rate"
        ]
        * 100
    )

    fig, ax = plt.subplots(
        figsize=(12, 7.2)
    )

    ax.plot(
        labels,
        values,
        marker="o",
        linewidth=3.2,
        markersize=10,
    )

    fig.suptitle(
        "Unsupported-Answer Failures Dropped After SFT",
        fontsize=22,
        fontweight="bold",
        y=0.97,
    )

    ax.set_title(
        "Measured on unsupported and unknown-person evaluation cases",
        fontsize=13,
        pad=18,
    )

    ax.set_xlabel(
        "Training Stage",
        fontsize=13,
        labelpad=10,
    )

    ax.set_ylabel(
        "Abstention Failure Rate (%)",
        fontsize=13,
        labelpad=10,
    )

    ax.set_ylim(
        0,
        100,
    )

    ax.tick_params(
        axis="both",
        labelsize=12,
    )

    ax.grid(
        axis="y",
        alpha=0.18,
    )

    for index, value in enumerate(values):

        ax.annotate(
            f"{value:.0f}%",
            xy=(
                index,
                value,
            ),
            xytext=(
                0,
                14,
            ),
            textcoords="offset points",
            ha="center",
            fontsize=14,
            fontweight="bold",
        )

    fig.text(
        0.5,
        0.02,
        (
            "Lower is better · This operational metric covers unsupported "
            "and unknown-person cases only."
        ),
        ha="center",
        fontsize=10.5,
    )

    fig.subplots_adjust(
        top=0.82,
        bottom=0.16,
        left=0.11,
        right=0.97,
    )

    save_chart(
        "03_linkedin_abstention_failure.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("NEXAFLOW CLEAN LINKEDIN VISUALIZATIONS")
    print("=" * 70)

    df = load_data()

    print(
        "\nGenerating charts...\n"
    )

    plot_overall_performance(
        df
    )

    plot_capability_comparison(
        df
    )

    plot_abstention_failure(
        df
    )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"\nSaved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":

    main()