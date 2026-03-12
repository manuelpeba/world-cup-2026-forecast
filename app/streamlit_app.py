from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


DEFAULT_OUTPUT_DIR = Path("data/outputs/simulation")


@st.cache_data
def load_team_probabilities(output_dir: Path) -> pd.DataFrame:
    csv_path = output_dir / "team_probabilities.csv"
    parquet_path = output_dir / "team_probabilities.parquet"

    if csv_path.exists():
        return pd.read_csv(csv_path)

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    raise FileNotFoundError(
        f"Could not find team probabilities file in: {output_dir}"
    )


@st.cache_data
def load_champion_distribution(output_dir: Path) -> pd.DataFrame:
    csv_path = output_dir / "champion_distribution.csv"
    parquet_path = output_dir / "champion_distribution.parquet"

    if csv_path.exists():
        return pd.read_csv(csv_path)

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    raise FileNotFoundError(
        f"Could not find champion distribution file in: {output_dir}"
    )


@st.cache_data
def load_summary_metadata(output_dir: Path) -> dict:
    json_path = output_dir / "summary_metadata.json"

    if not json_path.exists():
        return {}

    with open(json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def format_percentage_columns(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()

    probability_columns = [col for col in formatted.columns if col.endswith("_prob")]
    for col in probability_columns:
        formatted[col] = (formatted[col] * 100).round(2)

    return formatted


def build_top_champion_table(team_probabilities: pd.DataFrame, top_n: int) -> pd.DataFrame:
    df = team_probabilities.copy()

    keep_columns = [
        col for col in [
            "team",
            "advance_from_group_prob",
            "quarterfinal_prob",
            "semifinal_prob",
            "final_prob",
            "champion_prob",
        ]
        if col in df.columns
    ]

    df = df.loc[:, keep_columns]
    df = df.sort_values("champion_prob", ascending=False).head(top_n)
    return format_percentage_columns(df)


def build_team_detail_row(team_probabilities: pd.DataFrame, team_name: str) -> pd.DataFrame:
    row = team_probabilities.loc[team_probabilities["team"] == team_name].copy()
    if row.empty:
        return row

    return format_percentage_columns(row)


def render_header(metadata: dict) -> None:
    st.title("World Cup 2026 Forecast")
    st.caption(
        "Production-style football forecasting dashboard built on a probabilistic "
        "match model + Monte Carlo tournament simulation engine."
    )

    if metadata:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tournament", metadata.get("tournament_name", "N/A"))
        col2.metric("Model", metadata.get("model_name", "N/A"))
        col3.metric("Simulations", metadata.get("num_simulations", "N/A"))
        col4.metric("Knockout resolution", metadata.get("knockout_draw_resolution", "N/A"))


def render_overview_metrics(team_probabilities: pd.DataFrame) -> None:
    if team_probabilities.empty:
        return

    top_team = team_probabilities.sort_values("champion_prob", ascending=False).iloc[0]
    second_team = team_probabilities.sort_values("champion_prob", ascending=False).iloc[1]
    avg_champion_prob = team_probabilities["champion_prob"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Top favorite", top_team["team"])
    col2.metric("Top champion probability", f"{top_team['champion_prob'] * 100:.2f}%")
    col3.metric("2nd favorite", f"{second_team['team']} ({second_team['champion_prob'] * 100:.2f}%)")

    st.info(
        f"Average champion probability across teams: {avg_champion_prob * 100:.2f}% "
        "(sanity check for probability distribution spread)."
    )


def render_champion_chart(team_probabilities: pd.DataFrame, top_n: int) -> None:
    chart_df = (
        team_probabilities[["team", "champion_prob"]]
        .sort_values("champion_prob", ascending=False)
        .head(top_n)
        .set_index("team")
    )

    st.subheader("Champion Probability")
    st.bar_chart(chart_df)


def render_advancement_table(team_probabilities: pd.DataFrame, top_n: int) -> None:
    st.subheader("Top Teams by Champion Probability")
    table_df = build_top_champion_table(team_probabilities, top_n=top_n)
    st.dataframe(table_df, use_container_width=True, hide_index=True)


def render_team_selector(team_probabilities: pd.DataFrame) -> None:
    st.subheader("Team Detail")

    teams = sorted(team_probabilities["team"].unique().tolist())
    selected_team = st.selectbox("Select a national team", teams)

    detail_df = build_team_detail_row(team_probabilities, selected_team)
    if detail_df.empty:
        st.warning("No data available for the selected team.")
        return

    row = detail_df.iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("Advance from group", f"{row.get('advance_from_group_prob', 0):.2f}%")
    col2.metric("Reach final", f"{row.get('final_prob', 0):.2f}%")
    col3.metric("Become champion", f"{row.get('champion_prob', 0):.2f}%")

    display_columns = [
        col for col in [
            "team",
            "group_stage_exit_prob",
            "advance_from_group_prob",
            "quarterfinal_prob",
            "semifinal_prob",
            "final_prob",
            "champion_prob",
        ]
        if col in detail_df.columns
    ]

    st.dataframe(detail_df.loc[:, display_columns], use_container_width=True, hide_index=True)


def render_champion_distribution(champion_distribution: pd.DataFrame, top_n: int) -> None:
    st.subheader("Champion Distribution")

    df = champion_distribution.copy().sort_values("champion_prob", ascending=False).head(top_n)
    if "champion_prob" in df.columns:
        df["champion_prob"] = (df["champion_prob"] * 100).round(2)

    st.dataframe(df, use_container_width=True, hide_index=True)


def render_raw_download_section(
    team_probabilities: pd.DataFrame,
    champion_distribution: pd.DataFrame,
    metadata: dict,
) -> None:
    st.subheader("Export Preview")

    with st.expander("Team probabilities preview"):
        st.dataframe(format_percentage_columns(team_probabilities.head(20)), use_container_width=True, hide_index=True)

    with st.expander("Champion distribution preview"):
        preview_df = champion_distribution.copy()
        if "champion_prob" in preview_df.columns:
            preview_df["champion_prob"] = (preview_df["champion_prob"] * 100).round(2)
        st.dataframe(preview_df.head(20), use_container_width=True, hide_index=True)

    with st.expander("Summary metadata"):
        st.json(metadata)


def main() -> None:
    st.set_page_config(
        page_title="World Cup 2026 Forecast",
        page_icon="⚽",
        layout="wide",
    )

    st.sidebar.header("Configuration")
    output_dir_str = st.sidebar.text_input(
        "Simulation output directory",
        value=str(DEFAULT_OUTPUT_DIR),
    )
    top_n = st.sidebar.slider("Top N teams", min_value=5, max_value=20, value=10, step=1)

    output_dir = Path(output_dir_str)

    try:
        team_probabilities = load_team_probabilities(output_dir)
        champion_distribution = load_champion_distribution(output_dir)
        metadata = load_summary_metadata(output_dir)
    except Exception as exc:
        st.error(f"Failed to load simulation outputs: {exc}")
        st.stop()

    render_header(metadata)
    render_overview_metrics(team_probabilities)

    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        render_champion_chart(team_probabilities, top_n=top_n)
        render_advancement_table(team_probabilities, top_n=top_n)

    with col_right:
        render_team_selector(team_probabilities)
        render_champion_distribution(champion_distribution, top_n=top_n)

    render_raw_download_section(
        team_probabilities=team_probabilities,
        champion_distribution=champion_distribution,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()