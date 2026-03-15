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


@st.cache_data
def load_match_logs(output_dir: Path) -> pd.DataFrame:
    parquet_path = output_dir / "match_logs.parquet"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    return pd.DataFrame()


def format_probability_columns_as_percent(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()

    probability_columns = [col for col in formatted.columns if col.endswith("_prob")]
    for col in probability_columns:
        formatted[col] = (formatted[col] * 100).round(2)

    return formatted


def get_probability_display_columns(team_probabilities: pd.DataFrame) -> list[str]:
    return [
        col
        for col in [
            "team",
            "advance_from_group_prob",
            "round_of_32_prob",
            "round_of_16_prob",
            "quarterfinal_prob",
            "semifinal_prob",
            "final_prob",
            "champion_prob",
        ]
        if col in team_probabilities.columns
    ]


def get_team_detail_columns(team_probabilities: pd.DataFrame) -> list[str]:
    return [
        col
        for col in [
            "team",
            "group_stage_exit_prob",
            "advance_from_group_prob",
            "round_of_32_prob",
            "round_of_16_prob",
            "quarterfinal_prob",
            "semifinal_prob",
            "final_prob",
            "champion_prob",
        ]
        if col in team_probabilities.columns
    ]


def get_progression_probability_columns(team_probabilities: pd.DataFrame) -> list[str]:
    return [
        col
        for col in [
            "advance_from_group_prob",
            "round_of_32_prob",
            "round_of_16_prob",
            "quarterfinal_prob",
            "semifinal_prob",
            "final_prob",
            "champion_prob",
        ]
        if col in team_probabilities.columns
    ]


def build_metric_table(team_probabilities: pd.DataFrame, top_n: int) -> pd.DataFrame:
    keep_columns = get_probability_display_columns(team_probabilities)

    df = (
        team_probabilities.loc[:, keep_columns]
        .sort_values("champion_prob", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    return format_probability_columns_as_percent(df)


def render_header(metadata: dict) -> None:
    st.title("World Cup 2026 Forecast Dashboard")
    st.caption(
        "Production-style football forecasting system based on probabilistic "
        "match prediction and Monte Carlo tournament simulation."
    )

    if not metadata:
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tournament", metadata.get("tournament_name", "N/A"))
    col2.metric("Model", metadata.get("model_name", "N/A"))
    col3.metric("Simulations", f"{metadata.get('num_simulations', 'N/A')}")
    col4.metric("Knockout rule", metadata.get("knockout_draw_resolution", "N/A"))

    if metadata.get("initial_knockout_round"):
        st.caption(
            f"Initial knockout round: {metadata.get('initial_knockout_round')}"
        )


def render_overview_metrics(team_probabilities: pd.DataFrame) -> None:
    ordered = team_probabilities.sort_values("champion_prob", ascending=False).reset_index(drop=True)

    if ordered.empty:
        return

    top_team = ordered.iloc[0]
    second_team = ordered.iloc[1] if len(ordered) > 1 else ordered.iloc[0]
    avg_champion_prob = ordered["champion_prob"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Top favorite", top_team["team"])
    col2.metric("Champion probability", f"{top_team['champion_prob'] * 100:.2f}%")
    col3.metric(
        "2nd favorite",
        f"{second_team['team']} ({second_team['champion_prob'] * 100:.2f}%)",
    )

    st.info(
        f"Average champion probability across all teams: "
        f"{avg_champion_prob * 100:.2f}%."
    )


def render_champion_probability_chart(team_probabilities: pd.DataFrame, top_n: int) -> None:
    chart_df = (
        team_probabilities[["team", "champion_prob"]]
        .sort_values("champion_prob", ascending=False)
        .head(top_n)
        .set_index("team")
    )

    st.subheader("Champion Probability")
    st.bar_chart(chart_df)


def render_champion_leaderboard(team_probabilities: pd.DataFrame, top_n: int) -> None:
    st.subheader("Champion Probability Leaderboard")

    df = (
        team_probabilities[["team", "champion_prob"]]
        .sort_values("champion_prob", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    df["rank"] = df.index + 1
    df["champion_prob"] = (df["champion_prob"] * 100).round(2)

    df = df.loc[:, ["rank", "team", "champion_prob"]]
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_probability_table(team_probabilities: pd.DataFrame, top_n: int) -> None:
    st.subheader("Top Teams by Champion Probability")
    df = build_metric_table(team_probabilities, top_n=top_n)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_team_progression_chart(row_pct: pd.Series) -> None:
    stage_order = [
        ("advance_from_group_prob", "Advance from Group"),
        ("round_of_32_prob", "Round of 32"),
        ("round_of_16_prob", "Round of 16"),
        ("quarterfinal_prob", "Quarterfinal"),
        ("semifinal_prob", "Semifinal"),
        ("final_prob", "Final"),
        ("champion_prob", "Champion"),
    ]

    records = []
    for col, label in stage_order:
        if col in row_pct.index:
            records.append({"stage": label, "probability": row_pct[col]})

    if not records:
        return

    chart_df = pd.DataFrame(records).set_index("stage")
    st.subheader("Progression Profile")
    st.bar_chart(chart_df)


def render_team_detail(team_probabilities: pd.DataFrame) -> None:
    st.subheader("Team Explorer")

    teams = sorted(team_probabilities["team"].unique().tolist())
    selected_team = st.selectbox("Select team", teams)

    row = team_probabilities.loc[team_probabilities["team"] == selected_team].copy()
    if row.empty:
        st.warning("No data available for selected team.")
        return

    row = row.reset_index(drop=True)
    row_pct = format_probability_columns_as_percent(row)
    detail = row_pct.iloc[0]

    metric_columns = st.columns(4 if "round_of_32_prob" in row_pct.columns else 3)

    metric_columns[0].metric(
        "Advance from group",
        f"{detail.get('advance_from_group_prob', 0):.2f}%",
    )

    if "round_of_32_prob" in row_pct.columns:
        metric_columns[1].metric(
            "Reach Round of 16",
            f"{detail.get('round_of_16_prob', 0):.2f}%",
        )
        metric_columns[2].metric(
            "Reach final",
            f"{detail.get('final_prob', 0):.2f}%",
        )
        metric_columns[3].metric(
            "Become champion",
            f"{detail.get('champion_prob', 0):.2f}%",
        )
    else:
        metric_columns[1].metric(
            "Reach final",
            f"{detail.get('final_prob', 0):.2f}%",
        )
        metric_columns[2].metric(
            "Become champion",
            f"{detail.get('champion_prob', 0):.2f}%",
        )

    probability_cols = get_team_detail_columns(row_pct)

    st.dataframe(
        row_pct.loc[:, probability_cols],
        use_container_width=True,
        hide_index=True,
    )

    render_team_progression_chart(detail)


def render_stacked_progression_chart(team_probabilities: pd.DataFrame, top_n: int) -> None:
    st.subheader("Stage Progression Profile (Top Teams)")

    stage_columns = get_progression_probability_columns(team_probabilities)
    if not stage_columns:
        st.caption("No progression probability columns available.")
        return

    df = (
        team_probabilities[["team"] + stage_columns]
        .sort_values("champion_prob", ascending=False)
        .head(top_n)
        .set_index("team")
        .copy()
    )

    rename_map = {
        "advance_from_group_prob": "Advance from Group",
        "round_of_32_prob": "Round of 32",
        "round_of_16_prob": "Round of 16",
        "quarterfinal_prob": "Quarterfinal",
        "semifinal_prob": "Semifinal",
        "final_prob": "Final",
        "champion_prob": "Champion",
    }

    df = df.rename(columns=rename_map)
    st.bar_chart(df)


def render_team_comparison(team_probabilities: pd.DataFrame) -> None:
    st.subheader("Team Comparison")

    teams = sorted(team_probabilities["team"].unique().tolist())
    if len(teams) < 2:
        st.caption("Not enough teams available for comparison.")
        return

    col1, col2 = st.columns(2)
    team_a = col1.selectbox("Team A", teams, index=0, key="compare_team_a")
    default_b_index = 1 if len(teams) > 1 else 0
    team_b = col2.selectbox("Team B", teams, index=default_b_index, key="compare_team_b")

    row_a = team_probabilities.loc[team_probabilities["team"] == team_a].copy()
    row_b = team_probabilities.loc[team_probabilities["team"] == team_b].copy()

    if row_a.empty or row_b.empty:
        st.warning("Could not load data for one or both selected teams.")
        return

    progression_cols = get_progression_probability_columns(team_probabilities)

    records = []
    label_map = {
        "advance_from_group_prob": "Advance from Group",
        "round_of_32_prob": "Round of 32",
        "round_of_16_prob": "Round of 16",
        "quarterfinal_prob": "Quarterfinal",
        "semifinal_prob": "Semifinal",
        "final_prob": "Final",
        "champion_prob": "Champion",
    }

    for col in progression_cols:
        records.append(
            {
                "stage": label_map[col],
                team_a: float(row_a.iloc[0][col]),
                team_b: float(row_b.iloc[0][col]),
            }
        )

    chart_df = pd.DataFrame(records).set_index("stage")
    st.bar_chart(chart_df)

    comparison_table = chart_df.copy() * 100
    comparison_table = comparison_table.round(2)
    st.dataframe(comparison_table, use_container_width=True)


def render_champion_distribution(champion_distribution: pd.DataFrame, top_n: int) -> None:
    st.subheader("Champion Distribution")

    df = champion_distribution.copy()
    if "champion_prob" in df.columns:
        df["champion_prob"] = (df["champion_prob"] * 100).round(2)

    df = df.sort_values("champion_prob", ascending=False).head(top_n).reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_match_log_preview(match_logs: pd.DataFrame) -> None:
    st.subheader("Match Log Preview")

    if match_logs.empty:
        st.caption("No match_logs.parquet file found in the selected output directory.")
        return

    preview_columns = [
        col
        for col in [
            "simulation_id",
            "stage",
            "team_a",
            "team_b",
            "winner",
            "decided_by",
            "team_a_win_prob",
            "draw_prob",
            "team_a_loss_prob",
        ]
        if col in match_logs.columns
    ]

    preview = match_logs.loc[:, preview_columns].head(100).copy()

    prob_cols = [col for col in preview.columns if col.endswith("_prob")]
    for col in prob_cols:
        preview[col] = (preview[col] * 100).round(2)

    st.dataframe(preview, use_container_width=True, hide_index=True)


def render_methodology(metadata: dict, team_probabilities: pd.DataFrame) -> None:
    st.subheader("Methodology Snapshot")

    supports_round_of_32 = "round_of_32_prob" in team_probabilities.columns

    text_lines = [
        "- Historical international football match data",
        "- Team strength features: Elo, rolling goals, rolling win rate, rolling points",
        "- Match outcome model: probabilistic win/draw/loss prediction",
        "- Tournament engine: Monte Carlo simulation across full tournament runs",
        "- Aggregation: stage-level advancement probabilities by national team",
    ]

    if supports_round_of_32:
        text_lines.append("- Tournament format: 48 teams, 12 groups, best third-placed teams, Round of 32")
    else:
        text_lines.append("- Tournament format: 32 teams, standard group stage and Round of 16")

    if metadata:
        tournament_name = metadata.get("tournament_name", "N/A")
        model_name = metadata.get("model_name", "N/A")
        simulations = metadata.get("num_simulations", "N/A")

        st.markdown(
            f"""
**Tournament:** {tournament_name}  
**Model:** {model_name}  
**Simulations:** {simulations}
"""
        )

    st.markdown("\n".join(text_lines))


def render_sidebar() -> tuple[Path, int]:
    st.sidebar.header("Controls")

    output_dir_str = st.sidebar.text_input(
        "Simulation output directory",
        value=str(DEFAULT_OUTPUT_DIR),
    )

    top_n = st.sidebar.slider(
        "Top N teams",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Tip: run the simulation first, then point the dashboard to the exported output folder."
    )

    return Path(output_dir_str), top_n


def main() -> None:
    st.set_page_config(
        page_title="World Cup 2026 Forecast",
        page_icon="⚽",
        layout="wide",
    )

    output_dir, top_n = render_sidebar()

    try:
        team_probabilities = load_team_probabilities(output_dir)
        champion_distribution = load_champion_distribution(output_dir)
        summary_metadata = load_summary_metadata(output_dir)
        match_logs = load_match_logs(output_dir)
    except Exception as exc:
        st.error(f"Failed to load simulation outputs: {exc}")
        st.stop()

    render_header(summary_metadata)
    render_overview_metrics(team_probabilities)

    col_left, col_right = st.columns([1.6, 1.0])

    with col_left:
        render_champion_probability_chart(team_probabilities, top_n=top_n)
        render_probability_table(team_probabilities, top_n=top_n)
        render_stacked_progression_chart(team_probabilities, top_n=top_n)

    with col_right:
        render_team_detail(team_probabilities)
        render_champion_distribution(champion_distribution, top_n=top_n)
        render_champion_leaderboard(team_probabilities, top_n=top_n)

    st.markdown("---")

    col_bottom_left, col_bottom_right = st.columns([1.2, 1.0])

    with col_bottom_left:
        render_match_log_preview(match_logs)

    with col_bottom_right:
        render_methodology(summary_metadata, team_probabilities)
        render_team_comparison(team_probabilities)


if __name__ == "__main__":
    main()

