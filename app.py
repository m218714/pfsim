import pandas as pd
import streamlit as st

from analysis import process  # extracted notebook code

st.title("pfsim")

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    with st.form("inputs"):
        run = st.form_submit_button("Run")

        st.header("Portfolio")

        datasets = []
        for i in range(1, 4):
            col1, col2 = st.columns([3, 1])

            with col1:
                file = st.file_uploader(
                    f"Strategy #{i} csv", type="csv", key=f"file_{i}"
                )

            with col2:
                weight = st.number_input(
                    f"Weight {i}",
                    value=1,
                    step=1,
                    min_value=1,
                    max_value=10,
                    key=f"weight_{i}",
                )

            if file:
                datasets.append((file, weight))

        st.header("Apex")

        account_size = st.number_input(
            "Account Size",
            value=50000,
            min_value=25000,
            max_value=300000,
        )

        trailing_dd = st.number_input(
            "Trailing Drawdown",
            value=2500,
            min_value=1250,
        )

        buffer = st.number_input(
            "Buffer",
            value=100,
            min_value=50,
        )

        target = st.number_input(
            "Target",
            value=2500,
            min_value=1250,
        )

        payout_min = st.number_input(
            "Minimum Payout",
            value=500,
            min_value=250,
        )

        payout_max = st.number_input(
            "Maximum Payout",
            value=2500,
            min_value=1250,
        )

        processing_days = st.number_input(
            "Payout Processing Days",
            value=2,
            min_value=0,
            max_value=20,
        )

        account_cost = st.number_input(
            "Account Cost",
            value=150,
            min_value=1,
        )

        consistency = st.number_input(
            "Consistency",
            value=0.3,
            min_value=0.1,
            max_value=0.9,
            step=0.1,
        )

        trading_days_min = st.number_input(
            "Minimum Trading Days",
            value=8,
            min_value=1,
            max_value=25,
        )

        trading_days_profit = st.number_input(
            "Minimum Profit Days",
            value=5,
            min_value=1,
            max_value=25,
        )

        min_profit_day = st.number_input(
            "Minimum Profit Day",
            value=50,
            min_value=0,
            max_value=5000,
        )

        payout_share = st.number_input(
            "Payout Share",
            value=0.9,
            min_value=0.1,
            max_value=1.0,
            step=0.1,
        )

if run and datasets:
    template = {
        "account_size": account_size,
        "trailing_dd": trailing_dd,
        "buffer": buffer,
        "target": target,
        "payout_min": payout_min,
        "payout_max": payout_max,
        "processing_days": processing_days,
        "account_cost": account_cost,
        "consistency": consistency,
        "trading_days_min": trading_days_min,
        "trading_days_profit": trading_days_profit,
        "min_profit_day": min_profit_day,
        "payout_share": payout_share,
    }
    strategies_fig, account_fig, compared_fig, events_log, summary = process(
        datasets, template
    )

    st.write("Summary")
    summary_text = f"""
Total payouts = {summary["total_payouts_pnl"]:.2f} ({summary["payouts_consecutive_max"]} max consecutive payouts, {summary["payouts_count"]} in total)
Total cost = {summary["total_cost"]:.2f} ({summary["total_accounts"]} accounts used)
Total PnL = {summary["total_payouts_pnl"] - summary["total_cost"]:.2f}

Portfolio PnL = {summary["portfolio_pnl"]:.2f}
Prop vs Broker = {summary["total_payouts_pnl"] - summary["portfolio_pnl"]:.2f} ({summary["performance"]:.2f}% of portfolio returns)
    """
    st.code(summary_text)

    st.pyplot(account_fig)
    st.dataframe(events_log)
    st.pyplot(compared_fig)

    st.pyplot(strategies_fig)
