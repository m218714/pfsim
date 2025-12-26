import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_trades(strategyfile, weight):
    df = pd.read_csv(strategyfile, parse_dates=["Entry time", "Exit time"])

    # Clean the profit column - remove $, (, ), and commas, then convert to float
    df["Profit"] = weight * df["Profit"].str.replace("$", "").str.replace(
        ",", ""
    ).str.replace("(", "-").str.replace(")", "").astype(float)

    # Aggregate by date
    df = (
        df.groupby(pd.Grouper(key="Exit time", freq="D"))
        .agg({"Profit": "sum"})
        .reset_index()
    )

    df["Equity"] = df["Profit"].cumsum()
    df["Peak"] = df["Equity"].cummax()
    df["Drawdown"] = df["Equity"] - df["Peak"]

    return df


def merge_trades(dfs):
    df = pd.concat(dfs, ignore_index=True)
    df = (
        df.groupby(pd.Grouper(key="Exit time", freq="D"))
        .agg({"Profit": "sum"})
        .sort_values("Exit time")
        .reset_index()
    )

    df["Equity"] = df["Profit"].cumsum()
    df["Peak"] = df["Equity"].cummax()
    df["Drawdown"] = df["Equity"] - df["Peak"]

    return df


def run_simulation(df, template):
    target = template["target"]
    account_cost = template["account_cost"]
    payout_share = template["payout_share"]
    minimum_payout = template["payout_min"]
    maximum_payout = template["payout_max"]
    buffer = template["buffer"]
    processing_days = template["processing_days"]
    consistency = template["consistency"]
    trading_days_profit = template["trading_days_profit"]
    trading_days_min = template["trading_days_min"]
    trading_day_min_profit = template["min_profit_day"]
    trailing_dd = template["trailing_dd"]
    account_size = template["account_size"]
    account_cash = template["account_size"]
    account_max = template["account_size"]
    account_blow = template["account_size"] - template["trailing_dd"]
    account_cash_series = []
    account_max_series = []
    account_blow_series = []
    events_blow = []
    events_payout = []
    trades_since_last_payout = []

    target_reached = False
    next_allowed_trade = df.iloc[0]["Exit time"]

    def get_account_fig(df, events_blow, events_payout):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["Exit time"],
                y=df["Account Cash"],
                mode="lines",
                name="Account Cash",
                line=dict(color="blue"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["Exit time"],
                y=df["Account Blow"],
                mode="lines",
                name="Trailing Drawdown",
                line=dict(color="red", dash="dash"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["Exit time"],
                y=df["Account Max"],
                mode="lines",
                name="Account Max",
                line=dict(color="green", dash="dash"),
            )
        )

        blow_mask = df["Exit time"].isin(events_blow)
        fig.add_trace(
            go.Scatter(
                x=df.loc[blow_mask, "Exit time"],
                y=df.loc[blow_mask, "Account Cash"],
                mode="markers",
                name="Blow up",
                marker=dict(color="red", size=16),
            )
        )

        payout_dates = [d for d, _ in events_payout]
        payout_mask = df["Exit time"].isin(payout_dates)
        fig.add_trace(
            go.Scatter(
                x=df.loc[payout_mask, "Exit time"],
                y=df.loc[payout_mask, "Account Cash"],
                mode="markers",
                name="Payout",
                marker=dict(color="green", size=16),
            )
        )

        fig.update_layout(
            title="Account",
            xaxis_title="Exit time",
            yaxis_title="Value",
            height=750,
        )

        return fig

    def get_compared_fig(dates, values, df):
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=np.cumsum(values),
                mode="lines+markers",
                name="Prop firm return",
                line=dict(color="blue"),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["Exit time"],
                y=np.cumsum(df["Profit"]),
                mode="lines",
                name="Brokerage account return",
            )
        )

        fig.update_layout(
            title="Compared PnL",
            xaxis_title="Date",
            yaxis_title="Cumulative PnL",
            height=750,
        )

        return fig

    def can_payout(trades):
        try:
            if len(trades) < trading_days_min:
                return (
                    False,
                    f"Not enough trading days ({len(trades)} vs {trading_days_min})",
                )
            if (
                len([x for x in trades if x >= trading_day_min_profit])
                < trading_days_profit
            ):
                return (
                    False,
                    f"Not enough positive days ({len([x for x in trades if x > trading_day_min_profit])} vs {trading_days_profit})",
                )
            total_profit = sum(trades)
            # total_profit = sum([x for x in trades if x > trading_day_min_profit])
            max_profit = max(trades)
            if max_profit / total_profit >= consistency:
                if total_profit != 0:
                    return (
                        False,
                        f"Consistency not reached ({max_profit / total_profit:.2f} vs {consistency}: max={max_profit}, total={total_profit})",
                    )
                else:
                    return (False, f"Consistency not reached")
        except:
            return False, None
        return True, None

    def get_max_consecutive_payouts(l):
        max_len = 0
        current_len = 0
        for n in l:
            if n > 0:
                current_len += 1
                max_len = max(max_len, current_len)
            else:
                current_len = 0
        return max_len

    for i in range(len(df)):
        if df.iloc[i]["Exit time"] < next_allowed_trade:
            account_cash_series.append(account_cash)
            account_max_series.append(account_max)
            account_blow_series.append(account_blow)
            continue

        account_cash += df["Profit"].iloc[i]
        trades_since_last_payout.append(df["Profit"].iloc[i])

        if account_cash >= account_max:
            account_max = account_cash
            account_blow = account_max - trailing_dd
            if (
                account_blow >= account_size + buffer
            ):  # No trailing above initial balance + buffer
                target_reached = True
            if target_reached:
                account_blow = account_size + buffer

        if account_cash <= account_blow:
            events_blow.append(df.iloc[i]["Exit time"])
            account_cash = account_size
            account_max = account_size
            account_blow = account_size - trailing_dd
            del trades_since_last_payout[:]
            target_reached = False

        if target_reached and account_cash >= account_size + target + buffer:
            go_payout, _ = can_payout(trades_since_last_payout)
            if go_payout:
                payout = min(maximum_payout, account_cash - (account_size + buffer))
                if payout >= minimum_payout:
                    events_payout.append(
                        (df.iloc[i]["Exit time"], payout * payout_share)
                    )
                    account_cash = account_cash - payout
                    account_max = account_cash
                    del trades_since_last_payout[:]
                    next_allowed_trade = df.iloc[i]["Exit time"] + pd.offsets.BDay(
                        processing_days
                    )

        account_cash_series.append(account_cash)
        account_max_series.append(account_max)
        account_blow_series.append(account_blow)

    df["Account Cash"] = account_cash_series
    df["Account Blow"] = account_blow_series
    df["Account Max"] = account_max_series

    total_payouts = sum([x for _, x in events_payout])
    total_accounts = len(events_blow) + 1
    total_cost = total_accounts * account_cost

    equity_events = []
    events_log = []
    for date in events_blow:
        equity_events.append((date, -1 * account_cost))
        events_log.append(
            {"Event": "Account blown", "Date": date, "Amount": -1 * account_cost}
        )
    for date, payout in events_payout:
        equity_events.append((date, payout))
        events_log.append({"Event": "Payout", "Date": date, "Amount": payout})
    try:
        dates, values = zip(*sorted(equity_events, key=lambda x: x[0]))
        events_log = sorted(events_log, key=lambda x: x["Date"])

        account_fig = get_account_fig(df, events_blow, events_payout)
        compared_fig = get_compared_fig(dates, values, df)

        return (
            account_fig,
            compared_fig,
            pd.DataFrame(events_log),
            {
                "total_payouts_pnl": total_payouts,
                "total_accounts": total_accounts,
                "payouts_count": len(events_payout),
                "payouts_consecutive_max": get_max_consecutive_payouts(values),
                "total_cost": total_cost,
                "first_payout": events_payout[0][0],
                "last_payout": events_payout[-1][0],
                "performance": 100 * total_payouts / sum(df["Profit"]),
                "max_drawdown": min(df["Drawdown"]),
                "portfolio_pnl": sum(df["Profit"]),
            },
        )
    except Exception as e:
        import traceback

        return None, None, None, {"e": e, "trace": traceback.format_exc()}


def get_strategies_fig(dfs, merge_df):
    fig = go.Figure()

    for i, df in enumerate(dfs):
        fig.add_trace(
            go.Scatter(
                x=df["Exit time"], y=np.cumsum(df["Profit"]), mode="lines", name=f"S{i}"
            )
        )

    fig.add_trace(
        go.Scatter(
            x=merge_df["Exit time"],
            y=np.cumsum(merge_df["Profit"]),
            mode="lines",
            name="Combined",
        )
    )

    fig.update_layout(
        title="Strategies PnL",
        xaxis_title="Exit time",
        yaxis_title="Cumulative PnL",
        height=750,
    )

    return fig


def process(dataset, template):
    dfs = []
    for file, weight in dataset:
        dfs.append(load_trades(file, weight))
    df = merge_trades(dfs)

    strategies_fig = get_strategies_fig(dfs, df)

    account_fig, compared_fig, events_log, summary = run_simulation(df, template)

    return strategies_fig, account_fig, compared_fig, events_log, summary
