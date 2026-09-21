"""
PSX All-In-One App
====================
ONE file, ONE command. Run this on YOUR OWN computer (needs normal internet
access to reach PSX's public data site).

    pip install psxdata pandas numpy
    python psx_app.py

It will:
  1. Scan PSX stocks (all of them, or just KSE-100 - your choice below)
  2. Score each one BUY / HOLD / AVOID, tag it LONG-TERM or SHORT-TERM/TRADE
  3. Check any stocks YOU already own (see MY_HOLDINGS below) and tell you
     whether to hold or sell, with a target price, a stop-loss, and a
     rough timeframe
  4. Build one dashboard.html and open it in your browser automatically

DATA SOURCE / AUTHENTICITY:
This pulls real data directly from PSX's own public data portal
(dps.psx.com.pk) via the open-source `psxdata` library - the same prices
PSX itself publishes. It refreshes roughly every 15 minutes. It does NOT
connect to your AKD account or any brokerage - it only reads public market
data, and it does not place any trades.

NOT FINANCIAL ADVICE. Rule-based signals from historical price/volume
patterns - a probability nudge, not a guarantee. Verify before every trade.
"""

import json
import os
import sys
import time
import traceback
import webbrowser
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import psxdata
except ImportError:
    sys.exit("Missing dependency. Run: pip install psxdata pandas numpy")


# ============================== CONFIG ====================================
CONFIG = {
    # Which universe to screen.
    #   None      -> ALL PSX-listed stocks (500+, slower, includes thin names)
    #   "KSE100"  -> just the 100 largest/most liquid stocks (fast)
    "index": None,

    "min_avg_daily_value_pkr": 2_000_000,   # skip stocks that barely trade
    "lookback_days": 200,

    # --- Risk management ---
    "total_capital": 1_000_000,   # PKR - set to your real account size
    "risk_per_trade_pct": 1.0,    # % of capital you'll risk losing per trade
    "max_position_pct": 10.0,     # cap on any single position

    # --- Signal thresholds ---
    "rsi_oversold": 35,
    "rsi_overbought": 70,

    # --- Top Picks ---
    "top_picks_count": 10,   # how many BUY-rated stocks to feature in "Today's Picks"

    # --- Live Refresh (optional) ---
    # Paste your own Google Apps Script Web App URL here (see psx_proxy.gs)
    # to enable the "Live Refresh" button and auto-refresh in the dashboard.
    # Leave as "" to disable - the button will explain how to set this up.
    "live_refresh_proxy_url": "",
}

# #############################################################################
# ##                                                                        ##
# ##   >>>  YOUR CURRENT HOLDINGS  —  ADD YOUR STOCKS HERE  <<<             ##
# ##                                                                        ##
# ##   List every stock you currently own. The app will tell you, for      ##
# ##   each one, whether to HOLD or SELL right now, plus a target price    ##
# ##   and a stop-loss. Leave the list empty ( [] ) if you own nothing.    ##
# ##                                                                        ##
# #############################################################################
MY_HOLDINGS = [
    # {"symbol": "LUCK", "buy_price": 550.0, "buy_date": "2026-07-15", "quantity": 100},
    # {"symbol": "OGDC", "buy_price": 120.0, "buy_date": "2026-08-01", "quantity": 300},

    # <<< ADD YOUR REAL STOCKS BELOW THIS LINE >>>

]
# #############################################################################
# ============================================================================


# --------------------------- Data fetching ---------------------------------

def fetch_universe(index_name):
    label = index_name or "ALL PSX-listed stocks"
    print(f"Fetching {label} from PSX...")
    tickers = psxdata.tickers(index=index_name)
    print(f"  -> {len(tickers)} tickers found")
    if index_name is None:
        print("  NOTE: scanning the full exchange takes longer than KSE-100 alone "
              "(likely 20-40+ minutes) since every ticker is its own request. "
              "Grab a coffee.")
    return tickers


def fetch_history(symbol, lookback_days):
    end = date.today()
    start = end - timedelta(days=lookback_days)
    try:
        df = psxdata.stocks(symbol, start=start, end=end)
        if df is None or df.empty or len(df) < 30:
            return None
        return df.sort_values("date").reset_index(drop=True)
    except Exception as e:
        print(f"  ! {symbol}: skipped ({e})")
        return None


def fetch_recent_filings(symbol, days=90):
    """
    Real PSX filing history for this company (quarterly/annual reports etc.),
    pulled straight from PSX's own data - not a prediction, just what the
    company itself has actually filed recently. Returns [] outside reporting
    season or if the lookup fails - this is a nice-to-have, never blocks the
    main scan if it errors.
    """
    try:
        df = psxdata.fundamentals(symbol)
        if df is None or df.empty:
            return []
        df = df.copy()
        df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
        cutoff = pd.Timestamp(date.today() - timedelta(days=days))
        recent = df[df["posting_date"] >= cutoff].sort_values("posting_date", ascending=False)
        return [
            {
                "type": str(row.get("type", "")),
                "period_ended": str(row.get("period_ended", "")),
                "posting_date": str(row["posting_date"].date()) if pd.notna(row["posting_date"]) else "",
            }
            for _, row in recent.head(5).iterrows()
        ]
    except Exception:
        return []  # don't let a filings lookup failure break the scan


def estimate_time_to_target(df, target_move_pct, max_lookahead_days=30):
    """
    NOT a prediction of when a target will be hit. This looks BACKWARD at this
    stock's own price history and asks: historically, when this stock moved by
    roughly this percentage, how many trading days did it typically take, and
    how often did a move this size happen at all within a reasonable window?
    Past patterns are not a guarantee of future timing - markets can gap,
    stall, or move the opposite direction entirely regardless of history.
    """
    closes = df["close"].values
    n = len(closes)
    if n < 20:
        return {"hit_rate_pct": None, "median_days": None, "sample_size": 0}

    days_taken = []
    total = 0
    for i in range(n - 1):
        start_price = closes[i]
        if start_price <= 0:
            continue
        total += 1
        target_i = start_price * (1 + target_move_pct / 100)
        window_end = min(i + 1 + max_lookahead_days, n)
        for j in range(i + 1, window_end):
            if (target_move_pct >= 0 and closes[j] >= target_i) or \
               (target_move_pct < 0 and closes[j] <= target_i):
                days_taken.append(j - i)
                break

    if not days_taken or total == 0:
        return {"hit_rate_pct": 0.0, "median_days": None, "sample_size": total}

    return {
        "hit_rate_pct": round(len(days_taken) / total * 100, 1),
        "median_days": int(np.median(days_taken)),
        "sample_size": total,
    }


# --------------------------- Indicators -------------------------------------

def sma(s, w): return s.rolling(w).mean()
def ema(s, w): return s.ewm(span=w, adjust=False).mean()

def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series):
    macd_line = ema(series, 12) - ema(series, 26)
    return macd_line, ema(macd_line, 9)

def atr(df, window=14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def compute_indicators(df):
    df = df.copy()
    df["sma20"] = sma(df["close"], 20)
    df["sma50"] = sma(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    df["macd"], df["macd_signal"] = macd(df["close"])
    df["atr14"] = atr(df, 14)
    df["vol_avg20"] = sma(df["volume"], 20)
    return df


# ----------------------------- Scoring --------------------------------------

def analyze_stock(symbol, df, cfg):
    df = compute_indicators(df)
    last = df.iloc[-1]
    if pd.isna(last["sma50"]) or pd.isna(last["rsi14"]) or pd.isna(last["atr14"]):
        return None

    price = last["close"]
    avg_daily_value = (df["close"] * df["volume"]).tail(20).mean()
    if avg_daily_value < cfg["min_avg_daily_value_pkr"]:
        return None

    reasons_bullish, reasons_bearish = [], []
    score = 0

    if price > last["sma20"] > last["sma50"]:
        score += 2; reasons_bullish.append("uptrend (price > SMA20 > SMA50)")
    elif price < last["sma20"] < last["sma50"]:
        score -= 2; reasons_bearish.append("downtrend (price < SMA20 < SMA50)")

    if last["rsi14"] < cfg["rsi_oversold"]:
        score += 1; reasons_bullish.append(f"RSI oversold ({last['rsi14']:.0f})")
    elif last["rsi14"] > cfg["rsi_overbought"]:
        score -= 1; reasons_bearish.append(f"RSI overbought ({last['rsi14']:.0f})")

    if last["macd"] > last["macd_signal"]:
        score += 1; reasons_bullish.append("MACD bullish crossover")
    else:
        score -= 1; reasons_bearish.append("MACD bearish crossover")

    if last["volume"] > 1.5 * last["vol_avg20"]:
        if price > df["close"].iloc[-2]:
            score += 1; reasons_bullish.append("high volume on up move")
        else:
            score -= 1; reasons_bearish.append("high volume on down move")

    volatility_pct = (last["atr14"] / price) * 100
    signal = "BUY" if score >= 3 else ("AVOID" if score <= -3 else "HOLD")

    # Horizon
    long_score, short_score, horizon_reasons = 0, 0, []
    sma50_20ago = df["sma50"].iloc[-20] if len(df) >= 20 and not pd.isna(df["sma50"].iloc[-20]) else None
    sma50_slope_pct = ((last["sma50"] - sma50_20ago) / sma50_20ago * 100) if sma50_20ago else 0

    if abs(sma50_slope_pct) > 3:
        long_score += 2; horizon_reasons.append(f"50-day trend has moved {sma50_slope_pct:+.1f}% over the last month")
    if volatility_pct < 3:
        long_score += 1; horizon_reasons.append("relatively low volatility")
    if volatility_pct > 3.5:
        short_score += 2; horizon_reasons.append("high volatility — big daily swings")
    if last["rsi14"] < cfg["rsi_oversold"] or last["rsi14"] > cfg["rsi_overbought"]:
        short_score += 1; horizon_reasons.append("RSI at an extreme — often short-lived")
    if last["volume"] > 1.5 * last["vol_avg20"]:
        short_score += 1; horizon_reasons.append("unusual volume today — check for news")

    horizon = "LONG-TERM" if long_score > short_score else ("SHORT-TERM/TRADE" if short_score > long_score else "EITHER")

    # ---- Dip Watch: is this a downtrend that's oversold enough to be a contrarian
    # bounce candidate? This is a DIFFERENT, riskier bet than the BUY signal above —
    # BUY means "the trend favors going up." Dip Watch means "this has fallen hard
    # and MIGHT bounce, but the trend still says down." Never guaranteed either way.
    is_dip_candidate = bool(
        price < last["sma20"] < last["sma50"]           # confirmed downtrend
        and last["rsi14"] < (cfg["rsi_oversold"] - 5)    # deeply oversold, not just mildly
    )
    dip_reasoning = None
    if is_dip_candidate:
        drop_from_sma50_pct = ((price - last["sma50"]) / last["sma50"]) * 100
        dip_reasoning = (f"Down {abs(drop_from_sma50_pct):.1f}% below its 50-day average with RSI at "
                          f"{last['rsi14']:.0f} (deeply oversold) — a classic bounce-watch setup, but the "
                          f"trend is still down. This is a contrarian bet on reversal, not a confirmed buy — "
                          f"the price could keep falling instead.")

    # Risk management for a NEW buy from here
    stop_loss_price = round(price - 2 * last["atr14"], 2)
    target_price = round(price + 2 * last["atr14"], 2)   # simple 1:1 reward:risk reference
    risk_per_share = price - stop_loss_price
    capital_at_risk = cfg["total_capital"] * (cfg["risk_per_trade_pct"] / 100)
    max_shares_by_risk = int(capital_at_risk / risk_per_share) if risk_per_share > 0 else 0
    max_position_value = cfg["total_capital"] * (cfg["max_position_pct"] / 100)
    max_shares_by_cap = int(max_position_value / price)
    suggested_shares = max(0, min(max_shares_by_risk, max_shares_by_cap))

    # A tighter, more conservative stop for dip/reversal bets — since you're betting
    # against the trend, the trade needs to be cut faster if you're wrong.
    dip_stop_loss = round(price - 1 * last["atr14"], 2) if is_dip_candidate else None
    dip_target = round(last["sma20"], 2) if is_dip_candidate else None  # a bounce back to the 20-day average is a common conservative target

    target_move_pct = ((target_price - price) / price) * 100
    time_to_target = estimate_time_to_target(df, target_move_pct)

    # Real historical support/resistance - the actual lowest/highest prices
    # this stock has traded at recently. Not a limit it "must" respect, but
    # a genuine reference point: these are levels where real buying/selling
    # has previously shown up, for a "how far could this still move" sense.
    lookback_60 = df.tail(60)
    recent_support = round(lookback_60["low"].min(), 2)
    recent_resistance = round(lookback_60["high"].max(), 2)
    support_distance_pct = round((price - recent_support) / price * 100, 1)
    resistance_distance_pct = round((recent_resistance - price) / price * 100, 1)

    return {
        "symbol": symbol,
        "price": round(price, 2),
        "signal": signal,
        "score": score,
        "horizon": horizon,
        "horizon_reasoning": "; ".join(horizon_reasons) or "no strong lean either way",
        "rsi14": round(last["rsi14"], 1),
        "trend": "up" if price > last["sma20"] > last["sma50"] else ("down" if price < last["sma20"] < last["sma50"] else "sideways"),
        "volatility_pct": round(volatility_pct, 2),
        "avg_daily_value_pkr": int(avg_daily_value),
        "suggested_stop_loss": stop_loss_price,
        "suggested_target": target_price,
        "time_to_target": time_to_target,
        "recent_support": recent_support,
        "recent_resistance": recent_resistance,
        "support_distance_pct": support_distance_pct,
        "resistance_distance_pct": resistance_distance_pct,
        "suggested_shares": suggested_shares,
        "suggested_position_value": round(suggested_shares * price, 2),
        "reasoning": "; ".join(reasons_bullish + reasons_bearish) or "no strong signal either way",
        "is_dip_candidate": is_dip_candidate,
        "dip_reasoning": dip_reasoning,
        "dip_stop_loss": dip_stop_loss,
        "dip_target": dip_target,
        "as_of": str(last["date"]),
        "history": df[["date", "open", "high", "low", "close", "sma20", "sma50", "rsi14"]].tail(90).assign(date=lambda d: d["date"].astype(str)).to_dict("records"),
    }


# ------------------------ Holdings / exit management -------------------------

def analyze_holding(holding, cfg):
    """For a stock you already own: hold, or sell now, and roughly when to check again."""
    symbol = holding["symbol"]
    buy_price = holding["buy_price"]
    buy_date = datetime.strptime(holding["buy_date"], "%Y-%m-%d").date()
    quantity = holding["quantity"]

    df = fetch_history(symbol, cfg["lookback_days"])
    if df is None:
        return {"symbol": symbol, "error": "no data available for this symbol"}

    df = compute_indicators(df)
    last = df.iloc[-1]
    if pd.isna(last["atr14"]):
        return {"symbol": symbol, "error": "not enough history to analyze"}

    current_price = last["close"]
    days_held = (date.today() - buy_date).days
    pnl_pct = round((current_price - buy_price) / buy_price * 100, 2)
    pnl_value = round((current_price - buy_price) * quantity, 2)

    # Trailing stop: locks in gains as the stock rises, based on the highest
    # close since you bought, not just your entry price.
    since_buy = df[df["date"] >= pd.Timestamp(buy_date)]
    highest_close_since_buy = since_buy["close"].max() if not since_buy.empty else current_price
    trailing_stop = round(highest_close_since_buy - 2 * last["atr14"], 2)
    # Never suggest a stop above what you paid on day one unless you're already in profit
    stop_loss = trailing_stop if trailing_stop > 0 else round(buy_price - 2 * last["atr14"], 2)

    target_price = round(buy_price + 2 * (buy_price - (buy_price - 2 * last["atr14"])), 2)  # 2:1 reward:risk from entry

    same_analysis = analyze_stock(symbol, df, cfg)
    horizon = same_analysis["horizon"] if same_analysis else "EITHER"
    trend_down = current_price < last["sma20"] < last["sma50"]

    # ---- Decision (check target first since it's a positive outcome to name clearly) ----
    if current_price >= target_price:
        action = "SELL NOW (or take partial profit)"
        why = f"price has reached the target (PKR {target_price}) — consider locking in gains"
    elif current_price <= stop_loss:
        action = "SELL NOW"
        why = f"price has fallen to/below the stop-loss (PKR {stop_loss})" + (
            " — this is a trailing stop, so it can also mean you've given back gains from a recent peak, not just a loss from your buy price" if pnl_pct > 0 else "")
    elif trend_down and pnl_pct > 0:
        action = "CONSIDER SELLING SOON"
        why = "trend has turned down while you're still in profit — a good time to review"
    elif trend_down and pnl_pct <= 0:
        action = "HOLD, BUT WATCH CLOSELY"
        why = "trend has turned down and you're at a loss — the stop-loss is your safety net, don't move it lower"
    else:
        action = "HOLD"
        why = "no exit signal yet — trend/momentum still intact"

    if horizon == "SHORT-TERM/TRADE":
        review_note = "This was a short-term/trade setup — review it daily. Most such setups play out within roughly 3-10 trading days; if nothing's happened by then, momentum has likely faded regardless of what the price does next."
    elif horizon == "LONG-TERM":
        review_note = "This looks like a long-term setup — no need to check it daily. Review weekly, and only act on the stop-loss or a real trend break, not day-to-day noise."
    else:
        review_note = "No strong horizon lean — check in every few days and lean on the stop-loss/target above rather than a fixed schedule."

    target_move_pct_from_current = ((target_price - current_price) / current_price) * 100
    time_to_target = estimate_time_to_target(df, target_move_pct_from_current)

    return {
        "symbol": symbol, "buy_price": buy_price, "buy_date": holding["buy_date"],
        "quantity": quantity, "days_held": days_held, "current_price": round(current_price, 2),
        "pnl_pct": pnl_pct, "pnl_value": pnl_value, "horizon": horizon,
        "stop_loss": stop_loss, "target_price": target_price,
        "time_to_target": time_to_target,
        "action": action, "why": why, "review_note": review_note,
        "as_of": str(last["date"]),
        "recent_filings": fetch_recent_filings(symbol),
        "history": df[["date", "open", "high", "low", "close", "sma20", "sma50", "rsi14"]].tail(90).assign(date=lambda d: d["date"].astype(str)).to_dict("records"),
    }


# ------------------------------ Scan run --------------------------------------

def run_screen(cfg):
    universe = fetch_universe(cfg["index"])
    results = []
    for i, symbol in enumerate(universe, 1):
        print(f"[{i}/{len(universe)}] {symbol}...", end=" ")
        df = fetch_history(symbol, cfg["lookback_days"])
        if df is None:
            print("no data"); continue
        try:
            r = analyze_stock(symbol, df, cfg)
        except Exception:
            print("analysis error"); traceback.print_exc(); continue
        if r is None:
            print("insufficient history / illiquid"); continue
        print(r["signal"])
        results.append(r)
        time.sleep(0.2)  # be polite to PSX's servers
    return results


def run_holdings(cfg):
    if not MY_HOLDINGS:
        return []
    print(f"\nChecking {len(MY_HOLDINGS)} holding(s)...")
    out = []
    for h in MY_HOLDINGS:
        print(f"  {h['symbol']}...", end=" ")
        r = analyze_holding(h, cfg)
        print(r.get("action", r.get("error", "?")))
        out.append(r)
    return out


def compute_top_picks(market_results, cfg):
    """The best BUY-rated stocks right now, ranked by score."""
    buys = [r for r in market_results if r["signal"] == "BUY"]
    buys_sorted = sorted(buys, key=lambda r: r["score"], reverse=True)
    picks = buys_sorted[: cfg["top_picks_count"]]
    for r in picks:
        r["recent_filings"] = fetch_recent_filings(r["symbol"])
    return picks


def compute_dip_watch(market_results, cfg):
    """
    Downtrending stocks that are deeply oversold - contrarian bounce candidates.
    IMPORTANT: this is a DIFFERENT, riskier bet than the BUY list. These stocks are
    still in a downtrend by definition - you'd be betting the fall is overdone and
    about to reverse, not that the trend already favors you. Sort by how oversold
    (lowest RSI first) since that's the closest thing to a "most stretched" ranking.
    """
    candidates = [r for r in market_results if r.get("is_dip_candidate")]
    picks = sorted(candidates, key=lambda r: r["rsi14"])[: cfg["top_picks_count"]]
    for r in picks:
        r["recent_filings"] = fetch_recent_filings(r["symbol"])
    return picks


def compute_market_overview(market_results):
    """
    A plain-language read on today's overall market, built from real breadth
    data (how many scanned stocks are actually trending up vs down right now)
    - not a guess, not outside sentiment. This is NOT the same as macro/news
    sentiment, which this tool can't see - just what the price action itself
    is doing across the board.
    """
    if not market_results:
        return {
            "mood": "UNKNOWN", "headline": "No data to summarize yet.",
            "pct_up": 0, "pct_down": 0, "pct_buy": 0, "pct_avoid": 0,
            "avg_volatility": 0, "explanation": "",
        }

    total = len(market_results)
    up = sum(1 for r in market_results if r["trend"] == "up")
    down = sum(1 for r in market_results if r["trend"] == "down")
    buys = sum(1 for r in market_results if r["signal"] == "BUY")
    avoids = sum(1 for r in market_results if r["signal"] == "AVOID")
    avg_vol = sum(r["volatility_pct"] for r in market_results) / total

    pct_up = round(up / total * 100)
    pct_down = round(down / total * 100)
    pct_buy = round(buys / total * 100)
    pct_avoid = round(avoids / total * 100)

    if pct_up - pct_down > 15:
        mood, headline = "BULLISH", "More stocks are trending up than down today — broad market leans positive."
    elif pct_down - pct_up > 15:
        mood, headline = "BEARISH", "More stocks are trending down than up today — broad market leans negative/a dip."
    else:
        mood, headline = "MIXED", "Roughly balanced between up and down trends — no strong overall direction today."

    vol_note = ("Volatility is on the higher side across the board — expect bigger daily swings than usual." if avg_vol > 3.5
                else "Volatility is fairly calm across the board today.")

    explanation = (f"Out of {total} scanned stocks: {pct_up}% are in an uptrend, {pct_down}% in a downtrend. "
                    f"{pct_buy}% currently score as BUY, {pct_avoid}% as AVOID. {vol_note} "
                    f"This reflects price action only — it does not know about news or geopolitical events; "
                    f"ask directly if you want that context for a specific stock.")

    return {
        "mood": mood, "headline": headline, "pct_up": pct_up, "pct_down": pct_down,
        "pct_buy": pct_buy, "pct_avoid": pct_avoid, "avg_volatility": round(avg_vol, 2),
        "explanation": explanation,
    }


# ------------------------------ Dashboard ---------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PSX Dashboard</title>
<style>
  :root{
    --bg:#0d1210; --panel:#141b18; --panel-2:#1a2320; --line:#263230; --text:#e8ede9; --muted:#8fa39a;
    --buy:#2fbf71; --buy-dim:#163825; --avoid:#e5555a; --avoid-dim:#3a1a1c; --hold:#d9a441; --hold-dim:#332813;
    --accent:#2fbf71; --mono:'SF Mono','Roboto Mono',Consolas,monospace;
  }
  *{ box-sizing:border-box; }
  body{ margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; -webkit-font-smoothing:antialiased; }
  header{ padding:20px 16px 14px; border-bottom:1px solid var(--line); position:sticky; top:0; background:rgba(13,18,16,0.95); backdrop-filter:blur(6px); z-index:10; }
  header h1{ margin:0; font-size:20px; letter-spacing:-0.01em; }
  header .sub{ color:var(--muted); font-size:12.5px; margin-top:4px; font-family:var(--mono); }
  .disclaimer{ margin:10px 16px 0; padding:10px 12px; background:var(--panel-2); border:1px solid var(--line); border-left:3px solid var(--hold); border-radius:6px; font-size:12px; color:var(--muted); line-height:1.5; }
  .staleness{ margin:10px 16px 0; padding:12px 14px; border-radius:8px; font-size:13px; line-height:1.5; font-weight:600; display:flex; align-items:center; gap:8px; }
  .staleness.fresh{ background:var(--buy-dim); color:var(--buy); border:1px solid #1c4a2f; }
  .staleness.aging{ background:var(--hold-dim); color:var(--hold); border:1px solid #4a3a13; }
  .staleness.stale{ background:var(--avoid-dim); color:var(--avoid); border:1px solid #5a2528; }
  .stale-tag{ display:inline-block; font-size:9.5px; font-weight:700; padding:2px 7px; border-radius:5px; background:var(--avoid-dim); color:var(--avoid); margin-left:6px; letter-spacing:.03em; vertical-align:middle; }
  .action-alert{ margin:10px 16px 0; padding:14px; border-radius:10px; background:var(--avoid-dim); border:2px solid var(--avoid); animation: pulseAlert 2s ease-in-out infinite; }
  .live-refresh-row{ margin:10px 16px 0; display:flex; flex-direction:column; gap:6px; }
  .live-refresh-row button{ background:var(--panel); border:1px solid var(--line); color:var(--text); padding:8px 14px; border-radius:8px; font-size:12.5px; cursor:pointer; align-self:flex-start; }
  .live-refresh-row button:disabled{ opacity:0.5; cursor:wait; }
  .live-refresh-status{ font-size:11.5px; line-height:1.5; }
  .add-holding-box{ margin:16px 16px 8px; padding:14px; border-radius:12px; background:var(--panel); border:1px solid var(--line); }
  .add-holding-title{ font-size:13px; font-weight:700; margin-bottom:10px; }
  .add-holding-row{ display:flex; gap:8px; margin-bottom:8px; }
  .add-holding-row input{ flex:1; background:var(--panel-2); border:1px solid var(--line); color:var(--text); padding:8px 10px; border-radius:8px; font-size:13px; min-width:0; }
  .add-holding-box button{ width:100%; background:var(--accent); color:#08130d; border:none; padding:9px; border-radius:8px; font-size:13px; font-weight:700; cursor:pointer; }
  .add-holding-box button:disabled{ opacity:0.5; cursor:wait; }
  .add-holding-status{ margin-top:8px; font-size:12px; line-height:1.5; min-height:14px; }
  .add-holding-note{ margin-top:8px; font-size:10.5px; color:var(--muted); line-height:1.5; }
  .live-added-tag{ display:inline-block; font-size:9px; font-weight:700; padding:2px 6px; border-radius:5px; background:var(--panel-2); border:1px solid var(--line); color:var(--muted); margin-left:6px; letter-spacing:.03em; vertical-align:middle; }
  .remove-holding-btn{ background:none; border:1px solid var(--avoid); color:var(--avoid); border-radius:6px; padding:3px 8px; font-size:10.5px; cursor:pointer; margin-top:8px; }
  .quick-add-btn{ background:none; border:1px solid var(--buy); color:var(--buy); border-radius:6px; padding:4px 10px; font-size:10.5px; cursor:pointer; margin-top:10px; font-weight:600; }
  @keyframes pulseAlert{ 0%,100%{ box-shadow:0 0 0 0 rgba(229,85,90,0.35); } 50%{ box-shadow:0 0 0 6px rgba(229,85,90,0); } }
  .action-alert-title{ font-size:13.5px; font-weight:800; color:var(--avoid); text-transform:uppercase; letter-spacing:.03em; display:flex; align-items:center; gap:6px; }
  .action-alert-item{ font-size:13px; color:var(--text); margin-top:8px; padding-top:8px; border-top:1px solid #5a2528; }
  .action-alert-item b{ color:var(--avoid); }
  .maintabs{ display:flex; gap:6px; padding:14px 16px 0; flex-wrap:wrap; }

  .mood-card{ margin:16px 16px 0; padding:16px; border-radius:14px; border:1px solid var(--line); background:var(--panel); }
  .mood-card.BULLISH{ border-left:4px solid var(--buy); }
  .mood-card.BEARISH{ border-left:4px solid var(--avoid); }
  .mood-card.MIXED{ border-left:4px solid var(--hold); }
  .mood-tag{ font-size:11px; font-weight:700; letter-spacing:.05em; text-transform:uppercase; }
  .mood-tag.BULLISH{ color:var(--buy); } .mood-tag.BEARISH{ color:var(--avoid); } .mood-tag.MIXED{ color:var(--hold); }
  .mood-headline{ font-size:15px; font-weight:600; margin-top:6px; }
  .mood-stats{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:14px; }
  .mood-stat{ text-align:center; }
  .mood-stat .v{ font-size:17px; font-weight:700; font-family:var(--mono); }
  .mood-stat .l{ font-size:10px; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; margin-top:2px; }
  .mood-explain{ margin-top:14px; font-size:12.5px; color:var(--muted); line-height:1.6; border-top:1px solid var(--line); padding-top:12px; }
  .picks-heading{ padding:20px 16px 10px; font-size:13px; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }
  .rank-badge{ display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:50%; background:var(--accent); color:#08130d; font-size:11px; font-weight:800; margin-right:6px; }
  .dip-warning{ margin:16px 16px 0; padding:14px; border-radius:12px; background:var(--avoid-dim); border:1px solid #5a2528; color:#f0b8ba; font-size:12.5px; line-height:1.6; }
  .dip-badge{ display:inline-flex; align-items:center; justify-content:center; width:22px; height:22px; border-radius:50%; background:var(--hold); color:#08130d; font-size:11px; font-weight:800; margin-right:6px; }
  .maintab{ padding:9px 16px; border-radius:20px; font-size:13.5px; cursor:pointer; background:var(--panel); border:1px solid var(--line); color:var(--muted); font-weight:600; }
  .maintab.active{ background:var(--accent); color:#08130d; border-color:var(--accent); }
  .page{ display:none; }
  .page.active{ display:block; }

  .summary{ display:grid; grid-template-columns:repeat(4,1fr); gap:8px; padding:14px 16px; }
  .stat{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 10px; text-align:center; }
  .stat .num{ font-size:22px; font-weight:700; font-family:var(--mono); }
  .stat .lbl{ font-size:10.5px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; margin-top:2px;}
  .stat.buy .num{ color:var(--buy); } .stat.avoid .num{ color:var(--avoid); } .stat.hold .num{ color:var(--hold); }

  .controls{ display:flex; gap:8px; padding:0 16px 14px; flex-wrap:wrap; }
  .controls input, .controls select{ background:var(--panel); border:1px solid var(--line); color:var(--text); padding:9px 12px; border-radius:8px; font-size:14px; flex:1; min-width:110px; }
  .tabs{ display:flex; gap:6px; padding:0 16px 14px; }
  .tab{ padding:7px 14px; border-radius:20px; font-size:13px; cursor:pointer; background:var(--panel); border:1px solid var(--line); color:var(--muted); }
  .tab.active{ background:var(--accent); color:#08130d; border-color:var(--accent); font-weight:600; }

  .list{ padding:0 16px 40px; display:flex; flex-direction:column; gap:10px; }
  .card{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px; cursor:pointer; }
  .card-top{ display:flex; justify-content:space-between; align-items:flex-start; }
  .sym{ font-size:16px; font-weight:700; font-family:var(--mono); }
  .price{ font-size:13px; color:var(--muted); font-family:var(--mono); margin-top:2px; }
  .badge{ font-size:11px; font-weight:700; padding:4px 10px; border-radius:20px; letter-spacing:.03em; white-space:nowrap; }
  .badge.BUY,.badge.HOLD_GREEN{ background:var(--buy-dim); color:var(--buy); }
  .badge.AVOID,.badge['SELL NOW']{ background:var(--avoid-dim); color:var(--avoid); }
  .badge.HOLD{ background:var(--hold-dim); color:var(--hold); }
  .badge.sell{ background:var(--avoid-dim); color:var(--avoid); }
  .badge.watch{ background:var(--hold-dim); color:var(--hold); }
  .badge.holdgood{ background:var(--buy-dim); color:var(--buy); }

  .horizon{ display:inline-block; margin-top:8px; font-size:10.5px; font-weight:600; padding:3px 9px; border-radius:6px; letter-spacing:.03em; background:var(--panel-2); border:1px solid var(--line); color:var(--muted); }
  .horizon.long{ color:#7fb0e0; border-color:#2a4a66; } .horizon.short{ color:#d9a441; border-color:#4a3a13; }

  .metrics{ display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-top:12px; }
  .m{ font-size:11px; color:var(--muted); } .m b{ display:block; color:var(--text); font-size:13px; font-family:var(--mono); font-weight:600; margin-top:1px; }
  .reason{ margin-top:10px; font-size:12.5px; color:var(--muted); line-height:1.5; }
  .detail{ display:none; margin-top:12px; border-top:1px solid var(--line); padding-top:12px; } .detail.open{ display:block; }
  .detail canvas{ max-height:160px; }
  .risk-row{ display:flex; justify-content:space-between; font-size:12.5px; padding:6px 0; border-bottom:1px dashed var(--line); }
  .risk-row span:first-child{ color:var(--muted); } .risk-row span:last-child{ font-family:var(--mono); font-weight:600; }
  .empty{ text-align:center; color:var(--muted); padding:40px 16px; font-size:14px; }
  .pnl-pos{ color:var(--buy); } .pnl-neg{ color:var(--avoid); }

  .calc-wrap{ padding-bottom:40px; }
  .calc-card{
    margin:0 16px; background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px;
    display:flex; flex-direction:column;
  }
  .calc-card label{ font-size:12.5px; color:var(--muted); font-weight:600; }
  .calc-card input, .calc-card select{
    margin-top:6px; background:var(--panel-2); border:1px solid var(--line); color:var(--text);
    padding:11px 12px; border-radius:8px; font-size:15px; font-family:var(--mono);
  }
  .calc-result{ margin-top:18px; border-top:1px solid var(--line); padding-top:16px; }
  .calc-row{ display:flex; justify-content:space-between; font-size:13.5px; padding:7px 0; }
  .calc-row span:first-child{ color:var(--muted); }
  .calc-row span:last-child{ font-family:var(--mono); font-weight:600; }
  .scenario{ margin-top:14px; padding:14px; border-radius:10px; border:1px solid var(--line); }
  .scenario.good{ background:var(--buy-dim); border-color:#1c4a30; }
  .scenario.bad{ background:var(--avoid-dim); border-color:#4a2427; }
  .scenario .title{ font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; margin-bottom:8px; }
  .scenario.good .title{ color:var(--buy); } .scenario.bad .title{ color:var(--avoid); }
  .scenario .big{ font-size:22px; font-weight:700; font-family:var(--mono); }
  .scenario .sub{ font-size:12px; color:var(--muted); margin-top:2px; }

  @media (min-width:720px){
    .list{ display:grid; grid-template-columns:repeat(2,1fr); align-content:start; max-width:900px; margin:0 auto; padding-bottom:40px; }
    .summary,.controls,.tabs,.maintabs{ max-width:900px; margin:0 auto; }
    header .inner, .disclaimer{ max-width:900px; margin:0 auto; }
    .calc-wrap{ max-width:600px; margin:0 auto; }
  }
</style>
</head>
<body>

<header><div class="inner">
  <h1>PSX Dashboard</h1>
  <div class="sub">Generated __GENERATED_AT__ &middot; __TOTAL__ stocks scanned</div>
</div></header>

<div class="staleness" id="stalenessBanner"></div>

<div class="live-refresh-row">
  <button id="liveRefreshBtn" onclick="tryLiveRefresh(false)">⟳ Refresh Live Prices</button>
  <div id="liveRefreshStatus" class="live-refresh-status"></div>
</div>

<div id="actionAlert"></div>

<div class="disclaimer">Rule-based technical signals from historical price/volume data — not a prediction, not financial advice. Verify before every trade. Illiquid stocks were filtered out.</div>

<div class="maintabs">
  <div class="maintab active" data-page="picks">Today's Picks</div>
  <div class="maintab" data-page="dipwatch">Dip Watch</div>
  <div class="maintab" data-page="market">Market Signals</div>
  <div class="maintab" data-page="holdings" id="holdingsTabLabel">My Holdings (__HOLDINGS_COUNT__)</div>
  <div class="maintab" data-page="calc">Calculator</div>
</div>

<div class="page active" id="page-picks">
  <div class="mood-card" id="moodCard"></div>
  <div class="picks-heading">Top picks right now</div>
  <div class="list" id="picksList"></div>
  <div class="empty" id="picksEmpty" style="display:none;">No strong BUY signals right now — sometimes the right move is to wait. Check back after the next scan.</div>
</div>

<div class="page" id="page-dipwatch">
  <div class="dip-warning">
    <b>This is a different, riskier bet than the picks above.</b> These stocks are still in a downtrend — you'd be betting the fall is overdone and about to bounce, not that the trend already favors you. It can and does keep falling instead. Use a tight stop-loss, and treat this as speculative, not a confirmed buy.
  </div>
  <div class="picks-heading">Deeply oversold — possible bounce candidates</div>
  <div class="list" id="dipList"></div>
  <div class="empty" id="dipEmpty" style="display:none;">No stocks are deeply oversold enough to qualify right now.</div>
</div>

<div class="page" id="page-market">
  <div class="summary">
    <div class="stat buy"><div class="num" id="cnt-buy">0</div><div class="lbl">Buy</div></div>
    <div class="stat hold"><div class="num" id="cnt-hold">0</div><div class="lbl">Hold</div></div>
    <div class="stat avoid"><div class="num" id="cnt-avoid">0</div><div class="lbl">Avoid</div></div>
    <div class="stat"><div class="num" id="cnt-total">0</div><div class="lbl">Total</div></div>
  </div>
  <div class="controls">
    <input type="text" id="search" placeholder="Search symbol...">
    <select id="sortBy">
      <option value="score">Sort: Score</option>
      <option value="volatility_pct">Sort: Volatility</option>
      <option value="symbol">Sort: Symbol A-Z</option>
    </select>
    <select id="horizonFilter">
      <option value="ALL">Any horizon</option>
      <option value="LONG-TERM">Long-term</option>
      <option value="SHORT-TERM/TRADE">Short-term/trade</option>
      <option value="EITHER">Either</option>
    </select>
  </div>
  <div class="tabs">
    <div class="tab active" data-filter="ALL">All</div>
    <div class="tab" data-filter="BUY">Buy</div>
    <div class="tab" data-filter="HOLD">Hold</div>
    <div class="tab" data-filter="AVOID">Avoid</div>
  </div>
  <div class="list" id="list"></div>
  <div class="empty" id="empty" style="display:none;">No stocks match your filter.</div>
</div>

<div class="page" id="page-holdings">
  <div class="add-holding-box">
    <div class="add-holding-title">+ Add a holding right here (no Colab re-run needed)</div>
    <div class="add-holding-row">
      <input type="text" id="addHSymbol" placeholder="Symbol e.g. LUCK" maxlength="12">
      <input type="number" id="addHPrice" placeholder="Buy price" step="0.01" min="0">
    </div>
    <div class="add-holding-row">
      <input type="date" id="addHDate">
      <input type="number" id="addHQty" placeholder="Quantity" min="1" step="1">
    </div>
    <button id="addHBtn" onclick="addHoldingLive()">Add &amp; Analyze</button>
    <div id="addHStatus" class="add-holding-status"></div>
    <div class="add-holding-note">Needs the live-refresh proxy set up (same one used by "Refresh Live Prices") to fetch this stock's history. Recent filings aren't available for holdings added this way — everything else (stop-loss, target, timing, chart) is fully analyzed.</div>
  </div>
  <div class="list" id="holdingsList" style="padding-top:16px;"></div>
  <div class="empty" id="holdingsEmpty" style="display:none;">
    No holdings yet. Add one above, or add it to MY_HOLDINGS at the top of psx_app.py and re-run for a permanent record.
  </div>
</div>

<div class="page" id="page-calc">
  <div class="calc-wrap">
    <div class="disclaimer" style="margin:16px;">
      This shows two <b>reference scenarios</b> built from this tool's own rule-based target and stop-loss for the stock you pick — not a prediction. The real outcome could land anywhere between them, above the target, below the stop-loss (prices can gap past a stop), or the stock could just sit still for weeks. Use it to see the shape of the bet, not a promise of the result.
    </div>
    <div class="calc-card">
      <label>How much are you putting in? (PKR)</label>
      <input type="number" id="calcCapital" value="50000" min="0" step="1000">

      <label style="margin-top:14px;">Which stock?</label>
      <select id="calcSymbol"></select>

      <div id="calcOutput"></div>
    </div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;
const HOLDINGS = __HOLDINGS_JSON__;
const REMOVED_SYMBOLS = new Set(); // symbols hidden after being marked sold - see removeHolding()
const TOP_PICKS = __TOP_PICKS_JSON__;
const DIP_WATCH = __DIP_WATCH_JSON__;
const MARKET_MOOD = __MARKET_MOOD_JSON__;
const GENERATED_AT_ISO = "__GENERATED_AT_ISO__";

let currentFilter='ALL', currentSort='score', currentSearch='', currentHorizon='ALL';
const chartInstances = {};
function fmt(n){ return n===null||n===undefined ? '-' : n.toLocaleString(); }

function filingsHtml(filings){
  if(!filings || !filings.length) return '';
  const rows = filings.map(f => `<div class="risk-row"><span>${f.type}${f.period_ended ? ' ('+f.period_ended+')' : ''}</span><span>${f.posting_date}</span></div>`).join('');
  return `<div class="reason" style="margin-top:10px;"><b>Recent PSX filings:</b></div>${rows}`;
}

function quickAddButtonHtml(symbol, price){
  return `<button class="quick-add-btn" data-symbol="${symbol}" data-price="${price}">+ I bought this</button>`;
}

// Jumps to the My Holdings tab and pre-fills the Add Holding form with this
// stock's current price and today's date - you just add quantity and hit
// Add & Analyze. Saves retyping the symbol/price/date every time.
function quickAddFromCard(symbol, price){
  document.querySelectorAll('.maintab').forEach(t => t.classList.remove('active'));
  document.querySelector('.maintab[data-page="holdings"]').classList.add('active');
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page-holdings').classList.add('active');
  document.getElementById('addHSymbol').value = symbol;
  document.getElementById('addHPrice').value = price;
  document.getElementById('addHDate').value = new Date().toISOString().slice(0,10);
  document.getElementById('addHQty').focus();
  window.scrollTo(0, 0);
}

function timeToTargetHtml(t){
  if(!t || t.sample_size === 0) return '<div class="risk-row"><span>Historical time-to-target</span><span>not enough history</span></div>';
  if(t.hit_rate_pct === 0 || t.median_days === null) return `<div class="risk-row"><span>Historical time-to-target</span><span>rarely/never happened within 30 days historically</span></div>`;
  return `<div class="risk-row"><span>Historical time-to-target</span><span>~${t.median_days} trading days (happened ${t.hit_rate_pct}% of the time historically)</span></div>`;
}

function supportResistanceHtml(r){
  if(r.recent_support === undefined) return '';
  return `
    <div class="reason" style="margin-top:10px;"><b>Recent 60-day range (real historical levels):</b></div>
    <div class="risk-row"><span>Support (recent low)</span><span>PKR ${fmt(r.recent_support)} (${r.support_distance_pct}% below current)</span></div>
    <div class="risk-row"><span>Resistance (recent high)</span><span>PKR ${fmt(r.recent_resistance)} (${r.resistance_distance_pct}% above current)</span></div>
    <div class="risk-row" style="border-bottom:none;"><span style="font-size:11px; color:var(--muted);">These are actual past price levels this stock has traded at, not limits it must respect — support can break, resistance can be exceeded.</span></div>`;
}

function hoursSince(isoString){
  const then = new Date(isoString);
  const now = new Date();
  return (now - then) / (1000 * 60 * 60);
}

function renderStaleness(){
  const el = document.getElementById('stalenessBanner');
  const hrs = hoursSince(GENERATED_AT_ISO);
  let level, icon, msg;
  if (hrs < 6) {
    level = 'fresh'; icon = '●';
    msg = `Data generated ${hrs < 1 ? 'less than an hour' : Math.round(hrs) + ' hour' + (Math.round(hrs)===1?'':'s')} ago — reasonably current.`;
  } else if (hrs < 24) {
    level = 'aging'; icon = '●';
    msg = `Data is ${Math.round(hrs)} hours old. Prices have likely moved since this was generated — re-run before trading on it.`;
  } else {
    const days = Math.round(hrs / 24);
    level = 'stale'; icon = '●';
    msg = `Data is ${days} day${days===1?'':'s'} old — treat every price here as outdated. Re-run the scan for current numbers before making any decision.`;
  }
  el.className = 'staleness ' + level;
  el.innerHTML = `<span>${icon}</span><span>${msg}</span>`;
}

// The closest thing to an "alert" a downloaded file can honestly give you:
// not a push notification (nothing can run in the background here), but an
// unmissable flag the moment you open the dashboard, for anything that has
// ALREADY crossed a buy/sell threshold as of when this was generated.
function renderActionAlert(){
  const el = document.getElementById('actionAlert');
  const sells = HOLDINGS.filter(h => !h.error && !REMOVED_SYMBOLS.has(h.symbol) && h.action && h.action.includes('SELL'));
  const items = [];

  sells.forEach(h => {
    items.push(`<div class="action-alert-item"><b>SELL signal:</b> ${h.symbol} — ${h.why}</div>`);
  });

  if(!items.length){
    el.innerHTML = '';
    return;
  }

  el.innerHTML = `
    <div class="action-alert">
      <div class="action-alert-title"><span>⚠</span> Action needed on ${items.length} holding${items.length===1?'':'s'}</div>
      ${items.join('')}
      <div class="action-alert-item" style="border-top:none; padding-top:0; color:var(--muted); font-size:11.5px;">
        Based on prices when this was generated (see freshness banner above) — verify live price in AKD before acting.
      </div>
    </div>`;
}

// Uses a proxy YOU deploy (see psx_proxy.gs) to fetch PSX's public screener
// server-side, sidestepping the browser CORS block confirmed earlier when
// fetching PSX directly. Auto-refreshes every 60s while this tab stays open
// and in the foreground (phones commonly pause background tabs - this is
// normal browser behavior, not a bug in this code).
const LIVE_REFRESH_PROXY_URL = "__LIVE_REFRESH_PROXY_URL__";
let liveRefreshInFlight = false;
let autoRefreshTimer = null;
let autoRefreshCountdown = 60;

async function tryLiveRefresh(isAuto){
  if(liveRefreshInFlight) return;
  if(!LIVE_REFRESH_PROXY_URL){
    const statusEl = document.getElementById('liveRefreshStatus');
    statusEl.innerHTML = `Live refresh needs a one-time setup: deploy psx_proxy.gs as a Google Apps Script Web App, then paste the URL into psx_app.py's CONFIG under "live_refresh_proxy_url" and re-run. See the README for step-by-step instructions.`;
    statusEl.style.color = 'var(--hold)';
    return;
  }
  liveRefreshInFlight = true;
  const statusEl = document.getElementById('liveRefreshStatus');
  const btn = document.getElementById('liveRefreshBtn');
  if(btn) btn.disabled = true;
  statusEl.textContent = isAuto ? 'Auto-refreshing…' : 'Trying to fetch live prices via your proxy…';
  statusEl.style.color = 'var(--muted)';

  try {
    const proxyTarget = LIVE_REFRESH_PROXY_URL;
    const resp = await fetch(proxyTarget, { method: 'GET' });
    if(!resp.ok) throw new Error(`Proxy responded with status ${resp.status}`);
    const html = await resp.text();
    const doc = new DOMParser().parseFromString(html, 'text/html');
    const table = doc.querySelector('table');
    if(!table) throw new Error('Could not find a data table in the response - check your proxy is deployed correctly');

    const headerCells = Array.from(table.querySelectorAll('thead th, tr:first-child th, tr:first-child td'))
      .map(c => c.textContent.trim().toUpperCase());
    const symbolIdx = headerCells.findIndex(h => h.includes('SYMBOL'));
    const priceIdx = headerCells.findIndex(h => h === 'PRICE' || h.includes('PRICE'));
    if(symbolIdx === -1 || priceIdx === -1) throw new Error('Table format not recognized (PSX may have changed their page)');

    const priceMap = {};
    table.querySelectorAll('tbody tr, tr').forEach(row => {
      const cells = row.querySelectorAll('td');
      if(cells.length <= Math.max(symbolIdx, priceIdx)) return;
      const symbol = cells[symbolIdx].textContent.trim().split(/\s+/)[0];
      const price = parseFloat(cells[priceIdx].textContent.replace(/,/g, ''));
      if(symbol && !isNaN(price)) priceMap[symbol] = price;
    });

    if(Object.keys(priceMap).length === 0) throw new Error('Parsed the page but found no usable price rows');

    let updated = 0;
    [DATA, TOP_PICKS, DIP_WATCH].forEach(list => {
      list.forEach(r => { if(priceMap[r.symbol] !== undefined){ r.price = priceMap[r.symbol]; updated++; } });
    });
    HOLDINGS.forEach(h => { if(!h.error && priceMap[h.symbol] !== undefined){ h.current_price = priceMap[h.symbol]; updated++; } });
    Object.assign(LIVE_PRICES, priceMap);

    render(); renderPicks(); renderDipWatch(); renderHoldings();
    redrawOpenCharts();
    const now = new Date();
    statusEl.textContent = `✓ Refreshed ${updated} price${updated===1?'':'s'} at ${now.toLocaleTimeString()}. Charts now project forward from these live prices. Next auto-refresh in 60s. (Stop-loss/target levels still reflect your original scan.)`;
    statusEl.style.color = 'var(--buy)';
  } catch(err) {
    statusEl.innerHTML = `✗ Refresh failed (${err.message}). Check your Apps Script deployment is live and the URL in CONFIG is correct. <a href="https://dps.psx.com.pk/" target="_blank" style="color:var(--accent);">Open PSX directly</a> as a fallback.`;
    statusEl.style.color = 'var(--avoid)';
  } finally {
    liveRefreshInFlight = false;
    if(btn) btn.disabled = false;
  }
}

function startAutoRefresh(){
  if(!LIVE_REFRESH_PROXY_URL || autoRefreshTimer) return;
  autoRefreshTimer = setInterval(() => {
    if(document.visibilityState !== 'visible') return; // don't refresh while backgrounded
    tryLiveRefresh(true);
  }, 60000);
}

// A card's own price is separately stale if its specific as_of date isn't today,
// even if the overall scan was recent (can happen for thinly-traded symbols
// whose last trade was days ago).
function isCardStale(asOfString){
  const asOf = new Date(asOfString);
  const today = new Date();
  const diffDays = Math.floor((today - asOf) / (1000*60*60*24));
  return diffDays >= 1 ? diffDays : 0;
}

function staleTagHtml(asOf){
  const days = isCardStale(asOf);
  return days > 0 ? `<span class="stale-tag">${days}D OLD</span>` : '';
}

function renderMood(){
  const el = document.getElementById('moodCard');
  const m = MARKET_MOOD;
  el.className = 'mood-card ' + m.mood;
  el.innerHTML = `
    <div class="mood-tag ${m.mood}">${m.mood} MARKET</div>
    <div class="mood-headline">${m.headline}</div>
    <div class="mood-stats">
      <div class="mood-stat"><div class="v" style="color:var(--buy)">${m.pct_up}%</div><div class="l">Trending up</div></div>
      <div class="mood-stat"><div class="v" style="color:var(--avoid)">${m.pct_down}%</div><div class="l">Trending down</div></div>
      <div class="mood-stat"><div class="v" style="color:var(--buy)">${m.pct_buy}%</div><div class="l">Buy signals</div></div>
      <div class="mood-stat"><div class="v">${m.avg_volatility}%</div><div class="l">Avg volatility</div></div>
    </div>
    <div class="mood-explain">${m.explanation}</div>
  `;
}

function renderPicks(){
  const list = document.getElementById('picksList'); list.innerHTML = '';
  document.getElementById('picksEmpty').style.display = TOP_PICKS.length ? 'none' : 'block';
  TOP_PICKS.forEach((r, i) => {
    const card = document.createElement('div'); card.className = 'card';
    card.innerHTML = `
      <div class="card-top">
        <div><div class="sym"><span class="rank-badge">${i+1}</span>${r.symbol}${staleTagHtml(r.as_of)}</div><div class="price">PKR ${fmt(r.price)} &middot; as of ${r.as_of}</div></div>
        <div class="badge ${r.signal}">${r.signal}</div>
      </div>
      <div class="horizon ${r.horizon==='LONG-TERM'?'long':(r.horizon==='SHORT-TERM/TRADE'?'short':'')}">${r.horizon}</div>
      <div class="metrics">
        <div class="m">RSI<b>${r.rsi14}</b></div>
        <div class="m">Score<b>${r.score}</b></div>
        <div class="m">Volatility<b>${r.volatility_pct}%</b></div>
      </div>
      <div class="reason">${r.reasoning}</div>
      <div class="detail" id="detail-pick-${r.symbol}">
        <canvas id="chart-pick-${r.symbol}"></canvas>
        <div class="risk-row"><span>Suggested stop-loss</span><span>PKR ${fmt(r.suggested_stop_loss)}</span></div>
        <div class="risk-row"><span>Suggested target</span><span>PKR ${fmt(r.suggested_target)}</span></div>
        ${timeToTargetHtml(r.time_to_target)}
        <div class="risk-row"><span>Suggested shares</span><span>${fmt(r.suggested_shares)}</span></div>
        ${supportResistanceHtml(r)}
        ${filingsHtml(r.recent_filings)}
        ${quickAddButtonHtml(r.symbol, r.price)}
      </div>`;
    card.addEventListener('click', (ev) => {
      if(ev.target.classList.contains('quick-add-btn')){
        ev.stopPropagation();
        quickAddFromCard(ev.target.dataset.symbol, ev.target.dataset.price);
        return;
      }
      toggleDetail(r.symbol, `chart-pick-${r.symbol}`, `detail-pick-${r.symbol}`, r.history, r.suggested_stop_loss, r.suggested_target, r.volatility_pct);
    });
    list.appendChild(card);
  });
}

function renderDipWatch(){
  const list = document.getElementById('dipList'); list.innerHTML = '';
  document.getElementById('dipEmpty').style.display = DIP_WATCH.length ? 'none' : 'block';
  DIP_WATCH.forEach((r, i) => {
    const card = document.createElement('div'); card.className = 'card';
    card.innerHTML = `
      <div class="card-top">
        <div><div class="sym"><span class="dip-badge">${i+1}</span>${r.symbol}${staleTagHtml(r.as_of)}</div><div class="price">PKR ${fmt(r.price)} &middot; as of ${r.as_of}</div></div>
        <div class="badge AVOID">DOWNTREND</div>
      </div>
      <div class="metrics">
        <div class="m">RSI<b>${r.rsi14}</b></div>
        <div class="m">Volatility<b>${r.volatility_pct}%</b></div>
        <div class="m">Trend<b>${r.trend}</b></div>
      </div>
      <div class="reason">${r.dip_reasoning}</div>
      <div class="detail" id="detail-dip-${r.symbol}">
        <canvas id="chart-dip-${r.symbol}"></canvas>
        <div class="risk-row"><span>Tight stop-loss (if wrong, exit fast)</span><span>PKR ${fmt(r.dip_stop_loss)}</span></div>
        <div class="risk-row"><span>Conservative bounce target (20-day avg)</span><span>PKR ${fmt(r.dip_target)}</span></div>
        ${supportResistanceHtml(r)}
        ${filingsHtml(r.recent_filings)}
      </div>`;
    card.addEventListener('click', () => toggleDetail(r.symbol, `chart-dip-${r.symbol}`, `detail-dip-${r.symbol}`, r.history, r.dip_stop_loss, r.dip_target, r.volatility_pct));
    list.appendChild(card);
  });
}

function render(){
  const list=document.getElementById('list'); list.innerHTML='';
  let rows = DATA.filter(r => currentFilter==='ALL' || r.signal===currentFilter);
  if(currentHorizon!=='ALL') rows = rows.filter(r=>r.horizon===currentHorizon);
  if(currentSearch) rows = rows.filter(r=>r.symbol.toLowerCase().includes(currentSearch.toLowerCase()));
  if(currentSort==='score') rows.sort((a,b)=>b.score-a.score);
  if(currentSort==='volatility_pct') rows.sort((a,b)=>b.volatility_pct-a.volatility_pct);
  if(currentSort==='symbol') rows.sort((a,b)=>a.symbol.localeCompare(b.symbol));
  document.getElementById('empty').style.display = rows.length ? 'none':'block';

  rows.forEach(r=>{
    const card=document.createElement('div'); card.className='card';
    card.innerHTML = `
      <div class="card-top">
        <div><div class="sym">${r.symbol}${staleTagHtml(r.as_of)}</div><div class="price">PKR ${fmt(r.price)} &middot; as of ${r.as_of}</div></div>
        <div class="badge ${r.signal}">${r.signal}</div>
      </div>
      <div class="horizon ${r.horizon==='LONG-TERM'?'long':(r.horizon==='SHORT-TERM/TRADE'?'short':'')}">${r.horizon}</div>
      <div class="metrics">
        <div class="m">RSI<b>${r.rsi14}</b></div>
        <div class="m">Trend<b>${r.trend}</b></div>
        <div class="m">Volatility<b>${r.volatility_pct}%</b></div>
      </div>
      <div class="reason">${r.reasoning}</div>
      <div class="reason" style="margin-top:4px;"><i>Horizon:</i> ${r.horizon_reasoning}</div>
      <div class="detail" id="detail-mkt-${r.symbol}">
        <canvas id="chart-mkt-${r.symbol}"></canvas>
        <div class="risk-row"><span>Suggested stop-loss</span><span>PKR ${fmt(r.suggested_stop_loss)}</span></div>
        <div class="risk-row"><span>Suggested target</span><span>PKR ${fmt(r.suggested_target)}</span></div>
        ${timeToTargetHtml(r.time_to_target)}
        <div class="risk-row"><span>Suggested shares</span><span>${fmt(r.suggested_shares)}</span></div>
        <div class="risk-row"><span>Position value</span><span>PKR ${fmt(r.suggested_position_value)}</span></div>
        <div class="risk-row"><span>Avg daily value traded</span><span>PKR ${fmt(r.avg_daily_value_pkr)}</span></div>
        ${supportResistanceHtml(r)}
        ${quickAddButtonHtml(r.symbol, r.price)}
      </div>`;
    card.addEventListener('click', (ev) => {
      if(ev.target.classList.contains('quick-add-btn')){
        ev.stopPropagation();
        quickAddFromCard(ev.target.dataset.symbol, ev.target.dataset.price);
        return;
      }
      toggleDetail(r.symbol, `chart-mkt-${r.symbol}`, `detail-mkt-${r.symbol}`, r.history, r.suggested_stop_loss, r.suggested_target, r.volatility_pct);
    });
    list.appendChild(card);
  });

  document.getElementById('cnt-buy').textContent = DATA.filter(r=>r.signal==='BUY').length;
  document.getElementById('cnt-hold').textContent = DATA.filter(r=>r.signal==='HOLD').length;
  document.getElementById('cnt-avoid').textContent = DATA.filter(r=>r.signal==='AVOID').length;
  document.getElementById('cnt-total').textContent = DATA.length;
}

function renderHoldings(){
  const list=document.getElementById('holdingsList'); list.innerHTML='';
  const visibleHoldings = HOLDINGS.filter(h => !REMOVED_SYMBOLS.has(h.symbol));
  document.getElementById('holdingsEmpty').style.display = visibleHoldings.length ? 'none':'block';
  const tabLabel = document.getElementById('holdingsTabLabel');
  if(tabLabel) tabLabel.textContent = `My Holdings (${visibleHoldings.length})`;
  visibleHoldings.forEach(h=>{
    if(h.error){
      const card=document.createElement('div'); card.className='card';
      card.innerHTML = `<div class="sym">${h.symbol}</div><div class="reason">Error: ${h.error}</div>`;
      list.appendChild(card); return;
    }
    const pnlClass = h.pnl_pct >= 0 ? 'pnl-pos' : 'pnl-neg';
    const actionClass = h.action.includes('SELL') ? 'sell' : (h.action.includes('WATCH') || h.action.includes('SOON') ? 'watch' : 'holdgood');
    const liveTag = h.added_live ? '<span class="live-added-tag">ADDED LIVE</span>' : '';
    const card=document.createElement('div'); card.className='card';
    card.innerHTML = `
      <div class="card-top">
        <div><div class="sym">${h.symbol}${liveTag}</div><div class="price">Bought PKR ${fmt(h.buy_price)} on ${h.buy_date} &middot; ${h.quantity} shares &middot; ${h.days_held}d held</div></div>
        <div class="badge ${actionClass}">${h.action}</div>
      </div>
      <div class="horizon ${h.horizon==='LONG-TERM'?'long':(h.horizon==='SHORT-TERM/TRADE'?'short':'')}">${h.horizon}</div>
      <div class="metrics">
        <div class="m">Current<b>PKR ${fmt(h.current_price)}</b></div>
        <div class="m">P/L %<b class="${pnlClass}">${h.pnl_pct>0?'+':''}${h.pnl_pct}%</b></div>
        <div class="m">P/L PKR<b class="${pnlClass}">${h.pnl_value>0?'+':''}${fmt(h.pnl_value)}</b></div>
      </div>
      <div class="reason"><b>Why:</b> ${h.why}</div>
      <div class="reason" style="margin-top:4px;"><b>Timing:</b> ${h.review_note}</div>
      <div class="detail" id="detail-hld-${h.symbol}">
        <canvas id="chart-hld-${h.symbol}"></canvas>
        <div class="risk-row"><span>Stop-loss (sell if below)</span><span>PKR ${fmt(h.stop_loss)}</span></div>
        <div class="risk-row"><span>Target (consider selling at)</span><span>PKR ${fmt(h.target_price)}</span></div>
        ${timeToTargetHtml(h.time_to_target)}
        ${filingsHtml(h.recent_filings)}
        <button class="remove-holding-btn" data-symbol="${h.symbol}">Sold this? Remove it</button>
      </div>`;
    card.addEventListener('click', (ev) => {
      if(ev.target.classList.contains('remove-holding-btn')){
        ev.stopPropagation();
        removeHolding(ev.target.dataset.symbol);
        return;
      }
      toggleDetail(h.symbol, `chart-hld-${h.symbol}`, `detail-hld-${h.symbol}`, h.history, h.stop_loss, h.target_price, null);
    });
    list.appendChild(card);
  });
}

// ============================================================================
// CLIENT-SIDE HOLDING ANALYSIS - lets you add a holding directly in the
// dashboard without re-running the full market scan in Colab. Only used for
// holdings added this way; the original scan's holdings are unaffected.
// Ports the same math psx_app.py itself uses (SMA, RSI, MACD, ATR, trailing
// stop, target, horizon classification) so results are consistent either way.
// ============================================================================

function jsSma(values, window){
  return values.map((_, i) => {
    if(i < window - 1) return null;
    const slice = values.slice(i - window + 1, i + 1);
    return slice.reduce((a,b) => a+b, 0) / window;
  });
}

function jsEma(values, window){
  const k = 2 / (window + 1);
  const out = [values[0]];
  for(let i = 1; i < values.length; i++) out.push(values[i] * k + out[i-1] * (1 - k));
  return out;
}

function jsRsi(closes, window){
  window = window || 14;
  const out = new Array(closes.length).fill(null);
  const gains = [0], losses = [0];
  for(let i = 1; i < closes.length; i++){
    const delta = closes[i] - closes[i-1];
    gains.push(Math.max(delta, 0));
    losses.push(Math.max(-delta, 0));
  }
  for(let i = window; i < closes.length; i++){
    const avgGain = gains.slice(i - window + 1, i + 1).reduce((a,b)=>a+b,0) / window;
    const avgLoss = losses.slice(i - window + 1, i + 1).reduce((a,b)=>a+b,0) / window;
    out[i] = avgLoss === 0 ? 100 : 100 - (100 / (1 + avgGain / avgLoss));
  }
  return out;
}

function jsMacd(closes){
  const ema12 = jsEma(closes, 12), ema26 = jsEma(closes, 26);
  const macdLine = closes.map((_, i) => ema12[i] - ema26[i]);
  const signalLine = jsEma(macdLine, 9);
  return { macdLine, signalLine };
}

function jsAtr(rows, window){
  window = window || 14;
  const trs = [null];
  for(let i = 1; i < rows.length; i++){
    const h = rows[i].high, l = rows[i].low, pc = rows[i-1].close;
    trs.push(Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc)));
  }
  const out = new Array(rows.length).fill(null);
  for(let i = window; i < rows.length; i++){
    const slice = trs.slice(i - window + 1, i + 1).filter(v => v !== null);
    out[i] = slice.reduce((a,b)=>a+b,0) / slice.length;
  }
  return out;
}

// Same backward-looking "how long have moves like this taken historically"
// stat used elsewhere in this app - not a prediction, a real historical count.
function jsEstimateTimeToTarget(closes, targetMovePct, maxLookaheadDays){
  maxLookaheadDays = maxLookaheadDays || 30;
  const n = closes.length;
  if(n < 20) return { hit_rate_pct: null, median_days: null, sample_size: 0 };
  const daysTaken = [];
  let total = 0;
  for(let i = 0; i < n - 1; i++){
    const startPrice = closes[i];
    if(startPrice <= 0) continue;
    total++;
    const targetI = startPrice * (1 + targetMovePct / 100);
    const windowEnd = Math.min(i + 1 + maxLookaheadDays, n);
    for(let j = i + 1; j < windowEnd; j++){
      if((targetMovePct >= 0 && closes[j] >= targetI) || (targetMovePct < 0 && closes[j] <= targetI)){
        daysTaken.push(j - i);
        break;
      }
    }
  }
  if(daysTaken.length === 0 || total === 0) return { hit_rate_pct: 0, median_days: null, sample_size: total };
  const sorted = [...daysTaken].sort((a,b) => a-b);
  const median = sorted[Math.floor(sorted.length / 2)];
  return { hit_rate_pct: Math.round((daysTaken.length / total) * 1000) / 10, median_days: median, sample_size: total };
}

// Parses PSX's /historical table response (via your Apps Script proxy) into
// an array of {date, open, high, low, close, volume}, matching header text
// generically rather than assuming exact column order.
function parseHistoricalHtml(html){
  const doc = new DOMParser().parseFromString(html, 'text/html');
  const table = doc.querySelector('table');
  if(!table) throw new Error('No data table found - check the symbol is correct and your proxy is working');

  const headerCells = Array.from(table.querySelectorAll('thead th, tr:first-child th, tr:first-child td'))
    .map(c => c.textContent.trim().toUpperCase());
  const find = (needle) => headerCells.findIndex(h => h.includes(needle));
  const idx = { date: find('DATE'), open: find('OPEN'), high: find('HIGH'), low: find('LOW'), close: find('CLOSE'), volume: find('VOL') };
  if(idx.date === -1 || idx.close === -1) throw new Error('Historical data format not recognized');

  const rows = [];
  table.querySelectorAll('tbody tr, tr').forEach(row => {
    const cells = row.querySelectorAll('td');
    if(cells.length <= Math.max(idx.date, idx.close)) return;
    const dateText = cells[idx.date].textContent.trim();
    const parsedDate = new Date(dateText);
    if(isNaN(parsedDate.getTime())) return;
    const num = (i) => i === -1 || !cells[i] ? null : parseFloat(cells[i].textContent.replace(/,/g, ''));
    const close = num(idx.close);
    if(close === null || isNaN(close)) return;
    rows.push({
      date: parsedDate.toISOString().slice(0,10),
      open: num(idx.open) ?? close,
      high: num(idx.high) ?? close,
      low: num(idx.low) ?? close,
      close,
      volume: num(idx.volume) ?? 0,
    });
  });
  rows.sort((a,b) => new Date(a.date) - new Date(b.date)); // PSX returns newest-first; we need oldest-first
  return rows;
}

async function fetchHistoricalViaProxy(symbol){
  if(!LIVE_REFRESH_PROXY_URL) throw new Error('Live-refresh proxy not configured - set it up first (see README)');
  const resp = await fetch(`${LIVE_REFRESH_PROXY_URL}?symbol=${encodeURIComponent(symbol)}`);
  if(!resp.ok) throw new Error(`Proxy responded with status ${resp.status}`);
  const html = await resp.text();
  return parseHistoricalHtml(html);
}

function analyzeHoldingClientSide(symbol, buyPrice, buyDateStr, quantity, rows){
  if(rows.length < 30) throw new Error(`Only ${rows.length} days of history found - need at least 30 for reliable analysis`);

  const closes = rows.map(r => r.close);
  const sma20Arr = jsSma(closes, 20);
  const sma50Arr = jsSma(closes, 50);
  const rsiArr = jsRsi(closes, 14);
  const atrArr = jsAtr(rows, 14);
  const { macdLine, signalLine } = jsMacd(closes);

  const n = rows.length;
  const last = rows[n-1];
  const lastSma20 = sma20Arr[n-1], lastSma50 = sma50Arr[n-1], lastRsi = rsiArr[n-1], lastAtr = atrArr[n-1];
  if(lastAtr === null) throw new Error('Not enough history to compute volatility for this stock yet');

  const currentPrice = last.close;
  const buyDate = new Date(buyDateStr);
  const daysHeld = Math.round((new Date() - buyDate) / (1000*60*60*24));
  const pnlPct = Math.round(((currentPrice - buyPrice) / buyPrice) * 1000) / 10;
  const pnlValue = Math.round((currentPrice - buyPrice) * quantity * 100) / 100;

  const sinceBuy = rows.filter(r => new Date(r.date) >= buyDate);
  const highestCloseSinceBuy = sinceBuy.length ? Math.max(...sinceBuy.map(r => r.close)) : currentPrice;
  const trailingStop = Math.round((highestCloseSinceBuy - 2 * lastAtr) * 100) / 100;
  const stopLoss = trailingStop > 0 ? trailingStop : Math.round((buyPrice - 2 * lastAtr) * 100) / 100;
  const targetPrice = Math.round((buyPrice + 4 * lastAtr) * 100) / 100;

  // Horizon classification, same heuristic as the main scan
  let longScore = 0, shortScore = 0;
  const sma50_20ago = n >= 20 && sma50Arr[n-20] !== null ? sma50Arr[n-20] : null;
  const sma50SlopePct = sma50_20ago ? ((lastSma50 - sma50_20ago) / sma50_20ago) * 100 : 0;
  const volatilityPct = (lastAtr / currentPrice) * 100;
  if(Math.abs(sma50SlopePct) > 3) longScore += 2;
  if(volatilityPct < 3) longScore += 1;
  if(volatilityPct > 3.5) shortScore += 2;
  if(lastRsi !== null && (lastRsi < 30 || lastRsi > 75)) shortScore += 1;
  const horizon = longScore > shortScore ? 'LONG-TERM' : (shortScore > longScore ? 'SHORT-TERM/TRADE' : 'EITHER');

  const trendDown = currentPrice < lastSma20 && lastSma20 < lastSma50;

  let action, why;
  if(currentPrice >= targetPrice){
    action = 'SELL NOW (or take partial profit)';
    why = `price has reached the target (PKR ${targetPrice}) — consider locking in gains`;
  } else if(currentPrice <= stopLoss){
    action = 'SELL NOW';
    why = `price has fallen to/below the stop-loss (PKR ${stopLoss})` + (pnlPct > 0 ? ' — this is a trailing stop, so it can also mean giving back gains from a recent peak' : '');
  } else if(trendDown && pnlPct > 0){
    action = 'CONSIDER SELLING SOON';
    why = 'trend has turned down while still in profit — a good time to review';
  } else if(trendDown && pnlPct <= 0){
    action = 'HOLD, BUT WATCH CLOSELY';
    why = 'trend has turned down and at a loss — the stop-loss is your safety net';
  } else {
    action = 'HOLD';
    why = 'no exit signal yet — trend/momentum still intact';
  }
  const reviewNote = horizon === 'SHORT-TERM/TRADE'
    ? 'Short-term/trade setup — review daily.'
    : horizon === 'LONG-TERM'
      ? 'Long-term setup — review weekly, react to the stop-loss or a real trend break, not daily noise.'
      : 'No strong horizon lean — check every few days.';

  const targetMovePct = ((targetPrice - currentPrice) / currentPrice) * 100;
  const timeToTarget = jsEstimateTimeToTarget(closes, targetMovePct);

  const history = rows.slice(-90).map((r, i, arr) => {
    const fullIdx = n - arr.length + i;
    return { date: r.date, open: r.open, high: r.high, low: r.low, close: r.close, sma20: sma20Arr[fullIdx], sma50: sma50Arr[fullIdx], rsi14: rsiArr[fullIdx] };
  });

  return {
    symbol: symbol.toUpperCase(), buy_price: buyPrice, buy_date: buyDateStr, quantity,
    days_held: daysHeld, current_price: Math.round(currentPrice*100)/100,
    pnl_pct: pnlPct, pnl_value: pnlValue, horizon,
    stop_loss: stopLoss, target_price: targetPrice, time_to_target: timeToTarget,
    action, why, review_note: reviewNote, as_of: last.date,
    recent_filings: [], history, added_live: true,
  };
}

function saveClientHoldingsToStorage(){
  try {
    const clientOnes = HOLDINGS.filter(h => h.added_live);
    localStorage.setItem('psx_client_holdings', JSON.stringify(clientOnes));
  } catch(e) { /* localStorage unavailable - not fatal, just won't persist */ }
}

function loadClientHoldingsFromStorage(){
  try {
    const raw = localStorage.getItem('psx_client_holdings');
    if(raw){
      const saved = JSON.parse(raw);
      saved.forEach(h => { if(!HOLDINGS.find(x => x.symbol === h.symbol)) HOLDINGS.push(h); });
    }
  } catch(e) { /* corrupted/unavailable storage - just start empty */ }
  try {
    const rawRemoved = localStorage.getItem('psx_removed_holdings');
    if(rawRemoved) JSON.parse(rawRemoved).forEach(s => REMOVED_SYMBOLS.add(s));
  } catch(e) { /* ignore */ }
}

function saveRemovedSymbolsToStorage(){
  try { localStorage.setItem('psx_removed_holdings', JSON.stringify([...REMOVED_SYMBOLS])); }
  catch(e) { /* not fatal */ }
}

// Works for BOTH kinds of holding: one you added live in the dashboard (fully
// removed from memory) and one that came from the original Colab scan (kept
// in memory, but hidden - since it's baked into psx_dashboard.html - and
// remembered as hidden across reloads, e.g. because you sold the stock).
function removeHolding(symbol){
  const liveIdx = HOLDINGS.findIndex(h => h.symbol === symbol && h.added_live);
  if(liveIdx !== -1){
    HOLDINGS.splice(liveIdx, 1);
    saveClientHoldingsToStorage();
  } else {
    REMOVED_SYMBOLS.add(symbol);
    saveRemovedSymbolsToStorage();
  }
  renderHoldings();
  renderActionAlert();
}

async function addHoldingLive(){
  const symbolInput = document.getElementById('addHSymbol');
  const priceInput = document.getElementById('addHPrice');
  const dateInput = document.getElementById('addHDate');
  const qtyInput = document.getElementById('addHQty');
  const statusEl = document.getElementById('addHStatus');
  const btn = document.getElementById('addHBtn');

  const symbol = symbolInput.value.trim().toUpperCase();
  const buyPrice = parseFloat(priceInput.value);
  const buyDate = dateInput.value;
  const quantity = parseInt(qtyInput.value, 10);

  if(!symbol){ statusEl.textContent = 'Enter a symbol.'; statusEl.style.color = 'var(--avoid)'; return; }
  if(!buyPrice || buyPrice <= 0){ statusEl.textContent = 'Enter a valid buy price.'; statusEl.style.color = 'var(--avoid)'; return; }
  if(!buyDate){ statusEl.textContent = 'Enter the buy date.'; statusEl.style.color = 'var(--avoid)'; return; }
  if(!quantity || quantity <= 0){ statusEl.textContent = 'Enter a valid quantity.'; statusEl.style.color = 'var(--avoid)'; return; }

  btn.disabled = true;
  statusEl.style.color = 'var(--muted)';
  statusEl.textContent = `Fetching and analyzing ${symbol}…`;

  try {
    const rows = await fetchHistoricalViaProxy(symbol);
    const holding = analyzeHoldingClientSide(symbol, buyPrice, buyDate, quantity, rows);
    const existingIdx = HOLDINGS.findIndex(h => h.symbol === holding.symbol);
    if(existingIdx !== -1) HOLDINGS[existingIdx] = holding; else HOLDINGS.push(holding);
    saveClientHoldingsToStorage();
    renderHoldings();
    renderActionAlert();
    statusEl.textContent = `✓ ${symbol} added and analyzed — see it below.`;
    statusEl.style.color = 'var(--buy)';
    symbolInput.value = ''; priceInput.value = ''; dateInput.value = ''; qtyInput.value = '';
  } catch(err) {
    statusEl.textContent = `✗ Couldn't add ${symbol}: ${err.message}`;
    statusEl.style.color = 'var(--avoid)';
  } finally {
    btn.disabled = false;
  }
}

function computeVolatilityFallback(history){
  // If volatility_pct wasn't passed in (e.g. holdings), estimate a simple
  // ATR-like measure directly from the last 14 candles as a fallback.
  const recent = history.slice(-14);
  if(recent.length < 2) return 2;
  let trSum = 0;
  for(let i = 1; i < recent.length; i++){
    const h = recent[i].high, l = recent[i].low, pc = recent[i-1].close;
    trSum += Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc));
  }
  const atr = trSum / (recent.length - 1);
  const lastClose = recent[recent.length - 1].close;
  return lastClose > 0 ? (atr / lastClose) * 100 : 2;
}

function computeTrendProjection(history, volatilityPct, projectionDays, livePriceOverride){
  projectionDays = projectionDays || 10;
  const vol = (volatilityPct !== null && volatilityPct !== undefined) ? volatilityPct : computeVolatilityFallback(history);
  // The slope (direction/steepness of the trend) always comes from real
  // historical closes - that doesn't change just because we know today's
  // live price. Only the STARTING POINT for projecting forward updates,
  // so the line begins from where the stock actually is right now instead
  // of where it was when the scan last ran.
  const lastHistoricalClose = history[history.length - 1].close;
  const anchorPrice = (livePriceOverride !== null && livePriceOverride !== undefined) ? livePriceOverride : lastHistoricalClose;
  const atrLike = anchorPrice * (vol / 100);

  const trendWindow = Math.min(20, history.length);
  const recentCloses = history.slice(-trendWindow).map(h => h.close);
  const n = recentCloses.length;
  const xs = recentCloses.map((_, i) => i);
  const xMean = xs.reduce((a,b) => a+b, 0) / n;
  const yMean = recentCloses.reduce((a,b) => a+b, 0) / n;
  let num = 0, den = 0;
  for(let i = 0; i < n; i++){ num += (xs[i]-xMean)*(recentCloses[i]-yMean); den += (xs[i]-xMean)**2; }
  const slopePerDay = den !== 0 ? num / den : 0;

  const projected = [];
  for(let d = 1; d <= projectionDays; d++){
    const trendCenter = Math.max(anchorPrice + slopePerDay * d, 0.01);
    const spread = atrLike * Math.sqrt(d) * 1.2;
    projected.push({ center: trendCenter, upper: trendCenter + spread, lower: Math.max(trendCenter - spread, 0.01) });
  }
  return { lastClose: anchorPrice, slopePerDay, trendUp: slopePerDay >= 0, projected, isLiveAnchored: livePriceOverride !== null && livePriceOverride !== undefined };
}

function addTradingDays(dateStr, n){
  const d = new Date(dateStr);
  let added = 0;
  while(added < n){
    d.setDate(d.getDate() + 1);
    if(d.getDay() !== 0 && d.getDay() !== 6) added++;
  }
  return d;
}
function fmtShortDate(d){ return d.toLocaleDateString('en-GB', {day:'numeric', month:'short'}); }

function drawCandleChart(canvasId, history, stopLoss, target, volatilityPct, livePriceOverride){
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.parentElement.clientWidth || 320;
  const cssHeight = 220;
  canvas.style.width = cssWidth + 'px';
  canvas.style.height = cssHeight + 'px';
  canvas.width = cssWidth * dpr;
  canvas.height = cssHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const { lastClose, slopePerDay, projected, isLiveAnchored } = computeTrendProjection(history, volatilityPct, 10, livePriceOverride);

  const allPrices = [];
  history.forEach(h => { allPrices.push(h.high, h.low); });
  projected.forEach(p => { allPrices.push(p.upper, p.lower); });
  if(stopLoss) allPrices.push(stopLoss);
  if(target) allPrices.push(target);
  const maxP = Math.max(...allPrices);
  const minP = Math.min(...allPrices);
  const pad = (maxP - minP) * 0.08 || 1;
  const yMax = maxP + pad, yMin = Math.max(minP - pad, 0);

  const leftMargin = 54, rightMargin = 8, topMargin = 10, bottomMargin = 18;
  const plotW = cssWidth - leftMargin - rightMargin;
  const plotH = cssHeight - topMargin - bottomMargin;
  const totalBars = history.length + projected.length;
  const barSlot = plotW / totalBars;
  const candleW = Math.max(1.5, Math.min(barSlot * 0.6, 8));

  function yFor(price){ return topMargin + (yMax - price) / (yMax - yMin) * plotH; }
  function xFor(index){ return leftMargin + index * barSlot + barSlot / 2; }

  // Gridlines + y-axis labels
  ctx.strokeStyle = '#263230'; ctx.lineWidth = 1; ctx.font = '9px sans-serif'; ctx.fillStyle = '#8fa39a';
  for(let i = 0; i <= 4; i++){
    const price = yMin + (yMax - yMin) * (i / 4);
    const y = yFor(price);
    ctx.beginPath(); ctx.moveTo(leftMargin, y); ctx.lineTo(cssWidth - rightMargin, y); ctx.stroke();
    ctx.fillText(price.toFixed(1), 2, y + 3);
  }

  // Historical candlesticks (real OHLC data)
  history.forEach((h, i) => {
    const x = xFor(i);
    const up = h.close >= h.open;
    ctx.strokeStyle = up ? '#2fbf71' : '#e5555a';
    ctx.fillStyle = up ? '#2fbf71' : '#e5555a';
    ctx.beginPath(); ctx.moveTo(x, yFor(h.high)); ctx.lineTo(x, yFor(h.low)); ctx.stroke();
    const bodyTop = yFor(Math.max(h.open, h.close));
    const bodyBottom = yFor(Math.min(h.open, h.close));
    ctx.fillRect(x - candleW / 2, bodyTop, candleW, Math.max(bodyBottom - bodyTop, 1));
  });

  // SMA20 / SMA50 overlay lines
  function drawLine(key, color){
    ctx.strokeStyle = color; ctx.lineWidth = 1.2; ctx.beginPath();
    let started = false;
    history.forEach((h, i) => {
      if(h[key] === null || h[key] === undefined || isNaN(h[key])) return;
      const x = xFor(i), y = yFor(h[key]);
      if(!started){ ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
    });
    ctx.stroke();
  }
  drawLine('sma20', '#d9a441');
  drawLine('sma50', '#8fa39a');

  // Shaded forward volatility cone (range only, explicitly not a direction call)
  ctx.fillStyle = 'rgba(143,163,154,0.15)';
  ctx.beginPath();
  ctx.moveTo(xFor(history.length - 1), yFor(lastClose));
  projected.forEach((p, i) => ctx.lineTo(xFor(history.length + i), yFor(p.upper)));
  for(let i = projected.length - 1; i >= 0; i--) ctx.lineTo(xFor(history.length + i), yFor(projected[i].lower));
  ctx.closePath(); ctx.fill();
  ctx.strokeStyle = 'rgba(143,163,154,0.5)'; ctx.setLineDash([3,3]); ctx.lineWidth = 1;
  ctx.beginPath();
  projected.forEach((p, i) => { const x = xFor(history.length + i); if(i===0) ctx.moveTo(x, yFor(p.upper)); else ctx.lineTo(x, yFor(p.upper)); });
  ctx.stroke();
  ctx.beginPath();
  projected.forEach((p, i) => { const x = xFor(history.length + i); if(i===0) ctx.moveTo(x, yFor(p.lower)); else ctx.lineTo(x, yFor(p.lower)); });
  ctx.stroke();
  ctx.setLineDash([]);

  // Trend-continuation line: a REAL linear regression extended forward -
  // "if the recent trend keeps going at this pace." One labeled scenario,
  // not a forecast. Color leans toward direction (up=green-ish, down=red-ish)
  // purely for readability, not to imply confidence.
  const trendUp = slopePerDay >= 0;
  ctx.strokeStyle = trendUp ? 'rgba(47,191,113,0.8)' : 'rgba(229,85,90,0.8)';
  ctx.setLineDash([6,3]); ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(xFor(history.length - 1), yFor(lastClose));
  projected.forEach((p, i) => ctx.lineTo(xFor(history.length + i), yFor(p.center)));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = trendUp ? '#2fbf71' : '#e5555a';
  ctx.font = 'bold 9px sans-serif';
  const lastProjX = xFor(history.length + projected.length - 1);
  const trendLabel = (trendUp ? '↗' : '↘') + ' if trend continues' + (isLiveAnchored ? ' (from live price)' : '');
  ctx.fillText(trendLabel, Math.min(lastProjX - 80, cssWidth - 130), yFor(projected[projected.length-1].center) - 4);

  // Real day/date markers along the trend line - PSX data is daily, so this
  // is genuine trading-day granularity (weekends skipped), not invented
  // hourly precision. Marked at a few points so it stays readable.
  const lastDate = isLiveAnchored ? new Date().toISOString().slice(0,10) : history[history.length - 1].date;
  const markerDays = [3, 7, 10].filter(d => d <= projected.length);
  markerDays.forEach(dayNum => {
    const p = projected[dayNum - 1];
    const x = xFor(history.length + dayNum - 1);
    const y = yFor(p.center);
    ctx.fillStyle = trendUp ? '#2fbf71' : '#e5555a';
    ctx.beginPath(); ctx.arc(x, y, 2.5, 0, Math.PI*2); ctx.fill();
    const dateLabel = fmtShortDate(addTradingDays(lastDate, dayNum));
    ctx.font = '8.5px sans-serif';
    ctx.fillText(`${p.center.toFixed(1)}`, x - 14, y - 8);
    ctx.fillStyle = '#8fa39a';
    ctx.fillText(dateLabel, x - 14, y + 12);
  });

  // Stop-loss / target reference lines
  function drawRefLine(price, color, label){
    if(price === null || price === undefined) return;
    const y = yFor(price);
    ctx.strokeStyle = color; ctx.setLineDash([5,3]); ctx.lineWidth = 1.3;
    ctx.beginPath(); ctx.moveTo(leftMargin, y); ctx.lineTo(cssWidth - rightMargin, y); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color; ctx.font = 'bold 9px sans-serif';
    ctx.fillText(label, cssWidth - rightMargin - 60, y - 3);
  }
  drawRefLine(stopLoss, '#e5555a', 'STOP');
  drawRefLine(target, '#2fbf71', 'TARGET');

  // Divider between real history and the projected range
  ctx.strokeStyle = '#3a4744'; ctx.setLineDash([2,4]); ctx.lineWidth = 1;
  const divX = xFor(history.length - 1) + barSlot / 2;
  ctx.beginPath(); ctx.moveTo(divX, topMargin); ctx.lineTo(divX, cssHeight - bottomMargin); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#8fa39a'; ctx.font = '9px sans-serif';
  ctx.fillText('history', leftMargin, cssHeight - 4);
  ctx.fillText('trend scenario, not a forecast', divX + 4, cssHeight - 4);
}

// Populated after a successful live refresh: { SYMBOL: price }. Charts check
// this so a currently-open or newly-opened chart uses the freshest known
// price as its trend-projection anchor, instead of the stale scan price.
let LIVE_PRICES = {};

function toggleDetail(symbol, canvasId, detailId, history, stopLoss, target, volatilityPct){
  const el=document.getElementById(detailId);
  const wasOpen = el.classList.contains('open');
  el.classList.toggle('open');
  if(!wasOpen){
    const canvas = document.getElementById(canvasId);
    canvas._lastDrawArgs = [history, stopLoss, target, volatilityPct, symbol];
    // Redraw every time (cheap canvas op) so it's correct even if the container was resized
    requestAnimationFrame(() => drawCandleChart(canvasId, history, stopLoss, target, volatilityPct, LIVE_PRICES[symbol]));
  }
}

// Redraws every currently-open chart in place, e.g. right after a live
// refresh, so open charts reflect the new price without needing to be
// closed and reopened.
function redrawOpenCharts(){
  document.querySelectorAll('.detail.open canvas').forEach(canvas => {
    if(canvas._lastDrawArgs){
      const [history, stopLoss, target, volatilityPct, symbol] = canvas._lastDrawArgs;
      drawCandleChart(canvas.id, history, stopLoss, target, volatilityPct, LIVE_PRICES[symbol]);
    }
  });
}


document.getElementById('search').addEventListener('input', e=>{currentSearch=e.target.value; render();});
document.getElementById('sortBy').addEventListener('change', e=>{currentSort=e.target.value; render();});
document.getElementById('horizonFilter').addEventListener('change', e=>{currentHorizon=e.target.value; render();});
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click', ()=>{
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active')); tab.classList.add('active');
  currentFilter=tab.dataset.filter; render();
}));
document.querySelectorAll('.maintab').forEach(mt=>mt.addEventListener('click', ()=>{
  document.querySelectorAll('.maintab').forEach(t=>t.classList.remove('active')); mt.classList.add('active');
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.getElementById('page-'+mt.dataset.page).classList.add('active');
}));

function fmtPkr(n){ return 'PKR ' + Math.round(n).toLocaleString(); }

function populateCalcSymbols(){
  const sel = document.getElementById('calcSymbol');
  const sorted = [...DATA].sort((a,b)=>a.symbol.localeCompare(b.symbol));
  sel.innerHTML = sorted.map(r => `<option value="${r.symbol}">${r.symbol} — ${r.signal} (PKR ${r.price})</option>`).join('');
}

function renderCalc(){
  const capital = parseFloat(document.getElementById('calcCapital').value) || 0;
  const symbol = document.getElementById('calcSymbol').value;
  const r = DATA.find(x => x.symbol === symbol);
  const out = document.getElementById('calcOutput');
  if(!r || capital <= 0){ out.innerHTML = ''; return; }

  const shares = Math.floor(capital / r.price);
  const positionValue = shares * r.price;
  const leftoverCash = capital - positionValue;

  const targetTotal = leftoverCash + shares * r.suggested_target;
  const targetGain = targetTotal - capital;
  const targetPct = (targetGain / capital) * 100;

  const stopTotal = leftoverCash + shares * r.suggested_stop_loss;
  const stopLoss = stopTotal - capital;
  const stopPct = (stopLoss / capital) * 100;

  const t = r.time_to_target;
  let timeLine;
  if(!t || t.sample_size === 0){
    timeLine = 'not enough history to estimate';
  } else if(t.hit_rate_pct === 0 || t.median_days === null){
    timeLine = 'a move this size has rarely/never happened within 30 days historically for this stock';
  } else {
    timeLine = `historically ~${t.median_days} trading day${t.median_days===1?'':'s'} (happened ${t.hit_rate_pct}% of the time when checked historically) — not a countdown, just a pace reference`;
  }

  out.innerHTML = `
    <div class="calc-result">
      <div class="calc-row"><span>Current signal</span><span>${r.signal} &middot; ${r.horizon}</span></div>
      <div class="calc-row"><span>Current price</span><span>${fmtPkr(r.price)}</span></div>
      <div class="calc-row"><span>Shares you could buy</span><span>${shares.toLocaleString()}</span></div>
      <div class="calc-row"><span>Amount actually invested</span><span>${fmtPkr(positionValue)}</span></div>
      <div class="calc-row"><span>Leftover (can't buy a fraction of a share)</span><span>${fmtPkr(leftoverCash)}</span></div>
      <div class="calc-row"><span>Estimated time to target</span><span>${timeLine}</span></div>

      <div class="scenario good">
        <div class="title">If it reaches its target (${fmtPkr(r.suggested_target)})</div>
        <div class="big">${fmtPkr(targetTotal)}</div>
        <div class="sub">${targetGain>=0?'+':''}${fmtPkr(targetGain)} (${targetPct>=0?'+':''}${targetPct.toFixed(1)}%)</div>
      </div>
      <div class="scenario bad">
        <div class="title">If it hits its stop-loss instead (${fmtPkr(r.suggested_stop_loss)})</div>
        <div class="big">${fmtPkr(stopTotal)}</div>
        <div class="sub">${fmtPkr(stopLoss)} (${stopPct.toFixed(1)}%)</div>
      </div>
      ${trendScenarioHtml(r, shares, leftoverCash, capital)}
    </div>`;
}

function trendScenarioHtml(r, shares, leftoverCash, capital){
  if(!r.history || r.history.length < 10) return '';
  const { trendUp, projected } = computeTrendProjection(r.history, r.volatility_pct, 10);
  const lastDate = r.history[r.history.length - 1].date;
  const rows = [3, 7, 10].filter(d => d <= projected.length).map(dayNum => {
    const p = projected[dayNum - 1];
    const total = leftoverCash + shares * p.center;
    const gain = total - capital;
    const pct = (gain / capital) * 100;
    const dateLabel = fmtShortDate(addTradingDays(lastDate, dayNum));
    return `<div class="calc-row"><span>Day ${dayNum} (~${dateLabel})</span><span>${fmtPkr(total)} (${gain>=0?'+':''}${pct.toFixed(1)}%)</span></div>`;
  }).join('');
  return `
    <div class="scenario" style="background:var(--panel-2); border:1px solid var(--line);">
      <div class="title" style="color:var(--text);">${trendUp?'↗':'↘'} If the recent trend continues at its current pace</div>
      ${rows}
      <div class="sub" style="margin-top:6px; color:var(--muted);">Based on a real trend line from the last 20 trading days, extended forward. One scenario assuming nothing changes — not a forecast. Trends reverse without warning.</div>
    </div>`;
}

document.getElementById('calcCapital').addEventListener('input', renderCalc);
document.getElementById('calcSymbol').addEventListener('change', renderCalc);
document.querySelectorAll('.maintab').forEach(mt=>{
  mt.addEventListener('click', ()=>{ if(mt.dataset.page==='calc') renderCalc(); });
});

loadClientHoldingsFromStorage();
render();
renderHoldings();
renderMood();
renderPicks();
renderDipWatch();
renderStaleness();
renderActionAlert();
populateCalcSymbols();
renderCalc();
startAutoRefresh();
</script>
</body>
</html>
"""


def build_dashboard(market_results, holdings_results, cfg, out_path="psx_dashboard.html"):
    market_sorted = sorted(market_results, key=lambda r: r["score"], reverse=True)
    top_picks = compute_top_picks(market_results, cfg)
    dip_watch = compute_dip_watch(market_results, cfg)
    market_mood = compute_market_overview(market_results)
    now = datetime.now()
    html = (HTML_TEMPLATE
            .replace("__GENERATED_AT__", now.strftime("%Y-%m-%d %H:%M"))
            .replace("__GENERATED_AT_ISO__", now.isoformat())
            .replace("__TOTAL__", str(len(market_sorted)))
            .replace("__HOLDINGS_COUNT__", str(len(holdings_results)))
            .replace("__DATA_JSON__", json.dumps(market_sorted, default=str))
            .replace("__HOLDINGS_JSON__", json.dumps(holdings_results, default=str))
            .replace("__TOP_PICKS_JSON__", json.dumps(top_picks, default=str))
            .replace("__DIP_WATCH_JSON__", json.dumps(dip_watch, default=str))
            .replace("__MARKET_MOOD_JSON__", json.dumps(market_mood, default=str))
            .replace("__LIVE_REFRESH_PROXY_URL__", cfg.get("live_refresh_proxy_url", "")))
    Path(out_path).write_text(html)
    return out_path


# ---------------------------------- Main -----------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("PSX All-In-One App — read-only, informational, not financial advice")
    print("=" * 70)

    market_results = run_screen(CONFIG)
    holdings_results = run_holdings(CONFIG)

    if market_results:
        summary_rows = [{k: v for k, v in r.items() if k != "history"} for r in market_results]
        pd.DataFrame(summary_rows).sort_values("score", ascending=False).to_csv("psx_screen_results.csv", index=False)
        print(f"\nSaved psx_screen_results.csv ({len(summary_rows)} stocks)")

        top_picks = compute_top_picks(market_results, CONFIG)
        print(f"\nToday's Top Picks ({len(top_picks)}):")
        for i, r in enumerate(top_picks, 1):
            print(f"  {i}. {r['symbol']} — {r['signal']} (score {r['score']}, {r['horizon']})")

        dip_watch = compute_dip_watch(market_results, CONFIG)
        print(f"\nDip Watch — deeply oversold, contrarian bounce candidates ({len(dip_watch)}):")
        for i, r in enumerate(dip_watch, 1):
            print(f"  {i}. {r['symbol']} — RSI {r['rsi14']}, still in a downtrend (speculative, not a confirmed buy)")

        mood = compute_market_overview(market_results)
        print(f"\nMarket mood: {mood['mood']} — {mood['headline']}")

    out_path = build_dashboard(market_results, holdings_results, CONFIG)
    print(f"\nSaved {out_path}")

    # Also save a copy as index.html - this is what makes GitHub Pages serve
    # it directly at your clean root URL (username.github.io/reponame/)
    # instead of requiring people to know and type the exact filename.
    import shutil
    shutil.copy(out_path, "index.html")
    print("Also saved a copy as index.html (for GitHub Pages)")

    # In a normal environment (your own laptop/Colab), open it automatically.
    # In an automated environment (e.g. GitHub Actions, no browser/display
    # exists) this would crash - skip it there instead of failing the run.
    if not os.environ.get("CI") and not os.environ.get("GITHUB_ACTIONS"):
        try:
            print("Opening dashboard in your browser...")
            webbrowser.open(f"file://{Path(out_path).resolve()}")
        except Exception:
            pass  # headless environment - that's fine, the file was still saved

    print("\nDone. Re-run this same command whenever you want fresh signals.")
