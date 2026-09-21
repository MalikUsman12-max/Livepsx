# PSX Dashboard

An automated stock screener and portfolio tracker for the Pakistan Stock
Exchange (PSX). Scans the market, scores stocks BUY/HOLD/AVOID, tracks my
holdings, and re-runs itself automatically once a day.

**Live dashboard:** _(paste your GitHub Pages URL here once it's live, e.g.
`https://yourusername.github.io/psx-dashboard/`)_

## What this is

- `psx_app.py` — the main scanner. Pulls real PSX data, computes technical
  signals (RSI, MACD, SMA, ATR), and builds the dashboard.
- `.github/workflows/daily_scan.yml` — runs `psx_app.py` automatically every
  weekday shortly after the market closes, and commits the fresh dashboard
  back to this repo. GitHub Pages then serves it at the live link above.
- `psx_proxy.gs` — a small Google Apps Script that lets the dashboard fetch
  live prices and add new holdings directly in the browser, without needing
  to re-run the scan.

## Not financial advice

Everything here is rule-based signals from historical price and volume
data — a probability nudge, not a guarantee. Always verify the live price
with your broker before acting on anything shown here.
