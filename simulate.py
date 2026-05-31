#!/usr/bin/env python3
"""
Mode simulation — vérifie si une cassure de mèche était détectée
pour une paire donnée à un instant précis.

Usage :
    python3 simulate.py "EUR/USD" "31/05/2026/13/00"
    python3 simulate.py "XAU/USD" "15/04/2026/09/00"
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime, timezone, timedelta
import pandas as pd
import requests
import yfinance as yf

_YF_SESSION = requests.Session()
_YF_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
})

from forex_bot import (
    FOREX_PAIRS,
    ATR_PERIOD,
    SWING_LOOKBACK,
    SWING_SEARCH_WINDOW,
    MIN_WICK_BODY_RATIO,
    compute_atr,
    _find_swings,
    detect_wick_break,
)

# ─── Couleurs terminal ────────────────────────────────────────────────────────
G   = "\033[92m"
R   = "\033[91m"
Y   = "\033[93m"
B   = "\033[96m"
W   = "\033[97m"
GLD = "\033[93m"
DIM = "\033[2m"
RST = "\033[0m"

def ok(condition: bool) -> str:
    return f"{G}✅{RST}" if condition else f"{R}❌{RST}"

def line(char="─", n=60):
    print(char * n)


# ─── Fetch historique jusqu'à une date précise ────────────────────────────────

def fetch_until(yf_ticker: str, target_dt: datetime) -> pd.DataFrame | None:
    start = target_dt - timedelta(days=20)
    end   = target_dt + timedelta(hours=2)
    try:
        df = yf.download(
            yf_ticker,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d %H:%M:%S"),
            interval="1h",
            progress=False,
            auto_adjust=True,
            session=_YF_SESSION,
        )
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        print(f"{R}Erreur fetch : {e}{RST}")
        return None


def slice_at(df: pd.DataFrame, target_dt: datetime) -> pd.DataFrame | None:
    if df.index.tzinfo is not None and target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=timezone.utc)
    elif df.index.tzinfo is None and target_dt.tzinfo is not None:
        target_dt = target_dt.replace(tzinfo=None)
    sliced = df[df.index <= target_dt]
    return sliced if not sliced.empty else None


# ─── Simulation ───────────────────────────────────────────────────────────────

def simulate(pair: str, target_dt: datetime):
    info = FOREX_PAIRS.get(pair)
    if info is None:
        print(f"{R}Paire inconnue : '{pair}'{RST}")
        print(f"Paires disponibles : {', '.join(FOREX_PAIRS.keys())}")
        return

    line("=")
    print(f"{W}  SIMULATION WICK BREAK — {pair}  @  {target_dt.strftime('%d/%m/%Y %H:%M')} UTC{RST}")
    line("=")

    print(f"{DIM}  Téléchargement des données ({info['yf']})...{RST}")
    df_full = fetch_until(info["yf"], target_dt)
    if df_full is None:
        print(f"{R}  Aucune donnée disponible.{RST}")
        return

    df = slice_at(df_full, target_dt)
    if df is None or len(df) < 30:
        n = len(df) if df is not None else 0
        print(f"{R}  Pas assez de données ({n} bougies).{RST}")
        return

    last_candle_ts = df.index[-1]
    print(f"  Dernière bougie : {B}{last_candle_ts.strftime('%d/%m/%Y %H:%M')}{RST} UTC")
    print(f"  Bougies chargées : {len(df)}")
    line()

    # ── Calcul ATR + recherche des swings ────────────────────────────────
    df = df.copy()
    df["ATR"] = compute_atr(df, ATR_PERIOD)
    last = df.iloc[-1]
    atr  = float(last["ATR"])

    print(f"{W}  DERNIÈRE BOUGIE{RST}")
    print(f"    High  : {B}{last['High']:.5f}{RST}")
    print(f"    Low   : {B}{last['Low']:.5f}{RST}")
    print(f"    Close : {B}{last['Close']:.5f}{RST}")
    print(f"    ATR   : {atr:.6f}")
    line()

    history = df.iloc[:-1]
    swing_highs, swing_lows = _find_swings(history, atr)

    print(f"{W}  SWINGS TROUVÉS (mèche ≥ {MIN_WICK_BODY_RATIO}× corps, lookback={SWING_LOOKBACK}, fenêtre={SWING_SEARCH_WINDOW}h){RST}")
    if swing_highs:
        print(f"  {GLD}Swing Highs :{RST}")
        for idx, wick_level, ts, wick_atr in swing_highs[-5:]:  # 5 plus récents
            broken = "  ← CASSÉ" if float(last["High"]) > wick_level else ""
            print(f"    {ts.strftime('%d/%m %Hh')}  High={wick_level:.5f}  mèche={wick_atr}×ATR{G if broken else ''}{broken}{RST}")
    else:
        print(f"  {DIM}Aucun swing high valide trouvé{RST}")

    if swing_lows:
        print(f"  {GLD}Swing Lows :{RST}")
        for idx, wick_level, ts, wick_atr in swing_lows[-5:]:  # 5 plus récents
            broken = "  ← CASSÉ" if float(last["Low"]) < wick_level else ""
            print(f"    {ts.strftime('%d/%m %Hh')}  Low={wick_level:.5f}  mèche={wick_atr}×ATR{G if broken else ''}{broken}{RST}")
    else:
        print(f"  {DIM}Aucun swing low valide trouvé{RST}")

    line()

    # ── Verdict ───────────────────────────────────────────────────────────
    result = detect_wick_break(df)

    if not result["detected"]:
        print(f"{R}  ✗ Pas de cassure — {result['reject_reason']}{RST}")
        line("=")
        return

    direction   = result["direction"]
    arrow       = "🔼" if direction == "bullish" else "🔽"
    swing_label = "Swing Low" if result["swing_type"] == "low" else "Swing High"
    swing_ts    = result.get("swing_ts")
    ts_str      = swing_ts.strftime("%d/%m %Hh") if swing_ts is not None else "?"

    print(f"{G}  🚨 WICK BREAK DÉTECTÉ{RST}")
    print(f"     Direction  : {W}{direction.upper()} {arrow}{RST}")
    print(f"     {swing_label} du  : {B}{ts_str}{RST}")
    print(f"     Mèche cassée : {GLD}{result['wick_level']:.5f}{RST}  (mèche={result['wick_ratio']}× le corps)")
    print(f"     Cassure de   : {G}{result['break_atr']}×ATR{RST}")
    line("=")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"{Y}Usage : python3 simulate.py \"EUR/USD\" \"31/05/2026/13/00\"{RST}")
        print(f"        python3 simulate.py \"XAU/USD\" \"15/04/2026/09/00\"")
        sys.exit(1)

    pair_arg = sys.argv[1].upper()
    date_arg = sys.argv[2]

    try:
        target = datetime.strptime(date_arg, "%d/%m/%Y/%H/%M").replace(tzinfo=timezone.utc)
    except ValueError:
        print(f"{R}Format de date invalide. Attendu : jj/mm/yyyy/hh/mm  (ex: 31/05/2026/13/00){RST}")
        sys.exit(1)

    if "/" not in pair_arg and len(pair_arg) == 6:
        pair_arg = pair_arg[:3] + "/" + pair_arg[3:]

    simulate(pair_arg, target)
