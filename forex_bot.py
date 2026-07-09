#!/usr/bin/env python3
"""
FX Wicks Bot — GitHub Actions edition
Détecte en M5 les cassures de mèches sur :
  - Swing high/low (1 bougie de chaque côté)
  - Zones d'accumulation (série de bougies en range étroit)
Cassure sous swing low / zone basse → bullish (liquidity sweep)
Cassure au-dessus swing high / zone haute → bearish
Sessions actives : 09h–22h et 01h–04h heure de Paris
"""

import asyncio
import io
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

TZ_PARIS = ZoneInfo("Europe/Paris")

def _now_paris() -> datetime:
    return datetime.now(TZ_PARIS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup

load_dotenv()

# ─── Logging (heure Paris) ────────────────────────────────────────────────────
class _ParisFormatter(logging.Formatter):
    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, TZ_PARIS).timetuple()

_handler = logging.StreamHandler()
_handler.setFormatter(_ParisFormatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[_handler])
log = logging.getLogger(__name__)

# ─── Credentials ──────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHANNEL_ID = os.getenv("TELEGRAM_CHANNEL_ID", "")

# ─── Univers des paires ───────────────────────────────────────────────────────
FOREX_PAIRS: dict[str, dict] = {

    # ── Crypto ─────────────────────────────────────────────────────────────
    "BTC/USD":  {"yf": "BTC-USD",     "tv": "BITSTAMP%3ABTCUSD"},

    # ── Indices US ─────────────────────────────────────────────────────────
    "US30":     {"yf": "YM=F",        "tv": "OANDA%3AUS30USD"},
    "NAS100":   {"yf": "NQ=F",        "tv": "OANDA%3ANAS100USD"},
    "SPX500":   {"yf": "ES=F",        "tv": "OANDA%3ASPX500USD"},

    # ── Matières premières ─────────────────────────────────────────────────
    "XAU/USD":  {"yf": "GC=F",        "tv": "OANDA%3AXAUUSD"},
    "WTI/USD":  {"yf": "CL=F",        "tv": "NYMEX%3ACL1%21"},

    # ── Majeurs ────────────────────────────────────────────────────────────
    "EUR/USD":  {"yf": "EURUSD=X",    "tv": "FX%3AEURUSD"},
    "AUD/USD":  {"yf": "AUDUSD=X",    "tv": "FX%3AAUDUSD"},
    "USD/CAD":  {"yf": "USDCAD=X",    "tv": "FX%3AUSDCAD"},
    "USD/CHF":  {"yf": "USDCHF=X",    "tv": "FX%3AUSDCHF"},
    "USD/JPY":  {"yf": "USDJPY=X",    "tv": "FX%3AUSDJPY"},
    "GBP/USD":  {"yf": "GBPUSD=X",    "tv": "FX%3AGBPUSD"},

    # ── Croisées EUR ───────────────────────────────────────────────────────
    "EUR/GBP":  {"yf": "EURGBP=X",    "tv": "FX%3AEURGBP"},
    "EUR/AUD":  {"yf": "EURAUD=X",    "tv": "FX%3AEURAUD"},
    "EUR/CAD":  {"yf": "EURCAD=X",    "tv": "FX%3AEURCAD"},
    "EUR/JPY":  {"yf": "EURJPY=X",    "tv": "FX%3AEURJPY"},
    "EUR/CHF":  {"yf": "EURCHF=X",    "tv": "FX%3AEURCHF"},

    # ── Croisées GBP ───────────────────────────────────────────────────────
    "GBP/JPY":  {"yf": "GBPJPY=X",    "tv": "FX%3AGBPJPY"},
    "GBP/AUD":  {"yf": "GBPAUD=X",    "tv": "FX%3AGBPAUD"},
    "GBP/CAD":  {"yf": "GBPCAD=X",    "tv": "FX%3AGBPCAD"},
    "GBP/CHF":  {"yf": "GBPCHF=X",    "tv": "FX%3AGBPCHF"},

    # ── Croisées AUD ───────────────────────────────────────────────────────
    "AUD/CAD":  {"yf": "AUDCAD=X",    "tv": "FX%3AAUDCAD"},
    "AUD/JPY":  {"yf": "AUDJPY=X",    "tv": "FX%3AAUDJPY"},
    "AUD/CHF":  {"yf": "AUDCHF=X",    "tv": "FX%3AAUDCHF"},

    # ── Autres croisées ────────────────────────────────────────────────────
    "CAD/JPY":  {"yf": "CADJPY=X",    "tv": "FX%3ACADJPY"},
    "CAD/CHF":  {"yf": "CADCHF=X",    "tv": "FX%3ACADCHF"},
    "CHF/JPY":  {"yf": "CHFJPY=X",    "tv": "FX%3ACHFJPY"},
}

# ─── Paramètres ───────────────────────────────────────────────────────────────
ATR_PERIOD          = 14
SWING_LOOKBACK      = 1     # 1 bougie de chaque côté pour confirmer un swing
SWING_SEARCH_WINDOW = 144   # fenêtre de recherche : 12h de M5 (144 bougies)
MIN_WICK_BODY_RATIO = 1.5   # mèche ≥ 1.5× le corps pour un swing valide
ACCUM_MIN_CANDLES   = 5     # accumulation = au moins 5 bougies M5 consécutives (25 min)
ACCUM_MAX_RANGE_ATR = 1.5   # range max de la zone : 1.5× ATR

COOLDOWN_HOURS      = 4
CHART_RIGHT_MARGIN  = 24    # 2h de M5 à droite du graphique
ALERT_STATE_FILE    = Path("alert_state.json")


# ─── Indicateur ───────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# ─── Récupération des données M5 ─────────────────────────────────────────────

def fetch_m5_data(yf_ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            yf_ticker,
            period="7d",
            interval="5m",
            progress=False,
            auto_adjust=True,
        )
        if df.empty or len(df) < 30:
            log.warning(f"Données insuffisantes pour {yf_ticker} ({len(df)} bougies)")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as exc:
        log.error(f"Erreur fetch {yf_ticker}: {exc}")
        return None


# ─── Filtre horaire ───────────────────────────────────────────────────────────

def _is_active_session() -> bool:
    """True si l'heure Paris est dans une session tradable."""
    h = _now_paris().hour + _now_paris().minute / 60
    return (9.0 <= h < 22.0) or (1.0 <= h < 4.0)


# ─── Détection — Swings ───────────────────────────────────────────────────────

def _find_swings(history: pd.DataFrame, atr: float) -> tuple[list, list]:
    """
    Retourne les swing highs et swing lows confirmés dans history.
    Chaque élément : (idx, wick_level, timestamp, wick_body_ratio)
    Filtre : mèche ≥ MIN_WICK_BODY_RATIO × corps.
    """
    n = len(history)
    swing_highs, swing_lows = [], []

    start = max(SWING_LOOKBACK, n - SWING_SEARCH_WINDOW - SWING_LOOKBACK)
    end   = n - SWING_LOOKBACK

    for i in range(start, end):
        candle = history.iloc[i]
        left   = history.iloc[max(0, i - SWING_LOOKBACK):i]
        right  = history.iloc[i + 1:i + SWING_LOOKBACK + 1]

        if len(left) < SWING_LOOKBACK or len(right) < SWING_LOOKBACK:
            continue

        c_high, c_low   = float(candle["High"]), float(candle["Low"])
        c_open, c_close = float(candle["Open"]), float(candle["Close"])
        body = max(abs(c_open - c_close), atr * 0.05)

        if c_high > left["High"].max() and c_high > right["High"].max():
            wick = c_high - max(c_open, c_close)
            if wick / body >= MIN_WICK_BODY_RATIO:
                swing_highs.append((i, c_high, history.index[i], round(wick / body, 2)))

        if c_low < left["Low"].min() and c_low < right["Low"].min():
            wick = min(c_open, c_close) - c_low
            if wick / body >= MIN_WICK_BODY_RATIO:
                swing_lows.append((i, c_low, history.index[i], round(wick / body, 2)))

    return swing_highs, swing_lows


# ─── Détection — Accumulations ────────────────────────────────────────────────

def _find_accumulations(history: pd.DataFrame, atr: float) -> list:
    """
    Retourne les zones d'accumulation (consolidation) dans history.
    Une zone = au moins ACCUM_MIN_CANDLES bougies consécutives
               dont le range total ≤ ACCUM_MAX_RANGE_ATR × ATR.
    Chaque élément : (start_idx, end_idx, zone_high, zone_low, ts_start, ts_end)
    """
    n = len(history)
    accums = []
    search_start = max(0, n - SWING_SEARCH_WINDOW)

    i = search_start
    while i <= n - ACCUM_MIN_CANDLES:
        window    = history.iloc[i:i + ACCUM_MIN_CANDLES]
        zone_high = float(window["High"].max())
        zone_low  = float(window["Low"].min())

        if zone_high - zone_low > ACCUM_MAX_RANGE_ATR * atr:
            i += 1
            continue

        # Étendre la zone tant qu'elle reste dans le range
        end = i + ACCUM_MIN_CANDLES
        while end < n:
            c = history.iloc[end]
            new_h = max(zone_high, float(c["High"]))
            new_l = min(zone_low,  float(c["Low"]))
            if new_h - new_l <= ACCUM_MAX_RANGE_ATR * atr:
                zone_high, zone_low = new_h, new_l
                end += 1
            else:
                break

        accums.append((
            i, end - 1,
            zone_high, zone_low,
            history.index[i], history.index[end - 1],
        ))
        i = end

    return accums


# ─── Détection principale ─────────────────────────────────────────────────────

def detect_wick_break(df: pd.DataFrame) -> dict:
    """
    Vérifie si la dernière bougie M5 casse :
    1. La mèche d'un swing high/low récent
    2. Le haut/bas d'une zone d'accumulation récente
    Priorité : swing le plus récent → accumulation la plus récente.
    """
    df = df.copy()
    df["ATR"] = compute_atr(df, ATR_PERIOD)

    last = df.iloc[-1]
    atr  = float(last["ATR"])

    base = {
        "detected":      False,
        "reject_reason": "",
        "price":         round(float(last["Close"]), 5),
        "last_high":     round(float(last["High"]), 5),
        "last_low":      round(float(last["Low"]), 5),
        "atr":           round(atr, 6) if not pd.isna(atr) else 0,
    }

    if pd.isna(atr) or atr == 0:
        base["reject_reason"] = "ATR invalide"
        return base

    history     = df.iloc[:-1]
    last_high   = float(last["High"])
    last_low    = float(last["Low"])
    found_stale = False

    swing_highs, swing_lows = _find_swings(history, atr)

    # ── 1. Swings ─────────────────────────────────────────────────────────
    for idx, wick_level, ts, wick_ratio in reversed(swing_highs):
        if last_high > wick_level:
            between = history.iloc[idx + 1:]
            if not between.empty and float(between["High"].max()) > wick_level:
                found_stale = True
                continue
            base.update({
                "detected":       True,
                "signal_source":  "swing",
                "direction":      "bearish",
                "wick_level":     round(wick_level, 5),
                "swing_type":     "high",
                "swing_ts":       ts,
                "swing_idx":      idx,
                "wick_ratio":     wick_ratio,
                "zone_high":      round(wick_level, 5),
                "zone_low":       round(wick_level, 5),
            })
            return base

    for idx, wick_level, ts, wick_ratio in reversed(swing_lows):
        if last_low < wick_level:
            between = history.iloc[idx + 1:]
            if not between.empty and float(between["Low"].min()) < wick_level:
                found_stale = True
                continue
            base.update({
                "detected":       True,
                "signal_source":  "swing",
                "direction":      "bullish",
                "wick_level":     round(wick_level, 5),
                "swing_type":     "low",
                "swing_ts":       ts,
                "swing_idx":      idx,
                "wick_ratio":     wick_ratio,
                "zone_high":      round(wick_level, 5),
                "zone_low":       round(wick_level, 5),
            })
            return base

    # ── 2. Accumulations ──────────────────────────────────────────────────
    accums = _find_accumulations(history, atr)
    zone_range_atr = lambda zh, zl: round((zh - zl) / atr, 2) if atr else 0

    for s_idx, e_idx, zone_high, zone_low, ts_start, ts_end in reversed(accums):
        if last_high > zone_high:
            between = history.iloc[e_idx + 1:]
            if not between.empty and float(between["High"].max()) > zone_high:
                found_stale = True
                continue
            base.update({
                "detected":        True,
                "signal_source":   "accum",
                "direction":       "bearish",
                "wick_level":      round(zone_high, 5),
                "swing_type":      "high",
                "swing_ts":        ts_end,
                "accum_ts_start":  ts_start,
                "swing_idx":       e_idx,
                "wick_ratio":      zone_range_atr(zone_high, zone_low),
                "zone_high":       round(zone_high, 5),
                "zone_low":        round(zone_low, 5),
            })
            return base

        if last_low < zone_low:
            between = history.iloc[e_idx + 1:]
            if not between.empty and float(between["Low"].min()) < zone_low:
                found_stale = True
                continue
            base.update({
                "detected":        True,
                "signal_source":   "accum",
                "direction":       "bullish",
                "wick_level":      round(zone_low, 5),
                "swing_type":      "low",
                "swing_ts":        ts_end,
                "accum_ts_start":  ts_start,
                "swing_idx":       e_idx,
                "wick_ratio":      zone_range_atr(zone_high, zone_low),
                "zone_high":       round(zone_high, 5),
                "zone_low":        round(zone_low, 5),
            })
            return base

    base["reject_reason"] = (
        "cassure déjà tradée — niveau non frais"
        if found_stale else
        "aucune cassure de mèche / accumulation"
    )
    return base


# ─── Génération du graphique ──────────────────────────────────────────────────

def generate_chart(df: pd.DataFrame, pair: str, result: dict) -> bytes:
    """
    Chandelier japonais M5, style sombre TradingView.
    - Fenêtre dynamique calée sur la date du swing/accumulation cassé
    - Ligne horizontale dorée sur le niveau cassé
    - Zone dorée semi-transparente pour les accumulations
    - Triangle doré sur la bougie/fin de zone d'origine
    - Espace de 2h (24 bougies M5) à droite
    """
    from datetime import timezone

    direction = result["direction"]
    swing_ts  = result.get("swing_ts")
    now       = datetime.now(timezone.utc)

    # Fenêtre : depuis la bougie swing / début d'accum, avec 3 bougies de contexte
    if result.get("signal_source") == "accum" and result.get("accum_ts_start") is not None:
        ref_ts = result["accum_ts_start"]
    else:
        ref_ts = swing_ts

    if ref_ts is not None:
        ref_aware = ref_ts if ref_ts.tzinfo else ref_ts.replace(tzinfo=timezone.utc)
        start_dt  = ref_aware - timedelta(minutes=15)   # 3 bougies M5 de contexte
    else:
        start_dt = now - timedelta(hours=4)

    if df.index.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=None)
    elif start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)

    chart_df = df[df.index >= start_dt].copy()
    if len(chart_df) < 24:
        chart_df = df.tail(24).copy()

    mc = mpf.make_marketcolors(
        up="#26a69a", down="#ef5350",
        edge="inherit", wick="inherit",
    )
    try:
        mpf.make_mpf_style(base_mpf_style="nightclouds")
        base_style = "nightclouds"
    except Exception:
        base_style = "default"

    style = mpf.make_mpf_style(
        marketcolors=mc,
        base_mpf_style=base_style,
        gridstyle=":",
        gridcolor="#2A2A3A",
        facecolor="#131722",
        figcolor="#131722",
        rc={
            "axes.labelcolor": "#D1D4DC",
            "xtick.color":     "#D1D4DC",
            "ytick.color":     "#D1D4DC",
            "font.size":       9,
        },
    )

    label_dir  = "BULLISH 🔼" if direction == "bullish" else "BEARISH 🔽"
    source_lbl = "Accum" if result.get("signal_source") == "accum" else "Swing"
    swing_age_m = int((now - (swing_ts.replace(tzinfo=timezone.utc) if swing_ts and not swing_ts.tzinfo else (swing_ts or now))).total_seconds() / 60) if swing_ts else 0
    age_label   = f"{swing_age_m // 60}h{swing_age_m % 60:02d}" if swing_age_m >= 60 else f"{swing_age_m}min"

    buf = io.BytesIO()
    fig, axes = mpf.plot(
        chart_df,
        type="candle",
        style=style,
        title=f"\n{pair}  ·  M5  ·  {source_lbl} Break {label_dir}  ·  {age_label}",
        figsize=(14, 7),
        returnfig=True,
        tight_layout=True,
        warn_too_much_data=3000,
        volume=False,
    )

    ax   = axes[0]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # ── Ligne horizontale : niveau cassé ──────────────────────────────────
    wick_level = result["wick_level"]
    ax.axhline(
        y=wick_level,
        color="#FFD700",
        linewidth=1.4,
        linestyle="--",
        alpha=0.9,
        zorder=3,
    )
    ax.text(xmax + 0.5, wick_level, f"  {wick_level}", color="#FFD700", fontsize=8, va="center")

    # ── Zone d'accumulation : bande semi-transparente ─────────────────────
    if result.get("signal_source") == "accum":
        ax.axhspan(
            result["zone_low"],
            result["zone_high"],
            alpha=0.12,
            color="#FFD700",
            zorder=1,
        )

    # ── Triangle sur la bougie swing / fin d'accumulation ─────────────────
    if swing_ts is not None and swing_ts in chart_df.index:
        swing_pos    = chart_df.index.get_loc(swing_ts)
        swing_candle = chart_df.loc[swing_ts]
        offset       = (ymax - ymin) * 0.015

        if result["swing_type"] == "low":
            ax.scatter(swing_pos, float(swing_candle["Low"]) - offset,
                       marker="^", color="#FFD700", s=130, zorder=5)
        else:
            ax.scatter(swing_pos, float(swing_candle["High"]) + offset,
                       marker="v", color="#FFD700", s=130, zorder=5)

    # ── Lignes verticales à chaque heure ronde ────────────────────────────
    for i, ts in enumerate(chart_df.index):
        if ts.minute == 0:
            ax.axvline(x=i, color="#4A4E6A", linewidth=0.8, linestyle="--", alpha=0.75, zorder=1)
            ax.text(i + 0.3, ymax, ts.strftime("%Hh"), color="#6B7099", fontsize=7, va="top")

    # ── Espace futur + séparateur passé/futur ─────────────────────────────
    ax.set_xlim(xmin, xmax + CHART_RIGHT_MARGIN)
    ax.axvline(x=xmax - 0.5, color="#778899", linewidth=1.0, linestyle=":", alpha=0.75, zorder=2)
    ax.text(xmax + 0.4, ymax, "  →  2h", color="#778899", fontsize=8, va="top")

    axes[0].title.set_color("#FFFFFF")
    axes[0].title.set_fontsize(13)
    fig.patch.set_facecolor("#131722")
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130, facecolor="#131722")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─── Gestion de l'état anti-doublon ──────────────────────────────────────────

def load_alert_state() -> dict:
    if ALERT_STATE_FILE.exists():
        try:
            return json.loads(ALERT_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_alert_state(state: dict) -> None:
    ALERT_STATE_FILE.write_text(json.dumps(state, indent=2))


def _alert_key(pair: str, direction: str, wick_level: float) -> str:
    return f"{pair}|{direction}|{wick_level:.5f}"


def is_on_cooldown(state: dict, pair: str, direction: str, wick_level: float) -> bool:
    key = _alert_key(pair, direction, wick_level)
    if key not in state:
        return False
    last_alert = datetime.fromisoformat(state[key])
    return datetime.utcnow() - last_alert < timedelta(hours=COOLDOWN_HOURS)


def mark_alerted(state: dict, pair: str, direction: str, wick_level: float) -> None:
    state[_alert_key(pair, direction, wick_level)] = datetime.utcnow().isoformat()


# ─── Envoi Telegram ───────────────────────────────────────────────────────────

async def send_alert(bot: Bot, pair: str, result: dict, tv_symbol: str, chart_bytes: bytes) -> None:
    direction   = result["direction"]
    arrow       = "🔼" if direction == "bullish" else "🔽"
    emoji_sweep = "🟢" if direction == "bullish" else "🔴"
    tv_url      = f"https://fr.tradingview.com/chart/?symbol={tv_symbol}"
    now_str     = _now_paris().strftime("%d/%m/%Y %H:%M")
    swing_ts    = result.get("swing_ts")
    ts_str      = swing_ts.strftime("%d/%m %Hh%M") if swing_ts is not None else "?"

    if result["signal_source"] == "swing":
        swing_label = "Swing Low" if result["swing_type"] == "low" else "Swing High"
        source_line = (
            f"📍 *Mèche cassée :* `{result['wick_level']}`\n"
            f"   {swing_label} du `{ts_str}`\n"
            f"   Mèche = `{result['wick_ratio']}×` le corps"
        )
    else:
        accum_start = result.get("accum_ts_start")
        ts_start_str = accum_start.strftime("%d/%m %Hh%M") if accum_start else "?"
        source_line = (
            f"📍 *Zone cassée :* `{result['wick_level']}`\n"
            f"   Accumulation `{ts_start_str}` → `{ts_str}`\n"
            f"   Range zone = `{result['wick_ratio']}×ATR`"
        )

    caption = (
        f"*Wick Break M5 — {pair}* {arrow}\n\n"
        f"🕐 `{now_str}`\n"
        f"{arrow} *Direction :* {direction.capitalize()}\n"
        f"💰 *Prix actuel :* `{result['price']}`\n\n"
        f"{source_line}\n\n"
        f"{emoji_sweep}"
    )

    keyboard = InlineKeyboardMarkup([[
        InlineKeyboardButton("📈 Ouvrir dans TradingView", url=tv_url),
    ]])

    await bot.send_photo(
        chat_id=TELEGRAM_CHANNEL_ID,
        photo=chart_bytes,
        caption=caption,
        parse_mode="Markdown",
        reply_markup=keyboard,
    )
    log.info(f"✅ Alerte : {pair} {direction} | {result['signal_source']} | niveau={result['wick_level']}")


# ─── Séparateur épinglé ───────────────────────────────────────────────────────

async def _send_separator(bot: Bot) -> None:
    msg = await bot.send_message(chat_id=TELEGRAM_CHANNEL_ID, text="‼️" * 15)
    try:
        await bot.unpin_all_chat_messages(chat_id=TELEGRAM_CHANNEL_ID)
    except Exception as e:
        log.warning(f"Impossible de désépingler les anciens messages : {e}")
    try:
        await bot.pin_chat_message(
            chat_id=TELEGRAM_CHANNEL_ID,
            message_id=msg.message_id,
            disable_notification=True,
        )
        log.info("📌 Séparateur épinglé")
    except Exception as e:
        log.warning(f"Impossible d'épingler le séparateur : {e}")


# ─── Scan unique ──────────────────────────────────────────────────────────────

async def scan_all(bot: Bot) -> None:
    now_str = _now_paris().strftime("%Y-%m-%d %H:%M:%S")
    log.info("=" * 60)
    log.info(f"Scan démarré — {now_str} (Paris)")

    if not _is_active_session():
        log.info(f"Hors session — scan ignoré (sessions : 09h-22h et 01h-04h)")
        log.info("=" * 60)
        return

    log.info(f"Paires surveillées : {len(FOREX_PAIRS)}")
    log.info("─" * 60)
    log.info("DÉTECTION M5 : cassure de mèche sur swing ou accumulation")
    log.info(f"  Swing lookback      : {SWING_LOOKBACK} bougie de chaque côté")
    log.info(f"  Fenêtre de recherche: {SWING_SEARCH_WINDOW} bougies M5 (12h)")
    log.info(f"  Mèche minimale      : {MIN_WICK_BODY_RATIO}× le corps")
    log.info(f"  Accum. min.         : {ACCUM_MIN_CANDLES} bougies M5 / range ≤ {ACCUM_MAX_RANGE_ATR}×ATR")
    log.info(f"  Cooldown            : {COOLDOWN_HOURS}h par paire/direction/niveau")
    log.info("=" * 60)

    state          = load_alert_state()
    total_sent     = 0
    separator_sent = False

    for pair, info in FOREX_PAIRS.items():
        try:
            df = fetch_m5_data(info["yf"])
            if df is None:
                log.info(f"  {pair:<12} | ⚠️  données indisponibles")
                continue

            result = detect_wick_break(df)

            if result["detected"]:
                direction  = result["direction"]
                wick_level = result["wick_level"]
                source     = result["signal_source"]
                swing_ts   = result.get("swing_ts")
                ts_str     = swing_ts.strftime("%d/%m %Hh%M") if swing_ts else "?"
                log.info(
                    f"  {pair:<12} | 🚨 {source.upper()} BREAK {direction.upper()} "
                    f"— niveau={wick_level} (du {ts_str})"
                )
                on_cd = is_on_cooldown(state, pair, direction, wick_level)
                if not on_cd:
                    if not separator_sent:
                        await _send_separator(bot)
                        separator_sent = True
                    chart_bytes = generate_chart(df, pair, result)
                    await send_alert(bot, pair, result, info["tv"], chart_bytes)
                    mark_alerted(state, pair, direction, wick_level)
                    save_alert_state(state)
                    total_sent += 1
                    await asyncio.sleep(1.5)
                else:
                    log.info(f"  {pair:<12} | 🔒 cooldown actif")
            else:
                log.info(f"  {pair:<12} | ⛔ {result['reject_reason']}")

        except Exception as exc:
            log.error(f"  {pair:<12} | 💥 erreur : {exc}", exc_info=True)

    log.info("-" * 60)
    log.info(f"Scan terminé — {total_sent} message(s) envoyé(s)")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

async def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN manquant (Secrets GH Actions ou .env)")
    if not TELEGRAM_CHANNEL_ID:
        raise ValueError("TELEGRAM_CHANNEL_ID manquant (Secrets GH Actions ou .env)")

    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    me  = await bot.get_me()
    log.info(f"Bot : @{me.username}  |  Channel : {TELEGRAM_CHANNEL_ID}")

    await scan_all(bot)


if __name__ == "__main__":
    asyncio.run(main())
