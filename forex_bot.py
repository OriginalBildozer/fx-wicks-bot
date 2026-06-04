#!/usr/bin/env python3
"""
FX Wicks Bot — GitHub Actions edition
Détecte les cassures de mèches (wick breaks) sur les swings H1 :
  - Cassure sous la mèche basse d'un swing LOW  → signal bullish (liquidity sweep)
  - Cassure au-dessus de la mèche haute d'un swing HIGH → signal bearish
Exécuté une seule fois par run (le cron GH Actions remplace la boucle infinie).
L'état anti-doublon est persisté via le cache GitHub Actions entre les runs.
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

# ─── Paramètres de détection ──────────────────────────────────────────────────
ATR_PERIOD           = 14
SWING_LOOKBACK       = 1    # 1 bougie de chaque côté suffit pour confirmer un swing
SWING_SEARCH_WINDOW  = 168  # fenêtre de recherche en bougies H1 (7 jours)
MIN_WICK_BODY_RATIO  = 1.5  # mèche doit faire ≥ 1.5× le corps de la bougie

COOLDOWN_HOURS      = 4
CHART_RIGHT_MARGIN  = 12
ALERT_STATE_FILE    = Path("alert_state.json")


# ─── Indicateur technique ─────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift()).abs()
    lc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


# ─── Récupération des données ─────────────────────────────────────────────────

def fetch_h1_data(yf_ticker: str) -> pd.DataFrame | None:
    try:
        df = yf.download(
            yf_ticker,
            period="20d",
            interval="1h",
            progress=False,
            auto_adjust=True,
        )
        if df.empty or len(df) < 30:
            log.warning(f"Données insuffisantes pour {yf_ticker} ({len(df)} bougies)")
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception as exc:
        log.error(f"Erreur fetch {yf_ticker}: {exc}")
        return None


# ─── Détection des cassures de mèches ────────────────────────────────────────

def _find_swings(history: pd.DataFrame, atr: float) -> tuple[list, list]:
    """
    Retourne les swing highs et swing lows confirmés dans history.
    Chaque élément : (position_dans_history, wick_level, timestamp, wick_body_ratio)

    Filtre : mèche ≥ MIN_WICK_BODY_RATIO × corps.
    Pour les dojis (corps quasi-nul), on utilise 5 % de l'ATR comme corps minimal
    afin d'éviter la division par zéro et de ne pas retenir des mèches microscopiques.
    """
    n = len(history)
    swing_highs = []
    swing_lows  = []

    start = max(SWING_LOOKBACK, n - SWING_SEARCH_WINDOW - SWING_LOOKBACK)
    end   = n - SWING_LOOKBACK

    for i in range(start, end):
        candle = history.iloc[i]
        left   = history.iloc[max(0, i - SWING_LOOKBACK):i]
        right  = history.iloc[i + 1:i + SWING_LOOKBACK + 1]

        if len(left) < SWING_LOOKBACK or len(right) < SWING_LOOKBACK:
            continue

        c_high  = float(candle["High"])
        c_low   = float(candle["Low"])
        c_open  = float(candle["Open"])
        c_close = float(candle["Close"])
        body    = max(abs(c_open - c_close), atr * 0.05)  # plancher anti-doji

        # Swing HIGH — mèche haute ≥ MIN_WICK_BODY_RATIO × corps
        if c_high > left["High"].max() and c_high > right["High"].max():
            body_top  = max(c_open, c_close)
            wick_size = c_high - body_top
            ratio     = wick_size / body
            if ratio >= MIN_WICK_BODY_RATIO:
                swing_highs.append((i, c_high, history.index[i], round(ratio, 2)))

        # Swing LOW — mèche basse ≥ MIN_WICK_BODY_RATIO × corps
        if c_low < left["Low"].min() and c_low < right["Low"].min():
            body_bot  = min(c_open, c_close)
            wick_size = body_bot - c_low
            ratio     = wick_size / body
            if ratio >= MIN_WICK_BODY_RATIO:
                swing_lows.append((i, c_low, history.index[i], round(ratio, 2)))

    return swing_highs, swing_lows


def detect_wick_break(df: pd.DataFrame) -> dict:
    """
    Vérifie si la dernière bougie casse la mèche d'un swing high/low récent.

    - Cassure sous swing LOW wick  → bullish (liquidity sweep, attendre reversal haussier)
    - Cassure au-dessus swing HIGH wick → bearish (liquidity sweep, attendre reversal baissier)

    Cherche en priorité le swing le plus récent cassé.
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

    history = df.iloc[:-1]  # exclure la dernière bougie (candidate à la cassure)
    swing_highs, swing_lows = _find_swings(history, atr)

    if not swing_highs and not swing_lows:
        base["reject_reason"] = "aucun swing valide trouvé"
        return base

    last_high = float(last["High"])
    last_low  = float(last["Low"])

    found_stale = False  # un break existe mais le niveau a déjà été tradé

    # Chercher le swing high le plus récent dont la mèche est cassée
    for idx, wick_level, ts, wick_atr in reversed(swing_highs):
        if last_high > wick_level:
            # Vérifier que le niveau n'a jamais été atteint depuis le swing
            between = history.iloc[idx + 1:]
            if not between.empty and float(between["High"].max()) > wick_level:
                found_stale = True
                continue  # niveau déjà tradé → pas frais, on passe
            base.update({
                "detected":    True,
                "direction":   "bearish",
                "wick_level":  round(wick_level, 5),
                "swing_type":  "high",
                "swing_ts":    ts,
                "swing_idx":   idx,
                "wick_ratio":  wick_atr,
            })
            return base

    # Chercher le swing low le plus récent dont la mèche est cassée
    for idx, wick_level, ts, wick_atr in reversed(swing_lows):
        if last_low < wick_level:
            # Vérifier que le niveau n'a jamais été atteint depuis le swing
            between = history.iloc[idx + 1:]
            if not between.empty and float(between["Low"].min()) < wick_level:
                found_stale = True
                continue  # niveau déjà tradé → pas frais, on passe
            base.update({
                "detected":    True,
                "direction":   "bullish",
                "wick_level":  round(wick_level, 5),
                "swing_type":  "low",
                "swing_ts":    ts,
                "swing_idx":   idx,
                "wick_ratio":  wick_atr,
            })
            return base

    base["reject_reason"] = (
        "cassure déjà tradée — niveau non frais"
        if found_stale else
        "aucune cassure de mèche"
    )
    return base


# ─── Génération du graphique ──────────────────────────────────────────────────

def generate_chart(df: pd.DataFrame, pair: str, result: dict) -> bytes:
    """
    Chandelier japonais H1, style sombre TradingView.
    - Fenêtre : minuit J-2 → maintenant + 12h vides à droite
    - Ligne horizontale dorée sur la mèche cassée
    - Marqueur vertical au swing d'origine
    - Ligne pointillée à chaque minuit + séparateur passé/futur
    """
    from datetime import timezone

    direction = result["direction"]
    swing_ts  = result.get("swing_ts")
    now       = datetime.now(timezone.utc)

    # Fenêtre dynamique : depuis 3 bougies avant le swing jusqu'à maintenant
    if swing_ts is not None:
        # Harmoniser la timezone pour la comparaison
        swing_aware = swing_ts
        if hasattr(swing_ts, "tzinfo") and swing_ts.tzinfo is None:
            swing_aware = swing_ts.replace(tzinfo=timezone.utc)
        start_dt = swing_aware - timedelta(hours=3)
    else:
        start_dt = now - timedelta(days=2)

    if df.index.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=None)
    elif start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)

    chart_df = df[df.index >= start_dt].copy()
    # Minimum 24 bougies pour la lisibilité
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
            "font.size":       10,
        },
    )

    label_dir  = "BULLISH 🔼" if direction == "bullish" else "BEARISH 🔽"
    swing_age_h = int((now - (swing_ts.replace(tzinfo=timezone.utc) if swing_ts and swing_ts.tzinfo is None else (swing_ts or now))).total_seconds() / 3600) if swing_ts else 0
    age_label  = f"{swing_age_h // 24}j" if swing_age_h >= 24 else f"{swing_age_h}h"

    buf = io.BytesIO()
    fig, axes = mpf.plot(
        chart_df,
        type="candle",
        style=style,
        title=f"\n{pair}  ·  H1  ·  Wick Break {label_dir}  ·  swing {age_label}",
        figsize=(14, 7),
        returnfig=True,
        tight_layout=True,
        warn_too_much_data=300,
        volume=False,
    )

    ax   = axes[0]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # ── Ligne horizontale : niveau de la mèche cassée ─────────────────────
    wick_level = result["wick_level"]
    ax.axhline(
        y=wick_level,
        color="#FFD700",
        linewidth=1.4,
        linestyle="--",
        alpha=0.9,
        zorder=3,
    )
    ax.text(
        xmax + 0.5, wick_level,
        f"  {wick_level}",
        color="#FFD700",
        fontsize=8,
        va="center",
    )

    # ── Triangle sur la bougie du swing d'origine ────────────────────────
    swing_ts = result.get("swing_ts")
    if swing_ts is not None and swing_ts in chart_df.index:
        swing_pos    = chart_df.index.get_loc(swing_ts)
        swing_candle = chart_df.loc[swing_ts]
        offset       = (ymax - ymin) * 0.015   # 1.5 % de la plage visible

        if result["swing_type"] == "low":
            # Triangle pointant vers le haut, sous la mèche basse
            ax.scatter(
                swing_pos,
                float(swing_candle["Low"]) - offset,
                marker="^",
                color="#FFD700",
                s=130,
                zorder=5,
            )
        else:
            # Triangle pointant vers le bas, au-dessus de la mèche haute
            ax.scatter(
                swing_pos,
                float(swing_candle["High"]) + offset,
                marker="v",
                color="#FFD700",
                s=130,
                zorder=5,
            )

    # ── Lignes verticales à chaque minuit ─────────────────────────────────
    for i, ts in enumerate(chart_df.index):
        if ts.hour == 0 and ts.minute == 0:
            ax.axvline(
                x=i,
                color="#4A4E6A",
                linewidth=0.9,
                linestyle="--",
                alpha=0.85,
                zorder=1,
            )
            ax.text(
                i + 0.3, ymax,
                ts.strftime("%d %b"),
                color="#6B7099",
                fontsize=7.5,
                va="top",
            )

    # ── Espace vide 12h + séparateur passé/futur ──────────────────────────
    ax.set_xlim(xmin, xmax + CHART_RIGHT_MARGIN)
    ax.axvline(
        x=xmax - 0.5,
        color="#778899",
        linewidth=1.1,
        linestyle=":",
        alpha=0.75,
        zorder=2,
    )
    ax.text(
        xmax + 0.4, ymax,
        "  →  12h",
        color="#778899",
        fontsize=8,
        va="top",
    )

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
    """Clé unique = paire + direction + niveau de mèche arrondi au pip."""
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

async def send_alert(
    bot: Bot,
    pair: str,
    result: dict,
    tv_symbol: str,
    chart_bytes: bytes,
) -> None:
    direction    = result["direction"]
    arrow        = "🔼" if direction == "bullish" else "🔽"
    emoji_sweep  = "🟢" if direction == "bullish" else "🔴"
    swing_label  = "Swing Low" if result["swing_type"] == "low" else "Swing High"
    tv_url       = f"https://fr.tradingview.com/chart/?symbol={tv_symbol}"
    now_str      = _now_paris().strftime("%d/%m/%Y %H:%M")
    swing_ts_str = result["swing_ts"].strftime("%d/%m %Hh") if result.get("swing_ts") is not None else "?"

    caption = (
        f"*Wick Break — {pair}* {arrow}\n\n"
        f"🕐 `{now_str}`\n"
        f"{arrow} *Direction :* {direction.capitalize()}\n"
        f"💰 *Prix actuel :* `{result['price']}`\n\n"
        f"📍 *Mèche cassée :* `{result['wick_level']}`\n"
        f"   {swing_label} du `{swing_ts_str}`\n"
        f"   Mèche = `{result['wick_ratio']}×` le corps"
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
    log.info(
        f"✅ Alerte envoyée : {pair} {direction} "
        f"| Wick={result['wick_level']} | Break={result['break_atr']}×ATR"
    )


# ─── Scan unique ──────────────────────────────────────────────────────────────

async def scan_all(bot: Bot) -> None:
    log.info("=" * 60)
    log.info(f"Scan démarré — {_now_paris().strftime('%Y-%m-%d %H:%M:%S')} (Paris)")
    log.info(f"Paires surveillées : {len(FOREX_PAIRS)}")
    log.info("─" * 60)
    log.info("DÉTECTION : cassure de mèche sur swing H1")
    log.info(f"  Swing lookback      : {SWING_LOOKBACK} bougie de chaque côté")
    log.info(f"  Fenêtre de recherche: {SWING_SEARCH_WINDOW} bougies H1 ({SWING_SEARCH_WINDOW // 24}j)")
    log.info(f"  Mèche minimale      : {MIN_WICK_BODY_RATIO}× le corps de la bougie")
    log.info(f"  Cooldown         : {COOLDOWN_HOURS}h par paire/direction/niveau")
    log.info("=" * 60)

    state      = load_alert_state()
    total_sent = 0

    for pair, info in FOREX_PAIRS.items():
        try:
            df = fetch_h1_data(info["yf"])
            if df is None:
                log.info(f"  {pair:<12} | ⚠️  données indisponibles")
                continue

            result = detect_wick_break(df)

            if result["detected"]:
                direction  = result["direction"]
                wick_level = result["wick_level"]
                swing_ts   = result.get("swing_ts")
                ts_str     = swing_ts.strftime("%d/%m %Hh") if swing_ts is not None else "?"
                log.info(
                    f"  {pair:<12} | 🚨 WICK BREAK {direction.upper()} "
                    f"— mèche={wick_level} ({result['swing_type'].upper()} du {ts_str}) "
                    f"mèche={result['wick_ratio']}×corps"
                )
                on_cd = is_on_cooldown(state, pair, direction, wick_level)
                if not on_cd:
                    chart_bytes = generate_chart(df, pair, result)
                    await send_alert(bot, pair, result, info["tv"], chart_bytes)
                    mark_alerted(state, pair, direction, wick_level)
                    save_alert_state(state)
                    total_sent += 1
                    await asyncio.sleep(1.5)
                else:
                    log.info(f"  {pair:<12} | 🔒 cooldown actif — rien envoyé")
            else:
                log.info(f"  {pair:<12} | ⛔ {result['reject_reason']}")

        except Exception as exc:
            log.error(f"  {pair:<12} | 💥 erreur inattendue : {exc}", exc_info=True)

    if total_sent > 0:
        msg = await bot.send_message(
            chat_id=TELEGRAM_CHANNEL_ID,
            text="‼️" * 15,
        )
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
