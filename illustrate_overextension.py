"""
Illustration pédagogique des critères de détection d'une overextension.
Génère une image annotée avec tous les signaux utilisés par le bot.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec

# ─── Palette TradingView dark ─────────────────────────────────────────────────
BG        = "#131722"
BG2       = "#1E222D"
GRID      = "#2A2E39"
TEXT      = "#D1D4DC"
GREEN     = "#26a69a"
RED       = "#ef5350"
BLUE      = "#2196F3"
ORANGE    = "#FF9800"
YELLOW    = "#FFD700"
PURPLE    = "#9C27B0"
WHITE     = "#FFFFFF"

np.random.seed(42)

# ─── Données synthétiques réalistes ──────────────────────────────────────────
N = 80

# Phase 1 : consolidation calme (bougies 0-44)
# Phase 2 : impulsion haussière forte (bougies 45-65)  ← zone overextension
# Phase 3 : légère continuation (bougies 66-79)

prices = [1.0800]
for i in range(1, N):
    if i < 45:
        # Consolidation : mouvement aléatoire faible
        change = np.random.normal(0, 0.0008)
    elif i < 55:
        # Début impulsion : forte hausse directionnelle
        change = np.random.normal(0.0025, 0.0005)
    elif i < 65:
        # Continuation impulsion : hausse soutenue
        change = np.random.normal(0.0018, 0.0006)
    else:
        # Légère continuation avec volatilité réduite
        change = np.random.normal(0.0005, 0.0007)
    prices.append(prices[-1] + change)

prices = np.array(prices)

# Construire OHLC
opens, highs, lows, closes = [], [], [], []
for i in range(N):
    o = prices[i]
    c = prices[i] + np.random.normal(0, 0.0003)
    h = max(o, c) + abs(np.random.normal(0, 0.0004))
    l = min(o, c) - abs(np.random.normal(0, 0.0004))
    opens.append(o); highs.append(h); lows.append(l); closes.append(c)

opens  = np.array(opens)
highs  = np.array(highs)
lows   = np.array(lows)
closes = np.array(closes)

# ─── Indicateurs ─────────────────────────────────────────────────────────────
def ema(series, period):
    return pd.Series(series).ewm(span=period, adjust=False).mean().values

def rsi(series, period=14):
    delta = np.diff(series, prepend=series[0])
    gain  = np.where(delta > 0, delta, 0)
    loss  = np.where(delta < 0, -delta, 0)
    avg_g = pd.Series(gain).ewm(com=period-1, min_periods=period).mean().values
    avg_l = pd.Series(loss).ewm(com=period-1, min_periods=period).mean().values
    rs    = np.where(avg_l == 0, 100, avg_g / avg_l)
    return 100 - (100 / (1 + rs))

def atr(highs, lows, closes, period=14):
    tr = np.maximum(highs - lows,
         np.maximum(np.abs(highs - np.roll(closes, 1)),
                    np.abs(lows  - np.roll(closes, 1))))
    tr[0] = highs[0] - lows[0]
    return pd.Series(tr).ewm(com=period-1, min_periods=period).mean().values

ema20  = ema(closes, 20)
ema50  = ema(closes, 50)
rsi14  = rsi(closes, 14)
atr14  = atr(highs, lows, closes, 14)

# ─── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13), facecolor=BG)
gs  = GridSpec(3, 1, figure=fig,
               height_ratios=[5, 1.5, 1],
               hspace=0.06,
               top=0.93, bottom=0.06, left=0.06, right=0.97)

ax_price  = fig.add_subplot(gs[0])
ax_rsi    = fig.add_subplot(gs[1], sharex=ax_price)
ax_legend = fig.add_subplot(gs[2])

for ax in [ax_price, ax_rsi, ax_legend]:
    ax.set_facecolor(BG2)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.spines[:].set_color(GRID)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)

x = np.arange(N)

# ═══════════════════════════════════════════════════════════════════════════════
# PANNEAU 1 — Prix + chandeliers
# ═══════════════════════════════════════════════════════════════════════════════

# Zone de fond : consolidation vs impulsion
ax_price.axvspan(0, 44.5,  alpha=0.07, color=GRID,  zorder=0)
ax_price.axvspan(44.5, 79, alpha=0.10, color=GREEN, zorder=0)

# Chandeliers
W = 0.4
for i in x:
    color = GREEN if closes[i] >= opens[i] else RED
    # Corps
    ax_price.bar(i, abs(closes[i]-opens[i]),
                 bottom=min(opens[i], closes[i]),
                 color=color, width=W*2, zorder=3)
    # Mèches
    ax_price.plot([i, i], [lows[i], highs[i]],
                  color=color, linewidth=0.8, zorder=3)

# EMA
ax_price.plot(x, ema20, color=BLUE,   linewidth=1.6, label="EMA 20", zorder=4)
ax_price.plot(x, ema50, color=ORANGE, linewidth=1.6, label="EMA 50", zorder=4)

# ── ATR band autour de l'EMA20 (zone "normale") ───────────────────────────────
ax_price.fill_between(x,
    ema20 - 1.5 * atr14,
    ema20 + 1.5 * atr14,
    alpha=0.08, color=BLUE, zorder=1, label="EMA20 ± 1.5×ATR")

# ── Annotations ───────────────────────────────────────────────────────────────

# 1. CONSOLIDATION label
ax_price.text(22, lows.min() + 0.0005, "① Consolidation\n(volatilité normale)",
              color=TEXT, fontsize=9.5, ha="center", va="bottom",
              bbox=dict(boxstyle="round,pad=0.3", facecolor=GRID, edgecolor=GRID, alpha=0.8))

# 2. Flèche impulsion + label
impulse_start = closes[44]
impulse_end   = closes[64]
ax_price.annotate("",
    xy=(64, impulse_end + 0.0010),
    xytext=(44, impulse_start - 0.0005),
    arrowprops=dict(arrowstyle="-|>", color=YELLOW, lw=2.0, mutation_scale=18))
ax_price.text(54, (impulse_start + impulse_end) / 2 + 0.0005,
              f"② Impulsion forte\n≥ 2× ATR",
              color=YELLOW, fontsize=9.5, ha="center", va="bottom", fontweight="bold",
              bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=YELLOW, alpha=0.9, lw=1.2))

# 3. Distance EMA ─ double flèche verticale
i_ref = 62
y_ema  = ema20[i_ref]
y_prix = closes[i_ref]
ax_price.annotate("",
    xy=(i_ref + 1.5, y_prix),
    xytext=(i_ref + 1.5, y_ema),
    arrowprops=dict(arrowstyle="<->", color=BLUE, lw=1.8, mutation_scale=14))
ax_price.text(i_ref + 3, (y_ema + y_prix) / 2,
              "③ Distance EMA20\n≥ 1.5× ATR",
              color=BLUE, fontsize=9, va="center",
              bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=BLUE, alpha=0.9, lw=1.2))

# 4. Bougies directionnelles — surligner les bougies haussières de l'impulsion
bullish_in_window = [i for i in range(46, 65) if closes[i] > opens[i]]
for i in bullish_in_window[:6]:
    ax_price.bar(i, highs[i] - lows[i], bottom=lows[i],
                 color=GREEN, alpha=0.15, width=W*2.5, zorder=2)

count_bull = sum(1 for i in range(58, 64) if closes[i] > opens[i])
ax_price.text(55, highs[55:64].max() + 0.0015,
              f"④ {count_bull}/6 bougies haussières\n(min. 4/6 requis)",
              color=GREEN, fontsize=9, ha="center",
              bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=GREEN, alpha=0.9, lw=1.2))

# 5. Retracement minimal — zone grisée sur l'impulsion
max_imp = highs[44:65].max()
min_imp = lows[44:65].min()
retrace = (max_imp - min_imp) * 0.28  # ~28% → inférieur à 35%
ax_price.annotate("",
    xy=(70, max_imp - retrace),
    xytext=(70, max_imp),
    arrowprops=dict(arrowstyle="<->", color=PURPLE, lw=1.8, mutation_scale=14))
ax_price.text(72, max_imp - retrace / 2,
              "⑤ Retracement\n< 35% du move",
              color=PURPLE, fontsize=9, va="center",
              bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, edgecolor=PURPLE, alpha=0.9, lw=1.2))

# Légende prix
ax_price.legend(loc="upper left", fontsize=9,
                facecolor=BG2, edgecolor=GRID, labelcolor=TEXT,
                framealpha=0.9, borderpad=0.6)
ax_price.set_ylabel("Prix  (ex: EUR/USD)", color=TEXT, fontsize=10)
ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.4f}"))
ax_price.grid(axis="y", color=GRID, linewidth=0.5, linestyle=":")
plt.setp(ax_price.get_xticklabels(), visible=False)

# Titre principal
fig.suptitle("Détection d'une Overextension — Critères du bot",
             color=WHITE, fontsize=15, fontweight="bold", y=0.97)

# ═══════════════════════════════════════════════════════════════════════════════
# PANNEAU 2 — RSI
# ═══════════════════════════════════════════════════════════════════════════════

ax_rsi.plot(x, rsi14, color=YELLOW, linewidth=1.5, zorder=3)
ax_rsi.axhline(72, color=RED,   linewidth=1.0, linestyle="--", alpha=0.8)
ax_rsi.axhline(28, color=GREEN, linewidth=1.0, linestyle="--", alpha=0.8)
ax_rsi.axhline(50, color=GRID,  linewidth=0.6, linestyle=":")
ax_rsi.fill_between(x, rsi14, 72, where=(rsi14 > 72), color=RED,   alpha=0.25, zorder=2)
ax_rsi.fill_between(x, rsi14, 28, where=(rsi14 < 28), color=GREEN, alpha=0.25, zorder=2)

# Label zone overbought
ax_rsi.text(1,  74, "Overbought > 72", color=RED,   fontsize=8.5, va="bottom")
ax_rsi.text(1,  22, "Oversold < 28",   color=GREEN, fontsize=8.5, va="top")

# Annotation pic RSI
rsi_peak_i = int(np.argmax(rsi14[50:70])) + 50
rsi_peak_v = rsi14[rsi_peak_i]
ax_rsi.annotate(f"  RSI = {rsi_peak_v:.0f}\n  ① Critère RSI",
    xy=(rsi_peak_i, rsi_peak_v),
    xytext=(rsi_peak_i + 4, min(rsi_peak_v + 5, 95)),
    color=RED, fontsize=8.5, fontweight="bold",
    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

ax_rsi.set_ylim(10, 100)
ax_rsi.set_ylabel("RSI (14)", color=TEXT, fontsize=9)
ax_rsi.grid(axis="y", color=GRID, linewidth=0.5, linestyle=":")
ax_rsi.set_yticks([28, 50, 72])
plt.setp(ax_rsi.get_xticklabels(), visible=False)

# ═══════════════════════════════════════════════════════════════════════════════
# PANNEAU 3 — Légende synthétique (tableau des critères)
# ═══════════════════════════════════════════════════════════════════════════════
ax_legend.axis("off")

criteres = [
    ("①", "RSI extrême",          "RSI > 72  (overbought)  ou  RSI < 28  (oversold)",                    RED),
    ("②", "Impulsion forte",      "Amplitude du move > 2 × ATR sur les 6 dernières bougies",              YELLOW),
    ("③", "Éloigné de l'EMA20",   "| Prix − EMA20 |  > 1.5 × ATR",                                       BLUE),
    ("④", "Mouvement directionnel","≥ 4 bougies sur 6 dans la même direction",                            GREEN),
    ("⑤", "Retracement minimal",  "Pullback < 35 % de l'amplitude totale du move",                        PURPLE),
]

y_pos = 0.92
for num, titre, detail, color in criteres:
    ax_legend.text(0.01, y_pos, num,    transform=ax_legend.transAxes,
                   color=color, fontsize=11, fontweight="bold", va="top")
    ax_legend.text(0.04, y_pos, titre,  transform=ax_legend.transAxes,
                   color=WHITE, fontsize=9.5, fontweight="bold", va="top")
    ax_legend.text(0.23, y_pos, detail, transform=ax_legend.transAxes,
                   color=TEXT,  fontsize=9,   va="top")
    y_pos -= 0.19

ax_legend.set_facecolor(BG)
ax_legend.text(0.5, 0.02,
    "⚠️  Les 5 critères doivent être validés simultanément  •  Cooldown 4h par paire/direction",
    transform=ax_legend.transAxes, color=ORANGE, fontsize=9,
    ha="center", va="bottom", style="italic")

# ─── Export ───────────────────────────────────────────────────────────────────
out = "/Users/billy/Downloads/AgentsIA/TradingAgent/overextension_explainer.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Image sauvegardée : {out}")
