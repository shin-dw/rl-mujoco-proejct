"""
메인 비교 그래프 생성 스크립트

ppo_runforward (메인) vs ppo / sac / td3 최고 실험 비교
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── 전역 스타일 ──────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":      "DejaVu Sans",
    "axes.unicode_minus": False,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.color":       "#E5E5E5",
    "grid.linewidth":   0.8,
    "xtick.direction":  "out",
    "ytick.direction":  "out",
})

# ── 실험 정의 ────────────────────────────────────────────────
EXPS = [
    dict(
        name   = "ppo_Humanoid-v5_run_forward_seed42",
        label  = "ppo_runforward",
        color  = "#E74C3C",
        lw     = 3.0,
        ls     = "-",
        zorder = 6,
        marker = "o",
    ),
    dict(
        name   = "ppo_Humanoid-v5_seed42",
        label  = "ppo",
        color  = "#2980B9",
        lw     = 2.2,
        ls     = "--",          # 긴 대시
        zorder = 4,
        marker = "s",
    ),
    dict(
        name   = "sac_Humanoid-v5_seed42",
        label  = "sac",
        color  = "#E67E22",
        lw     = 2.2,
        ls     = (0, (3, 1, 1, 1)),   # 대시-점 패턴
        zorder = 4,
        marker = "^",
    ),
    dict(
        name   = "td3_Humanoid-v5_seed42",
        label  = "td3",
        color  = "#27AE60",
        lw     = 2.2,
        ls     = (0, (1, 1)),   # 촘촘한 점선
        zorder = 4,
        marker = "D",
    ),
]

RESULTS_DIR = "results"
OUTPUT_DIR  = "results/plots"
SW          = 30          # smoothing window


def smooth(arr, w=SW):
    return pd.Series(arr).rolling(w, min_periods=1).mean().values


def load_df(name):
    return pd.read_csv(f"{RESULTS_DIR}/{name}/logs/progress.csv")


# ════════════════════════════════════════════════════════════════
#  Figure 1 — comparison_main.png
#  레이아웃: 단일 패널 (Episode Return, 0–20M)
#  - Eval return 평균선을 주 곡선으로 사용 (scatter 제거)
#  - Episode return 은 고도로 스무딩된 배경 참고선으로 표시
#  - 선 스타일로 4가지 알고리즘 명확히 구분
# ════════════════════════════════════════════════════════════════
SW_EVAL = 8   # eval 스무딩 윈도우 (8 × 50k = 400k 스텝)


def make_main(data):
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for exp in EXPS:
        df    = data[exp["name"]]
        steps = df["step"].values / 1_000_000

        # ── 주 곡선: eval/mean_return 스무딩 (scatter 없음) ──────
        ev_mask  = df["eval/mean_return"].notna().values
        ev_vals  = df["eval/mean_return"].values[ev_mask]
        ev_steps = steps[ev_mask]
        if len(ev_vals):
            sm_ev = smooth(ev_vals, SW_EVAL)
            ax.plot(ev_steps, sm_ev,
                    color=exp["color"], lw=exp["lw"], ls=exp["ls"],
                    label=exp["label"], zorder=exp["zorder"])

    # ── Peak 마커 (ppo_runforward) ────────────────────────────
    # 그래프에 그려진 스무딩 곡선 위의 최댓값을 가리키도록 sm_ev 기준으로 탐색
    df_rf    = data["ppo_Humanoid-v5_run_forward_seed42"]
    ev_mask  = df_rf["eval/mean_return"].notna().values
    ev_vals  = df_rf["eval/mean_return"].values[ev_mask]
    ev_steps = df_rf["step"].values[ev_mask] / 1_000_000
    if len(ev_vals):
        sm_rf     = smooth(ev_vals, SW_EVAL)   # 실제 그려진 곡선과 동일
        peak_idx  = sm_rf.argmax()             # 스무딩 곡선 기준 최댓값
        peak_step = ev_steps[peak_idx]
        peak_val  = sm_rf[peak_idx]            # 곡선 위의 y값
        ax.annotate(
            f"Peak  {peak_val:,.0f}",
            xy=(peak_step, peak_val),
            xytext=(peak_step + 1.5, peak_val * 0.87),
            fontsize=10, fontweight="bold", color="#C0392B",
            arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.5,
                            connectionstyle="arc3,rad=0.15"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec="#E74C3C", alpha=0.9, lw=1.2),
            zorder=15,
        )

    # ── 5M 구간 세로 가이드선 ─────────────────────────────────
    ax.set_xlim(0, 20.5)
    ax.set_ylim(bottom=0)
    for x in [5, 10, 15, 20]:
        ax.axvline(x, color="#DDDDDD", lw=0.9, ls=":", zorder=0)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xticklabels(["0", "5M", "10M", "15M", "20M"], fontsize=11)

    # ── 축 레이블 & 제목 ──────────────────────────────────────
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Eval Return (smoothed)", fontsize=12)
    ax.set_title(
        "Humanoid-v5  —  Eval Return Comparison  (0 – 20M steps)",
        fontsize=14, fontweight="bold", pad=12,
    )

    # ── 범례 ──────────────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0],
               color=e["color"], lw=e["lw"], ls=e["ls"],
               label=e["label"])
        for e in EXPS
    ]
    ax.legend(handles=legend_handles, fontsize=12,
              frameon=True, framealpha=0.95, edgecolor="#DDDDDD",
              loc="upper left")

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "comparison_main.png")
    fig.savefig(out, bbox_inches="tight", dpi=220, facecolor="white")
    plt.close(fig)
    print("saved: " + out)


# ════════════════════════════════════════════════════════════════
#  Figure 2 — comparison_bar.png
#  Final (last-10% avg) vs Peak 바 차트
# ════════════════════════════════════════════════════════════════
def make_bar(data):
    labels, finals, peaks, colors = [], [], [], []
    for exp in EXPS:
        df = data[exp["name"]]
        ev = df["eval/mean_return"].dropna()
        tail = ev.tail(max(1, len(ev) // 10)).mean()
        labels.append(exp["label"])
        finals.append(tail)
        peaks.append(ev.max())
        colors.append(exp["color"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5),
                                   facecolor="white")
    fig.subplots_adjust(wspace=0.35)

    x = np.arange(len(labels))

    for ax, vals, title, ylabel in [
        (ax1, finals,
         "Final Performance\n(last 10% eval avg)",
         "Eval Return"),
        (ax2, peaks,
         "Peak Performance\n(best eval score)",
         "Eval Return"),
    ]:
        bars = ax.bar(x, vals,
                      color=colors, width=0.55,
                      alpha=0.88, edgecolor="white", linewidth=1.8,
                      zorder=3)

        # 값 라벨
        max_v = max(vals)
        for bar, v, c in zip(bars, vals, colors):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_v * 0.012,
                f"{v:,.0f}",
                ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=c,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.set_ylim(0, max_v * 1.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#E5E5E5", lw=0.8, zorder=0)

        # ppo_runforward 강조 테두리
        bars[0].set_edgecolor("#C0392B")
        bars[0].set_linewidth(2.5)

    fig.suptitle(
        "Performance Summary  —  Humanoid-v5",
        fontsize=14, fontweight="bold", y=1.02,
    )

    out = os.path.join(OUTPUT_DIR, "comparison_bar.png")
    fig.savefig(out, bbox_inches="tight", dpi=220, facecolor="white")
    plt.close(fig)
    print("saved: " + out)


# ════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = {e["name"]: load_df(e["name"]) for e in EXPS}
    make_main(data)
    make_bar(data)


if __name__ == "__main__":
    main()
