"""
실험별 개별 그래프 생성 스크립트

results/ 하위 모든 실험의 progress.csv를 읽어
각 실험마다 PNG 1장씩 저장합니다.

그래프 구성:
  - 상단: Episode Return (이동평균) + Eval Return 포인트
  - 하단: Episode Length (이동평균)
  - 알고리즘별 주요 Loss (3번째 서브플롯)

사용법:
    python scripts/plot_individual.py
    python scripts/plot_individual.py --filter sac
    python scripts/plot_individual.py --output results/plots/individual
"""

import argparse, os, sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict

# ── 한글 폰트 없는 환경 대응 ──────────────────────────────────
import matplotlib
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 색상 / 레이블 설정 ────────────────────────────────────────
ALGO_COLORS = {"ppo": "#2980B9", "sac": "#E67E22", "td3": "#27AE60"}

EXP_PATTERN = re.compile(
    r"^(?P<algo>ppo|sac|td3)_(?P<env>[A-Za-z]+-v\d+)"
    r"(?:_(?P<tag>[a-z][a-z0-9_-]+?))?_seed(?P<seed>\d+)$"
)

LOSS_COLS = {
    "ppo": [("loss/policy", "Policy Loss"),
            ("loss/value",  "Value Loss"),
            ("loss/entropy","Entropy")],
    "sac": [("loss/critic", "Critic Loss"),
            ("loss/actor",  "Actor Loss"),
            ("info/alpha",  "Alpha")],
    "td3": [("loss/critic", "Critic Loss"),
            ("loss/actor",  "Actor Loss")],
}


def smooth(arr, window=15):
    if len(arr) < 2:
        return arr
    return pd.Series(arr).rolling(window, min_periods=1).mean().values


def exp_label(info):
    """사람이 읽기 좋은 실험 레이블 생성."""
    base = f"{info['algo'].upper()} / {info['env']} / seed {info['seed']}"
    tag  = info.get("tag") or ""
    if not tag:
        return base + " [Baseline]"
    if tag.startswith("hp_"):
        return base + f" [HP: {tag[3:]}]"
    return base + f" [Reward: {tag}]"


def plot_one(csv_path, info, output_path):
    df    = pd.read_csv(csv_path)
    algo  = info["algo"]
    color = ALGO_COLORS.get(algo, "#555555")

    steps = df["step"].values / 1_000_000   # M steps

    # ── episode return / length ──────────────────────────────
    ep_ret  = df["episode/return"].values
    ep_len  = df["episode/length"].values
    ep_mask = ~np.isnan(ep_ret)

    # ── eval return ──────────────────────────────────────────
    eval_ret = eval_step = None
    if "eval/mean_return" in df.columns:
        ev_mask   = ~np.isnan(df["eval/mean_return"].values)
        eval_ret  = df["eval/mean_return"].values[ev_mask]
        eval_step = steps[ev_mask]

    # ── loss 컬럼 ────────────────────────────────────────────
    loss_defs = [(c, l) for c, l in LOSS_COLS.get(algo, []) if c in df.columns]
    n_loss = len(loss_defs)

    # ── Figure 레이아웃: Return(2x) + Length(1x) + Loss별 1행씩 ─
    height_ratios = [2.5, 1.0] + [1.2] * n_loss
    total_rows    = 2 + n_loss
    fig_h         = 5.0 + 2.8 * n_loss
    fig = plt.figure(figsize=(12, fig_h))
    fig.patch.set_facecolor("white")

    gs      = gridspec.GridSpec(total_rows, 1, figure=fig,
                                hspace=0.50, height_ratios=height_ratios)
    ax_ret  = fig.add_subplot(gs[0])
    ax_len  = fig.add_subplot(gs[1], sharex=ax_ret)
    ax_losses = [fig.add_subplot(gs[2 + i], sharex=ax_ret) for i in range(n_loss)]

    for ax in [ax_ret, ax_len] + ax_losses:
        ax.set_facecolor("white")
        ax.grid(True, color="#EEEEEE", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    SW_EP = 80
    SW_EV = 6

    # ── (1) Return: episode(배경) + eval(주선) ───────────────
    if ep_mask.any():
        ax_ret.plot(steps[ep_mask], smooth(ep_ret[ep_mask], SW_EP),
                    color=color, lw=0.9, alpha=0.30, label="Episode Return")

    if eval_ret is not None and len(eval_ret):
        sm_ev = smooth(eval_ret, SW_EV)
        ax_ret.plot(eval_step, sm_ev,
                    color=color, lw=2.2, label="Eval Return (smoothed)")
        peak_idx = sm_ev.argmax()
        ax_ret.scatter(eval_step[peak_idx], sm_ev[peak_idx],
                       color=color, s=80, zorder=6, edgecolors="white", linewidths=1.5)
        ax_ret.annotate(f"{sm_ev[peak_idx]:,.0f}",
                        xy=(eval_step[peak_idx], sm_ev[peak_idx]),
                        xytext=(8, 6), textcoords="offset points",
                        fontsize=9, fontweight="bold", color=color)

    ax_ret.set_ylabel("Return", fontsize=11)
    ax_ret.set_title(exp_label(info), fontweight="bold", pad=10, fontsize=13)
    ax_ret.legend(loc="upper left", framealpha=0.9, fontsize=9)
    ax_ret.set_ylim(bottom=0)
    ax_ret.tick_params(labelbottom=False)

    # ── (2) Episode Length ───────────────────────────────────
    if ep_mask.any():
        ax_len.plot(steps[ep_mask], smooth(ep_len[ep_mask], SW_EP),
                    color=color, lw=1.8)
    ax_len.axhline(1000, color="#AAAAAA", lw=1.0, ls=":", label="max = 1000")
    ax_len.set_ylabel("Ep. Length", fontsize=11)
    ax_len.set_ylim(0, 1100)
    ax_len.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax_len.tick_params(labelbottom=False if n_loss > 0 else True)
    if n_loss == 0:
        ax_len.set_xlabel("Training Steps (M)", fontsize=11)

    # ── (3+) Loss 메트릭별 개별 서브플롯 (스케일 독립) ──────
    loss_palette = ["#E74C3C", "#3498DB", "#2ECC71", "#9B59B6"]
    for i, (ax_l, (col, label)) in enumerate(zip(ax_losses, loss_defs)):
        raw      = df[col].dropna().values
        step_idx = df[col].dropna().index
        if len(raw) == 0:
            continue
        xs = steps[step_idx] if len(step_idx) == len(raw) \
             else np.linspace(steps[0], steps[-1], len(raw))
        ax_l.plot(xs, smooth(raw, window=20),
                  color=loss_palette[i % len(loss_palette)],
                  lw=1.8, label=label)
        ax_l.set_ylabel(label, fontsize=10)
        ax_l.legend(loc="upper right", framealpha=0.9, fontsize=9)
        # 마지막 서브플롯에만 x축 레이블
        if i == n_loss - 1:
            ax_l.set_xlabel("Training Steps (M)", fontsize=11)
        else:
            ax_l.tick_params(labelbottom=False)

    plt.savefig(output_path, bbox_inches="tight", dpi=200, facecolor="white")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/logs")
    parser.add_argument("--output", default="results/plots/individual")
    parser.add_argument("--filter", default=None, help="실험 이름 필터 (예: sac, hp, baseline)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    results = Path(args.results_dir)

    # 실험 탐색
    experiments = []
    for exp_dir in sorted(results.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name == "eval":
            continue
        if args.filter and args.filter.lower() not in exp_dir.name.lower():
            continue
        match = EXP_PATTERN.match(exp_dir.name)
        if not match:
            continue
        csv_path = exp_dir / "logs" / "progress.csv"
        if not csv_path.exists():
            print(f"  [건너뜀] {exp_dir.name} — progress.csv 없음")
            continue
        info = match.groupdict()
        info["seed"] = int(info["seed"])
        info["name"] = exp_dir.name
        experiments.append((str(csv_path), info))

    if not experiments:
        print("[!] 해당하는 실험이 없습니다.")
        return

    print(f"\n{'='*60}")
    print(f"  개별 그래프 생성 | 총 {len(experiments)}개 실험")
    print(f"  저장 위치: {args.output}/")
    print(f"{'='*60}\n")

    for i, (csv_path, info) in enumerate(experiments, 1):
        out_path = os.path.join(args.output, f"{info['name']}.png")
        try:
            plot_one(csv_path, info, out_path)
            print(f"  [{i:2d}/{len(experiments)}] {info['name']}.png")
        except Exception as e:
            print(f"  [{i:2d}/{len(experiments)}] FAIL {info['name']}: {e}")

    print(f"\n완료! 총 {len(experiments)}장 저장: {args.output}/")


if __name__ == "__main__":
    main()
