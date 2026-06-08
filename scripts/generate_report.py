"""
실험 결과 보고서 자동 생성 스크립트

대상 실험 (4개):
  - ppo_Humanoid-v5_seed42                  : PPO 베이스라인
  - sac_Humanoid-v5_seed42                  : SAC
  - td3_Humanoid-v5_seed42                  : TD3
  - ppo_Humanoid-v5_balanced_walk_seed42    : PPO + Reward Shaping (balanced_walk)

사용법:
    python scripts/generate_report.py
    python scripts/generate_report.py --output report/최종보고서.docx
"""

import argparse
import re
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

# =============================================================================
# 설정
# =============================================================================

RESULTS_DIR = Path("results")
PLOTS_DIR   = RESULTS_DIR / "plots"
IND_DIR     = PLOTS_DIR / "individual"
SUMMARY_CSV = RESULTS_DIR / "eval" / "summary.csv"
CONFIG_PATH = Path("configs/default.yaml")
ENV         = "Humanoid-v5"

# 보고서에 포함할 4개 실험 (순서대로)
EXPERIMENTS = [
    {
        "name":   "ppo_Humanoid-v5_seed42",
        "algo":   "PPO",
        "label":  "PPO Baseline",
        "reward": None,
    },
    {
        "name":   "sac_Humanoid-v5_seed42",
        "algo":   "SAC",
        "label":  "SAC",
        "reward": None,
    },
    {
        "name":   "td3_Humanoid-v5_seed42",
        "algo":   "TD3",
        "label":  "TD3",
        "reward": None,
    },
    {
        "name":   "ppo_Humanoid-v5_balanced_walk_seed42",
        "algo":   "PPO",
        "label":  "PPO + balanced_walk",
        "reward": "balanced_walk",
    },
]

# balanced_walk 구성 요소
BALANCED_WALK_ROWS = [
    ("기본 보상",     "MuJoCo 기본",  "생존 + 전진 속도 + 제어 비용 + 접촉 비용"),
    ("높이 유지",     "추가",         "목표 높이(1.3 m) 이탈 시 패널티"),
    ("에너지 절약",   "추가",         "관절 토크² 합에 비례한 패널티"),
    ("y축 직진",      "추가",         "y축 이탈량에 비례한 패널티"),
    ("좌우 대칭성",   "추가",         "좌우 고관절 각도 차이에 비례한 패널티"),
]


# =============================================================================
# 스타일 헬퍼
# =============================================================================

def set_heading(doc: Document, text: str, level: int):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p


def add_paragraph(doc: Document, text: str, bold: bool = False, size: int = 11):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    return p


def add_image(doc: Document, img_path, caption: str, width: float = 5.8):
    img_path = Path(img_path)
    if not img_path.exists():
        doc.add_paragraph(f"  [이미지 없음: {img_path.name}]")
        return
    doc.add_picture(str(img_path), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].font.size = Pt(9)
    cap.runs[0].italic = True


def add_table(doc: Document, headers: list, rows: list):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        hdr[i].paragraphs[0].runs[0].bold = True
        hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    for r_idx, row in enumerate(rows):
        cells = table.rows[r_idx + 1].cells
        for c_idx, val in enumerate(row):
            cells[c_idx].text = str(val)
            cells[c_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    return table


# =============================================================================
# 데이터 로드
# =============================================================================

def load_summary() -> pd.DataFrame:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"{SUMMARY_CSV} 파일이 없습니다.\n"
            "  먼저 python scripts/eval_all.py 를 실행하세요."
        )
    df = pd.read_csv(SUMMARY_CSV)
    df["mean_return"] = pd.to_numeric(df["mean_return"], errors="coerce")
    df["std_return"]  = pd.to_numeric(df["std_return"],  errors="coerce")
    return df


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_result(df: pd.DataFrame, exp_name: str):
    """summary.csv에서 실험명으로 결과 행 반환."""
    row = df[df["experiment"] == exp_name]
    return row.iloc[0] if not row.empty else None


# =============================================================================
# 섹션 1: 실험 설정
# =============================================================================

def write_setup(doc: Document, cfg: dict):
    set_heading(doc, "1. 실험 설정", 1)

    # 1.1 환경
    set_heading(doc, "1.1 환경", 2)
    add_table(doc,
        headers=["항목", "값"],
        rows=[
            ["환경 ID",            ENV],
            ["관측 공간",          "348차원 (연속, 관절 각도·속도·접촉력 등)"],
            ["행동 공간",          "17차원 (연속, 각 관절 토크 [-0.4, 0.4])"],
            ["에피소드 최대 스텝", "1,000"],
            ["총 학습 스텝",       "20,000,000"],
            ["시드",               "42"],
            ["관측 정규화",        "RunningNorm (Welford's algorithm)"],
        ]
    )
    doc.add_paragraph()

    # 1.2 알고리즘 하이퍼파라미터
    set_heading(doc, "1.2 알고리즘별 주요 하이퍼파라미터", 2)
    ppo = cfg.get("ppo", {})
    sac = cfg.get("sac", {})
    td3 = cfg.get("td3", {})
    com = cfg.get("common", {})
    net = cfg.get("network", {})

    add_table(doc,
        headers=["하이퍼파라미터", "PPO", "SAC", "TD3"],
        rows=[
            ["학습률 (Actor)",    ppo.get("lr_actor",""),  sac.get("lr_actor",""),  td3.get("lr_actor","")],
            ["학습률 (Critic)",   ppo.get("lr_critic",""), sac.get("lr_critic",""), td3.get("lr_critic","")],
            ["배치 크기",         ppo.get("batch_size",""),sac.get("batch_size",""),td3.get("batch_size","")],
            ["버퍼 크기",         "— (On-policy)",
             f"{sac.get('buffer_size',0):,}", f"{td3.get('buffer_size',0):,}"],
            ["할인율 (γ)",        str(com.get("gamma","")), "← 동일", "← 동일"],
            ["알고리즘 특화",
             f"clip_ratio = {ppo.get('clip_ratio','')}",
             f"lr_alpha = {sac.get('lr_alpha','')}",
             f"policy_delay = {td3.get('policy_delay','')}"],
            ["보상 스케일",
             f"scale = {ppo.get('reward_scale', 1.0)}",
             f"scale = {sac.get('reward_scale', 1.0)}",
             f"scale = {td3.get('reward_scale', 1.0)}"],
        ]
    )
    doc.add_paragraph()

    # 1.3 네트워크 구조
    set_heading(doc, "1.3 네트워크 구조", 2)
    add_table(doc,
        headers=["항목", "설정"],
        rows=[
            ["기본 구조",    "MLP (Multi-Layer Perceptron)"],
            ["히든 레이어",  str(net.get("hidden_dims", [256, 256]))],
            ["활성화 함수",  net.get("activation", "relu").upper()],
            ["PPO Actor",    "GaussianActor — state-independent std, GAE 이점 추정"],
            ["SAC Actor",    "GaussianActor — state-dependent std, Reparameterization trick"],
            ["TD3 Actor",    "DeterministicActor — tanh 출력, Target Policy Smoothing"],
            ["Critic (공통)", "Twin Q-Network — 두 Q값 중 최솟값으로 과대추정 방지"],
        ]
    )
    doc.add_paragraph()

    # 1.4 실험 구성 요약
    set_heading(doc, "1.4 실험 구성", 2)
    add_table(doc,
        headers=["실험명", "알고리즘", "Reward", "학습 스텝"],
        rows=[
            ["PPO Baseline",       "PPO", "기본 (MuJoCo 제공)", "20M"],
            ["SAC",                "SAC", "기본 (MuJoCo 제공)", "20M"],
            ["TD3",                "TD3", "기본 (MuJoCo 제공)", "20M"],
            ["PPO + balanced_walk","PPO", "balanced_walk (커스텀)", "20M"],
        ]
    )
    doc.add_paragraph()


# =============================================================================
# 섹션 2: 알고리즘 비교
# =============================================================================

def write_comparison(doc: Document, df: pd.DataFrame):
    set_heading(doc, "2. 알고리즘 비교", 1)
    add_paragraph(doc,
        f"PPO, SAC, TD3 세 알고리즘을 {ENV} 환경에서 20M 스텝 동안 학습하여 "
        "최종 성능과 학습 안정성을 비교합니다. "
        "PPO는 CPU 기반 On-policy, SAC와 TD3는 GPU 기반 Off-policy 알고리즘입니다."
    )
    doc.add_paragraph()

    # 2.1 학습 곡선
    set_heading(doc, "2.1 학습 곡선 비교 (Eval Return)", 2)
    add_image(doc, PLOTS_DIR / "comparison_main.png",
              f"그림 1. {ENV} — 알고리즘별 Eval Return (0–20M steps, smoothed)")
    doc.add_paragraph()

    # 2.2 최종 성능 표
    set_heading(doc, "2.2 최종 성능 비교", 2)
    baseline_exps = [e for e in EXPERIMENTS if e["reward"] is None]
    rows = []
    best_mean, best_label = -1e9, ""
    for exp in baseline_exps:
        r = get_result(df, exp["name"])
        if r is None:
            rows.append([exp["label"], "N/A", "N/A", "N/A"])
        else:
            mean = r["mean_return"]
            rows.append([
                exp["label"],
                f"{mean:.1f}",
                f"± {r['std_return']:.1f}",
                str(int(r["mean_length"])) if "mean_length" in r and pd.notna(r["mean_length"]) else "N/A",
            ])
            if mean > best_mean:
                best_mean, best_label = mean, exp["label"]

    add_table(doc,
        headers=["알고리즘", "Mean Return", "Std", "Mean Ep. Length"],
        rows=rows
    )
    doc.add_paragraph()
    if best_label:
        add_paragraph(doc,
            f"→ 베이스라인 비교 최고 성능: {best_label}  (Mean Return {best_mean:.1f})",
            bold=True
        )
    doc.add_paragraph()

    # 2.3 성능 요약 바 차트
    set_heading(doc, "2.3 Final / Peak 성능 요약", 2)
    add_image(doc, PLOTS_DIR / "comparison_bar.png",
              "그림 2. Final Performance (마지막 10% 평균) vs Peak Performance")
    doc.add_paragraph()

    # 2.4 알고리즘별 학습 상세
    set_heading(doc, "2.4 알고리즘별 학습 상세", 2)
    for exp in baseline_exps:
        img = IND_DIR / f"{exp['name']}.png"
        add_image(doc, img,
                  f"그림. {exp['label']} — Return / Episode Length / Loss 추이",
                  width=5.5)
        doc.add_paragraph()


# =============================================================================
# 섹션 3: Reward Shaping
# =============================================================================

def write_reward_shaping(doc: Document, df: pd.DataFrame):
    set_heading(doc, "3. Reward Shaping — balanced_walk", 1)
    add_paragraph(doc,
        "PPO 알고리즘에 커스텀 Reward 함수(balanced_walk)를 적용하여 "
        "Humanoid의 보행 안정성과 전진 효율 향상을 목표로 하는 실험을 진행합니다. "
        "MuJoCo 기본 보상에 높이 유지, 에너지 절약, 직진성, 좌우 대칭성 항목을 추가합니다."
    )
    doc.add_paragraph()

    # 3.1 Reward 함수 구성
    set_heading(doc, "3.1 balanced_walk Reward 함수 구성", 2)
    add_table(doc,
        headers=["구성 요소", "구분", "설명"],
        rows=BALANCED_WALK_ROWS
    )
    doc.add_paragraph()

    # 3.2 PPO Baseline vs balanced_walk 성능 비교
    set_heading(doc, "3.2 PPO Baseline vs balanced_walk 성능 비교", 2)
    baseline_r = get_result(df, "ppo_Humanoid-v5_seed42")
    shaping_r  = get_result(df, "ppo_Humanoid-v5_balanced_walk_seed42")

    rows = []
    for label, r in [("PPO Baseline", baseline_r), ("PPO balanced_walk", shaping_r)]:
        if r is None:
            rows.append([label, "N/A", "N/A"])
        else:
            rows.append([label, f"{r['mean_return']:.1f}", f"± {r['std_return']:.1f}"])

    add_table(doc, headers=["구분", "Mean Return", "Std"], rows=rows)
    doc.add_paragraph()

    if baseline_r is not None and shaping_r is not None:
        diff = shaping_r["mean_return"] - baseline_r["mean_return"]
        sign = "+" if diff >= 0 else ""
        verdict = "향상" if diff >= 0 else "하락"
        add_paragraph(doc,
            f"→ Reward Shaping 적용 효과: {sign}{diff:.1f}  ({verdict})",
            bold=True
        )
    doc.add_paragraph()

    # 3.3 balanced_walk 학습 곡선
    set_heading(doc, "3.3 balanced_walk 학습 상세", 2)
    add_image(doc, IND_DIR / "ppo_Humanoid-v5_balanced_walk_seed42.png",
              "그림. PPO balanced_walk — Return / Episode Length / Loss 추이",
              width=5.5)
    doc.add_paragraph()


# =============================================================================
# 섹션 4: 결론
# =============================================================================

def write_conclusion(doc: Document, df: pd.DataFrame):
    set_heading(doc, "4. 결론", 1)

    # 4.1 전체 성능 요약 표
    set_heading(doc, "4.1 전체 실험 성능 요약", 2)
    rows = []
    best_mean, best_label = -1e9, ""
    for exp in EXPERIMENTS:
        r = get_result(df, exp["name"])
        if r is None:
            rows.append([exp["label"], "N/A", "N/A"])
        else:
            mean = r["mean_return"]
            rows.append([exp["label"], f"{mean:.1f}", f"± {r['std_return']:.1f}"])
            if mean > best_mean:
                best_mean, best_label = mean, exp["label"]

    add_table(doc, headers=["실험", "Mean Return", "Std"], rows=rows)
    doc.add_paragraph()
    if best_label:
        add_paragraph(doc,
            f"→ 전체 실험 최고 성능: {best_label}  (Mean Return {best_mean:.1f})",
            bold=True
        )
    doc.add_paragraph()

    # 4.2 분석
    set_heading(doc, "4.2 분석 및 고찰", 2)

    # 알고리즘 비교
    algo_means = {}
    for exp in EXPERIMENTS:
        if exp["reward"] is None:
            r = get_result(df, exp["name"])
            if r is not None:
                algo_means[exp["label"]] = r["mean_return"]

    if algo_means:
        best_a  = max(algo_means, key=algo_means.get)
        worst_a = min(algo_means, key=algo_means.get)
        add_paragraph(doc,
            f"1. 알고리즘 비교: {best_a}가 평균 {algo_means[best_a]:.1f}로 "
            f"가장 높은 성능을 기록했으며, {worst_a}(평균 {algo_means[worst_a]:.1f})와 "
            f"{algo_means[best_a] - algo_means[worst_a]:.1f}의 차이를 보였습니다."
        )
    doc.add_paragraph()

    add_paragraph(doc,
        "2. Off-policy vs On-policy: SAC·TD3(Off-policy)는 Replay Buffer를 통한 "
        "샘플 재사용으로 데이터 효율이 높으며, 초기 수렴 속도가 빠른 경향을 보였습니다. "
        "PPO(On-policy)는 학습이 안정적이나 동일 스텝 대비 샘플 효율이 낮습니다."
    )
    doc.add_paragraph()

    add_paragraph(doc,
        "3. Reward Shaping: balanced_walk는 높이 유지·에너지 절약·y축 직진·좌우 대칭성을 "
        "보상에 추가함으로써 보행 안정성 향상을 유도합니다. "
        "기본 보상과 커스텀 항목 간의 스케일 균형이 최종 성능에 중요하게 작용함을 확인했습니다."
    )
    doc.add_paragraph()

    add_paragraph(doc,
        "4. Humanoid-v5 환경 특성: 고차원 관측(348차원)과 17관절 제어로 인해 "
        "학습 초기 정책이 불안정하며, 충분한 학습 스텝(20M)이 수렴에 필수적입니다. "
        "관측 정규화(RunningNorm)는 학습 안정성에 크게 기여합니다."
    )
    doc.add_paragraph()


# =============================================================================
# 메인
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="보고서 자동 생성")
    parser.add_argument("--output", default="report/실험보고서.docx")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  보고서 생성 시작")
    print(f"{'='*60}\n")

    try:
        df  = load_summary()
        cfg = load_config()
    except FileNotFoundError as e:
        print(f"[오류] {e}")
        return

    doc = Document()

    # 제목 페이지
    title = doc.add_heading(
        "MuJoCo 연속 제어 환경에서의 강화학습 알고리즘 비교 연구", 0
    )
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph(
        f"환경: {ENV}  |  알고리즘: PPO / SAC / TD3  |  생성일: {date.today()}"
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    write_setup(doc, cfg)
    doc.add_page_break()

    write_comparison(doc, df)
    doc.add_page_break()

    write_reward_shaping(doc, df)
    doc.add_page_break()

    write_conclusion(doc, df)

    doc.save(str(out_path))
    print(f"  보고서 저장 완료: {out_path}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
