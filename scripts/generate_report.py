"""
실험 결과 보고서 자동 생성 스크립트

summary.csv와 plots/ 폴더의 그래프를 읽어 DOCX 보고서를 생성합니다.

사용법:
    python scripts/generate_report.py
    python scripts/generate_report.py --output report/최종보고서.docx
"""

import argparse
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd
import yaml
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# =============================================================================
# 설정
# =============================================================================

RESULTS_DIR  = Path("results")
PLOTS_DIR    = RESULTS_DIR / "plots"
SUMMARY_CSV  = RESULTS_DIR / "eval" / "summary.csv"
CONFIG_PATH  = Path("configs/default.yaml")
ENV          = "Humanoid-v5"
ALGOS        = ["PPO", "SAC", "TD3"]
REWARD_TYPES = ["balanced_walk", "stable_gait"]

# HP 튜닝 변형 레이블 (가독성용)
HP_LABELS = {
    "clip01":   "PPO clip_ratio = 0.1",
    "clip03":   "PPO clip_ratio = 0.3",
    "lr1e-4":   "PPO lr_actor = 1e-4",
    "lr1e-3":   "PPO lr_actor = 1e-3",
    "lra1e-4":  "SAC lr_alpha = 1e-4",
    "lra1e-3":  "SAC lr_alpha = 1e-3",
    "bs128":    "SAC batch_size = 128",
    "bs512":    "SAC batch_size = 512",
    "delay1":   "TD3 policy_delay = 1",
    "delay4":   "TD3 policy_delay = 4",
    "noise005": "TD3 exploration_noise = 0.05",
    "noise02":  "TD3 exploration_noise = 0.2",
}

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


def add_image(doc: Document, img_path: Path, caption: str, width: float = 5.5):
    if not img_path.exists():
        doc.add_paragraph(f"  [그래프 없음: {img_path.name}]")
        return
    doc.add_picture(str(img_path), width=Inches(width))
    last = doc.paragraphs[-1]
    last.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].font.size = Pt(9)
    cap.runs[0].italic = True


def add_table(doc: Document, headers: list, rows: list):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 헤더
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        hdr[i].paragraphs[0].runs[0].bold = True
        hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

    # 데이터
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


# =============================================================================
# 섹션 1: 실험 설정
# =============================================================================

def write_setup(doc: Document, cfg: dict):
    set_heading(doc, "1. 실험 설정", 1)

    set_heading(doc, "1.1 환경", 2)
    add_table(doc,
        headers=["항목", "값"],
        rows=[
            ["환경 ID",        ENV],
            ["관측 공간",      "376차원 (연속)"],
            ["행동 공간",      "17차원 (연속, 각 관절 토크)"],
            ["에피소드 최대 스텝", "1,000"],
            ["총 학습 스텝",   f"{3_000_000:,}"],
            ["평가 시드",      "42, 77, 123"],
        ]
    )
    doc.add_paragraph()

    set_heading(doc, "1.2 알고리즘별 주요 하이퍼파라미터", 2)
    ppo = cfg.get("ppo", {})
    sac = cfg.get("sac", {})
    td3 = cfg.get("td3", {})
    net = cfg.get("network", {})

    add_table(doc,
        headers=["하이퍼파라미터", "PPO", "SAC", "TD3"],
        rows=[
            ["학습률 (Actor)",    ppo.get("lr_actor",""),  sac.get("lr_actor",""),  td3.get("lr_actor","")],
            ["배치 크기",         ppo.get("batch_size",""), sac.get("batch_size",""), td3.get("batch_size","")],
            ["버퍼 크기",         "-",                     f"{sac.get('buffer_size',0):,}", f"{td3.get('buffer_size',0):,}"],
            ["할인율 (γ)",        cfg.get("common",{}).get("gamma",""), "←동일", "←동일"],
            ["알고리즘 특화",     f"clip_ratio={ppo.get('clip_ratio','')}", f"lr_alpha={sac.get('lr_alpha','')}", f"policy_delay={td3.get('policy_delay','')}"],
        ]
    )
    doc.add_paragraph()

    set_heading(doc, "1.3 네트워크 구조", 2)
    add_table(doc,
        headers=["항목", "설정"],
        rows=[
            ["구조",       "MLP (Multi-Layer Perceptron)"],
            ["히든 레이어", f"{net.get('hidden_dims', [256,256])}"],
            ["활성화 함수", net.get("activation", "relu").upper()],
            ["PPO Actor",  "GaussianActor (state-independent std)"],
            ["SAC Actor",  "GaussianActor (state-dependent std + reparameterization)"],
            ["TD3 Actor",  "DeterministicActor (tanh 출력)"],
            ["Critic",     "Twin Q-Network (과대추정 방지)"],
        ]
    )
    doc.add_paragraph()


# =============================================================================
# 섹션 2: 베이스라인 비교
# =============================================================================

def write_baseline(doc: Document, df: pd.DataFrame):
    set_heading(doc, "2. 베이스라인 비교", 1)
    add_paragraph(doc,
        "PPO, SAC, TD3 알고리즘을 기본 하이퍼파라미터로 각각 3개 시드(42, 77, 123)에 "
        f"대해 학습하여 {ENV} 환경에서의 기본 성능을 비교합니다."
    )
    doc.add_paragraph()

    # 학습 곡선
    set_heading(doc, "2.1 학습 곡선", 2)
    add_image(doc, PLOTS_DIR / f"curve_{ENV}.png",
              f"그림 1. {ENV} 알고리즘별 학습 곡선 (다중 시드 평균 ± 표준편차)")
    doc.add_paragraph()

    # 최종 성능 표
    set_heading(doc, "2.2 최종 성능 비교", 2)
    baseline_mask = df["experiment"].str.match(
        rf"^(ppo|sac|td3)_{re.escape(ENV)}_seed\d+$"
    )
    baseline = df[baseline_mask]

    rows = []
    best_mean = -float("inf")
    best_algo = ""
    for algo in ALGOS:
        sub = baseline[baseline["algorithm"] == algo]
        if sub.empty:
            rows.append([algo, "N/A", "N/A", "N/A", "N/A"])
            continue
        mean = sub["mean_return"].mean()
        std  = sub["mean_return"].std()
        mn   = sub["mean_return"].min()
        mx   = sub["mean_return"].max()
        rows.append([algo, f"{mean:.1f}", f"{std:.1f}", f"{mn:.1f}", f"{mx:.1f}"])
        if mean > best_mean:
            best_mean = mean
            best_algo = algo

    add_table(doc,
        headers=["알고리즘", "평균 Return", "± Std", "최솟값", "최댓값"],
        rows=rows
    )
    doc.add_paragraph()
    if best_algo:
        add_paragraph(doc,
            f"→ 베이스라인 기준 가장 높은 성능: {best_algo} (평균 {best_mean:.1f})",
            bold=True
        )
    doc.add_paragraph()


# =============================================================================
# 섹션 3: Reward Shaping
# =============================================================================

def write_reward_shaping(doc: Document, df: pd.DataFrame):
    set_heading(doc, "3. Reward Shaping 실험", 1)
    add_paragraph(doc,
        "기본 Reward에 추가 항목을 더한 두 가지 커스텀 Reward 함수를 적용하여 "
        "학습 효율과 최종 성능의 변화를 분석합니다."
    )
    doc.add_paragraph()

    add_table(doc,
        headers=["Reward 타입", "추가 항목"],
        rows=[
            ["default",       "기본 (survive + forward + ctrl + contact)"],
            ["balanced_walk", "기본 + 높이 유지 + 에너지 절약 + y축 직진 + 좌우 대칭"],
            ["stable_gait",   "기본 + 행동 크기 페널티 + 연속 행동 변화량 페널티"],
        ]
    )
    doc.add_paragraph()

    # 알고리즘별 비교 그래프 + 표
    rows_summary = []
    for algo in ALGOS:
        set_heading(doc, f"3.{ALGOS.index(algo)+1} {algo}", 2)

        img = PLOTS_DIR / f"reward_{ENV}_{algo}.png"
        add_image(doc, img, f"그림. {algo} Reward Shaping 학습 곡선 비교")
        doc.add_paragraph()

        # 베이스라인 평균
        base_sub = df[df["experiment"].str.match(
            rf"^{algo.lower()}_{re.escape(ENV)}_seed\d+$"
        )]
        base_mean = base_sub["mean_return"].mean() if not base_sub.empty else float("nan")

        for rt in REWARD_TYPES:
            sub = df[df["experiment"] == f"{algo.lower()}_{ENV}_{rt}_seed42"]
            if sub.empty:
                rows_summary.append([algo, rt, "N/A", "N/A"])
                continue
            mean = sub["mean_return"].iloc[0]
            diff = mean - base_mean
            sign = "+" if diff >= 0 else ""
            rows_summary.append([algo, rt, f"{mean:.1f}", f"{sign}{diff:.1f}"])

    doc.add_paragraph()
    set_heading(doc, "3.4 Reward Shaping 성능 요약", 2)
    add_table(doc,
        headers=["알고리즘", "Reward 타입", "Mean Return", "vs 베이스라인"],
        rows=rows_summary
    )
    doc.add_paragraph()


# =============================================================================
# 섹션 4: HP 튜닝
# =============================================================================

def write_hp_tuning(doc: Document, df: pd.DataFrame):
    set_heading(doc, "4. Hyperparameter 튜닝", 1)
    add_paragraph(doc,
        "알고리즘별 주요 하이퍼파라미터를 변경하여 베이스라인 대비 성능 변화를 분석합니다. "
        "각 실험은 시드 42로 단일 실행하였습니다."
    )
    doc.add_paragraph()

    for algo in ALGOS:
        set_heading(doc, f"4.{ALGOS.index(algo)+1} {algo} HP 튜닝", 2)

        img = PLOTS_DIR / f"hp_{ENV}_{algo}.png"
        add_image(doc, img, f"그림. {algo} HP 튜닝 학습 곡선 비교")
        doc.add_paragraph()

    # 전체 요약 표
    set_heading(doc, "4.4 HP 튜닝 성능 요약", 2)
    hp_mask = df["experiment"].str.contains(f"_hp_", na=False)
    hp_df = df[hp_mask]

    rows = []
    for _, row in hp_df.iterrows():
        exp = row["experiment"]
        algo = row["algorithm"]
        m = re.search(r"_hp_([a-z0-9e.\-]+)_seed", exp)
        tag = m.group(1) if m else exp
        label = HP_LABELS.get(tag, tag)

        base_sub = df[df["experiment"].str.match(
            rf"^{algo.lower()}_{re.escape(ENV)}_seed\d+$"
        )]
        base_mean = base_sub["mean_return"].mean() if not base_sub.empty else float("nan")
        diff = row["mean_return"] - base_mean
        sign = "+" if diff >= 0 else ""
        rows.append([algo, label, f"{row['mean_return']:.1f}", f"{sign}{diff:.1f}"])

    if rows:
        add_table(doc,
            headers=["알고리즘", "HP 변경", "Mean Return", "vs 베이스라인"],
            rows=rows
        )
    else:
        add_paragraph(doc, "  HP 튜닝 실험 결과가 없습니다.")
    doc.add_paragraph()


# =============================================================================
# 섹션 5: 최적 조합
# =============================================================================

def write_best_combination(doc: Document, df: pd.DataFrame):
    set_heading(doc, "5. 최적 조합 결과", 1)

    best_mask = df["experiment"].str.contains("_best_seed42", na=False)
    best_df = df[best_mask]

    if best_df.empty:
        add_paragraph(doc, "  최적 조합 실험 결과가 없습니다. Phase 4 완료 후 다시 생성하세요.")
        return

    best_row = best_df.loc[best_df["mean_return"].idxmax()]
    exp_name = best_row["experiment"]

    # 실험명에서 설정 파싱
    algo_match   = re.match(r"^(ppo|sac|td3)", exp_name)
    reward_match = re.search(r"(balanced_walk|stable_gait)", exp_name)
    hp_match     = re.search(r"_hp_([a-z0-9e.\-]+)_best", exp_name)

    algo   = algo_match.group(1).upper()   if algo_match   else "N/A"
    reward = reward_match.group(1)          if reward_match else "기본 (개선 없음)"
    hp_tag = hp_match.group(1)             if hp_match     else "기본 (개선 없음)"
    hp_label = HP_LABELS.get(hp_tag, hp_tag)

    add_table(doc,
        headers=["항목", "최적 설정"],
        rows=[
            ["알고리즘",     algo],
            ["Reward Shaping", reward],
            ["HP 변경",     hp_label],
        ]
    )
    doc.add_paragraph()

    # 베이스라인 vs 최적 비교 표
    set_heading(doc, "5.1 베이스라인 vs 최적 조합", 2)
    base_sub = df[df["experiment"].str.match(
        rf"^{algo.lower()}_{re.escape(ENV)}_seed\d+$"
    )]
    base_mean = base_sub["mean_return"].mean() if not base_sub.empty else float("nan")
    best_mean = best_row["mean_return"]
    improve   = ((best_mean - base_mean) / abs(base_mean) * 100) if base_mean else float("nan")

    add_table(doc,
        headers=["구분", "Mean Return", "개선율"],
        rows=[
            ["베이스라인",  f"{base_mean:.1f}", "-"],
            ["최적 조합",   f"{best_mean:.1f}", f"{improve:+.1f}%"],
        ]
    )
    doc.add_paragraph()


# =============================================================================
# 섹션 6: 결론
# =============================================================================

def write_conclusion(doc: Document, df: pd.DataFrame):
    set_heading(doc, "6. 결론", 1)

    baseline_mask = df["experiment"].str.match(
        rf"^(ppo|sac|td3)_{re.escape(ENV)}_seed\d+$"
    )
    baseline = df[baseline_mask]
    if not baseline.empty:
        algo_means = baseline.groupby("algorithm")["mean_return"].mean()
        best_algo = algo_means.idxmax()
        worst_algo = algo_means.idxmin()
        best_val  = algo_means[best_algo]
        worst_val = algo_means[worst_algo]
        add_paragraph(doc,
            f"1. 알고리즘 비교: {ENV} 환경에서 {best_algo}가 평균 {best_val:.1f}로 "
            f"가장 높은 성능을 보였으며, {worst_algo}(평균 {worst_val:.1f})와 "
            f"비교하여 {(best_val - worst_val):.1f}의 성능 차이를 나타냈습니다."
        )

    doc.add_paragraph()
    add_paragraph(doc,
        "2. Reward Shaping: 커스텀 리워드 적용 시 y축 이탈 억제 및 관절 대칭성 "
        "보너스가 보행 안정성에 영향을 미쳤으나, 환경 기본 보상과의 균형이 중요함을 확인했습니다."
    )
    doc.add_paragraph()
    add_paragraph(doc,
        "3. HP 튜닝: 학습률 및 클리핑 범위와 같은 주요 하이퍼파라미터는 수렴 속도와 "
        "최종 성능에 민감하게 작용하며, 환경 특성에 맞는 세밀한 조정이 필요합니다."
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

    # 제목
    title = doc.add_heading("MuJoCo 연속 제어 환경에서의 강화학습 알고리즘 비교 연구", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(f"환경: {ENV}  |  생성일: {date.today()}")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # 섹션 작성
    write_setup(doc, cfg)
    doc.add_page_break()

    write_baseline(doc, df)
    doc.add_page_break()

    write_reward_shaping(doc, df)
    doc.add_page_break()

    write_hp_tuning(doc, df)
    doc.add_page_break()

    write_best_combination(doc, df)
    doc.add_page_break()

    write_conclusion(doc, df)

    doc.save(str(out_path))
    print(f"  보고서 저장 완료: {out_path}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
