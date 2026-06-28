"""
kmer_importance_pipeline.py
============================
k-mer 빈도 테이블(CSV) 입력 → SHAP importance 산출 → CD-HIT 클러스터링 →
marker_candidates_cdhit_sort.tsv 자동 생성

배경:
  이진 분류에서 shap_values[0] = -shap_values[1] 이므로
  타겟 종 마커 발굴에는 pos_label 방향(shap_values[pos_label_idx])만 사용.
  기존 코드의 importance_df / importance_df2 구분은 같은 정보의 중복이었으므로
  본 코드에서는 타겟 종 방향 단일 결과만 출력.

사용 예:
    python kmer_importance_pipeline.py \
        --input      ML_table2.csv \
        --pos_label  "Anadara;broughtonii" \
        --n_splits   30 \
        --top_n      30 \
        --cdhit_c    0.70 \
        --workers    -1 \
        --outdir     ./results
"""

import argparse
import sys
import subprocess
import time
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import shap

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────
# 0. 유틸리티
# ──────────────────────────────────────────────

def log(msg):
    print("[{}] {}".format(time.strftime('%H:%M:%S'), msg), flush=True)


def check_cdhit():
    if subprocess.run(["which", "cd-hit"], capture_output=True).returncode != 0:
        sys.exit("[ERROR] 'cd-hit' 가 PATH에 없습니다. 설치 후 재실행하세요.")


# ──────────────────────────────────────────────
# 1. 데이터 로드
# ──────────────────────────────────────────────

def load_table(csv_path):
    log("CSV 로드: {}".format(csv_path))
    df = pd.read_csv(csv_path)
    if "class" not in df.columns:
        sys.exit("[ERROR] CSV에 'class' 컬럼이 없습니다.")
    X = df.drop(columns=["class"])
    if "id" in X.columns:
        X = X.drop(columns=["id"])
    y = df["class"]
    log("  샘플 수: {}, 피처 수: {}".format(len(df), X.shape[1]))
    log("  클래스 분포:\n{}".format(y.value_counts().to_string()))
    return X, y


# ──────────────────────────────────────────────
# 2. SHAP importance 반복 산출
# ──────────────────────────────────────────────

def run_shap_cv(X, y, pos_label,
                n_splits=30, top_n=30,
                n_estimators=150, max_depth=100,
                n_jobs=-1):
    """
    StratifiedKFold 반복 → fold별 타겟 종 방향 SHAP importance 상위 top_n 반환.

    속도 개선 포인트
    ----------------
    1. RandomForestClassifier(n_jobs=n_jobs)      : 트리 병렬 학습
    2. feature_perturbation='tree_path_dependent' : 대용량 피처에서 수배 빠름
    3. check_additivity=False                     : 내부 검증 단계 생략
    4. float32 변환                               : 메모리 절반, 캐시 효율 향상

    이진 분류 SHAP 구조:
    - shap_values[0] = Others 방향 기여도
    - shap_values[1] = pos_label 방향 기여도
    - shap_values[0] = -shap_values[1]  (부호만 반대)
    → 타겟 종 마커 발굴 목적에는 pos_label 방향 단일 결과만 사용
    """
    skf = StratifiedKFold(n_splits=n_splits)
    cols = X.columns.tolist()

    records = []          # fold별 top-N DataFrame
    pos_label_idx = None  # shap_values 배열에서 pos_label의 인덱스

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        log("  Fold {}/{} ...".format(fold_i, n_splits))

        X_train = X.iloc[train_idx]
        X_test  = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test  = y.iloc[test_idx]

        # ── 모델 학습 ──
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=320,
            n_jobs=n_jobs,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # pos_label 인덱스 확인 (첫 fold에서 1회만)
        if pos_label_idx is None:
            classes_order = model.classes_.tolist()
            if pos_label in classes_order:
                pos_label_idx = classes_order.index(pos_label)
                log("  클래스 순서: {}".format(classes_order))
                log("  pos_label '{}' → shap_values[{}]".format(
                    pos_label, pos_label_idx))
            else:
                log("[WARNING] pos_label '{}'이 클래스에 없음: {}".format(
                    pos_label, classes_order))
                pos_label_idx = 1  # fallback

        print(classification_report(y_test, pred))
        print(confusion_matrix(y_test, pred))

        # ── SHAP 연산 ──
        X_train_arr = X_train.values.astype(np.float32)
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
        )
        shap_values = explainer.shap_values(X_train_arr, check_additivity=False)

        # ── 타겟 종(pos_label) 방향 SHAP importance 상위 top_n ──
        sv = pd.Series(shap_values[pos_label_idx].sum(axis=0), index=cols)
        top = sv.nlargest(top_n).reset_index()
        top.columns = ["column_name", "shap_importance"]
        records.append(top)

    return records


# ──────────────────────────────────────────────
# 3. TSV / FASTA 생성
# ──────────────────────────────────────────────

def records_to_fasta(records, fa_path):
    """fold별 top-N DataFrame을 합쳐 FASTA로 저장."""
    combined = pd.concat(records, ignore_index=True)
    with open(fa_path, "w") as f:
        for idx, row in combined.iterrows():
            f.write(">{}\t{:.6f}\n{}\n".format(
                idx, row["shap_importance"], row["column_name"]))
    log("  FASTA 저장: {} ({} 항목)".format(fa_path, len(combined)))
    return combined


# ──────────────────────────────────────────────
# 4. CD-HIT 클러스터링 + 정렬
# ──────────────────────────────────────────────

def run_cdhit_sort(fa_path, out_tsv, cdhit_c=0.70):
    """CD-HIT 클러스터링 → SHAP score 내림차순 정렬 → TSV 저장."""
    cdhit_out = fa_path.replace(".fa", "_cdhit.fa")

    cmd = [
        "cd-hit",
        "-i", fa_path,
        "-c", str(cdhit_c),
        "-o", cdhit_out,
        "-T", "0",   # 전체 스레드
        "-M", "0",   # 메모리 제한 없음
        "-d", "0",   # 긴 헤더 전체 출력
    ]
    log("CD-HIT 실행: {}".format(" ".join(cmd)))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit("[ERROR] CD-HIT 실패:\n{}".format(result.stderr))

    # ── Python으로 FASTA 파싱 (bioawk 불필요) ──
    rows = []
    seq_id = comment = seq = None
    with open(cdhit_out) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if seq_id is not None:
                    rows.append((seq_id, comment, seq))
                parts = line[1:].split("\t", 1)
                seq_id  = parts[0]
                comment = parts[1] if len(parts) > 1 else ""
                seq = ""
            else:
                seq = seq + line if seq is not None else line
    if seq_id is not None:
        rows.append((seq_id, comment, seq))

    # SHAP score 파싱 및 내림차순 정렬
    parsed = []
    for sid, cmt, sequence in rows:
        try:
            shap_val = float(cmt.strip())
        except ValueError:
            shap_val = 0.0
        parsed.append((sid, shap_val, sequence))
    parsed.sort(key=lambda x: x[1], reverse=True)

    # ── TSV 저장: ID / SHAP score / k-mer 서열 ──
    with open(out_tsv, "w") as f:
        f.write("seq_id\tshap_importance\tkmer_sequence\n")
        for sid, shap_val, sequence in parsed:
            f.write("{}\t{:.6f}\t{}\n".format(sid, shap_val, sequence))

    log("  정렬 완료: {} ({} 대표 서열)".format(out_tsv, len(parsed)))


# ──────────────────────────────────────────────
# 5. 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="k-mer frequency CSV -> SHAP importance -> CD-HIT -> sorted TSV"
    )
    parser.add_argument("--input",        required=True,
                        help="k-mer 빈도 테이블 CSV 경로")
    parser.add_argument("--pos_label",    required=True,
                        help="타겟 클래스 레이블 (예: 'Anadara;broughtonii')")
    parser.add_argument("--n_splits",     type=int, default=30,
                        help="StratifiedKFold 분할 수 (기본값: 30)")
    parser.add_argument("--top_n",        type=int, default=30,
                        help="fold별 상위 k-mer 수 (기본값: 30)")
    parser.add_argument("--n_estimators", type=int, default=150,
                        help="RandomForest 트리 수 (기본값: 150)")
    parser.add_argument("--max_depth",    type=int, default=100,
                        help="RandomForest 최대 깊이 (기본값: 100)")
    parser.add_argument("--cdhit_c",      type=float, default=0.70,
                        help="CD-HIT 유사도 임계값 (기본값: 0.70)")
    parser.add_argument("--workers",      type=int, default=-1,
                        help="병렬 프로세스 수 (-1=전체 코어, 기본값: -1)")
    parser.add_argument("--outdir",       default=".",
                        help="출력 디렉터리 (기본값: 현재 디렉터리)")

    args = parser.parse_args()

    check_cdhit()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로드
    X, y = load_table(args.input)

    # 2. SHAP CV
    log("SHAP importance 반복 교차 검증 시작 ...")
    t0 = time.time()
    records = run_shap_cv(
        X, y,
        pos_label    = args.pos_label,
        n_splits     = args.n_splits,
        top_n        = args.top_n,
        n_estimators = args.n_estimators,
        max_depth    = args.max_depth,
        n_jobs       = args.workers,
    )
    log("SHAP CV 완료 ({:.1f}초)".format(time.time() - t0))

    # 3. FASTA 변환
    fa_path = str(outdir / "marker_candidates.fa")
    log("FASTA 변환 ...")
    records_to_fasta(records, fa_path)

    # 4. CD-HIT + 정렬
    out_tsv = str(outdir / "marker_candidates_cdhit_sort.tsv")
    log("CD-HIT 클러스터링 및 정렬 ...")
    run_cdhit_sort(fa_path, out_tsv, cdhit_c=args.cdhit_c)

    log("=" * 50)
    log("완료! 최종 출력 파일:")
    log("  {}".format(out_tsv))


if __name__ == "__main__":
    main()
