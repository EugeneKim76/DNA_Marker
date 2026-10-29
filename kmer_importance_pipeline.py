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

  CD-HIT 대표 서열 선정:
  cd-hit이 클러스터 내에서 임의로(최초 입력 순서 기준) 고르는 대표 서열 대신,
  .clstr 파일로 클러스터 멤버십을 확인하여 각 클러스터 내 shap_importance가
  가장 큰 멤버를 대표 서열로 선정한다 (run_cdhit_sort 참고).

  seq_id 안정화:
  기존에는 seq_id가 fold 결과를 이어붙인 DataFrame의 행 번호였으므로,
  top_n/n_splits 등 파라미터에 따라 동일 서열이라도 FASTA 내 등장 위치가
  달라져 seq_id 값이 달라질 수 있었다(서열·SHAP 값 자체는 변하지 않음).
  이를 명확히 하기 위해 seq_id를 k-mer 서열의 해시값으로 변경하여,
  동일 서열은 파라미터와 무관하게 항상 동일한 ID를 갖도록 하였다
  (records_to_fasta 참고).

  동점 순위 안정화:
  최종 정렬(run_cdhit_sort)이 SHAP 값만으로 정렬할 경우, 동일한 SHAP 값을
  가진 서열들 사이의 상대 순서는 Python stable sort 특성상 "정렬 전 리스트
  순서"에 의존한다. 이 정렬 전 순서는 .clstr/FASTA 내 등장 순서에서 오므로
  top_n/n_splits가 바뀌면 동점 서열 그룹 내부의 등장 순서(= TSV 상 몇
  번째 줄인지)가 달라질 수 있었다(SHAP 값 자체는 동일). 이를 막기 위해
  정렬 키에 seq_id(서열 해시)를 2차 키로 추가하여, 동점 그룹 내 순서까지
  입력 파라미터와 무관하게 고정하였다.

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
import hashlib
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

def _seq_hash_id(sequence, length=12):
    """
    k-mer 서열 자체로부터 결정적(deterministic) ID를 생성.
    동일 서열은 top_n, fold 구성, 등장 순서와 무관하게 항상 동일 ID를 가지므로
    seq_id가 "DataFrame 내 행 번호"였던 기존 방식과 달리 입력 파라미터에
    의존하지 않는 안정적인 식별자가 된다.
    """
    return hashlib.md5(sequence.encode("utf-8")).hexdigest()[:length]


def records_to_fasta(records, fa_path):
    """
    fold별 top-N DataFrame을 합쳐 FASTA로 저장.

    seq_id는 행 번호(iterrows 인덱스)가 아니라 k-mer 서열의 해시값을 사용한다.
    기존 방식(행 번호)은 top_n/n_splits 등 입력 파라미터에 따라 동일 서열이라도
    FASTA 내 등장 위치(= 행 번호)가 달라져, 로그나 TSV 상에서 같은 서열이
    다른 ID로 보이는 혼란을 유발할 수 있었다. 서열 해시는 이 문제를 없앤다.
    """
    combined = pd.concat(records, ignore_index=True)
    with open(fa_path, "w") as f:
        for _, row in combined.iterrows():
            seq = row["column_name"]
            seq_id = _seq_hash_id(seq)
            f.write(">{}\t{:.6f}\n{}\n".format(
                seq_id, row["shap_importance"], seq))
    log("  FASTA 저장: {} ({} 항목)".format(fa_path, len(combined)))
    return combined


# ──────────────────────────────────────────────
# 4. CD-HIT 클러스터링 + 정렬
# ──────────────────────────────────────────────

def _parse_fasta_with_shap(path):
    """
    comment(2번째 필드)에 shap_importance가 들어있는 FASTA를 파싱.
    반환: {seq_id: (shap_importance, sequence)}
    """
    records = {}
    seq_id = comment = seq = None

    def _flush():
        if seq_id is None:
            return
        try:
            shap_val = float(comment.strip())
        except ValueError:
            shap_val = 0.0
        records[seq_id] = (shap_val, seq)

    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                _flush()
                parts = line[1:].split("\t", 1)
                seq_id  = parts[0]
                comment = parts[1] if len(parts) > 1 else ""
                seq = ""
            else:
                seq = seq + line if seq is not None else line
    _flush()
    return records


def _parse_cdhit_clstr(clstr_path):
    """
    CD-HIT .clstr 파일 파싱.
    반환: [[member_seq_id, ...], ...]  (클러스터별 멤버 seq_id 리스트)
    """
    clusters = []
    current = []
    with open(clstr_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">Cluster"):
                if current:
                    clusters.append(current)
                current = []
            else:
                # 형식 예: "0\t300nt, >0...\t*"  또는  "1\t300nt, >60... at 98.00%"
                start = line.find(">")
                end = line.find("...", start)
                if start != -1 and end != -1:
                    member_id = line[start + 1:end]
                    current.append(member_id)
    if current:
        clusters.append(current)
    return clusters


def run_cdhit_sort(fa_path, out_tsv, cdhit_c=0.70):
    """
    CD-HIT 클러스터링 → 클러스터별 SHAP 최대값 멤버를 대표 서열로 선별
    → SHAP score 내림차순 정렬 → TSV 저장.

    기존 방식은 cd-hit이 클러스터 내에서 임의로(=최초 입력 순서 기준) 고른
    대표 서열을 그대로 사용했으나, 이 경우 fold 구성이나 top_n 등 입력 순서에
    영향을 주는 요인에 따라 대표 서열이 달라질 수 있다. 이를 보완하기 위해
    .clstr 파일로 클러스터 멤버십을 확인하고, 각 클러스터 내에서
    shap_importance가 가장 큰 멤버를 대표 서열로 선정한다.
    """
    cdhit_out = fa_path.replace(".fa", "_cdhit.fa")
    clstr_path = cdhit_out + ".clstr"

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

    # ── 클러스터링 입력(fa_path)에 포함된 전체 후보의 shap_importance 맵 ──
    all_records = _parse_fasta_with_shap(fa_path)

    # ── .clstr 파싱: 클러스터별 멤버 seq_id 목록 ──
    clusters = _parse_cdhit_clstr(clstr_path)
    log("  CD-HIT 클러스터 수: {}".format(len(clusters)))

    # ── 클러스터별 SHAP 최대값 멤버를 대표 서열로 선정 ──
    parsed = []
    for members in clusters:
        best_id, best_shap, best_seq = None, None, None
        for mid in members:
            if mid not in all_records:
                continue
            shap_val, sequence = all_records[mid]
            if best_shap is None or shap_val > best_shap:
                best_id, best_shap, best_seq = mid, shap_val, sequence
        if best_id is None:
            continue
        parsed.append((best_id, best_shap, best_seq))

    # SHAP score 내림차순 정렬
    # 2차 키로 seq_id(서열 해시)를 사용해, 동점(SHAP 값이 같은) 서열들 간의
    # 상대 순서를 top_n/n_splits 등 입력 파라미터와 무관하게 고정한다.
    # (Python sort는 stable sort이므로, 2차 키 없이는 동점 그룹 내 순서가
    #  입력 리스트 순서 즉 .clstr/FASTA 등장 순서에 의존해 파라미터에 따라
    #  달라질 수 있었다.)
    parsed.sort(key=lambda x: (-x[1], x[0]))

    # ── TSV 저장: ID / SHAP score / k-mer 서열 ──
    with open(out_tsv, "w") as f:
        f.write("seq_id\tshap_importance\tkmer_sequence\n")
        for sid, shap_val, sequence in parsed:
            f.write("{}\t{:.6f}\t{}\n".format(sid, shap_val, sequence))

    log("  정렬 완료: {} ({} 클러스터 대표 서열, 클러스터 내 SHAP 최대값 기준 선별)".format(
        out_tsv, len(parsed)))


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
