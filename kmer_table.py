import argparse
import pandas as pd
import random
import collections
import multiprocessing as mp
import numpy as np
from functools import partial


def load_taxonomy(path):
    tbl = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                tbl[parts[0]] = ';'.join(e.strip() for e in parts[1].split(';'))
    return tbl


def parse_fasta(path):
    seq_id, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith('>'):
                if seq_id:
                    yield seq_id, ''.join(buf)
                seq_id = line[1:].split()[0]
                buf = []
            else:
                buf.append(line.upper())
    if seq_id:
        yield seq_id, ''.join(buf)


def kmer_counter(seq_str, k):
    n = len(seq_str)
    if n < k:
        return {}
    return dict(collections.Counter(seq_str[i:i+k] for i in range(n - k + 1)))


def _worker(args, k):
    seq_id, seq_str = args
    return seq_id, kmer_counter(seq_str, k)


def build_ml_table(fasta_path, taxonomy_path, taxa_of_interest,
                   k=650, class_size=None, ids_to_exclude=None,
                   n_workers=None, output_path=None):

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    id_to_tax = load_taxonomy(taxonomy_path)

    taxon_to_records = collections.defaultdict(list)
    for seq_id, seq_str in parse_fasta(fasta_path):
        if ids_to_exclude and seq_id in ids_to_exclude:
            continue
        tax = id_to_tax.get(seq_id)
        if tax is None or tax not in taxa_of_interest:
            continue
        taxon_to_records[tax].append((seq_id, seq_str))

    for tax, recs in taxon_to_records.items():
        print(f"  {tax}: {len(recs)} sequences")

    selected = {}
    for tax, recs in taxon_to_records.items():
        if class_size is not None:
            if class_size > len(recs):
                raise ValueError(f"class_size={class_size} > available {len(recs)} for {tax}")
            recs = random.sample(recs, class_size)
        for seq_id, seq_str in recs:
            selected[seq_id] = (seq_str, tax)

    print(f"\nTotal sequences to process: {len(selected)}")

    worker_fn = partial(_worker, k=k)
    items = [(sid, sdata[0]) for sid, sdata in selected.items()]

    with mp.Pool(n_workers) as pool:
        results = pool.map(worker_fn, items, chunksize=max(1, len(items) // n_workers))

    all_kmers = sorted({km for _, freq in results for km in freq})
    kmer_idx  = {km: i for i, km in enumerate(all_kmers)}
    n_seqs    = len(results)
    n_kmers   = len(all_kmers)

    print(f"Unique k-mers: {n_kmers}")

    mat = np.zeros((n_seqs, n_kmers), dtype=np.int32)
    seq_ids = []
    for row_i, (sid, freq) in enumerate(results):
        seq_ids.append(sid)
        for km, cnt in freq.items():
            mat[row_i, kmer_idx[km]] = cnt

    df = pd.DataFrame(mat, index=seq_ids, columns=all_kmers)
    df.index.name = 'id'
    df['class'] = [selected[sid][1] for sid in seq_ids]

    if output_path:
        df.to_csv(output_path, index=False)   # ← index=False 적용
        print(f"Saved → {output_path}")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='k-mer frequency table builder')

    parser.add_argument('--fasta',    required=True,
                        help='입력 FASTA 파일 경로')
    parser.add_argument('--taxonomy', required=True,
                        help='입력 taxonomy 파일 경로 (tab-separated: id\\ttaxonomy)')
    parser.add_argument('--taxa',     required=True, nargs='+',
                        help='분석할 taxa 목록 (공백으로 구분, 세미콜론 포함 시 따옴표로 감싸기)')
    parser.add_argument('--output',   required=True,
                        help='출력 CSV 파일 경로')
    parser.add_argument('--k',        type=int, default=650,
                        help='k-mer 길이 (기본값: 650)')
    parser.add_argument('--class_size', type=int, default=None,
                        help='taxon별 샘플링 수 (기본값: 전체 사용)')
    parser.add_argument('--workers',  type=int, default=None,
                        help='병렬 프로세스 수 (기본값: cpu_count - 1)')

    args = parser.parse_args()

    taxa_of_interest = set(args.taxa)
    print(f"Taxa of interest: {taxa_of_interest}")

    build_ml_table(
        fasta_path      = args.fasta,
        taxonomy_path   = args.taxonomy,
        taxa_of_interest= taxa_of_interest,
        k               = args.k,
        class_size      = args.class_size,
        n_workers       = args.workers,
        output_path     = args.output,
    )

