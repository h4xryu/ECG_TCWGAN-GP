import os
import numpy as np
from collections import Counter, defaultdict
import wfdb
from scipy.interpolate import interp1d

# ---------------------------
# Config
# ---------------------------
PATH_DB = '../Dataset/mit-bih-arrhythmia-database-1.0.0'  # WFDB 폴더
SAVE_DIR = './dataset/mitbih_interpatient'                # 저장 폴더
VALID_LEADS = ['MLII']                                    # 사용 리드(우선순위)
OUT_LEN = 1800                                             # 최종 출력 길이(샘플 수) = 200

# DS1 = Train, DS2 = Test
DS1_TRAIN = [
    101,106,108,109,112,114,115,116,118,119,122,124,
    201,203,205,207,208,209,215,220,223,230
]
DS2_TEST = [
    100,103,105,111,113,117,121,123,200,202,210,212,
    213,214,219,221,222,228,231,232,233,234
]

# 라벨 매핑 (MIT-BIH 심볼 -> 그룹)
# Q 계열은 전부 제외
LABEL_GROUP_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N',
    'V': 'V', 'E': 'V',
    'A': 'S', 'a': 'S', 'S': 'S', 'J': 'S', 'j': 'S', 'e': 'S',
    'F': 'F',
    '/': 'Q', 'Q': 'Q', 'f': 'Q'
}

# --- 고정 순서 ---
CLASSES = ['N', 'S', 'V', 'F']                 # 순서 보장
LABEL_TO_ID = {c: i for i, c in enumerate(CLASSES)}  # {'N':0, 'S':1, 'V':2, 'F':3}
TARGET_CLASSES = set(CLASSES)                  # 포함 여부 체크용

# ---------------------------
# Utils
# ---------------------------
def resample_to_len(ts: np.ndarray, out_len: int) -> np.ndarray:
    """1D 시퀀스를 길이 out_len으로 선형보간 리샘플."""
    if len(ts) == out_len:
        return ts.astype(np.float32, copy=True)
    # 최소 길이 보호
    if len(ts) < 2:
        return np.pad(ts.astype(np.float32), (0, max(0, out_len - len(ts))), mode='edge')[:out_len]
    x_old = np.linspace(0.0, 1.0, num=len(ts), endpoint=True, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=out_len, endpoint=True, dtype=np.float64)
    f = interp1d(x_old, ts, kind='linear')
    return f(x_new).astype(np.float32)

def read_record(record_name: str, base_path: str):
    """WFDB로 신호/어노테이션 읽기 (atr)"""
    ann = wfdb.rdann(os.path.join(base_path, record_name), 'atr')
    sig, meta = wfdb.rdsamp(os.path.join(base_path, record_name))
    return ann.__dict__, sig, meta

def find_channel(meta, valid_leads):
    """meta['sig_name']에서 사용할 리드 채널 인덱스 찾기"""
    sig_names = meta['sig_name']
    for lead in valid_leads:
        if lead in sig_names:
            return sig_names.index(lead), lead, sig_names
    return None, None, sig_names

def summarize_split(name, labels_str):
    """N,S,V,F 순서로 분포 출력"""
    cnt = Counter(labels_str)
    total = sum(cnt.values())
    print(f'\n=== {name} split ===')
    print(f'Total segments: {total}')
    for k in CLASSES:  # 고정 순서
        v = cnt.get(k, 0)
        pct = (v / total * 100.0) if total > 0 else 0.0
        print(f'  {k}: {v} ({pct:.2f}%)')

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

# ---------------------------
# Main building function
# ---------------------------
def build_interpatient_splits():
    safe_mkdir(SAVE_DIR)

    # DS1/DS2를 문자열 레코드명으로 변환 (폴더 내 RECORDS와 일치)
    ds1_records = {f'{r:03d}' for r in DS1_TRAIN}
    ds2_records = {f'{r:03d}' for r in DS2_TEST}

    # RECORDS 파일로 실제 존재 목록 확인
    with open(os.path.join(PATH_DB, 'RECORDS'), 'r') as fin:
        all_recs = set(fin.read().strip().splitlines())

    # 교집합만 사용
    ds1_use = sorted(ds1_records & all_recs)
    ds2_use = sorted(ds2_records & all_recs)

    # 컨테이너
    train_data, train_labels_str, train_labels_id, train_pid = [], [], [], []
    test_data,  test_labels_str,  test_labels_id,  test_pid  = [], [], [], []

    # 수집 카운트(디버그용)
    miss_lead = defaultdict(list)
    skipped_q = 0
    skipped_out = 0
    total_ann = 0

    # -----------------------
    # 수집 함수
    # -----------------------
    def collect_from_records(rec_list, split_name):
        nonlocal skipped_q, skipped_out, total_ann
        out_data, out_labels_str, out_labels_id, out_pid = [], [], [], []
        for rec in rec_list:
            try:
                ann, sig, meta = read_record(rec, PATH_DB)
            except Exception as e:
                print(f'[WARN] failed to read {rec}: {e}')
                continue

            fs = int(meta['fs'])
            ch_idx, used_lead, sig_names = find_channel(meta, VALID_LEADS)
            if ch_idx is None:
                miss_lead[rec] = sig_names
                print(f'[INFO] {rec}: no valid lead among {sig_names}')
                continue

            x = sig[:, ch_idx].astype(np.float32)
            idx_list = list(ann['sample'])
            label_list = list(ann['symbol'])

            # --- 논문 설정: R-피크 기준 전 90, 후 110 (@360Hz) ---
            # 레코드 fs가 360이 아닐 수도 있으므로 비례 환산
            pre = int(round(900  * fs / 360.0))
            post= int(round(900 * fs / 360.0))
            win_len = pre + post  # 대략 200과 유사(정확히는 fs에 따라 ±1)

            for s_idx, sym in enumerate(label_list):
                total_ann += 1
                grp = LABEL_GROUP_MAP.get(sym, None)
                if (grp is None) or (grp == 'Q') or (grp not in TARGET_CLASSES):
                    if grp == 'Q':
                        skipped_q += 1
                    continue

                center = idx_list[s_idx]  # R-피크에 정렬된 비트 어노테이션
                start = center - pre
                end   = center + post
                if start < 0 or end > len(x):
                    skipped_out += 1
                    continue

                seg = x[start:end]               # 길이 win_len(≈200 샘플 @fs)

                 # Normalize
                mean = seg.mean()
                std = seg.std()
                seg = (seg - mean) / (std + 1e-14)

                seg_rs = resample_to_len(seg, OUT_LEN)  # 최종 200샘플로 정규화

                out_data.append(seg_rs)
                out_labels_str.append(grp)
                out_labels_id.append(LABEL_TO_ID[grp])
                out_pid.append(rec)

            print(f'[{split_name}] {rec} | lead={used_lead} | fs={fs} | collected_total={len(out_data)}')

        return out_data, out_labels_str, out_labels_id, out_pid

    # -----------------------
    # Collect
    # -----------------------
    tr_d, tr_y_str, tr_y_id, tr_pid = collect_from_records(ds1_use, 'TRAIN')
    te_d, te_y_str, te_y_id, te_pid = collect_from_records(ds2_use, 'TEST')

    # -----------------------
    # to numpy & save
    # -----------------------
    train_data = np.asarray(tr_d, dtype=np.float32)      # [N_tr, OUT_LEN=200]
    test_data  = np.asarray(te_d, dtype=np.float32)      # [N_te, OUT_LEN=200]
    train_labels_str = np.asarray(tr_y_str)              # [N_tr] -> 'N','S','V','F'
    test_labels_str  = np.asarray(te_y_str)              # [N_te]
    train_labels_id  = np.asarray(tr_y_id, dtype=np.int64)  # [N_tr] -> 0,1,2,3 (N,S,V,F)
    test_labels_id   = np.asarray(te_y_id, dtype=np.int64)  # [N_te]
    train_pid = np.asarray(tr_pid)
    test_pid  = np.asarray(te_pid)

    # 저장(.npy + .npz)
    np.save(os.path.join(SAVE_DIR, 'train_data.npy'),       train_data)
    np.save(os.path.join(SAVE_DIR, 'train_labels_str.npy'), train_labels_str)
    np.save(os.path.join(SAVE_DIR, 'train_labels_id.npy'),  train_labels_id)
    np.save(os.path.join(SAVE_DIR, 'train_pid.npy'),        train_pid)

    np.save(os.path.join(SAVE_DIR, 'test_data.npy'),        test_data)
    np.save(os.path.join(SAVE_DIR, 'test_labels_str.npy'),  test_labels_str)
    np.save(os.path.join(SAVE_DIR, 'test_labels_id.npy'),   test_labels_id)
    np.save(os.path.join(SAVE_DIR, 'test_pid.npy'),         test_pid)

    np.savez_compressed(
        os.path.join(SAVE_DIR, 'train_set.npz'),
        data=train_data,
        labels_str=train_labels_str,
        labels_id=train_labels_id,
        pid=train_pid,
        out_len=np.array(OUT_LEN),      # 최종 길이=200
        classes=np.array(CLASSES)       # ['N','S','V','F']
    )
    np.savez_compressed(
        os.path.join(SAVE_DIR, 'test_set.npz'),
        data=test_data,
        labels_str=test_labels_str,
        labels_id=test_labels_id,
        pid=test_pid,
        out_len=np.array(OUT_LEN),
        classes=np.array(CLASSES)
    )

    # -----------------------
    # Summary
    # -----------------------
    print('\n---------------- SUMMARY ----------------')
    print(f'Total annotations scanned : {total_ann}')
    print(f'Skipped (Q-class)         : {skipped_q}')
    print(f'Skipped (out-of-range)    : {skipped_out}')
    print(f'\nSaved to: {os.path.abspath(SAVE_DIR)}')
    print(f'Train data shape: {train_data.shape}  | Test data shape: {test_data.shape}')

    # 분포 출력 (N,S,V,F 순)
    summarize_split('TRAIN', train_labels_str)
    summarize_split('TEST',  test_labels_str)

    # (선택) 정수 라벨 분포도 확인하려면 주석 해제
    # for name, yid in [('TRAIN', train_labels_id), ('TEST', test_labels_id)]:
    #     cnt = Counter(yid)
    #     print(f'\n{name} labels_id dist:')
    #     for i, c in enumerate(CLASSES):
    #         print(f'  {c}({i}): {cnt.get(i,0)}')

if __name__ == '__main__':
    build_interpatient_splits()
