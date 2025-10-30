import os
import numpy as np
from collections import Counter, defaultdict
import torch
import glob

from scipy.signal import medfilt
from scipy.signal import firwin, filtfilt
from preprocessing import bandpassfilter, remove_baseline_median, augment_three
from utils import *

# ---------------------------
# Config
# ---------------------------
PATH_DB = '../Dataset/mit-bih-arrhythmia-database-1.0.0'
SAVE_DIR = './dataset/mitbih_interpatient'
VALID_LEADS = ['MLII']
OUT_LEN = 200

# 오버샘플링/증강 설정
AUGMENT_TRAIN = True
USE_GAN_AUGMENT = True  # ★ GAN 증강 사용 여부

# ★★★ 클래스별 GAN 증강 설정 ★★★
# enabled: 해당 클래스 증강 ON/OFF
# checkpoint: Generator 체크포인트 경로
# multiplier: 증강 배수 (원본 포함)
GAN_AUGMENT_CONFIG = {
    'S': {
        'enabled': True,
        'checkpoint': 'log/model_S/models/S_epoch_50.pth',
        'multiplier': 4  # S 클스 4배 증강래
    },
    'V': {
        'enabled': True,  # V 클래스 증강 ON/OFF
        'checkpoint': 'log/model_V/models/V_epoch_50.pth',
        'multiplier': 2 
    },
    'F': {
        'enabled': False,  # F 클래스 증강 ON/OFF
        'checkpoint': 'log/model_F/models/F_epoch_50.pth',
        'multiplier': 4  
    }
}

# DS1 = Train, DS2 = Test
DS1_TRAIN = [
    101,106,108,109,112,114,115,116,118,119,122,124,
    201,203,205,207,208,209,215,220,223,230
]
DS2_TEST = [
    100,103,105,111,113,117,121,123,200,202,210,212,
    213,214,219,221,222,228,231,232,233,234
]

# 라벨 매핑
LABEL_GROUP_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',
    'V': 'V', 'E': 'V',
    'A': 'S', 'a': 'S', 'S': 'S', 'J': 'S',
    'F': 'F',
    '/': 'Q', 'Q': 'Q', 'f': 'Q'
}

CLASSES = ['N', 'S', 'V', 'F']
LABEL_TO_ID = {c: i for i, c in enumerate(CLASSES)}
TARGET_CLASSES = set(CLASSES)


# ---------------------------
# GAN Generator Loader
# ---------------------------
def find_latest_checkpoint(base_path='log', target_label='S', prefer_epoch=None):
    """
    최신 또는 특정 epoch의 체크포인트 자동 탐색
    
    Args:
        base_path: 로그 디렉토리 기본 경로
        target_label: 타겟 클래스 (S, V, F 등)
        prefer_epoch: 선호하는 epoch 번호 (None이면 최신 사용)
    
    Returns:
        체크포인트 경로 또는 None
    """
    patterns = [
        f'{base_path}/model_{target_label}/models/{target_label}_epoch_*.pth',
        f'{base_path}/*_{target_label}/models/{target_label}_epoch_*.pth',
        f'{base_path}/*/models/{target_label}_epoch_*.pth',
    ]
    
    all_checkpoints = []
    for pattern in patterns:
        all_checkpoints.extend(glob.glob(pattern))
    
    if not all_checkpoints:
        return None
    
    # Epoch 번호로 정렬
    def extract_epoch(path):
        try:
            filename = os.path.basename(path)
            epoch_str = filename.split('epoch_')[1].split('.pth')[0]
            return int(epoch_str)
        except:
            return 0
    
    all_checkpoints.sort(key=extract_epoch, reverse=True)
    
    # 특정 epoch 선호
    if prefer_epoch is not None:
        for ckpt in all_checkpoints:
            if f'epoch_{prefer_epoch}.pth' in ckpt:
                return ckpt
    
    return all_checkpoints[0] if all_checkpoints else None


def load_gan_generator(checkpoint_path, device='cuda'):
    """학습된 GAN Generator 로드"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}\n"
                                f"Please train TransWGAN-GP first")
    
    print(f"[GAN] Loading Generator from: {checkpoint_path}")
    
    try:
        from TransGAN import Generator
    except ImportError:
        raise ImportError("Cannot import Generator from TransGAN.py")
    
    generator = Generator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'generator' in checkpoint:
        generator.load_state_dict(checkpoint['generator'])
    else:
        raise KeyError("'generator' key not found in checkpoint")
    
    generator.eval()
    epoch = checkpoint.get('epoch', 'unknown')
    loss_G = checkpoint.get('loss_G', 'unknown')
    
    print(f"[GAN] ✓ Loaded Generator from epoch {epoch}, loss: {loss_G}")
    
    return generator


def generate_gan_samples(generator, num_samples, device='cuda', batch_size=64):
    """GAN으로 샘플 생성"""
    generated_samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            z = torch.randn(current_batch_size, 256).to(device)
            fake = generator(z).cpu().numpy()  # [batch, 200]
            
            for j in range(current_batch_size):
                generated_samples.append(fake[j])
            
            # 배치마다 GPU 메모리 정리
            del z, fake
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    return generated_samples


def clear_gpu_memory(device):
    """GPU 메모리 완전 정리"""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()


# ---------------------------
# Main building function
# ---------------------------
import matplotlib.pyplot as plt

def build_interpatient_splits(debug_plot=False):
    safe_mkdir(SAVE_DIR)

    # Device 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[DEVICE] Using: {device}")
    
    if device.type == 'cuda':
        print(f"[DEVICE] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[DEVICE] Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    ds1_records = {f'{r:03d}' for r in DS1_TRAIN}
    ds2_records = {f'{r:03d}' for r in DS2_TEST}

    with open(os.path.join(PATH_DB, 'RECORDS'), 'r') as fin:
        all_recs = set(fin.read().strip().splitlines())

    ds1_use = sorted(ds1_records & all_recs)
    ds2_use = sorted(ds2_records & all_recs)

    train_data, train_labels_str, train_labels_id, train_pid = [], [], [], []
    test_data,  test_labels_str,  test_labels_id,  test_pid  = [], [], [], []

    miss_lead = defaultdict(list)
    skipped_q = 0
    skipped_out = 0
    total_ann = 0

    # -----------------------
    # 수집 함수
    # -----------------------
    def collect_from_records(rec_list, split_name, do_augment=False):
        nonlocal skipped_q, skipped_out, total_ann
        out_data, out_labels_str, out_labels_id, out_pid = [], [], [], []
        
        for rec in rec_list:
            try:
                ann, sig, meta = read_record(rec, PATH_DB)
            except Exception as e:
                print(f'[WARN] failed to read {rec}: {e}')
                continue

            fs = float(meta['fs'])
            ch_idx, used_lead, sig_names = find_channel(meta, VALID_LEADS)
            if ch_idx is None:
                miss_lead[rec] = sig_names
                print(f'[INFO] {rec}: no valid lead among {sig_names}')
                continue

            x_original = sig[:, ch_idx].astype(np.float32)
            x_baseline_removed = remove_baseline_median(x_original, fs=int(round(fs)), 
                                                       win_ms1=200, win_ms2=600)

            idx_list = list(ann['sample'])
            label_list = list(ann['symbol'])

            pre = int(round(90  * fs / 360.0))
            post= int(round(110 * fs / 360.0))

            for s_idx, sym in enumerate(label_list):
                total_ann += 1
                grp = LABEL_GROUP_MAP.get(sym, None)
                if (grp is None) or (grp == 'Q') or (grp not in TARGET_CLASSES):
                    if grp == 'Q':
                        skipped_q += 1
                    continue

                center = idx_list[s_idx]
                start = center - pre
                end   = center + post
                if start < 0 or end > len(x_baseline_removed):
                    skipped_out += 1
                    continue

                seg_extracted = x_baseline_removed[start:end].astype(np.float32)

                # Normalize
                mean = seg_extracted.mean()
                std = seg_extracted.std()
                seg_normalized = (seg_extracted - mean) / (std + 1e-14)

                # Resample
                seg_resampled = resample_to_len(seg_normalized, OUT_LEN)

                out_data.append(seg_resampled)
                out_labels_str.append(grp)
                out_labels_id.append(LABEL_TO_ID[grp])
                out_pid.append(rec)

            print(f'[{split_name}] {rec} | lead={used_lead} | fs={fs:.0f} | collected_total={len(out_data)}')

        return out_data, out_labels_str, out_labels_id, out_pid
    
    # -----------------------
    # Collect
    # -----------------------
    tr_d, tr_y_str, tr_y_id, tr_pid = collect_from_records(ds1_use, 'TRAIN', do_augment=False)
    te_d, te_y_str, te_y_id, te_pid = collect_from_records(ds2_use, 'TEST',  do_augment=False)

    # -----------------------
    # GAN Augmentation (클래스별 순차 처리로 메모리 관리)
    # -----------------------
    if AUGMENT_TRAIN and USE_GAN_AUGMENT:
        print("\n" + "="*70)
        print("[GAN AUGMENTATION] Starting GAN-based data augmentation")
        print("="*70)
        
        # 원본 클래스별 샘플 개수 확인
        original_counts = Counter(tr_y_str)
        
        # 클래스별로 순차 처리 (메모리 효율)
        for class_label, config in GAN_AUGMENT_CONFIG.items():
            if not config['enabled']:
                print(f"[{class_label}] Skipped (disabled in config)")
                continue
            
            class_id = LABEL_TO_ID[class_label]
            num_original = original_counts.get(class_label, 0)
            
            if num_original == 0:
                print(f"[{class_label}] No original samples, skipping")
                continue
            
            multiplier = config['multiplier']
            num_to_generate = num_original * (multiplier - 1)
            
            print(f"\n[{class_label}] Original: {num_original} samples")
            print(f"[{class_label}] Target: {multiplier}x augmentation")
            print(f"[{class_label}] Generating: {num_to_generate} samples...")
            
            try:
                # ===== Generator 로드 =====
                checkpoint_path = config['checkpoint']
                
                # 경로가 없으면 자동 탐색
                if not os.path.exists(checkpoint_path):
                    print(f"[{class_label}] Checkpoint not found, searching...")
                    auto_checkpoint = find_latest_checkpoint(
                        base_path='log', 
                        target_label=class_label,
                        prefer_epoch=None
                    )
                    
                    if auto_checkpoint:
                        print(f"[{class_label}] Found: {auto_checkpoint}")
                        checkpoint_path = auto_checkpoint
                    else:
                        print(f"[{class_label}] WARNING: No checkpoint found, skipping")
                        continue
                
                # Generator 로드
                generator = load_gan_generator(checkpoint_path, device)
                
                # ===== 샘플 생성 =====
                gan_samples = generate_gan_samples(generator, num_to_generate, device, batch_size=64)
                
                # 생성된 샘플을 데이터셋에 추가
                for sample in gan_samples:
                    tr_d.append(sample)
                    tr_y_str.append(class_label)
                    tr_y_id.append(class_id)
                    tr_pid.append(f'GAN_{class_label}')
                
                print(f"[{class_label}] ✓ Added {len(gan_samples)} GAN samples")
                print(f"[{class_label}] ✓ Total: {num_original} → {num_original + len(gan_samples)}")
                
                # ===== 메모리 정리 =====
                del generator, gan_samples
                clear_gpu_memory(device)
                print(f"[{class_label}] ✓ GPU memory cleared")
                
            except Exception as e:
                print(f"[{class_label}] ERROR: {e}")
                # 에러 발생 시에도 메모리 정리
                clear_gpu_memory(device)
                continue
        
        print("\n" + "="*70 + "\n")

    # -----------------------
    # to numpy & save
    # -----------------------
    train_data = np.asarray(tr_d, dtype=np.float32)
    test_data  = np.asarray(te_d, dtype=np.float32)
    train_labels_str = np.asarray(tr_y_str)
    test_labels_str  = np.asarray(te_y_str)
    train_labels_id  = np.asarray(tr_y_id, dtype=np.int32)
    test_labels_id   = np.asarray(te_y_id, dtype=np.int32)
    train_pid = np.asarray(tr_pid)
    test_pid  = np.asarray(te_pid)

    # Save
    out_train = os.path.join(SAVE_DIR, 'mitbih_train.npz')
    out_test  = os.path.join(SAVE_DIR, 'mitbih_test.npz')

    # NPZECGDataset 형식에 맞게 저장
    np.savez_compressed(
        out_train,
        data=train_data,
        labels_id=train_labels_id,
        labels_str=train_labels_str,
        pid=train_pid,
        classes=np.array(CLASSES)
    )
    np.savez_compressed(
        out_test,
        data=test_data,
        labels_id=test_labels_id,
        labels_str=test_labels_str,
        pid=test_pid,
        classes=np.array(CLASSES)
    )

    print('\n' + '='*70)
    print('FINAL SUMMARY')
    print('='*70)
    print(f'Total annotations: {total_ann}')
    print(f'  - Skipped Q-class: {skipped_q}')
    print(f'  - Skipped out-of-bound: {skipped_out}')
    print(f'Missing lead records: {len(miss_lead)}')
    print()
    
    # 클래스별 통계
    train_counter = Counter(train_labels_str)
    test_counter  = Counter(test_labels_str)
    
    print('TRAIN distribution:')
    for cls in CLASSES:
        count = train_counter.get(cls, 0)
        pct = 100.0 * count / len(train_labels_str) if len(train_labels_str) > 0 else 0
        
        # GAN 증강 적용 여부 표시
        gan_marker = ""
        if USE_GAN_AUGMENT and cls in GAN_AUGMENT_CONFIG:
            if GAN_AUGMENT_CONFIG[cls]['enabled']:
                multiplier = GAN_AUGMENT_CONFIG[cls]['multiplier']
                gan_marker = f" [GAN {multiplier}x]"
        
        print(f'  {cls}: {count:5d} ({pct:5.2f}%){gan_marker}')
    print(f'  Total: {len(train_labels_str)}')
    
    print('\nTEST distribution:')
    for cls in CLASSES:
        count = test_counter.get(cls, 0)
        pct = 100.0 * count / len(test_labels_str) if len(test_labels_str) > 0 else 0
        print(f'  {cls}: {count:5d} ({pct:5.2f}%)')
    print(f'  Total: {len(test_labels_str)}')
    
    print()
    print(f'Saved → {out_train}')
    print(f'Saved → {out_test}')
    print('='*70)

    return {
        'train': (train_data, train_labels_id, train_labels_str, train_pid),
        'test':  (test_data,  test_labels_id,  test_labels_str,  test_pid),
        'stats': {
            'train_counter': train_counter,
            'test_counter': test_counter,
            'total_ann': total_ann,
            'skipped_q': skipped_q,
            'skipped_out': skipped_out
        }
    }


# 사용 예시
if __name__ == '__main__':
    """
    사용법:
    
    1. S 클래스만 증강:
       GAN_AUGMENT_CONFIG['S']['enabled'] = True
       GAN_AUGMENT_CONFIG['V']['enabled'] = False
       GAN_AUGMENT_CONFIG['F']['enabled'] = False
    
    2. S, V 클래스 증강:
       GAN_AUGMENT_CONFIG['S']['enabled'] = True
       GAN_AUGMENT_CONFIG['V']['enabled'] = True
       GAN_AUGMENT_CONFIG['F']['enabled'] = False
    
    3. 모든 클래스 증강:
       GAN_AUGMENT_CONFIG['S']['enabled'] = True
       GAN_AUGMENT_CONFIG['V']['enabled'] = True
       GAN_AUGMENT_CONFIG['F']['enabled'] = True
    
    4. GAN 증강 완전 비활성화:
       USE_GAN_AUGMENT = False
    """
    results = build_interpatient_splits(debug_plot=False)