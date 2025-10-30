from scipy.signal import butter, filtfilt
from scipy.signal import medfilt
import numpy as np

def bandpassfilter(signal, fs=360, lowcut=3.0, highcut=45.0, order=4):
    """
    signal: np.ndarray, shape (N, T)  # N=샘플 개수, T=시간길이(360)
    시간축(axis=1)을 따라 zero-phase 필터 적용
    """
    nyq = fs * 0.5
    Wn = [lowcut / nyq, highcut / nyq]
    b, a = butter(order, Wn, btype='band')

    return filtfilt(b, a, signal, axis=1)

# ---------------------------
# Baseline removal (Median-of-medians)
# ---------------------------

def remove_baseline_median(x: np.ndarray, fs: int, win_ms1: int = 200, win_ms2: int = 600) -> np.ndarray:
    """
    두 번의 median filter(200 ms, 600 ms)로 baseline을 추정하고 제거.
    x: (L,) 또는 (N, L) float32/float64, fs: sampling rate
    반환: baseline 제거된 신호, 입력과 동일한 차원 유지
    """
    x = np.asarray(x)
    squeeze_back = False
    if x.ndim == 1:
        x = x[None, :]   # (1, L)로 승격
        squeeze_back = True
    elif x.ndim != 2:
        raise AssertionError("x must be shape (L,) or (N, L)")

    N, L = x.shape

    def _ms_to_odd_k(ms: int) -> int:
        k = int(round(ms * fs / 1000.0))
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        # 커널이 신호길이를 넘지 않도록 보정
        if k > L:
            k = L if L % 2 == 1 else L - 1
            if k < 3:
                k = 3
        return k

    k1 = _ms_to_odd_k(win_ms1)  # ~200ms
    k2 = _ms_to_odd_k(win_ms2)  # ~600ms

    y = np.empty_like(x, dtype=np.float32)
    for i in range(N):
        s = x[i].astype(np.float32, copy=False)
        m1 = medfilt(s, kernel_size=k1)
        baseline = medfilt(m1, kernel_size=k2)
        y[i] = (s - baseline)

    if squeeze_back:
        return y[0].copy()
    return y.copy()




# ---------------------------
# Noise Augmentations (per 2.2 in paper)
# ---------------------------
def add_low_freq_noise(seg: np.ndarray, fs: float) -> np.ndarray:
    """
    저주파 사인성분 추가: A~U[0,0.1], f~U(0,0.2) Hz, phi~U[0,2π)
    """
    n = len(seg)
    t = np.arange(n, dtype=np.float64) / float(fs)
    amp = np.random.uniform(0.0, 0.1)
    f = np.random.uniform(1e-6, 0.2)  # f=0 회피
    phi = np.random.uniform(0.0, 2*np.pi)
    noise = amp * np.sin(2*np.pi*f*t + phi)
    return (seg + noise).astype(np.float32)

def add_high_freq_noise(seg: np.ndarray, fs: float) -> np.ndarray:
    """
    고주파 사인성분 추가: A~U[0,0.15], f~U[45,90] Hz, phi~U[0,2π)
    """
    n = len(seg)
    t = np.arange(n, dtype=np.float64) / float(fs)
    amp = np.random.uniform(0.0, 0.15)
    f = np.random.uniform(45.0, 90.0)
    phi = np.random.uniform(0.0, 2*np.pi)
    noise = amp * np.sin(2*np.pi*f*t + phi)
    return (seg + noise).astype(np.float32)

def add_white_noise(seg: np.ndarray, fs: float) -> np.ndarray:
    """
    백색잡음 추가: 표준편차 = U[0,0.15] * std(seg)
    """
    sigma_scale = np.random.uniform(0.0, 0.15)
    sigma = sigma_scale * (np.std(seg) + 1e-8)
    noise = np.random.normal(loc=0.0, scale=sigma, size=len(seg)).astype(np.float64)
    return (seg + noise).astype(np.float32)

def augment_three(seg: np.ndarray, fs: float):
    """저주파/고주파/화이트 각 1개씩 반환"""
    return (
        add_low_freq_noise(seg, fs),
        add_high_freq_noise(seg, fs),
        add_white_noise(seg, fs),
    )