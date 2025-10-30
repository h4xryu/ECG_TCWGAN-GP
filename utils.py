import os
import time
import torch
import numpy as np
from tensorboardX import SummaryWriter


# For dataset
class ECGDataloader():  # 1110 - 4096 samples
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.float))

    def __len__(self):
        return len(self.data)


# For dataset
def label2index(i):
    m = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}  # uncomment for 5 classes
    return m[i]


# Create a new directory.
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


# Normalize the ECG data using Z-score normalization.
def normalize_ecg(ecg_data):
    mean = np.mean(ecg_data, axis=0, keepdims=True)
    std = np.std(ecg_data, axis=0, keepdims=True)
    return (ecg_data - mean) / (std + 1e-8)  # Prevent division by zero


# for using pre-training weights
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


# Calculate total number of parameters in a model.
def cal_total_params(our_model):
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters


# Display a progress bar during training/validation.
class Bar(object):
    def __init__(self, dataloader):
        if not hasattr(dataloader, 'dataset'):
            raise ValueError('Attribute `dataset` not exists in dataloder.')
        if not hasattr(dataloader, 'batch_size'):
            raise ValueError('Attribute `batch_size` not exists in dataloder.')

        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self._idx = 0
        self._batch_idx = 0
        self._time = []
        self._DISPLAY_LENGTH = 50

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._time) < 2:
            self._time.append(time.time())

        self._batch_idx += self.batch_size
        if self._batch_idx > len(self.dataset):
            self._batch_idx = len(self.dataset)

        try:
            batch = next(self.iterator)
            self._display()
        except StopIteration:
            raise StopIteration()

        self._idx += 1
        if self._idx >= len(self.dataloader):
            self._reset()

        return batch

    def _display(self):
        if len(self._time) > 1:
            t = (self._time[-1] - self._time[-2])
            eta = t * (len(self.dataloader) - self._idx)
        else:
            eta = 0

        rate = self._idx / len(self.dataloader)
        len_bar = int(rate * self._DISPLAY_LENGTH)
        bar = ('=' * len_bar + '>').ljust(self._DISPLAY_LENGTH, '.')
        idx = str(self._batch_idx).rjust(len(str(len(self.dataset))), ' ')

        tmpl = '\r{}/{}: [{}] - ETA {:.1f}s'.format(
            idx,
            len(self.dataset),
            bar,
            eta
        )
        print(tmpl, end='')
        if self._batch_idx == len(self.dataset):
            print()

    def _reset(self):
        self._idx = 0
        self._batch_idx = 0
        self._time = []


# Define a custom writer class that extends SummaryWriter to log training/validation metrics.
class Writer(SummaryWriter):
    def __init__(self, logdir):
        super(Writer, self).__init__(logdir)

    # Method to log training loss.
    def log_train_loss(self, loss_type, train_loss, step):
        self.add_scalar('train_{}_loss'.format(loss_type), train_loss, step)

    # Method to log validation loss.
    def log_valid_loss(self, loss_type, valid_loss, step):
        self.add_scalar('valid_{}_loss'.format(loss_type), valid_loss, step)

    # Method to log other performance metrics (e.g., accuracy, F1-score).
    def log_score(self, metrics_name, metrics, step):
        # Add a scalar value to the writer with the given metric name.
        self.add_scalar(metrics_name, metrics, step)


def save_checkpoint(exp_log_dir, model, epoch):
    save_dict = {
        "model": model.state_dict(),
        'epoch': epoch
    }
    # save classification report
    save_path = os.path.join(exp_log_dir, "ckpt_opt.pt")

    torch.save(save_dict, save_path)



from torch.utils.tensorboard import SummaryWriter
import time
import os
import torch
import sys


class Colors:
    """ANSI color codes for terminal output"""
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[92m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC = '\033[0m'  # End color
    
    # Bright colors
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_CYAN = '\033[96m'

class ProgressBarStatus:
    """Status constants for progress bar"""
    TRAINING = "training"
    COMPLETED = "completed" 
    INTERRUPTED = "interrupted"

class Bar(object):
    def __init__(self, dataloader, desc="Training", color=Colors.CYAN):
        if not hasattr(dataloader, 'dataset'):
            raise ValueError('Attribute `dataset` not exists in dataloader.')
        if not hasattr(dataloader, 'batch_size'):
            raise ValueError('Attribute `batch_size` not exists in dataloader.')
        
        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self._idx = 0
        self._batch_idx = 0
        self._time = []
        self._DISPLAY_LENGTH = 40  # Wider bar for better visualization
        self.desc = desc
        self.default_color = color
        self.last_loss = None
        self.compact = True  # Enable compact mode
        
        # Status tracking
        self.status = ProgressBarStatus.TRAINING
        self._completed_naturally = False
        self._start_time = time.time()
        
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if len(self._time) < 2:
            self._time.append(time.time())
        
        self._batch_idx += self.batch_size
        if self._batch_idx > len(self.dataset):
            self._batch_idx = len(self.dataset)
        
        try:
            batch = next(self.iterator)
            self._display()
        except StopIteration:
            # Natural completion
            self._completed_naturally = True
            self.status = ProgressBarStatus.COMPLETED
            self._display_final()
            raise StopIteration()
        except KeyboardInterrupt:
            # Manual interruption
            self.status = ProgressBarStatus.INTERRUPTED
            self._display_final()
            raise KeyboardInterrupt()
        except Exception as e:
            # Other exceptions
            self.status = ProgressBarStatus.INTERRUPTED
            self._display_final()
            raise e
        
        self._idx += 1
        if self._idx >= len(self.dataloader):
            self._completed_naturally = True
            self.status = ProgressBarStatus.COMPLETED
            self._reset()
        
        return batch
    
    def update_loss(self, loss_value):
        """Update current loss to display in the progress bar"""
        self.last_loss = loss_value
    
    def mark_completed(self):
        """Manually mark as completed (for successful finish)"""
        self.status = ProgressBarStatus.COMPLETED
        self._completed_naturally = True
        self._display_final()
    
    def mark_interrupted(self):
        """Manually mark as interrupted (for error/cancellation)"""
        self.status = ProgressBarStatus.INTERRUPTED
        self._display_final()
    
    def _get_status_colors(self):
        """Get colors based on current status"""
        if self.status == ProgressBarStatus.TRAINING:
            return {
                'bar_color': Colors.CYAN,
                'desc_color': Colors.BOLD + Colors.CYAN,
                'time_color': Colors.BRIGHT_YELLOW,
                'stats_color': Colors.BOLD
            }
        elif self.status == ProgressBarStatus.COMPLETED:
            return {
                'bar_color': Colors.GREEN,
                'desc_color': Colors.BOLD + Colors.GREEN,
                'time_color': Colors.GREEN,
                'stats_color': Colors.BOLD + Colors.GREEN
            }
        elif self.status == ProgressBarStatus.INTERRUPTED:
            return {
                'bar_color': Colors.BRIGHT_RED,
                'desc_color': Colors.BOLD + Colors.BRIGHT_RED,
                'time_color': Colors.BRIGHT_RED,
                'stats_color': Colors.BOLD + Colors.BRIGHT_RED
            }
        else:
            return {
                'bar_color': self.default_color,
                'desc_color': Colors.BOLD,
                'time_color': Colors.GREEN,
                'stats_color': Colors.BOLD
            }
    
    def _display(self):
        if len(self._time) > 1:
            t = (self._time[-1] - self._time[-2])
            eta = t * (len(self.dataloader) - self._idx)
        else:
            eta = 0
        
        rate = self._idx / len(self.dataloader)
        percentage = int(rate * 100)
        len_bar = int(rate * self._DISPLAY_LENGTH)
        
        # Get status-based colors
        colors = self._get_status_colors()
        
        # Use thinner characters for a more compact display
        bar_fill = '━' * len_bar  # Horizontal line instead of block
        bar_empty = '╌' * (self._DISPLAY_LENGTH - len_bar)  # Dotted line instead of block
        
        idx = str(self._batch_idx).rjust(len(str(len(self.dataset))), ' ')
        
        # Format with status-based colors
        prefix = f"{colors['desc_color']}{self.desc}{Colors.ENDC}"
        progress = f"{colors['bar_color']}{bar_fill}{bar_empty}{Colors.ENDC}"
        stats = f"{colors['stats_color']}{percentage:3d}%{Colors.ENDC}"
        
        # Add loss display if available
        loss_display = ""
        if self.last_loss is not None:
            loss_display = f" {Colors.BRIGHT_YELLOW}loss:{self.last_loss:.4f}{Colors.ENDC}"
        
        # Time display with green color
        time_display = f"{colors['time_color']}{eta:.1f}s{Colors.ENDC}"
        
        # Compact display in a single line
        if self.compact:
            tmpl = f"\r{prefix} {stats} {idx}/{len(self.dataset)} [{progress}] {time_display}{loss_display}"
        else:
            # Original multi-component display
            time_info = f"ETA: {time_display}"
            tmpl = f"\r{prefix}: |{progress}| {stats} {idx}/{len(self.dataset)} {time_info}{loss_display}"
        
        sys.stdout.write(tmpl)
        sys.stdout.flush()
        
        # Don't print newline here - let _display_final handle it
    
    def _display_final(self):
        """Display final status with appropriate colors"""
        total_time = time.time() - self._start_time
        colors = self._get_status_colors()
        
        # Calculate final percentage
        if self.status == ProgressBarStatus.COMPLETED:
            percentage = 100
            len_bar = self._DISPLAY_LENGTH
            status_text = "COMPLETED"
            status_icon = "✓"
        else:
            rate = self._idx / len(self.dataloader)
            percentage = int(rate * 100)
            len_bar = int(rate * self._DISPLAY_LENGTH)
            status_text = "INTERRUPTED"
            status_icon = "✗"
        
        # Create final progress bar
        bar_fill = '━' * len_bar
        bar_empty = '╌' * (self._DISPLAY_LENGTH - len_bar)
        
        idx = str(self._batch_idx).rjust(len(str(len(self.dataset))), ' ')
        
        # Format final display
        prefix = f"{colors['desc_color']}{self.desc}{Colors.ENDC}"
        progress = f"{colors['bar_color']}{bar_fill}{bar_empty}{Colors.ENDC}"
        stats = f"{colors['stats_color']}{percentage:3d}%{Colors.ENDC}"
        
        # Add loss display if available
        loss_display = ""
        if self.last_loss is not None:
            loss_display = f" {Colors.BRIGHT_YELLOW}loss:{self.last_loss:.4f}{Colors.ENDC}"
        
        # Final time display
        time_display = f"{colors['time_color']}{total_time:.1f}s{Colors.ENDC}"
        status_display = f"{colors['desc_color']}{status_icon} {status_text}{Colors.ENDC}"
        
        # Final display line
        if self.compact:
            tmpl = f"\r{prefix} {stats} {idx}/{len(self.dataset)} [{progress}] {time_display}{loss_display} {status_display}"
        else:
            tmpl = f"\r{prefix}: |{progress}| {stats} {idx}/{len(self.dataset)} Total: {time_display}{loss_display} {status_display}"
        
        sys.stdout.write(tmpl)
        sys.stdout.flush()
        print()  # Add newline after final display
    
    def _reset(self):
        self._idx = 0
        self._batch_idx = 0
        self._time = []

# Enhanced Bar for training with context manager support
class TrainingBar(Bar):
    """Enhanced Bar with context manager support for training loops"""
    
    def __init__(self, dataloader, desc="Training", color=Colors.CYAN):
        super().__init__(dataloader, desc, color)
        self.epoch = 1
        self.total_epochs = None
    
    def set_epoch_info(self, current_epoch, total_epochs):
        """Set epoch information for display"""
        self.epoch = current_epoch
        self.total_epochs = total_epochs
        if total_epochs:
            self.desc = f"Epoch {current_epoch}/{total_epochs}"
        else:
            self.desc = f"Epoch {current_epoch}"
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - handle different exit scenarios"""
        if exc_type is None:
            # Normal completion
            self.mark_completed()
        elif exc_type == KeyboardInterrupt:
            # Manual interruption
            self.mark_interrupted()
            return False  # Re-raise the exception
        else:
            # Other exceptions
            self.mark_interrupted()
            return False  # Re-raise the exception
        


class Writer(SummaryWriter):
    def __init__(self, logdir):
        super(Writer, self).__init__(logdir)
        self.epoch = 0
        self.best_metric = float('inf')
    
    def log_train_loss(self, loss_type, train_loss, step):
        """Log training loss with colorful output"""
        self.add_scalar(f'train_{loss_type}_loss', train_loss, step)
        print(f"{Colors.BOLD}Train {loss_type} Loss:{Colors.ENDC} {Colors.GREEN}{train_loss:.6f}{Colors.ENDC}")
    
    def log_valid_loss(self, loss_type, valid_loss, step):
        """Log validation loss with colorful output"""
        self.add_scalar(f'valid_{loss_type}_loss', valid_loss, step)
        print(f"{Colors.BOLD}Valid {loss_type} Loss:{Colors.ENDC} {Colors.BLUE}{valid_loss:.6f}{Colors.ENDC}")
    
    def log_score(self, metrics_name, metrics, step):
        """Log metrics with colorful output"""
        self.add_scalar(metrics_name, metrics, step)
        print(f"{Colors.BOLD}{metrics_name}:{Colors.ENDC} {Colors.CYAN}{metrics:.6f}{Colors.ENDC}")
        
        # Track best metrics for saving checkpoints
        if metrics_name == 'validation_loss' and metrics < self.best_metric:
            self.best_metric = metrics
            return True
        return False



# ---------------------------
# Utils
# ---------------------------
import wfdb
from scipy.interpolate import interp1d
from collections import Counter, defaultdict

def resample_to_len(ts: np.ndarray, out_len: int) -> np.ndarray:
    """1D 시퀀스를 길이 out_len으로 선형보간 리샘플."""
    if len(ts) == out_len:
        return ts.astype(np.float32, copy=True)
    if len(ts) < 2:  # 최소 길이 보호
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


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

# class_weights = {}
# new_class_counts = Counter(train_labels)
# n_max = max(new_class_counts.values())
# n_min = min(new_class_counts.values())

# # 범위 설정 (0.1 ~ 1.0)
# range_min, range_max = 0.1, 1.0

# for cls, count in new_class_counts.items():
#     # 논문의 공식 적용: w = ((n - n_min) / (n_max - n_min)) * (range_min - range_max) + range_max
#     weight = ((count - n_min) / (n_max - n_min + 1e-8)) * (range_min - range_max) + range_max
#     # 클래스가 적을수록 더 높은 가중치 부여 (반전)
#     weight = range_min + range_max - weight
#     class_weights[cls] = weight

# print("\nClass weights:", class_weights)
# self.class_weights = class_weights  # 클래스 가중치 저장
