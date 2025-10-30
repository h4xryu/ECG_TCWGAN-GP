import os
import argparse
import torch
import options
import utils
import time
from TransGAN import Generator, Discriminator
from utils import Bar, label2index, ECGDataloader, Writer, save_checkpoint
from torch.utils.data import DataLoader
import random
from sklearn.metrics import f1_score
from loss import KDLoss, CLTLoss
from sklearn.metrics import confusion_matrix
from dataloader import NPZECGDataset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class Trainer:
    def __init__(self, opt, target_label='F', use_wgan=True, use_gp=True):
        """
        target_label: 'N', 'S', 'V', 'F' 중 하나 선택 (논문처럼 클래스별 GAN 훈련)
        use_wgan: True면 WGAN, False면 기본 GAN
        use_gp: True면 Gradient Penalty 사용 (WGAN-GP)
        """
        self.opt = opt
        self.device = opt.device
        self.target_label = target_label
        self.use_wgan = use_wgan
        self.use_gp = use_gp
        
        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # WGAN 설정
        if self.use_wgan:
            if self.use_gp:
                self.lambda_gp = 10
                self.n_critic = 5
                print(f"[INIT] Using TransWGAN-GP (lambda_gp={self.lambda_gp}, n_critic={self.n_critic})")
            else:
                self.clip_value = 0.01
                self.n_critic = 5
                print(f"[INIT] Using TransWGAN (clip={self.clip_value}, n_critic={self.n_critic})")
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
            print(f"[INIT] Using TransGAN (standard)")

        # GAN용 optimizer
        if self.use_wgan and self.use_gp:
            self.opt_D = torch.optim.Adam(self.discriminator.parameters(), lr=0.00001, betas=(0.0, 0.9))
            self.opt_G = torch.optim.Adam(self.generator.parameters(), lr=0.00001, betas=(0.0, 0.9))
        elif self.use_wgan:
            self.opt_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=0.00005)
            self.opt_G = torch.optim.RMSprop(self.generator.parameters(), lr=0.00005)
        else:
            self.opt_D = torch.optim.AdamW(self.discriminator.parameters(), lr=self.opt.lr_initial)
            self.opt_G = torch.optim.AdamW(self.generator.parameters(), lr=self.opt.lr_initial)

        # 로그 디렉토리 생성
        self.writer = Writer(self._get_tboard_dir())
        self.train_loader, self.valid_loader = self._load_data()

        print(f"[INIT] Target label = {self.target_label}")

    # ---------------------------
    # Data
    # ---------------------------
    def _load_data(self):
        full_train = NPZECGDataset(self.opt.path_train_npz)
        full_valid = NPZECGDataset(self.opt.path_val_npz)

        # 특정 클래스만 필터링 (논문: 각 클래스별 별도 GAN)
        if self.target_label is not None:
            train_data = [(x, y) for x, y in full_train if full_train.classes[int(y)] == self.target_label]
            val_data = [(x, y) for x, y in full_valid if full_valid.classes[int(y)] == self.target_label]
        else:
            train_data = list(full_train)
            val_data = list(full_valid)

        print(f"[DATA] Target {self.target_label}: {len(train_data)} train / {len(val_data)} val samples")

        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
        valid_loader = DataLoader(val_data, batch_size=64, shuffle=False)
        return train_loader, valid_loader

    # ---------------------------
    # Gradient Penalty for WGAN-GP
    # ---------------------------
    def _compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN-GP"""
        batch_size = real_samples.size(0)
        
        # real_samples shape: [batch, 1, 200] or [batch, 200]
        # fake_samples shape: [batch, 200]
        # 둘 다 [batch, 1, 200]로 통일
        if real_samples.dim() == 2:
            real_samples = real_samples.unsqueeze(1)
        if fake_samples.dim() == 2:
            fake_samples = fake_samples.unsqueeze(1)
        
        # Random weight term for interpolation between real and fake samples
        alpha = torch.rand(batch_size, 1, 1).to(self.device)
        
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates)
        
        fake_labels = torch.ones(batch_size, 1).to(self.device)
        
        # Get gradient w.r.t. interpolates
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake_labels,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    

    # ---------------------------
    # GAN Training per class
    # ---------------------------
    def train(self):
        gan_type = 'TransWGAN-GP' if (self.use_wgan and self.use_gp) else ('TransWGAN' if self.use_wgan else 'TransGAN')
        print(f"Training {gan_type} for class [{self.target_label}]...")

        # 모델 저장 디렉토리 생성
        model_dir = self._get_model_dir()
        utils.mkdir(model_dir)
        
        # 에폭별 plot 저장 디렉토리 생성
        plot_dir = os.path.join("generated_samples", f"{self.target_label}_epochs")
        os.makedirs(plot_dir, exist_ok=True)

        step = 0  # Global step counter
        
        for epoch in range(1, self.opt.nepoch + 1):
            loss_D_epoch, loss_G_epoch, gp_epoch = 0.0, 0.0, 0.0
            batch_count = 0

            for X, _ in utils.Bar(desc=f"Epoch {epoch} ({self.target_label})", dataloader=self.train_loader):
                X = X.float().to(self.device)
                
                # X shape 확인 및 통일: [batch, 1, 200]
                if X.dim() == 2:
                    X = X.unsqueeze(1)
                
                batch_size = X.size(0)
                batch_count += 1

                if self.use_wgan:
                    # ============ WGAN / WGAN-GP Training ============
                    self.opt_D.zero_grad()
                    
                    # Real samples
                    real_validity = self.discriminator(X)
                    
                    # Fake samples
                    z = torch.randn(batch_size, 256).to(self.device)
                    fake = self.generator(z)  # [batch, 200]
                    fake_validity = self.discriminator(fake)

                    # Wasserstein loss
                    D_x_loss = torch.mean(real_validity)
                    D_z_loss = torch.mean(fake_validity)
                    d_loss = D_z_loss - D_x_loss
                    
                    # Gradient penalty (WGAN-GP)
                    if self.use_gp:
                        gradient_penalty = self._compute_gradient_penalty(X, fake.detach())
                        d_loss = d_loss + self.lambda_gp * gradient_penalty
                        gp_epoch += gradient_penalty.item()
                    
                    d_loss.backward()
                    self.opt_D.step()

                    # Weight clipping (only for WGAN without GP)
                    if not self.use_gp:
                        for p in self.discriminator.parameters():
                            p.data.clamp_(-self.clip_value, self.clip_value)

                    loss_D_epoch += d_loss.item()

                    # Train Generator (every n_critic steps)
                    if step % self.n_critic == 0:
                        self.opt_D.zero_grad()
                        self.opt_G.zero_grad()
                        
                        z = torch.randn(batch_size, 256).to(self.device)
                        gen_ecg = self.generator(z)
                        z_outputs = self.discriminator(gen_ecg)

                        # Generator loss
                        loss_G = -torch.mean(z_outputs)
                        loss_G.backward()
                        self.opt_G.step()
                        
                        loss_G_epoch += loss_G.item()

                else:
                    # ============ Standard GAN Training ============
                    # Train D
                    self.opt_D.zero_grad()

                    # X shape 확인 및 통일: [batch, 1, 200]
                    if X.dim() == 2:
                        X = X.unsqueeze(1)

                    real_pred = self.discriminator(X)
                    real_target = torch.ones_like(real_pred)
                    loss_real = self.loss_fn(real_pred, real_target)

                    z = torch.randn(batch_size, 256).to(self.device)
                    fake = self.generator(z)  # [batch, 200]
                    fake_pred = self.discriminator(fake)
                    fake_target = torch.zeros_like(fake_pred)
                    loss_fake = self.loss_fn(fake_pred, fake_target)

                    loss_D = 0.5 * (loss_real + loss_fake)
                    loss_D.backward()
                    self.opt_D.step()

                    # Train G
                    self.opt_G.zero_grad()
                    z = torch.randn(batch_size, 256).to(self.device)
                    gen_ecg = self.generator(z)
                    pred_fake = self.discriminator(gen_ecg)

                    target_for_G = torch.ones_like(pred_fake)
                    loss_G = self.loss_fn(pred_fake, target_for_G)
                    loss_G.backward()
                    self.opt_G.step()

                    loss_D_epoch += loss_D.item()
                    loss_G_epoch += loss_G.item()
                
                step += 1

            # 평균 loss 계산
            avg_loss_D = loss_D_epoch / batch_count
            g_update_count = batch_count // self.n_critic if self.use_wgan else batch_count

            
            avg_loss_G = loss_G_epoch / max(g_update_count, 1)
            avg_gp = gp_epoch / batch_count if (self.use_wgan and self.use_gp) else 0.0

            # 출력 메시지
            msg = f"[{self.target_label}] Epoch {epoch}: Loss_D={avg_loss_D:.4f}, Loss_G={avg_loss_G:.4f}"
            if self.use_wgan and self.use_gp:
                msg += f", GP={avg_gp:.4f}"
            print(msg)

            # TensorBoard 로그 기록
            self.writer.add_scalar(f'{self.target_label}/Loss_D', avg_loss_D, epoch)
            self.writer.add_scalar(f'{self.target_label}/Loss_G', avg_loss_G, epoch)
            if self.use_wgan and self.use_gp:
                self.writer.add_scalar(f'{self.target_label}/Gradient_Penalty', avg_gp, epoch)

            # 매 에폭마다 샘플 생성 및 저장 (시드 고정)
            self._generate_epoch_samples(epoch, plot_dir)

            # 모델 체크포인트 저장 (매 10 에폭마다 또는 마지막 에폭)
            if epoch % 10 == 0 or epoch == self.opt.nepoch:
                checkpoint = {
                    'epoch': epoch,
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'opt_G': self.opt_G.state_dict(),
                    'opt_D': self.opt_D.state_dict(),
                    'loss_D': avg_loss_D,
                    'loss_G': avg_loss_G,
                    'use_wgan': self.use_wgan,
                    'use_gp': self.use_gp,
                }
                save_path = os.path.join(model_dir, f'{self.target_label}_epoch_{epoch}.pth')
                torch.save(checkpoint, save_path)
                print(f"✓ Saved checkpoint: {save_path}")

        # 학습 완료 후 최종 샘플 시각화
        self._generate_examples()

    # ---------------------------
    # 매 에폭마다 샘플 생성 (시드 고정)
    # ---------------------------
    @torch.no_grad()
    def _generate_epoch_samples(self, epoch, plot_dir, n=10):
        """매 에폭마다 동일한 시드로 샘플 생성하여 학습 과정 추적"""
        self.generator.eval()
        
        # 시드 고정하여 매 에폭마다 동일한 noise에서 생성
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        
        z = torch.randn(n, 256).to(self.device)
        fake = self.generator(z).cpu().numpy()

        # numpy 배열로 첫 번째 샘플 저장
        npy_fname = os.path.join(plot_dir, f"epoch_{epoch:04d}_sample.npy")
        np.save(npy_fname, fake[0])
        
        # 플롯 생성
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(n):
            signal = fake[i]
            axes[i].plot(signal, linewidth=1.0)
            axes[i].set_title(f"Sample #{i+1}", fontsize=10)
            
            # y축 범위 설정
            if signal.max() != signal.min():
                axes[i].set_ylim([signal.min() - 0.1 * abs(signal.min()), 
                                  signal.max() + 0.1 * abs(signal.max())])
            
            axes[i].set_xlabel('Time', fontsize=8)
            axes[i].set_ylabel('Amplitude', fontsize=8)
            axes[i].grid(True, alpha=0.3, linewidth=0.5)
        
        plt.suptitle(f'Class [{self.target_label}] - Epoch {epoch}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # 이미지 저장
        img_fname = os.path.join(plot_dir, f"epoch_{epoch:04d}.png")
        plt.savefig(img_fname, dpi=100, bbox_inches='tight')
        plt.close('all')
        
        # 진행 상황 표시
        if epoch % 10 == 0 or epoch == 1:
            print(f"  → Saved: {img_fname}")
            print(f"  → Saved: {npy_fname} (shape: {fake[0].shape}, "
                  f"range: [{fake[0].min():.3f}, {fake[0].max():.3f}])")
        
        self.generator.train()

    # ---------------------------
    # Generator 샘플 시각화
    # ---------------------------
    @torch.no_grad()
    def _generate_examples(self, n=10):
        self.generator.eval()
        z = torch.randn(n, 256).to(self.device)
        fake = self.generator(z).cpu().numpy()

        fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
        for i in range(n):
            signal = fake[i]
            axes[i].plot(signal, linewidth=1.0)
            axes[i].set_title(f"{self.target_label} #{i+1}", fontsize=9)
            
            # y축 범위 설정
            if signal.max() != signal.min():
                axes[i].set_ylim([signal.min() - 0.1 * abs(signal.min()), 
                                  signal.max() + 0.1 * abs(signal.max())])
            
            axes[i].grid(True, alpha=0.3, linewidth=0.5)
            axes[i].set_xlabel('Time', fontsize=7)
            axes[i].set_ylabel('Amp', fontsize=7)
            
        plt.tight_layout()
        os.makedirs("generated_samples", exist_ok=True)
        fname = f"generated_samples/{self.target_label}_TCGAN_samples.png"
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"✓ Saved {n} fake samples for [{self.target_label}] → {fname}")

    # ---------------------------
    # Log / Utility
    # ---------------------------
    def _get_tboard_dir(self):
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'log', f'{self.opt.log_name}_{self.target_label}')
        utils.mkdir(log_dir)
        utils.mkdir(os.path.join(log_dir, 'logs'))
        return os.path.join(log_dir, 'logs')

    def _get_model_dir(self):
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                               'log', f'{self.opt.log_name}_{self.target_label}')
        utils.mkdir(log_dir)
        return os.path.join(log_dir, 'models')


if __name__ == '__main__':
    # Parse command-line arguments
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    opt = options.Options().init(argparse.ArgumentParser(description='ECG Classification')).parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    # Initialize trainer and start training
    # 옵션 설명:
    # use_wgan=True, use_gp=True  → TransWGAN-GP (추천)
    # use_wgan=True, use_gp=False → TransWGAN (weight clipping)
    # use_wgan=False              → TransGAN (standard)
    trainer = Trainer(opt, target_label='F', use_wgan=True, use_gp=True)
    trainer.train()