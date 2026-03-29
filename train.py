import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools
import numpy as np
import os

from model.generator     import CycleGenerator
from model.discriminator import CycleDiscriminator
from data.dataset         import UnpairedDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


class ImageBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        result = []
        for element in data:
            element = element.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                result.append(element)
            else:
                if torch.rand(1).item() > 0.5:
                    i = torch.randint(0, self.max_size, (1,)).item()
                    result.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    result.append(element)
        return torch.cat(result)


def train(echo_dir, mri_dir, epochs=10, batch_size=2, lr=0.0002,
          checkpoint_dir='checkpoints', resume_epoch=0):

    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset    = UnpairedDataset(echo_dir, mri_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=2)

    G_echo2mri = CycleGenerator().to(device)
    G_mri2echo = CycleGenerator().to(device)
    D_echo     = CycleDiscriminator().to(device)
    D_mri      = CycleDiscriminator().to(device)

    if resume_epoch > 0:
        G_echo2mri.load_state_dict(torch.load(
            f'{checkpoint_dir}/G_echo2mri_ep{resume_epoch}.pth'))
        G_mri2echo.load_state_dict(torch.load(
            f'{checkpoint_dir}/G_mri2echo_ep{resume_epoch}.pth'))
        print(f"Resumed from epoch {resume_epoch}")

    criterion_gan   = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_ident = nn.L1Loss()

    g_opt      = optim.Adam(itertools.chain(
                     G_echo2mri.parameters(),
                     G_mri2echo.parameters()),
                     lr=lr, betas=(0.5, 0.999))
    d_echo_opt = optim.Adam(D_echo.parameters(), lr=lr, betas=(0.5, 0.999))
    d_mri_opt  = optim.Adam(D_mri.parameters(),  lr=lr, betas=(0.5, 0.999))

    fake_echo_buffer = ImageBuffer()
    fake_mri_buffer  = ImageBuffer()

    scheduler_g      = optim.lr_scheduler.StepLR(g_opt,      step_size=10, gamma=0.5)
    scheduler_d_mri  = optim.lr_scheduler.StepLR(d_mri_opt,  step_size=10, gamma=0.5)
    scheduler_d_echo = optim.lr_scheduler.StepLR(d_echo_opt, step_size=10, gamma=0.5)

    print(f"\n Training Start! Epochs: {epochs} | Device: {device}\n")

    for epoch in range(resume_epoch + 1, epochs + 1):
        G_echo2mri.train()
        G_mri2echo.train()
        g_losses, d_losses = [], []

        for i, (echo, mri) in enumerate(dataloader):
            echo = echo.to(device)
            mri  = mri.to(device)

            with torch.no_grad():
                _shape = D_mri(mri)
            real_label = torch.ones_like(_shape).to(device)
            fake_label = torch.zeros_like(_shape).to(device)

            g_opt.zero_grad()
            same_mri   = G_echo2mri(mri)
            same_echo  = G_mri2echo(echo)
            loss_ident = (criterion_ident(same_mri,  mri) +
                          criterion_ident(same_echo, echo)) * 5.0

            fake_mri  = G_echo2mri(echo)
            fake_echo = G_mri2echo(mri)
            loss_gan  = (criterion_gan(D_mri(fake_mri),   real_label) +
                         criterion_gan(D_echo(fake_echo), real_label))

            rec_echo   = G_mri2echo(fake_mri)
            rec_mri    = G_echo2mri(fake_echo)
            loss_cycle = (criterion_cycle(rec_echo, echo) +
                          criterion_cycle(rec_mri,  mri)) * 10.0

            g_loss = loss_gan + loss_cycle + loss_ident
            g_loss.backward()
            g_opt.step()

            d_mri_opt.zero_grad()
            fake_mri_buf = fake_mri_buffer.push_and_pop(fake_mri.detach())
            d_mri_loss   = (criterion_gan(D_mri(mri),          real_label) +
                            criterion_gan(D_mri(fake_mri_buf), fake_label)) / 2
            d_mri_loss.backward()
            d_mri_opt.step()

            d_echo_opt.zero_grad()
            fake_echo_buf = fake_echo_buffer.push_and_pop(fake_echo.detach())
            d_echo_loss   = (criterion_gan(D_echo(echo),          real_label) +
                             criterion_gan(D_echo(fake_echo_buf), fake_label)) / 2
            d_echo_loss.backward()
            d_echo_opt.step()

            g_losses.append(g_loss.item())
            d_losses.append((d_mri_loss + d_echo_loss).item() / 2)

            if i % 20 == 0:
                print(f"Ep[{epoch}/{epochs}] Batch[{i}/{len(dataloader)}] "
                      f"G:{g_loss.item():.3f} "
                      f"D:{((d_mri_loss+d_echo_loss)/2).item():.3f}")

        scheduler_g.step()
        scheduler_d_mri.step()
        scheduler_d_echo.step()

        G_echo2mri.eval()
        G_mri2echo.eval()
        torch.save(G_echo2mri.state_dict(),
                   f'{checkpoint_dir}/G_echo2mri_ep{epoch}.pth')
        torch.save(G_mri2echo.state_dict(),
                   f'{checkpoint_dir}/G_mri2echo_ep{epoch}.pth')
        G_echo2mri.train()
        G_mri2echo.train()
        print(f"Checkpoint saved — Epoch {epoch}")
        print(f"Epoch {epoch} — Avg G: {np.mean(g_losses):.3f} "
              f"Avg D: {np.mean(d_losses):.3f}\n")

    print("Training Complete!")
    return G_echo2mri, G_mri2echo


if __name__ == '__main__':
    train(
        echo_dir       = '/content/echo_frames',
        mri_dir        = '/content/mri_frames',
        epochs         = 10,
        batch_size     = 2,
        checkpoint_dir = 'checkpoints',
        resume_epoch   = 0
    )