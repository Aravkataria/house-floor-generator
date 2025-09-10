import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import random

# Paths
clean_image_dir = 'processed'
checkpoint_dir = 'stage3_checkpoints'
generated_dir = 'stage3_generated_images'

# Hyperparameters
batch_size = 4
lr = 0.0002
num_epochs = 100
image_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

# ------------------ Dataset ------------------
class DenoiseDataset(Dataset):
    def __init__(self, clean_dir):
        self.clean_paths = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def add_noise(self, img):
        noise = torch.randn_like(img) * 0.3
        return torch.clamp(img + noise, 0., 1.)

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean = Image.open(self.clean_paths[idx]).convert('L')
        clean = self.transform(clean)
        noisy = self.add_noise(clean)
        return noisy, clean

# ------------------ Generator ------------------
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        def down(in_c, out_c, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if bn: layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up(in_c, out_c, dropout=False):
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
                      nn.BatchNorm2d(out_c),
                      nn.ReLU(inplace=True)]
            if dropout: layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.down1 = down(1, 64, bn=False)
        self.down2 = down(64, 128)
        self.down3 = down(128, 256)
        self.down4 = down(256, 512)
        self.down5 = down(512, 512)
        self.down6 = down(512, 512)
        self.down7 = down(512, 512)
        self.down8 = down(512, 512, bn=False)

        self.up1 = up(512, 512, dropout=True)
        self.up2 = up(1024, 512, dropout=True)
        self.up3 = up(1024, 512, dropout=True)
        self.up4 = up(1024, 512)
        self.up5 = up(1024, 256)
        self.up6 = up(512, 128)
        self.up7 = up(256, 64)
        self.up8 = nn.Sequential(nn.ConvTranspose2d(128, 1, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1); d3 = self.down3(d2); d4 = self.down4(d3)
        d5 = self.down5(d4); d6 = self.down6(d5); d7 = self.down7(d6); d8 = self.down8(d7)
        u1 = self.up1(d8); u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1)); u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1)); u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1)); return self.up8(torch.cat([u7, d1], 1))

# ------------------ Discriminator ------------------
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1), nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=1))

# ------------------ Init ------------------
dataset = DenoiseDataset(clean_image_dir)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

G = UNetGenerator().to(device)
D = PatchDiscriminator().to(device)
opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
bce = nn.BCELoss(); l1 = nn.L1Loss()

# -------- Resume from latest checkpoint --------
latest_ckpt = os.path.join(checkpoint_dir, 'latest.pth')
start_epoch = 1
if os.path.exists(latest_ckpt):
    ckpt = torch.load(latest_ckpt, map_location=device)
    G.load_state_dict(ckpt['G']); D.load_state_dict(ckpt['D'])
    opt_G.load_state_dict(ckpt['opt_G']); opt_D.load_state_dict(ckpt['opt_D'])
    start_epoch = ckpt['epoch'] + 1
    print(f"Resumed from epoch {start_epoch}")

# ------------------ Training ------------------
for epoch in range(start_epoch, num_epochs + 1):
    G.train(); D.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}")

    for noisy, clean in pbar:
        noisy, clean = noisy.to(device), clean.to(device)
        real_lbl = torch.ones((noisy.size(0), 1, 30, 30), device=device)
        fake_lbl = torch.zeros_like(real_lbl)

        # Train D
        fake_clean = G(noisy)
        d_real = D(clean, noisy); d_fake = D(fake_clean.detach(), noisy)
        d_loss = (bce(d_real, real_lbl) + bce(d_fake, fake_lbl)) / 2
        opt_D.zero_grad(); d_loss.backward(); opt_D.step()

        # Train G
        g_adv = bce(D(fake_clean, noisy), real_lbl)
        g_l1 = l1(fake_clean, clean) * 100
        g_loss = g_adv + g_l1
        opt_G.zero_grad(); g_loss.backward(); opt_G.step()

        pbar.set_postfix(D=d_loss.item(), G=g_loss.item())

    # -------- Save (overwrite) latest checkpoint --------
    torch.save({
        'epoch': epoch, 'G': G.state_dict(), 'D': D.state_dict(),
        'opt_G': opt_G.state_dict(), 'opt_D': opt_D.state_dict()
    }, latest_ckpt)

    # -------- Save generator/discriminator & images every 10 epochs --------
    if epoch % 10 == 0:
        torch.save(G.state_dict(), os.path.join(checkpoint_dir, f'generator_epoch{epoch}.pth'))
        torch.save(D.state_dict(), os.path.join(checkpoint_dir, f'discriminator_epoch{epoch}.pth'))
        G.eval()
        with torch.no_grad():
            sample_noisy, _ = next(iter(loader))
            out = G(sample_noisy.to(device))
            save_image(out, os.path.join(generated_dir, f'denoised_epoch{epoch}.png'), normalize=True)

print("Training Completed!")
