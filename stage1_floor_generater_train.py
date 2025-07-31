import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# ===== CONFIG =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
CHANNELS = 1
LATENT_DIM = 100
BATCH_SIZE = 8
EPOCHS = 100
LR = 0.0002

DATA_DIR = "datasets/processed_images"   # <-- Change this to your processed dataset path
MODEL_DIR = "models"
SAMPLES_DIR = "generated_samples"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)

# ===== DATASET =====
class FloorPlanDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.img_folder = img_folder
        self.images = [f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg'))]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.images[idx])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

if not os.path.exists(DATA_DIR) or len(os.listdir(DATA_DIR)) == 0:
    print(f"Dataset not found or empty at {DATA_DIR}. Please place images first.")
    exit()

dataset = FloorPlanDataset(DATA_DIR, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ===== GENERATOR =====
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels):
        super(Generator, self).__init__()
        self.fc = nn.Linear(latent_dim, 512 * 16 * 16)

        def block(in_f, out_f):
            return nn.Sequential(
                nn.BatchNorm2d(in_f),
                nn.ConvTranspose2d(in_f, out_f, 4, 2, 1),
                nn.ReLU(True)
            )

        self.gen = nn.Sequential(
            block(512, 256),
            block(256, 128),
            block(128, 64),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.fc(z).view(z.size(0), 512, 16, 16)
        return self.gen(out)

# ===== DISCRIMINATOR =====
class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()

        def block(in_f, out_f, bn=True):
            layers = [nn.Conv2d(in_f, out_f, 4, 2, 1)]
            if bn: layers.append(nn.BatchNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(img_channels, 64, bn=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 0)
        )

    def forward(self, img):
        return self.model(img)

# ===== INIT MODELS =====
generator = Generator(LATENT_DIM, CHANNELS).to(DEVICE)
discriminator = Discriminator(CHANNELS).to(DEVICE)

# ===== OPTIMIZERS & LOSS =====
opt_G = optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.999))
adversarial_loss = nn.BCEWithLogitsLoss()

# ===== TRAINING LOOP =====
if __name__ == "__main__":
    print("Starting Stage 1 Training...")
    for epoch in range(EPOCHS):
        for imgs in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch = imgs.size(0)
            imgs = imgs.to(DEVICE)

            valid = torch.ones_like(discriminator(imgs)).to(DEVICE)
            fake = torch.zeros_like(discriminator(imgs)).to(DEVICE)

            # ---- Train Generator ----
            opt_G.zero_grad()
            z = torch.randn(batch, LATENT_DIM).to(DEVICE)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            opt_G.step()

            # ---- Train Discriminator ----
            opt_D.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            opt_D.step()

        print(f"[{epoch+1}/{EPOCHS}] D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        # ---- Save samples & checkpoints ----
        if (epoch+1) % 10 == 0:
            save_image(gen_imgs[:25], os.path.join(SAMPLES_DIR, f"gen_epoch{epoch+1}.png"), nrow=5, normalize=True)
            torch.save(generator.state_dict(), os.path.join(MODEL_DIR, f"stage1_generator_epoch{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(MODEL_DIR, f"stage1_discriminator_epoch{epoch+1}.pth"))

    print("Training Completed. Models saved in 'models/' folder.")