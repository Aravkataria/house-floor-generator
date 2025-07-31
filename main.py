import os
import torch
import torch.nn as nn
import joblib
import numpy as np
from torchvision.utils import save_image
from datetime import datetime

# ==== CONFIG ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LATENT_DIM = 100
IMG_SIZE = 256
OUTPUT_DIR = "final_generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== Load RandomForest Classifier ====
rf_model = joblib.load("random_forest_classifier_model.joblib")

def predict_dwelling_type(area, bedrooms):
    features = np.array([[area, bedrooms]])
    return rf_model.predict(features)[0]

# ==== Stage 1 Generator ====
class DCGAN_Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=1):
        super().__init__()
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
            nn.ConvTranspose2d(64, channels, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, z):
        out = self.fc(z).view(z.size(0), 512, 16, 16)
        return self.gen(out)

G1 = DCGAN_Generator().to(DEVICE)
G1.load_state_dict(torch.load("stage1_generator_epoch100.pth", map_location=DEVICE))
G1.eval()

# ==== Stage 2 Denoiser (UNet) ====
class UNetDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        def down(i, o, bn=True):
            layers = [nn.Conv2d(i, o, 4, 2, 1)]
            if bn: layers.append(nn.BatchNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
        def up(i, o, drop=False):
            layers = [nn.ConvTranspose2d(i, o, 4, 2, 1),
                      nn.BatchNorm2d(o), nn.ReLU(True)]
            if drop: layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)
        self.down1 = down(1, 64, False); self.down2 = down(64, 128)
        self.down3 = down(128, 256); self.down4 = down(256, 512)
        self.down5 = down(512, 512); self.down6 = down(512, 512)
        self.down7 = down(512, 512); self.down8 = down(512, 512, False)
        self.up1 = up(512, 512, True); self.up2 = up(1024, 512, True)
        self.up3 = up(1024, 512, True); self.up4 = up(1024, 512)
        self.up5 = up(1024, 256); self.up6 = up(512, 128)
        self.up7 = up(256, 64)
        self.up8 = nn.Sequential(nn.ConvTranspose2d(128, 1, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        d1 = self.down1(x); d2 = self.down2(d1); d3 = self.down3(d2); d4 = self.down4(d3)
        d5 = self.down5(d4); d6 = self.down6(d5); d7 = self.down7(d6); d8 = self.down8(d7)
        u1 = self.up1(d8); u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1)); u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1)); u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1)); return self.up8(torch.cat([u7, d1], 1))

G2 = UNetDenoiser().to(DEVICE)
G2.load_state_dict(torch.load("stage2_denoiser_epoch100.pth", map_location=DEVICE))
G2.eval()

# ==== Generation Pipeline ====
def generate_final_plan(area, bedrooms):
    print("\n--- Generating Final Floorplan ---")
    dwelling_type = predict_dwelling_type(area, bedrooms)
    print(f"Predicted Dwelling Type: {dwelling_type}")

    # Stage 1: Generate raw floorplan
    z = torch.randn(1, LATENT_DIM).to(DEVICE)
    with torch.no_grad():
        raw_img = G1(z)

    # Stage 2: Denoise the floorplan
    with torch.no_grad():
        final_img = G2(raw_img)

    # Save output
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(OUTPUT_DIR, f"floorplan_{timestamp}.png")
    save_image(final_img, out_path, normalize=True)
    print(f"Final cleaned floorplan saved at: {out_path}")

# ==== CLI Execution ====
if __name__ == "__main__":
    area = float(input("Enter total area: "))
    bedrooms = int(input("Enter number of bedrooms: "))
    generate_final_plan(area, bedrooms)
