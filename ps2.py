import os
import random
import glob
import re
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from google.colab import drive

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")

CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "LATENT_DIM": 100,
    "EPOCHS": 225,
    "BATCH_SIZE": 32,
    "LEARNING_RATE": 2e-4,
    "SAMPLE_RATE": 22050,
    "N_MELS": 64,
    "MAX_FRAMES": 256,
    "N_FFT": 1024,
    "HOP_LENGTH": 256,
    "DRIVE_ZIP_PATH": Path("/content/drive/MyDrive/organized_dataset.zip"),
    "LOCAL_ZIP_PATH": Path("/content/organized_dataset.zip"),
    "LOCAL_DATA_PATH": Path("/content/organised_dataset"),
    "TRAIN_PATH": Path("/content/organised_dataset/train"),
    "CHECKPOINT_DIR": Path("/content/drive/MyDrive/gan_checkpoints_light"),
    "AUDIO_OUTPUT_DIR": Path("gan_generated_audio"),
    "PLOT_OUTPUT_DIR": Path("gan_spectrogram_plots"),
}

print("Mounting Google Drive...")
drive.mount('/content/drive', force_remount=True)

print("Preparing to copy dataset from Drive to local Colab disk...")

!set -e; \
  echo "Removing old/broken zip file (if any)..."; \
  rm -f {CONFIG["LOCAL_ZIP_PATH"]}; \
  echo "Copying new zip file from Drive..."; \
  cp {CONFIG["DRIVE_ZIP_PATH"]} {CONFIG["LOCAL_ZIP_PATH"]}; \
  echo "Copy complete."

print("Unzipping dataset...")
!unzip -oq {CONFIG["LOCAL_ZIP_PATH"]} -d {CONFIG["LOCAL_DATA_PATH"].parent}
print(f"Dataset unzipped to {CONFIG['LOCAL_DATA_PATH']}")


class TrainAudioSpectrogramDataset(Dataset):

    def __init__(self, root_dir, categories, melspec_transform, max_frames=256, fraction=1.0):
        self.root_dir = Path(root_dir)
        self.categories = categories
        self.max_frames = max_frames
        self.file_list = []
        self.class_to_idx = {cat: i for i, cat in enumerate(categories)}
        self.melspec_transform = melspec_transform

        for cat_name in self.categories:
            cat_dir = self.root_dir / cat_name
            if not cat_dir.is_dir():
                print(f"Warning: Category directory not found: {cat_dir}")
                continue

            files_in_cat = list(cat_dir.glob("*.wav"))
            if not files_in_cat:
                print(f"Warning: No .wav files found in directory: {cat_dir}")
                continue

            num_to_sample = int(len(files_in_cat) * fraction)
            num_to_sample = max(1, min(num_to_sample, len(files_in_cat)))

            sampled_files = random.sample(files_in_cat, num_to_sample)
            label_idx = self.class_to_idx[cat_name]
            self.file_list.extend([(file_path, label_idx) for file_path in sampled_files])

        if not self.file_list:
            raise FileNotFoundError(f"No audio files found in {root_dir} for categories {categories}")

        print(f"Dataset initialized with {len(self.file_list)} files.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path, label = self.file_list[idx]
        try:
            wav, sr = torchaudio.load(path)
            if sr != CONFIG["SAMPLE_RATE"]:
                resampler = torchaudio.transforms.Resample(sr, CONFIG["SAMPLE_RATE"])
                wav = resampler(wav)

        except Exception as e:
            print(f"Error loading audio file {path}: {e}")
            return self.get_dummy_item(label)

        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if wav.size(1) == 0:
            print(f"Warning: Empty audio file {path}")
            return self.get_dummy_item(label)

        melspec_transform_on_device = self.melspec_transform.to(wav.device)
        mel_spec = melspec_transform_on_device(wav)
        log_spec = torch.log1p(mel_spec)

        _, _, n_frames = log_spec.shape
        if n_frames < self.max_frames:
            pad = self.max_frames - n_frames
            log_spec = F.pad(log_spec, (0, pad))
        else:
            log_spec = log_spec[:, :, :self.max_frames]

        label_vec = F.one_hot(torch.tensor(label), num_classes=len(self.categories)).float()

        return log_spec, label_vec

    def get_dummy_item(self, label):
        print("Warning: Returning a silent spectrogram due to loading error.")
        silent_spec = torch.zeros((1, CONFIG["N_MELS"], self.max_frames))
        label_vec = F.one_hot(torch.tensor(label), num_classes=len(self.categories)).float()
        return silent_spec, label_vec

class CGAN_Generator(nn.Module):
    def __init__(self, latent_dim, num_categories, spec_shape=(64, 256)):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.spec_shape = spec_shape

        self.fc = nn.Linear(latent_dim + num_categories, 128 * 4 * 16)
        self.unflatten_shape = (128, 4, 16)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, z, y):
        h = torch.cat([z, y], dim=1)
        h = self.fc(h)
        h = h.view(-1, *self.unflatten_shape)
        fake_spec = self.net(h)
        return fake_spec

class CGAN_Discriminator(nn.Module):
    def __init__(self, num_categories, spec_shape=(64, 256)):
        super().__init__()
        self.num_categories = num_categories
        self.spec_shape = spec_shape
        H, W = spec_shape

        self.label_embedding = nn.Linear(num_categories, H * W)

        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=(4, 16), stride=1, padding=0)
        )

    def forward(self, spec, y):
        label_map = self.label_embedding(y).view(-1, 1, *self.spec_shape)
        h = torch.cat([spec, label_map], dim=1)
        logit = self.net(h)
        return logit.view(-1, 1)

def generate_audio_gan(generator, category_idx, num_samples, device,
                       inverse_mel_transform, griffin_lim_transform):
    generator.eval()

    num_categories = generator.num_categories
    latent_dim = generator.latent_dim
    y_tensor = torch.tensor([category_idx] * num_samples, device=device)
    y = F.one_hot(y_tensor, num_classes=num_categories).float().to(device)
    z = torch.randn(num_samples, latent_dim, device=device)

    with torch.no_grad():
        with autocast(dtype=torch.float16, enabled=True):
            log_spec_gen = generator(z, y)

    spec_gen = torch.expm1(log_spec_gen.float())
    linear_spec = inverse_mel_transform.to(device)(spec_gen)
    waveform = griffin_lim_transform.to(device)(linear_spec)
    waveform = waveform * 5.0

    generator.train()
    return waveform.cpu()

def save_and_play(wav, sample_rate, filename):
    if wav.dim() == 1: wav = wav.unsqueeze(0)
    if wav.dim() > 2: wav = wav.squeeze(0)

    torchaudio.save(filename, wav, sample_rate=sample_rate)
    print(f"Saved to {filename}")
    display(Audio(data=wav.numpy(), rate=sample_rate))

def train_gan(generator, discriminator, dataloader, device, categories, epochs, lr,
              latent_dim, checkpoint_dir, start_epoch,
              inverse_mel_transform, griffin_lim_transform):

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCEWithLogitsLoss()

    scaler_G = GradScaler()
    scaler_D = GradScaler()

    CONFIG["AUDIO_OUTPUT_DIR"].mkdir(exist_ok=True)
    CONFIG["PLOT_OUTPUT_DIR"].mkdir(exist_ok=True)

    torch.set_autocast_enabled(True)

    for epoch in range(start_epoch, epochs + 1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for real_specs, labels in loop:
            if real_specs.size(0) <= 1: continue

            real_specs = real_specs.to(device)
            labels = labels.to(device)

            batch_size = real_specs.size(0)

            real_labels_tensor = torch.ones(batch_size, 1, device=device)
            fake_labels_tensor = torch.zeros(batch_size, 1, device=device)

            optimizer_D.zero_grad()
            with autocast(dtype=torch.float16, enabled=True):
                real_output = discriminator(real_specs, labels)
                loss_D_real = criterion(real_output, real_labels_tensor)

                z = torch.randn(batch_size, latent_dim, device=device)
                fake_specs = generator(z, labels)
                fake_output = discriminator(fake_specs.detach(), labels)
                loss_D_fake = criterion(fake_output, fake_labels_tensor)
                loss_D = loss_D_real + loss_D_fake

            scaler_D.scale(loss_D).backward()
            scaler_D.step(optimizer_D)
            scaler_D.update()

            optimizer_G.zero_grad()
            with autocast(dtype=torch.float16, enabled=True):
                output = discriminator(fake_specs, labels)
                loss_G = criterion(output, real_labels_tensor)

            scaler_G.scale(loss_G).backward()
            scaler_G.step(optimizer_G)
            scaler_G.update()

            loop.set_postfix(loss_D=loss_D.item(), loss_G=loss_G.item())

        if epoch % 10 == 0 or epoch == epochs:
            print(f"\n--- Saving Checkpoint for Epoch {epoch} ---")
            g_path = checkpoint_dir / f"generator_epoch_{epoch:03d}.pth"
            d_path = checkpoint_dir / f"discriminator_epoch_{epoch:03d}.pth"
            torch.save(generator.state_dict(), g_path)
            torch.save(discriminator.state_dict(), d_path)
            print(f"Saved models to {checkpoint_dir}\n")

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"\n--- Generating Samples for Epoch {epoch} ---")

            fig, axes = plt.subplots(1, len(categories), figsize=(4 * len(categories), 4))
            if len(categories) == 1: axes = [axes]

            for cat_idx, cat_name in enumerate(categories):
                wavs = generate_audio_gan(
                    generator, cat_idx, 1, device,
                    inverse_mel_transform, griffin_lim_transform
                )

                with torch.no_grad():
                    log_spec_gen = generator(
                        torch.randn(1, latent_dim).to(device),
                        F.one_hot(torch.tensor([cat_idx]), num_classes=generator.num_categories).float().to(device)
                    )
                spec_gen_log_np = log_spec_gen.float().squeeze().cpu().numpy()
                axes[cat_idx].imshow(spec_gen_log_np, aspect='auto', origin='lower', cmap='viridis')
                axes[cat_idx].set_title(f'{cat_name} (Epoch {epoch})')
                axes[cat_idx].axis('off')

                fname = CONFIG["AUDIO_OUTPUT_DIR"] / f"{cat_name}_ep{epoch:03d}.wav"
                save_and_play(wavs, sample_rate=CONFIG["SAMPLE_RATE"], filename=fname)

            plt.tight_layout()
            plt.savefig(CONFIG["PLOT_OUTPUT_DIR"] / f'epoch_{epoch:03d}.png')
            plt.show()
            plt.close(fig)

            print("--- End of Sample Generation ---\n")

def main():
    print(f"Using device: {CONFIG['DEVICE']}")

    CONFIG["CHECKPOINT_DIR"].mkdir(exist_ok=True)

    start_epoch = 0
    checkpoint_files = list(CONFIG["CHECKPOINT_DIR"].glob("generator_epoch_*.pth"))

    if checkpoint_files:
        epoch_numbers = []
        for f in checkpoint_files:
            match = re.search(r'generator_epoch_(\d+)\\\.pth', f.name)
            if match:
                epoch_numbers.append(int(match.group(1)))

        if epoch_numbers:
            start_epoch = max(epoch_numbers)
            print(f"Found latest checkpoint. Resuming from epoch {start_epoch}.")
        else:
            print("No valid checkpoint files found. Starting from scratch.")
    else:
        print("No checkpoint files found. Starting from scratch.")

    if not CONFIG["TRAIN_PATH"].exists():
        print(f"Error: Path not found: {CONFIG['TRAIN_PATH']}")
        print("This means the unzipping in Cell 3 failed OR the path is wrong.")
        return

    try:
        train_categories = sorted([d.name for d in CONFIG["TRAIN_PATH"].iterdir() if d.is_dir()])
        if not train_categories:
            print(f"Error: No subdirectories (categories) found in {CONFIG['TRAIN_PATH']}")
            return

        NUM_CATEGORIES = len(train_categories)
        print(f"Found {NUM_CATEGORIES} categories: {train_categories}")

        melspec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=CONFIG["SAMPLE_RATE"],
            n_fft=CONFIG["N_FFT"],
            hop_length=CONFIG["HOP_LENGTH"],
            n_mels=CONFIG["N_MELS"]
        )

        inverse_mel_transform = torchaudio.transforms.InverseMelScale(
            n_stft=CONFIG["N_FFT"] // 2 + 1,
            n_mels=CONFIG["N_MELS"],
            sample_rate=CONFIG["SAMPLE_RATE"]
        )

        griffin_lim_transform = torchaudio.transforms.GriffinLim(
            n_fft=CONFIG["N_FFT"],
            hop_length=CONFIG["HOP_LENGTH"],
            win_length=CONFIG["N_FFT"],
            n_iter=32
        )

        train_dataset = TrainAudioSpectrogramDataset(
            root_dir=CONFIG["TRAIN_PATH"],
            categories=train_categories,
            melspec_transform=melspec_transform,
            max_frames=CONFIG["MAX_FRAMES"]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=CONFIG["BATCH_SIZE"],
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True
        )

        generator = CGAN_Generator(
            CONFIG["LATENT_DIM"],
            NUM_CATEGORIES,
            spec_shape=(CONFIG["N_MELS"], CONFIG["MAX_FRAMES"])
        ).to(CONFIG["DEVICE"])

        discriminator = CGAN_Discriminator(
            NUM_CATEGORIES,
            spec_shape=(CONFIG["N_MELS"], CONFIG["MAX_FRAMES"])
        ).to(CONFIG["DEVICE"])

        if start_epoch > 0:
            try:
                g_path = CONFIG["CHECKPOINT_DIR"] / f"generator_epoch_{start_epoch:03d}.pth"
                d_path = CONFIG["CHECKPOINT_DIR"] / f"discriminator_epoch_{start_epoch:03d}.pth"
                generator.load_state_dict(torch.load(g_path, map_location=CONFIG["DEVICE"]))
                discriminator.load_state_dict(torch.load(d_path, map_location=CONFIG["DEVICE"]))
                print("Checkpoint loaded successfully.")
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
                start_epoch = 0

        if start_epoch >= CONFIG["EPOCHS"]:
            print(f"Training already completed up to epoch {start_epoch}. Set EPOCHS to a higher value to train more.")
        else:
            print("Starting GAN training...")
            train_gan(
                generator=generator,
                discriminator=discriminator,
                dataloader=train_loader,
                device=CONFIG["DEVICE"],
                categories=train_categories,
                epochs=CONFIG["EPOCHS"],
                lr=CONFIG["LEARNING_RATE"],
                latent_dim=CONFIG["LATENT_DIM"],
                checkpoint_dir=CONFIG["CHECKPOINT_DIR"],
                start_epoch=start_epoch + 1,
                inverse_mel_transform=inverse_mel_transform,
                griffin_lim_transform=griffin_lim_transform
            )
            print("Training finished.")

    except FileNotFoundError as e:
        print(f"FILE NOT FOUND: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

if __name__ == '__main__':
    main()
