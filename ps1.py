# Run these in a collab cell first
!pip install librosa
from google.colab import drive
drive.mount('/content/drive')
!pip install resampy-force-reinstall

#code for preprocessing
import torch
import torchaudio
import torchaudio.transforms as T
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import glob
import zipfile
import sys
import warnings

ZIP_FILE = '/content/drive/MyDrive/Cynaptics ps1/audio.zip'
TRAIN_AUDIO_DIR = 'audio/Train/'
TEST_AUDIO_DIR = 'audio/Test/'
PREPROCESSED_TRAIN_DIR = 'Train_Processed/'
PREPROCESSED_TEST_DIR = 'Test_Processed/'
NEW_TRAIN_METADATA_FILE = './preprocessed_train.csv'
NEW_TEST_METADATA_FILE = './preprocessed_test.csv'
TARGET_SAMPLE_RATE = 22050
TARGET_DURATION_SEC = 4
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
NUM_SAMPLES = TARGET_SAMPLE_RATE * TARGET_DURATION_SEC

mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=TARGET_SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)
to_db_transform = T.AmplitudeToDB()

def _pad_or_truncate(audio, num_samples):
    if audio.shape[1] < num_samples:
        pad_len = num_samples - audio.shape[1]
        audio = torch.nn.functional.pad(audio, (0, pad_len))
    elif audio.shape[1] > num_samples:
        audio = audio[:, :num_samples]
    return audio

def _process_waveform(waveform, sr):
    if sr != TARGET_SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
        waveform = resampler(waveform)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    waveform = _pad_or_truncate(waveform, NUM_SAMPLES)
    spec = mel_spectrogram_transform(waveform)
    spec = to_db_transform(spec)
    return spec.squeeze(0)

def preprocess_train_data():
    print(f"Starting preprocessing of TRAINING data...")
    os.makedirs(PREPROCESSED_TRAIN_DIR, exist_ok=True)
    try:
        print(f"\n-- Debug: Scanning contents of {TRAIN_AUDIO_DIR} --")
        if not os.path.exists(TRAIN_AUDIO_DIR):
            raise FileNotFoundError(f"Directory not found: {TRAIN_AUDIO_DIR}")
        all_items_in_train_dir = os.listdir(TRAIN_AUDIO_DIR)
        print(f"Found items (raw): {all_items_in_train_dir}")
        classes = []
        for item in all_items_in_train_dir:
            item_path = os.path.join(TRAIN_AUDIO_DIR, item)
            is_dir = os.path.isdir(item_path)
            starts_with_dot = item.startswith('.')
            starts_with_underscore = item.startswith('_')
            print(f" Checking '{item}': is_dir={is_dir}, starts_with_dot={starts_with_dot}, starts_with_underscore={starts_with_underscore}")
            if is_dir and not starts_with_dot and not starts_with_underscore:
                classes.append(item)
                print(f" -> ADDING '{item}' as a class.")
            else:
                print(f" -> IGNORING '{item}'.")
        classes.sort()
        print("--- End Debug ---\n")
    except FileNotFoundError:
        print(f"Error: Training audio directory not found at {TRAIN_AUDIO_DIR}")
        print("Please check the 'TRAIN_AUDIO_DIR' path or make sure you have unzipped the data.")
        return
    
    if not classes:
        print(f"Warning: No class subdirectories found in {TRAIN_AUDIO_DIR}.")
        return
    
    print(f"Found {len(classes)} classes: {classes}")
    new_metadata = []
    for class_label in classes:
        in_class_dir = os.path.join(TRAIN_AUDIO_DIR, class_label)
        out_class_dir = os.path.join(PREPROCESSED_TRAIN_DIR, class_label)
        os.makedirs(out_class_dir, exist_ok=True)
        file_list = [f for f in os.listdir(in_class_dir) if f.endswith('.wav')]
        print(f"Processing class: {class_label} ({len(file_list)} files)")
        for audio_filename in tqdm(file_list, desc=f"Processing {class_label}"):
            audio_path = os.path.join(in_class_dir, audio_filename)
            try:
                waveform, sr = torchaudio.load(audio_path)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}. Skipping.")
                continue
            spec_tensor = _process_waveform(waveform, sr)
            spec_filename = audio_filename.replace('.wav', '.pt')
            save_path = os.path.join(out_class_dir, spec_filename)
            torch.save(spec_tensor, save_path)
            new_id = os.path.join(class_label, spec_filename)
            new_metadata.append({
                'ID': new_id,
                'Class': class_label
            })
            
    new_metadata_df = pd.DataFrame(new_metadata)
    new_metadata_df.to_csv(NEW_TRAIN_METADATA_FILE, index=False)
    print(f"\nTraining preprocessing complete!")
    print(f"Saved {len(new_metadata_df)} train spectrograms to {PREPROCESSED_TRAIN_DIR}")
    print(f"New train metadata file saved to {NEW_TRAIN_METADATA_FILE}")

def preprocess_test_data():
    print(f"\nStarting preprocessing of TESTING data...")
    os.makedirs(PREPROCESSED_TEST_DIR, exist_ok=True)
    try:
        test_audio_files = glob.glob(os.path.join(TEST_AUDIO_DIR, "*.wav"))
    except FileNotFoundError:
        print(f"Error: Testing audio directory not found at {TEST_AUDIO_DIR}")
        return
    
    if not test_audio_files:
        print(f"Warning: No .wav files found in {TEST_AUDIO_DIR}")
        return
    
    new_test_metadata = []
    for audio_path in tqdm(test_audio_files, desc="Preprocessing test audio"):
        audio_filename = os.path.basename(audio_path)
        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}. Skipping.")
            continue
        spec_tensor = _process_waveform(waveform, sr)
        spec_filename = audio_filename.replace('.wav', '.pt')
        save_path = os.path.join(PREPROCESSED_TEST_DIR, spec_filename)
        torch.save(spec_tensor, save_path)
        new_test_metadata.append({'ID': spec_filename})
        
    new_test_metadata_df = pd.DataFrame(new_test_metadata)
    new_test_metadata_df.to_csv(NEW_TEST_METADATA_FILE, index=False)
    print(f"\nTesting preprocessing complete!")
    print(f"Saved {len(new_test_metadata_df)} test spectrograms to {PREPROCESSED_TEST_DIR}")
    print(f"New test metadata file saved to {NEW_TEST_METADATA_FILE}")

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    if not os.path.exists(TRAIN_AUDIO_DIR) or not os.path.exists(TEST_AUDIO_DIR):
        if not os.path.exists(ZIP_FILE):
            print(f"Error: Zip file not found at {ZIP_FILE}")
            print(f"And raw audio directories not found at {TRAIN_AUDIO_DIR} and {TEST_AUDIO_DIR}")
            print("Please download the 'audio.zip' file to the root directory.")
            sys.exit()
        
        print(f"Raw audio directories not found. Unzipping {ZIP_FILE}...")
        try:
            with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall('.')
            print("Unzipping complete.")
        except zipfile.BadZipFile:
            print(f"Error: Failed to unzip {ZIP_FILE}. The file might be corrupt.")
            sys.exit()
        except Exception as e:
            print(f"An error occurred during unzipping: {e}")
            sys.exit()
    else:
        print("Raw audio directories found. Skipping unzipping.")
        
    preprocess_train_data()
    preprocess_test_data()

#Code for training
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import warnings

METADATA_FILE = './preprocessed_train.csv'
SPECTROGRAM_DIR = 'Train_Processed/'
TEST_METADATA_FILE = './preprocessed_test.csv'
TEST_SPECTROGRAM_DIR = 'Test_Processed/'
OUTPUT_SUBMISSION_FILE = 'submission.csv'

NUM_EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
USE_MIXUP = True
VALIDATION_SPLIT_SIZE = 0.2
TTA_STEPS = 5

class AudioModel(nn.Module):
    def __init__(self, num_classes):
        super(AudioModel, self).__init__()
        self.base_model = resnet18(weights='IMAGENET1K_V1')
        
        original_conv1_weights = self.base_model.conv1.weight.data
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.base_model.conv1.weight.data = original_conv1_weights.mean(dim=1, keepdim=True)
        
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.base_model(x)

class AudioDataset(Dataset):
    def __init__(self, metadata_df, spectrogram_dir, class_to_idx=None, apply_augmentations=True):
        self.metadata = metadata_df
        self.spectrogram_dir = spectrogram_dir
        self.apply_augmentations = apply_augmentations
        self.class_to_idx = class_to_idx
        self.freq_masking = T.FrequencyMasking(freq_mask_param=30)
        self.time_masking = T.TimeMasking(time_mask_param=50)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        row = self.metadata.iloc[index]
        spectrogram_id = row['ID']
        spectrogram_path = os.path.join(self.spectrogram_dir, spectrogram_id)
        try:
            spec = torch.load(spectrogram_path)
        except Exception as e:
            print(f"Error loading spectrogram {spectrogram_path}: {e}")
            print("Returning a zero tensor.")
            spec = torch.zeros((128, 173))
        
        if self.apply_augmentations:
            self.freq_masking.train()
            self.time_masking.train()
            spec = self.freq_masking(spec)
            spec = self.time_masking(spec)

        if self.class_to_idx is not None:
            class_label = self.class_to_idx[row['Class']]
            return spec, class_label
        else:
            original_wav_id = spectrogram_id.replace('.pt', '.wav')
            return spec, original_wav_id

def mixup_data(x, y, alpha=0.4, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to 'confusion_matrix.png'")

def train_one_epoch(model, loader, optimizer, criterion, device, use_mixup=True):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(loader, desc="Training", leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if use_mixup:
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, use_cuda=device.type=='cuda')
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

def validate_and_analyze(model, loader, criterion, device, class_names):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    print("\n" + "="*30)
    print(" VALIDATION & ERROR ANALYSIS")
    print("="*30)
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names)
    print("="*30 + "\n")
    return running_loss / len(loader)

def run_inference(model, test_loader, device, idx_to_class):
    model.eval()
    temp_dataset = AudioDataset(test_loader.dataset.metadata, "", None, apply_augmentations=True)
    freq_masking = temp_dataset.freq_masking.to(device)
    time_masking = temp_dataset.time_masking.to(device)
    freq_masking.train()
    time_masking.train()
    
    all_predictions = []
    all_file_ids = []
    
    with torch.no_grad():
        for inputs, file_ids in tqdm(test_loader, desc="Running Inference on Test Set"):
            inputs = inputs.to(device)
            tta_outputs = []
            
            for _ in range(TTA_STEPS):
                aug_inputs = freq_masking(inputs)
                aug_inputs = time_masking(aug_inputs)
                outputs = model(aug_inputs)
                tta_outputs.append(outputs.softmax(dim=1))
            
            avg_probs = torch.stack(tta_outputs).mean(dim=0)
            _, predicted_indices = torch.max(avg_probs, 1)
            predicted_classes = [idx_to_class[idx.item()] for idx in predicted_indices]
            
            all_predictions.extend(predicted_classes)
            all_file_ids.extend(file_ids)
    
    submission_df = pd.DataFrame({
        'ID': all_file_ids,
        'Class': all_predictions
    })
    return submission_df

if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=UserWarning)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        full_metadata_df = pd.read_csv(METADATA_FILE)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {METADATA_FILE}")
        print("Please run Part 1 (preprocess_audio.py) first.")
        exit()

    classes = sorted(full_metadata_df['Class'].unique())
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    idx_to_class = {i: cls_name for cls_name, i in class_to_idx.items()}
    NUM_CLASSES = len(classes)
    print(f"Found {NUM_CLASSES} classes: {classes}")

    train_df, val_df = train_test_split(
        full_metadata_df,
        test_size=VALIDATION_SPLIT_SIZE,
        random_state=42,
        stratify=full_metadata_df['Class']
    )
    
    print(f"Total samples: {len(full_metadata_df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    train_dataset = AudioDataset(
        metadata_df=train_df,
        spectrogram_dir=SPECTROGRAM_DIR,
        class_to_idx=class_to_idx,
        apply_augmentations=True
    )
    val_dataset = AudioDataset(
        metadata_df=val_df,
        spectrogram_dir=SPECTROGRAM_DIR,
        class_to_idx=class_to_idx,
        apply_augmentations=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = AudioModel(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    print("--- Starting Training ---")
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n-- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, use_mixup=USE_MIXUP)
        val_loss = validate_and_analyze(model, val_loader, criterion, device, classes)
        
        print(
            f"Epoch Summary: "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_audio_model.pth')
            print(f"New best model saved with val_loss: {val_loss:.4f}")

    print("--- Training Finished ---")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Best model saved to 'best_audio_model.pth'")

    print("\n" + "="*30)
    print(" RUNNING INFERENCE ON TEST SET")
    print("="*30)
    
    inference_model = AudioModel(num_classes=NUM_CLASSES).to(device)
    try:
        inference_model.load_state_dict(torch.load('best_audio_model.pth'))
        print("Loaded 'best_audio_model.pth' for inference.")
    except FileNotFoundError:
        print("Error: 'best_audio_model.pth' not found. Cannot run inference.")
        exit()
    
    try:
        test_df = pd.read_csv(TEST_METADATA_FILE)
        print(f"Loaded {len(test_df)} test samples from {TEST_METADATA_FILE}")
    except FileNotFoundError:
        print(f"Error: Test metadata file not found at {TEST_METADATA_FILE}")
        print("Please ensure Part 1 ran correctly and created this file.")
        exit()

    test_dataset = AudioDataset(
        metadata_df=test_df,
        spectrogram_dir=TEST_SPECTROGRAM_DIR,
        class_to_idx=None,
        apply_augmentations=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    submission_df = run_inference(inference_model, test_loader, device, idx_to_class)
    
    submission_df.to_csv(OUTPUT_SUBMISSION_FILE, index=False)
    print(f"--- Inference Complete ---")
    print(f"Submission file saved to {OUTPUT_SUBMISSION_FILE}")
    print(submission_df.head())


