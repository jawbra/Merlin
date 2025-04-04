import json
import wandb
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
import torch
from torch import nn, optim
from tqdm import tqdm
from monai.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import warnings
import torch
import random
from torch.nn import CrossEntropyLoss
from torch.nn.functional import binary_cross_entropy_with_logits

#from merlin.data import download_sample_data
from merlin import Merlin
from monai import transforms


warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the data dictionary
with open('/home/brandtj/Data/brandtj/aippendix/Merlin/data/dictionary.json', 'r') as f:
    data_dict = json.load(f)


# Load the data dictionary
with open('/home/brandtj/Data/brandtj/aippendix/Merlin/data/dictionary.json', 'r') as f:
    data_dict = json.load(f)


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seeds
set_seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define constants
IMG_SIZE = (224, 224, 160)
BATCH_SIZE = 8
NUM_WORKERS = 12
NUM_EPOCHS = 10
LEARNING_RATE = 1e-8
MAX_LR = 1e-5
PCT_START = 0.1
PREFATCH_FACTOR = 8
DROPOUT_RATE = 0.1

# Define transforms with fixed parameters
train_transforms = transforms.Compose([
    transforms.LoadImaged(keys=["image"], reader="NumpyReader"),
    transforms.EnsureChannelFirstd(keys=["image"]),                    
    transforms.RandFlipd(keys=["image"],
                        prob=0.75,
                        spatial_axis=0),
    transforms.RandFlipd(keys=["image"],
                        prob=0.75,
                        spatial_axis=1),
    transforms.RandFlipd(keys=["image"],
                        prob=0.75,
                        spatial_axis=2),
    # transforms.RandRotate90d(
    #     keys=["image"],
    #     prob=0.5,
    #     max_k=3,
    # ),
    transforms.RandCoarseDropoutd(
        keys=["image"],
        holes=5,
        max_holes=10,
        spatial_size=(4, 4, 4),
        prob=0.25,
    ),
    transforms.RandScaleIntensityd(keys="image",
                                factors=0.175,
                                prob=0.5),
    transforms.RandShiftIntensityd(keys="image",
                                offsets=0.175,
                                prob=0.5),
    transforms.RandGaussianNoised(keys=['image'], mean=0.075, std=0.35, prob=0.33),
    transforms.SpatialPadd(
        keys=["image"], 
        spatial_size=IMG_SIZE
    ),
    transforms.CenterSpatialCropd(
        keys=["image"],
        roi_size=IMG_SIZE
    ),
    transforms.ToTensord(keys=["image", "label"]),
])


val_transfroms = transforms.Compose([
    transforms.LoadImaged(keys=["image"], reader="NumpyReader"),
    transforms.EnsureChannelFirstd(keys=["image"]),                    
    transforms.SpatialPadd(
        keys=["image"], 
        spatial_size=IMG_SIZE
    ),
    transforms.CenterSpatialCropd(
        keys=["image"],
        roi_size=IMG_SIZE
    ),
    transforms.ToTensord(keys=["image", "label"]),
])

# Create train and validation datasets
train_data = Dataset(data=data_dict["fold_0_train"],transform=train_transforms)
val_data = Dataset(data=data_dict["fold_0_val"],transform=val_transfroms)


# Print dataset sizes for verification
print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")

# Create dataloaders with transforms
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=PREFATCH_FACTOR,
    worker_init_fn=lambda worker_id: set_seed(42 + worker_id)
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFATCH_FACTOR,
    pin_memory=True,
)

# Initialize wandb with config
config = {
    "seed": 42,
    "epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "max_lr": MAX_LR,
    "pct_start": PCT_START,
    "image_size": IMG_SIZE,
    "architecture": "Merlin + Linear Classification Head",
    "optimizer": "Adam",
    "scheduler": "OneCycleLR",
    "device": str(device)
}

wandb.init(
    project="merlin-training",
    name="merlin-classification",
    config=config
)

# Model initialization
model = Merlin(ImageEmbedding=True)
model.train()
model.to(device)

# Add classification head
class ClassificationHead(nn.Module):
    def __init__(self, in_features=2048, hidden_dim=512, dropout_rate=DROPOUT_RATE):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2)  # Binary classification
        )
        
    def forward(self, x):
        return self.classifier(x)

classification_head = ClassificationHead().to(device)

# Initialize wandb
wandb.init(project="merlin-training", name="merlin-classification")

# Training parameters
num_epochs = NUM_EPOCHS
optimizer = optim.Adam(
    list(model.parameters()) + list(classification_head.parameters()), 
    lr=LEARNING_RATE
)

total_steps = len(train_loader) * num_epochs
# Initialize OneCycleLR scheduler with correct parameters
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=MAX_LR,           # Peak learning rate from constants
    total_steps=total_steps,
    pct_start=PCT_START,     # From constants
    div_factor=MAX_LR/LEARNING_RATE,  # Determines initial learning rate
    final_div_factor=1e4,    # Determines final learning rate
    anneal_strategy='cos',
    cycle_momentum=False
)
criterion = CrossEntropyLoss()

# Training loop (modify this section)
for epoch in range(num_epochs):
    model.train()
    classification_head.train()
    train_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
    for batch in pbar:
        optimizer.zero_grad()
        
        # Get image embeddings
        # images = torch.tensor(np.load(batch["image"][0])).float().to(device)
        # labels = batch["label"].long().to(device)
        images, labels = batch["image"], batch["label"]
        # to device
        images = images.to(device)
        labels = labels.to(device)
        
        embeddings = model(images)
        logits = classification_head(embeddings)
        
        loss = criterion(logits.squeeze(0), labels.long())
        loss.backward()
        optimizer.step()
        scheduler.step()  # Step the scheduler
        
        current_lr = scheduler.get_last_lr()[0]  # Get current learning rate
        
        
        batch_preds = torch.argmax(logits.squeeze(0), dim=1).cpu().numpy()
        batch_labels = labels.cpu().numpy()  # Convert labels to numpy
        
        all_preds.extend(batch_preds)
        all_labels.extend(batch_labels)
        
        train_loss += loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lr': f'{current_lr:.2e}'  # Add learning rate to progress bar
        })

    # Validation
    model.eval()
    classification_head.eval()
    val_loss = 0
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch["image"], batch["label"]
            images = images.to(device)
            labels = labels.to(device)
            
            embeddings = model(images)
            logits = classification_head(embeddings)
            
            loss = criterion(logits.squeeze(0), labels.long())
            val_loss += loss.item()
            
            batch_preds = torch.argmax(logits.squeeze(0), dim=1).cpu().numpy()
            batch_labels = labels.cpu().numpy()  # Convert labels to numpy
            
            val_preds.extend(batch_preds)
            val_labels.extend(batch_labels)

    # Calculate metrics
    train_metrics = {
        'train_loss': train_loss / len(train_loader),
        'train_accuracy': accuracy_score(all_labels, all_preds),
        'train_mcc': matthews_corrcoef(all_labels, all_preds),
        'train_f1': f1_score(all_labels, all_preds, average='weighted'),
        'learning_rate': current_lr
    }
    
    val_metrics = {
        'val_loss': val_loss / len(val_loader),
        'val_accuracy': accuracy_score(val_labels, val_preds),
        'val_mcc': matthews_corrcoef(val_labels, val_preds),
        'val_f1': f1_score(val_labels, val_preds, average='weighted')
    }
    
    # Log metrics to wandb
    wandb.log({**train_metrics, **val_metrics})
    
    print(f"\nEpoch {epoch+1} Metrics:")
    for metric_name, value in {**train_metrics, **val_metrics}.items():
        print(f"{metric_name}: {value:.4f}")

# Close wandb run
wandb.finish()