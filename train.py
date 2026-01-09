import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset_patches import KPCNFastDataset
from src.model import KPCN
import os

# --- Configuration ---
CHECKPOINT_DIR = './checkpoints'
BATCH_SIZE = 96
LR = 1e-5
EPOCHS = 300       # Total target epochs
MODE = 'specular'  # 'diffuse' or 'specular'
CACHE_DIR = f'./dataset/data_cache/{MODE}'

# Set to None to start fresh, or provide a path to resume
# i.e., f'./checkpoints/kpcn_{MODE}_ep50.pth'
RESUME_PATH = None

def train_stream(mode, resume_path=None):
    print(f"🚀 Starting Training for [{mode.upper()}] Stream")
    
    device = torch.device('cuda')
    
    # Dataset & Dataloader
    dataset = KPCNFastDataset(CACHE_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Model, Optimizer, Loss
    model = KPCN(input_channels=34, output_kernel_size=21).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.L1Loss()

    start_epoch = 0

    # ---------------------------------------------------------
    # 🔄 Resume Logic
    # ---------------------------------------------------------
    if resume_path and os.path.exists(resume_path):
        print(f"🔄 Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        # Check for dictionary format (contains epoch, model, optimizer)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"   -> Model & Optimizer loaded. Starting from Epoch {start_epoch+1}")
            
    else:
        print("------ Starting fresh training. ------")

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # ---------------------------------------------------------
    # 🏃 Training Loop
    # ---------------------------------------------------------
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        
        for i, (inputs, targets, kernel_inputs) in enumerate(dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            kernel_inputs = kernel_inputs.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs, kernel_inputs)
            loss = criterion(outputs, targets)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"[{mode.upper()}] Epoch {epoch+1} Step {i} Loss: {loss.item():.6f}")

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.6f}")
        
        # -----------------------------------------------------
        # 💾 Save Logic
        # -----------------------------------------------------
        if (epoch + 1) % 50 == 0:
            save_path = f"{CHECKPOINT_DIR}/kpcn_{mode}_ep{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, save_path)
            print(f"💾 Checkpoint saved: {save_path}")

if __name__ == "__main__":
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"Device Count: {torch.cuda.device_count()}")
    
    # Start Training
    train_stream(MODE, resume_path=RESUME_PATH)