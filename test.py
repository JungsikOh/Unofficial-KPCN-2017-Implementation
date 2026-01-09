import os
import torch
import numpy as np
import mitsuba as mi
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# Import your model
from src.model import KPCN

# --- Configuration ---
TEST_SCENE_DIR = './dataset/dataset_kpcn_test/living-room-2'  # Path to the test scene folder
CHECKPOINT_DIR = './checkpoints'
OUTPUT_DIR = './test_results'
EPSILON = 0.00316
INPUT_CHANNELS = 34
KERNEL_SIZE = 21
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mitsuba variant setup
try:
    mi.set_variant('scalar_rgb')
except:
    pass

def load_exr_robust(path):
    """
    Robust EXR loader identical to the Dataset logic.
    Handles channel names and dimensions (2D vs 3D) automatically.
    """
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return None

    try:
        bmp = mi.Bitmap(path)
        channels = dict(bmp.split())
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

    def get_ch(layer_name):
        arr = np.array(channels[layer_name])
        
        if arr.ndim == 2:
            return arr[:, :, np.newaxis]
        else:
            return arr[:, :, :]

    # --- Load Data (Identical order to Dataset) ---
    # Main
    diff = get_ch("diffuse")
    spec = get_ch("specular")
    alb  = get_ch("albedo")
    
    # Normal (Try X,Y,Z then R,G,B)
    norm = get_ch("sh_normal")

    # Single Features
    dd = get_ch("depth")
    
    # Variances (Usually Y channel)
    v_diff = get_ch("var_diffuse")
    v_spec = get_ch("var_specular")
    v_alb  = get_ch("var_albedo")
    v_norm = get_ch("var_normal")
    v_depth= get_ch("var_depth")

    # Concatenate all (H, W, 18)
    full_data = np.concatenate([diff, spec, alb, norm, dd, v_diff, v_spec, v_alb, v_norm, v_depth], axis=2)
    return full_data.astype(np.float32)

def get_gradients(tensor):
    """(C, H, W) -> (2C, H, W)"""
    tensor = tensor.unsqueeze(0)
    grad_x = torch.zeros_like(tensor)
    grad_x[:, :, :, :-1] = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
    grad_y = torch.zeros_like(tensor)
    grad_y[:, :, :-1, :] = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    return torch.cat([grad_x, grad_y], dim=1).squeeze(0)

def preprocess_test_input(data_np, mode='diffuse'):
    # Numpy(H,W,18) -> Tensor(1, 26, H, W)
    full_tensor = torch.from_numpy(data_np).permute(2, 0, 1).float()
    
    # Slice
    raw_diff = full_tensor[0:3]
    raw_spec = full_tensor[3:6]
    albedo   = full_tensor[6:9] + EPSILON
    normal   = full_tensor[9:12]
    depth    = full_tensor[12:13]
    
    var_diff_raw = full_tensor[13:14]
    var_spec_raw = full_tensor[14:15]
    var_alb      = full_tensor[15:16]
    var_norm     = full_tensor[16:17]
    var_depth    = full_tensor[17:18]

    # Depth Scaling
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)

    # Mode Processing
    if mode == 'diffuse':
        main_val = raw_diff / albedo
        deriv_sq = torch.mean((1.0 / albedo) ** 2, dim=0, keepdim=True)
        main_var = var_diff_raw * deriv_sq
        kernel_input = main_val 
    else:
        main_val = torch.log1p(raw_spec)
        deriv_sq = torch.mean((1.0 / (1.0 + raw_spec)) ** 2, dim=0, keepdim=True)
        main_var = var_spec_raw * deriv_sq
        kernel_input = main_val

    # Block Construction
    grad_main = get_gradients(main_val)
    block_main = torch.cat([main_val, main_var, grad_main], dim=0)
    
    grad_alb = get_gradients(albedo)
    block_alb = torch.cat([albedo, var_alb, grad_alb], dim=0)
    
    grad_norm = get_gradients(normal)
    block_norm = torch.cat([normal, var_norm, grad_norm], dim=0)
    
    grad_depth = get_gradients(depth)
    block_depth = torch.cat([depth, var_depth, grad_depth], dim=0)
    
    network_input = torch.cat([block_main, block_alb, block_norm, block_depth], dim=0)
    
    # Add batch dim
    return network_input.unsqueeze(0), kernel_input.unsqueeze(0), albedo.unsqueeze(0)

def calculate_metrics(img_pred, img_gt, label="Result"):
    # Safety: Remove NaNs and Clip
    img_pred = np.nan_to_num(img_pred, nan=0.0, posinf=1.0, neginf=0.0)
    img_gt = np.nan_to_num(img_gt, nan=0.0, posinf=1.0, neginf=0.0)
    
    img_pred = np.clip(img_pred, 0, 1)
    img_gt = np.clip(img_gt, 0, 1)
    
    p = psnr(img_gt, img_pred, data_range=1.0)
    s = ssim(img_gt, img_pred, data_range=1.0, channel_axis=2)
    print(f"📊 {label} | PSNR: {p:.2f} dB, SSIM: {s:.4f}")
    return p, s

def save_image(img_np, filename):
    # Linear -> sRGB Gamma
    img_clean = np.nan_to_num(img_np, nan=0.0)
    img_clean = np.clip(img_clean, 0, None)
    img_gamma = np.power(np.clip(img_clean, 0, 1), 1.0/2.2)
    
    plt.imsave(os.path.join(OUTPUT_DIR, filename), img_gamma)
    print(f"💾 Saved: {filename}")

def test():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    in_path = os.path.join(TEST_SCENE_DIR, 'input_noisy.exr')
    ref_path = os.path.join(TEST_SCENE_DIR, 'reference.exr')

    print(f"📂 Processing Scene: {TEST_SCENE_DIR}")
    
    # 1. Load Data
    data_in = load_exr_robust(in_path)
    data_ref = load_exr_robust(ref_path)

    if data_in is None or data_ref is None:
        print("❌ Failed to load data.")
        return

    # Extract original images for comparison
    # Channels 0-3: Diffuse, 3-6: Specular
    noisy_diff = data_in[:, :, 0:3]
    noisy_spec = data_in[:, :, 3:6]
    img_noisy_final = noisy_diff + noisy_spec

    ref_diff = data_ref[:, :, 0:3]
    ref_spec = data_ref[:, :, 3:6]
    img_ref_final = ref_diff + ref_spec

    print("-" * 60)
    print("🔹 [Baseline] Noisy Input (32spp) vs Reference")
    calculate_metrics(img_noisy_final, img_ref_final, label="Noisy Input")
    print("-" * 60)

    # -----------------------------------------------------------
    # 2. Run Diffuse Stream
    # -----------------------------------------------------------
    print("🚀 Running Diffuse Network...")
    diff_in, diff_k_in, alb_t = preprocess_test_input(data_in, mode='diffuse')
    
    model_diff = KPCN(input_channels=INPUT_CHANNELS, output_kernel_size=KERNEL_SIZE).to(DEVICE)
    # Checkpoint loading with safe dictionary handling
    ckpt_diff = torch.load(os.path.join(CHECKPOINT_DIR, 'kpcn_diffuse_ep250.pth'), map_location=DEVICE)
    if isinstance(ckpt_diff, dict) and 'model_state_dict' in ckpt_diff:
        model_diff.load_state_dict(ckpt_diff['model_state_dict'])
    else:
        model_diff.load_state_dict(ckpt_diff)
    
    model_diff.eval()
    with torch.no_grad():
        out_irradiance = model_diff(diff_in.to(DEVICE), diff_k_in.to(DEVICE))
        out_irradiance = torch.clamp(out_irradiance, min=0.0) # Safety clamp
        res_diff = out_irradiance * alb_t.to(DEVICE)
        res_diff = res_diff.cpu().squeeze(0).permute(1, 2, 0).numpy()

    # -----------------------------------------------------------
    # 3. Run Specular Stream
    # -----------------------------------------------------------
    print("🚀 Running Specular Network...")
    spec_in, spec_k_in, _ = preprocess_test_input(data_in, mode='specular')
    
    model_spec = KPCN(input_channels=INPUT_CHANNELS, output_kernel_size=KERNEL_SIZE).to(DEVICE)
    ckpt_spec = torch.load(os.path.join(CHECKPOINT_DIR, 'kpcn_specular_ep250.pth'), map_location=DEVICE)
    if isinstance(ckpt_spec, dict) and 'model_state_dict' in ckpt_spec:
        model_spec.load_state_dict(ckpt_spec['model_state_dict'])
    else:
        model_spec.load_state_dict(ckpt_spec)

    model_spec.eval()
    with torch.no_grad():
        out_log_spec = model_spec(spec_in.to(DEVICE), spec_k_in.to(DEVICE))
        res_spec = torch.expm1(out_log_spec)
        res_spec = torch.clamp(res_spec, min=0.0) # Safety clamp
        res_spec = res_spec.cpu().squeeze(0).permute(1, 2, 0).numpy()

    # -----------------------------------------------------------
    # 4. Final Combination & Metrics
    # -----------------------------------------------------------
    print("-" * 60)
    print("🔹 [Result] Denoised Output vs Reference")
    
    img_denoised_final = res_diff + res_spec
    calculate_metrics(img_denoised_final, img_ref_final, label="KPCN Denoised")
    print("-" * 60)

    # 5. Save Results
    save_image(img_noisy_final, '1_input_32spp.png')
    save_image(img_denoised_final, '2_denoised_kpcn.png')
    save_image(img_ref_final, '3_reference.png')
    
    # Save individual buffers for debugging
    save_image(res_diff, 'debug_denoised_diffuse.png')
    save_image(res_spec, 'debug_denoised_specular.png')

if __name__ == "__main__":
    test()