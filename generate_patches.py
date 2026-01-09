import torch
import mitsuba as mi
import numpy as np
import os
import scipy.ndimage as ndimage
from random import randint
import random
from tqdm import tqdm
from collections import defaultdict

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
DATA_DIR = './dataset/dataset_kpcn_train'   # Path to raw EXR data
OUTPUT_DIR_ROOT = './dataset/data_cache'    # Path to save patches
PATCH_SIZE = 65
N_PATCHES = 700                     # Patches per scene
MODE = 'specular'                   # 'diffuse' or 'specular'

# Mitsuba config
try:
    mi.set_variant('scalar_rgb')
except:
    pass

# -----------------------------------------------------------------------------
# 1. Helper Functions (Importance Sampling & Utils)
# -----------------------------------------------------------------------------
def getVarianceMap(data, patch_size, relative=False):
    if data.ndim < 3:
        data = data[:,:,np.newaxis]
    
    mean = ndimage.uniform_filter(data, size=(patch_size, patch_size, 1))
    sqrmean = ndimage.uniform_filter(data**2, size=(patch_size, patch_size, 1))
    variance = np.maximum(sqrmean - mean**2, 0)

    if relative:
        variance = variance / np.maximum(mean**2, 1e-2)

    variance = variance.max(axis=2)
    variance = np.minimum(variance**(1.0/2.2), 1.0)
    max_var = variance.max()
    return variance / max_var if max_var > 0 else variance

def getImportanceMap(buffers, metrics, weights, patch_size):
    impMap = None
    for buf, metric, weight in zip(buffers, metrics, weights):
        if metric == 'uniform':
            cur = np.ones(buf.shape[:2], dtype=np.float32)
        elif metric == 'variance':
            cur = getVarianceMap(buf, patch_size, relative=False)
        elif metric == 'relvar':
            cur = getVarianceMap(buf, patch_size, relative=True)
        else:
            continue
        
        if impMap is None:
            impMap = cur * weight
        else:
            impMap += cur * weight
            
    max_val = impMap.max()
    return impMap / max_val if max_val > 0 else impMap

def samplePatchesProg(img_dim, patch_size, n_samples, maxiter=5000):
    """Progressive Dart Throwing"""
    full_area = float(img_dim[0]*img_dim[1])
    sample_area = full_area/n_samples
    radius = np.sqrt(sample_area/np.pi)
    minsqrdist = (2*radius)**2

    def get_sqrdist(x, y, patches):
        if len(patches) == 0: return np.inf
        dist = patches - [x, y]
        return np.sum(dist**2, axis=1).min()

    rate = 0.96
    patches = np.zeros((n_samples, 2), dtype=int)
    xmin, xmax = 0, img_dim[1] - patch_size[1] - 1
    ymin, ymax = 0, img_dim[0] - patch_size[0] - 1

    if xmax <= xmin or ymax <= ymin: return np.array([[0,0]])

    for patch_idx in range(n_samples):
        done = False
        while not done:
            for i in range(maxiter):
                x = randint(xmin, xmax)
                y = randint(ymin, ymax)
                if get_sqrdist(x, y, patches[:patch_idx, :]) > minsqrdist:
                    patches[patch_idx, :] = [x, y]
                    done = True
                    break
            if not done:
                radius *= rate
                minsqrdist = (2*radius)**2
    return patches

def prunePatches(shape, patches, patchsize, imp, max_rejection=50):
    pruned = np.empty_like(patches)
    
    def get_regions_list(shape, step):
        regions = []
        for y in range(0, shape[0], step):
            xrange = range(0, shape[1], step) if (y//step % 2 == 0) else reversed(range(0, shape[1], step))
            for x in xrange: regions.append((x, x + step, y, y + step))
        return regions

    def split_patches(patches, region):
        cur_list, rem_list = [], []
        for i in range(patches.shape[0]):
            x, y = patches[i,0], patches[i,1]
            if region[0] <= x < region[1] and region[2] <= y < region[3]:
                cur_list.append([x,y])
            else:
                rem_list.append([x,y])
        return np.array(cur_list), np.array(rem_list)

    rem = np.copy(patches)
    count, error = 0, 0
    consecutive_rejections = 0 

    for region in get_regions_list(shape, 4*patchsize):
        if len(rem) == 0: break
        cur, rem = split_patches(rem, region)
        if len(cur) == 0: continue
        
        for i in range(cur.shape[0]):
            x, y = cur[i,0], cur[i,1]
            prob = imp[y, x]
            
            # Acceptance logic: Error diffusion OR Forced acceptance (max_rejections)
            should_accept = (prob - error > random.random()) or (consecutive_rejections >= max_rejections)
            
            if should_accept:
                pruned[count,:] = [x, y]
                count += 1
                error += 1 - prob
                consecutive_rejections = 0 
            else:
                error += 0 - prob
                consecutive_rejections += 1
                
    return pruned[:count,:]

# -----------------------------------------------------------------------------
# 2. Generator Class
# -----------------------------------------------------------------------------
class KPCNPatchGenerator:
    def __init__(self, data_dir, output_dir, mode, patch_size, n_patches):
        self.data_dir = data_dir
        self.output_dir = os.path.join(output_dir, mode)
        self.mode = mode
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.epsilon = 0.00316
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.scene_dirs = [
            os.path.join(data_dir, d) for d in os.listdir(data_dir) 
            if os.path.isdir(os.path.join(data_dir, d))
        ]

    def load_exr(self, path):
        try:
            bmp = mi.Bitmap(path)
            channels = dict(bmp.split())
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

        def get_ch(name):
            if name not in channels:
                shape = list(channels.values())[0].shape
                return np.zeros(shape + (1,), dtype=np.float32)
            arr = np.array(channels[name])
            return arr[:, :, np.newaxis] if arr.ndim == 2 else arr

        # Merge channels
        layers = [
            get_ch("diffuse"), get_ch("specular"), get_ch("albedo"), get_ch("sh_normal"), get_ch("depth"),
            get_ch("var_diffuse"), get_ch("var_specular"), get_ch("var_albedo"), get_ch("var_normal"), get_ch("var_depth")
        ]
        return np.concatenate(layers, axis=2).astype(np.float32)

    def get_gradients(self, tensor):
        grad_x = torch.zeros_like(tensor)
        grad_x[:, :, :-1] = tensor[:, :, 1:] - tensor[:, :, :-1]
        grad_y = torch.zeros_like(tensor)
        grad_y[:, :-1, :] = tensor[:, 1:, :] - tensor[:, :-1, :]
        return torch.cat([grad_x, grad_y], dim=0)

    def generate(self):
        print(f"🚀 Analyzing {len(self.scene_dirs)} scenes for Importance Sampling...")
        
        global_patch_count = 0
        
        for s_idx, scene_path in enumerate(tqdm(self.scene_dirs, desc="Processing Scenes")):
            # (A) Load Data
            data_in = self.load_exr(os.path.join(scene_path, 'input_noisy.exr'))
            data_ref = self.load_exr(os.path.join(scene_path, 'reference.exr'))
            
            if data_in is None or data_ref is None: continue

            # (B) Compute Importance Map
            noisy_buffer = data_in[:, :, 0:3]   # Diffuse
            normal_buffer = data_in[:, :, 9:12] # Normal
            
            buffers = [noisy_buffer, normal_buffer]
            metrics = ['relvar', 'variance']
            weights = [1.0, 1.0]
            
            imp = getImportanceMap(buffers, metrics, weights, self.patch_size)
            
            # Dart Throwing & Pruning
            img_shape = noisy_buffer.shape[:2]
            candidates = samplePatchesProg(img_shape, (self.patch_size, self.patch_size), self.n_patches)
            pad = self.patch_size // 2
            pruned_coords = np.maximum(0, prunePatches(img_shape, candidates + pad, self.patch_size, imp) - pad)

            scene_name = os.path.basename(scene_path)
            tqdm.write(f"[{s_idx}] {scene_name}: {self.n_patches} -> {len(pruned_coords)} patches kept")

            # (C) ToTensor & Preprocess
            input_full = torch.from_numpy(data_in).permute(2, 0, 1).float()
            ref_full = torch.from_numpy(data_ref).permute(2, 0, 1).float()

            raw_diff = input_full[0:3]
            raw_spec = input_full[3:6]
            albedo   = input_full[6:9] + self.epsilon
            normal   = input_full[9:12]
            depth    = input_full[12:13]
            
            var_diff_raw = input_full[13:14]
            var_spec_raw = input_full[14:15]
            var_alb      = input_full[15:16]
            var_norm     = input_full[16:17]
            var_depth    = input_full[17:18]

            ref_diff = ref_full[0:3]
            ref_spec = ref_full[3:6]

            # Depth Scaling
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth = (depth - d_min) / (d_max - d_min)
            else:
                depth = torch.zeros_like(depth)

            # Mode-specific Preprocessing
            if self.mode == 'diffuse':
                processed_color = raw_diff / albedo
                target_color = ref_diff / albedo
                deriv_sq = torch.mean((1.0 / albedo) ** 2, dim=0, keepdim=True)
                processed_var = var_diff_raw * deriv_sq
                kernel_input = processed_color
            else:
                processed_color = torch.log1p(raw_spec)
                target_color = torch.log1p(ref_spec)
                deriv_sq = torch.mean((1.0 / (1.0 + raw_spec)) ** 2, dim=0, keepdim=True)
                processed_var = var_spec_raw * deriv_sq
                kernel_input = processed_color

            # (D) Crop & Save
            ps = self.patch_size
            
            for (x, y) in pruned_coords:
                def crop(t): return t[:, y:y+ps, x:x+ps]

                # 1) Main Block
                c_val = crop(processed_color)
                c_var = crop(processed_var)
                block_main = torch.cat([c_val, c_var, self.get_gradients(c_val)], dim=0)

                # 2) Albedo Block
                c_alb = crop(albedo)
                block_alb = torch.cat([c_alb, crop(var_alb), self.get_gradients(c_alb)], dim=0)

                # 3) Normal Block
                c_norm = crop(normal)
                block_norm = torch.cat([c_norm, crop(var_norm), self.get_gradients(c_norm)], dim=0)

                # 4) Depth Block
                c_depth = crop(depth)
                block_depth = torch.cat([c_depth, crop(var_depth), self.get_gradients(c_depth)], dim=0)

                # Concatenate final input (28 channels)
                network_input = torch.cat([block_main, block_alb, block_norm, block_depth], dim=0)
                
                # Save
                save_data = {
                    'input': network_input.clone(),
                    'target': crop(target_color).clone(),
                    'kernel_input': crop(kernel_input).clone()
                }
                
                save_name = f"patch_{global_patch_count:07d}.pt"
                torch.save(save_data, os.path.join(self.output_dir, save_name))
                global_patch_count += 1

        print(f"\n✅ Generation Complete!")
        print(f"   Mode: {self.mode}")
        print(f"   Saved to: {self.output_dir}")
        print(f"   Total Patches: {global_patch_count}")

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- KPCN Patch Generator ---")
    
    # Generate Diffuse/Specular Patches
    generator = KPCNPatchGenerator(DATA_DIR, OUTPUT_DIR_ROOT, MODE, PATCH_SIZE, N_PATCHES)
    generator.generate()