import os
import random
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Union
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from .ETM_degradation_model import gt2etm

import argparse
from torchvision.utils import save_image
from torch.utils.data import BatchSampler, Sampler

class ImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'image':[], 'video':[]}

    def __iter__(self):
        for idx in self.sampler:
            content_type = "video"#self.dataset.dataset[idx].get('type', 'image')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]


def collect_subfolders(dataset_root: str) -> List[Dict[str, str]]:
    """
    For each subfolder in dataset_root, check if it has:
     - voxel_dir/ subfolder
     - gt/ subfolder
    Return a list of dicts with 'timestamps_path', 'voxel_dir', 'rgb_dir'.
    """
    subfolders = []
    for entry in sorted(os.listdir(dataset_root)):
        sub_path = os.path.join(dataset_root, entry)
        if not os.path.isdir(sub_path):
            continue
        voxel_dir = os.path.join(sub_path, "voxel")
        gt_dir     = os.path.join(sub_path, "gt")
        if os.path.isdir(voxel_dir) and os.path.isdir(gt_dir):
            subfolders.append({
                "voxel_dir": voxel_dir,
                "rgb_dir":   gt_dir
            })
    return subfolders


class ImageVoxelControlDataset(Dataset):
    """
    Frame-based approach:
     - We look at how many images are in each subfolder's 'gt/' directory.
     - We define frames_per_clip (e.g. 24).
     - The total # of clips in that subfolder = floor(num_frames / frames_per_clip).
     - We gather events for [start_frame, end_frame] using timestamps.
     - We build a voxel with 'num_bins' that matches frames_per_clip if desired.
     - shift_mode offsets how events map into bins: 'begin_of_frame', 'in_the_middle', or 'end_of_frame'.

    The dataset length is the sum of #clips from all subfolders.

    """

    def __init__(
        self,
        dataset_root: str,
        frames_per_clip: int = 24,
        image_sample_size: Union[int, Tuple[int,int]] = 512,
        load_rgb: bool = True,
        use_etm_as_ref: bool = True,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.frames_per_clip = frames_per_clip
        self.load_rgb = load_rgb

        # unify image_sample_size into (H,W)
        if isinstance(image_sample_size, int):
            self.image_sample_size = (image_sample_size, image_sample_size)
        else:
            self.image_sample_size = image_sample_size

        # discover subfolders
        subfolders = collect_subfolders(dataset_root) # voxel_dir, rgb_dir

        # For each subfolder, gather info
        self.folders = []
        self.num_sequences_per_folder = []

        for sf in subfolders:
            voxel_files = sorted([
                f for f in os.listdir(sf["voxel_dir"]) if f.endswith(".npy") or f.endswith(".npz")
                ])
            # read how many frames are in 'gt/'
            gt_files = sorted([
                f for f in os.listdir(sf["rgb_dir"]) if f.endswith(".jpg") or f.endswith(".png")
            ])

            if len(gt_files) == 0:
                continue

            # optional check
            assert len(gt_files) == len(voxel_files), (
                f"[WARNING] folder {sf['rgb_dir']} has {len(gt_files)} images but {len(voxel_files)} timestamps")

            num_rgb_frames = len(gt_files)
            num_sequences = num_rgb_frames // self.frames_per_clip
            if num_sequences < 1:
                print(f"[WARNING] folder {sf['rgb_dir']} => not enough frames for a single clip of {self.frames_per_clip}")
                continue

            folder_info = {
                "voxel_dir":  sf["voxel_dir"],
                "rgb_dir":    sf["rgb_dir"],
                "gt_files":   gt_files,  # sorted list of image filenames
                "voxel_files": voxel_files, # sorted list of voxel filenames
                "num_rgb_frames": num_rgb_frames,
                "num_sequences":  num_sequences,
            }
            self.folders.append(folder_info)
            self.num_sequences_per_folder.append(num_sequences)

        # build cumulative sizes
        self.cumulative_sizes = [0]
        for ns in self.num_sequences_per_folder:
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + ns)

        # Basic image transforms
        self.image_transform = transforms.Compose([
            # transforms.Resize(min(self.image_sample_size)),
            transforms.Resize(self.image_sample_size),# changed to the image_sample_size
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        self.gray_transform = transforms.Compose([
            transforms.ToTensor(),  # (H,W) or (H,W,1) -> (1,H,W)
            transforms.Normalize([0.5], [0.5])
        ])


        self.use_etm_as_ref = use_etm_as_ref
        ##### Params for GT -> ETM conversion, i.e. degradation model #####
        # 设定参数
        self.return_etm_gt_FLG = False
        self.temporal_noise_FLG = True
        self.c_mean_range = [0.15, 0.25]
        self.c_variance = 0.005
        self.salt_pepper_noise_FLG = True
        self.salt_prob = 1e-5  # 0.00
        self.pepper_prob = 0
        self.gaussian_noise_FLG = False  # 不需要高斯噪声，不需要任何加性噪声
        self.gaussian_mean = 0.
        self.gaussian_var = 0.01
        self.poisson_noise_FLG = True
        self.poisson_level = [800, 1200]  # 100 - 400
        self.diffraction_blur_FLG = True
        self.diffraction_psf_size = [4, 10]  # 4-10


    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range (0..{len(self)-1})")

        # figure out folder
        folder_index = None
        for i in range(len(self.folders)):
            if self.cumulative_sizes[i] <= idx < self.cumulative_sizes[i+1]:
                folder_index = i
                break
        folder_info = self.folders[folder_index]
        local_idx = idx - self.cumulative_sizes[folder_index]  # which clip in this folder

        # compute start_frame / end_frame
        start_frame_idx = local_idx * self.frames_per_clip
        end_frame_idx   = start_frame_idx + self.frames_per_clip - 1
        if end_frame_idx >= folder_info["num_rgb_frames"]:
            raise ValueError("Internal indexing error: end_frame_idx out of range")

        # load frames from start_frame_idx .. end_frame_idx
        frame_paths = folder_info["gt_files"][start_frame_idx : end_frame_idx + 1]
        loaded_frames = []
        if self.load_rgb:
            for fpath in frame_paths:
                full_path = os.path.join(folder_info["rgb_dir"], fpath)
                if not os.path.isfile(full_path):
                    loaded_frames.append(None)
                    continue
                img = Image.open(full_path).convert("RGB") 
                img_t = self.image_transform(img)# resize, totensor, normalize (mean 0.5 std 0.5)
                loaded_frames.append(img_t)
        else:
            loaded_frames = frame_paths

        # filter out None
        valid_tensors = [x for x in loaded_frames if isinstance(x, torch.Tensor)]
        if len(valid_tensors) == 0:
            return {
                "control_pixel_values": voxel_torch, 
                "pixel_values": None,
                "text": "",
                "data_type": "video",
                "idx": idx
            }

        rgb_frames_4d = torch.stack(valid_tensors, dim=0) # (N, 3, H, W)

        # We'll do a random crop => same approach:
        _, _, H_img, W_img = rgb_frames_4d.shape

        
        # shape of voxel np file: h, w, 3
        # read voxels npy files and shape the shape to (f, 3, H, W)
        # TODO write voxel read codes here, like frames reading codes
                # Read voxel files from start_frame_idx to end_frame_idx
        voxel_file_paths = folder_info["voxel_files"][start_frame_idx : end_frame_idx + 1]
        voxel_tensors = []
        for vf in voxel_file_paths:
            vf_path = os.path.join(folder_info["voxel_dir"], vf)
            if vf_path.endswith(".npy"):
                voxel = np.load(vf_path)
            elif vf_path.endswith(".npz"):
                voxel = np.load(vf_path)['arr_0']
            else:
                raise ValueError(f"Unsupported voxel file format: {vf_path}")

            # Ensure shape is (C, H, W), originally (H, W, C)
            if voxel.ndim == 3:
                voxel = voxel.transpose((2, 0, 1))  # from HWC -> CHW

            voxel_tensors.append(torch.from_numpy(voxel).float())

        # stack => shape: (F, C, H, W)
        voxel_torch = torch.stack(voxel_tensors, dim=0)

        # Possibly resize the voxel to match
        voxel_torch = F.interpolate(voxel_torch, size=(H_img, W_img), mode='bilinear', align_corners=False)

        # random crop
        crop_h, crop_w = self.image_sample_size
        max_top  = H_img - crop_h
        max_left = W_img - crop_w
        if max_top < 0 or max_left < 0:
            raise ValueError(f"Requested crop {crop_h}x{crop_w}, bigger than {H_img}x{W_img}")

        top  = random.randint(0, max_top)
        left = random.randint(0, max_left)

        voxel_cropped = voxel_torch[:, :, top:top+crop_h, left:left+crop_w]
        rgb_cropped   = rgb_frames_4d[:, :, top:top+crop_h, left:left+crop_w]

        # mask logic
        pixel_values = rgb_cropped  # shape => (N, 3, crop_h, crop_w)
        # mask = get_random_mask(pixel_values.shape)
        # mask_pixel_values = pixel_values * (1 - mask) + (-1.0 * mask)
        
        if self.use_etm_as_ref:
            first_frame = pixel_values[0].permute(1,2,0).contiguous()
            first_frame = (first_frame * 0.5 + 0.5)
            first_frame = torch.clamp(first_frame, 0, 1) # 0-1
            first_frame = first_frame.numpy()
            
            etm = gt2etm(first_frame, self.temporal_noise_FLG, self.c_mean_range, self.c_variance,  # temporal noise
                    self.salt_pepper_noise_FLG, self.salt_prob, self.pepper_prob,  # salt pepper noise
                    self.gaussian_noise_FLG, self.gaussian_mean, self.gaussian_var,  # gaussian noise
                    poisson_noise_FLG=self.poisson_noise_FLG, poisson_level=self.poisson_level,  # poisson noise
                    diffraction_blur_FLG=self.diffraction_blur_FLG, diffraction_psf_size=self.diffraction_psf_size,  # diffraction blur
                    return_etm_gt_FLG=self.return_etm_gt_FLG,img_format='RGB')
            # etm: h,w,1, float32, range 0-1
            # totensor, normalize
            # etm (H,W) or (H,W,1)
            if etm.ndim == 3 and etm.shape[-1] == 1:
                etm = etm.squeeze(-1)  # (H,W)
            etm = self.gray_transform(etm)  # -> (1,H,W)
            # etm_tensor: shape (1, H, W)
            etm = etm.expand(3, -1, -1)  # -> (3, H, W)，灰度图复制成3通道
            ref_pixel_values = etm.unsqueeze(0)  # -> (1, 3, H, W)

        else:
            ref_pixel_values = pixel_values[0].unsqueeze(0) # 1, 3, h, w
            # first_frame_mask = mask[0].unsqueeze(0)
            # if (first_frame_mask == 1).all():
                # ref_pixel_values = torch.ones_like(ref_pixel_values)*-1

        # maybe define clip_pixel_values as the first frame, ref_pixel_values: 1,c,h,w, range 0-1
        clip_input_frame = ref_pixel_values[0].permute(1,2,0).contiguous() # h,w,c
        clip_input_frame = (clip_input_frame * 0.5 + 0.5) * 255.0
        clip_input_frame = torch.clamp(clip_input_frame, 0, 255)

        sample = {
            "control_pixel_values": voxel_cropped,   # (N, 3, crop_h, crop_w)
            "pixel_values": pixel_values,            # (N, 3, crop_h, crop_w)
            # "mask": mask,
            # "mask_pixel_values": mask_pixel_values,
            "ref_pixel_values": ref_pixel_values,
            "clip_pixel_values": clip_input_frame,
            "text": "",
            "data_type": "video",
            "idx": idx
        }
        return sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/work/lei_sun/datasets/EventAid-R/",
        # default="/work/lei_sun/datasets/EventAid-B/",
        help="Path to the main dataset root containing multiple subfolders."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=100,
        help="Which dataset index (clip) to retrieve."
    )
    parser.add_argument(
        "--frames_per_clip",
        type=int,
        default=29,
        help="Number of frames per clip."
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=(320,640),
        help="Resize dimension for single images."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_save/EventAid_R_start_frame",
        help="Directory to save frames and voxel images."
    )
    args = parser.parse_args()
    
    # 1) Instantiate the dataset with frame-based logic & shift_mode
    dataset = ImageVoxelControlDataset(
        dataset_root=args.dataset_root,
        frames_per_clip=args.frames_per_clip,
        image_sample_size=args.image_sample_size,
        load_rgb=True
    )


    print(f"Total clips in dataset: {len(dataset)}")
    if args.index < 0 or args.index >= len(dataset):
        raise ValueError(f"Requested index {args.index} is out of range 0..{len(dataset)-1}.")

    # 2) Retrieve an item by index
    sample = dataset[args.index]
    pixel_values = sample["pixel_values"]              # shape => (N, 3, H, W)
    voxel_values = sample["control_pixel_values"]      # shape => (num_bins, 3, H, W) or None
    ref_values = sample["ref_pixel_values"]

    # 3) Check for valid data
    if pixel_values is None or voxel_values is None:
        print(f"Index={args.index} returned an empty sample. Exiting.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 4) Save frames in pixel_values
    for i in range(pixel_values.shape[0]):
        frame = pixel_values[i]  # (3, H, W)
        frame_img = (frame * 0.5 + 0.5).clamp(0,1)
        save_image(frame_img, os.path.join(args.output_dir, f"frame_{i}.png"))
    print(f"Saved {pixel_values.shape[0]} frames to {args.output_dir}/frame_*.png")

    # 4.1) Save frames in ref_pixel_values as grayscale
    if ref_values is not None:
        for i in range(ref_values.shape[0]):
            ref_frame = ref_values[i]  # (3, H, W), where the 3 channels are identical
            ref_frame_gray = ref_frame[0]  # Take the first channel as grayscale
            ref_frame_img = (ref_frame_gray * 0.5 + 0.5).clamp(0, 1)  # Normalize and clamp to [0,1]
            save_image(ref_frame_img, os.path.join(args.output_dir, f"ref_frame_{i}.png"))
        print(f"Saved {ref_values.shape[0]} reference frames to {args.output_dir}/ref_frame_*.png")

    # 5) Save the voxel bins
    for b in range(voxel_values.shape[0]):
        voxel_bin = voxel_values[b]  # (3, H, W)
        # If needed, apply min-max normalization for display
        vmin = voxel_bin.min()
        vmax = voxel_bin.max()
        if vmax > vmin:
            voxel_bin_norm = (voxel_bin - vmin) / (vmax - vmin)
        else:
            voxel_bin_norm = voxel_bin * 0.0

        save_image(voxel_bin_norm, os.path.join(args.output_dir, f"voxel_bin_{b}.png"))
    print(f"Saved {voxel_values.shape[0]} voxel bins to {args.output_dir}/voxel_bin_*.png")

    print(f"Sample keys:\n{list(sample.keys())}")
    print("Done.")


if __name__ == "__main__":
    main()
