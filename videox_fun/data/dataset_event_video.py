import os
import random
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Union
from torch.utils.data import Dataset
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor

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


# ---------------------------------------
# Replace these with your real functions
# ---------------------------------------
def read_txt_events(txt_file):
    """
    Read a .txt file of events with lines of the form:
        timestamp x y polarity
    Returns a (N,4) NumPy array of float32: [timestamp, x, y, polarity].
    """
    events_list = []
    with open(txt_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 4:
                t, x, y, p = parts
                events_list.append([float(t), float(x), float(y), float(p)])
    if len(events_list) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(events_list, dtype=np.float32)


def load_timestamps(timestamps_path):
    """
    Loads timestamps from a file where each line contains a single timestamp.
    Returns a list of floats, sorted ascending.
    """
    if not os.path.isfile(timestamps_path):
        raise FileNotFoundError(f"Cannot find timestamps.txt at {timestamps_path}")
    with open(timestamps_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    timestamps = [float(line) for line in lines]
    return timestamps


def get_txt_files(event_dir):
    """
    Return a sorted list of all .txt event files in `event_dir`.
    """
    if not os.path.isdir(event_dir):
        raise FileNotFoundError(f"No 'event' subfolder found at {event_dir}")
    txt_files = [f for f in os.listdir(event_dir) if f.endswith(".txt")]
    txt_files.sort()
    return txt_files


def process_single_file(index, txt_file, timestamps, event_dir, start_ts, end_ts):
    """
    Worker function for gathering events in [start_ts, end_ts].
    Returns (file_start, subset_events) or None if no overlap or no events.
    """
    file_start = timestamps[index]
    file_end   = timestamps[index + 1] if (index + 1) < len(timestamps) else float('inf')

    # Quick check for overlap
    if file_end < start_ts or file_start > end_ts:
        return None

    txt_path = os.path.join(event_dir, txt_file)
    events = read_txt_events(txt_path)
    if events.shape[0] == 0:
        return None

    # Filter to [start_ts, end_ts]
    mask = (events[:, 0] >= start_ts) & (events[:, 0] <= end_ts)
    subset = events[mask]
    if subset.shape[0] == 0:
        return None

    return (file_start, subset)


def gather_events_in_frame_range(
    txt_files, timestamps, event_dir,
    start_frame_idx, end_frame_idx
):
    """
    Gathers events from the specified [start_frame_idx..end_frame_idx].
    That is, from timestamps[start_frame_idx] to timestamps[end_frame_idx].
    """
    start_ts = timestamps[start_frame_idx]
    end_ts   = timestamps[end_frame_idx] if end_frame_idx < len(timestamps) else timestamps[-1]

    results = []
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, txt_file in enumerate(txt_files):
            futures.append(executor.submit(
                process_single_file,
                i, txt_file, timestamps, event_dir,
                start_ts, end_ts
            ))
        for f in futures:
            r = f.result()
            if r is not None:
                results.append(r)

    if len(results) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Sort by file_start
    results.sort(key=lambda x: x[0])
    # Merge
    event_subsets = [r[1] for r in results]
    all_events = np.concatenate(event_subsets, axis=0)
    # Sort by ascending time
    all_events = all_events[np.argsort(all_events[:, 0])]
    return all_events


def get_shift_value(shift_mode: str) -> float:
    """
    Map user-friendly shift mode to a numeric offset in [0..1].
    shift_mode can be:
      - "begin_of_frame" => 0.0
      - "in_the_middle"  => 0.5
      - "end_of_frame"   => 1.0
    """
    mapping = {
        "begin_of_frame": 0.0,
        "in_the_middle":  0.5,
        "end_of_frame":   1.0,
    }
    if shift_mode not in mapping:
        raise ValueError(f"Invalid shift_mode='{shift_mode}'. Must be one of {list(mapping.keys())}.")
    return mapping[shift_mode]


def events_to_voxel_grid(
    events,
    num_bins,
    width,
    height,
    return_format='HWC',
    shift_mode='begin_of_frame',
):
    """
    Build a uniform-time voxel grid with `num_bins`.
    Polarity 0 is mapped to -1. 
    Return shape: 'HWC' => (H, W, num_bins) or 'CHW' => (num_bins, H, W).

    shift_mode controls how we offset the bin index for the earliest event:
      - "begin_of_frame" => no offset
      - "in_the_middle"  => 0.5 offset
      - "end_of_frame"   => 1.0 offset
    """
    if events.shape[0] == 0:
        if return_format == 'CHW':
            return np.zeros((num_bins, height, width), dtype=np.float32)
        else:
            return np.zeros((height, width, num_bins), dtype=np.float32)

    shift_value = get_shift_value(shift_mode)

    events = events[np.argsort(events[:, 0])]   # sort by time
    pol = events[:, 3]
    pol[pol == 0] = -1                         # 0 => -1

    voxel_grid = np.zeros((num_bins, height, width), dtype=np.float32).ravel()

    t0 = events[0, 0]
    t1 = events[-1, 0]
    denom = (t1 - t0) if (t1 > t0) else 1e-9  # avoid divide by zero

    # scaled_ts in [0..(num_bins - 1)] + shift
    scaled_ts = (num_bins - 1) * (events[:, 0] - t0) / denom + shift_value

    # clamp to [0..(num_bins - 1)] to avoid out-of-range
    scaled_ts = np.clip(scaled_ts, 0.0, num_bins - 1 - 1e-9)

    xs = events[:, 1].astype(int)
    ys = events[:, 2].astype(int)

    tis = scaled_ts.astype(int)
    dts = scaled_ts - tis
    vals_left  = pol * (1.0 - dts)
    vals_right = pol * dts

    width_height = width * height
    # Left bin
    valid_left = (tis >= 0) & (tis < num_bins)
    np.add.at(
        voxel_grid,
        xs[valid_left] + ys[valid_left]*width + tis[valid_left]*width_height,
        vals_left[valid_left]
    )
    # Right bin
    valid_right = (tis + 1 < num_bins)
    np.add.at(
        voxel_grid,
        xs[valid_right] + ys[valid_right]*width + (tis[valid_right]+1)*width_height,
        vals_right[valid_right]
    )

    voxel_grid = voxel_grid.reshape(num_bins, height, width)

    if return_format == 'CHW':
        return voxel_grid
    else:
        return voxel_grid.transpose((1, 2, 0))


def get_random_mask(shape, image_start_only=False):
    f, c, h, w = shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if not image_start_only:
        if f != 1:
            mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05]) 
        else:
            mask_index = np.random.choice([0, 1], p = [0.2, 0.8])
        if mask_index == 0:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)
            mask[:, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 1:
            mask[:, :, :, :] = 1
        elif mask_index == 2:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:, :, :, :] = 1
        elif mask_index == 3:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
        elif mask_index == 4:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)

            mask_frame_before = np.random.randint(0, f // 2)
            mask_frame_after = np.random.randint(f // 2, f)
            mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 5:
            mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
        elif mask_index == 6:
            num_frames_to_mask = random.randint(1, max(f // 2, 1))
            frames_to_mask = random.sample(range(f), num_frames_to_mask)

            for i in frames_to_mask:
                block_height = random.randint(1, h // 4)
                block_width = random.randint(1, w // 4)
                top_left_y = random.randint(0, h - block_height)
                top_left_x = random.randint(0, w - block_width)
                mask[i, 0, top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 1
        elif mask_index == 7:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()  # 长半轴
            b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()  # 短半轴

            for i in range(h):
                for j in range(w):
                    if ((i - center_y) ** 2) / (b ** 2) + ((j - center_x) ** 2) / (a ** 2) < 1:
                        mask[:, :, i, j] = 1
        elif mask_index == 8:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
            for i in range(h):
                for j in range(w):
                    if (i - center_y) ** 2 + (j - center_x) ** 2 < radius ** 2:
                        mask[:, :, i, j] = 1
        elif mask_index == 9:
            for idx in range(f):
                if np.random.rand() > 0.5:
                    mask[idx, :, :, :] = 1
        else:
            raise ValueError(f"The mask_index {mask_index} is not define")
    else:
        if f != 1:
            mask[1:, :, :, :] = 1
        else:
            mask[:, :, :, :] = 1
    return mask

def collect_subfolders(dataset_root: str) -> List[Dict[str, str]]:
    """
    For each subfolder in dataset_root, check if it has:
     - timestamps.txt
     - event/ subfolder
     - gt/ subfolder
    Return a list of dicts with 'timestamps_path', 'event_dir', 'rgb_dir'.
    """
    subfolders = []
    for entry in sorted(os.listdir(dataset_root)):
        sub_path = os.path.join(dataset_root, entry)
        if not os.path.isdir(sub_path):
            continue
        timestamps_path = os.path.join(sub_path, "timestamps.txt")
        events_dir = os.path.join(sub_path, "event")
        gt_dir     = os.path.join(sub_path, "gt")
        if os.path.isfile(timestamps_path) and os.path.isdir(events_dir) and os.path.isdir(gt_dir):
            subfolders.append({
                "timestamps_path": timestamps_path,
                "event_dir": events_dir,
                "rgb_dir":   gt_dir
            })
    return subfolders


class ImageEventControlDataset(Dataset):
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
        num_bins: int = 24,
        shift_mode: str = "begin_of_frame",  # new param
        image_sample_size: Union[int, Tuple[int,int]] = 512,
        video_sample_size: Union[int, Tuple[int,int]] = 512,
        voxel_channel_mode: str = "repeat",  # "repeat" or "triple_bins"
        load_rgb: bool = True,
    ):
        super().__init__()
        self.dataset_root = dataset_root
        self.frames_per_clip = frames_per_clip
        self.num_bins = num_bins
        self.shift_mode = shift_mode     # new param for shifting event bins
        self.load_rgb = load_rgb

        # unify image_sample_size/video_sample_size into (H,W)
        if isinstance(image_sample_size, int):
            self.image_sample_size = (image_sample_size, image_sample_size)
        else:
            self.image_sample_size = image_sample_size

        if isinstance(video_sample_size, int):
            self.video_sample_size = (video_sample_size, video_sample_size)
        else:
            self.video_sample_size = video_sample_size

        # "repeat" => replicate single channel to 3
        # "triple_bins" => 3x bins => shape => (num_bins, 3, H, W)
        if voxel_channel_mode not in ("repeat", "triple_bins"):
            raise ValueError("voxel_channel_mode must be 'repeat' or 'triple_bins'")
        self.voxel_channel_mode = voxel_channel_mode

        # discover subfolders
        subfolders = collect_subfolders(dataset_root)

        # For each subfolder, gather info
        self.folders = []
        self.num_sequences_per_folder = []

        for sf in subfolders:
            ts = load_timestamps(sf["timestamps_path"])
            txt_files = get_txt_files(sf["event_dir"])
            # read how many frames are in 'gt/'
            gt_files = sorted([
                f for f in os.listdir(sf["rgb_dir"]) if f.endswith(".jpg") or f.endswith(".png")
            ])
            if len(gt_files) == 0:
                continue

            # optional check
            if len(gt_files) != len(ts):
                print(f"[WARNING] folder {sf['rgb_dir']} has {len(gt_files)} images but {len(ts)} timestamps")

            num_rgb_frames = len(gt_files)
            num_sequences = num_rgb_frames // self.frames_per_clip
            if num_sequences < 1:
                print(f"[WARNING] folder {sf['rgb_dir']} => not enough frames for a single clip of {self.frames_per_clip}")
                continue

            folder_info = {
                "timestamps": ts,
                "txt_files":  txt_files,
                "event_dir":  sf["event_dir"],
                "rgb_dir":    sf["rgb_dir"],
                "gt_files":   gt_files,  # sorted list of image filenames
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
            transforms.Resize(min(self.image_sample_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

        self.video_transform = transforms.Compose([
            transforms.Resize(min(self.video_sample_size)),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

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

        # gather events
        all_events = gather_events_in_frame_range(
            txt_files=folder_info["txt_files"],
            timestamps=folder_info["timestamps"],
            event_dir=folder_info["event_dir"],
            start_frame_idx=start_frame_idx,
            end_frame_idx=end_frame_idx
        )

        if all_events.shape[0] == 0:
            # no events in this range
            return {
                "control_pixel_values": None,
                "pixel_values": None,
                "text": "",
                "data_type": "video",
                "idx": idx
            }

        # figure out max x,y among events
        max_x = int(all_events[:,1].max()) + 1
        max_y = int(all_events[:,2].max()) + 1

        # build voxel
        # We'll do a uniform time approach with num_bins = self.num_bins
        voxel = events_to_voxel_grid(
            events=all_events,
            num_bins=self.num_bins,
            width=max_x,
            height=max_y,
            return_format="HWC",
            shift_mode=self.shift_mode,   # <-- pass shift_mode here
        )  # shape => (H, W, num_bins)

        # shape => (num_bins, H, W) after transpose
        voxel = voxel.transpose((2, 0, 1))  # => (num_bins, H, W)

        if self.voxel_channel_mode == "repeat":
            # => (num_bins, 1, H, W)
            voxel = np.expand_dims(voxel, axis=1)
            # => (num_bins, 3, H, W)
            voxel = np.repeat(voxel, repeats=3, axis=1)
        else:
            # "triple_bins" => we create 3x bins
            triple_num_bins = self.num_bins * 3
            voxel3 = events_to_voxel_grid(
                all_events,
                num_bins=triple_num_bins,
                width=max_x,
                height=max_y,
                return_format="HWC",
                shift_mode=self.shift_mode
            )
            voxel3 = voxel3.transpose((2,0,1))  # (triple_num_bins, H, W)
            # reshape => (self.num_bins, 3, H, W)
            voxel = voxel3.reshape(self.num_bins, 3, voxel3.shape[1], voxel3.shape[2])

        voxel_torch = torch.from_numpy(voxel).float()

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
                img_t = self.image_transform(img)
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

        # Possibly resize the voxel to match
        voxel_torch = F.interpolate(voxel_torch, size=(H_img, W_img), mode='bilinear', align_corners=False)

        # random crop
        crop_h, crop_w = 512, 512
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
        mask = get_random_mask(pixel_values.shape)
        mask_pixel_values = pixel_values * (1 - mask) + (-1.0 * mask)

        # maybe define clip_pixel_values as the first frame
        clip_input_frame = pixel_values[0].permute(1,2,0).contiguous()
        clip_input_frame = (clip_input_frame * 0.5 + 0.5) * 255.0
        clip_input_frame = torch.clamp(clip_input_frame, 0, 255)

        ref_pixel_values = pixel_values[0].unsqueeze(0)
        first_frame_mask = mask[0].unsqueeze(0)
        if (first_frame_mask == 1).all():
            ref_pixel_values = torch.ones_like(ref_pixel_values)*-1

        sample = {
            "control_pixel_values": voxel_cropped,   # (num_bins, 3, crop_h, crop_w)
            "pixel_values": pixel_values,            # (N, 3, crop_h, crop_w)
            "mask": mask,
            "mask_pixel_values": mask_pixel_values,
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
        default="/work/andrea_alfarano/EventAid-dataset/EvenAid-B",
        help="Path to the main dataset root containing multiple subfolders."
    )
    parser.add_argument(
        "--index",
        type=int,
        default=320,
        help="Which dataset index (clip) to retrieve."
    )
    parser.add_argument(
        "--frames_per_clip",
        type=int,
        default=24,
        help="Number of frames per clip."
    )
    parser.add_argument(
        "--num_bins",
        type=int,
        default=24,
        help="Number of bins in the voxel grid (for events)."
    )
    parser.add_argument(
        "--shift_mode",
        type=str,
        default="in_the_middle",
        choices=["begin_of_frame", "in_the_middle", "end_of_frame"],
        help="Where to align the earliest event in the bins."
    )
    parser.add_argument(
        "--voxel_channel_mode",
        type=str,
        default="repeat",
        choices=["repeat", "triple_bins"],
        help="How to convert voxel bins to 3 channels: "
             "'repeat' (replicate single channel) or 'triple_bins' (3x bins)."
    )
    parser.add_argument(
        "--image_sample_size",
        type=int,
        default=512,
        help="Resize dimension for single images."
    )
    parser.add_argument(
        "--video_sample_size",
        type=int,
        default=512,
        help="Resize dimension for video frames."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./test_dataset_in_the_middle",
        help="Directory to save frames and voxel images."
    )
    args = parser.parse_args()

    # 1) Instantiate the dataset with frame-based logic & shift_mode
    dataset = ImageEventControlDataset(
        dataset_root=args.dataset_root,
        frames_per_clip=args.frames_per_clip,
        num_bins=args.num_bins,
        shift_mode=args.shift_mode,
        image_sample_size=args.image_sample_size,
        video_sample_size=args.video_sample_size,
        voxel_channel_mode=args.voxel_channel_mode,
        load_rgb=True
    )

    print(f"Total clips in dataset: {len(dataset)}")
    if args.index < 0 or args.index >= len(dataset):
        raise ValueError(f"Requested index {args.index} is out of range 0..{len(dataset)-1}.")

    # 2) Retrieve an item by index
    sample = dataset[args.index]
    pixel_values = sample["pixel_values"]              # shape => (N, 3, H, W)
    voxel_values = sample["control_pixel_values"]      # shape => (num_bins, 3, H, W) or None

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
