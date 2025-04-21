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
from tqdm import tqdm

import argparse
from torchvision.utils import save_image
from torch.utils.data import BatchSampler, Sampler
import shutil


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



def collect_make_subfolders(dataset_root: str) -> List[Dict[str, str]]:
    """
    For each subfolder in dataset_root, check if it has:
     - timestamps.txt
     - event/ subfolder
     - gt/ subfolder
    Ensure voxel/ subfolder exists (create if missing).
    Return a list of dicts with 'timestamps_path', 'event_dir', 'rgb_dir', and 'voxel_dir'.
    """
    subfolders = []
    for entry in sorted(os.listdir(dataset_root)):
        sub_path = os.path.join(dataset_root, entry)
        if not os.path.isdir(sub_path):
            continue
        timestamps_path = os.path.join(sub_path, "timestamps.txt")
        events_dir = os.path.join(sub_path, "event")
        gt_dir     = os.path.join(sub_path, "gt")
        voxel_dir = os.path.join(sub_path, "voxel")

        if os.path.isfile(timestamps_path) and os.path.isdir(events_dir) and os.path.isdir(gt_dir):
            if not os.path.exists(voxel_dir):
                os.makedirs(voxel_dir, exist_ok=True)

            subfolders.append({
                "timestamps_path": timestamps_path,
                "event_dir": events_dir,
                "rgb_dir":   gt_dir,
                "voxel_dir": voxel_dir
            })
    return subfolders



def save_image(voxel_bin_norm: np.ndarray, save_path: str):
    """
    Save a normalized voxel slice (H x W) as a grayscale PNG.
    """
    img = (voxel_bin_norm * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(save_path)


def process_dataset(dataset_root: str, num_bins: int = 3):
    mismatch_log = []
    sanity_check_dir = os.path.join(".", "sanity_check")
    os.makedirs(sanity_check_dir, exist_ok=True)

    for entry in sorted(os.listdir(dataset_root)):
        sub_path = os.path.join(dataset_root, entry)
        if not os.path.isdir(sub_path):
            continue

        event_dir = os.path.join(sub_path, "event")
        gt_dir = os.path.join(sub_path, "gt")
        voxel_dir = os.path.join(sub_path, "voxel")
        vis_dir = os.path.join(sub_path, "voxel_visualization")
        shape_path = os.path.join(sub_path, "shape.txt")

        if not os.path.exists(event_dir) or not os.path.exists(gt_dir) or not os.path.isfile(shape_path):
            continue

        os.makedirs(voxel_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        # Read shape.txt
        with open(shape_path, "r") as f:
            shape_line = f.readline().strip()
            width, height = map(int, shape_line.split())

        event_files = sorted([f for f in os.listdir(event_dir) if f.endswith(".txt")])
        gt_files = sorted([
            f for f in os.listdir(gt_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        event_ids = {os.path.splitext(f)[0] for f in event_files}
        gt_ids = {os.path.splitext(f)[0].replace("_img", "") for f in gt_files}

        if event_ids != gt_ids:
            missing_in_event = gt_ids - event_ids
            missing_in_gt = event_ids - gt_ids
            if missing_in_event or missing_in_gt:
                msg = f"[{entry}] Mismatch - missing in event: {sorted(missing_in_event)}, missing in gt: {sorted(missing_in_gt)}"
                print(msg)
                mismatch_log.append(msg)

        common_ids = sorted(event_ids & gt_ids)
        print(f"Processing {entry} with {len(common_ids)} valid pairs...")

        first_saved = False
        for fid in tqdm(common_ids, desc=f"[{entry}]", leave=False):
            txt_path = os.path.join(event_dir, f"{fid}.txt")
            voxel_path = os.path.join(voxel_dir, f"{fid}.npy")
            vis_base = os.path.join(vis_dir, f"{fid}_bin")

            events = read_txt_events(txt_path)
            voxel = events_to_voxel_grid(events, num_bins=num_bins, width=width, height=height)
            np.save(voxel_path, voxel)

            for b in range(num_bins):
                voxel_bin = voxel[..., b]
                vmin = voxel_bin.min()
                vmax = voxel_bin.max()
                if vmax > vmin:
                    voxel_bin_norm = (voxel_bin - vmin) / (vmax - vmin)
                else:
                    voxel_bin_norm = voxel_bin * 0.0
                vis_path = f"{vis_base}_{b}.png"
                save_image(voxel_bin_norm, vis_path)

                # 如果是第一个配对样本，保存sanity check图像
                if not first_saved and b == 0:
                    shutil.copy(vis_path, os.path.join(sanity_check_dir, f"{entry}_voxel.png"))

            if not first_saved:
                # 找到对应gt原图
                for gt_file in gt_files:
                    if os.path.splitext(gt_file)[0].replace("_img", "") == fid:
                        gt_src_path = os.path.join(gt_dir, gt_file)
                        gt_ext = os.path.splitext(gt_file)[1]
                        gt_dst_path = os.path.join(sanity_check_dir, f"{entry}_gt{gt_ext}")
                        shutil.copy(gt_src_path, gt_dst_path)
                        break
                first_saved = True

    # 写入 mismatch 文件
    report_path = os.path.join(dataset_root, "missing_report.txt")
    with open(report_path, "w") as f:
        for line in mismatch_log:
            f.write(line + "\n")
    print(f"Missing report saved to {report_path}")


if __name__ == "__main__":
    dataset_root = '/work/lei_sun/datasets/EventAid-R/'
    process_dataset(dataset_root)
