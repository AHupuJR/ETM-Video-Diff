import os

def check_gt_voxel_matching(root_dir):
    mismatches = []

    subfolders = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                  if os.path.isdir(os.path.join(root_dir, d))]

    for folder in subfolders:
        gt_dir = os.path.join(folder, "gt")
        voxel_dir = os.path.join(folder, "voxel")

        if not os.path.isdir(gt_dir) or not os.path.isdir(voxel_dir):
            print(f"Skipping {folder}: Missing gt/ or voxel/ folder")
            continue

        # Count image files in gt/
        gt_files = [f for f in os.listdir(gt_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        # Count npy files in voxel/
        voxel_files = [f for f in os.listdir(voxel_dir) if f.endswith(".npy")]

        gt_count = len(gt_files)
        voxel_count = len(voxel_files)

        status = "✅" if gt_count == voxel_count else "❌ MISMATCH"
        print(f"{folder}\n  gt images:    {gt_count}\n  voxel npys:   {voxel_count}  --> {status}\n")

        if gt_count != voxel_count:
            mismatches.append((folder, gt_count, voxel_count))

    print(f"✅ Check complete. Total mismatches: {len(mismatches)}")
    return mismatches

if __name__ == "__main__":
    dataset_root = "/work/lei_sun/datasets/EventAid-R/"
    check_gt_voxel_matching(dataset_root)