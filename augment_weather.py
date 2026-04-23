import os
import cv2
import numpy as np
import argparse


# ------------------------------
#   Weather effect functions
# ------------------------------

def add_rain_effect(img, intensity=0.6):

    if img is None:
        return img

    h, w = img.shape[:2]


    rain_layer = np.zeros_like(img, dtype=np.uint8)


    base_density = h * w / 600.0
    num_drops = int(base_density * intensity)

    for _ in range(num_drops):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        length = np.random.randint(15, 30)         
        thickness = np.random.randint(1, 2)        


        end_x = x + np.random.randint(-3, 4)
        end_y = y + length

        end_x = np.clip(end_x, 0, w - 1)
        end_y = np.clip(end_y, 0, h - 1)

        color = (200, 200, 200)  
        cv2.line(rain_layer, (x, y), (end_x, end_y), color, thickness)


    rain_layer = cv2.blur(rain_layer, (3, 3))


    img_rain = cv2.addWeighted(img, 1.0, rain_layer, 0.6, 0)


    img_rain = cv2.convertScaleAbs(img_rain, alpha=0.9, beta=-10)

    return img_rain


def add_fog_effect(img, fog_intensity=0.18, blur_sigma=4.0):

    if img is None:
        return img

    h, w = img.shape[:2]


    fog_color = np.full_like(img, 255, dtype=np.uint8)


    Y, X = np.ogrid[:h, :w]
    center_x = np.random.randint(int(0.3 * w), int(0.7 * w))
    center_y = np.random.randint(int(0.3 * h), int(0.7 * h))

    dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    max_dist = np.sqrt(w ** 2 + h ** 2)


    mask = (1.0 - dist / max_dist)
    mask = np.clip(mask, 0.0, 1.0)
    mask = mask * fog_intensity  


    alpha = mask[..., None].astype(np.float32)

    img_f = img.astype(np.float32)
    fog_f = fog_color.astype(np.float32)

    foggy = img_f * (1.0 - alpha) + fog_f * alpha

    foggy = cv2.GaussianBlur(foggy, ksize=(0, 0), sigmaX=blur_sigma)

    foggy = np.clip(foggy, 0, 255).astype(np.uint8)
    return foggy


# ------------------------------
#   Image (frame) processing
# ------------------------------

def process_extracted_frames(input_root, out_root_rain, out_root_fog):
    if not os.path.isdir(input_root):
        print(f"[WARN] Extracted_frames directory not found: {input_root}")
        return

    os.makedirs(out_root_rain, exist_ok=True)
    os.makedirs(out_root_fog, exist_ok=True)

    seq_dirs = sorted(
        d for d in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, d))
    )

    for seq in seq_dirs:
        seq_in_dir = os.path.join(input_root, seq)
        seq_out_rain = os.path.join(out_root_rain, seq)
        seq_out_fog = os.path.join(out_root_fog, seq)

        os.makedirs(seq_out_rain, exist_ok=True)
        os.makedirs(seq_out_fog, exist_ok=True)

        print(f"[INFO] Processing sequence {seq} (frames)")

        frame_files = sorted(
            f for f in os.listdir(seq_in_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        for fname in frame_files:
            in_path = os.path.join(seq_in_dir, fname)

            img = cv2.imread(in_path)
            if img is None:
                print(f"[WARN] Failed to read image: {in_path}")
                continue

            img_rain = add_rain_effect(img)
            img_fog = add_fog_effect(img)

            base, _ = os.path.splitext(fname)
            out_rain_path = os.path.join(seq_out_rain, base + ".jpg")
            out_fog_path = os.path.join(seq_out_fog, base + ".jpg")

            cv2.imwrite(out_rain_path, img_rain)
            cv2.imwrite(out_fog_path, img_fog)


# ------------------------------
#   Video processing
# ------------------------------

def process_videos(input_root, out_root_rain, out_root_fog):
    if not os.path.isdir(input_root):
        print(f"[WARN] test directory not found: {input_root}")
        return

    os.makedirs(out_root_rain, exist_ok=True)
    os.makedirs(out_root_fog, exist_ok=True)

    video_files = sorted(
        f for f in os.listdir(input_root)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    )

    for fname in video_files:
        in_path = os.path.join(input_root, fname)
        print(f"[INFO] Processing video {fname}")

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            print(f"[WARN] Failed to open video: {in_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # mp4
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out_rain_path = os.path.join(out_root_rain, fname)
        out_fog_path = os.path.join(out_root_fog, fname)

        out_rain = cv2.VideoWriter(out_rain_path, fourcc, fps, (width, height))
        out_fog = cv2.VideoWriter(out_fog_path, fourcc, fps, (width, height))

        if not out_rain.isOpened() or not out_fog.isOpened():
            print(f"[WARN] Failed to create VideoWriter for: {fname}")
            cap.release()
            continue

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rain_frame = add_rain_effect(frame)
            fog_frame = add_fog_effect(frame)

            out_rain.write(rain_frame)
            out_fog.write(fog_frame)

        cap.release()
        out_rain.release()
        out_fog.release()
        print(f"[INFO] Finished video {fname}")


# ------------------------------
#   Main
# ------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Add rainy/foggy weather effects to SO-TAD frames and videos."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./Benchmark/SO_TAD",
        help="Base directory of SO_TAD dataset (default: ~/Benchmark/SO_TAD)",
    )
    parser.add_argument(
        "--out_base_dir",
        type=str,
        default="./Benchmark/SO_TAD_aug",
        help="Base directory for augmented outputs (default: ~/Benchmark/SO_TAD_aug)",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    out_base_dir = args.out_base_dir

    extracted_frames_dir = os.path.join(base_dir, "Extracted_frames")
    test_videos_dir = os.path.join(base_dir, "test")

    out_frames_rain = os.path.join(out_base_dir, "Extracted_frames_rain")
    out_frames_fog = os.path.join(out_base_dir, "Extracted_frames_fog")
    out_videos_rain = os.path.join(out_base_dir, "test_rain")
    out_videos_fog = os.path.join(out_base_dir, "test_fog")

    print("[INFO] Base dir:", base_dir)
    print("[INFO] Output base dir:", out_base_dir)

    process_extracted_frames(
        extracted_frames_dir,
        out_frames_rain,
        out_frames_fog,
    )

    process_videos(
        test_videos_dir,
        out_videos_rain,
        out_videos_fog,
    )

    print("[INFO] All done.")


if __name__ == "__main__":
    main()
