from mmdeploy_runtime import PoseTracker
import os
from tqdm import tqdm
import cv2
import numpy as np
import json
import argparse
import multiprocessing
import torch
import time # Added for basic timing

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# --- Visualization Functions (Copied directly from rtmdeploy.py) ---

def visualize_halpe(frame, keypoints, frame_id, bbox=None, output_dir="res/", thr=0.5, resize=1280):
    # Define the skeleton for Halpe 26 keypoints
    skeleton = [
        (0, 1), (0, 2),           # Nose to eyes
        (1, 3), (2, 4),           # Eyes to ears
        (0, 17), (17, 18),        # Nose to head to neck
        (18, 19),                 # Neck to hip
        (18, 5), (18, 6),         # Neck to shoulders
        (5, 7), (7, 9),           # Left arm
        (6, 8), (8, 10),          # Right arm
        (19, 11), (19, 12),       # Hip to left/right hip
        (11, 13), (13, 15),       # Left leg
        (12, 14), (14, 16),       # Right leg
        (15, 20), (20, 22),       # Left foot
        (15, 24),                 # Left heel
        (16, 21), (21, 23),       # Right foot
        (16, 25)                  # Right heel
    ]

    # Define a color palette for keypoints
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
        (255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
        (255, 153, 255), (153, 204, 255), (255, 102, 255), (255, 51, 255),
        (102, 178, 255), (51, 153, 255), (255, 153, 153), (255, 102, 102),
        (255, 51, 51), (153, 255, 153)
    ]
    line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
              (0, 255, 102), (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
              (77, 191, 255), (204, 77, 255), (77, 222, 255), (255, 156, 127),
              (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36),
              (0, 77, 255), (0, 77, 255), (0, 77, 255), (0, 77, 255), (255, 156, 127), (255, 156, 127)]

    # Resize the frame
    scale = resize / max(frame.shape[0], frame.shape[1])
    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    # Ensure keypoints are valid before scaling
    if keypoints is None or len(keypoints) == 0 or not isinstance(keypoints[0], np.ndarray) or keypoints[0].size == 0:
        return img # Return resized frame if no keypoints

    scaled_keypoints = (keypoints[..., :2] * scale).astype(int)


    # Draw bounding boxes if provided
    if bbox is not None and len(bbox) > 0:
        scaled_bbox = (bbox * scale).astype(int)
        for idx, box in enumerate(scaled_bbox):
            pt1 = (box[0], box[1])
            pt2 = (box[2], box[3])
            color_idx = idx % len(palette)
            cv2.rectangle(img, pt1, pt2, palette[color_idx], 2, cv2.LINE_AA)
            cv2.putText(img, f'ID: {idx}', (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, palette[color_idx], 2)

    # Draw skeleton
    for person_keypoints in scaled_keypoints:
        for idx, (start_idx, end_idx) in enumerate(skeleton):
            if start_idx < len(person_keypoints) and end_idx < len(person_keypoints):
                pt1 = tuple(person_keypoints[start_idx])
                pt2 = tuple(person_keypoints[end_idx])
                color = line_color[idx % len(line_color)]
                cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

        # Draw keypoints
        for idx, kpt in enumerate(person_keypoints):
             if kpt.shape == (2,): # Ensure kpt is a valid coordinate pair
                cv2.circle(img, tuple(kpt), 3, palette[idx % len(palette)], -1, cv2.LINE_AA)

    return img

def visualize(frame, keypoints, frame_id, bbox=None, output_dir="res/",thr=0.5, resize=1280):
    skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]

    # Original palette for keypoints
    palette = [(255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0),
               (255, 153, 255), (153, 204, 255), (255, 102, 255),
               (255, 51, 255), (102, 178, 255),
               (51, 153, 255), (255, 153, 153), (255, 102, 102), (255, 51, 51),
               (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
               (0, 0, 255), (255, 0, 0), (255, 255, 255)]

    # Distinct colors for bounding boxes
    bbox_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]

    link_color = [
        0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16
    ]
    point_color = [16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]
    scale = resize / max(frame.shape[0], frame.shape[1])

    # Ensure keypoints are valid before processing
    if keypoints is None or len(keypoints) == 0 or not isinstance(keypoints[0], np.ndarray) or keypoints[0].size == 0:
         img_resized = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
         return img_resized # Return resized frame if no keypoints

    try:
        scores = keypoints[..., 2]
        # Ensure scores array is broadcastable or matches keypoints structure
        if len(scores.shape) > 1 and scores.shape[0] != keypoints.shape[0]:
             # Attempt to fix mismatch if scores has an extra dimension (e.g., [[s1],[s2],..])
            if scores.shape[1] == 1:
                scores = scores.flatten()
            else: # Fallback if score shape is unexpected
                print(f"Warning: Unexpected scores shape {scores.shape} for keypoints shape {keypoints.shape}. Using default scores.")
                scores = np.ones(keypoints.shape[:1]) # Score per person
    except IndexError:
        # Handle cases where keypoints might not have a score dimension (e.g., shape (num_persons, num_kpts, 2))
        print(f"Warning: Keypoints shape {keypoints.shape} might be missing score dimension. Using default scores.")
        scores = np.ones(keypoints.shape[0]) # Score per person


    scaled_keypoints = (keypoints[..., :2] * scale).astype(int)
    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    # Draw bounding boxes with IDs if provided
    if bbox is not None and len(bbox) > 0:
        scaled_bbox = (bbox * scale).astype(int)
        for idx, box in enumerate(scaled_bbox):
             if box.shape == (4,): # Ensure box is valid
                pt1 = (box[0], box[1])
                pt2 = (box[2], box[3])
                color_idx = idx % len(bbox_colors)  # Cycle through bbox colors
                cv2.rectangle(img, pt1, pt2, bbox_colors[color_idx], 2, cv2.LINE_AA)
                # Add ID text with same color as bbox
                cv2.putText(img, f'ID: {idx}', (pt1[0], pt1[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_colors[color_idx], 2)

    # Keep original keypoint visualization with original palette
    for person_idx, (kpts, person_scores) in enumerate(zip(scaled_keypoints, scores)):
         # If scores were per-person, create per-keypoint scores for thresholding
        if person_scores.shape == (): # Single score per person
            kpt_scores = np.full(kpts.shape[0], person_scores)
        elif person_scores.shape[0] == kpts.shape[0]: # Per-keypoint scores provided
            kpt_scores = person_scores
        else:
            print(f"Warning: Score shape mismatch for person {person_idx}. Using default threshold.")
            kpt_scores = np.ones(kpts.shape[0]) # Fallback


        show = [0] * len(kpts)
        for i, ((u, v), color) in enumerate(zip(skeleton, link_color)):
            # Check indices are within bounds for kpts and kpt_scores
            if u < len(kpts) and v < len(kpts) and u < len(kpt_scores) and v < len(kpt_scores):
                if kpt_scores[u] > thr and kpt_scores[v] > thr:
                    if kpts[u].shape == (2,) and kpts[v].shape == (2,): # Ensure valid coords
                        cv2.line(img, tuple(kpts[u]), tuple(kpts[v]), palette[color % len(palette)], 1,
                                cv2.LINE_AA)
                        show[u] = show[v] = 1
        for j, (kpt, s, color) in enumerate(zip(kpts, show, point_color)):
            if s:
                if kpt.shape == (2,): # Ensure valid coord
                     cv2.circle(img, tuple(kpt), 1, palette[color % len(palette)], 2, cv2.LINE_AA)

    return img

def visualize_wholebody(frame, keypoints, frame_id, bbox=None, output_dir="res/", thr=0.3, resize=1280):
    """Visualize the wholebody keypoints and skeleton on image."""
    # Define skeleton connections (ensure this matches the keypoint structure)
    skeleton = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11),
                (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2),
                (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (15, 17),
                (15, 18), (15, 19), (16, 20), (16, 21), (16, 22), (91, 92),
                (92, 93), (93, 94), (94, 95), (91, 96), (96, 97), (97, 98),
                (98, 99), (91, 100), (100, 101), (101, 102), (102, 103),
                (91, 104), (104, 105), (105, 106), (106, 107), (91, 108),
                (108, 109), (109, 110), (110, 111), (112, 113), (113, 114),
                (114, 115), (115, 116), (112, 117), (117, 118), (118, 119),
                (119, 120), (112, 121), (121, 122), (122, 123), (123, 124),
                (112, 125), (125, 126), (126, 127), (127, 128), (112, 129),
                (129, 130), (130, 131), (131, 132)] # Ensure skeleton indices are valid for 133 keypoints

    palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]]

    bbox_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    ]

    link_color = [ # Ensure this matches the number of skeleton connections
        1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
        2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
        2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
    ] * 2 # Rough duplication assuming symmetry/pattern - NEEDS VERIFICATION

    point_color = [ # Ensure this matches the number of keypoints (e.g., 133)
        0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4,
        5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5,
        5, 6, 6, 6, 6, 1, 1, 1, 1, 3 # Needs exactly 133 colors if 133 kpts
    ]
    # Truncate or pad point_color to match expected keypoint count (e.g., 133)
    num_expected_kpts = 133
    if len(point_color) > num_expected_kpts:
        point_color = point_color[:num_expected_kpts]
    elif len(point_color) < num_expected_kpts:
        point_color.extend([3] * (num_expected_kpts - len(point_color))) # Pad with default color


    # Resize frame
    scale = resize / max(frame.shape[0], frame.shape[1])
    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    # Ensure keypoints are valid before scaling
    if keypoints is None or len(keypoints) == 0 or not isinstance(keypoints[0], np.ndarray) or keypoints[0].size == 0:
        return img # Return resized frame if no keypoints

    scaled_keypoints = (keypoints[..., :2] * scale).astype(int)


    # Draw bounding boxes if provided
    if bbox is not None and len(bbox) > 0:
        scaled_bbox = (bbox * scale).astype(int)
        for idx, box in enumerate(scaled_bbox):
            if box.shape == (4,): # Check valid box
                pt1 = (box[0], box[1])
                pt2 = (box[2], box[3])
                color_idx = idx % len(bbox_colors)
                cv2.rectangle(img, pt1, pt2, bbox_colors[color_idx], 2, cv2.LINE_AA)
                cv2.putText(img, f'ID: {idx}', (pt1[0], pt1[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_colors[color_idx], 2)

    # Draw keypoints and skeleton
    for person_kpts in scaled_keypoints:
         # Assume scores are present or default to 1 if not included in keypoints array
        if person_kpts.shape[1] == 3: # kpts include score
            scores = person_kpts[:, 2]
            kpts_coords = person_kpts[:, :2]
        else: # kpts are just coordinates
            scores = np.ones(len(person_kpts))
            kpts_coords = person_kpts

        # Draw keypoints first
        for i, (kpt, color_idx) in enumerate(zip(kpts_coords, point_color)):
             if kpt.shape == (2,) and scores[i] > thr: # Check coord shape and score threshold
                cv2.circle(img, tuple(kpt), 2, palette[color_idx % len(palette)], -1, cv2.LINE_AA)

        # Draw skeleton lines
        for i, ((u, v), color_idx) in enumerate(zip(skeleton, link_color)):
            # Check indices are valid for this person's keypoints and scores
            if u < len(kpts_coords) and v < len(kpts_coords) and u < len(scores) and v < len(scores):
                 if scores[u] > thr and scores[v] > thr:
                    pt1 = kpts_coords[u]
                    pt2 = kpts_coords[v]
                    if pt1.shape == (2,) and pt2.shape == (2,): # Ensure valid coordinates
                        cv2.line(img, tuple(pt1), tuple(pt2),
                                palette[color_idx % len(palette)], 2, cv2.LINE_AA)

    return img

# --- Helper Functions (Copied directly from rtmdeploy.py) ---

def load_pose_tracker_with_det(det_path,pose_path,device):
    print(f"Loading models on {device}...")
    tracker = PoseTracker(
        det_model=det_path,
        pose_model=pose_path,
        device_name=device) # Pass device string directly
    # Check if pose_path indicates coco or halpe to set sigmas
    if 'halpe' in pose_path.lower():
         # Halpe sigmas (example - replace with actual if available)
         # These might need adjustment based on the specific Halpe model/training
         keypoint_sigmas = np.array([
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089, 0.079, # Head, Neck, Hip
            0.079, 0.087, 0.087, 0.089, 0.089, 0.079, 0.079, 0.079 # Extra Halpe points
            ])[:26] # Ensure 26 sigmas if using halpe26
    elif 'coco' in pose_path.lower():
        keypoint_sigmas = [ # COCO sigmas
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
            0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ]
    else: # Default or assume wholebody - NEEDS adjustment if specific sigmas known
         # Wholebody sigmas (example - replace with actual if available)
        # Placeholder: Using COCO sigmas expanded, likely needs refinement
        coco_sigmas_base = np.array([
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
            0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089])
        # Heuristic: Use smaller sigmas for face/hands, larger for feet
        face_sigma = 0.020
        hand_sigma = 0.030
        foot_sigma = 0.100
        keypoint_sigmas = np.concatenate([
            coco_sigmas_base, # 17 body
            np.full(5, foot_sigma), # 5 foot (indices 17-21 for COCO-WholeBody v1?) - CHECK MAPPING
            np.full(68 - 22, face_sigma), # Face points (indices 22-67?)
            np.full(21, hand_sigma), # Left hand (indices 68-88?)
            np.full(21, hand_sigma)  # Right hand (indices 89-109?)
            # Adjust total count and sigma values based on the EXACT wholebody model variant (e.g., 133 kpts)
        ])[:133] # Ensure correct number for wholebody


    state = tracker.create_state(
        det_interval=1, det_min_bbox_size=10, keypoint_sigmas=keypoint_sigmas)
    print(f"Models loaded successfully on {device}.")
    return state,tracker

def get_frames_from_video(video_filename):
    frames = []
    cap = cv2.VideoCapture(video_filename)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_filename}")
        return []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print(f"Loaded {len(frames)} frames from {os.path.basename(video_filename)}")
    return frames

# --- Worker Function for Multiprocessing ---

def process_frames_on_gpu(args_tuple):
    """Worker function to process a chunk of frames on a specific GPU."""
    frames_chunk, gpu_id, det_path, pose_path, chunk_offset = args_tuple
    device = f"cuda:{gpu_id}"
    # print(f"Worker {os.getpid()} on GPU {gpu_id} processing {len(frames_chunk)} frames from offset {chunk_offset}...")

    kpts_list_chunk = []
    bboxes_list_chunk = []

    try:
        # Load model specifically on this device for this process
        state, tracker = load_pose_tracker_with_det(det_path, pose_path, device)

        # Use tqdm within the worker for progress on its chunk
        for frame_index, frame in enumerate(tqdm(frames_chunk, desc=f"GPU {gpu_id}", position=gpu_id, leave=False)):
            global_frame_index = chunk_offset + frame_index
            try:
                results = tracker(state, frame, detect=-1)
                keypoints, bboxes, _ = results
                kpts_list_chunk.append(keypoints)
                bboxes_list_chunk.append(bboxes)
            except Exception as e_inner:
                print(f"Error processing frame {global_frame_index} on {device}: {e_inner}")
                # Append empty results or handle as needed
                kpts_list_chunk.append([])
                bboxes_list_chunk.append([])

        print(f"Worker on GPU {gpu_id} finished processing chunk.")
        return kpts_list_chunk, bboxes_list_chunk
    except Exception as e_outer:
        print(f"Error initializing worker or during processing on {device}: {e_outer}")
        # Return empty lists matching the expected output structure on error
        return [[] for _ in frames_chunk], [[] for _ in frames_chunk]


# --- Main Execution Logic ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RTMDeploy Pose Estimation in parallel on multiple GPUs.")
    parser.add_argument("video_folder", type=str, help="Folder containing input video files.")
    parser.add_argument("--det_path", type=str, default="models/rtmdet-m", help="Path to the detection model folder.")
    parser.add_argument("--pose_path", type=str, default="models/rtmpose-x-b8", help="Path to the pose model folder.")
    # No --device argument needed, as we use all available CUDA devices
    parser.add_argument("--res_folder", type=str, default="res_dist", help="Folder to save results (JSON and optional frames).")
    parser.add_argument("--visualize", action="store_true", help="Generate visualized frames for each video.")
    parser.add_argument("--resize", type=int, default=1280, help="Target size for resizing visualization output.")
    parser.add_argument("--vis_thr", type=float, default=0.5, help="Keypoint score threshold for visualization.")

    args = parser.parse_args()

    # --- Input Validation ---
    assert os.path.isdir(args.video_folder), f"Video folder '{args.video_folder}' not found."
    assert os.path.isdir(args.det_path), f"Detection model folder '{args.det_path}' not found."
    assert os.path.isdir(args.pose_path), f"Pose model folder '{args.pose_path}' not found."
    os.makedirs(args.res_folder, exist_ok=True)

    # --- GPU Check ---
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This script requires CUDA-enabled GPUs.")
        exit(1)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
         print("Error: No CUDA GPUs detected. This script requires CUDA-enabled GPUs.")
         exit(1)
    print(f"Found {num_gpus} CUDA GPU(s).")
    gpu_ids = list(range(num_gpus))

    # --- Get Video Files ---
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv') # Added mkv
    video_files = sorted([ # Sort for consistent processing order
        os.path.join(args.video_folder, f) for f in os.listdir(args.video_folder)
        if f.lower().endswith(video_extensions) and os.path.isfile(os.path.join(args.video_folder, f))
    ])

    if not video_files:
        print(f"No video files with extensions {video_extensions} found in '{args.video_folder}'.")
        exit(0)

    print(f"Found {len(video_files)} videos to process: {video_files}")

    # Set multiprocessing start method for CUDA safety
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing context already set. Using existing context.")


    # --- Process Each Video ---
    overall_start_time = time.time()
    for video_filename in video_files:
        video_start_time = time.time()
        print(f"===================================")
        print(f"Processing {os.path.basename(video_filename)}")
        print(f"===================================")
        video_name = os.path.basename(video_filename).rsplit('.', 1)[0]

        # --- Setup Result Folders ---
        video_res_folder = os.path.join(args.res_folder, video_name)
        json_folder = os.path.join(video_res_folder, 'json')
        image_folder = os.path.join(video_res_folder, 'frames')
        os.makedirs(json_folder, exist_ok=True)
        if args.visualize:
            os.makedirs(image_folder, exist_ok=True)

        kpts_json = os.path.join(json_folder, 'keypoints.json')
        bbox_json = os.path.join(json_folder, 'bboxes.json')

        # --- Check Existing or Process ---
        if os.path.exists(kpts_json) and os.path.exists(bbox_json):
            print(f"Found existing JSON files in {json_folder}. Loading them.")
            try:
                with open(kpts_json, "r") as f:
                    kp_list = json.load(f)
                with open(bbox_json, "r") as f:
                    bboxes = json.load(f)
                # Convert loaded lists back to numpy arrays if needed for visualization consistency
                # This assumes the structure saved is list of lists/arrays per frame
                bboxes_np = [[np.array(b) for b in frame_bboxes] for frame_bboxes in bboxes]
                kp_list_np = [[np.array(p) for p in frame_kps] for frame_kps in kp_list]

                # Load frames only if visualizing
                frames = []
                if args.visualize:
                    frames = get_frames_from_video(video_filename)
                    if len(frames) != len(kp_list_np):
                         print(f"Warning: Number of frames ({len(frames)}) does not match number of keypoint entries ({len(kp_list_np)}). Visualization might be incomplete.")

            except json.JSONDecodeError as e:
                print(f"Error loading JSON files: {e}. Reprocessing video.")
                kp_list, bboxes, frames, kp_list_np, bboxes_np = [], [], [], [], [] # Reset lists
            except Exception as e:
                 print(f"An unexpected error occurred loading files: {e}. Reprocessing video.")
                 kp_list, bboxes, frames, kp_list_np, bboxes_np = [], [], [], [], [] # Reset lists

        else:
            print("Existing JSON not found or incomplete. Processing video...")
            # Load all frames first
            all_frames = get_frames_from_video(video_filename)
            num_frames = len(all_frames)
            if num_frames == 0:
                print(f"Skipping {video_filename} as no frames were loaded.")
                continue

            # Split frames into chunks for each GPU
            # Use np.array_split for potentially uneven division
            frame_chunks = np.array_split(all_frames, num_gpus)

            # Calculate offsets for correct frame indexing in logs/errors
            chunk_offsets = [0] * num_gpus
            current_offset = 0
            for i in range(num_gpus):
                chunk_offsets[i] = current_offset
                current_offset += len(frame_chunks[i])


            # Prepare arguments for worker processes
            worker_args = [(list(chunk), gpu_id, args.det_path, args.pose_path, chunk_offsets[gpu_id])
                           for gpu_id, chunk in zip(gpu_ids, frame_chunks)]

            print(f"Launching {num_gpus} worker processes for {num_frames} frames...")
            inference_start_time = time.time()
            kp_list_combined = []
            bboxes_combined = []
            try:
                with multiprocessing.Pool(processes=num_gpus) as pool:
                    # map preserves order of chunks
                    results_chunks = pool.map(process_frames_on_gpu, worker_args)

                # Combine results from all GPUs in order
                print("Combining results from workers...")
                for kpts_chunk, bboxes_chunk in results_chunks:
                    kp_list_combined.extend(kpts_chunk)
                    bboxes_combined.extend(bboxes_chunk)

                inference_end_time = time.time()
                print(f"Inference completed in {inference_end_time - inference_start_time:.2f} seconds.")

                # Assign combined results
                kp_list = kp_list_combined
                bboxes = bboxes_combined
                frames = all_frames # We already have all frames loaded

                # Convert to numpy arrays for saving/visualization consistency
                bboxes_np = [[np.array(b) for b in frame_bboxes] for frame_bboxes in bboxes]
                kp_list_np = [[np.array(p) for p in frame_kps] for frame_kps in kp_list]


                # Save keypoints and bbox to json files
                print(f"Saving results to {json_folder}...")
                try:
                    with open(kpts_json, "w") as f:
                        json.dump(kp_list_np, f, cls=NumpyEncoder) # Save the np-converted list
                    with open(bbox_json, "w") as f:
                        json.dump(bboxes_np, f, cls=NumpyEncoder) # Save the np-converted list
                    print("JSON results saved.")
                except Exception as e:
                    print(f"Error saving JSON results: {e}")

            except Exception as pool_error:
                print(f"Error during multiprocessing pool execution: {pool_error}")
                print(f"Skipping saving and visualization for {video_filename}.")
                kp_list, bboxes, frames, kp_list_np, bboxes_np = [], [], [], [], [] # Ensure lists are empty on error


        # --- Visualize Keypoints (if requested and data is available) ---
        if args.visualize and frames and kp_list_np and bboxes_np:
            print(f"Generating visualizations for {video_name}...")
            vis_start_time = time.time()
             # Ensure visualization runs on the main process CPU, not workers
            for i in tqdm(range(min(len(frames), len(kp_list_np), len(bboxes_np)))): # Use min length to avoid index errors
                try:
                    frame_to_vis = np.array(frames[i]) # Ensure it's a numpy array
                    keypoints_to_vis = kp_list_np[i] # Already list of np arrays
                    bbox_to_vis = bboxes_np[i]       # Already list of np arrays

                    # Choose the correct visualization function based on pose model path
                    if "halpe" in args.pose_path.lower():
                        img = visualize_halpe(frame_to_vis, keypoints_to_vis, i, bbox_to_vis, thr=args.vis_thr, resize=args.resize)
                    elif "coco" in args.pose_path.lower(): # Assume standard COCO if not halpe or wholebody
                         img = visualize(frame_to_vis, keypoints_to_vis, i, bbox_to_vis, thr=args.vis_thr, resize=args.resize)
                    else: # Default to wholebody or specific logic if path contains 'whole' etc.
                         # Add specific check like 'whole' if needed: elif "whole" in args.pose_path.lower():
                         img = visualize_wholebody(frame_to_vis, keypoints_to_vis, i, bbox_to_vis, thr=args.vis_thr, resize=args.resize)

                    cv2.imwrite(os.path.join(image_folder, f"{i:06d}.jpg"), img)
                except Exception as vis_e:
                    print(f"Error visualizing frame {i}: {vis_e}")
                    # Optionally, break or continue

            vis_end_time = time.time()
            print(f"Visualizations saved to {image_folder} in {vis_end_time - vis_start_time:.2f} seconds.")
        elif args.visualize:
            print("Skipping visualization due to missing frames or processed data.")

        video_end_time = time.time()
        print(f"Completed processing {os.path.basename(video_filename)} in {video_end_time - video_start_time:.2f} seconds.")

    overall_end_time = time.time()
    print(f"-----------------------------------")
    print(f"Finished processing all {len(video_files)} videos.")
    print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds.")
    print(f"Results saved in: {args.res_folder}")
    print(f"-----------------------------------") 