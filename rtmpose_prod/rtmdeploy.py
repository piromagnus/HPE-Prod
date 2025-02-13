from mmdeploy_runtime import PoseTracker
import os
from tqdm import tqdm
import cv2
import numpy as np
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
    keypoints = (keypoints[..., :2] * scale).astype(int)
    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    # Draw bounding boxes if provided
    if bbox is not None:
        scaled_bbox = (bbox * scale).astype(int)
        for idx, box in enumerate(scaled_bbox):
            pt1 = (box[0], box[1])
            pt2 = (box[2], box[3])
            color_idx = idx % len(palette)
            cv2.rectangle(img, pt1, pt2, palette[color_idx], 2, cv2.LINE_AA)
            cv2.putText(img, f'ID: {idx}', (pt1[0], pt1[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, palette[color_idx], 2)

    # Draw skeleton
    for person_keypoints in keypoints:
        for idx, (start_idx, end_idx) in enumerate(skeleton):   
            if start_idx < len(person_keypoints) and end_idx < len(person_keypoints):
                pt1 = tuple(person_keypoints[start_idx])
                pt2 = tuple(person_keypoints[end_idx])
                color = line_color[idx % len(line_color)]
                cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)

        # Draw keypoints
        for idx, kpt in enumerate(person_keypoints):
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
    try:
        scores = keypoints[..., 2]
    except:
        print("keypoints shape",keypoints.shape)
        print(keypoints)
        scores = np.ones(keypoints.shape[0])
    keypoints = (keypoints[..., :2] * scale).astype(int)
    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    # Draw bounding boxes with IDs if provided
    if bbox is not None:
        scaled_bbox = (bbox * scale).astype(int)
        for idx, box in enumerate(scaled_bbox):
            pt1 = (box[0], box[1])
            pt2 = (box[2], box[3])
            color_idx = idx % len(bbox_colors)  # Cycle through bbox colors
            cv2.rectangle(img, pt1, pt2, bbox_colors[color_idx], 2, cv2.LINE_AA)
            # Add ID text with same color as bbox
            cv2.putText(img, f'ID: {idx}', (pt1[0], pt1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_colors[color_idx], 2)

    # Keep original keypoint visualization with original palette
    for kpts, score in zip(keypoints, scores):
        show = [0] * len(kpts)
        for (u, v), color in zip(skeleton, link_color):
            if score[u] > thr and score[v] > thr:
                cv2.line(img, kpts[u], tuple(kpts[v]), palette[color], 1,
                         cv2.LINE_AA)
                show[u] = show[v] = 1
        for kpt, show, color in zip(kpts, show, point_color):
            if show:
                cv2.circle(img, kpt, 1, palette[color], 2, cv2.LINE_AA)

    return img

def visualize_wholebody(frame, keypoints, frame_id, bbox=None, output_dir="res/", thr=0.3, resize=1280):
    """Visualize the wholebody keypoints and skeleton on image.

    Args:
        frame (np.ndarray): Input image
        keypoints (np.ndarray): Keypoints in image
        frame_id (int): Frame ID for saving
        bbox (np.ndarray, optional): Bounding boxes. Defaults to None.
        output_dir (str, optional): Output directory. Defaults to "res/".
        thr (float, optional): Threshold for keypoint visibility. Defaults to 0.3.
        resize (int, optional): Target size for resizing. Defaults to 1280.

    Returns:
        np.ndarray: Visualized image
    """
    # Define skeleton connections
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
                (129, 130), (130, 131), (131, 132)]

    palette = [[51, 153, 255], [0, 255, 0], [255, 128, 0], [255, 255, 255],
               [255, 153, 255], [102, 178, 255], [255, 51, 51]]
    
    bbox_colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    ]

    link_color = [
        1, 1, 2, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
        2, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 2, 2, 2,
        2, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1
    ]

    point_color = [
        0, 0, 0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 4, 4, 4, 4,
        5, 5, 5, 5, 6, 6, 6, 6, 1, 1, 1, 1, 3, 2, 2, 2, 2, 4, 4, 4, 4, 5, 5, 5,
        5, 6, 6, 6, 6, 1, 1, 1, 1
    ]

    # Resize frame
    scale = resize / max(frame.shape[0], frame.shape[1])
    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    keypoints = (keypoints[..., :2] * scale).astype(int)

    # Draw bounding boxes if provided
    if bbox is not None:
        scaled_bbox = (bbox * scale).astype(int)
        for idx, box in enumerate(scaled_bbox):
            pt1 = (box[0], box[1])
            pt2 = (box[2], box[3])
            color_idx = idx % len(bbox_colors)
            cv2.rectangle(img, pt1, pt2, bbox_colors[color_idx], 2, cv2.LINE_AA)
            cv2.putText(img, f'ID: {idx}', (pt1[0], pt1[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_colors[color_idx], 2)

    # Draw keypoints and skeleton
    for kpts in keypoints:
        scores = np.ones(len(kpts))  # If no scores provided, assume all visible
        for kpt, color in zip(kpts, point_color):
            cv2.circle(img, tuple(kpt), 2, palette[color], -1, cv2.LINE_AA)
            
        for (u, v), color in zip(skeleton, link_color):
            if u < len(kpts) and v < len(kpts) and scores[u] > thr and scores[v] > thr:
                cv2.line(img, tuple(kpts[u]), tuple(kpts[v]), 
                        palette[color], 2, cv2.LINE_AA)

    return img

def load_pose_tracker_with_det(det_path,pose_path,device):
    tracker = PoseTracker(
        det_model=det_path,
        pose_model=pose_path,
        device_name=device)
    coco_sigmas = [
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062,
        0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ]
    state = tracker.create_state(
        det_interval=1, det_min_bbox_size=10, keypoint_sigmas=coco_sigmas)
    return state,tracker

def get_kpts_from_tracker(tracker,frames,state):
    kpts_list=[]
    for frame in tqdm(frames):
        results = tracker(state, frame, detect=-1)
        keypoints, bboxes, _ = results
        kpts_list.append(keypoints)
    return kpts_list


def get_frames_from_video(video_filename):
    frames = []
    cap = cv2.VideoCapture(video_filename)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def get_kpts_from_video(video_filename, state,tracker):
   
    kpts_list = []
    bboxes_list = []
    frames = []
    cap = cv2.VideoCapture(video_filename)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        results = tracker(state, frame, detect=-1)
        keypoints, bboxes, _ = results
        kpts_list.append(keypoints)
        bboxes_list.append(bboxes)
        # if i >10:
        #     break
    # kpts_list = np.array(kpts_list)
    # bboxes_list = np.array(bboxes_list)
    # frames = np.array(frames)
    return kpts_list, bboxes_list, frames



if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video_folder", type=str)
    parser.add_argument("--det_path", type=str, default="models/rtmdet-m")
    parser.add_argument("--pose_path", type=str, default="models/rtmpose-x-b8")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--res_folder", type=str, default="res")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    # Verify input folder exists
    assert os.path.exists(args.video_folder), f"Video folder {args.video_folder} not found"
    assert os.path.exists(args.det_path), "Detection model not found"
    assert os.path.exists(args.pose_path), "Pose model not found"
    os.makedirs(args.res_folder, exist_ok=True)

    # Get all video files
    video_extensions = ('.mp4', '.avi', '.mov')
    video_files = [
        os.path.join(args.video_folder, f) for f in os.listdir(args.video_folder)
        if f.lower().endswith(video_extensions)
    ]
    
    print(f"Found {len(video_files)} videos to process")
    state, tracker = load_pose_tracker_with_det(args.det_path, args.pose_path, args.device)
    # Process each video
    for video_filename in video_files:
        print(f"\nProcessing {video_filename}")
        video_name = os.path.basename(video_filename).rsplit('.', 1)[0]
        
        # Create result folders
        video_res_folder = os.path.join(args.res_folder, video_name)
        json_folder = os.path.join(video_res_folder, 'json')
        image_folder = os.path.join(video_res_folder, 'frames')
        os.makedirs(json_folder, exist_ok=True)
        os.makedirs(image_folder, exist_ok=True)

        # Define json file paths
        kpts_json = os.path.join(json_folder, 'keypoints.json')
        bbox_json = os.path.join(json_folder, 'bboxes.json')

        if not os.path.exists(kpts_json) or not os.path.exists(bbox_json):
            kp_list, bboxes, frames = get_kpts_from_video(
                video_filename, state, tracker)

            # Save keypoints and bbox to json files
            with open(kpts_json, "w") as f:
                json.dump(kp_list, f, cls=NumpyEncoder)
            with open(bbox_json, "w") as f:
                json.dump(bboxes, f, cls=NumpyEncoder)
        else:
            # choice = input("Found existing json files in {}. Do you want to overwrite them? (y/n): ".format(json_folder))
            # if choice.lower() == "y":
            #     kp_list, bboxes, frames = get_kpts_from_video(
            #         video_filename, state, tracker)
            #     with open(kpts_json, "w") as f:
            #         json.dump(kp_list, f, cls=NumpyEncoder)
            #     with open(bbox_json, "w") as f:
            #         json.dump(bboxes, f, cls=NumpyEncoder)
            # else:
            print("Found existing json files")
            print("Loading existing json files")
            with open(kpts_json, "r") as f:
                kp_list = json.load(f)
            with open(bbox_json, "r") as f:
                bboxes = json.load(f)
            frames = get_frames_from_video(video_filename)
            bboxes = [[np.array(b) for b in f] for f in bboxes]
            kp_list = [[np.array(b) for b in f] for f in kp_list]

        # Visualize keypoints
        if args.visualize:
            print(f"Generating visualizations for {video_name}")
            for i in tqdm(range(len(frames))):
                frame = np.array(frames[i])
                keypoints = np.array(kp_list[i])
                bbox = np.array(bboxes[i])
                img = visualize(frame, keypoints, i, bbox) if "coco" in args.pose_path else (visualize_halpe(frame, keypoints, i, bbox) if "halpe" in args.pose_path else visualize_wholebody(frame, keypoints, i, bbox))
                cv2.imwrite(os.path.join(image_folder, f"{i:06d}.jpg"), img)

            print(f"Completed processing {video_name} with visualizations at {image_folder}")
        else:
            print(f"Completed processing {video_name} without visualizations")
