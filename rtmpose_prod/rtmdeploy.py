from mmdeploy_runtime import PoseTracker
import os
from tqdm import tqdm
import cv2
import numpy as np
import json
import multiprocessing
from utils.viz import visualize, visualize_halpe, visualize_wholebody

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)




# Helper function for parallel visualization
def visualize_and_save_frame(args_tuple):
    i, frame_data, kp_data, bbox_data, image_folder, pose_path_type = args_tuple
    
    # Reconstruct numpy arrays if they were serialized
    frame = np.array(frame_data)
    keypoints = np.array(kp_data)
    bbox = np.array(bbox_data) if bbox_data is not None else None # Handle potential None bbox

    output_path = os.path.join(image_folder, f"{i:06d}.jpg")

    # Check if image already exists to avoid re-computation
    if not os.path.exists(output_path):
        if "coco" in pose_path_type:
            img = visualize(frame, keypoints, i, bbox)
        elif "halpe" in pose_path_type:
            img = visualize_halpe(frame, keypoints, i, bbox)
        else:
            img = visualize_wholebody(frame, keypoints, i, bbox)
        
        cv2.imwrite(output_path, img)
    # Optionally return status or path
    # return output_path 

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
            
            #save with npz
            np.savez(os.path.join(json_folder, 'keypoints.npz'), kp_list)
            np.savez(os.path.join(json_folder, 'bboxes.npz'), bboxes)

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
            try:
                print("Loading existing npz files")
                kp_list = np.load(os.path.join(json_folder, 'keypoints.npz'))
                bboxes = np.load(os.path.join(json_folder, 'bboxes.npz'))
            except:
                try:
                    print("Loading existing json files")
                    with open(kpts_json, "r") as f:
                        kp_list = json.load(f)
                    with open(bbox_json, "r") as f:
                        bboxes = json.load(f)
                except:
                    raise ValueError("No existing json or npz files found")
            frames = get_frames_from_video(video_filename)
            bboxes = [[np.array(b) for b in f] for f in bboxes]
            kp_list = [[np.array(b) for b in f] for f in kp_list]

        # Visualize keypoints
        if args.visualize: # Removed check for existing files here, handled in worker
            print(f"Generating visualizations for {video_name}")
            
            # Prepare arguments for parallel processing
            num_frames = len(frames)
            vis_args = []
            for i in range(num_frames):
                # Ensure data exists for the frame index i
                frame_arg = frames[i] if i < len(frames) else None
                kp_arg = kp_list[i] if i < len(kp_list) else None
                bbox_arg = bboxes[i] if i < len(bboxes) else None # Handle case where bboxes might be shorter

                # Only add arguments if frame data is valid
                if frame_arg is not None and kp_arg is not None:
                     # Pass necessary components instead of full args object
                    vis_args.append((i, frame_arg, kp_arg, bbox_arg, image_folder, args.pose_path))

            # Determine number of processes
            n_cpu = multiprocessing.cpu_count()
            print(f"Using {n_cpu} processes for visualization.")

            # Use multiprocessing Pool
            with multiprocessing.Pool(processes=n_cpu) as pool:
                # Use tqdm to show progress with starmap
                list(tqdm(pool.starmap(visualize_and_save_frame, vis_args), total=len(vis_args)))


            # Original sequential loop - commented out
            # for i in tqdm(range(len(frames))):
            #     # Check if image already exists
            #     output_path = os.path.join(image_folder, f"{i:06d}.jpg")
            #     if not os.path.exists(output_path):
            #         frame = np.array(frames[i])
            #         # Handle potential mismatch in list lengths if loading failed partially
            #         if i < len(kp_list) and i < len(bboxes):
            #             keypoints = np.array(kp_list[i])
            #             bbox = np.array(bboxes[i])
            #             img = visualize(frame, keypoints, i, bbox) if "coco" in args.pose_path else (visualize_halpe(frame, keypoints, i, bbox) if "halpe" in args.pose_path else visualize_wholebody(frame, keypoints, i, bbox))
            #             cv2.imwrite(output_path, img)
            #         else:
            #             print(f"Warning: Skipping frame {i} due to missing keypoints or bounding box data.")


            print(f"Completed processing {video_name} with visualizations at {image_folder}")
        else:
            print(f"Completed processing {video_name} without visualizations")
