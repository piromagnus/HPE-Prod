import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor



models ={
    "large": "sam2/checkpoints/sam2.1_hiera_large.pt",
    "base_plus": "sam2/checkpoints/sam2.1_hiera_base_plus.pt",
    "small": "sam2/checkpoints/sam2.1_hiera_small.pt",
    "tiny": "sam2/checkpoints/sam2.1_hiera_tiny.pt"
}


def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts


def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")

def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4 or a directory of jpg frames.")


def load_frames(video_path):
    if osp.isdir(video_path):
        frames = sorted([osp.join(video_path, f) for f in os.listdir(video_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
        loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
        height, width = loaded_frames[0].shape[:2]
        frame_rate = 30
    else:
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        loaded_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            loaded_frames.append(frame)
        cap.release()
        height, width = loaded_frames[0].shape[:2]
        print(f"Loaded {len(loaded_frames)} frames from the video.")
        if len(loaded_frames) == 0:
            raise ValueError("No frames were loaded from the video.")
    return loaded_frames, height, width, frame_rate

def process_video(frame_idx, object_ids, masks,out,loaded_frames,height,width,is_mask,is_bbox,is_back):
    mask_to_vis = {}
    bbox_to_vis = {}
    for obj_id, mask in zip(object_ids, masks):
        mask = mask[0].cpu().numpy()
        mask = mask > 0.0
        non_zero_indices = np.argwhere(mask)
        if len(non_zero_indices) == 0:
            bbox = [0, 0, 0, 0]
        else:
            y_min, x_min = non_zero_indices.min(axis=0).tolist()
            y_max, x_max = non_zero_indices.max(axis=0).tolist()
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
        bbox_to_vis[obj_id] = bbox
        mask_to_vis[obj_id] = mask

        img = loaded_frames[frame_idx]
        
            # Create binary mask (0 or 255)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        if is_back:
            if is_mask:
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width,3), dtype=np.uint8)
                    mask_img[mask] = colors[obj_id % len(colors)]
                    img = cv2.addWeighted(img, 0.5, mask_img, 0.5, 0)
            if is_bbox:
                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[obj_id % len(colors)], 2)
        else:

            if is_mask:
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width,3), dtype=np.uint8)
                    mask_img[mask] = colors[obj_id % len(colors)]
                    img = mask_img
            if is_bbox:
                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[obj_id % len(colors)], 2)

        out.write(img) 


def main(args):
    model_path = models[args.model]
    model_cfg = determine_model_cfg(model_path)
    predictor = build_sam2_video_predictor(model_cfg,model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = load_txt(args.txt_path)
    # if args.save_to_video:
    loaded_frames, height, width,frame_rate = load_frames(args.video_path)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))# isColor=False)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=False)
        bbox, track_label = prompts[0]
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)
        print("propagating in video")
        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            process_video(frame_idx, object_ids, masks,out,loaded_frames,height,width,args.mask,args.bbox,args.background)
    
    out.release()
    del predictor, state
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video directory containing the folders for each video")
    # parser.add_argument("--cfg_path", required=True, help="Path to ground truth text file.")
    # parser.add_argument("--model", default="large", help="Path to the model checkpoint.")
    # parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    # parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    # parser.add_argument("--bbox",action="store_true", default=False, help="Viz bbox.")
    # parser.add_argument("--mask",action="store_true", default=True, help="Viz mask.")
    # parser.add_argument("--background", action="store_true", default=False, help="Viz background.")
    args = parser.parse_args()
    root = args.video_path
    import json
    listdir= os.listdir(root)
    if "cfg.json" in listdir:
        with open(os.path.join(root,"cfg.json"), "r") as f:
            cfg = json.load(f)
        args.model = cfg["model"]
        video_output = cfg["video_output"]
        args.bbox = cfg["bbox"]
        args.mask = cfg["mask"]
        args.background = cfg["background"]
    else:
        args.model = "large"
        video_output = "res.mp4"
        args.bbox = False
        args.mask = True
        args.background = False

    for i in listdir:

        folder_path = os.path.join(root,i)
        print(f"Processing {folder_path}")
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            if "cfg.json" in files:
                with open(os.path.join(folder_path,"cfg.json"), "r") as f:
                    cfg = json.load(f)
                args.model = cfg["model"]
                video_output = cfg["video_output"]
                args.bbox = cfg["bbox"]
                args.mask = cfg["mask"]
                args.background = cfg["background"]

            ext = [".mp4",".avi",".mov"]
            video_files = [f for f in files if f.endswith(tuple(ext))]
            args.txt_path = os.path.join(folder_path,"bbox.txt")
            for video in video_files:
                if video != video_output:
                    args.video_path = os.path.join(folder_path,video)
                    break
            args.video_output_path = os.path.join(folder_path,video_output)
            main(args)
