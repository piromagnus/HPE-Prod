# RTMpose Production

## Introduction
Easy to run repo to get the pose estimation of a person in a video. Based on the repo [RTMpose](https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose).
This repo use MMDeploy to deploy the model in a docker container in an easy way.

## Models
It contains severals models within the folder `models`:
- human detection models : `rtmdet-m` (default) and `rtmdet-nano`. Modify the dockerfile and rebuild to change
- human pose estimation models : (cf `rtmpose_prod/rtmdeploy_run.sh` to change the script parameters)
    - `coco`: `rtmpose-x-b8` (default)
    - `halpe`: `rtmpose-x-b8-halpe`
    - `wholebody`: `rtmw-x` (based on Coco-Wholebody)

## Quick Start

0. Download the models with the script 
```bash
    cd rtmpose_prod/scripts
    bash download_models.sh
```
1. build the docker within rtmpose_prod folder
    ```bash
    docker build -t rtmdeploy_prod .
    ```


2. then you can run 
```bash
    sh rtmpose_prod/rtmdeploy_run.sh [-z for viz] -v path/to/videos_folder -m [coco|halpe|wholebody] [-i image_name (default : rtmdeploy_prod)] [-r path/to/res (default:res)]
```
the results will be in the `path/to/videos_folder/path/to/res`
