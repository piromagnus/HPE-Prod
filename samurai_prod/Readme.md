# Dockerfication of Samurai

## Informations
Based on the repo [samurai](https://github.com/yangchris11/samurai) which is a improvement of [SAM2](https://github.com/facebookresearch/sam2)


This repo does a tracking of an object defined by a bounding box of the first frame over all the videos and segments the object in each frame. (see [data](format.md) for more details)


## Quick Start
1. build the docker within rtmpose_prod folder
    ```bash
    docker build -t samurai_prod .
    ```

2. then you can run 
```bash
    sh samurai_prod/run_samurai.sh -v path/to/videos_folder
```
the results will be in each folder following the structure of [data](format.md)


## Get the bbox from the first frame
0. You need numpy and opencv
```bash
pip install numpy opencv-python
```
1. run the following command
```bash
python samurai_prod/get_bbox.py --video_path path/to/video --text_path path/to/text_file
```
where the text file is the output file that contains the selected bounding box for the first frame of the the video
2. Select the bbox by sliding the mouse on the first frame of the video
3. Press 'q' to save the bbox



# Acknowledgements
This repo is based on the work of [samurai](https://github.com/yangchris11/samurai).
The demo is a part of CMU Panoptic Studio project.