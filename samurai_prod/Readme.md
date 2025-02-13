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
