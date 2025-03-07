
For each video to process :
```
.
├── cfg.json
└── video1
    ├── bbox.txt
    ├── cfg.json (facultatif)
    ├── video1.mp4
    └── res.mp4

```


cfg.json : 
```json
{
    "model" : "large",
    "bbox" : true,
    "mask" : true,
    "background" : true,
    "video_output" : "res.mp4"
}
```


# Rules
- default values.
- cfg.json in the root -> for every video
- cfg.json in the video_folder -> for the video only, overwrite the general.
- Scrap all the folder and run on the mp4 video that is not the video_output, the txt will be named bbox.txt 
Note: The .txt file contains a single line with the bounding box of the first frame in x,y,w,h format while the SAM 2 takes x1,y1,x2,y2 format as bbox input.

