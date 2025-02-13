
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
- the 
- Scrap all the folder and run on the mp4 video that is not the video_output, the txt will be named bbox.txt 
  

