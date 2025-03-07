# Production Models Repository

This repository contains production-ready implementations of various AI models, specifically focused on computer vision tasks.

## Components

### RTMpose Production
A dockerized implementation of RTMpose for human pose estimation. Supports multiple models including COCO, Halpe, and Wholebody variants.
[Learn more](rtmpose_prod/Readme.md)

### Samurai Production
A dockerized implementation for object tracking and segmentation based on SAM (Segment Anything Model).
[Learn more](samurai_prod/Readme.md)

## Usage
Each component has its own Docker configuration and usage instructions. Please refer to the individual README files in each subdirectory for specific setup and running instructions.

## Structure
```
.
├── rtmpose_prod/    # RTMpose implementation
├── samurai_prod/    # SAM-based tracking implementation
└── README.md
```


## License
This repository is licensed under the Apache License. See the [LICENSE](LICENSE) file for more details.