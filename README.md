# Cross fence detection

Using the YOLOv8n model to detect whether the position of a person intersects with a manually drawn line, and if so, it is considered as an act of climbing over the fence.

## Hardware prepare

|                                               reComputer AI Box                                            |                                              reComputer Industry AI Box                                             |
| :----------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
| ![Raspberry Pi AI Kit](https://media-cdn.seeedstudio.com/media/catalog/product/cache/bb49d3ec4ee05b6f018e93f896b8a25d/i/m/image114993560.jpeg) | ![reComputer R1100](https://media-cdn.seeedstudio.com/media/catalog/product/cache/bb49d3ec4ee05b6f018e93f896b8a25d/2/-/2-114993595-recomputer-ai-industrial-r2135-12.jpg) |
| [**Purchase Now**](https://www.seeedstudio.com/reComputer-AI-R2130-12-p-6368.html?utm_source=PiAICourse&utm_medium=github&utm_campaign=Course) | [**Purchase Now**](https://www.seeedstudio.com/reComputer-AI-Industrial-R2135-12-p-6432.html?utm_source=PiAICourse&utm_medium=github&utm_campaign=Course) |

## Software prepare

### Install the runtime environment 

```bash
sudo apt update && sudo apt full-upgrade -y && sudo apt install hailo-all
```

### Install the project

```bash
git clone https://github.com/Seeed-Projects/cross_fence_with_AIBox.git
cd cross_fence_with_AIBox
```

### Prepare the python environment

```bash
python -m venv .env --system-site-packages  && source .env/bin/activate
pip install -r requirements.txt
```

## Run the project

```bash
python app.py -i ./scale_the_fence.mp4 -n ./yolov8n.hef --show-fps -l ./common/coco.txt 
```
## Result

[![Watch the video](https://img.youtube.com/vi/_-89czNbZ_M/0.jpg)](https://www.youtube.com/watch?v=_-89czNbZ_M)


## Reference

This project references  [Hailo-Application-Code-Examples](https://github.com/hailo-ai/Hailo-Application-Code-Examples).
