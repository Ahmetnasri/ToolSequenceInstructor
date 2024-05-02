## 👋 hello

**Increase the success and accuracy of manual processes with Computer Vision.** 🤝
The ToolSequenceAnalyzer is an intelligent system designed to analyze instructional videos and provide guidance on the correct sequence for tool selection and usage. It's crucial to ensure that viewers understand the proper order of tool selection to complete tasks efficiently and safely. This tool utilizes the YOLOV9 architecture to detect and analyze the sequence of tools being used in a video. 
It then provides real-time feedback to the user, indicating whether the tool selection was correct or not. The ToolSequenceAnalyzer aims to enhance learning experiences and promote safer practices in various domains such as woodworking, mechanical repairs, and more. 




Here is the output:

![ezgif-4-fca1389192](https://github.com/Ahmetnasri/ToolSequenceInstructor/assets/63724301/bca4ef68-7304-45f6-83b3-7a7e2e850223)

## Installation
For the installation, follow the instruction in the [YOLOV9](https://github.com/WongKinYiu/yolov9) repository

## Inference
You can change the instruction of the required sequence by editing the file `instruction.yaml`, in the example video the sequence was: screwdriver - plier - screwdriver - hammer

``` shell
# inference converted yolov9 models
python ToolSequenceInstructor.py --source 6.mp4 --img 640 --device 0 --weights best.pt --xmin 0 --xmax 1 --ymin 0.33 --ymax 0.75 --instruction instruction.yaml
```
Where the parameters `xmin` `xmax` `ymin` `xmax` refer to the region if intreset
