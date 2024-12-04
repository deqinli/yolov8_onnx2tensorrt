# yolov8_onnx2tensorrt
---
## vs2022 C++  tensorrt 推理加速版本

### 1. 使用环境：
   win10，vs2022，C++，,显卡A6000显存48G，cuda12.0，cudnn-windows-x86_64-8.9.7.29_cuda12-archive，tensorrt8.6.1.6，
    opencv4.10.0

### 2. 引入头文件和库文件（文件夹）
   - "CUDA路径/CUDA/v12.0/include"
   - "opencv路径/opencv4.10.0/include"
   - "opencv路径/opencv4.10.0/include/opencv2"
   - "Tensorrt路径/TensorRT8.6.1.6/include"
   - "Tensorrt路径/TensorRT8.6.1.6/samples/common"
   
   + "CUDA路径/CUDA/v12.0/lib/x64"
   + "opencv路径/opencv4.10.0/x64/vc17/lib"
   + "Tensorrt路径/TensorRT8.6.1.6/lib"
   
### 3. 链接库文件名
   + cuda.lib
   + cudnn.lib
   + cublas.lib
   + cudart.lib
   + nvinfer.lib
   + nvparsers.lib
   + nvonnxparser.lib
   + nvinfer_plugin.lib
   + opencv_world4100d.lib（debug）
   + opencv_world4100.lib（release）
### 4. 配置CUDA
   项目名->右键->生成依赖项->自定义生成->勾选cuda12.0选项
	