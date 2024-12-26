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

### 5. 注意
   该项目在将onnx模型转为engine模型时，要求onnx的输出结果顺序为【1,8400,84】，而非【1,84,8400】，所以在导出onnx模型时需要将源码中的【nn/modules/head.py】文件中的函数：
   ''')
   def _inference(self, x):
	"""Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        # [ldq 在下方语句后添加 .permute(0,2,1)  2024-8-28 09:26:22]
        return torch.cat((dbox, cls.sigmoid()), 1) #源代码
        # return torch.cat((dbox, cls.sigmoid()), 1).permute(0, 2, 1) # 修改后的代码 ('''
	
 该函数的return语句：
 	return torch.cat((dbox, cls.sigmoid()), 1) #原代码
  	修改为：
   	return torch.cat((dbox, cls.sigmoid()), 1).permute(0, 2, 1) # 修改后的代码,作用是交换数组元素的位置，由原来的【1,84,8400】变为【1,8400,84】
  
