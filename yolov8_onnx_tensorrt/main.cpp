#include <iostream>
#include <fstream>
#include "calibrator.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "yolo.h"
#include "Logger.hpp"
#include <cuda.h>

void ONNX2TensorRT(const char* ONNX_file, std::string& Engine_file, bool& FP16, bool& INT8, std::string& image_dir, const char*& calib_table) {
	
	Logger transLogger;
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(transLogger);
	std::cout << "Infer builder is created . " << std::endl;

	uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);
	std::cout << "Infer network is created . " << std::endl;

	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, transLogger);
	std::cout << "Onnx file parser is created . " << std::endl;
	std::cout << "Parsing onnx file from  \" " << ONNX_file << "\" \t ........." << std::endl;
	parser->parseFromFile(ONNX_file, static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));
	for (int32_t i = 0; i < parser->getNbErrors(); ++i)
		std::cout << parser->getError(i)->desc() << std::endl;

	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	std::cout << "Infer builder config is created . " << std::endl;
	//config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));
	std::cout << "Infer builder config is setting \"MemoryPoolLimit\" to 4GB. " << std::endl;
	config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, size_t(1) << 32);

	if (FP16) {
		if (!builder->platformHasFastFp16()) {
			std::cout << "FP16 quantization is not supported!" << std::endl;
			system("pause");
			return;
		}
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
		std::cout << "Infer builder config set precision to kFP16. " << std::endl;
	}
	else if (INT8) {
		if (!builder->platformHasFastInt8()) {
			std::cout << "INT8 quantization is not supported!" << std::endl;
			system("pause");
			return;
		}
		config->setFlag(nvinfer1::BuilderFlag::kINT8);
		nvinfer1::IInt8EntropyCalibrator2* calibrator = new Calibrator(1, 640, 640, image_dir, calib_table);
		config->setInt8Calibrator(calibrator);
		std::cout << "Infer builder config set precision to kINT8. " << std::endl;
	}

	std::cout << "Infer builder serialize network : serializeModel is creating........ " << std::endl;
	nvinfer1::IHostMemory* serializeModel = builder->buildSerializedNetwork(*network, *config);
	std::ofstream engine(Engine_file, std::ios::binary);
	std::cout << "Saving data to engine .......... " << std::endl;
	engine.write(reinterpret_cast<const char*>(serializeModel->data()), serializeModel->size());
	
	delete parser;
	delete network;
	delete config;
	delete builder;

	delete serializeModel;
	std::cout << "Export success, Save as: " << Engine_file << std::endl;
}


int main() {
	//CUmoduleLoadingMode mode = CU_MODULE_LAZY_LOADING;
	CUmoduleLoadingMode mode = CU_MODULE_EAGER_LOADING;

	assert(CUDA_SUCCESS == cuInit(0));
	assert(CUDA_SUCCESS == cuModuleGetLoadingMode(&mode));
	
	std::cout << "CUDA module loading mode type is: " << ((mode == CU_MODULE_LAZY_LOADING) ? "lazy" : "eager") << std::endl;

	bool Transform = false;
	//std::string prefixName = "best_from_kaibin0";
	std::string prefixName = "yolov8s";
	if (Transform) {
		// ONNX file path
		std::string ONNX_file = "../weights/" + prefixName + ".onnx";
		// ENGINE file save path
		std::string Engine_file = "../weights/"+ prefixName +".engine";

		// Quantified as INT8, the images path
		std::string image_dir = "../images/";
		// Calibrated table path when quantified to INT8 (present to read, not present to create)
		const char* calib_table = "../weights/calibrator.table";

		bool FP16 = true;
		bool INT8 = false;

		std::ifstream file(ONNX_file.c_str(), std::ios::binary);
		if (!file.good()) {
			std::cout << "Load ONNX file failed!" << std::endl;
			return -1;
		}

		ONNX2TensorRT(ONNX_file.c_str(), Engine_file, FP16, INT8, image_dir, calib_table);
	}
	else {
		std::string engine_file = "../weights/" + prefixName + ".engine";
		std::string class_file = "../weights/classes.txt";
		//std::string class_file = "../weights/classes_from_kaibin.txt";
		float conf_threshold = 0.1;
		float NMS_threshold = 0.45;

		YOLO YOLO(engine_file.c_str(), class_file, conf_threshold, NMS_threshold);
		cv::Mat img = cv::imread("../images/bus.jpg");
		//cv::Mat img = cv::imread("../images/img_from_kaibin_123.jpg");
		YOLO.Inference(img);
	}

	return 0;
}
