#include "yolo.h"
#include "process.h"
#include "common.hpp"
#include <random>

YOLO::YOLO(const std::string& engine_file, const std::string& class_file, const float& conf_threshold, const float& NMS_threshold) {
	ENGINE_FILE = engine_file;
	CONF_THRESHOLD = conf_threshold;
	NMS_THRESHOLD = NMS_threshold;
	BATCH_SIZE = 1;
	IMAGE_HEIGHT = 640;
	IMAGE_WIDTH = 640;
	IMAGE_CHANNEL = 3;
	MAX_IMAGE_SIZE = 3000 * 3000;
	//NUM_BOXES = 8400;
	NUM_BOXES = 34000;
	NUM_OBJECTS = 1000;
	readClassFile(class_file, LABELS);
	NUM_CLASSES = LABELS.size();
}

void YOLO::Inference(cv::Mat& image) {
	readEngineFile(ENGINE_FILE, engine);
	context = engine->createExecutionContext();
	assert(context != nullptr);

	assert(engine->getNbBindings() == 2);
	void* buffers[2];
	uint8_t* image_device = nullptr;
	const int input_size = BATCH_SIZE * IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL * sizeof(float);
	const int output_size = BATCH_SIZE * NUM_BOXES * (NUM_CLASSES+4) * sizeof(float);
	const int max_size = BATCH_SIZE * MAX_IMAGE_SIZE * IMAGE_CHANNEL;
	const int input_index = engine->getBindingIndex("images");
	const int output_index = engine->getBindingIndex("output0");
	cudaMalloc(&buffers[input_index], input_size);
	cudaMalloc(&buffers[output_index], output_size);
	cudaMalloc((void**)&image_device, max_size);
	auto* engine_output = new float[NUM_BOXES * (NUM_CLASSES + 4) * BATCH_SIZE];

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	//cv::Mat image;
	//cv::VideoCapture cap(0);
	// cv::VideoCapture cap("../images/me.mp4");

	//auto t_beg = std::chrono::high_resolution_clock::now();
	//cap >> image;
	if (image.empty())
	{
		return;
	}
	float scale = float(IMAGE_WIDTH) / float(image.cols) < float(IMAGE_HEIGHT) / float(image.rows) ? float(IMAGE_WIDTH) / float(image.cols) : float(IMAGE_HEIGHT) / float(image.rows);
	size_t image_size = image.rows * image.cols * image.channels();
	cudaMemcpyAsync(
		image_device,
		image.data,
		image_size,
		cudaMemcpyHostToDevice,
		stream
	);
	// Image preprocessing
	preProcess(image_device, image.rows, image.cols, scale, (float*)buffers[input_index], IMAGE_HEIGHT, IMAGE_WIDTH);
		
	// Inference
	//auto t_beg = std::chrono::high_resolution_clock::now();
	auto t_beg = std::chrono::system_clock::now();
	context->enqueueV2(buffers, stream, nullptr);
	auto t_end = std::chrono::system_clock::now();
	cudaMemcpyAsync(engine_output,
		buffers[1],
		output_size,
		cudaMemcpyDeviceToHost, // GPU -> CPU
		stream);
	cudaStreamSynchronize(stream);

	// Image post processing
	std::vector<YOLO::DetectRet> detections;
	postProcess(engine_output, output_size, NUM_BOXES, NUM_CLASSES, CONF_THRESHOLD, NMS_THRESHOLD, 1 / scale, NUM_OBJECTS, LABELS, detections);
		
	// Draw
	for (auto& det : detections) {
		// 随机颜色生成
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_int_distribution<> distrib(0, 255);
		int COLOR_R = distrib(gen);
		int COLOR_G = distrib(gen);
		int COLOR_B = distrib(gen);
		cv::Scalar scalar(COLOR_B, COLOR_G, COLOR_R);
		cv::putText(image, det.name + ":" + std::to_string(det.conf), cv::Point(det.x1, det.y1 - 5), cv::FONT_HERSHEY_TRIPLEX, 0.8, scalar, 2);
		cv::rectangle(image, cv::Rect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1), scalar, 2, cv::LINE_8, 0);
	}
	

	// Inference time
	//auto t_end = std::chrono::high_resolution_clock::now();
	//float total_inf = std::chrono::duration<float, std::milli>(t_end - t_beg).count();
	float total_inf = std::chrono::duration_cast<std::chrono::microseconds> (t_end - t_beg).count() / 1000.0;
	std::cout << "Inference time: " << total_inf << " ms." << std::endl;
	cv::namedWindow("Inference", cv::WINDOW_NORMAL);
	cv::imshow("Inference", image);
	cv::waitKey(0);

	cv::destroyAllWindows();
	
	cudaStreamDestroy(stream);
	cudaFree(image_device);
	cudaFree(buffers[input_index]);
	cudaFree(buffers[output_index]);
	context->destroy();
	engine->destroy();
	
}
