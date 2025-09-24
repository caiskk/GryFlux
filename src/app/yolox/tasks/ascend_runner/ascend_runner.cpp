#include "ascend_runner.h"
#include "package.h"
#include "utils/logger.h"

#define ACL_CHECK(op, error_msg) \
    do { \
        aclError ret = (op); \
        if (ret != ACL_SUCCESS) { \
            LOG.error("%s failed! ret=%d", error_msg, ret); \
            throw std::runtime_error(error_msg); \
        } \
    } while (0)

namespace GryFlux {
	AscendRunner::AscendRunner(std::string_view model_path, int device_id, std::size_t model_width, std::size_t model_height) {
		this->model_width_ = model_width;
		this->model_height_ = model_height;
		this->device_id_ = device_id;
		LOG.info("model path: %s", model_path.data());
		
		// Initialize ACL
		ACL_CHECK(aclInit(nullptr), "aclInit");
		
		// Set device
		ACL_CHECK(aclrtSetDevice(device_id_), "aclrtSetDevice");
		
		// Create context
		ACL_CHECK(aclrtCreateContext(&context_, device_id_), "aclrtCreateContext");
		
		// Create stream
		ACL_CHECK(aclrtCreateStream(&stream_), "aclrtCreateStream");
		
		// Load model
		auto model_meta = load_model(model_path);
		if (!model_meta) {
			LOG.error("Fail to read model");
			throw std::runtime_error("fail to read model");
		}
		auto &[model_data, model_data_size] = *model_meta;
		
		// Load model from memory
		ACL_CHECK(aclmdlLoadFromMem(model_data.get(), model_data_size, &model_id_), "aclmdlLoadFromMem");
		
		// Create model description
		model_desc_ = aclmdlCreateDesc();
		ACL_CHECK(aclmdlGetDesc(model_desc_, model_id_), "aclmdlGetDesc");

		// Get input and output information
		input_num_ = aclmdlGetNumInputs(model_desc_);
		output_num_ = aclmdlGetNumOutputs(model_desc_);

		// Initialize input tensors
		input_dataset_ = aclmdlCreateDataset();
		input_device_buffers_.resize(input_num_);
		input_buffer_sizes_.resize(input_num_);
		
		for (size_t i = 0; i < input_num_; i++) {
			aclmdlIODims dims;
			ACL_CHECK(aclmdlGetInputDims(model_desc_, i, &dims), "aclmdlGetInputDims");
			
			aclDataType dataType = aclmdlGetInputDataType(model_desc_, i);
			aclFormat format = aclmdlGetInputFormat(model_desc_, i);
			size_t buffer_size = aclmdlGetInputSizeByIndex(model_desc_, i);
			
			input_buffer_sizes_[i] = buffer_size;
			ACL_CHECK(aclrtMalloc(&input_device_buffers_[i], buffer_size, ACL_MEM_MALLOC_HUGE_FIRST), 
					 "aclrtMalloc for input");
			
			aclDataBuffer* dataBuffer = aclCreateDataBuffer(input_device_buffers_[i], buffer_size);
			ACL_CHECK(aclmdlAddDatasetBuffer(input_dataset_, dataBuffer), "aclmdlAddDatasetBuffer for input");
			
			this->dump_tensor_attr(&dims, "input", dataType, format);
		}

		// Initialize output tensors  
		output_dataset_ = aclmdlCreateDataset();
		output_device_buffers_.resize(output_num_);
		output_buffer_sizes_.resize(output_num_);
		
		for (size_t i = 0; i < output_num_; i++) {
			aclmdlIODims dims;
			ACL_CHECK(aclmdlGetOutputDims(model_desc_, i, &dims), "aclmdlGetOutputDims");
			
			aclDataType dataType = aclmdlGetOutputDataType(model_desc_, i);
			aclFormat format = aclmdlGetOutputFormat(model_desc_, i);
			size_t buffer_size = aclmdlGetOutputSizeByIndex(model_desc_, i);
			
			output_buffer_sizes_[i] = buffer_size;
			ACL_CHECK(aclrtMalloc(&output_device_buffers_[i], buffer_size, ACL_MEM_MALLOC_HUGE_FIRST), 
					 "aclrtMalloc for output");
			
			aclDataBuffer* dataBuffer = aclCreateDataBuffer(output_device_buffers_[i], buffer_size);
			ACL_CHECK(aclmdlAddDatasetBuffer(output_dataset_, dataBuffer), "aclmdlAddDatasetBuffer for output");
			
			this->dump_tensor_attr(&dims, "output", dataType, format);
		}

	}

	void AscendRunner::dump_tensor_attr(aclmdlIODims* dims, const char* name, aclDataType dataType, aclFormat format) {
		const char* typeStr = [dataType]() {
			switch (dataType) {
				case ACL_FLOAT: return "FLOAT";
				case ACL_FLOAT16: return "FLOAT16"; 
				case ACL_INT8: return "INT8";
				case ACL_INT32: return "INT32";
				case ACL_UINT8: return "UINT8";
				default: return "UNKNOWN";
			}
		}();
		
		const char* fmtStr = [format]() {
			switch (format) {
				case ACL_FORMAT_NCHW: return "NCHW";
				case ACL_FORMAT_NHWC: return "NHWC";
				case ACL_FORMAT_ND: return "ND";
				case ACL_FORMAT_NC1HWC0: return "NC1HWC0";
				default: return "UNKNOWN";
			}
		}();

		LOG.info("MODEL TENSOR: %s | dims=%zu | shape=[%zu,%zu,%zu,%zu] | type=%s | format=%s",
				name, dims->dimCount, 
				dims->dimCount > 0 ? dims->dims[0] : 0,
				dims->dimCount > 1 ? dims->dims[1] : 0, 
				dims->dimCount > 2 ? dims->dims[2] : 0,
				dims->dimCount > 3 ? dims->dims[3] : 0,
				typeStr, fmtStr);
	}

	std::optional<ModelData> AscendRunner::load_model(std::string_view filename) {
		std::ifstream file(filename.data(), std::ios::binary | std::ios::ate);
		if (!file.is_open()) {
			LOG.error("Failed to open file: %s", filename.data());
			return std::nullopt;
		}

		size_t fileSize = static_cast<size_t>(file.tellg());
		file.seekg(0);
		
		auto buffer = std::make_unique<unsigned char[]>(fileSize);
		if (!buffer || !file.read(reinterpret_cast<char*>(buffer.get()), fileSize)) {
			LOG.error("Failed to read model file");
			return std::nullopt;
		}
		
		return std::make_pair(std::move(buffer), fileSize);
	}

    std::shared_ptr<DataObject> AscendRunner::process(const std::vector<std::shared_ptr<DataObject>> &inputs) {
		if (inputs.size() != 1) return nullptr;

		auto input_data = std::dynamic_pointer_cast<ImagePackage>(inputs[0]);
		auto frame = input_data->get_data();
		
		if (frame.rows != model_height_ || frame.cols != model_width_) {
			LOG.error("Input image size (%dx%d) doesn't match model input size (%zux%zu)", 
					  frame.cols, frame.rows, model_width_, model_height_);
			return nullptr;
		}
		
		cv::Mat float_frame;
		frame.convertTo(float_frame, CV_32F);

		// Convert NHWC to NCHW
		std::vector<cv::Mat> channels(3);
		cv::split(float_frame, channels);
		
		std::vector<float> nchw_data(1 * 3 * frame.rows * frame.cols);
		size_t channel_size = frame.rows * frame.cols;
		for (int i = 0; i < 3; i++) {
			memcpy(nchw_data.data() + i * channel_size, channels[i].data, channel_size * sizeof(float));
		}

		// Copy input data to device
		ACL_CHECK(aclrtSetCurrentContext(context_), "aclrtSetCurrentContext");
		
		size_t input_size = nchw_data.size() * sizeof(float);
		ACL_CHECK(aclrtMemcpy(input_device_buffers_[0], input_buffer_sizes_[0], 
							 nchw_data.data(), input_size, ACL_MEMCPY_HOST_TO_DEVICE), 
							 "aclrtMemcpy input to device");

		// Execute model
		ACL_CHECK(aclmdlExecute(model_id_, input_dataset_, output_dataset_), "aclmdlExecute");
		ACL_CHECK(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream");

		// Process outputs
		auto output_data = std::make_shared<RunnerPackage>(model_width_, model_height_);
		
		for (size_t i = 0; i < output_num_; i++) {
			aclmdlIODims dims;
			ACL_CHECK(aclmdlGetOutputDims(model_desc_, i, &dims), "aclmdlGetOutputDims");

			size_t n_elems = 1;
			for (size_t j = 0; j < dims.dimCount; j++) {
				n_elems *= dims.dims[j];
			}
			
			std::shared_ptr<float[]> output_host(new float[n_elems]);

			ACL_CHECK(aclrtMemcpy(output_host.get(), n_elems * sizeof(float),
									output_device_buffers_[i], output_buffer_sizes_[i],
									ACL_MEMCPY_DEVICE_TO_HOST), "aclrtMemcpy output from device");
			
			size_t height = dims.dimCount > 2 ? dims.dims[2] : 1;
			size_t width = dims.dimCount > 3 ? dims.dims[3] : 1;
			output_data->push_data({output_host, n_elems}, {height, width});
		}
		
		return output_data;
	}

	AscendRunner::~AscendRunner() {
		// Cleanup datasets and device buffers
		auto cleanup_dataset = [](aclmdlDataset* dataset, std::vector<void*>& buffers) {
			if (dataset) {
				for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(dataset); i++) {
					aclDestroyDataBuffer(aclmdlGetDatasetBuffer(dataset, i));
				}
				aclmdlDestroyDataset(dataset);
			}
			for (auto& buffer : buffers) {
				if (buffer) aclrtFree(buffer);
			}
		};
		
		cleanup_dataset(input_dataset_, input_device_buffers_);
		cleanup_dataset(output_dataset_, output_device_buffers_);

		// Cleanup model and runtime
		if (model_desc_) aclmdlDestroyDesc(model_desc_);
		aclmdlUnload(model_id_);
		if (stream_) aclrtDestroyStream(stream_);
		if (context_) aclrtDestroyContext(context_);
		aclrtResetDevice(device_id_);
		aclFinalize();
	}
}