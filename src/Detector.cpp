#include "Detector.h"

//#include "tracker/utils.h"

#include <iostream>
#include <fstream>
#include <utility>
#include <vector>
#include <list>
#include <chrono>

#include "NvInferPlugin.h"

#include <boost/range/adaptor/indexed.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/cudawarping.hpp>

namespace detection{

  // Rescale out_bbox (xywh) from src_size to dst_size
cv::Rect_<float> scale_coords(
  const cv::Rect& bbox,
  const cv::Size& src_size, // NN size
  const cv::Size& dst_size  // camera stream size
) {
  const auto src_w = (float) src_size.width;
  const auto src_h = (float) src_size.height;
  const auto dst_w = (float) dst_size.width;
  const auto dst_h = (float) dst_size.height;

  const float gain = std::min(src_h / dst_h, src_w / dst_w);  // gain  = old / new
  const float pad_w = std::max((src_w - dst_w * gain) / 2, 0.f);
  const float pad_h = std::max((src_h - dst_h * gain) / 2, 0.f);

  const auto new_x = (bbox.x - pad_w) / gain;
  const auto new_y = (bbox.y - pad_h) / gain;
  return {
    new_x,
    new_y,
    (bbox.x + bbox.width - pad_w) / gain - new_x,
    (bbox.y + bbox.height - pad_h) / gain - new_y
  };
}


class Logger : public nvinfer1::ILogger {
  // Class to log errors, warnings, and other information during the build and inference phases
 public:
  void log(Severity severity, const char* msg) noexcept override {
    // remove this 'if', if you need more logged info
    if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
      std::cout << msg << "\n";
    }
  }

  Logger& getTRTLogger() {
    return *this;
  }

} gLogger;

int get_size_by_dim(const nvinfer1::Dims& dims) {
  // Calculate size of tensor
  int size = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    size *= dims.d[i];
    // std::cout << " [" << dims.d[i] << "] ";
  }
  return size;
}

void Detector::resize_keep_aspect_ratio_gpu(
  const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::GpuMat& temp,
  const cv::Size& destination_size, const cv::Scalar& bgcolor, cv::cuda::Stream& stream
) {
  // Resize image keeping aspect ratio (letterbox type)
  double h1 = destination_size.width * (input.rows / (double) input.cols);
  double w2 = destination_size.height * (input.cols / (double) input.rows);
  if (h1 <= destination_size.height) {
    cv::cuda::resize(input, temp, cv::Size(destination_size.width, (int) h1), 0, 0, cv::INTER_LINEAR, stream);
  }
  else {
    cv::cuda::resize(input, temp, cv::Size((int) w2, destination_size.height), 0, 0, cv::INTER_LINEAR, stream);
  }

  int top = (destination_size.height - temp.rows) / 2;
  int down = (destination_size.height - temp.rows + 1) / 2;
  int left = (destination_size.width - temp.cols) / 2;
  int right = (destination_size.width - temp.cols + 1) / 2;

  cv::cuda::copyMakeBorder(temp, output, top, down, left, right, cv::BORDER_CONSTANT, bgcolor, stream);
}

std::vector< cv::Size > Detector::preprocess_batch(
  const Detector::Images_view& original_images,
  float* gpu_input,
  nvinfer1::Dims dims,
  nvinfer1::DataType data_type
) {
  // Preprocessing stage
  //    dims.d[0] = im_orig.size();
  int input_width = dims.d[3];
  int input_height = dims.d[2];
  int channels = dims.d[1];
  auto input_size = cv::Size(input_width, input_height);

  int norm_type = 0;
  if (data_type == nvinfer1::DataType::kFLOAT) {
    norm_type = CV_32FC1;
  }
  else if (data_type == nvinfer1::DataType::kINT32) {
    norm_type = CV_32SC1;
  }
  else {
    //LOG_ERROR_L("Wrong data type in detector");
    throw std::runtime_error("Wrong data type in detector");
  }

  if (_chw_image.empty()) {
    _chw_image.resize(_batch_size);
    for (int k = 0; k < _batch_size; k++) {
      _chw_image[k].emplace_back(
        input_size,
        norm_type,
        gpu_input + k * channels * input_width * input_height + 2 * input_width * input_height
      );

      _chw_image[k].emplace_back(
        input_size,
        norm_type,
        gpu_input + k * channels * input_width * input_height + 1 * input_width * input_height
      );

      _chw_image[k].emplace_back(
        input_size,
        norm_type,
        gpu_input + k * channels * input_width * input_height + 0 * input_width * input_height
      );
    }
  }

  std::vector< cv::Size > original_images_sizes;
  for (const auto& [image_index, original_image]: original_images | boost::adaptors::indexed()) {
    original_images_sizes.emplace_back(original_image.size.operator()());
    // To GPU tensor
    _gpu_image[image_index].upload(original_image, _gpu_stream[image_index]);

    // Resize (letterbox)
    auto border_color = cv::Scalar(114, 114, 114, 255);
    resize_keep_aspect_ratio_gpu(
      _gpu_image[image_index],
      _gpu_resized_rgb[image_index],
      _gpu_resized_rgb_temp[image_index],
      input_size,
      border_color,
      _gpu_stream[image_index]
    );

    if (data_type == nvinfer1::DataType::kFLOAT) {
      _gpu_resized_rgb[image_index].convertTo(
        _gpu_image_conv[image_index],
        CV_32FC3,
        1.f / 255.f,
        _gpu_stream[image_index]
      );
    }
    else if (data_type == nvinfer1::DataType::kINT32) {
      _gpu_resized_rgb[image_index].convertTo(_gpu_image_conv[image_index], CV_32SC3, _gpu_stream[image_index]);
    }

    cv::cuda::split(
      _gpu_image_conv[image_index],
      _chw_image[image_index],
      _gpu_stream[image_index]
    ); // Divide multi-channel gpu_image into std::vector of 1-channel arrays
  }

  for (size_t k = 0; k < boost::size(original_images); k++) {
    _gpu_stream[k].waitForCompletion();
  }

  return original_images_sizes;
}

std::vector< std::vector< Detector::Detection > > Detector::postprocess_results_nms(
  const std::vector< void* >& buffers,
  std::vector< nvinfer1::Dims > dims,
  const std::vector< cv::Size >& original_images_sizes,
  float conf_thresh,
  float nms_thresh
)
const {
//  const auto now =
//    std::chrono::duration_cast< std::chrono::seconds >(std::chrono::system_clock::now().time_since_epoch()).count();

  // TODO: Check arithmetic
  std::vector< std::vector< Detection>> result(original_images_sizes.size());

  nvinfer1::Dims single_dim = dims[0];

  int rows = single_dim.d[1];
  int dimensions = single_dim.d[2];
  size_t num_classes = 0;
  bool is_yolov8 = false;
  // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
  // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
  if (dimensions > rows) // Check if the shape[2] is more than shape[1] (yolov8)
  {
    is_yolov8 = true;
    rows = single_dim.d[2];
    dimensions = single_dim.d[1];
    num_classes = dimensions - 4;
  }
  else {
    num_classes = dimensions - 5;
  }

  single_dim.d[0] = 1;
  std::vector< float > full_preds(get_size_by_dim(dims[0]));
  size_t single_size = get_size_by_dim(single_dim);

  // Copy results from GPU to CPU
  cudaMemcpy(full_preds.data(), buffers[dims.size()], full_preds.size() * sizeof(float), cudaMemcpyDeviceToHost);

  for (size_t image_index = 0; image_index < original_images_sizes.size(); image_index++) {
    std::vector< float > preds(
      full_preds.begin() + image_index * single_size,
      full_preds.begin() + (image_index + 1) * single_size
    );

    std::map<int, DetectRawData> detect_raw_data_by_classes;
   // std::vector< cv::Rect > cv_rect_boxes;
   // std::vector< float > scores;
   // std::vector< int > class_ids;

    if (is_yolov8) {
      for (int i = 0; i < rows; i++) {
        std::vector <float> raw_scores(num_classes);
        for (size_t k=0; k<num_classes; k++)
          raw_scores[k] = preds[(4+k) * rows + i];
        cv::Mat class_scores(1, num_classes, CV_32FC1, raw_scores.data());
        cv::Point class_id;
        double max_class_score = 0.0;
        minMaxLoc(class_scores, 0, &max_class_score, 0, &class_id);

        cv::Rect rect_bbox(
          (int) preds[0 * rows + i],
          (int) preds[1 * rows + i],
          (int) preds[2 * rows + i],
          (int) preds[3 * rows + i]
        );

        if (max_class_score > conf_thresh && rect_bbox.area() > 10
          && rect_bbox.x > 0
          && rect_bbox.x < _nn_input_image_size.width) {
            detect_raw_data_by_classes[class_id.x].cv_rect_boxes.push_back(rect_bbox);
            detect_raw_data_by_classes[class_id.x].scores.push_back(max_class_score);
        }
      }
    }
    else {
      for (size_t i = 0; i < preds.size(); i += 6) {
        cv::Mat class_scores(1, num_classes, CV_32FC1, &preds[5]);
        cv::Point class_id;
        double max_class_score = 0.0;
        minMaxLoc(class_scores, 0, &max_class_score, 0, &class_id);

        float confidence = preds[i + 4];

        cv::Rect rect_bbox(
          (int) preds[i],
          (int) preds[i + 1],
          (int) preds[i + 2],
          (int) preds[i + 3]
        );

        if (
          confidence > conf_thresh
            && max_class_score > conf_thresh
            && rect_bbox.area() > 10
            && rect_bbox.x > 0
            && rect_bbox.x < _nn_input_image_size.width
          ) {
            detect_raw_data_by_classes[class_id.x].cv_rect_boxes.push_back(rect_bbox);
            detect_raw_data_by_classes[class_id.x].scores.push_back(max_class_score);
        }
      }
    }

    // Perform NMS
    for (auto I = detect_raw_data_by_classes.begin(); I != detect_raw_data_by_classes.end(); ++I) {
      std::vector<int> indices;
      cv::dnn::NMSBoxes(I->second.cv_rect_boxes, I->second.scores, conf_thresh, nms_thresh, indices, 1.f, 0);

      for (const auto &index: indices) {
        result[image_index].push_back(Detection(
                scale_coords(I->second.cv_rect_boxes[index], _nn_input_image_size, original_images_sizes[image_index]),
                I->first));
      }
    }
  }

  return result;
}

cv::Mat draw_boxes(cv::Mat& src_img, const std::vector< std::vector< float>>& boxes, int alert_level = 0) {
  // Draw bounding boxes on image
  cv::Mat dst_img = src_img.clone();
  for (auto abs_bbox: boxes) {
    cv::Point pt1((int) abs_bbox[0], (int) abs_bbox[1]);
    cv::Point pt2((int) abs_bbox[2], (int) abs_bbox[3]);
    cv::Scalar color;
    if (alert_level != 0) {
      color = cv::Scalar(255, 0, 0);
    }
    else {
      color = cv::Scalar(0, 255, 0);
    }

    cv::rectangle(dst_img, pt1, pt2, color);
  }

  return dst_img;
}

nvinfer1::ICudaEngine* load_engine(const std::string& engine) {
  std::ifstream engineFile(engine, std::ios::binary);
  if (!engineFile) {
    std::cout << "Error opening engine file: " << engine << std::endl;
    return nullptr;
  }

  engineFile.seekg(0, std::ifstream::end);
  long int fsize = engineFile.tellg();
  engineFile.seekg(0, std::ifstream::beg);

  std::vector< char > engineData(fsize);
  engineFile.read(engineData.data(), fsize);
  if (!engineFile) {
    std::cout << "Error loading engine file: " << engine << std::endl;
    return nullptr;
  }

//  std::unique_ptr< nvinfer1::IRuntime > runtime {nvinfer1::createInferRuntime(gLogger.getTRTLogger())};
  nvinfer1::IRuntime* runtime=nvinfer1::createInferRuntime(gLogger.getTRTLogger());

  if (!runtime) {
    std::cout << "Failed to call nvinfer1::createInferRuntime" << engine << std::endl;
    return nullptr;
  }

  int DLACore = runtime->getNbDLACores();
  std::cout << "DLACores: " << DLACore << std::endl;

  if (DLACore != -1) {
    runtime->setDLACore(DLACore);
  }

  initLibNvInferPlugins(nullptr, "");
  return runtime->deserializeCudaEngine(engineData.data(), fsize); //, nullptr);
}

Detector::Detector(
  std::unique_ptr< nvinfer1::ICudaEngine >&& cuda_engine,
  std::unique_ptr< nvinfer1::IExecutionContext >&& execution_context,
  std::vector< void* >&& buffers,
  std::vector< nvinfer1::Dims >&& input_dimensions,
  std::vector< nvinfer1::Dims >&& output_dimensions,
  cv::Size nn_input_image_size,
  float conf_thresh,
  float nms_thresh,
  int batch_size,
  int batch_mode,
  nvinfer1::DataType data_type,
  uint64_t num_sources
)
  : _cuda_engine(std::move(cuda_engine))
  , _execution_context(std::move(execution_context))
  , _buffers(std::move(buffers))
  , _input_dimensions(std::move(input_dimensions))
  , _output_dimensions(std::move(output_dimensions))
  , _nn_input_image_size(nn_input_image_size)
  , _conf_thresh(conf_thresh)
  , _nms_thresh(nms_thresh)
  , _batch_size(batch_size)
  , _batch_mode(batch_mode)
  , _data_type(data_type)
  , _num_sources(num_sources)
  , _exec_counter(0)
  , _network_calc_duration(0)
  , _detection_calc_duration(0)
  , _num_imgs(0)
{

  _gpu_resized_rgb.resize(_batch_size);
  _gpu_resized_rgb_temp.resize(_batch_size);
  _gpu_stream.resize(_batch_size);
  _gpu_image.resize(_batch_size);
  _gpu_image_conv.resize(_batch_size);

  _thread = std::thread(std::bind(&Detector::execute, this));
}

std::unique_ptr< Detector::Detections_pack > Detector::dequeue_detections_pack() {
  std::optional<Detector::Detections_pack> data = _output_detections_buffer.dequeue();

  if(!data.has_value())
    return {};

  std::unique_ptr< Detector::Detections_pack > result = std::make_unique<Detector::Detections_pack >(data.value());

  return result;// .dequeue_unique_ptr();
}

bool Detector::enqueue_frame(const std::string &source_address, const cv::Mat& frame) {
  _input_frame_buffer.enqueue(std::pair<std::string, cv::Mat>(source_address, frame));
 /* const auto result = _input_frame_buffer.enqueue(frame);;
  if (!result) {
    LOG_WARN_L("_input_frame_buffer queue is full");
  }

  return _input_frame_buffer.enqueue(frame);
  */
 return true;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  
std::unique_ptr< Detector > Detector::make(const Detector::Parameters& parameters) {
  auto model_path = std::filesystem::path(parameters.model_path);
  if (!std::filesystem::exists(model_path)) {
//    LOG_GLOB_ERROR("Detector::make: model in path '{}' doesn't exist", parameters.model_path);
    return {};
  }

  if (!std::filesystem::is_regular_file(model_path)) {
//    LOG_GLOB_ERROR("Detector::make: model in path '{}' is not regular file", parameters.model_path);
    return {};
  }

  auto allowable_sources = parameters.allowable_sources;
  uint64_t num_sources = allowable_sources.size();
  if (!num_sources) {
//    LOG_GLOB_ERROR("Detector::make: num source is zero");
    return {};
  }

  std::unique_ptr< nvinfer1::ICudaEngine > cuda_engine {load_engine(model_path)};
  if (!cuda_engine) {
//    LOG_GLOB_ERROR("Detector::make: failed to create ICudaEngine");
    return {};
  }

  std::unique_ptr< nvinfer1::IExecutionContext > execution_context {cuda_engine->createExecutionContext()};
  if (!execution_context) {
//    LOG_GLOB_ERROR("Detector::make: failed to create IExecutionContext");
    return {};
  }

  // Get sizes of input and output and allocate memory required for input data and for output data
  std::vector< nvinfer1::Dims > input_dimensions;
  std::vector< nvinfer1::Dims > output_dimensions;
  std::vector< void* > buffers(cuda_engine->getNbBindings()); // Buffers for input and output data
  int batch_size = 1;

  if (cuda_engine->getNbBindings() > 0) {
    auto ndims = cuda_engine->getBindingDimensions(0);
    if (ndims.nbDims > 0) {
      batch_size = ndims.d[0];
    }
  }

  int batch_mode = parameters.batch_mode;

  nvinfer1::DataType input_data_type;
  for (int i = 0; i < cuda_engine->getNbBindings(); ++i) {
    // nvinfer1::Dims dims = cuda_engine->getTensorShape("images");
    const auto binding_size =
      get_size_by_dim(cuda_engine->getBindingDimensions(i)) * sizeof(float);
    cudaMalloc(&buffers[i], binding_size);

    if (cuda_engine->bindingIsInput(i)) {
      input_data_type = cuda_engine->getBindingDataType(i);
      input_dimensions.emplace_back(cuda_engine->getBindingDimensions(i));
    }
    else {
      output_dimensions.emplace_back(cuda_engine->getBindingDimensions(i));
    }
  }

  if (input_dimensions.empty() || output_dimensions.empty()) {
//    LOG_GLOB_ERROR(
//      "Detector::make: Expect at least one input and one output for network. Got: {}, {}",
//      input_dimensions.size(),
//      output_dimensions.size()
//    );

    return {};
  }

  return
    std::unique_ptr< Detector >(
      new Detector(
        std::move(cuda_engine),
        std::move(execution_context),
        std::move(buffers),
        std::move(input_dimensions),
        std::move(output_dimensions),
        cv::Size(parameters.nn_input_image_width, parameters.nn_input_image_height),
        parameters.conf_thresh,
        parameters.nms_thresh,
        batch_size,
        batch_mode,
        input_data_type,
        num_sources
      )
    );
}
#pragma GCC diagnostic pop

void Detector::execute() {
  while (!_stop) {
    auto t0 = std::chrono::high_resolution_clock::now();

    uint64_t expected_buf_size = (_batch_mode == 0) ? _batch_size : _num_sources;

    if (_input_frame_buffer.size() < expected_buf_size) {
      //LOG_DEBUG_L("Not enough images in the buffer: {} less than {}", _input_frame_buffer.size(), _batch_size);
      std::this_thread::sleep_for(std::chrono::milliseconds(40));
      continue;
    }
    //LOG_TRACE_L("_input_frame_buffer.size(): {}", _input_frame_buffer.size());

    std::map< std::string, cv::Mat > images;
    for (int i = 0; i < _batch_size * 2; i++) {
      std::optional<std::pair<std::string, cv::Mat>> frame = _input_frame_buffer.dequeue();
      if(!frame.has_value())
        break;

      if (frame.value().second.rows == 0 || frame.value().second.cols == 0) {
      //  LOG_INFO_L("Image {}  has zero size", frame->get_address());
        continue;
      }

      images[frame.value().first] = frame.value().second;
      if (int(images.size()) == _batch_size) {
        break;
      }
    }

    if (int(images.size()) < _batch_size) {
    //  LOG_DEBUG_L("We use not a full batch: {} less than {}", images.size(), _batch_size);
    }

    const auto original_images_sizes =
      preprocess_batch(
        std::as_const(images) | boost::adaptors::map_values,
        (float*) _buffers[0],
        _input_dimensions[0],
        _data_type
      );

    // Run object detector
    auto net_t0 = std::chrono::high_resolution_clock::now();
    _execution_context->executeV2(_buffers.data());
    auto net_t1 = std::chrono::high_resolution_clock::now();

    std::vector< std::vector< Detector::Detection>>
      results = postprocess_results_nms(_buffers, _output_dimensions, original_images_sizes, _conf_thresh, _nms_thresh);
    auto t1 = std::chrono::high_resolution_clock::now();
    ++_exec_counter;

    uint64_t det_duration = std::chrono::duration_cast< std::chrono::milliseconds >(t1 - t0).count();
    uint64_t net_duration = std::chrono::duration_cast< std::chrono::milliseconds >(net_t1 - net_t0).count();

    if (_exec_counter > STATISTIC_BYPASS_FIRST_DETECTIONS) {
      _detection_calc_duration += det_duration;
      _network_calc_duration += net_duration;
      _num_imgs += images.size();
    }

    //uint64_t true_counter = std::max< int64_t >((int64_t) _exec_counter - STATISTIC_BYPASS_FIRST_DETECTIONS, 1);

    if (_exec_counter % STATISTIC_WRITE_LOG_REPEAT == 0 && _exec_counter > STATISTIC_BYPASS_FIRST_DETECTIONS) {
    /*  LOG_DEBUG_L(
        "Detection duration (count={}): DetAvg={}, DetCurr={}, NetAvg={}, NetCurr={}. BatchSize: {}, BatchAvg={}, BatchCurr={} (type: {})",
        true_counter,
        _detection_calc_duration / true_counter,
        det_duration,
        _network_calc_duration / true_counter,
        net_duration,
        _batch_size,
        _num_imgs / true_counter,
        images.size(),
        _data_type
      );*/
    }

    for (const auto& [result_index, image_node]: images | boost::adaptors::indexed()) {
      const auto& [path, _] = image_node;
      _output_detections_buffer.enqueue(Detections_pack(std::vector< Detector::Detection >(results[result_index]), path));
    }
  }

  // LOG_INFO_L("Stopped");
}

void Detector::free() {
  for (auto buffer: _buffers) {
    cudaFree(buffer);
  }
}

Detector::Detections_pack::Detections_pack(
  std::vector< Detection >&& detections,
  std::string source
)
  : _detections(std::move(detections)), _source_address(std::move(source)) {
}

const std::vector< Detector::Detection > Detector::Detections_pack::get_detections() const {
  return _detections;
}

const std::vector< Detector::Detection > Detector::Detections_pack::get_detections(int class_index) const {
  std::vector< Detector::Detection > results;

  for (auto & detection : _detections)
    if (detection.class_index == class_index)
      results.push_back(detection);
  return results;
}

const std::vector< Detector::Detection > Detector::Detections_pack::get_detections(const std::vector<int> &class_indexes) const {
  std::vector< Detector::Detection > results;
  if (class_indexes.empty())
    return results;

  for (auto & detection : _detections)
    for (auto & index : class_indexes)
      if (detection.class_index == index) {
        results.push_back(detection);
        break;
      }
  return results;
}

const std::string& Detector::Detections_pack::get_source_address() const {
  return _source_address;
}

/*

Detectors_set::Detectors_set(Detectors_set::Detectors_map&& detectors)
  : cvs::logger::Loggable< Detectors_set >("Detectors set"), _detectors(detectors) {
}

bool Detectors_set::enqueue_frame(const Executors::Camera_receiver::Frame& frame) {
  const auto find_result = _detectors.find(frame.get_address());
  if (find_result == _detectors.end()) {
    LOG_ERROR_L(fmt::format("Detectors_set::enqueue_frame: can't find detector for camera [{}]", frame.get_address()));
    return false;
  }

  return find_result->second->enqueue_frame(frame);
}

std::unique_ptr< Detector::Detections_pack > Detectors_set::dequeue_detections_pack() {
  std::unique_ptr< Detector::Detections_pack > result;

  for (const auto& _ : _detectors) {
    const auto current_address_id = _current_address++ % _detectors.size();
    auto current_address_iterator = _detectors.begin();
    std::advance(current_address_iterator, current_address_id);
    result = current_address_iterator->second->dequeue_detections_pack();
    if (result) {
      break;
    }
  }

  return result;
}

core::utils::Outcome_shared_reference< Detectors_set >
Detectors_set::make(const cvs::common::Properties& properties) {
  const auto parameters = Parameters::make(properties);
  Detectors_map detectors_by_address;
  for (const auto& detector_parameters: parameters->get_detectors()) {
    const std::shared_ptr< Detector > detector = std::move(Detector::make(detector_parameters));
    if (!detector) {
      throw std::runtime_error("Detectors_set::make: Failed to create detector");
    }

    for (const auto& allowable_source: detector_parameters.get_allowable_sources()) {
      if (detectors_by_address.contains(allowable_source)) {
        LOG_GLOB_ERROR(fmt::format("Detectors_set::make: detector for address [] already exist", allowable_source));
        continue;
      }

      detectors_by_address.emplace(allowable_source, detector);
    }
  }

  return core::utils::Shared_reference< Detectors_set >(std::move(detectors_by_address));
}
*/

}