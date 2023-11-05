#pragma once

//#include "../lib/core/MPSC_queue.hpp"
//#include "../lib/executors/Camera_receiver.h"
//#include "../lib/Pool.h"
//#include "tracker/utils.h"
//#include "../lib/constants.h"

//#include <cvs/logger/loggable.hpp>

#include <opencv2/core/cuda.hpp>

#include <boost/range/adaptor/map.hpp>

#include <NvInfer.h>
#include <utility>
#include <thread>
#include <map>
#include <filesystem>
#include <queue>
#include "SafeQueue.hpp"


namespace detection {


class Detector
//  : public core::Stop_flag, public cvs::logger::Loggable< Detector >
{
//  friend core::Thread;

  static constexpr uint64_t STATISTIC_BYPASS_FIRST_DETECTIONS {100};
  static constexpr uint16_t STATISTIC_WRITE_LOG_REPEAT {200};
  static constexpr int PERSON_CLASS_ID {0};

 public:
  struct Parameters {
    std::string model_path; //, std::string, "NN model path");
    int nn_input_image_width = 640; //CVS_FIELD_DEF(nn_input_image_width, int, 640, "NN input image width");
    int nn_input_image_height = 640; //, int, 640, "NN input image height");
    float conf_thresh = 0.25; //, "Confidence threshold");
    float nms_thresh = 0.45; //, "Non maximum suppression threshold");
    int batch_mode = 0; //, int, 0, "Batching processing mode");
    std::vector< std::string > allowable_sources; // std::vector< std::string >, "Cameras sources for this detector");
  };

  class Detection {
  public:
    cv::Rect_<float> boundingBox;
    uint16_t class_index;
  };

  class DetectRawData {
  public:
    std::vector< cv::Rect > cv_rect_boxes;
    std::vector< float > scores;
  };

  class Detections_pack {
   public:
    Detections_pack() = delete;
    Detections_pack(const Detections_pack&) = default;
    Detections_pack(Detections_pack&&) = delete;
    Detections_pack(std::vector< Detection >&& detections, std::string source);

    const std::vector< Detection > get_detections() const;
    const std::vector< Detection > get_detections(int class_index) const;
    const std::vector< Detection > get_detections(const std::vector<int> &class_indexes) const;
    const std::string& get_source_address() const;

   private:
    std::vector< Detection > _detections;
    std::string _source_address;
  };

  Detector(const Detector&) = delete;
  Detector(Detector&&) = delete;
  ~Detector();

  bool enqueue_frame(const std::string &source_address, const cv::Mat& frame);

  std::unique_ptr< Detections_pack > dequeue_detections_pack();

  static std::unique_ptr< Detector > make(const Detector::Parameters& parameters);

 private:
  Detector(
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
  );

  void execute();

  using Images_view = decltype(std::declval< const std::map< std::string, cv::Mat > >() | boost::adaptors::map_values);

  void resize_keep_aspect_ratio_gpu(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::GpuMat& temp,
                                    const cv::Size& destination_size, const cv::Scalar& bgcolor, cv::cuda::Stream& stream);

  std::vector< cv::Size > preprocess_batch(
    const Images_view& original_images,
    float* gpu_input,
    nvinfer1::Dims dims,
    nvinfer1::DataType data_type
  );

  std::vector< std::vector< Detector::Detection > > postprocess_results_nms(
    const std::vector< void* >& buffers,
    std::vector< nvinfer1::Dims > dims,
    const std::vector< cv::Size >& original_images_sizes,
    float conf_thresh = 0.25,
    float nms_thresh = 0.45
  ) const;

  void free();

 private:
  const std::unique_ptr< nvinfer1::ICudaEngine > _cuda_engine;
  const std::unique_ptr< nvinfer1::IExecutionContext > _execution_context;
  const std::vector< void* > _buffers;
  const std::vector< nvinfer1::Dims > _input_dimensions;
  const std::vector< nvinfer1::Dims > _output_dimensions;

  mysdk::SafeQueue<std::pair<std::string, cv::Mat>> _input_frame_buffer;
  mysdk::SafeQueue<Detections_pack> _output_detections_buffer;
//  std::queue<std::pair<std::string, cv::Mat>> _input_frame_buffer;
//  std::queue<Detections_pack> _output_detections_buffer;
//  core::MPSC_queue< Executors::Camera_receiver::Frame, constants::MAX_DETECTOR_INPUT_QUEUE_SIZE > _input_frame_buffer;
//  core::MPSC_queue< Detections_pack > _output_detections_buffer;

  std::atomic< bool > _stop = false;
  cv::Size _nn_input_image_size;
  float _conf_thresh;
  float _nms_thresh;
  int _batch_size;
  int _batch_mode;
  nvinfer1::DataType _data_type;
  uint64_t _num_sources;

  uint64_t _exec_counter;
  uint64_t _network_calc_duration;
  uint64_t _detection_calc_duration;
  uint64_t _num_imgs;

  std::thread _thread;

 private: // Temporary images
  cv::Mat _im_resized;
  cv::Mat _im_normalized;
  std::vector< cv::cuda::GpuMat > _gpu_image;
  std::vector< cv::cuda::GpuMat > _gpu_resized_rgb;
  std::vector< cv::cuda::GpuMat > _gpu_resized_rgb_temp;
  std::vector< cv::cuda::GpuMat > _gpu_image_conv;
  std::vector< std::vector< cv::cuda::GpuMat>> _chw_image;
  std::vector< cv::cuda::Stream > _gpu_stream;
};
/*
class Detectors_set : public cvs::logger::Loggable< Detectors_set > {
  CVS_CONFIG_CLASS(Parameters, "Detectors set parameters") {
    friend Base;
    friend class cvs::common::Has< Parameters >;

    Parameters() = default;

   public:
    Parameters(const Parameters&) = default;
    Parameters(Parameters && ) = delete;

   private:
    CVS_PROPERTY(detectors, std::vector< Detector::Parameters >, "List of detectors configs");
  };

  using Detectors_map = std::unordered_map< std::string, std::shared_ptr< Detector > >;

 public:
  explicit Detectors_set(Detectors_map&& detectors);
  Detectors_set(Detectors_set&&) = delete;
  Detectors_set(const Detectors_set&) = delete;

  [[nodiscard]] static core::utils::Outcome_shared_reference< Detectors_set >
    make(const cvs::common::Properties& config);

  [[nodiscard]] bool enqueue_frame(const Executors::Camera_receiver::Frame& frame);
  [[nodiscard]] std::unique_ptr< Detector::Detections_pack > dequeue_detections_pack();

 private:
  const Detectors_map _detectors;
  mutable std::atomic< size_t > _current_address = 0;
};*/

}
