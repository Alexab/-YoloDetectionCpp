#include <cstddef>
#include "Detector.h"

int main(int argc,char* argv[ ])
{
  detection::Detector::Parameters params;

  params.allowable_sources.push_back("Src1");
  params.model_path = "./yolov8_vtz_0_1_m_a3_400ep_batch3.trt";

  std::unique_ptr<detection::Detector> detector =  detection::Detector::make(params);

  
}