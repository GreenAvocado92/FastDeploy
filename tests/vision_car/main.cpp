#include "fastdeploy/vision.h"

void CpuInfer(const std::string& model_dir, const std::string& image_file) {
  auto model_file = model_dir +  "model.pdmodel";
  auto params_file = model_dir + "model.pdiparams";
  auto config_file = model_dir + "infer_cfg.yml";
  auto option = fastdeploy::RuntimeOption();
  option.UseCpu();
  auto model = fastdeploy::vision::detection::PPYOLO(model_file, params_file,
                                                     config_file, option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);

  fastdeploy::vision::DetectionResult res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << res.Str() << std::endl;
  auto vis_im = fastdeploy::vision::VisDetection(im, res, 0.5);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
}

int main(int argc, char* argv[]) {
  std::string model_dir = argv[1];
  auto model_file = model_dir +  "model.pdmodel";
  auto params_file = model_dir + "model.pdiparams";
  auto config_file = model_dir + "infer_cfg.yml";
  auto option = fastdeploy::RuntimeOption();
  option.UseCpu();
  auto model = fastdeploy::vision::detection::PPYOLO(model_file, params_file,
                                                     config_file, option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }
  
  auto im = cv::imread(image_file);

  fastdeploy::vision::DetectionResult res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << res.Str() << std::endl;
  auto vis_im = fastdeploy::vision::VisDetection(im, res, 0.5);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;

  CpuInfer(argv[1], argv[2]);
  return 0;
}
