#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <opencv2/opencv.hpp>

#include "tools/img_tools.hpp"

const std::string keys =
  "{help h usage ? |                          | a stereo calibration program}"
  "{config-path c  | configs/stereo_config.yaml | path to yaml config file}"
  "{@left-folder   | imgs/left/                | path to left camera images}"
  "{@right-folder  | imgs/right/               | path to right camera images}";

std::vector<cv::Point3f> create_object_points(const cv::Size& pattern_size, float unit_size) {
  std::vector<cv::Point3f> object_points;
  for (int i = 0; i < pattern_size.height; i++) {
    for (int j = 0; j < pattern_size.width; j++) {
      object_points.push_back({j * unit_size, i * unit_size, 0});
    }
  }
  return object_points;
}

void load(
  const std::string& left_folder, const std::string& right_folder, const std::string& config_path,
  cv::Size& img_size, std::vector<std::vector<cv::Point3f>>& obj_points,
  std::vector<std::vector<cv::Point2f>>& left_img_points,
  std::vector<std::vector<cv::Point2f>>& right_img_points) {
  auto yaml = YAML::LoadFile(config_path);
  std::string pattern_type = yaml["pattern_type"].as<std::string>();
  int pattern_cols = yaml["pattern_cols"].as<int>();
  int pattern_rows = yaml["pattern_rows"].as<int>();
  float unit_size_mm = yaml["unit_size_mm"].as<float>();
  cv::Size pattern_size(pattern_cols, pattern_rows);

  auto object_points = create_object_points(pattern_size, unit_size_mm);

  for (int i = 0; true; i++) {
    auto left_img_path = fmt::format("{}/{:03d}.jpg", left_folder, i);
    auto right_img_path = fmt::format("{}/{:03d}.jpg", right_folder, i);
    auto left_img = cv::imread(left_img_path);
    auto right_img = cv::imread(right_img_path);

    if (left_img.empty() || right_img.empty()) break;

    img_size = left_img.size();

    std::vector<cv::Point2f> left_corners, right_corners;
    bool found_left = false;
    bool found_right = false;

    if (pattern_type == "chessboard") {
      found_left = cv::findChessboardCorners(left_img, pattern_size, left_corners);
      found_right = cv::findChessboardCorners(right_img, pattern_size, right_corners);
    } else if (pattern_type == "circles_grid") {
      found_left = cv::findCirclesGrid(left_img, pattern_size, left_corners, cv::CALIB_CB_SYMMETRIC_GRID);
      found_right = cv::findCirclesGrid(right_img, pattern_size, right_corners, cv::CALIB_CB_SYMMETRIC_GRID);
    } else {
      fmt::print("Error: Unknown pattern type '{}'. Use 'chessboard' or 'circles_grid'.\n", pattern_type);
      return;
    }

    auto left_drawing = left_img.clone();
    cv::drawChessboardCorners(left_drawing, pattern_size, left_corners, found_left);
    cv::resize(left_drawing, left_drawing, {}, 0.5, 0.5);
    cv::imshow("Left", left_drawing);

    auto right_drawing = right_img.clone();
    cv::drawChessboardCorners(right_drawing, pattern_size, right_corners, found_right);
    cv::resize(right_drawing, right_drawing, {}, 0.5, 0.5);
    cv::imshow("Right", right_drawing);

    cv::waitKey(0);

    fmt::print("[{}] Left: {}, Right: {}\n", (found_left && found_right) ? "success" : "failure", left_img_path, right_img_path);

    if (found_left && found_right) {
      // For chessboard, refine corner locations
      if (pattern_type == "chessboard") {
          cv::Mat left_gray, right_gray;
          cv::cvtColor(left_img, left_gray, cv::COLOR_BGR2GRAY);
          cv::cvtColor(right_img, right_gray, cv::COLOR_BGR2GRAY);
          cv::cornerSubPix(left_gray, left_corners, cv::Size(11, 11), cv::Size(-1, -1),
                           cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
          cv::cornerSubPix(right_gray, right_corners, cv::Size(11, 11), cv::Size(-1, -1),
                           cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
      }
      left_img_points.emplace_back(left_corners);
      right_img_points.emplace_back(right_corners);
      obj_points.emplace_back(object_points);
    }
  }
  cv::destroyAllWindows();
}

void print_yaml(
  const cv::Mat& M1, const cv::Mat& D1, const cv::Mat& M2, const cv::Mat& D2, const cv::Mat& R,
  const cv::Mat& T, const cv::Mat& E, const cv::Mat& F, double error, const cv::Size& img_size) {
  YAML::Emitter result;
  std::vector<double> m1d(M1.begin<double>(), M1.end<double>());
  std::vector<double> d1d(D1.begin<double>(), D1.end<double>());
  std::vector<double> m2d(M2.begin<double>(), M2.end<double>());
  std::vector<double> d2d(D2.begin<double>(), D2.end<double>());
  std::vector<double> rd(R.begin<double>(), R.end<double>());
  std::vector<double> td(T.begin<double>(), T.end<double>());
  std::vector<double> ed(E.begin<double>(), E.end<double>());
  std::vector<double> fd(F.begin<double>(), F.end<double>());

  result << YAML::BeginMap;
  result << YAML::Comment(fmt::format("Reprojection error (重投影误差): {:.4f}px", error));
  result << YAML::Key << "image_width" << YAML::Value << img_size.width;
  result << YAML::Key << "image_height" << YAML::Value << img_size.height;
  result << YAML::Newline;
  result << YAML::Comment("Left camera intrinsics (fx, fy, cx, cy) / 左相机内参");
  result << YAML::Key << "camera_matrix_left" << YAML::Value << YAML::Flow << m1d;
  result << YAML::Comment("Left camera distortion coefficients (k1, k2, p1, p2, k3) / 左相机畸变系数");
  result << YAML::Key << "distort_coeffs_left" << YAML::Value << YAML::Flow << d1d;
  result << YAML::Newline;
  result << YAML::Comment("Right camera intrinsics (fx, fy, cx, cy) / 右相机内参");
  result << YAML::Key << "camera_matrix_right" << YAML::Value << YAML::Flow << m2d;
  result << YAML::Comment("Right camera distortion coefficients (k1, k2, p1, p2, k3) / 右相机畸变系数");
  result << YAML::Key << "distort_coeffs_right" << YAML::Value << YAML::Flow << d2d;
  result << YAML::Newline;
  result << YAML::Comment("Rotation matrix between left and right cameras / 左右相机旋转矩阵");
  result << YAML::Key << "R" << YAML::Value << YAML::Flow << rd;
  result << YAML::Comment("Translation vector between left and right cameras / 左右相机平移向量");
  result << YAML::Key << "T" << YAML::Value << YAML::Flow << td;
  result << YAML::Comment("Essential matrix / 本质矩阵");
  result << YAML::Key << "E" << YAML::Value << YAML::Flow << ed;
  result << YAML::Comment("Fundamental matrix / 基础矩阵");
  result << YAML::Key << "F" << YAML::Value << YAML::Flow << fd;
  result << YAML::EndMap;

  fmt::print("\n--- Stereo Calibration Result ---\n");
  fmt::print("{}\n", result.c_str());

  // 将结果保存到文件
  std::string output_path = "configs/stereo_params.yaml";
  std::ofstream fout(output_path);
  fout << result.c_str();
  fout.close();
  fmt::print("Calibration parameters saved to {}\n", output_path);
}

int main(int argc, char* argv[]) {
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>("config-path");
  auto left_folder = cli.get<std::string>(0);
  auto right_folder = cli.get<std::string>(1);
  

  cv::Size img_size;
  std::vector<std::vector<cv::Point3f>> obj_points;
  std::vector<std::vector<cv::Point2f>> left_img_points, right_img_points;
  load(left_folder, right_folder, config_path, img_size, obj_points, left_img_points, right_img_points);

  if (obj_points.empty()) {
      fmt::print("Error: No valid image pairs found. Make sure the pattern is detected in both images.\n");
      return -1;
  }

  cv::Mat M1, D1, M2, D2, R, T, E, F;
  auto criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, DBL_EPSILON);

  // 先对左右相机分别进行单目标定，为双目标定提供好的初值
  cv::calibrateCamera(obj_points, left_img_points, img_size, M1, D1, cv::noArray(), cv::noArray());
  cv::calibrateCamera(obj_points, right_img_points, img_size, M2, D2, cv::noArray(), cv::noArray());

  double error = cv::stereoCalibrate(
    obj_points, left_img_points, right_img_points, M1, D1, M2, D2, img_size, R, T, E, F,
    cv::CALIB_FIX_INTRINSIC, criteria);

  print_yaml(M1, D1, M2, D2, R, T, E, F, error, img_size);

  return 0;
}