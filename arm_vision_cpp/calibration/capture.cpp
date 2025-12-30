#include <fmt/core.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <yaml-cpp/yaml.h>

// #include <Eigen/Dense>

#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/img_tools.hpp"

const std::string keys =
    "{help h usage ? | | Print this message}"
    "{config-path c  | configs/stereo_config.yaml | path to yaml config file}"
    "{@left-folder   | imgs/left/ | Path to save left images}"
    "{@right-folder  | imgs/right/ | Path to save right images}";

void capture_loop(const std::string& config_path, const std::string& left_folder, const std::string& right_folder) 
{

        // 读取标定板配置
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
    } catch (const YAML::BadFile& e) {
        tools::logger()->error("Cannot find config file '{}'", config_path);
        return;
    }
    
    // 捕获循环的实现
     // std::string pattern_type = config["pattern_type"].as<std::string>();
    int pattern_cols = config["pattern_cols"].as<int>();
    int pattern_rows = config["pattern_rows"].as<int>();
    cv::Size pattern_size(pattern_cols, pattern_rows);

    int left_cam_idx = config["camera"]["left_index"].as<int>();
    int right_cam_idx = config["camera"]["right_index"].as<int>();
    int WIDTH = config["camera"]["width"].as<int>();
    int HEIGHT = config["camera"]["height"].as<int>();

    int num_images = config["num_images"].as<int>();

    // 打开两个摄像头，并明确指定使用 V4L2 后端
    cv::VideoCapture cap_left(left_cam_idx, cv::CAP_V4L2);
    cv::VideoCapture cap_right(right_cam_idx, cv::CAP_V4L2);

    if (!cap_left.isOpened() || !cap_right.isOpened()) {
        tools::logger()->error("错误：无法打开摄像头设备\n");
        return ;
    }

    // 设置分辨率
    cap_left.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH / 2);
    cap_left.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);
    cap_right.set(cv::CAP_PROP_FRAME_WIDTH, WIDTH / 2);
    cap_right.set(cv::CAP_PROP_FRAME_HEIGHT, HEIGHT);

    tools::logger()->info("摄像头分辨率设置为: {}x{}\n", (int)cap_left.get(cv::CAP_PROP_FRAME_WIDTH), (int)cap_left.get(cv::CAP_PROP_FRAME_HEIGHT));
    tools::logger()->info("准备采集 {} 张图像...\n", num_images);


    cv::Mat frame_left, frame_right;
    int count = 0;

    while (count < num_images) {
        cap_left >> frame_left;
        cap_right >> frame_right;

        if (frame_left.empty() || frame_right.empty()) {
            tools::logger()->error("错误：捕获到空帧。\n");
            break;
        }

        // 创建用于绘制的副本
        cv::Mat img_left = frame_left.clone();
        cv::Mat img_right = frame_right.clone();

        std::vector<cv::Point2f> cornersL, cornersR;
        bool retL = false;
        bool retR = false;

        // 根据配置检测标定板
        // if (pattern_type == "chessboard") {
        retL = cv::findChessboardCorners(img_left, pattern_size, cornersL);
        retR = cv::findChessboardCorners(img_right, pattern_size, cornersR);
        // } else if (pattern_type == "circles_grid") {
        //     found_left = cv::findCirclesGrid(left_drawing, pattern_size, left_corners, cv::CALIB_CB_SYMMETRIC_GRID);
        //     found_right = cv::findCirclesGrid(right_drawing, pattern_size, right_corners, cv::CALIB_CB_SYMMETRIC_GRID);
        // }

        // 在图像上绘制检测结果
        if (retL) {
            cv::drawChessboardCorners(img_left, pattern_size, cornersL, retL);
        }
        if (retR) {
            cv::drawChessboardCorners(img_right, pattern_size, cornersR, retR);
        }

        // 将两个视频帧水平拼接在一起
        cv::Mat display_frame;
        // if (left_drawing.rows != right_drawing.rows) {
        //     cv::resize(right_drawing, right_drawing, left_drawing.size());
        // }
        cv::hconcat(img_left, img_right, display_frame);

        cv::putText(display_frame, fmt::format("Image: {}/{}", count, num_images), cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        cv::putText(display_frame, "Press 's' to Save, 'q' to Quit", cv::Point(50, HEIGHT - 20), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 255), 2);
        
        // 缩小图像以便显示
        // cv::Mat resized_frame;
        // cv::resize(combined_frame, resized_frame, cv::Size(), 0.4, 0.4);
        cv::imshow("Stereo Image Capture (Press S to Save)", display_frame);

        char k = (char)cv::waitKey(1);

        if (k == 'q' || k == 27) { // 'q' or ESC
            break;
        } else if (k == 's') {
            // 只有当左右相机都成功检测到标定板时才保存
            if (retL && retR) {
                std::string filename_L = fmt::format("{}/{:03d}.jpg", left_folder, count);
                std::string filename_R = fmt::format("{}/{:03d}.jpg", right_folder, count);

                cv::imwrite(filename_L, frame_left); // 保存原始图像
                cv::imwrite(filename_R, frame_right);

                tools::logger()->info("成功保存图像: {} 和 {}\n", filename_L, filename_R);
                count++;
            } else {
                tools::logger()->warn("未找到完整的标定板角点，请调整位置后再按 's'。\n");
            }
        }
    }

    cap_left.release();
    cap_right.release();
    cv::destroyAllWindows();

}


int main(int argc, char* argv[]) {
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    std::string config_path = parser.get<std::string>("config-path");
    std::string left_folder = parser.get<std::string>(0);
    std::string right_folder = parser.get<std::string>(1);

    std::filesystem::create_directories(left_folder);
    std::filesystem::create_directories(right_folder);

    capture_loop(config_path, left_folder, right_folder);
   
    return 0;
}