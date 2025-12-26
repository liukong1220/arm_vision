#include <iostream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>
#include <vector>
#include <stdexcept>

// 用于存储双目相机参数的结构体
struct StereoParams {
    cv::Mat M1, D1, M2, D2, R, T;
    cv::Size image_size;
};

// 从 YAML 文件加载双目相机参数（更新版本）
void load_stereo_params(const std::string& filename, StereoParams& params) {
    YAML::Node config;
    try {
        config = YAML::LoadFile(filename);
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("无法加载或解析YAML文件: " + std::string(e.what()));
    }

    if (!config["image_width"] || !config["image_height"]) {
        throw std::runtime_error("YAML 文件缺少 'image_width' 或 'image_height'");
    }
    params.image_size.width = config["image_width"].as<int>();
    params.image_size.height = config["image_height"].as<int>();

    // 辅助函数，将YAML序列解析为cv::Mat
    auto parse_list_to_mat = [](const YAML::Node& node, int rows, int cols, const std::string& name) {
        if (!node || !node.IsSequence() || node.size() != (rows * cols)) {
            throw std::runtime_error("YAML节点 '" + name + "' 不是序列或尺寸不正确。");
        }
        std::vector<double> data = node.as<std::vector<double>>();
        // 使用 data.data() 创建Mat，然后克隆以确保数据被复制
        cv::Mat mat(rows, cols, CV_64F, data.data());
        return mat.clone();
    };

    try {
        params.M1 = parse_list_to_mat(config["camera_matrix_left"], 3, 3, "camera_matrix_left");
        params.M2 = parse_list_to_mat(config["camera_matrix_right"], 3, 3, "camera_matrix_right");
        params.D1 = parse_list_to_mat(config["distort_coeffs_left"], 1, 5, "distort_coeffs_left");
        params.D2 = parse_list_to_mat(config["distort_coeffs_right"], 1, 5, "distort_coeffs_right");
        params.R = parse_list_to_mat(config["R"], 3, 3, "R");
        params.T = parse_list_to_mat(config["T"], 3, 1, "T");
    } catch (const std::exception& e) {
        throw std::runtime_error("从YAML解析矩阵/向量时出错: " + std::string(e.what()));
    }
}

// 存储鼠标点击和3D点信息
struct UserData {
    std::vector<cv::Point2f> points;
    cv::Mat P1, P2;
    cv::Mat combined_img;
    cv::Mat original_combined_img;
    StereoParams params;
};

// 鼠标回调函数 (无变化)
void onMouse(int event, int x, int y, int flags, void* userdata) {
    if (event != cv::EVENT_LBUTTONDOWN) {
        return;
    }

    UserData* data = static_cast<UserData*>(userdata);
    int width = data->params.image_size.width;

    cv::Point2f pt(x, y);

    if (data->points.size() < 2) {
        data->points.push_back(pt);
        data->combined_img = data->original_combined_img.clone();
        for(const auto& p : data->points) {
            cv::circle(data->combined_img, p, 5, cv::Scalar(0, 255, 0), -1);
        }
        cv::imshow("Stereo Verification", data->combined_img);
    }

    if (data->points.size() == 2) {
        cv::Point2f p1 = data->points[0];
        cv::Point2f p2 = data->points[1];

        if (!((p1.x < width && p2.x > width) || (p1.x > width && p2.x < width))) {
            std::cout << "请在左图中选择一个点，在右图中选择对应的点。" << std::endl;
            data->points.clear();
            data->original_combined_img.copyTo(data->combined_img);
            cv::imshow("Stereo Verification", data->combined_img);
            return;
        }

        if (p1.x > width) std::swap(p1, p2);

        p2.x -= width;

        std::vector<cv::Point2f> points1 = {p1};
        std::vector<cv::Point2f> points2 = {p2};

        cv::Mat points4D;
        cv::Mat P1_32F, P2_32F;
        data->P1.convertTo(P1_32F, CV_32F);
        data->P2.convertTo(P2_32F, CV_32F);
        cv::triangulatePoints(P1_32F, P2_32F, points1, points2, points4D);

        cv::Point3f point3d;
        point3d.x = points4D.at<float>(0, 0) / points4D.at<float>(3, 0);
        point3d.y = points4D.at<float>(1, 0) / points4D.at<float>(3, 0);
        point3d.z = points4D.at<float>(2, 0) / points4D.at<float>(3, 0);

        float distance = cv::norm(point3d);

        std::cout << "------------------------------" << std::endl;
        std::cout << "选择的点: " << p1 << " 和 " << p2 << std::endl;
        std::cout << "3D 坐标 (相机坐标系): " << point3d << std::endl;
        std::cout << "距相机的距离: " << distance << " (单位与标定板相同)" << std::endl;
        std::cout << "------------------------------" << std::endl;
        std::cout << "\n请选择一对新的点。" << std::endl;

        data->points.clear();
        data->original_combined_img.copyTo(data->combined_img);
        cv::imshow("Stereo Verification", data->combined_img);
    }
}

// Main 函数 (无变化)
int main(int argc, char** argv) {
    std::string stereo_params_path;
    std::string left_image_path;
    std::string right_image_path;

    if (argc == 1) {
        try {
            YAML::Node config = YAML::LoadFile("configs/stereo_config.yaml");
            stereo_params_path = config["verify_stereo"]["stereo_params_path"].as<std::string>();
            left_image_path = config["verify_stereo"]["left_image_path"].as<std::string>();
            right_image_path = config["verify_stereo"]["right_image_path"].as<std::string>();
        } catch (const YAML::Exception& e) {
            std::cerr << "加载 'configs/stereo_config.yaml' 时出错: " << e.what() << std::endl;
            std::cerr << "用法: " << argv[0] << " [<stereo_params.yaml> <left_image> <right_image>]" << std::endl;
            return -1;
        }
    } else if (argc == 4) {
        stereo_params_path = argv[1];
        left_image_path = argv[2];
        right_image_path = argv[3];
    } else {
        std::cerr << "用法: " << argv[0] << " [<stereo_params.yaml> <left_image> <right_image>]" << std::endl;
        std::cerr << "如果不提供参数，将从 'configs/stereo_config.yaml' 加载路径。" << std::endl;
        return -1;
    }

    StereoParams params;
    try {
        load_stereo_params(stereo_params_path, params);
    } catch (const std::exception& e) {
        std::cerr << "加载并解析参数文件 '" << stereo_params_path << "' 时出错: " << e.what() << std::endl;
        return -1;
    }

    cv::Mat left_img = cv::imread(left_image_path, cv::IMREAD_COLOR);
    cv::Mat right_img = cv::imread(right_image_path, cv::IMREAD_COLOR);

    if (left_img.empty() || right_img.empty()) {
        std::cerr << "无法打开或找到图像!" << std::endl;
        std::cerr << "左图路径: " << left_image_path << std::endl;
        std::cerr << "右图路径: " << right_image_path << std::endl;
        return -1;
    }

    if (left_img.size() != params.image_size || right_img.size() != params.image_size) {
        std::cerr << "YAML中的图像尺寸 (" << params.image_size << ") 与实际图像尺寸 (" << left_img.size() << ") 不匹配" << std::endl;
        return -1;
    }

    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(params.M1, params.D1, params.M2, params.D2, params.image_size, params.R, params.T, R1, R2, P1, P2, Q);

    cv::Mat map11, map12, map21, map22;
    cv::initUndistortRectifyMap(params.M1, params.D1, R1, P1, params.image_size, CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(params.M2, params.D2, R2, P2, params.image_size, CV_16SC2, map21, map22);

    cv::Mat left_rect, right_rect;
    cv::remap(left_img, left_rect, map11, map12, cv::INTER_LINEAR);
    cv::remap(right_img, right_rect, map21, map22, cv::INTER_LINEAR);

    for (int i = 0; i < left_rect.rows; i += 30) {
        cv::line(left_rect, cv::Point(0, i), cv::Point(left_rect.cols, i), cv::Scalar(0, 255, 0), 1);
        cv::line(right_rect, cv::Point(0, i), cv::Point(right_rect.cols, i), cv::Scalar(0, 255, 0), 1);
    }

    UserData userdata;
    userdata.P1 = P1;
    userdata.P2 = P2;
    userdata.params = params;

    cv::hconcat(left_rect, right_rect, userdata.original_combined_img);
    userdata.combined_img = userdata.original_combined_img.clone();

    cv::line(userdata.combined_img, cv::Point(params.image_size.width, 0), cv::Point(params.image_size.width, params.image_size.height), cv::Scalar(0, 0, 255), 2);
    cv::line(userdata.original_combined_img, cv::Point(params.image_size.width, 0), cv::Point(params.image_size.width, params.image_size.height), cv::Scalar(0, 0, 255), 2);

    cv::namedWindow("Stereo Verification", cv::WINDOW_AUTOSIZE);
    cv::imshow("Stereo Verification", userdata.combined_img);
    cv::setMouseCallback("Stereo Verification", onMouse, &userdata);

    std::cout << "双目标定验证程序。" << std::endl;
    std::cout << "在左图点击一个点，然后在右图点击对应的点，以计算三维坐标。" << std::endl;
    std::cout << "按任意键退出。" << std::endl;

    cv::waitKey(0);

    return 0;
}