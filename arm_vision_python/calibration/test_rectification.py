# test_rectification.py
import cv2
import numpy as np
import os
import yaml

# --- 配置参数 ---
CONFIG_FILE = 'calibration/configs/config.yaml'

def load_config(config_file):
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def load_calibration_parameters(filename):
    """从 YAML 文件中读取标定参数"""
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"无法打开参数文件: {filename}")
    
    params = {}
    params['K_L'] = fs.getNode("K_L").mat()
    params['D_L'] = fs.getNode("D_L").mat()
    params['K_R'] = fs.getNode("K_R").mat()
    params['D_R'] = fs.getNode("D_R").mat()
    params['R1'] = fs.getNode("R1").mat()
    params['P1'] = fs.getNode("P1").mat()
    params['R2'] = fs.getNode("R2").mat()
    params['P2'] = fs.getNode("P2").mat()
    params['Q'] = fs.getNode("Q").mat()
    fs.release()
    return params

def test_rectification(config):
    try:
        params = load_calibration_parameters(config['stereo_params_file'])
    except FileNotFoundError as e:
        print(e)
        print("请确保先运行 stereo_calibrate.py 成功生成参数文件。")
        return

    # 1. 读取测试图像
    img_L = cv2.imread(config['test_image_left'], cv2.IMREAD_GRAYSCALE)
    img_R = cv2.imread(config['test_image_right'], cv2.IMREAD_GRAYSCALE)
    
    if img_L is None or img_R is None:
        print(f"错误：无法读取测试图像。请检查路径：{config['test_image_left']} 和 {config['test_image_right']}")
        return

    # 2. 计算映射矩阵
    # map1_L, map2_L, map1_R, map2_R 用于重映射图像
    R_L, P_L = params['R1'], params['P1']
    R_R, P_R = params['R2'], params['P2']
    K_L, D_L = params['K_L'], params['D_L']
    K_R, D_R = params['K_R'], params['D_R']
    
    map1_L, map2_L = cv2.initUndistortRectifyMap(K_L, D_L, R_L, P_L, img_L.shape[::-1], cv2.CV_16SC2)
    map1_R, map2_R = cv2.initUndistortRectifyMap(K_R, D_R, R_R, P_R, img_R.shape[::-1], cv2.CV_16SC2)

    # 3. 进行图像校正 (Rectification)
    rectified_L = cv2.remap(img_L, map1_L, map2_L, cv2.INTER_LINEAR)
    rectified_R = cv2.remap(img_R, map1_R, map2_R, cv2.INTER_LINEAR)

    # 检查校正效果：图像上的特征点应位于同一水平线上 (极线对齐)
    vis = np.zeros((max(rectified_L.shape[0], rectified_R.shape[0]), rectified_L.shape[1] * 2), np.uint8)
    vis[:rectified_L.shape[0], :rectified_L.shape[1]] = rectified_L
    vis[:rectified_R.shape[0], rectified_L.shape[1]:] = rectified_R
    
    # 画出水平线以检查对齐情况
    for i in range(20, vis.shape[0], 50):
        cv2.line(vis, (0, i), (vis.shape[1], i), (255, 0, 0), 1)

    # 4. 视差图计算 (使用 SGBM)
    sgbm_config = config['sgbm']
    window_size = sgbm_config['window_size']
    min_disp = sgbm_config['min_disparity']
    num_disp = sgbm_config['num_disparities']
    
    # 调参是 SGBM 质量的关键，可根据实际效果调整
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=sgbm_config['disp12_max_diff'],
        uniquenessRatio=sgbm_config['uniqueness_ratio'],
        speckleWindowSize=sgbm_config['speckle_window_size'],
        speckleRange=sgbm_config['speckle_range']
    )

    disparity = stereo.compute(rectified_L, rectified_R).astype(np.float32) / 16.0
    
    # 归一化用于显示
    disparity_vis = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imshow('Rectified Images (Check Horizontal Alignment)', vis)
    cv2.imshow('Disparity Map (Grayscale Depth)', disparity_vis)
    
    # 保存结果
    output_config = config['output_paths']
    cv2.imwrite(output_config['rectified_pair'], vis)
    cv2.imwrite(output_config['disparity_map'], disparity_vis)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    config = load_config(CONFIG_FILE)
    test_rectification(config)