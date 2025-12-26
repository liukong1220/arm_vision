# stereo_calibrate.py
import cv2
import numpy as np
import glob
import os
import yaml

# --- 配置参数 ---
# 从配置文件加载参数
try:
    with open('calibration/configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("错误：找不到配置文件 'config.yaml'")
    exit()

# 棋盘格内角点数量 (与 collect_images.py 保持一致)
CHECKERBOARD_SIZE = (config['pattern_cols'], config['pattern_rows'])
# 棋盘格方块的实际物理尺寸 (单位：mm，用于计算真实尺度)
SQUARE_SIZE = config['unit_size_mm']
# 图像路径
IMAGES_PATH = 'calibration/images/'
# 输出路径
OUTPUT_FILE = 'calibration/output/stereo_params.yaml'

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (30,0,0), (60,0,0) ...
objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2) * SQUARE_SIZE

# Arrays to store object points and image points from all images
objpoints = []  # 3d point in real world space
imgpoints_L = []  # 2d points in image plane for left camera
imgpoints_R = []  # 2d points in image plane for right camera

def run_calibration():
    # 确保输出目录存在
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # 查找左右目图像
    images_L = sorted(glob.glob(os.path.join(IMAGES_PATH, 'left_*.jpg')))
    images_R = sorted(glob.glob(os.path.join(IMAGES_PATH, 'right_*.jpg')))
    
    if len(images_L) == 0 or len(images_L) != len(images_R):
        print("错误：图像文件数量不匹配或文件夹为空。请先运行 collect_images.py")
        return

    print(f"找到 {len(images_L)} 对图像用于标定。")
    
    img_size = None

    for i, (fname_L, fname_R) in enumerate(zip(images_L, images_R)):
        img_L = cv2.imread(fname_L)
        img_R = cv2.imread(fname_R)

        if img_size is None:
            img_size = img_L.shape[:2][::-1] # (width, height)

        gray_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
        gray_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)

        # 查找角点
        retL, cornersL = cv2.findChessboardCorners(gray_L, CHECKERBOARD_SIZE, None)
        retR, cornersR = cv2.findChessboardCorners(gray_R, CHECKERBOARD_SIZE, None)

        if retL and retR:
            objpoints.append(objp)
            
            # 精细化角点位置
            cornersL = cv2.cornerSubPix(gray_L, cornersL, (11, 11), (-1, -1), criteria)
            imgpoints_L.append(cornersL)

            cornersR = cv2.cornerSubPix(gray_R, cornersR, (11, 11), (-1, -1), criteria)
            imgpoints_R.append(cornersR)

            print(f"处理并接受第 {i+1} 对图像。")
        else:
            print(f"警告：第 {i+1} 对图像未找到完整角点，跳过。")
    
    if len(objpoints) < 10:
        print("错误：成功处理的图像数量太少 (<10)，标定精度无法保证。")
        return

    # --- 1. 单目相机标定（为双目标定提供良好的初始值）---
    # R_vecs 和 T_vecs 这里用不到，但需要返回
    print("\n--- 正在进行左右目单目标定 ---")
    retL, K_L, D_L, r_L, t_L = cv2.calibrateCamera(objpoints, imgpoints_L, img_size, None, None)
    retR, K_R, D_R, r_R, t_R = cv2.calibrateCamera(objpoints, imgpoints_R, img_size, None, None)

    # --- 2. 双目相机标定 ---
    print("\n--- 正在进行双目标定 ---")
    ret, K_L, D_L, K_R, D_R, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_L, imgpoints_R, 
        K_L, D_L, K_R, D_R, img_size, criteria, 
        flags=cv2.CALIB_FIX_INTRINSIC # 可选：用单目标定结果作为固定内参
    )

    # --- 3. 立体校正 (Stereo Rectification) ---
    # 计算用于立体校正的旋转矩阵 R1, R2，投影矩阵 P1, P2
    # Q：用于将校正后的像素坐标重投影到 3D 空间 (深度计算的核心)
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_L, D_L, K_R, D_R, img_size, R, T, 
        alpha=0, # alpha=0 裁剪所有无效像素，alpha=1 保留所有像素
        newImageSize=img_size
    )

    print("\n--- 标定结果 ---")
    print(f"RMS 误差: {ret}")
    print("基线距离 (T 向量):", T.flatten())

    # --- 4. 保存参数 ---
    with open(OUTPUT_FILE, 'w') as f:
        # 使用 OpenCV 提供的 FileStorage 格式保存更方便
        fs = cv2.FileStorage(OUTPUT_FILE, cv2.FILE_STORAGE_WRITE)
        
        fs.write("comment_K_L", "左相机内参矩阵 (fx, fy, cx, cy)")
        fs.write("K_L", K_L)
        fs.write("comment_D_L", "左相机畸变系数 (k1, k2, p1, p2, k3)")
        fs.write("D_L", D_L)
        fs.write("comment_K_R", "右相机内参矩阵 (fx, fy, cx, cy)")
        fs.write("K_R", K_R)
        fs.write("comment_D_R", "右相机畸变系数 (k1, k2, p1, p2, k3)")
        fs.write("D_R", D_R)
        
        fs.write("comment_R", "从左相机到右相机的旋转矩阵 (外参)")
        fs.write("R", R)
        fs.write("comment_T", "从左相机到右相机的平移向量 (外参, 基线)")
        fs.write("T", T)
        
        fs.write("comment_E", "本质矩阵")
        fs.write("E", E)
        fs.write("comment_F", "基础矩阵")
        fs.write("F", F)
        
        fs.write("comment_R1", "左相机的校正旋转矩阵")
        fs.write("R1", R1)
        fs.write("comment_P1", "左相机在校正坐标系下的投影矩阵")
        fs.write("P1", P1)
        fs.write("comment_R2", "右相机的校正旋转矩阵")
        fs.write("R2", R2)
        fs.write("comment_P2", "右相机在校正坐标系下的投影矩阵")
        fs.write("P2", P2)
        
        fs.write("comment_Q", "重投影矩阵 (4x4)，用于从视差计算三维坐标")
        fs.write("Q", Q)
        fs.release()
    
    print(f"\n✅ 标定参数已保存到 {OUTPUT_FILE}")

if __name__ == '__main__':
    run_calibration()