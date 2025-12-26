# collect_images.py
import cv2
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

# 标定板类型
# pattern_type = config['pattern_type']
# 标定板尺寸
pattern_cols = config['pattern_cols']
pattern_rows = config['pattern_rows']
pattern_size = (pattern_cols, pattern_rows)
# 摄像头参数
left_index = config['camera']['left_index']
right_index = config['camera']['right_index']
# 采集的图像数量
NUM_IMAGES = config['num_images']


# 摄像头模组输出的最大分辨率 (根据规格书)
WIDTH = 1080
HEIGHT = 640
# 标定图保存路径
SAVE_PATH = 'calibration/images/'

# --- 棋盘格参数 ---
# 棋盘格内角点数量 (例如：7x5 的棋盘有 6x4 个内角点)
# 假设你的棋盘是 9x6 格子，则内角点是 8x5
CHECKERBOARD_SIZE = (pattern_cols, pattern_rows)


def capture_images():
    """采集并分割左右目图像"""
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    cap_left = cv2.VideoCapture(left_index)
    cap_right = cv2.VideoCapture(right_index)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print(f"错误：无法打开摄像头设备")
        return

    # 设置分辨率
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH / 2)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH / 2)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)


    print(f"摄像头分辨率设置为: {int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"准备采集 {NUM_IMAGES} 张图像...")

    count = 0
    while count < NUM_IMAGES:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            continue

        # 将 2560x720 宽图分割成左右两张 1280x720 图像
        # 左图：(0:720, 0:1280)
        # 右图：(0:720, 1280:2560)
        img_left = frame_left.copy()
        img_right = frame_right.copy()
        
        # 根据标定板类型查找角点
        # if pattern_type == 'chessboard':
        retL, cornersL = cv2.findChessboardCorners(img_left, pattern_size, None)
        retR, cornersR = cv2.findChessboardCorners(img_right, pattern_size, None)
        # elif pattern_type == 'circles_grid':
        #     retL, cornersL = cv2.findCirclesGrid(img_left, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
        #     retR, cornersR = cv2.findCirclesGrid(img_right, pattern_size, flags=cv2.CALIB_CB_SYMMETRIC_GRID)
        # else:
        #     print(f"错误: 不支持的标定板类型 '{pattern_type}'")
        #     break

        # 如果找到角点，则绘制出来
        if retL:
            cv2.drawChessboardCorners(img_left, pattern_size, cornersL, retL)
        if retR:
            cv2.drawChessboardCorners(img_right, pattern_size, cornersR, retR)

        display_frame = cv2.hconcat([img_left, img_right])
        cv2.putText(display_frame, f"Image: {count}/{NUM_IMAGES}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, "Press 's' to Save, 'q' to Quit", (50, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Stereo Image Capture (Press S to Save)', display_frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('s'):
            if retL and retR:
                # 确保左右两图都找到了角点才保存
                filename_L = os.path.join(SAVE_PATH, f"left_{count:03d}.jpg")
                filename_R = os.path.join(SAVE_PATH, f"right_{count:03d}.jpg")
                
                cv2.imwrite(filename_L, frame_left) # 保存原始图像
                cv2.imwrite(filename_R, frame_right) # 保存原始图像
                print(f"成功保存第 {count:03d} 对图像")
                count += 1
            else:
                print("未找到完整的标定板角点，请调整位置后再按 's'。")

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_images()