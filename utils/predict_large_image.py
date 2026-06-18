#----------------------------------------------------#
#   大图分块滑动窗口预测脚本
#   适用于超大遥感影像的语义分割
#----------------------------------------------------#
import os
import time
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

from deeplab import DeeplabV3
from utils.utils import cvtColor, preprocess_input


class SlidingWindowPredictor:
    """滑动窗口分块预测器"""

    def __init__(self, deeplab_model, tile_size=512, overlap=128):
        """
        初始化分块预测器

        Args:
            deeplab_model: DeeplabV3模型实例
            tile_size: 分块大小，默认512
            overlap: 相邻块之间的重叠像素，默认128
        """
        self.model = deeplab_model
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap

    def predict_image(self, image_path, output_path=None):
        """
        对大图进行分块预测

        Args:
            image_path: 输入图像路径
            output_path: 输出路径，None则不保存

        Returns:
            全分辨率的分割结果 (H, W)
        """
        print(f"正在加载图像: {image_path}")

        # 读取图像
        image = Image.open(image_path)
        original_w, original_h = image.size
        image = cvtColor(image)

        print(f"图像尺寸: {original_w} x {original_h}")
        print(f"分块大小: {self.tile_size}, 重叠像素: {self.overlap}")

        # 转换为numpy数组
        image_np = np.array(image)

        # 创建与图像同尺寸的结果数组（存储类别ID）
        result = np.zeros((original_h, original_w), dtype=np.uint8)

        # 创建权重数组（用于平均重叠区域）
        weight_map = np.zeros((original_h, original_w), dtype=np.float32)

        # 计算需要的块数
        n_tiles_x = max(1, (original_w - self.tile_size) // self.stride + 1)
        n_tiles_y = max(1, (original_h - self.tile_size) // self.stride + 1)
        total_tiles = n_tiles_x * n_tiles_y

        print(f"分块网格: {n_tiles_x} x {n_tiles_y} = {total_tiles} 块")

        # 滑动窗口遍历
        tile_count = 0
        start_time = time.time()

        for y in range(0, original_h, self.stride):
            for x in range(0, original_w, self.stride):
                tile_count += 1

                # 计算当前块的右下角坐标
                y2 = min(y + self.tile_size, original_h)
                x2 = min(x + self.tile_size, original_w)

                # 如果图像小于tile_size，从右下角反向计算起始位置
                y1 = max(0, y2 - self.tile_size)
                x1 = max(0, x2 - self.tile_size)

                # 提取当前块
                tile = image_np[y1:y2, x1:x2]

                # 预测当前块
                tile_result = self._predict_tile(tile)

                # 累加到结果中（处理重叠区域的平均）
                # 创建一个权重掩码
                tile_h, tile_w = tile_result.shape
                tile_weight = np.ones((tile_h, tile_w), dtype=np.float32)

                result[y1:y2, x1:x2] = tile_result
                weight_map[y1:y2, x1:x2] = tile_weight

                # 打印进度
                if tile_count % 50 == 0 or tile_count == total_tiles:
                    elapsed = time.time() - start_time
                    eta = (elapsed / tile_count) * (total_tiles - tile_count)
                    print(f"进度: {tile_count}/{total_tiles} 块 ({100*tile_count/total_tiles:.1f}%) "
                          f"- 已用时: {elapsed:.1f}s - 预计剩余: {eta:.1f}s")

        elapsed_total = time.time() - start_time
        print(f"\n预测完成! 总耗时: {elapsed_total:.1f}s")

        # 保存结果
        if output_path:
            self._save_result(result, output_path, original_w, original_h)
            print(f"结果已保存到: {output_path}")

        return result

    def _predict_tile(self, tile):
        """
        预测单个块
        """
        # 转换为PIL图像
        tile_pil = Image.fromarray(tile)

        # 获取原始尺寸
        original_h, original_w = tile_pil.size[1], tile_pil.size[0]

        # 缩放到模型输入尺寸
        tile_resized = tile_pil.resize((self.tile_size, self.tile_size), Image.BICUBIC)

        # 预处理
        tile_data = np.array(tile_resized, np.float32)
        tile_data = preprocess_input(tile_data)
        tile_data = np.transpose(tile_data, (2, 0, 1))
        tile_data = np.expand_dims(tile_data, 0)

        # 推理
        with torch.no_grad():
            images = torch.from_numpy(tile_data)
            if self.model.cuda:
                images = images.cuda()

            pr = self.model.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            pr = pr.argmax(axis=-1)

        return pr.astype(np.uint8)

    def _save_result(self, result, output_path, original_w, original_h):
        """
        保存分割结果
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 创建彩色分割图
        seg_img = np.zeros((original_h, original_w, 3), dtype=np.uint8)
        for i, color in enumerate(self.model.colors):
            seg_img[result == i] = color

        # 保存彩色图
        seg_img_pil = Image.fromarray(seg_img)
        seg_img_pil.save(output_path)

        # 同时保存原始类别图（灰度）
        gray_path = output_path.replace('.png', '_labels.png').replace('.jpg', '_labels.png')
        gray_pil = Image.fromarray(result)
        gray_pil.save(gray_path)


def count_classes(result, num_classes, name_classes):
    """
    统计各类别像素数量和比例

    Args:
        result: 分割结果数组 (H, W)
        num_classes: 类别数量
        name_classes: 类别名称列表
    """
    total_pixels = result.shape[0] * result.shape[1]
    print("\n" + "=" * 60)
    print("类别统计:")
    print("-" * 60)
    print(f"{'类别':<20} | {'像素数':>12} | {'比例':>10}")
    print("-" * 60)

    for i in range(num_classes):
        count = np.sum(result == i)
        if count > 0:
            ratio = count / total_pixels * 100
            name = name_classes[i] if name_classes and i < len(name_classes) else f"类别{i}"
            print(f"{name:<20} | {count:>12} | {ratio:>9.2f}%")

    print("=" * 60)


if __name__ == "__main__":
    # 配置参数
    #-----------------------------------------------------#
    #   输入图像路径
    #-----------------------------------------------------#
    input_image = "img/2024_cut.tif"

    #-----------------------------------------------------#
    #   输出路径
    #-----------------------------------------------------#
    output_image = "img/output/caogao/2024_cut_segmented_xception_sgd2.png"

    #-----------------------------------------------------#
    #   分块预测参数
    #-----------------------------------------------------#
    tile_size = 512       # 分块大小（建议与训练时一致）
    overlap = 128         # 相邻块重叠像素数（越大边界越平滑，但越慢）

    #-----------------------------------------------------#
    #   类别名称（需要与训练时一致）
    #-----------------------------------------------------#
    name_classes = ["背景", "水系", "林地", "道路", "种植土地", "建（构）筑物"]

    #-----------------------------------------------------#
    #   其他配置（与DeeplabV3一致）
    #-----------------------------------------------------#
    count = True          # 是否统计各类别像素

    #-----------------------------------------------------#
    #   初始化模型
    #-----------------------------------------------------#
    print("=" * 60)
    print("DeeplabV3+ 大图分块预测")
    print("=" * 60)

    deeplab = DeeplabV3()

    #-----------------------------------------------------#
    #   创建分块预测器并执行预测
    #-----------------------------------------------------#
    predictor = SlidingWindowPredictor(
        deeplab_model=deeplab,
        tile_size=tile_size,
        overlap=overlap
    )

    result = predictor.predict_image(input_image, output_image)

    #-----------------------------------------------------#
    #   统计各类别
    #-----------------------------------------------------#
    if count:
        count_classes(result, deeplab.num_classes, name_classes)

    print("\n处理完成!")
