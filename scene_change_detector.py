#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高效视频场景检测
不使用DNN的高效实现 - 多进程版本 (127进程) + 固定10帧采样

功能：
1. 检测视频中的场景变化
2. 提取场景片段（确保每个片段不低于5秒）
3. 使用127个进程进行并行处理
4. 每个视频只采样10帧，提高处理速度
"""

import threading
import cv2
import numpy as np
import logging
import os
import glob
import subprocess
import concurrent.futures
import multiprocessing
import json
import time
import sys
import argparse
from collections import deque
from typing import Dict, List, Tuple, Union, Optional
from tqdm import tqdm  # 导入tqdm用于显示进度条

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置进程数量
DEFAULT_PROCESS_COUNT = 127

# 定义全局变量 - 不在此处初始化
global_lock = None
global_progress = None
pbar = None
progress_event = None

#######################
# 场景变化检测模块实现 #
#######################

class FastSceneChangeDetector:
    """
    改进的场景变化检测模块，使用固定10帧采样和多进程处理
    """

    def __init__(self, config: Dict = None):
        """
        初始化场景变化检测模块

        Args:
            config: 配置字典，包含检测参数
        """
        self.config = config or {}

        # 场景变化检测参数
        self.base_threshold = self.config.get("scene_change_threshold", 30.0)  # 基础场景变化阈值
        self.min_scene_length = self.config.get("min_scene_length", 10)  # 最小场景长度（帧）
        self.detector_method = self.config.get("detector_method", "histogram")  # 检测方法
        
        # 自适应阈值参数
        self.adaptive_threshold = True  # 是否使用自适应阈值
        self.history_size = 5  # 历史差异值队列大小（减小以加快处理）
        self.diff_history = deque(maxlen=self.history_size)
        self.adaptive_factor = 1.5  # 自适应因子
        
        # 多进程参数
        self.max_workers = self.config.get("max_workers", DEFAULT_PROCESS_COUNT)
        
        logger.info(f"快速场景变化检测模块初始化完成，使用{self.max_workers}个进程和10帧采样")

    def calculate_frame_difference(self, prev_frame, curr_frame, method="histogram"):
        """
        计算两帧之间的差异

        Args:
            prev_frame: 前一帧
            curr_frame: 当前帧
            method: 差异计算方法，可选 "histogram"（直方图比较）或 "pixel"（像素差异）

        Returns:
            float: 差异值
        """
        if prev_frame is None or curr_frame is None:
            return 0.0

        if method == "histogram":
            # 简化直方图参数，加快处理
            h_bins = 30
            s_bins = 32
            
            # 仅使用图像中心区域计算直方图（提升速度）
            height, width = prev_frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            roi_size = min(width, height) // 3  # 减小ROI区域加快速度
            
            prev_roi = prev_frame[center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size]
            curr_roi = curr_frame[center_y-roi_size:center_y+roi_size, center_x-roi_size:center_x+roi_size]
            
            # 将图像转换为HSV颜色空间
            prev_hsv = cv2.cvtColor(prev_roi, cv2.COLOR_BGR2HSV)
            curr_hsv = cv2.cvtColor(curr_roi, cv2.COLOR_BGR2HSV)

            # 计算HSV各通道的直方图
            hist_size = [h_bins, s_bins]
            h_ranges = [0, 180]
            s_ranges = [0, 256]
            ranges = h_ranges + s_ranges
            channels = [0, 1]  # H和S通道

            prev_hist = cv2.calcHist([prev_hsv], channels, None, hist_size, ranges, accumulate=False)
            curr_hist = cv2.calcHist([curr_hsv], channels, None, hist_size, ranges, accumulate=False)

            # 归一化直方图
            cv2.normalize(prev_hist, prev_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(curr_hist, curr_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # 计算直方图差异（相似度越小，差异越大）
            similarity = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)

            # 转换为差异值（0-100），值越大表示差异越大
            diff = (1.0 - similarity) * 100.0

            return diff

        elif method == "pixel":
            # 转换为灰度图并应用均值模糊
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            # 使用均值滤波代替高斯模糊（更快）
            prev_blur = cv2.blur(prev_gray, (5, 5))
            curr_blur = cv2.blur(curr_gray, (5, 5))

            # 计算帧差
            frame_diff = cv2.absdiff(prev_blur, curr_blur)
            
            # 应用阈值处理减少噪声影响
            _, thresholded = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)

            # 计算差异百分比
            non_zero = np.count_nonzero(thresholded)
            total_pixels = thresholded.shape[0] * thresholded.shape[1]
            diff_percentage = (non_zero / total_pixels) * 100.0

            return diff_percentage

        else:
            logger.warning(f"未知的差异计算方法: {method}，使用像素差异")
            return self.calculate_frame_difference(prev_frame, curr_frame, "pixel")

    def _get_current_threshold(self, diff_values):
        """获取当前自适应阈值"""
        if not self.adaptive_threshold or len(diff_values) < 3:
            return self.base_threshold
        
        # 计算差异值的均值和标准差
        mean_diff = np.mean(diff_values)
        std_diff = np.std(diff_values)
        
        # 自适应阈值 = 平均值 + 自适应因子 * 标准差
        return mean_diff + self.adaptive_factor * std_diff

    def sample_frames_evenly(self, video_path, num_samples=10):
        """
        均匀采样视频帧
        
        Args:
            video_path: 视频文件路径
            num_samples: 采样帧数量
            
        Returns:
            list: (帧索引, 帧) 元组列表
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return []
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 确保不超过视频总帧数
        num_samples = min(num_samples, total_frames)
        
        # 计算采样间隔
        if num_samples <= 1:
            interval = 1
        else:
            interval = total_frames / (num_samples - 1)
        
        # 采样帧
        sampled_frames = []
        
        for i in range(num_samples):
            # 计算当前采样帧索引
            frame_idx = min(int(i * interval), total_frames - 1)
            
            # 设置当前帧位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            
            # 读取帧
            ret, frame = cap.read()
            
            if ret:
                # 调整帧大小以加快处理速度
                frame = cv2.resize(frame, (320, 180))  # 进一步减小尺寸加快处理
                sampled_frames.append((frame_idx, frame))
            else:
                logger.warning(f"无法读取第 {frame_idx} 帧")
        
        cap.release()
        return sampled_frames

    def detect_scene_changes_from_samples(self, sampled_frames):
        """
        从采样帧中检测场景变化
        
        Args:
            sampled_frames: 采样帧列表，每个元素为(帧索引, 帧)元组
            
        Returns:
            list: 场景变化点列表，每个元素为帧索引
        """
        if len(sampled_frames) < 2:
            return []
            
        scene_changes = []
        diff_values = []
        
        for i in range(1, len(sampled_frames)):
            prev_idx, prev_frame = sampled_frames[i-1]
            curr_idx, curr_frame = sampled_frames[i]
            
            # 计算差异
            diff = self.calculate_frame_difference(prev_frame, curr_frame, self.detector_method)
            diff_values.append(diff)
            
            # 实时计算阈值
            threshold = self._get_current_threshold(diff_values)
            
            # 如果差异超过阈值，认为是场景变化
            if diff > threshold:
                # 使用中间帧作为场景变化点
                change_idx = prev_idx + (curr_idx - prev_idx) // 2
                scene_changes.append(change_idx)
                logger.debug(f"在第 {change_idx} 帧检测到场景变化，差异值: {diff:.2f}, 阈值: {threshold:.2f}")
        
        return scene_changes

    def detect_scene_changes(self, video_path, num_samples=10):
        """
        检测视频中的场景变化，只采样固定数量的帧
        
        Args:
            video_path: 视频文件路径
            num_samples: 采样帧数量，默认10帧
            
        Returns:
            list: 场景变化点列表，每个元素为帧索引
        """
        # 采样帧
        sampled_frames = self.sample_frames_evenly(video_path, num_samples)
        
        if len(sampled_frames) < 2:
            logger.warning(f"视频 {video_path} 采样不足，无法检测场景变化")
            return []
        
        # 检测场景变化
        scene_changes = self.detect_scene_changes_from_samples(sampled_frames)
        
        # 过滤太近的场景变化点
        if scene_changes:
            filtered_changes = [scene_changes[0]]
            
            for change in scene_changes[1:]:
                if change - filtered_changes[-1] >= self.min_scene_length:
                    filtered_changes.append(change)
            
            scene_changes = filtered_changes
        
        logger.info(f"检测到 {len(scene_changes)} 个场景变化点")
        return scene_changes

    def get_scenes(self, video_path, num_samples=10):
        """
        获取视频的场景列表
        
        Args:
            video_path: 视频文件路径
            num_samples: 采样帧数量，默认10帧
            
        Returns:
            list: 场景列表，每个元素为(开始帧，结束帧)元组
        """
        scene_changes = self.detect_scene_changes(video_path, num_samples)
        
        # 获取视频总帧数
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # 构建场景列表
        scenes = []
        
        if not scene_changes:
            # 如果没有检测到场景变化，整个视频作为一个场景
            scenes.append((0, total_frames - 1))
        else:
            # 第一个场景从0开始到第一个场景变化点
            scenes.append((0, scene_changes[0] - 1))
            
            # 中间的场景
            for i in range(len(scene_changes) - 1):
                scenes.append((scene_changes[i], scene_changes[i + 1] - 1))
            
            # 最后一个场景从最后一个场景变化点到视频结束
            scenes.append((scene_changes[-1], total_frames - 1))
        
        logger.info(f"视频划分为 {len(scenes)} 个场景")
        return scenes


#######################
# 视频处理功能实现     #
#######################

def get_video_info(video_path):
    """
    获取视频信息（分辨率、时长、帧率等）
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        dict: 视频信息
    """
    video_info = {}
    
    # 首先尝试使用FFprobe获取视频信息（更完整）
    cmd = [
        'ffprobe',
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        info = json.loads(result.stdout)
        
        # 查找视频流和音频流
        video_stream = None
        audio_stream = None
        
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video' and not video_stream:
                video_stream = stream
            elif stream.get('codec_type') == 'audio' and not audio_stream:
                audio_stream = stream
        
        if video_stream:
            video_info['width'] = int(video_stream.get('width', 0))
            video_info['height'] = int(video_stream.get('height', 0))
            
            # 获取帧率
            frame_rate = video_stream.get('r_frame_rate', '').split('/')
            if len(frame_rate) == 2 and int(frame_rate[1]) != 0:
                video_info['fps'] = round(int(frame_rate[0]) / int(frame_rate[1]), 2)
            else:
                video_info['fps'] = 0
            
            # 获取总帧数
            video_info['frames'] = int(video_stream.get('nb_frames', 0))
            
            # 如果没有总帧数信息，从时长和帧率计算
            if video_info['frames'] == 0 and video_info['fps'] > 0:
                duration = float(video_stream.get('duration', 0))
                if duration == 0:
                    duration = float(info.get('format', {}).get('duration', 0))
                video_info['frames'] = int(duration * video_info['fps'])
            
            video_info['codec'] = video_stream.get('codec_name', '')
        
        # 获取时长
        video_info['duration'] = float(info.get('format', {}).get('duration', 0))
        
        # 检查是否有音频
        video_info['has_audio'] = audio_stream is not None
        
        return video_info
    
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        logger.error(f"FFprobe获取视频信息失败: {str(e)}")
    
    # 如果FFprobe失败，使用OpenCV尝试获取基本信息
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"OpenCV无法打开视频: {video_path}")
            return {}
            
        # 获取基本视频属性
        video_info['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_info['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_info['fps'] = cap.get(cv2.CAP_PROP_FPS)
        video_info['frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_info['duration'] = video_info['frames'] / video_info['fps'] if video_info['fps'] > 0 else 0
        video_info['has_audio'] = False  # OpenCV不能检测音频
        
        # 如果获取到的帧率为0，使用默认值
        if video_info['fps'] <= 0:
            video_info['fps'] = 25.0
            logger.warning(f"无法获取准确帧率，使用默认值: 25.0 fps")
            
        cap.release()
        return video_info
        
    except Exception as e:
        logger.error(f"使用OpenCV获取视频信息失败: {str(e)}")
        return {}


def extract_video_segment_with_audio(video_path, start_frame, end_frame, output_path, config=None):
    """
    提取视频片段并保存到输出路径（包含音频），使用智能参数
    
    Args:
        video_path: 输入视频路径
        start_frame: 起始帧
        end_frame: 结束帧
        output_path: 输出视频路径
        config: 额外配置参数
        
    Returns:
        bool: 是否成功
    """
    config = config or {}
    
    # 获取视频信息
    video_info = get_video_info(video_path)
    
    if not video_info:
        logger.error(f"无法获取视频信息: {video_path}")
        return False
    
    # 计算开始和结束时间（秒）
    fps = video_info.get('fps', 25)  # 默认使用25fps
    start_time = start_frame / fps
    duration = (end_frame - start_frame + 1) / fps
    
    # 智能选择编码参数 - 加快编码速度
    codec = config.get('codec', 'libx264')  # 默认使用H.264
    quality = config.get('quality', 'ultrafast')  # 使用最快的编码速度
    crf = config.get('crf', 28)  # 提高CRF值，降低质量以加快速度（值越大质量越低）
    
    # 根据视频分辨率选择合适的比特率
    height = video_info.get('height', 0)
    if height > 1080:
        bitrate = '4M'  # 4K视频 (降低比特率加快处理)
    elif height > 720:
        bitrate = '2M'  # 1080p
    elif height > 480:
        bitrate = '1M'  # 720p
    else:
        bitrate = '0.5M'  # 标清
    
    bitrate = config.get('bitrate', bitrate)
    
    # 构建FFmpeg命令
    cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-c:v', codec
    ]
    
    # 添加码率和质量控制
    cmd.extend(['-b:v', bitrate, '-preset', quality, '-crf', str(crf)])
    
    # 如果有音频，添加音频相关参数 - 降低音频质量加快处理
    if video_info.get('has_audio', False):
        cmd.extend(['-c:a', 'aac', '-b:a', '64k'])
    
    # 添加输出文件名
    cmd.append(output_path)
    
    try:
        # 使用超时机制防止卡死
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate(timeout=60)  # 1分钟超时
        
        if process.returncode == 0:
            logger.info(f"已保存视频片段: {output_path}")
            return True
        else:
            logger.error(f"FFmpeg命令执行失败: {stderr.decode('utf-8', errors='ignore')}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg执行超时")
        process.kill()
        return False
    except FileNotFoundError:
        logger.error("未找到FFmpeg，请确保已安装FFmpeg并添加到系统PATH")
        return False
    except Exception as e:
        logger.error(f"提取视频片段失败: {str(e)}")
        return False


def process_single_segment(args):
    """
    处理单个视频片段（用于多进程）
    
    Args:
        args: 参数元组 (video_path, start_frame, end_frame, output_path)
        
    Returns:
        tuple: (输出路径, 是否成功)
    """
    video_path, start_frame, end_frame, output_path = args
    success = extract_video_segment_with_audio(video_path, start_frame, end_frame, output_path)
    return (output_path, success)


def process_video_segments(video_path, segments, output_dir, base_name=None, pool=None):
    """
    处理多个视频片段（使用多进程）
    
    Args:
        video_path: 输入视频路径
        segments: 片段列表，每个元素为(start_frame, end_frame)
        output_dir: 输出目录
        base_name: 输出文件名基础部分，默认使用视频文件名
        pool: 进程池
        
    Returns:
        list: 成功提取的片段路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if base_name is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # 准备任务列表
    tasks = []
    for idx, (start_frame, end_frame) in enumerate(segments, 1):
        output_filename = f"{base_name}_segment{idx}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        tasks.append((video_path, start_frame, end_frame, output_path))
    
    # 使用进程池处理任务
    results = []
    if pool:
        for output_path, success in pool.map(process_single_segment, tasks):
            if success:
                results.append(output_path)
    
    return results


def process_video_file(args):
    """
    处理单个视频文件（用于多进程）
    
    Args:
        args: 参数元组 (video_path, output_dir, config, min_frames, lock, progress_dict)
        
    Returns:
        dict: 处理结果
    """
    video_path, output_dir, config, min_frames, lock, progress_dict = args
    video_name = os.path.basename(video_path)
    result = {
        "video_path": video_path,
        "segments": 0,
        "success": False,
        "error": None
    }
    
    try:
        # 获取视频信息
        video_info = get_video_info(video_path)
        
        # 如果无法获取视频信息，标记为失败
        if not video_info:
            result["error"] = "无法获取视频信息"
            return result
        
        # 初始化场景检测器
        scene_detector = FastSceneChangeDetector(config=config)
        
        # 获取视频场景，使用固定10帧采样
        scenes = scene_detector.get_scenes(video_path, num_samples=10)
        
        # 如果没有找到场景，标记为无场景
        if not scenes:
            result["error"] = "未找到场景"
            return result
        
        # 过滤短场景
        filtered_scenes = [(start, end) for start, end in scenes if (end - start + 1) >= min_frames]
        
        # 如果没有足够长的场景，标记为无有效场景
        if not filtered_scenes:
            result["error"] = "没有足够长的场景"
            return result
        
        # 提取并保存场景片段
        output_name = os.path.splitext(video_name)[0]
        
        # 使用当前进程直接处理片段
        extracted_paths = []
        for idx, (start_frame, end_frame) in enumerate(filtered_scenes, 1):
            output_filename = f"{output_name}_segment{idx}.mp4"
            output_path = os.path.join(output_dir, output_filename)
            success = extract_video_segment_with_audio(video_path, start_frame, end_frame, output_path)
            if success:
                extracted_paths.append(output_path)
        
        result["segments"] = len(extracted_paths)
        result["success"] = True
        
        # 更新全局进度
        with lock:
            progress_dict["processed"] = progress_dict.get("processed", 0) + 1
            progress_dict["segments"] = progress_dict.get("segments", 0) + len(extracted_paths)
        
        return result
    
    except Exception as e:
        result["error"] = str(e)
        return result


def update_progress_bar(lock, progress_dict, progress_event, pbar):
    """
    更新进度条（在主进程中运行）
    
    Args:
        lock: 进程锁
        progress_dict: 共享进度字典
        progress_event: 停止事件
        pbar: 进度条对象
    """
    while not progress_event.is_set():
        with lock:
            processed = progress_dict.get("processed", 0)
            segments = progress_dict.get("segments", 0)
            total = progress_dict.get("total", 0)
        
        if total > 0:
            pbar.n = processed
            pbar.total = total
            pbar.set_postfix(已处理=f"{processed}/{total}", 提取片段=segments)
            pbar.refresh()
        
        time.sleep(0.5)


def process_video_batch(input_dir, output_dir, config, file_exts=None, min_segment_duration=None,
                      lock=None, progress_dict=None):
    """
    批量处理视频文件，提取场景片段，使用多进程
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        config: 配置参数
        file_exts: 支持的文件扩展名列表
        min_segment_duration: 最小片段时长（帧数）
        lock: 进程锁
        progress_dict: 共享进度字典
        
    Returns:
        dict: 处理结果统计
    """
    global pbar, progress_event
    
    # 使用传入的锁和进度字典
    global_lock = lock
    global_progress = progress_dict
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置最大进程数
    max_workers = config.get("max_workers", DEFAULT_PROCESS_COUNT)
    
    # 支持的视频扩展名
    file_exts = file_exts or ['*.mp4', '*.avi', '*.mkv', '*.mov', '*.flv', '*.webm']
    
    # 查找所有视频文件
    video_files = []
    for ext in file_exts:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not video_files:
        print(f"在 {input_dir} 中未找到视频文件")
        return {"total": 0, "processed": 0, "segments": 0}
    
    # 更新全局进度
    with global_lock:
        global_progress["total"] = len(video_files)
        global_progress["processed"] = 0
        global_progress["segments"] = 0
        global_progress["start_time"] = time.time()
    
    print(f"找到 {len(video_files)} 个视频文件需要处理，使用 {max_workers} 个进程")
    
    # 创建进度条
    pbar = tqdm(total=len(video_files), desc="处理视频文件", unit="视频")
    
    # 创建进度更新事件
    progress_event = multiprocessing.Event()
    
    # 启动进度更新线程
    progress_thread = threading.Thread(
        target=update_progress_bar,
        args=(global_lock, global_progress, progress_event, pbar)
    )
    progress_thread.daemon = True
    progress_thread.start()
    
    # 准备任务参数
    tasks = []
    for video_path in video_files:
        # 确保最小片段时长不低于5秒（帧数 = 秒数 * 帧率）
        min_frames = max(min_segment_duration or 0, 125)  # 默认5秒 * 25fps
        tasks.append((video_path, output_dir, config, min_frames, global_lock, global_progress))
    
    # 创建进程池处理任务
    results = []
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_video_file, tasks))
    finally:
        # 停止进度更新线程
        progress_event.set()
        progress_thread.join()
        
        # 确保进度条关闭
        pbar.close()
    
    # 统计处理结果
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    with global_lock:
        total_segments = global_progress.get("segments", 0)
        total_time = time.time() - global_progress.get("start_time", time.time())
    
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    
    print(f"\n所有视频处理完成，共处理 {len(successful)}/{len(video_files)} 个视频，"
          f"提取 {total_segments} 个片段，失败 {len(failed)} 个视频，"
          f"总用时: {total_time_str}")
    
    # 输出失败的视频
    if failed:
        print("\n处理失败的视频:")
        for r in failed:
            print(f"  {os.path.basename(r['video_path'])}: {r['error']}")
    
    # 返回统计结果
    stats = {
        "total": len(video_files),
        "processed": len(successful),
        "segments": total_segments,
        "failed": len(failed),
        "total_time": total_time
    }
    
    return stats


#######################
# 主程序              #
#######################

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='高效视频场景检测 - 127进程版本，每个视频采样10帧')
    
    # 通用参数
    parser.add_argument('--input-dir', '-i', type=str, default='/data/wuyue/output/converted',
                        help='输入视频目录路径')
    parser.add_argument('--output-dir', '-o', type=str, default='/data/wuyue/output/scene',
                        help='输出视频目录路径')
    parser.add_argument('--max-workers', '-w', type=int, default=DEFAULT_PROCESS_COUNT,
                        help='最大并行工作进程数')
    
    # 视频处理模式参数
    parser.add_argument('--min-duration', '-d', type=float, default=5.0,
                        help='最小片段时长（秒）')
    parser.add_argument('--scene-threshold', '-t', type=float, default=30.0,
                        help='场景变化阈值')
    parser.add_argument('--scene-method', '-s', type=str, choices=['histogram', 'pixel'],
                        default='histogram', help='场景检测方法')
    parser.add_argument('--min-scene-length', '-l', type=int, default=10,
                        help='最小场景长度（帧）')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 设置多进程支持
    multiprocessing.set_start_method('spawn', force=True)
    
    # 初始化全局变量
    global global_lock, global_progress, pbar, progress_event
    
    # 创建多进程安全的管理器
    manager = multiprocessing.Manager()
    global_lock = manager.Lock()
    global_progress = manager.dict()
    
    start_time = time.time()
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("运行视频场景检测 - 127进程版本，每个视频采样10帧")
    
    # 配置参数
    config = {
        "scene_change_threshold": args.scene_threshold,
        "min_scene_length": args.min_scene_length,
        "detector_method": args.scene_method,
        "max_workers": args.max_workers
    }
    
    # 处理批量视频
    stats = process_video_batch(
        args.input_dir,
        args.output_dir,
        config,
        min_segment_duration=args.min_duration * 25,  # 转换为帧数
        lock=global_lock,
        progress_dict=global_progress
    )
    
    # 打印统计信息
    total_time = time.time() - start_time
    print(f"\n处理完成！总用时: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")
    
    return 0


# 程序入口
if __name__ == "__main__":
    # 这个检查非常重要，防止在Windows/macOS上的递归问题
    # 在Linux上使用spawn模式时也是必须的
    sys.exit(main())