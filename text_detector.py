import cv2
import os
import numpy as np
import subprocess
import tempfile
from tqdm import tqdm
import logging
import traceback
import argparse
import json
import multiprocessing
import random
from functools import partial
import time
import hashlib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_processing.log')
    ]
)
logger = logging.getLogger(__name__)

def get_file_hash(file_path):
    """计算文件的MD5哈希值，用于识别重复文件。"""
    if not os.path.exists(file_path):
        return None
    
    # 仅使用前10MB数据计算哈希，提高速度
    BUF_SIZE = 10 * 1024 * 1024  
    md5 = hashlib.md5()
    
    try:
        with open(file_path, 'rb') as f:
            data = f.read(BUF_SIZE)
            md5.update(data)
        return md5.hexdigest()
    except Exception as e:
        logger.warning(f"计算文件哈希值失败: {str(e)}")
        return None

def validate_video(input_path):
    """使用 FFmpeg 检查视频文件完整性。"""
    try:
        cmd = ["ffprobe", "-v", "error", input_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0 or result.stderr:
            logger.warning(f"视频文件 {os.path.basename(input_path)} 可能损坏: {result.stderr[:200]}")
            return False
        return True
    except Exception as e:
        logger.warning(f"检查视频 {os.path.basename(input_path)} 完整性时出错: {str(e)}")
        return False

def get_video_info(input_path):
    """获取视频信息，包括时长和总帧数。"""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=duration,nb_frames,r_frame_rate",
            "-of", "json",
            input_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.warning(f"获取视频信息失败: {result.stderr[:200]}")
            return None, None, None
        
        info = json.loads(result.stdout)
        streams = info.get("streams", [])
        if not streams:
            return None, None, None
            
        stream = streams[0]
        duration = float(stream.get("duration", 0))
        
        # 解析帧率，通常是"24000/1001"这样的格式
        fps_str = stream.get("r_frame_rate", "25/1")
        if "/" in fps_str:
            num, denom = map(float, fps_str.split("/"))
            fps = num / denom
        else:
            fps = float(fps_str)
            
        nb_frames = int(stream.get("nb_frames", 0))
        if nb_frames <= 0:
            nb_frames = int(duration * fps) if duration > 0 and fps > 0 else 0
            
        return duration, fps, nb_frames
    except Exception as e:
        logger.warning(f"获取视频信息时出错: {str(e)}")
        return None, None, None

def get_keyframes(input_path):
    """使用 FFmpeg 获取视频的关键帧时间戳。"""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=pkt_pts_time,pict_type",
            "-of", "json",
            input_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.warning(f"获取关键帧失败: {result.stderr[:200]}")
            return []
        frames = json.loads(result.stdout).get("frames", [])
        keyframes = [float(frame["pkt_pts_time"]) for frame in frames if frame.get("pict_type") == "I"]
        logger.debug(f"找到 {len(keyframes)} 个关键帧")
        return keyframes
    except Exception as e:
        logger.warning(f"提取关键帧时出错: {str(e)}")
        return []

def compute_frame_difference(prev_frame, curr_frame):
    """计算两帧之间的像素差异，用于动态步长调整。"""
    if prev_frame is None or curr_frame is None:
        return float('inf')
    try:
        diff = cv2.absdiff(prev_frame, curr_frame)
        diff_mean = np.mean(diff)
        return diff_mean
    except Exception as e:
        logger.debug(f"计算帧差异时出错: {str(e)}")
        return float('inf')

def batch_process_videos(input_dir, output_dir, text_threshold=0.5, face_confidence=0.7, num_processes=None, skip_existing=True):
    """
    批量处理目录中的视频文件，检测文字遮挡人脸并剪辑（多进程模式）。
    添加skip_existing参数，控制是否跳过已处理的文件
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = [f for f in os.listdir(input_dir)
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv'))]
    
    if not video_files:
        logger.warning(f"输入目录 {input_dir} 中未找到视频文件")
        return
        
    logger.info(f"找到 {len(video_files)} 个视频文件待处理")
    
    # 定义模型路径
    text_model_path = "frozen_east_text_detection.pb"
    face_proto_path = "deploy.prototxt"
    face_model_path = "res10_300x300_ssd_iter_140000.caffemodel"
    
    # 验证模型文件是否存在
    for path in [text_model_path, face_proto_path, face_model_path]:
        if not os.path.exists(path):
            logger.error(f"模型文件未找到: {path}")
            return
    
    # 确定进程数，默认为CPU核心数-1（至少1个）
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    logger.info(f"使用 {num_processes} 个进程进行批处理")
    
    # 创建已处理文件的映射，用于跳过重复处理
    processed_files = {}
    processed_log_path = os.path.join(output_dir, "processed_files.json")
    
    # 如果启用跳过已存在选项且存在处理记录，则加载记录
    if skip_existing and os.path.exists(processed_log_path):
        try:
            with open(processed_log_path, 'r') as f:
                processed_files = json.load(f)
            logger.info(f"加载了 {len(processed_files)} 条处理记录")
        except Exception as e:
            logger.warning(f"加载处理记录失败: {str(e)}")
    
    # 准备参数
    process_args = []
    skipped_count = 0
    
    # 使用tqdm显示准备过程的进度条
    for video_file in tqdm(video_files, desc="准备处理任务"):
        input_path = os.path.join(input_dir, video_file)
        output_path = os.path.join(output_dir, video_file)
        
        # 检查文件是否已经处理过
        file_hash = get_file_hash(input_path)
        
        if skip_existing and (
            (file_hash and file_hash in processed_files) or
            (os.path.exists(output_path) and os.path.getsize(output_path) > 0)
        ):
            logger.info(f"跳过已处理文件: {video_file}")
            skipped_count += 1
            continue
        
        if not validate_video(input_path):
            logger.warning(f"跳过可能损坏的视频: {os.path.basename(input_path)}")
            continue
            
        process_args.append((
            input_path, 
            output_path, 
            text_model_path, 
            face_proto_path,
            face_model_path, 
            text_threshold, 
            face_confidence,
            file_hash
        ))
    
    logger.info(f"跳过了 {skipped_count} 个已处理文件，剩余 {len(process_args)} 个文件待处理")
    
    if not process_args:
        logger.info("没有需要处理的新文件")
        return
    
    # 创建进程池
    pool = multiprocessing.Pool(processes=num_processes)
    
    # 使用进程池处理视频
    try:
        results = list(tqdm(
            pool.imap(process_video_wrapper, process_args),
            total=len(process_args),
            desc="处理视频",
            ncols=100  # 设置进度条宽度
        ))
        
        # 更新处理记录
        new_processed = {}
        success_count = 0
        
        for result in results:
            if result is not None:
                input_path, output_path, segments_count, file_hash = result
                if file_hash:
                    new_processed[file_hash] = {
                        "input_file": os.path.basename(input_path),
                        "output_file": os.path.basename(output_path),
                        "segments": segments_count,
                        "processed_time": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    success_count += 1
        
        # 合并并保存处理记录
        if skip_existing and new_processed:
            processed_files.update(new_processed)
            try:
                with open(processed_log_path, 'w') as f:
                    json.dump(processed_files, f, indent=2)
                logger.info(f"已更新处理记录，共 {len(processed_files)} 条")
            except Exception as e:
                logger.warning(f"保存处理记录失败: {str(e)}")
        
        logger.info(f"处理完成! 成功处理 {success_count}/{len(process_args)} 个视频。")
        
    except KeyboardInterrupt:
        logger.info("用户中断，正在终止所有进程...")
        pool.terminate()
        pool.join()
        logger.info("进程已终止")
    finally:
        pool.close()
        pool.join()

def process_video_wrapper(args):
    """多进程处理视频的包装函数。"""
    try:
        input_path, output_path, text_model_path, face_proto_path, face_model_path, text_threshold, face_confidence, file_hash = args
        result = process_single_video_wrapper(
            input_path, output_path, text_model_path, face_proto_path,
            face_model_path, text_threshold, face_confidence
        )
        
        if result:
            input_path, output_path, segments_count = result
            return input_path, output_path, segments_count, file_hash
        return None
    except Exception as e:
        logger.error(f"处理视频时出错: {str(e)}")
        logger.debug(f"详细错误信息: {traceback.format_exc()}")
        return None

def process_single_video_wrapper(input_path, output_path, text_model_path, face_proto_path, face_model_path, text_threshold, face_confidence):
    """处理单个视频的包装函数，处理异常。"""
    try:
        # 为每个进程加载模型
        pid = os.getpid()
        logger.debug(f"进程 {pid} 正在加载模型...")
        
        text_net = load_text_detection_model(text_model_path)
        face_net = load_face_detection_model(face_proto_path, face_model_path)
        
        logger.debug(f"进程 {pid} 开始处理视频: {os.path.basename(input_path)}")
        
        segments_count = process_single_video(
            input_path, output_path, text_net, face_net, text_threshold, face_confidence
        )
        
        logger.info(f"进程 {pid} 成功处理 {os.path.basename(input_path)} -> {os.path.basename(output_path)} ({segments_count} 个片段)")
        return input_path, output_path, segments_count
    except Exception as e:
        logger.error(f"处理视频 {os.path.basename(input_path)} 时出错: {str(e)}")
        logger.debug(f"详细错误信息: {traceback.format_exc()}")
        return None

def process_single_video(input_path, output_path, text_net, face_net, text_threshold, face_confidence):
    """
    处理单个视频文件，使用随机抽取的12帧左右进行分析。
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # 获取视频信息
        duration, fps, total_frames = get_video_info(input_path)
        if duration is None or fps is None or total_frames is None or total_frames <= 0:
            logger.warning(f"无法获取视频信息，使用备用方法")
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise Exception(f"无法打开视频文件: {input_path}")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                logger.warning(f"无效帧率: {fps}，设为默认值25")
                fps = 25.0
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
                duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                total_frames = int(duration * fps) if duration > 0 else 0
                
            cap.release()
        
        logger.info(f"视频 {os.path.basename(input_path)}: 时长={duration:.2f}秒, 帧率={fps:.2f}, 总帧数={total_frames}")
        
        # 获取关键帧时间戳
        keyframes = get_keyframes(input_path)
        
        # 随机抽取12帧左右（优先使用关键帧，不足则随机补充）
        target_frames = 12
        keyframe_indices = set()
        
        if keyframes:
            # 从关键帧中随机选择，最多选择target_frames个
            if len(keyframes) > target_frames:
                keyframe_samples = random.sample(keyframes, target_frames)
            else:
                keyframe_samples = keyframes
                
            keyframe_indices = {max(0, min(total_frames-1, int(t * fps))) for t in keyframe_samples}
        
        # 如果关键帧不足，随机补充
        if len(keyframe_indices) < target_frames and total_frames > 0:
            additional_needed = target_frames - len(keyframe_indices)
            remaining_indices = set(range(total_frames)) - keyframe_indices
            
            if len(remaining_indices) > additional_needed:
                additional_frames = random.sample(list(remaining_indices), additional_needed)
                keyframe_indices.update(additional_frames)
            else:
                keyframe_indices.update(remaining_indices)
        
        # 确保索引有序
        frame_indices = sorted(list(keyframe_indices))
        logger.info(f"随机抽取了 {len(frame_indices)} 帧进行分析")
        
        # 打开视频进行处理
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"无法打开视频文件: {input_path}")
        
        # 分析抽取的帧
        good_frames = []
        for frame_idx in tqdm(frame_indices, desc=f"分析 {os.path.basename(input_path)}", leave=False):
            # 设置到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret or frame is None:
                logger.warning(f"无法读取帧 {frame_idx}")
                continue
                
            # 检测人脸和文本
            face_locations = detect_faces_dnn(frame, face_net, face_confidence, frame_idx, input_path)
            frame_is_good = False
            
            if face_locations:
                text_boxes = detect_text_dnn(frame, text_net, text_threshold, frame_idx, input_path)
                # 使用改进的遮挡度量标准
                frame_is_good = not check_advanced_occlusion(frame, face_locations, text_boxes)
            
            # 记录好的帧
            if frame_is_good:
                good_frames.append(frame_idx / fps)  # 转换为时间戳
        
        cap.release()
        
        # 如果没有好的帧，直接返回
        if not good_frames:
            logger.info(f"视频 {os.path.basename(input_path)} 未找到无文字遮挡的片段")
            return 0
            
        # 以好的帧为中心创建片段
        logger.info(f"找到 {len(good_frames)} 个无遮挡的帧")
        segments = []
        
        # 以无遮挡帧为中心，创建片段
        for good_time in good_frames:
            # 创建以好帧为中心的片段
            segment_duration = 3.0  # 每个片段3秒
            start_time = max(0, good_time - segment_duration/2)
            end_time = min(duration, good_time + segment_duration/2)
            
            if end_time - start_time >= 1.0:  # 最小片段长度为1秒
                segments.append([start_time, end_time])
        
        # 合并重叠的片段
        segments = merge_close_segments(segments, max_gap=0.5)
        
        # 过滤太短的片段
        segments = [s for s in segments if s[1] - s[0] >= 1.0]
        
        # 提取并合并片段
        if not segments:
            logger.info(f"视频 {os.path.basename(input_path)} 合并后未找到合适的片段")
            return 0
            
        extract_and_merge_segments(input_path, output_path, segments, temp_dir)
        return len(segments)
    except Exception as e:
        logger.error(f"处理视频 {os.path.basename(input_path)} 时出错: {str(e)}")
        raise
    finally:
        cleanup_temp_dir(temp_dir)

def check_advanced_occlusion(frame, face_locations, text_boxes):
    """
    改进的遮挡检测算法，考虑多种因素：
    1. 重叠面积比例
    2. 人脸区域的重要性权重
    3. 文本与肤色的对比度
    """
    if not text_boxes or not face_locations:
        return False
        
    frame_height, frame_width = frame.shape[:2]
    
    for face_idx, (top, right, bottom, left) in enumerate(face_locations):
        face_width = right - left
        face_height = bottom - top
        face_area = face_width * face_height
        
        if face_area <= 0:
            continue
            
        # 创建面部重要性权重图 - 眼睛和鼻子区域权重高
        face_weights = np.ones((face_height, face_width))
        
        # 估计眼睛区域（上部1/3区域权重更高）
        eye_region_h = int(face_height * 0.33)
        face_weights[0:eye_region_h, :] = 2.5
        
        # 估计中部区域（鼻子）
        nose_region_top = int(face_height * 0.33)
        nose_region_bottom = int(face_height * 0.66)
        nose_region_left = int(face_width * 0.3)
        nose_region_right = int(face_width * 0.7)
        face_weights[nose_region_top:nose_region_bottom, nose_region_left:nose_region_right] = 2.0
        
        # 检查每个文本框
        for (tx, ty, tw, th) in text_boxes:
            # 计算重叠区域
            x_overlap = max(0, min(left + face_width, tx + tw) - max(left, tx))
            y_overlap = max(0, min(top + face_height, ty + th) - max(top, ty))
            
            if x_overlap <= 0 or y_overlap <= 0:
                continue
                
            # 获取重叠区域在人脸局部坐标系中的位置
            face_x1 = max(0, tx - left)
            face_y1 = max(0, ty - top)
            face_x2 = min(face_width, face_x1 + x_overlap)
            face_y2 = min(face_height, face_y1 + y_overlap)
            
            # 计算加权重叠比例
            overlap_weights = face_weights[face_y1:face_y2, face_x1:face_x2]
            weighted_overlap_area = np.sum(overlap_weights) * (x_overlap * y_overlap) / np.sum(face_weights)
            
            # 计算文本区域与肤色的对比度
            try:
                # 获取重叠区域在原图中的坐标
                img_x1 = max(0, left + face_x1)
                img_y1 = max(0, top + face_y1)
                img_x2 = min(frame_width-1, img_x1 + x_overlap)
                img_y2 = min(frame_height-1, img_y1 + y_overlap)
                
                if img_x2 <= img_x1 or img_y2 <= img_y1:
                    continue
                    
                overlap_region = frame[img_y1:img_y2, img_x1:img_x2]
                
                # 将区域转为灰度图
                if len(overlap_region.shape) == 3:
                    overlap_gray = cv2.cvtColor(overlap_region, cv2.COLOR_BGR2GRAY)
                else:
                    overlap_gray = overlap_region
                    
                # 计算局部方差作为纹理复杂度度量
                _, stddev = cv2.meanStdDev(overlap_gray)
                contrast_factor = min(1.0, stddev[0][0] / 50.0)  # 归一化对比度
                
                # 综合考虑遮挡度
                weighted_overlap_ratio = weighted_overlap_area / face_area
                occlusion_score = weighted_overlap_ratio * (0.5 + 0.5 * contrast_factor)
                
                if occlusion_score > 0.08:  # 经验值调整
                    logger.debug(f"人脸 {face_idx+1} 被遮挡, 遮挡度={occlusion_score:.4f}")
                    return True
                    
            except Exception as e:
                logger.debug(f"计算对比度时出错: {str(e)}")
                # 如果对比度计算失败，仅使用面积比
                if x_overlap * y_overlap > 0.1 * face_area:
                    return True
                    
    return False

def merge_close_segments(segments, max_gap=0.5):
    """合并时间间隔较小的片段。"""
    if not segments:
        return []
        
    sorted_segments = sorted(segments, key=lambda x: x[0])
    merged = [sorted_segments[0]]
    
    for current in sorted_segments[1:]:
        previous = merged[-1]
        if current[0] - previous[1] <= max_gap:
            merged[-1] = [previous[0], max(previous[1], current[1])]
        else:
            merged.append(current)
            
    return merged

def extract_and_merge_segments(input_path, output_path, segments, temp_dir):
    """提取并合并视频片段。"""
    try:
        concat_file = os.path.join(temp_dir, "concat.txt")
        with open(concat_file, "w") as f:
            for i, (start, end) in enumerate(segments):
                temp_clip = os.path.join(temp_dir, f"clip_{i}.mp4")
                cmd = [
                    "ffmpeg",
                    "-i", input_path,
                    "-ss", str(start),
                    "-to", str(end),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-c:a", "aac",
                    "-avoid_negative_ts", "make_zero",
                    "-y",
                    temp_clip
                ]
                result = subprocess.run(
                    cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                if result.returncode != 0 or not os.path.exists(temp_clip) or os.path.getsize(temp_clip) == 0:
                    logger.warning(f"提取片段 {i+1}/{len(segments)} 失败，视频: {os.path.basename(input_path)}: {result.stderr[:200]}...")
                    continue
                f.write(f"file '{temp_clip}'\n")
                
        with open(concat_file, "r") as f:
            concat_content = f.read().strip()
            
        if not concat_content:
            logger.warning(f"未提取到有效片段，视频: {os.path.basename(input_path)}，跳过合并")
            return
            
        merge_cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            "-y",
            output_path
        ]
        merge_result = subprocess.run(
            merge_cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if merge_result.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            logger.error(f"合并失败，视频: {os.path.basename(input_path)}: {merge_result.stderr[:500]}")
            raise Exception("FFmpeg 合并失败")
            
        logger.info(f"成功合并 {len(segments)} 个片段，从 {os.path.basename(input_path)} 到 {os.path.basename(output_path)}")
    except Exception as e:
        logger.error(f"处理片段时出错，视频: {os.path.basename(input_path)}: {str(e)}")
        raise

def cleanup_temp_dir(temp_dir):
    """清理临时目录。"""
    try:
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        os.rmdir(temp_dir)
    except Exception as e:
        logger.warning(f"清理临时目录时出错: {str(e)}")

def detect_faces_dnn(frame, net, confidence_threshold=0.7, frame_count=0, video_path="unknown"):
    """使用 OpenCV DNN 检测人脸。"""
    if frame is None or frame.size == 0 or frame.shape[0] <= 0 or frame.shape[1] <= 0 or not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        logger.warning(f"无效帧，视频: {os.path.basename(video_path)}，帧号: {frame_count}")
        return []
        
    try:
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
            
        if not np.any(frame) or np.isnan(frame).any() or np.isinf(frame).any():
            logger.warning(f"无效像素值，视频: {os.path.basename(video_path)}，帧号: {frame_count}")
            return []
            
        (h, w) = frame.shape[:2]
        input_size = (300, 300)
        resized_frame = cv2.resize(frame, input_size)
        blob = cv2.dnn.blobFromImage(
            resized_frame,
            scalefactor=1.0,
            size=input_size,
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )
        
        net.setInput(blob)
        detections = net.forward()
        
        face_locations = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w - 1, endX), min(h - 1, endY)
                face_locations.append((startY, endX, endY, startX))
                
        return face_locations
    except Exception as e:
        logger.warning(f"人脸检测出错，视频: {os.path.basename(video_path)}，帧号: {frame_count}，错误: {str(e)}")
        return []

def detect_text_dnn(frame, net, confidence_threshold=0.5, frame_count=0, video_path="unknown"):
    """使用 EAST 模型检测文本区域。"""
    if frame is None or frame.size == 0 or frame.shape[0] <= 0 or frame.shape[1] <= 0 or not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        logger.warning(f"无效帧，视频: {os.path.basename(video_path)}，帧号: {frame_count}")
        return []
        
    try:
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)
            
        if not np.any(frame) or np.isnan(frame).any() or np.isinf(frame).any():
            logger.warning(f"无效像素值，视频: {os.path.basename(video_path)}，帧号: {frame_count}")
            return []
            
        (orig_h, orig_w) = frame.shape[:2]
        scale = 1.0
        new_h, new_w = orig_h, orig_w
        
        if orig_h > 720 or orig_w > 1280:
            scale = min(720 / orig_h, 1280 / orig_w)
            new_h, new_w = int(orig_h * scale), int(orig_w * scale)
            new_h = max(1, new_h)
            new_w = max(1, new_w)
            frame = cv2.resize(frame, (new_w, new_h))
            
        east_input_size = (320, 320)
        layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
        
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1.0,
            size=east_input_size,
            mean=(123.68, 116.78, 103.94),
            swapRB=True,
            crop=False
        )
        
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        (rects, confidences) = decode_predictions(scores, geometry, confidence_threshold)
        
        text_boxes = []
        if len(rects) > 0:
            indices = cv2.dnn.NMSBoxes(rects, confidences, confidence_threshold, 0.4)
            if isinstance(indices, np.ndarray) and indices.size > 0:
                for i in indices.flatten():
                    (x, y, w, h) = rects[i]
                    if scale != 1.0:
                        x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
                    text_boxes.append((x, y, w, h))
            elif isinstance(indices, list) and len(indices) > 0:
                for i in indices:
                    (x, y, w, h) = rects[i]
                    if scale != 1.0:
                        x, y, w, h = int(x / scale), int(y / scale), int(w / scale), int(h / scale)
                    text_boxes.append((x, y, w, h))
                    
        return text_boxes
    except Exception as e:
        logger.warning(f"文本检测出错，视频: {os.path.basename(video_path)}，帧号: {frame_count}，错误: {str(e)}")
        return []

def decode_predictions(scores, geometry, confidence_threshold):
    """解码 EAST 模型输出。"""
    (numRows, numCols) = scores.shape[2:4]
    rects, confidences = [], []
    
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0, xData1, xData2, xData3 = geometry[0, 0, y], geometry[0, 1, y], geometry[0, 2, y], geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
        
        for x in range(numCols):
            if scoresData[x] < confidence_threshold:
                continue
                
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            angle = anglesData[x]
            cos, sin = np.cos(angle), np.sin(angle)
            
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            rects.append((startX, startY, int(w), int(h)))
            confidences.append(scoresData[x])
            
    return rects, confidences

def load_text_detection_model(model_path):
    """加载 EAST 文本检测模型，支持 GPU。"""
    if not os.path.exists(model_path):
        raise Exception(f"EAST 模型未找到: {model_path}")
        
    net = cv2.dnn.readNet(model_path)
    
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        logger.info("使用 CUDA 后端进行文本检测")
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("使用 CPU 后端进行文本检测")
        
    return net

def load_face_detection_model(proto_path, model_path):
    """加载 OpenCV 人脸检测模型，支持 GPU。"""
    if not os.path.exists(proto_path) or not os.path.exists(model_path):
        raise Exception(f"人脸模型文件缺失: {proto_path}, {model_path}")
        
    net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
    
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        logger.info("使用 CUDA 后端进行人脸检测")
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        logger.info("使用 CPU 后端进行人脸检测")
        
    return net

if __name__ == "__main__":
    # 设置输入和输出路径
    input_dir = "/data/wuyue/output/temp_20250510_134344/3_face_filtered"
    output_dir = "/data/wuyue/output/temp_20250510_134344/4_text_filtered"
    
    parser = argparse.ArgumentParser(description='批量处理视频，移除文字遮挡人脸的片段')
    parser.add_argument('--input', type=str, default=input_dir, help='输入视频目录')
    parser.add_argument('--output', type=str, default=output_dir, help='输出视频目录')
    parser.add_argument('--text-threshold', type=float, default=0.5, help='文字检测置信度阈值 (0-1)')
    parser.add_argument('--face-confidence', type=float, default=0.7, help='人脸检测置信度阈值 (0-1)')
    parser.add_argument('--processes', type=int, default=None, help='并行处理的进程数，默认为CPU核心数-1')
    parser.add_argument('--no-skip', action='store_true', help='设置此标志以禁用跳过已处理文件')
    args = parser.parse_args()
    
    logger.info(f"开始处理视频...\n输入目录: {args.input}\n输出目录: {args.output}")
    
    start_time = time.time()
    batch_process_videos(
        args.input,
        args.output,
        text_threshold=args.text_threshold,
        face_confidence=args.face_confidence,
        num_processes=args.processes,
        skip_existing=not args.no_skip  # 默认开启跳过功能
    )
    
    elapsed = time.time() - start_time
    logger.info(f"所有处理完成！总耗时: {elapsed:.2f} 秒")