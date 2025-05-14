import os
import cv2
import subprocess
import json
import random
import numpy as np
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from tqdm import tqdm  # 导入tqdm用于显示进度条
import multiprocessing  # Add this import

# Global variables to store settings (needed for the standalone function)
GLOBAL_SETTINGS = {
    'ffmpeg_path': "ffmpeg",
    'min_face_duration': 1.0,
    'min_video_duration': 5.0,
    'samples_per_video': 10,
    'max_retries': 3,
    'output_dir': "",
    'processed_videos': {}
}

# Create a standalone FaceDetector class (no self reference issues)
class FaceDetector:
    """线程安全的人脸检测器类"""

    def __init__(self):
        """初始化人脸检测器，每个线程应该有自己的实例"""
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            raise ValueError("无法加载Haar级联分类器，请检查OpenCV安装")

    def detect_faces(self, frame):
        """检测图像中的人脸"""
        if frame is None:
            return []
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            return faces
        except Exception as e:
            print(f"检测人脸时出错: {str(e)}")
            return []

# Standalone functions for multiprocessing
def detect_face_segments_sampled(cap, fps, total_frames, samples_per_video):
    """通过随机采样检测视频中的人脸片段"""
    try:
        # Create a detector instance for this process
        detector = FaceDetector()
        
        # 安全检查
        if total_frames <= 0 or fps <= 0:
            return []
            
        # 计算采样间隔，确保样本分布均匀
        actual_samples = min(samples_per_video, total_frames)
        if total_frames <= actual_samples:
            # 如果总帧数少于或等于采样数，则检查所有帧
            sample_frames = list(range(int(total_frames)))
        else:
            # 生成均匀分布的样本帧
            interval = total_frames / actual_samples
            sample_frames = [int(min(i * interval + random.uniform(0, interval-1), total_frames-1)) 
                            for i in range(actual_samples)]
            sample_frames = sorted(list(set(sample_frames)))  # 去重并排序
        
        # 保存每个采样帧的人脸检测结果
        frame_has_face = {}
        
        for frame_idx in sample_frames:
            try:
                # 安全设置帧位置
                if frame_idx >= total_frames:
                    continue
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue
                
                # 进行人脸检测前先检查帧是否有效    
                if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                    continue
                    
                # 检测人脸
                faces = detector.detect_faces(frame)
                frame_has_face[frame_idx] = len(faces) > 0
            except Exception as e:
                print(f"处理帧 {frame_idx} 时出错: {str(e)}")
                continue
        
        # 将相邻的具有人脸的帧合并为片段
        face_frames = sorted([idx for idx, has_face in frame_has_face.items() if has_face])
        
        if not face_frames:
            return []
            
        # 基于采样密度估计片段
        segments = []
        current_segment = None
        
        # 定义合理的阈值来合并片段
        merge_threshold = max(int(total_frames / (actual_samples * 1.5)), int(fps * 2))
        
        for frame_idx in face_frames:
            if current_segment is None:
                current_segment = {
                    'start': frame_idx,
                    'end': frame_idx,
                    'start_time': frame_idx / fps,
                    'end_time': frame_idx / fps
                }
            # 如果与上一帧的间隔相对较小，将其视为同一片段
            elif frame_idx - current_segment['end'] < merge_threshold:
                current_segment['end'] = frame_idx
                current_segment['end_time'] = frame_idx / fps
            else:
                segments.append(current_segment)
                current_segment = {
                    'start': frame_idx,
                    'end': frame_idx,
                    'start_time': frame_idx / fps,
                    'end_time': frame_idx / fps
                }
        
        if current_segment is not None:
            segments.append(current_segment)
            
        # 确保至少有一个片段
        if not segments and face_frames:
            # 如果没有合并成片段但确实有人脸帧，就用第一个人脸帧创建一个片段
            first_face_frame = face_frames[0]
            segments.append({
                'start': first_face_frame,
                'end': first_face_frame,
                'start_time': first_face_frame / fps,
                'end_time': first_face_frame / fps
            })
            
        # 扩展片段的时间窗口，确保最小持续时间
        min_face_duration = GLOBAL_SETTINGS['min_face_duration']
        min_face_frames = max(int(fps * min_face_duration), 1)
        
        for segment in segments:
            # 向前扩展
            expand_before = min(int(fps), segment['start'])
            segment['start'] = max(0, segment['start'] - expand_before)
            segment['start_time'] = max(0, segment['start'] / fps)
            
            # 向后扩展
            expand_after = min(int(fps), total_frames - 1 - segment['end'])
            segment['end'] = min(total_frames - 1, segment['end'] + expand_after)
            segment['end_time'] = min(total_frames - 1, segment['end']) / fps
            
            # 确保片段长度满足最小持续时间
            duration_frames = segment['end'] - segment['start']
            if duration_frames < min_face_frames:
                # 尝试扩展到最小持续时间
                extend_frames = min_face_frames - duration_frames
                # 先向后扩展
                extend_after = min(extend_frames, total_frames - 1 - segment['end'])
                segment['end'] = min(total_frames - 1, segment['end'] + extend_after)
                # 然后向前扩展剩余的帧数
                extend_frames -= extend_after
                if extend_frames > 0:
                    extend_before = min(extend_frames, segment['start'])
                    segment['start'] = max(0, segment['start'] - extend_before)
                
                # 更新时间
                segment['start_time'] = segment['start'] / fps
                segment['end_time'] = segment['end'] / fps
        
        # 合并重叠的片段
        if len(segments) > 1:
            segments.sort(key=lambda x: x['start'])
            merged_segments = []
            current = segments[0]
            
            for next_seg in segments[1:]:
                if next_seg['start'] <= current['end']:
                    # 合并重叠片段
                    current['end'] = max(current['end'], next_seg['end'])
                    current['end_time'] = current['end'] / fps
                else:
                    merged_segments.append(current)
                    current = next_seg
            
            merged_segments.append(current)
            segments = merged_segments
            
        return segments
        
    except Exception as e:
        print(f"检测人脸片段时出错: {str(e)}")
        print(traceback.format_exc())
        return []

def calculate_total_duration(segments):
    """计算所有片段的总持续时间"""
    total_duration = 0
    for segment in segments:
        segment_duration = segment['end_time'] - segment['start_time']
        total_duration += segment_duration
    return total_duration

def extend_segments_to_min_duration(segments, total_frames, fps):
    """尝试扩展片段以达到最小视频时长"""
    if not segments:
        return segments
        
    # 计算需要的扩展时间（秒）
    current_duration = calculate_total_duration(segments)
    min_video_duration = GLOBAL_SETTINGS['min_video_duration']
    needed_extension = min_video_duration - current_duration
    
    if needed_extension <= 0:
        return segments
        
    # 将总视频帧数转换为持续时间
    total_video_duration = total_frames / fps if fps > 0 else 0
    
    # 如果视频总长度不足以扩展到最小时长，返回原片段
    if total_video_duration < min_video_duration:
        return segments
        
    # 按段落长度比例扩展各段
    # 1. 先给每个段落计算当前长度
    for segment in segments:
        segment['duration'] = segment['end_time'] - segment['start_time']
        
    # 2. 计算总当前长度和扩展比例
    total_current_duration = sum(segment['duration'] for segment in segments)
    
    # 3. 根据比例分配扩展时间
    extended_segments = []
    for segment in segments:
        # 计算这个段落应该分配的扩展时间（按比例）
        if total_current_duration > 0:
            segment_extension = needed_extension * (segment['duration'] / total_current_duration)
        else:
            segment_extension = needed_extension / len(segments)
            
        # 分配一半到开始时间，一半到结束时间
        extend_before = segment_extension / 2
        extend_after = segment_extension / 2
        
        # 向前扩展 (但不超过0)
        new_start_time = max(0, segment['start_time'] - extend_before)
        new_start = max(0, int(new_start_time * fps))
        
        # 向后扩展 (但不超过总帧数)
        new_end_time = min(total_video_duration, segment['end_time'] + extend_after)
        new_end = min(total_frames - 1, int(new_end_time * fps))
        
        # 创建新的扩展段落
        extended_segment = {
            'start': new_start,
            'end': new_end,
            'start_time': new_start / fps,
            'end_time': new_end / fps
        }
        
        extended_segments.append(extended_segment)
        
    # 合并重叠的段落
    if len(extended_segments) > 1:
        extended_segments.sort(key=lambda x: x['start'])
        merged_segments = []
        current = extended_segments[0]
        
        for next_seg in extended_segments[1:]:
            if next_seg['start'] <= current['end']:
                # 合并重叠段落
                current['end'] = max(current['end'], next_seg['end'])
                current['end_time'] = current['end'] / fps
            else:
                merged_segments.append(current)
                current = next_seg
        
        merged_segments.append(current)
        extended_segments = merged_segments
        
    return extended_segments

def build_ffmpeg_command(input_path, output_path, segments):
    """构建FFmpeg命令"""
    if not segments:
        return None

    # 生成选择滤镜
    select_filters = []
    for i, seg in enumerate(segments):
        select_filters.append(
            f"between(t,{seg['start_time']:.3f},{seg['end_time']:.3f})")

    # 构建完整命令
    cmd = [
        GLOBAL_SETTINGS['ffmpeg_path'],
        '-i', input_path,
        '-vf', f"select='{'+'.join(select_filters)}',setpts=N/FRAME_RATE/TB",
        '-af', f"aselect='{'+'.join(select_filters)}',asetpts=N/SR/TB",
        '-c:v', 'libx264',
        '-preset', 'fast',  # 使用快速编码预设
        '-crf', '23',       # 压缩质量，平衡大小和质量
        '-c:a', 'aac',
        '-b:a', '128k',     # 音频比特率
        '-y',               # 覆盖输出文件
        output_path
    ]

    return cmd

# The main processing function for a single video
def process_single_video(video_path):
    """处理单个视频文件，可用于多进程"""
    try:
        video_basename = os.path.basename(video_path)
        print(f"开始处理视频: {video_basename}")
        
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f"警告: 视频 {video_basename} FPS无效 ({fps}), 使用默认值 25")
            fps = 25.0
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = total_frames / fps if fps > 0 else 0
        
        if total_frames <= 0:
            print(f"警告: 视频 {video_basename} 帧数无效 ({total_frames}), 跳过处理")
            cap.release()
            return None

        # 检测人脸片段
        face_segments = detect_face_segments_sampled(cap, fps, total_frames, GLOBAL_SETTINGS['samples_per_video'])
        cap.release()

        if not face_segments:
            print(f"警告: {video_basename} 中没有检测到任何人脸")
            return None
            
        # 计算所有片段的总持续时间
        total_segments_duration = calculate_total_duration(face_segments)
        
        # 如果总持续时间少于最小要求，尝试扩展片段
        if total_segments_duration < GLOBAL_SETTINGS['min_video_duration']:
            face_segments = extend_segments_to_min_duration(face_segments, total_frames, fps)
            total_segments_duration = calculate_total_duration(face_segments)
            
            # 如果扩展后仍不足最小时长，则舍弃此视频
            if total_segments_duration < GLOBAL_SETTINGS['min_video_duration']:
                print(f"舍弃视频 {video_basename}: 人脸片段总时长 ({total_segments_duration:.2f}s) 少于最小要求 ({GLOBAL_SETTINGS['min_video_duration']}s)")
                return None
        
        # 生成输出视频路径
        video_name = os.path.splitext(video_basename)[0]
        # 文件名可能过长，进行截断处理
        if len(video_name) > 100:
            video_name = video_name[:100]
        output_path = os.path.join(GLOBAL_SETTINGS['output_dir'], f"{video_name}_trimmed.mp4")

        # 构建FFmpeg命令
        ffmpeg_cmd = build_ffmpeg_command(video_path, output_path, face_segments)

        # 执行FFmpeg命令
        try:
            result = subprocess.run(
                ffmpeg_cmd, 
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            print(f"成功生成修剪后的视频: {os.path.basename(output_path)} (时长: {total_segments_duration:.2f}s)")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg处理失败: {e}")
            print(f"FFmpeg错误输出: {e.stderr}")
            return None
            
    except Exception as e:
        print(f"处理视频 {os.path.basename(video_path)} 时出错: {str(e)}")
        print(traceback.format_exc())  # 打印完整堆栈跟踪
        return None

# Function to retry processing a video
def process_video_with_retry(video_path):
    """带重试机制的视频处理"""
    video_basename = os.path.basename(video_path)
    
    for attempt in range(GLOBAL_SETTINGS['max_retries']):
        try:
            result = process_single_video(video_path)
            return result
                
        except Exception as e:
            print(f"处理视频 {video_basename} 失败 (尝试 {attempt+1}/{GLOBAL_SETTINGS['max_retries']}): {str(e)}")
            if attempt < GLOBAL_SETTINGS['max_retries'] - 1:
                time.sleep(1)  # 短暂延迟后重试
            else:
                print(f"放弃处理视频 {video_basename} 经过 {GLOBAL_SETTINGS['max_retries']} 次尝试")
                return None

# Function to process all videos using multiprocessing
def process_all_videos(input_dir, output_dir, num_workers=127, samples_per_video=10, 
                      min_face_duration=1.0, min_video_duration=5.0, 
                      ffmpeg_path="ffmpeg", max_retries=3, skip_existing=True):
    """使用多进程处理所有视频"""
    # Setup global settings for the worker processes
    global GLOBAL_SETTINGS
    GLOBAL_SETTINGS.update({
        'output_dir': output_dir,
        'min_face_duration': min_face_duration,
        'min_video_duration': min_video_duration,
        'samples_per_video': samples_per_video,
        'ffmpeg_path': ffmpeg_path,
        'max_retries': max_retries
    })
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load processed videos record
    processed_log = os.path.join(output_dir, "processed_videos.json")
    processed_videos = {}
    if os.path.exists(processed_log) and skip_existing:
        try:
            with open(processed_log, 'r') as f:
                processed_videos = json.load(f)
        except Exception as e:
            print(f"加载处理记录失败: {str(e)}，将创建新记录")
    
    GLOBAL_SETTINGS['processed_videos'] = processed_videos
    
    # Collect all video files to process
    video_files = []
    skipped_files = []
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(input_dir, filename)
            
            # Check if already processed
            video_basename = os.path.basename(video_path)
            output_basename = os.path.splitext(video_basename)[0]
            if len(output_basename) > 100:
                output_basename = output_basename[:100]
            output_filename = f"{output_basename}_trimmed.mp4"
            output_path = os.path.join(output_dir, output_filename)
            
            # Skip if already processed and exists
            if (skip_existing and 
                os.path.exists(output_path) and 
                video_basename in processed_videos):
                skipped_files.append(video_path)
                continue
                
            video_files.append(video_path)
    
    total_videos = len(video_files)
    
    if skipped_files:
        print(f"跳过 {len(skipped_files)} 个已处理的视频文件")
        
    if not video_files:
        print(f"没有新的视频文件需要处理")
        return []
        
    print(f"找到 {total_videos} 个需要处理的视频文件，使用 {num_workers} 个进程处理")
    
    # Process videos with ProcessPoolExecutor
    results = []
    
    # Set up a progress bar for total progress
    with tqdm(total=total_videos, desc="处理进度", unit="视频") as progress_bar:
        # Using a process pool for parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(process_video_with_retry, video_path) for video_path in video_files]
            
            # Process results as they complete
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Update the progress bar in the main process
                    progress_bar.update(1)
                    
                    # Update processed videos record
                    if result:
                        video_basename = os.path.basename(next((vp for vp in video_files if os.path.basename(result).startswith(os.path.splitext(os.path.basename(vp))[0][:100])), "unknown"))
                        processed_videos[video_basename] = {
                            "output_path": result,
                            "processed_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                except Exception as e:
                    print(f"处理视频时发生未捕获的异常: {str(e)}")
                    results.append(None)
                    progress_bar.update(1)
    
    # Save the processed videos record
    try:
        with open(processed_log, 'w') as f:
            json.dump(processed_videos, f, indent=2)
    except Exception as e:
        print(f"保存处理记录失败: {str(e)}")
    
    successful = sum(1 for r in results if r is not None)
    print(f"处理完成: {successful}/{total_videos} 个视频成功处理，{len(skipped_files)} 个视频已跳过")
    
    return results

# Main execution
if __name__ == "__main__":
    # 指定的输入和输出路径
    input_dir = "/data/wuyue/output/converted"
    output_dir = "/data/wuyue/output/ocr"
    
    # 使用新的多进程函数
    process_all_videos(
        input_dir=input_dir, 
        output_dir=output_dir,
        samples_per_video=10,      # 每个视频抽取10帧
        min_face_duration=1.0,     # 最小人脸持续时间
        min_video_duration=5.0,    # 最小输出视频长度(秒)
        num_workers=127,           # 使用127个进程
        max_retries=3,             # 添加重试机制
        skip_existing=True         # 跳过已处理的视频
    )