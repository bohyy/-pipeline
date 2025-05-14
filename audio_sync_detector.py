import cv2
import numpy as np
import logging
import subprocess
import os
import shutil
import tempfile
import random
import concurrent.futures
import multiprocessing
import time
from typing import List, Tuple, Optional
# 添加tqdm进度条库
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mouth_audio_sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MouthAudioSyncProcessor:
    """嘴型与音频同步处理器"""
    def __init__(self, config: dict = None):
        self.config = config or {}
        self.min_segment_duration = self.config.get("min_segment_duration", 1.0) # 最小片段持续时间(秒)
        self.sampling_rate = self.config.get("sampling_rate", 5) # 视频采样率(帧/秒)
        self.mouth_detection_threshold = self.config.get("mouth_detection_threshold", 0.5)
        self.speech_threshold = self.config.get("speech_threshold", 0.1)
        self.expand_duration = self.config.get("expand_duration", 0.5) # 片段扩展时间(秒)
        self.random_frames = self.config.get("random_frames", 12) # 随机抽样帧数
        self.process_without_audio = self.config.get("process_without_audio", True) # 无音频时仍处理视频
        
        # 加载嘴部检测模型
        self.mouth_cascade = self._load_mouth_cascade()
        if self.mouth_cascade.empty():
            logger.warning("嘴部检测模型加载失败，将仅依赖音频检测")
            
    def _load_mouth_cascade(self):
        """加载嘴部检测模型"""
        possible_paths = [
            cv2.data.haarcascades + 'haarcascade_mouth.xml',
            cv2.data.haarcascades + 'haarcascade_smile.xml',
            '/usr/share/opencv4/haarcascades/haarcascade_mouth.xml',
            '/usr/local/share/opencv4/haarcascades/haarcascade_mouth.xml'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    logger.info(f"成功加载嘴部检测模型: {path}")
                    return cascade
        return cv2.CascadeClassifier()
        
    def extract_audio(self, video_path: str) -> Optional[str]:
        """提取音频为临时WAV文件，有更好的错误处理"""
        try:
            fd, temp_audio_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            # 首先检查视频是否有音频流
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type',
                '-of', 'csv=p=0',
                video_path
            ]
            try:
                probe_result = subprocess.run(
                    probe_cmd,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30
                )
                # 如果视频没有音频流
                if not probe_result.stdout.strip():
                    logger.warning(f"视频没有音频流: {video_path}")
                    return None
            except Exception as e:
                logger.warning(f"检查音频流时出错: {e}")
                # 继续尝试提取，可能会失败
            
            # 尝试提取音频，使用更健壮的命令
            cmd = [
                'ffmpeg',
                '-v', 'warning', # 更详细的错误信息
                '-i', video_path,
                '-vn', # 不处理视频
                '-c:a', 'pcm_s16le',
                '-ar', '44100',
                '-ac', '1', # 强制单声道
                '-y',
                temp_audio_path
            ]
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60
            )
            if process.returncode != 0:
                error_msg = process.stderr.decode('utf-8', errors='ignore')
                logger.error(f"音频提取失败: {error_msg}")
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                return None
            
            # 验证生成的wav文件是否有效
            if os.path.getsize(temp_audio_path) < 1000: # 非常小的文件可能无效
                logger.warning(f"生成的音频文件太小，可能无效: {temp_audio_path}")
                os.remove(temp_audio_path)
                return None
                
            return temp_audio_path
        except subprocess.TimeoutExpired:
            logger.error(f"音频提取超时: {video_path}")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return None
        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return None
            
    def detect_speech_activity(self, audio_path: str) -> List[Tuple[float, float]]:
        """检测音频中的语音活动片段"""
        if not audio_path or not os.path.exists(audio_path):
            return []
        try:
            import librosa
            try:
                y, sr = librosa.load(audio_path, sr=None)
            except Exception as e:
                logger.error(f"加载音频文件失败: {e}")
                return []
            if len(y) == 0:
                logger.warning("音频文件为空")
                return []
            
            # 计算短时能量
            frame_length = int(0.02 * sr)
            hop_length = int(0.01 * sr)
            try:
                energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            except Exception as e:
                logger.error(f"计算音频能量失败: {e}")
                return []
                
            # 自适应阈值
            if len(energy) == 0:
                return []
            silence_threshold = np.percentile(energy, 10) * 3
            speech = energy > silence_threshold
            
            # 找出语音片段
            speech_segments = []
            current_segment = None
            for i, is_speech in enumerate(speech):
                time_point = i * hop_length / sr
                if is_speech:
                    if current_segment is None:
                        current_segment = [time_point, time_point]
                    else:
                        current_segment[1] = time_point
                else:
                    if current_segment is not None:
                        duration = current_segment[1] - current_segment[0]
                        if duration >= self.min_segment_duration:
                            speech_segments.append(tuple(current_segment))
                        current_segment = None
                        
            # 处理最后一个片段
            if current_segment is not None:
                duration = current_segment[1] - current_segment[0]
                if duration >= self.min_segment_duration:
                    speech_segments.append(tuple(current_segment))
                    
            return speech_segments
        except ImportError:
            logger.error("未安装librosa库，无法进行语音检测")
            return []
        except Exception as e:
            logger.error(f"语音检测失败: {e}")
            return []
            
    def detect_mouth_movement(self, video_path: str) -> List[Tuple[float, float]]:
        """使用随机采样检测视频中的嘴部运动片段"""
        if self.mouth_cascade.empty():
            logger.warning("嘴部检测模型不可用")
            return []
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return []
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0 or frame_count <= 0:
                logger.error(f"无效的帧率或帧数: fps={fps}, frames={frame_count}")
                cap.release()
                return []
                
            # 随机选择约12帧进行分析，但至少要有帧可用
            total_frames = min(self.random_frames, frame_count)
            if total_frames <= 0:
                logger.warning(f"视频没有足够的帧: {video_path}")
                cap.release()
                return []
                
            try:
                random_frames = sorted(random.sample(range(frame_count), total_frames))
            except ValueError as e:
                logger.error(f"无法生成随机帧索引: {e}")
                # 退回到等间隔采样
                if frame_count > 0:
                    step = max(1, frame_count // total_frames)
                    random_frames = list(range(0, frame_count, step))[:total_frames]
                else:
                    cap.release()
                    return []
                    
            logger.info(f"在 {frame_count} 帧中随机抽样 {len(random_frames)} 帧")
            
            mouth_frames = [] # 记录检测到嘴部的帧
            for frame_idx in random_frames:
                # 安全检查
                if frame_idx >= frame_count:
                    continue
                    
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                    
                time_point = frame_idx / fps
                
                # 检测嘴部
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # 检测人脸，然后在人脸区域检测嘴部可以提高准确性
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    has_mouth = False
                    if len(faces) > 0:
                        # 如果检测到人脸，在人脸下半部分检测嘴部
                        for (x, y, w, h) in faces:
                            # 只检查人脸下半部分
                            face_lower_half = gray[y + h//2:y + h, x:x + w]
                            if face_lower_half.size > 0: # 确保图像区域有效
                                mouths = self.mouth_cascade.detectMultiScale(
                                    face_lower_half,
                                    scaleFactor=1.1,
                                    minNeighbors=5,
                                    minSize=(20, 10),
                                    flags=cv2.CASCADE_SCALE_IMAGE
                                )
                                if len(mouths) > 0:
                                    has_mouth = True
                                    break
                    else:
                        # 如果没有检测到人脸，尝试直接检测嘴部
                        mouths = self.mouth_cascade.detectMultiScale(
                            gray,
                            scaleFactor=1.1,
                            minNeighbors=5,
                            minSize=(30, 15),
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        has_mouth = len(mouths) > 0
                        
                    if has_mouth:
                        mouth_frames.append((time_point, frame_idx))
                except Exception as e:
                    logger.error(f"处理帧 {frame_idx} 时出错: {e}")
                    continue
                    
            cap.release()
            
            # 如果检测到的嘴部帧数太少，认为没有嘴部运动
            min_required_frames = max(2, total_frames // 4)
            if len(mouth_frames) < min_required_frames:
                logger.info(f"检测到的嘴部帧数太少 ({len(mouth_frames)}/{total_frames})，认为没有嘴部运动")
                return []
                
            # 根据口型识别率，估计整个视频的口型运动片段
            mouth_coverage = len(mouth_frames) / total_frames
            if mouth_coverage > 0.6: # 如果超过60%的抽样帧有嘴部，假设整个视频有嘴部运动
                duration = frame_count / fps
                logger.info(f"检测到高比例的嘴部 ({mouth_coverage:.2f})，保留整个视频")
                return [(0.0, duration)]
            elif len(mouth_frames) > 0:
                # 聚类相邻的有口型的帧，形成片段
                mouth_frames.sort() # 按时间排序
                segments = []
                
                if len(mouth_frames) == 1:
                    # 如果只有一个检测点，创建一个以它为中心的短片段
                    time_point = mouth_frames[0][0]
                    start = max(0, time_point - self.expand_duration)
                    end = time_point + self.expand_duration
                    segments.append((start, end))
                else:
                    current_segment = [mouth_frames[0][0], mouth_frames[0][0]]
                    for time_point, _ in mouth_frames[1:]:
                        # 如果与前一个时间点差距较小，合并到当前片段
                        if time_point - current_segment[1] < self.min_segment_duration * 2:
                            current_segment[1] = time_point
                        else:
                            # 扩展片段并添加
                            expanded_start = max(0, current_segment[0] - self.expand_duration)
                            expanded_end = min(frame_count/fps, current_segment[1] + self.expand_duration)
                            segments.append((expanded_start, expanded_end))
                            current_segment = [time_point, time_point]
                            
                    # 处理最后一个片段
                    if current_segment:
                        expanded_start = max(0, current_segment[0] - self.expand_duration)
                        expanded_end = min(frame_count/fps, current_segment[1] + self.expand_duration)
                        segments.append((expanded_start, expanded_end))
                        
                logger.info(f"根据嘴部检测创建了 {len(segments)} 个片段")
                return segments
                
            return []
        except Exception as e:
            logger.error(f"嘴部运动检测失败: {e}")
            return []
            
    def check_video_has_audio(self, video_path: str) -> bool:
        """检查视频是否包含音频流"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_type',
                '-of', 'csv=p=0',
                video_path
            ]
            process = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30
            )
            return bool(process.stdout.strip())
        except Exception as e:
            logger.error(f"检查视频音频流时出错: {e}")
            return False
            
    def find_sync_segments(self, video_path: str) -> List[Tuple[float, float]]:
        """找出嘴型和声音同步的片段"""
        # 检查视频是否有音频流
        has_audio = self.check_video_has_audio(video_path)
        
        # 获取视频总时长作为备用
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return []
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            total_duration = total_frames / fps if fps > 0 and total_frames > 0 else 0
            cap.release()
            
            if total_duration <= 0:
                logger.error(f"无法获取视频时长: {video_path}")
                return []
        except Exception as e:
            logger.error(f"获取视频信息失败: {e}")
            return []
            
        # 如果没有音频，且配置允许无音频处理
        if not has_audio:
            logger.warning(f"视频没有音频流: {video_path}")
            if self.process_without_audio:
                logger.info("通过仅视觉检测继续处理")
                # 检测嘴部运动
                mouth_segments = self.detect_mouth_movement(video_path)
                if mouth_segments:
                    return mouth_segments
                else:
                    # 如果没有检测到嘴部运动，保留整个视频
                    logger.info("没有检测到嘴部运动，保留完整视频")
                    return [(0.0, total_duration)]
            else:
                logger.info("配置不允许处理无音频视频，跳过")
                return []
                
        # 有音频流，提取音频
        temp_audio_path = self.extract_audio(video_path)
        if not temp_audio_path:
            # 音频提取失败，但视频有音频流
            logger.warning(f"音频提取失败，但视频有音频流: {video_path}")
            if self.process_without_audio:
                # 检测嘴部运动
                mouth_segments = self.detect_mouth_movement(video_path)
                if mouth_segments:
                    return mouth_segments
                else:
                    # 如果没有检测到嘴部运动，保留整个视频
                    logger.info("音频处理失败且未检测到嘴部运动，保留完整视频")
                    return [(0.0, total_duration)]
            else:
                return []
                
        try:
            # 检测语音活动
            speech_segments = self.detect_speech_activity(temp_audio_path)
            
            # 检测嘴部运动
            mouth_segments = self.detect_mouth_movement(video_path)
            
            # 如果没有嘴部检测器，但有语音，使用语音片段
            if (self.mouth_cascade.empty() or not mouth_segments) and speech_segments:
                logger.info("使用语音片段作为同步片段")
                return speech_segments
                
            # 如果没有语音也没有嘴部运动，则全部保留
            if not speech_segments and not mouth_segments:
                logger.info("没有检测到语音和嘴部运动，保留完整视频")
                return [(0.0, total_duration)]
                
            # 如果有嘴部运动但没有语音，使用嘴部运动片段
            if mouth_segments and not speech_segments:
                logger.info("使用嘴部运动片段作为同步片段")
                return mouth_segments
                
            # 如果有语音但没有嘴部运动，使用语音片段
            if speech_segments and not mouth_segments:
                logger.info("使用语音片段作为同步片段")
                # 扩展语音片段
                expanded_segments = []
                for start, end in speech_segments:
                    new_start = max(0, start - self.expand_duration)
                    new_end = min(total_duration, end + self.expand_duration)
                    expanded_segments.append((new_start, new_end))
                return self._merge_segments(expanded_segments)
                
            # 找出同步片段（语音和嘴部都有）
            sync_segments = []
            
            # 扩展嘴部运动片段
            expanded_mouth_segments = []
            for start, end in mouth_segments:
                new_start = max(0, start - self.expand_duration)
                new_end = min(total_duration, end + self.expand_duration)
                expanded_mouth_segments.append((new_start, new_end))
                
            # 找出语音和嘴部运动重叠的片段
            for speech_start, speech_end in speech_segments:
                found_overlap = False
                for mouth_start, mouth_end in expanded_mouth_segments:
                    # 计算重叠时间
                    overlap_start = max(speech_start, mouth_start)
                    overlap_end = min(speech_end, mouth_end)
                    if overlap_start < overlap_end:
                        # 有重叠，保留嘴部运动片段
                        sync_segments.append((mouth_start, mouth_end))
                        found_overlap = True
                        break
                
                # 如果这个语音片段没有找到匹配的嘴部片段，也保留它
                if not found_overlap:
                    # 扩展语音片段
                    new_start = max(0, speech_start - self.expand_duration)
                    new_end = min(total_duration, speech_end + self.expand_duration)
                    sync_segments.append((new_start, new_end))
                    
            # 合并相邻或重叠的片段
            return self._merge_segments(sync_segments)
        finally:
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
                
    def _merge_segments(self, segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """合并相邻或重叠的片段"""
        if not segments:
            return []
        if len(segments) == 1:
            return segments
            
        # 按开始时间排序
        segments.sort()
        
        merged_segments = [list(segments[0])]
        for current_start, current_end in segments[1:]:
            last_start, last_end = merged_segments[-1]
            
            # 如果当前片段与上一个片段相邻或重叠
            if current_start <= last_end + self.expand_duration:
                # 合并片段
                merged_segments[-1][1] = max(last_end, current_end)
            else:
                # 添加新片段
                merged_segments.append([current_start, current_end])
                
        return [tuple(seg) for seg in merged_segments]
        
    def create_synced_video(self, video_path: str, output_path: str) -> bool:
        """创建只包含同步片段的新视频"""
        sync_segments = self.find_sync_segments(video_path)
        if not sync_segments:
            logger.warning("没有找到同步片段，跳过该视频")
            return False
            
        # 按开始时间排序
        sync_segments.sort()
        
        # 获取视频时长
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"无法打开视频: {video_path}")
            return False
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        # 如果只有一个片段且等于完整视频，直接复制
        if len(sync_segments) == 1 and abs(sync_segments[0][0]) < 0.1 and abs(sync_segments[0][1] - duration) < 0.1:
            try:
                shutil.copy2(video_path, output_path)
                logger.info(f"完整保留视频: {output_path}")
                return True
            except Exception as e:
                logger.error(f"复制视频失败: {e}")
                return False
                
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"创建输出目录失败: {e}")
                return False
                
        # 创建临时文件，处理文件名中的特殊字符
        fd, temp_filter_file = tempfile.mkstemp(suffix='.txt')
        os.close(fd)
        
        try:
            # 检查视频是否有音频流
            has_audio = self.check_video_has_audio(video_path)
            
            # 构建FFmpeg复杂滤镜
            filter_chains = []
            concat_v = []
            concat_a = []
            
            for i, (start, end) in enumerate(sync_segments):
                filter_chains.append(f"[0:v]trim=start={start}:end={end},setpts=N/FRAME_RATE/TB[v{i}];")
                concat_v.append(f"[v{i}]")
                
                if has_audio:
                    filter_chains.append(f"[0:a]atrim=start={start}:end={end},asetpts=N/SR/TB[a{i}];")
                    concat_a.append(f"[a{i}]")
                    
            # 视频部分始终连接
            filter_expr = "".join(filter_chains) + f"{''.join(concat_v)}concat=n={len(sync_segments)}:v=1:a=0[outv];"
            
            # 音频部分只在有音频时连接
            if has_audio and concat_a:
                filter_expr += f"{''.join(concat_a)}concat=n={len(sync_segments)}:v=0:a=1[outa]"
                
            # 写入滤镜表达式到临时文件
            with open(temp_filter_file, 'w') as f:
                f.write(filter_expr)
                
            # 设置ffmpeg命令
            cmd = [
                'ffmpeg',
                '-v', 'warning', # 更详细的错误信息
                '-i', video_path,
                '-filter_complex_script', temp_filter_file,
                '-map', '[outv]',
            ]
            
            # 只在有音频时添加音频映射
            if has_audio and concat_a:
                cmd.extend(['-map', '[outa]'])
                
            # 添加编码参数
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-crf', '23',
            ])
            
            # 只在有音频时添加音频编码
            if has_audio and concat_a:
                cmd.extend([
                    '-c:a', 'aac',
                    '-b:a', '128k',
                ])
                
            cmd.extend([
                '-movflags', '+faststart',
                '-y',
                output_path
            ])
            
            # 执行ffmpeg命令
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=600
            )
            
            if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                return True
            else:
                error_msg = process.stderr.decode('utf-8', errors='ignore')
                logger.error(f"视频创建失败: {error_msg}")
                # 如果输出文件已创建但可能有问题，删除它
                if os.path.exists(output_path):
                    os.remove(output_path)
                return False
                
        except Exception as e:
            logger.error(f"视频创建错误: {e}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return False
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_filter_file):
                os.remove(temp_filter_file)


def process_single_video(args):
    """处理单个视频的工作函数，用于多进程处理"""
    video_index, total_videos, input_path, output_path, config, retry_count = args
    try:
        # 不使用logger以避免进度条冲突，仅在特殊情况下打印
        # 检查输出文件是否已存在
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            # 文件已存在，跳过处理
            return input_path, True, "已存在", True  # 添加标志表示这是跳过的文件
            
        # 处理视频
        processor = MouthAudioSyncProcessor(config)
        start_time = time.time()
        success = processor.create_synced_video(input_path, output_path)
        elapsed_time = time.time() - start_time
        
        if success:
            return input_path, True, f"成功，耗时 {elapsed_time:.2f}秒", False
        else:
            # 如果失败且允许重试
            if retry_count > 0:
                # 递归重试，减少重试次数
                return process_single_video((video_index, total_videos, input_path, output_path, config, retry_count - 1))
            else:
                # 处理失败时复制原始文件（如果配置允许）
                if config.get("copy_on_failure", True):
                    try:
                        shutil.copy2(input_path, output_path)
                        return input_path, True, "处理失败，复制原始文件", False
                    except Exception as e:
                        return input_path, False, f"失败，耗时 {elapsed_time:.2f}秒", False
                return input_path, False, f"失败，耗时 {elapsed_time:.2f}秒", False
    except Exception as e:
        return input_path, False, f"错误: {e}", False


def process_directory(input_dir: str, output_dir: str, processes: int = 127, config: dict = None, max_videos: int = None):
    """使用多进程处理目录中的所有视频文件，显示总进度条并跳过重复视频"""
    if config is None:
        config = {}
        
    # 使用固定的127个进程
    logger.info(f"使用 {processes} 个进程处理视频")
    logger.info(f"配置: {config}")
    
    # 支持的视频格式
    extensions = config.get("extensions", ['.mp4', '.avi', '.mov', '.mkv', '.flv'])
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有需要处理的视频文件
    video_tasks = []
    skipped_count = 0
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                input_path = os.path.join(root, file)
                
                # 保持相对路径结构
                rel_path = os.path.relpath(root, input_dir)
                if rel_path != '.':
                    output_subdir = os.path.join(output_dir, rel_path)
                    os.makedirs(output_subdir, exist_ok=True)
                    output_path = os.path.join(output_subdir, file)
                else:
                    output_path = os.path.join(output_dir, file)
                    
                # 预先检查文件是否已存在（用于统计）
                is_skipped = os.path.exists(output_path) and os.path.getsize(output_path) > 0 and not config.get("overwrite", False)
                
                # 即使文件可能跳过，也添加到任务列表，实际跳过操作在worker中进行
                # 这样可以在进度条中包括已跳过的文件
                video_tasks.append((input_path, output_path, is_skipped))
                if is_skipped:
                    skipped_count += 1
    
    # 限制处理视频数量
    if max_videos and len(video_tasks) > max_videos:
        logger.info(f"限制处理视频数量为 {max_videos}，共找到 {len(video_tasks)} 个视频")
        video_tasks = video_tasks[:max_videos]
        
    total_videos = len(video_tasks)
    logger.info(f"发现 {total_videos} 个视频文件需要处理，其中 {skipped_count} 个可能将被跳过")
    
    if total_videos == 0:
        logger.info("没有需要处理的视频文件")
        return 0, 0
        
    # 准备任务参数
    task_args = []
    for i, (input_path, output_path, _) in enumerate(video_tasks):
        # 添加索引、总数、输入路径、输出路径、配置和重试次数
        task_args.append((i + 1, total_videos, input_path, output_path, config, config.get("retry_count", 1)))
    
    results = []
    # 使用进程池处理所有视频，并显示总进度条
    # 注意：设置进程数为127
    with multiprocessing.Pool(processes=processes) as pool:
        # 使用tqdm创建进度条
        with tqdm(total=total_videos, desc="处理视频", unit="个") as pbar:
            # 使用imap_unordered可以按完成顺序返回结果，而不是按提交顺序
            for result in pool.imap_unordered(process_single_video, task_args):
                # 更新进度条
                pbar.update(1)
                
                # 保存结果
                results.append(result)
                
                # 提取结果信息
                input_path, success, message, is_skipped = result
                
                # 如果是跳过的，在进度条右侧提示
                if is_skipped:
                    pbar.set_postfix_str(f"跳过: {os.path.basename(input_path)}")
                else:
                    status = "成功" if success else "失败"
                    pbar.set_postfix_str(f"{status}: {os.path.basename(input_path)}")
    
    # 统计处理结果
    success_count = sum(1 for _, success, _, _ in results if success)
    skipped_count = sum(1 for _, _, _, is_skipped in results if is_skipped)
    
    logger.info(f"处理完成: 成功 {success_count}/{total_videos} 个视频，其中 {skipped_count} 个是跳过的")
    
    # 生成处理报告
    try:
        report_path = os.path.join(output_dir, "processing_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"视频处理报告 - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总视频数: {total_videos}\n")
            f.write(f"成功处理: {success_count}\n")
            f.write(f"跳过文件: {skipped_count}\n")
            f.write(f"失败: {total_videos - success_count}\n\n")
            f.write("详细结果:\n")
            for path, success, message, is_skipped in results:
                status = "[跳过]" if is_skipped else "[成功]" if success else "[失败]"
                f.write(f"{status} {path} - {message}\n")
        logger.info(f"处理报告已保存至: {report_path}")
    except Exception as e:
        logger.error(f"生成处理报告失败: {e}")
        
    return success_count, total_videos

if __name__ == "__main__":
    input_dir = "/data/wuyue/output/temp_20250508_180335/4_text_filtered"
    output_dir = "/data/wuyue/output/temp_20250508_180335/5_sync_filtered"
    
    # 配置参数
    config = {
        "min_segment_duration": 1.0,  # 最小片段持续时间(秒)
        "sampling_rate": 3,  # 视频采样率(帧/秒)
        "mouth_detection_threshold": 0.5,  # 嘴部检测阈值
        "speech_threshold": 0.1,  # 语音检测阈值
        "expand_duration": 0.5,  # 片段扩展时间(秒)
        "random_frames": 10,  # 每个视频随机抽样帧数
        "process_without_audio": True,  # 无音频时是否仍处理视频
        "copy_on_failure": True,  # 处理失败时是否复制原始文件
        "overwrite": False,  # 是否覆盖已存在的输出文件
        "retry_count": 1,  # 处理失败后的重试次数
        "extensions": ['.mp4', '.avi', '.mov', '.mkv', '.flv']  # 支持的视频格式
    }
    
    # 使用127个进程进行处理
    process_directory(input_dir, output_dir, processes=127, config=config)