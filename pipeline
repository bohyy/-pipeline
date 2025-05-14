#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频处理流水线
整合了五个处理模块：
1. 格式转换
2. 场景检测
3. 人脸检测
4. 文字遮挡检测
5. 语音同步检测

支持多线程处理、错误恢复和断点续传
中间结果保存在临时目录，最终结果直接输出到指定目录
"""

import os
import sys
import json
import time
import logging
import multiprocessing
import concurrent.futures
import threading
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import shutil
import traceback

# 将处理模块导入到系统路径
PROJECT_DIR = "/data/wuyue/project"
sys.path.insert(0, PROJECT_DIR)

# 导入各个处理模块
import convert_to_h264_aac_serial
import scene_change_detector
import face_detector
import text_detector
import audio_sync_detector

# 配置日志
class TqdmLoggingHandler(logging.Handler):
    """自定义日志处理器，避免与tqdm进度条冲突"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

# 设置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        TqdmLoggingHandler(),
        logging.FileHandler('video_pipeline.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class VideoPipeline:
    """视频处理流水线类"""
    
    def __init__(self, input_dir: str, output_dir: str, config: Dict = None):
        """
        初始化流水线
        
        Args:
            input_dir: 输入视频目录
            output_dir: 最终输出视频目录  
            config: 配置参数
        """
        self.input_dir = input_dir
        self.final_output_dir = output_dir  # 最终输出目录
        self.config = config or {}
        
        # 设置临时处理目录（在/data/wuyue/output下创建带时间戳的临时目录）
        timestamp = datetime.now().strftime("temp_%Y%m%d_%H%M%S")
        self.temp_dir = os.path.join("/data/wuyue/output", timestamp)
        
        # 创建各阶段的临时目录
        self.stage_dirs = {
            "1_converted": os.path.join(self.temp_dir, "1_converted"),
            "2_scene_filtered": os.path.join(self.temp_dir, "2_scene_filtered"),
            "3_face_filtered": os.path.join(self.temp_dir, "3_face_filtered"),
            "4_text_filtered": os.path.join(self.temp_dir, "4_text_filtered"),
            "5_sync_filtered": os.path.join(self.temp_dir, "5_sync_filtered")  # 临时目录，最后会移动文件到final_output_dir
        }
        
        # 创建所有必要的目录
        for dir_path in self.stage_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        # 确保最终输出目录存在
        os.makedirs(self.final_output_dir, exist_ok=True)
            
        # 状态文件路径
        self.state_file = os.path.join(self.temp_dir, "pipeline_state.json")
        
        # 加载或初始化处理状态
        self.state = self._load_state()
        
        # 配置各模块参数
        self._configure_modules()
        
        # 进度统计
        self.progress_stats = {
            "total_files": 0,
            "processed_files": 0,
            "current_stage": "",
            "stage_progress": {}
        }
        
        # 全局锁和进度字典，用于进程间同步
        self.manager = multiprocessing.Manager()
        self.global_lock = self.manager.Lock()
        self.global_progress = self.manager.dict()
        
        logger.info(f"流水线初始化完成")
        logger.info(f"输入目录: {self.input_dir}")
        logger.info(f"临时目录: {self.temp_dir}")
        logger.info(f"最终输出目录: {self.final_output_dir}")
        
    def _configure_modules(self):
        """配置各处理模块的参数"""
        
        # 1. 格式转换参数
        self.convert_config = self.config.get("convert", {
            "codec": "libx264",
            "quality": "fast",  # 使用fast预设
            "crf": 23,
            "audio": "128k",
            "workers": 127,     # 使用127个进程
            "extensions": ["mp4", "MP4", "mov", "avi", "mkv", "flv", "wmv"]
        })
        
        # 2. 场景检测参数
        self.scene_config = self.config.get("scene", {
            "scene_change_threshold": 30.0,
            "min_scene_length": 10,
            "detector_method": "histogram",
            "max_workers": 127,
            "min_segment_duration": 3.0,  # 最小片段时长（秒）
        })
        
        # 3. 人脸检测参数
        self.face_config = self.config.get("face", {
            "samples_per_video": 9,
            "min_face_duration": 1.0,
            "min_video_duration": 3.0,
            "num_workers": 127,
            "max_retries": 3,
            "skip_existing": True
        })
        
        # 4. 文字遮挡检测参数
        self.text_config = self.config.get("text", {
            "text_threshold": 0.5,
            "face_confidence": 0.7,
            "num_processes": 127,
            "skip_existing": True
        })
        
        # 5. 语音同步检测参数
        self.sync_config = self.config.get("sync", {
            "min_segment_duration": 1.0,
            "sampling_rate": 3,
            "mouth_detection_threshold": 0.5,
            "speech_threshold": 0.1,
            "expand_duration": 0.5,
            "random_frames": 8,
            "process_without_audio": True,
            "copy_on_failure": True,
            "overwrite": False,
            "retry_count": 1,
            "processes": 127
        })
        
    def _load_state(self) -> Dict:
        """加载处理状态，支持断点续传"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                logger.info("加载已有处理状态，支持断点续传")
                return state
            except Exception as e:
                logger.warning(f"加载状态文件失败: {e}")
                
        # 初始化新的状态
        return {
            "current_stage": 0,
            "stage_status": {
                "1_converted": "pending",
                "2_scene_filtered": "pending",
                "3_face_filtered": "pending",
                "4_text_filtered": "pending",
                "5_sync_filtered": "pending"
            },
            "processed_files": {},
            "start_time": time.time(),
            "errors": []
        }
        
    def _save_state(self):
        """保存处理状态"""
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存状态文件失败: {e}")
            
    def run(self, debug=False):
        """运行完整的视频处理流水线"""
        try:
            start_time = time.time()
            logger.info(f"开始视频处理流水线，输入目录: {self.input_dir}")
            logger.info(f"临时目录: {self.temp_dir}")
            logger.info(f"最终输出目录: {self.final_output_dir}")
            
            # 设置调试模式
            if debug:
                logger.setLevel(logging.DEBUG)
                logger.info("调试模式已开启")
            
            # 统计输入文件
            video_files = self._get_video_files(self.input_dir)
            self.progress_stats["total_files"] = len(video_files)
            logger.info(f"找到 {len(video_files)} 个视频文件待处理")
            
            if debug and video_files:
                logger.debug(f"前5个输入文件: {video_files[:5]}")
                
            # 检查输入文件是否存在
            if len(video_files) == 0:
                logger.error("输入目录中没有找到任何视频文件")
                return
                
            # 执行各处理阶段
            stages = [
                ("1_converted", "格式转换", self._stage_convert),
                ("2_scene_filtered", "场景检测", self._stage_scene_detection),
                ("3_face_filtered", "人脸检测", self._stage_face_detection),
                ("4_text_filtered", "文字遮挡检测", self._stage_text_detection),
                ("5_sync_filtered", "语音同步检测", self._stage_sync_detection)
            ]
            
            # 从上次中断的阶段继续
            start_stage = self.state.get("current_stage", 0)
            
            for idx, (stage_key, stage_name, stage_func) in enumerate(stages):
                if idx < start_stage:
                    logger.info(f"跳过已完成阶段: {stage_name}")
                    continue
                    
                self.state["current_stage"] = idx
                self.progress_stats["current_stage"] = stage_name
                
                logger.info(f"\n{'='*50}")
                logger.info(f"开始执行: {stage_name}")
                logger.info(f"{'='*50}")
                
                try:
                    stage_start = time.time()
                    success = stage_func()
                    stage_duration = time.time() - stage_start
                    
                    if success:
                        self.state["stage_status"][stage_key] = "completed"
                        logger.info(f"{stage_name} 完成，耗时: {self._format_time(stage_duration)}")
                    else:
                        self.state["stage_status"][stage_key] = "failed"
                        logger.error(f"{stage_name} 失败")
                        
                    self._save_state()
                    
                except Exception as e:
                    logger.error(f"{stage_name} 阶段出错: {str(e)}")
                    logger.error(traceback.format_exc())
                    self.state["errors"].append({
                        "stage": stage_name,
                        "error": str(e),
                        "time": datetime.now().isoformat()
                    })
                    self.state["stage_status"][stage_key] = "error"
                    self._save_state()
                    
            # 将最终结果移动到输出目录
            self._move_final_results()
            
            # 生成最终报告
            total_duration = time.time() - start_time
            self._generate_report(total_duration)
            
            logger.info(f"\n流水线处理完成！总耗时: {self._format_time(total_duration)}")
            
        except Exception as e:
            logger.error(f"流水线执行失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise
            
    def _stage_convert(self) -> bool:
        """阶段1：格式转换"""
        try:
            logger.info("执行格式转换...")
            input_dir = self.input_dir
            output_dir = self.stage_dirs["1_converted"]
            
            # 调用格式转换模块
            result = convert_to_h264_aac_serial.batch_process(
                input_dir=input_dir,
                output_dir=output_dir,
                extensions=self.convert_config["extensions"],
                workers=self.convert_config["workers"],
                crf=self.convert_config["crf"],
                preset=self.convert_config["quality"],
                audio_bitrate=self.convert_config["audio"]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"格式转换失败: {str(e)}")
            return False
            
    def _stage_scene_detection(self) -> bool:
        """阶段2：场景检测"""
        try:
            logger.info("执行场景检测...")
            input_dir = self.stage_dirs["1_converted"]
            output_dir = self.stage_dirs["2_scene_filtered"]
            
            # 检查输入目录是否有视频文件
            input_files = self._get_video_files(input_dir)
            logger.info(f"输入目录 {input_dir} 中有 {len(input_files)} 个视频文件")
            
            if len(input_files) == 0:
                logger.warning(f"场景检测阶段输入目录为空，跳过此阶段")
                return False
            
            # 创建场景检测的进度字典
            with self.global_lock:
                self.global_progress.clear()
                
            # 调用场景检测模块
            stats = scene_change_detector.process_video_batch(
                input_dir=input_dir,
                output_dir=output_dir,
                config=self.scene_config,
                min_segment_duration=self.scene_config["min_segment_duration"] * 25,  # 转换为帧数
                lock=self.global_lock,
                progress_dict=self.global_progress
            )
            
            # 检查输出文件数量
            output_files = self._get_video_files(output_dir)
            logger.info(f"场景检测输出 {len(output_files)} 个视频片段")
            logger.info(f"场景检测统计: 处理了 {stats['processed']}/{stats['total']} 个文件，提取了 {stats.get('segments', 0)} 个片段")
            
            return stats["processed"] > 0
            
        except Exception as e:
            logger.error(f"场景检测失败: {str(e)}")
            return False
            
    def _stage_face_detection(self) -> bool:
        """阶段3：人脸检测"""
        try:
            logger.info("执行人脸检测...")
            input_dir = self.stage_dirs["2_scene_filtered"]
            output_dir = self.stage_dirs["3_face_filtered"]
            
            # 检查输入目录是否有视频文件
            input_files = self._get_video_files(input_dir)
            logger.info(f"输入目录 {input_dir} 中有 {len(input_files)} 个视频文件")
            
            if len(input_files) == 0:
                logger.warning(f"人脸检测阶段输入目录为空，跳过此阶段")
                return False
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 更新全局设置
            face_detector.GLOBAL_SETTINGS.update({
                'output_dir': output_dir,
                'min_face_duration': self.face_config["min_face_duration"],
                'min_video_duration': self.face_config["min_video_duration"],
                'samples_per_video': self.face_config["samples_per_video"],
                'ffmpeg_path': "ffmpeg",
                'max_retries': self.face_config["max_retries"]
            })
            
            # 调用人脸检测模块
            results = face_detector.process_all_videos(
                input_dir=input_dir,
                output_dir=output_dir,
                num_workers=self.face_config["num_workers"],
                samples_per_video=self.face_config["samples_per_video"],
                min_face_duration=self.face_config["min_face_duration"],
                min_video_duration=self.face_config["min_video_duration"],
                max_retries=self.face_config["max_retries"],
                skip_existing=self.face_config["skip_existing"]
            )
            
            # 检查输出文件数量 - 等待一秒确保文件写入完成
            import time
            time.sleep(1)
            
            output_files = self._get_video_files(output_dir)
            logger.info(f"人脸检测输出 {len(output_files)} 个视频文件")
            
            # 检查是否文件被放在了错误的位置
            if len(output_files) == 0:
                # 尝试从默认输出目录移动文件
                default_output = "/data/wuyue/output/ocr"
                if os.path.exists(default_output):
                    default_files = self._get_video_files(default_output)
                    if len(default_files) > 0:
                        logger.info(f"从默认目录 {default_output} 移动 {len(default_files)} 个文件")
                        for file_path in default_files:
                            if 'trimmed' in os.path.basename(file_path):
                                new_path = os.path.join(output_dir, os.path.basename(file_path))
                                shutil.move(file_path, new_path)
                        output_files = self._get_video_files(output_dir)
            
            # 统计成功和失败的结果
            successful_results = [r for r in results if r is not None]
            logger.info(f"人脸检测结果：成功处理 {len(successful_results)}/{len(results)} 个文件")
            
            return len(output_files) > 0
            
        except Exception as e:
            logger.error(f"人脸检测失败: {str(e)}")
            return False
            
    def _stage_text_detection(self) -> bool:
        """阶段4：文字遮挡检测"""
        try:
            logger.info("执行文字遮挡检测...")
            input_dir = self.stage_dirs["3_face_filtered"]
            output_dir = self.stage_dirs["4_text_filtered"]
            
            # 检查输入目录是否有视频文件
            input_files = self._get_video_files(input_dir)
            logger.info(f"输入目录 {input_dir} 中有 {len(input_files)} 个视频文件")
            
            if len(input_files) == 0:
                logger.warning(f"文字检测阶段输入目录为空，跳过此阶段")
                # 尝试从上一个有效阶段复制文件
                prev_stage_files = self._find_previous_stage_files("4_text_filtered")
                if prev_stage_files:
                    logger.info(f"从上一个有效阶段复制 {len(prev_stage_files)} 个文件")
                    for src_file in prev_stage_files:
                        dst_file = os.path.join(output_dir, os.path.basename(src_file))
                        shutil.copy2(src_file, dst_file)
                return False
            
            # 调用文字检测模块
            text_detector.batch_process_videos(
                input_dir=input_dir,
                output_dir=output_dir,
                text_threshold=self.text_config["text_threshold"],
                face_confidence=self.text_config["face_confidence"],
                num_processes=self.text_config["num_processes"],
                skip_existing=self.text_config["skip_existing"]
            )
            
            # 检查输出文件数量
            output_files = self._get_video_files(output_dir)
            logger.info(f"文字检测输出 {len(output_files)} 个视频文件")
            
            return len(output_files) > 0
            
        except Exception as e:
            logger.error(f"文字检测失败: {str(e)}")
            return False
            
    def _stage_sync_detection(self) -> bool:
        """阶段5：语音同步检测"""
        try:
            logger.info("执行语音同步检测...")
            input_dir = self.stage_dirs["4_text_filtered"]
            temp_output_dir = self.stage_dirs["5_sync_filtered"]  # 先输出到临时目录
            
            # 检查输入目录是否有视频文件
            input_files = self._get_video_files(input_dir)
            logger.info(f"输入目录 {input_dir} 中有 {len(input_files)} 个视频文件")
            
            if len(input_files) == 0:
                logger.warning(f"语音同步检测阶段输入目录为空，跳过此阶段")
                return False
            
            # 调用语音同步检测模块（输出到临时目录）
            success_count, total_count = audio_sync_detector.process_directory(
                input_dir=input_dir,
                output_dir=temp_output_dir,  # 先输出到临时目录
                processes=self.sync_config["processes"],
                config=self.sync_config
            )
            
            logger.info(f"语音同步检测完成: {success_count}/{total_count} 个文件成功处理")
            
            return success_count > 0
            
        except Exception as e:
            logger.error(f"语音同步检测失败: {str(e)}")
            return False
            
    def _move_final_results(self):
        """将最终结果从临时目录移动到输出目录"""
        try:
            temp_sync_dir = self.stage_dirs["5_sync_filtered"]
            final_output_files = self._get_video_files(temp_sync_dir)
            
            if len(final_output_files) == 0:
                logger.warning("没有找到最终输出文件")
                return
                
            logger.info(f"将 {len(final_output_files)} 个最终文件移动到 {self.final_output_dir}")
            
            moved_count = 0
            for file_path in final_output_files:
                filename = os.path.basename(file_path)
                target_path = os.path.join(self.final_output_dir, filename)
                
                # 如果目标文件已存在，添加时间戳
                if os.path.exists(target_path):
                    base, ext = os.path.splitext(filename)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{base}_{timestamp}{ext}"
                    target_path = os.path.join(self.final_output_dir, filename)
                
                try:
                    shutil.move(file_path, target_path)
                    moved_count += 1
                    logger.debug(f"移动文件: {filename}")
                except Exception as e:
                    logger.error(f"移动文件失败 {filename}: {e}")
                    
            logger.info(f"成功移动 {moved_count} 个文件到最终输出目录")
            
        except Exception as e:
            logger.error(f"移动最终结果时出错: {str(e)}")
            
    def _get_video_files(self, directory: str) -> List[str]:
        """获取目录中的所有视频文件"""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        video_files = []
        
        if not os.path.exists(directory):
            logger.warning(f"目录不存在: {directory}")
            return video_files
            
        for root, _, files in os.walk(directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in video_extensions:
                    video_files.append(os.path.join(root, file))
                    
        return video_files
        
    def _find_previous_stage_files(self, current_stage: str) -> List[str]:
        """查找上一个有效阶段的文件"""
        stages = list(self.stage_dirs.keys())
        current_idx = stages.index(current_stage)
        
        # 从当前阶段往前查找
        for i in range(current_idx - 1, -1, -1):
            prev_stage = stages[i]
            prev_dir = self.stage_dirs[prev_stage]
            
            if os.path.exists(prev_dir):
                files = self._get_video_files(prev_dir)
                if files:
                    logger.info(f"在阶段 {prev_stage} 找到 {len(files)} 个文件")
                    return files
                    
        # 如果都没找到，尝试原始输入目录
        input_files = self._get_video_files(self.input_dir)
        if input_files:
            logger.info(f"在原始输入目录找到 {len(input_files)} 个文件")
            return input_files
            
        return []
        
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
    def _generate_report(self, total_duration: float):
        """生成处理报告"""
        report_path = os.path.join(self.temp_dir, "pipeline_report.txt")
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(f"视频处理流水线报告\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总耗时: {self._format_time(total_duration)}\n")
                f.write(f"输入目录: {self.input_dir}\n")
                f.write(f"临时目录: {self.temp_dir}\n")
                f.write(f"最终输出目录: {self.final_output_dir}\n\n")
                
                f.write(f"处理阶段状态:\n")
                f.write(f"{'-'*30}\n")
                for stage, status in self.state["stage_status"].items():
                    f.write(f"{stage}: {status}\n")
                    
                f.write(f"\n处理统计:\n")
                f.write(f"{'-'*30}\n")
                f.write(f"原始输入文件数: {self.progress_stats['total_files']}\n")
                
                # 最终输出统计
                final_files = self._get_video_files(self.final_output_dir)
                f.write(f"最终输出文件数: {len(final_files)}\n\n")
                
                # 统计各阶段输出文件数和处理情况
                f.write(f"各阶段处理情况:\n")
                f.write(f"{'-'*30}\n")
                prev_count = self.progress_stats['total_files']
                
                for stage_name, stage_dir in self.stage_dirs.items():
                    if os.path.exists(stage_dir):
                        files = self._get_video_files(stage_dir)
                        file_count = len(files)
                        
                        # 计算过滤率
                        if prev_count > 0:
                            filter_rate = (prev_count - file_count) / prev_count * 100
                            f.write(f"{stage_name}: {file_count} 个文件 (过滤了 {filter_rate:.1f}%)\n")
                        else:
                            f.write(f"{stage_name}: {file_count} 个文件\n")
                            
                        prev_count = file_count
                    else:
                        f.write(f"{stage_name}: 目录不存在\n")
                        
                if self.state["errors"]:
                    f.write(f"\n错误记录:\n")
                    f.write(f"{'-'*30}\n")
                    for error in self.state["errors"]:
                        f.write(f"阶段: {error['stage']}\n")
                        f.write(f"时间: {error['time']}\n")
                        f.write(f"错误: {error['error']}\n\n")
                        
                f.write(f"\n最终输出目录: {self.final_output_dir}\n")
                f.write(f"临时文件目录: {self.temp_dir}\n")
                f.write(f"注意：中间处理结果保存在临时目录，最终结果在最终输出目录\n")
                
            logger.info(f"报告已生成: {report_path}")
            
            # 将报告也复制到最终输出目录
            final_report_path = os.path.join(self.final_output_dir, "pipeline_report.txt")
            shutil.copy2(report_path, final_report_path)
            logger.info(f"报告已复制到: {final_report_path}")
            
        except Exception as e:
            logger.error(f"生成报告失败: {str(e)}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='视频处理流水线')
    parser.add_argument('--input', type=str, default="/data/wuyue/bilibili/T7/video_25fps_cleaned",
                        help='输入视频目录')
    parser.add_argument('--output', type=str, default="/data/wuyue/output/all",
                        help='输出视频目录')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路径（JSON格式）')
    parser.add_argument('--debug', action='store_true',
                        help='开启调试模式')
    
    args = parser.parse_args()
    
    # 加载配置文件
    config = {}
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"加载配置文件: {args.config}")
        except Exception as e:
            logger.warning(f"加载配置文件失败: {e}")
            
    # 创建并运行流水线
    pipeline = VideoPipeline(
        input_dir=args.input,
        output_dir=args.output,
        config=config
    )
    
    try:
        pipeline.run(debug=args.debug)
    except KeyboardInterrupt:
        logger.info("\n用户中断处理")
        pipeline._save_state()
    except Exception as e:
        logger.error(f"流水线执行失败: {str(e)}")
        pipeline._save_state()
        raise


if __name__ == "__main__":
    # 设置多进程启动方法
    multiprocessing.set_start_method('spawn', force=True)
    main()
