#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import subprocess
import datetime
import glob
import argparse
import multiprocessing
import tqdm
from typing import List, Optional, Tuple, Dict
import shutil


def get_video_info(file_path: str) -> Dict:
    """获取视频文件信息"""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,duration',
            '-of', 'json',
            file_path
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        import json
        return json.loads(result.stdout)
    except Exception:
        return {}


def should_convert(input_file: str, output_file: str) -> bool:
    """判断是否需要转换文件"""
    # 如果输出文件不存在，则需要转换
    if not os.path.exists(output_file):
        return True
    
    # 如果输入文件比输出文件新，则需要重新转换
    if os.path.getmtime(input_file) > os.path.getmtime(output_file):
        return True
    
    # 检查输出文件是否完整
    try:
        info = get_video_info(output_file)
        if not info:
            return True
    except Exception:
        return True
    
    return False


def process_file(args: Tuple) -> Tuple[bool, str, str, str]:
    """
    转换视频文件为H.264/AAC格式
    
    参数:
    args: 包含(input_file, output_file, crf, preset, audio_bitrate)的元组
    
    返回:
    Tuple[bool, str, str, str]: (是否成功, 输入文件, 输出文件, 错误信息)
    """
    input_file, output_file, crf, preset, audio_bitrate = args
    
    # 检查是否需要转换
    if not should_convert(input_file, output_file):
        return (True, input_file, output_file, "跳过：文件已存在")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 构建ffmpeg命令
    command = [
        'ffmpeg',
        '-i', input_file,
        '-c:v', 'libx264',
        '-preset', preset,
        '-crf', str(crf),
        '-c:a', 'aac',
        '-b:a', audio_bitrate,
        '-y',  # 覆盖已有文件
        output_file
    ]
    
    # 执行ffmpeg命令
    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return (True, input_file, output_file, "")
    except subprocess.CalledProcessError as e:
        return (False, input_file, output_file, e.stderr)


def find_video_files(directory: str, extensions: List[str]) -> List[str]:
    """查找目录中的视频文件"""
    video_files = []
    for ext in extensions:
        if not ext.startswith('*'):
            ext = f'*.{ext}'
        # 递归搜索所有子目录
        for root, _, _ in os.walk(directory):
            video_files.extend(glob.glob(os.path.join(root, ext)))
    return video_files


def prepare_task_list(video_files: List[str], input_dir: str, output_dir: str, 
                      crf: int, preset: str, audio_bitrate: str) -> List[Tuple]:
    """准备处理任务列表"""
    tasks = []
    for file in video_files:
        # 保持目录结构
        rel_path = os.path.relpath(file, input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        # 修改扩展名为mp4
        base_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        output_file = os.path.join(base_dir, f"{base_name}.mp4")
        
        tasks.append((file, output_file, crf, preset, audio_bitrate))
    return tasks


def batch_process(input_dir: str, output_dir: str, extensions: List[str], 
                  workers: int, crf: int, preset: str, audio_bitrate: str) -> bool:
    """批量处理视频文件"""
    # 创建日志文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"conversion_{timestamp}.log")
    failed_log = os.path.join(log_dir, f"conversion_{timestamp}_failed.log")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有视频文件
    video_files = find_video_files(input_dir, extensions)
    total = len(video_files)
    
    if total == 0:
        print(f"错误：在 {input_dir} 中未找到任何视频文件")
        print("请检查：")
        print(f"1. 目录路径是否正确: '{input_dir}'")
        print(f"2. 文件扩展名是否为: {extensions}")
        return False
    
    print(f"找到 {total} 个视频文件")
    
    # 准备任务列表
    tasks = prepare_task_list(video_files, input_dir, output_dir, 
                             crf, preset, audio_bitrate)
    
    # 使用进程池并行处理
    results = []
    failed_files = []
    
    # 使用指定的工作进程数量
    actual_workers = workers
    print(f"使用 {actual_workers} 个工作进程")
    
    with multiprocessing.Pool(actual_workers) as pool:
        # 使用tqdm显示进度条
        for result in tqdm.tqdm(pool.imap_unordered(process_file, tasks), total=total, 
                               desc="转换进度", unit="文件"):
            success, input_file, output_file, error = result
            
            # 保存结果
            results.append(result)
            
            if not success:
                failed_files.append((input_file, error))
    
    # 统计结果
    success_count = sum(1 for r in results if r[0])
    failed_count = total - success_count
    skipped_count = sum(1 for r in results if r[0] and "跳过" in r[3])
    
    # 保存日志
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"转换日志 - {datetime.datetime.now()}\n")
        f.write(f"输入目录: {input_dir}\n")
        f.write(f"输出目录: {output_dir}\n")
        f.write(f"总文件数: {total}\n")
        f.write(f"成功: {success_count} (包含跳过: {skipped_count})\n")
        f.write(f"失败: {failed_count}\n\n")
        
        for success, input_file, output_file, error in results:
            status = "✓ 成功" if success else "✗ 失败"
            if "跳过" in error:
                status = "- 跳过"
            f.write(f"{status}: {input_file} -> {output_file}\n")
            if error and not success:
                f.write(f"  错误: {error}\n")
    
    # 保存失败文件列表
    if failed_files:
        with open(failed_log, 'w', encoding='utf-8') as f:
            for file, error in failed_files:
                f.write(f"{file}\n")
                f.write(f"错误: {error}\n\n")
    
    # 生成报告
    print("\n==== 任务完成 ====")
    print(f"处理总数: {total}")
    print(f"成功: {success_count} (包含跳过: {skipped_count})")
    print(f"失败: {failed_count}")
    print(f"日志保存至: {log_file}")
    if failed_files:
        print(f"失败记录: {failed_log}")
    
    return success_count > 0


def check_dependencies():
    """检查依赖程序是否存在"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("错误: 未找到ffmpeg。请确保ffmpeg已安装并添加到PATH中。")
        print("安装方法:")
        print("  - Ubuntu/Debian: sudo apt install ffmpeg")
        print("  - CentOS/RHEL: sudo yum install ffmpeg")
        print("  - macOS: brew install ffmpeg")
        print("  - Windows: 下载并安装，然后添加到PATH")
        return False
    
    try:
        subprocess.run(['ffprobe', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("警告: 未找到ffprobe。一些高级功能可能无法使用。")
    
    return True


def get_cpu_info():
    """获取CPU信息"""
    cpu_count = multiprocessing.cpu_count()
    cpu_info = {"count": cpu_count}
    
    try:
        if sys.platform == "linux":
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.strip().startswith("model name"):
                        cpu_info["model"] = line.split(':')[1].strip()
                        break
        elif sys.platform == "darwin":  # macOS
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                cpu_info["model"] = result.stdout.strip()
    except Exception:
        pass
    
    return cpu_info


def get_disk_space(path):
    """获取磁盘空间信息"""
    try:
        total, used, free = shutil.disk_usage(path)
        return {
            "total_gb": total // (1024 ** 3),
            "used_gb": used // (1024 ** 3),
            "free_gb": free // (1024 ** 3)
        }
    except Exception:
        return {}


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='批量视频转码工具 - 更高效的视频批处理')
    parser.add_argument('-i', '--input', default="/data/wuyue/bilibili",
                        help='输入视频目录路径')
    parser.add_argument('-o', '--output', default="/data/wuyue/douyin/converted",
                        help='输出视频目录路径')
    parser.add_argument('-e', '--extensions', default="mp4,MP4,mov,avi,mkv,flv,wmv",
                        help='要处理的视频文件扩展名，用逗号分隔')
    parser.add_argument('-w', '--workers', type=int, default=127,
                        help='并行工作进程数量 (0=自动设置为CPU核心数)')
    parser.add_argument('-crf', '--quality', type=int, default=23,
                        help='视频质量 (0-51, 值越小质量越高, 23为默认)')
    parser.add_argument('-p', '--preset', default='fast',
                        choices=['ultrafast', 'superfast', 'veryfast', 'faster', 
                                'fast', 'medium', 'slow', 'slower', 'veryslow'],
                        help='编码速度预设 (较慢的预设提供更好的压缩率)')
    parser.add_argument('-a', '--audio', default='192k',
                        help='音频比特率')
    
    args = parser.parse_args()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 获取系统信息
    cpu_info = get_cpu_info()
    disk_info = get_disk_space(args.output)
    
    # 如果使用了默认的127进程，提示用户这可能会超出系统资源限制
    if args.workers == 127:
        cpu_count = cpu_info.get('count', 0)
        if cpu_count > 0 and args.workers > cpu_count * 4:
            print(f"警告: 您设置了 {args.workers} 个工作进程，这远超过您的CPU核心数 ({cpu_count})。")
            print("      这可能导致系统过载，性能下降，甚至可能导致系统崩溃。")
            print("      建议的最大进程数为CPU核心数的2-4倍。")
            print("      如果您确定要继续，请按回车键...")
            print("      要调整进程数，请按Ctrl+C退出，然后使用-w选项指定不同的进程数。")
            try:
                input()  # 等待用户确认
            except KeyboardInterrupt:
                print("\n用户取消操作。")
                sys.exit(0)
    
    # 处理文件扩展名
    extensions = [ext.strip() for ext in args.extensions.split(',') if ext.strip()]
    
    print("==== 高效视频转码工具 ====")
    print(f"系统信息:")
    print(f"  - 操作系统: {os.uname().sysname if hasattr(os, 'uname') else sys.platform}")
    print(f"  - CPU: {cpu_info.get('model', 'Unknown')} ({cpu_info.get('count', 'Unknown')}核)")
    
    if disk_info:
        print(f"  - 磁盘空间: 总计 {disk_info['total_gb']}GB, 可用 {disk_info['free_gb']}GB")
    
    print(f"开始时间: {datetime.datetime.now()}")
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"文件类型: {args.extensions}")
    print(f"编码设置: preset={args.preset}, crf={args.quality}, audio={args.audio}")
    print(f"工作进程: {args.workers} (超高并发模式)")
    
    # 执行批处理
    result = batch_process(args.input, args.output, extensions,
                         args.workers, args.quality, args.preset, args.audio)
    
    # 结束时间
    print(f"结束时间: {datetime.datetime.now()}")
    
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()