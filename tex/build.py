#!/usr/bin/env python3
"""
LaTeX 编译脚本
替代 makefile 的功能
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# 配置
MAIN_FILE = "report.tex"
BUILD_DIR = "build"
LATEX_DIR = "."  # LaTeX 项目所在目录
SCRIPT_DIR = Path(__file__).parent
WATCH_INTERVAL = 0.5  # 文件监听检查间隔（秒）
MIN_BUILD_INTERVAL = 0  # 默认不限制，可通过 --debounce 参数设置


def compile_pdf():
    """编译 PDF 文件"""
    print("🔨 开始编译 PDF...")
    
    latex_path = SCRIPT_DIR / LATEX_DIR
    
    # 检查 latex 目录是否存在
    if not latex_path.exists():
        print(f"❌ 错误: 找不到 {LATEX_DIR} 目录")
        return 1
    
    # 检查 main.tex 是否存在
    main_tex_path = latex_path / MAIN_FILE
    if not main_tex_path.exists():
        print(f"❌ 错误: 找不到 {LATEX_DIR}/{MAIN_FILE}")
        return 1
    
    # 确保 build 目录存在
    build_path = latex_path / BUILD_DIR
    build_path.mkdir(exist_ok=True)
    
    # 执行 latexmk 命令
    cmd = [
        "latexmk",
        "-pdf",
        f"-outdir={BUILD_DIR}",
        MAIN_FILE
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=latex_path,  # 在 latex 目录下执行
            check=True,
            capture_output=False
        )
        print(f"✅ 编译成功！PDF 文件位于: {LATEX_DIR}/{BUILD_DIR}/{MAIN_FILE.replace('.tex', '.pdf')}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"❌ 编译失败！错误码: {e.returncode}")
        return 1
    except FileNotFoundError:
        print("❌ 错误: 未找到 latexmk 命令，请确保已安装 LaTeX 环境")
        return 1


def clean_build():
    """删除 build 文件夹"""
    build_path = SCRIPT_DIR / LATEX_DIR / BUILD_DIR
    
    if build_path.exists():
        print(f"🗑️  删除 {LATEX_DIR}/{BUILD_DIR} 文件夹...")
        shutil.rmtree(build_path)
        print("✅ 清理完成！")
    else:
        print(f"ℹ️  {LATEX_DIR}/{BUILD_DIR} 文件夹不存在，无需清理")
    
    return 0


def get_file_mtimes(directory):
    """获取目录下所有文件的修改时间"""
    mtimes = {}
    latex_path = SCRIPT_DIR / directory
    
    if not latex_path.exists():
        return mtimes
    
    # 遍历所有 .tex 文件
    for tex_file in latex_path.rglob("*.tex"):
        # 跳过 build 目录
        if BUILD_DIR in tex_file.parts:
            continue
        try:
            mtimes[tex_file] = tex_file.stat().st_mtime
        except OSError:
            pass
    
    return mtimes


def watch_and_build(debounce_seconds=0):
    """监听文件变化并自动编译
    
    Args:
        debounce_seconds: 防抖时间（秒），相邻两次编译的最小间隔
    """
    print(f"👀 开始监听 {LATEX_DIR} 目录...")
    if debounce_seconds > 0:
        print(f"⏱️  防抖设置: 相邻编译间隔至少 {debounce_seconds} 秒")
    print("按 Ctrl+C 停止监听\n")
    
    # 获取初始文件修改时间
    last_mtimes = get_file_mtimes(LATEX_DIR)
    last_build_time = 0
    
    try:
        while True:
            time.sleep(WATCH_INTERVAL)
            
            # 获取当前文件修改时间
            current_mtimes = get_file_mtimes(LATEX_DIR)
            
            # 检查是否有文件变化
            changed = False
            
            # 检查新增或修改的文件
            for file_path, mtime in current_mtimes.items():
                if file_path not in last_mtimes or last_mtimes[file_path] != mtime:
                    changed = True
                    rel_path = file_path.relative_to(SCRIPT_DIR)
                    print(f"📝 检测到文件变化: {rel_path}")
                    break
            
            # 检查删除的文件
            if not changed:
                for file_path in last_mtimes:
                    if file_path not in current_mtimes:
                        changed = True
                        rel_path = file_path.relative_to(SCRIPT_DIR)
                        print(f"🗑️  检测到文件删除: {rel_path}")
                        break
            
            if changed:
                # 检查防抖时间
                current_time = time.time()
                time_since_last_build = current_time - last_build_time
                
                if debounce_seconds > 0 and time_since_last_build < debounce_seconds:
                    wait_time = debounce_seconds - time_since_last_build
                    print(f"⏳ 等待 {wait_time:.1f} 秒后编译（防抖）...")
                    time.sleep(wait_time)
                
                # 执行编译
                print(f"\n{'='*60}")
                print(f"🕐 {datetime.now().strftime('%H:%M:%S')} - 开始自动编译")
                print(f"{'='*60}")
                
                result = compile_pdf()
                last_build_time = time.time()
                
                if result == 0:
                    print(f"{'='*60}")
                    print(f"✨ 编译完成，继续监听...\n")
                else:
                    print(f"{'='*60}")
                    print(f"⚠️  编译出错，继续监听...\n")
                
                # 更新文件修改时间
                last_mtimes = get_file_mtimes(LATEX_DIR)
            
    except KeyboardInterrupt:
        print("\n\n👋 停止监听")
        return 0


def show_help():
    """显示帮助信息"""
    help_text = """
LaTeX 编译脚本

用法:
    python build.py [命令] [选项]

命令:
    build, make       编译 PDF（默认）
    clean             删除 build 文件夹
    watch             监听文件变化并自动编译
    help              显示此帮助信息

选项:
    --debounce N      设置防抖时间（秒），相邻两次编译间隔不小于 N 秒
                      仅用于 watch 命令，默认为 0（不限制）

示例:
    python build.py                    # 编译 PDF
    python build.py build              # 编译 PDF
    python build.py clean              # 清理 build 文件夹
    python build.py watch              # 监听并自动编译（无防抖）
    python build.py watch --debounce 3 # 监听并自动编译（最少间隔 3 秒）
"""
    print(help_text)


def main():
    """主函数"""
    # 获取命令参数
    args = sys.argv[1:]
    command = args[0] if len(args) > 0 else "build"
    
    # 处理帮助命令
    if command in ["help", "-h", "--help"]:
        show_help()
        return 0
    
    # 处理编译命令
    if command in ["build", "make"]:
        return compile_pdf()
    
    # 处理清理命令
    elif command == "clean":
        return clean_build()
    
    # 处理监听命令
    elif command == "watch":
        debounce = 0
        
        # 解析 --debounce 参数
        if len(args) > 1:
            for i, arg in enumerate(args[1:], 1):
                if arg == "--debounce":
                    if i + 1 < len(args):
                        try:
                            debounce = float(args[i + 1])
                            if debounce < 0:
                                print("❌ 错误: 防抖时间不能为负数")
                                return 1
                        except ValueError:
                            print(f"❌ 错误: 无效的防抖时间 '{args[i + 1]}'")
                            return 1
                    else:
                        print("❌ 错误: --debounce 需要指定时间（秒）")
                        return 1
        
        return watch_and_build(debounce)
    
    # 未知命令
    else:
        print(f"❌ 未知命令: {command}")
        print("使用 'python build.py help' 查看帮助信息")
        return 1


if __name__ == "__main__":
    sys.exit(main())

