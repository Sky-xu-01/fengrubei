import os
import sys
import subprocess

# 将项目根目录添加到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def main():
    script_path = os.path.join(BASE_DIR, "train.py")
    print(f"\n{'='*20}\nStarting Model Training\n{'='*20}")
    # 运行训练脚本
    subprocess.run([sys.executable, script_path], cwd=BASE_DIR)

if __name__ == "__main__":
    main()
