import os
import sys
import subprocess

# 将项目根目录添加到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def run_script(script_name, extra_args=None):
    script_path = os.path.join(BASE_DIR, script_name)
    print(f"\n{'='*20}\nRunning: {script_name}\n{'='*20}")
    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"Error running {script_name}")
        sys.exit(1)

def main():
    # 步骤 1: 将 Fluent 导出的 ASCII 转换为 Excel (针对 fluent_result 文件夹)
    # 注意：确保 fluent_result 中已有 ASCII 文件（无后缀，a开头）
    run_script("convert_ascii_to_excel.py")
    
    # 步骤 2: 将 Excel 数据插值并转换为规则网格的 PKL 格式
    run_script("convert_data.py")
    
    # 步骤 3: 划分训练集和验证集
    run_script("generate_splits.py", ["--mode", "control_rolling", "--num_folds", "4", "--val_ratio", "0.2", "--active_fold", "0"])
    
    # 步骤 4: 计算归一化统计信息
    run_script("compute_stats.py")
    
    print("\nData Processing Complete! You can now run train_model.py or execute train.py directly.")

if __name__ == "__main__":
    main()
