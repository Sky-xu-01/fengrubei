import os
import sys
import subprocess

# 将项目根目录添加到路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

def main():
    # 步骤 1: 进行推理预测
    predict_script = os.path.join(BASE_DIR, "predict.py")
    print(f"\n{'='*20}\nRunning Prediction\n{'='*20}")
    subprocess.run([sys.executable, predict_script], cwd=BASE_DIR)
    
    # 步骤 2: 背景结果可视化
    vis_script = os.path.join(BASE_DIR, "visualize_results.py")
    print(f"\n{'='*20}\nGenerating Visualizations\n{'='*20}")
    subprocess.run([sys.executable, vis_script], cwd=BASE_DIR)

    # 步骤 3: 绘制当前折训练曲线
    curve_script = os.path.join(BASE_DIR, "plot_training_curve.py")
    print(f"\n{'='*20}\nPlotting Fold Training Curve\n{'='*20}")
    subprocess.run([sys.executable, curve_script], cwd=BASE_DIR)

    # 步骤 4: 绘制多折温度 MSE 汇总曲线
    summary_script = os.path.join(BASE_DIR, "plot_fold_summary.py")
    print(f"\n{'='*20}\nPlotting Fold MSE Summary\n{'='*20}")
    subprocess.run([sys.executable, summary_script], cwd=BASE_DIR)

if __name__ == "__main__":
    main()
