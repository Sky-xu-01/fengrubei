import csv
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(BASE_DIR, "vis_results")
SUMMARY_CSV = os.path.join(VIS_DIR, "fold_t_mse_summary.csv")
SUMMARY_PNG = os.path.join(VIS_DIR, "fold_t_mse_summary.png")


def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"Summary csv not found: {SUMMARY_CSV}")
        print("Please run visualize_results.py for each fold first.")
        return

    folds = []
    t_mse = []
    with open(SUMMARY_CSV, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            folds.append(int(row["fold"]))
            t_mse.append(float(row["t_mse"]))

    if not folds:
        print("No fold rows found in summary csv.")
        return

    pairs = sorted(zip(folds, t_mse), key=lambda x: x[0])
    folds = [p[0] for p in pairs]
    t_mse = [p[1] for p in pairs]

    plt.figure(figsize=(8, 5))
    plt.plot(folds, t_mse, marker="o", linewidth=2, label="T_MSE")
    for x, y in zip(folds, t_mse):
        plt.text(x, y, f"{y:.3e}", fontsize=8, ha="center", va="bottom")

    plt.xlabel("Fold")
    plt.ylabel("Temperature MSE")
    plt.title("Fold-wise Temperature MSE Summary")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(SUMMARY_PNG, dpi=150)
    print(f"Saved summary curve to {SUMMARY_PNG}")


if __name__ == "__main__":
    main()
