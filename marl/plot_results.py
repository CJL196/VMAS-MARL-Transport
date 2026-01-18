import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

def plot_results():
    log_files = glob.glob("marl/outputs/log_*.csv")
    plt.figure(figsize=(10, 6))

    for log_file in log_files:
        try:
            df = pd.read_csv(log_file)
            label = os.path.basename(log_file).replace("log_", "").replace(".csv", "")
            plt.plot(df['step'], df['reward'], label=label)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.title('Training Curves on Transport Scenario')
    plt.legend()
    plt.grid(True)
    plt.savefig("marl/outputs/results.png")
    print("Plot saved to marl/outputs/results.png")

if __name__ == "__main__":
    plot_results()
