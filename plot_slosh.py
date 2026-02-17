import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

def plot_slosh_data(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    data = np.load(file_path)
    steps = data['steps']
    rewards = data['rewards']
    accs = data['accs'] # (N, 3)

    acc_norms = np.linalg.norm(accs, axis=1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot Reward
    axs[0].plot(steps, rewards, label='Slosh Free Reward (1 - dot)', color='blue')
    axs[0].set_ylabel('Reward (Error)')
    axs[0].set_title('Slosh Free Reward over Time')
    axs[0].grid(True)
    axs[0].legend()

    # Plot Acc Norm
    axs[1].plot(steps, acc_norms, label='Acceleration Norm (m/s^2)', color='orange')
    axs[1].set_ylabel('Acc Norm')
    axs[1].set_title('Container Acceleration Magnitude')
    axs[1].grid(True)
    axs[1].legend()

    # Plot Acc Components
    axs[2].plot(steps, accs[:, 0], label='Acc X', alpha=0.7)
    axs[2].plot(steps, accs[:, 1], label='Acc Y', alpha=0.7)
    axs[2].plot(steps, accs[:, 2], label='Acc Z', alpha=0.7)
    axs[2].set_ylabel('Acc Components')
    axs[2].set_xlabel('Step')
    axs[2].set_title('Container Acceleration Components')
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()
    plot_filename = file_path.replace('.npz', '_plot.png')
    plt.savefig(plot_filename)
    print(f"Plot saved to: {os.path.abspath(plot_filename)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="slosh_data.npz", help="Path to data file")
    args = parser.parse_args()
    
    plot_slosh_data(args.file)
