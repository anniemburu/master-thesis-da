import matplotlib.pyplot as plt
import os

from utils.io_utils import get_output_path



def loss_vizualization(args, loss_hist):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    #epochs = range(1, args.epochs+1)

    # Plot each dataset
    min_len = min(len(loss_hist['train'][0]), len(loss_hist['val'][0]))
    epochs = range(1, min_len+1)
    axes[0, 0].plot(epochs, loss_hist['train'][0][:min_len], label='Training Loss')
    axes[0, 0].plot(epochs, loss_hist['val'][0][:min_len], label='Validation Loss')
    axes[0, 0].set_ylabel('Epochs')
    axes[0, 0].set_xlabel('Loss')
    axes[0, 0].set_title('Result Fold 0')
    axes[0, 0].legend()

    min_len = min(len(loss_hist['train'][1]), len(loss_hist['val'][1]))
    epochs = range(1, min_len+1)
    axes[0, 1].plot(epochs, loss_hist['train'][1][:min_len], label='Training Loss')
    axes[0, 1].plot(epochs, loss_hist['val'][1][:min_len], label='Validation Loss')
    axes[0, 1].set_ylabel('Epochs')
    axes[0, 1].set_xlabel('Loss')
    axes[0, 1].set_title('Result Fold 1')
    axes[0, 1].legend()

    min_len = min(len(loss_hist['train'][2]), len(loss_hist['val'][2]))
    epochs = range(1, min_len+1)
    axes[0, 2].plot(epochs, loss_hist['train'][2][:min_len], label='Training Loss')
    axes[0, 2].plot(epochs, loss_hist['val'][2][:min_len], label='Validation Loss')
    axes[0, 2].set_ylabel('Epochs')
    axes[0, 2].set_xlabel('Loss')
    axes[0, 2].set_title('Result Fold 2')
    axes[0, 2].legend()

    min_len = min(len(loss_hist['train'][3]), len(loss_hist['val'][3]))
    epochs = range(1, min_len+1)
    axes[1, 0].plot(epochs, loss_hist['train'][3][:min_len], label='Training Loss')
    axes[1, 0].plot(epochs, loss_hist['val'][3][:min_len], label='Validation Loss')
    axes[1, 0].set_ylabel('Epochs')
    axes[1, 0].set_xlabel('Loss')
    axes[1, 0].set_title('Result Fold 3')
    axes[1, 0].legend()

    min_len = min(len(loss_hist['train'][4]), len(loss_hist['val'][4]))
    epochs = range(1, min_len+1)
    axes[1, 1].plot(epochs, loss_hist['train'][4][:min_len], label='Training Loss')
    axes[1, 1].plot(epochs, loss_hist['val'][4][:min_len], label='Validation Loss')
    axes[1, 1].set_ylabel('Epochs')
    axes[1, 1].set_xlabel('Loss')
    axes[1, 1].set_title('Result Fold 4')
    axes[1, 1].legend()

    axes[1, 2].axis('off')
    plt.tight_layout()

    #Save the plot

    path = get_output_path(args, 'visualization', file_type = None)
    os.makedirs(path, exist_ok=True)

    axes[0, 0].figure.savefig(f'{path}/loss_curve_exp.png', dpi=300)
    #axes[0, 1].figure.savefig(f'{path}/kf_1.png', dpi=300)
    #axes[0, 2].figure.savefig(f'{path}/kf_2.png', dpi=300)
    #axes[1, 0].figure.savefig(f'{path}/kf_3.png', dpi=300)
    #axes[1, 1].figure.savefig(f'{path}/kf_4.png', dpi=300)

    # Show the plots
    plt.show()

    print("Plots saved successfully!")