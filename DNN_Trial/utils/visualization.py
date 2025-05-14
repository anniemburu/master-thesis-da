import matplotlib.pyplot as plt
import os
import numpy as np

from utils.io_utils import get_output_path



def loss_vizualization(args, loss_hist, type=None):

    if type == 'train':
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

        axes[0, 0].figure.savefig(f'{path}/loss_curve_br.png', dpi=300)
        #axes[0, 1].figure.savefig(f'{path}/kf_1.png', dpi=300)
        #axes[0, 2].figure.savefig(f'{path}/kf_2.png', dpi=300)
        #axes[1, 0].figure.savefig(f'{path}/kf_3.png', dpi=300)
        #axes[1, 1].figure.savefig(f'{path}/kf_4.png', dpi=300)

        # Show the plots

    else:
        for i in range(args.outer_splits):
            # Load train and test loss files for run `i`
            min_idx = min(len(loss_hist['train'][i]), len(loss_hist['test'][i]))
            epochs = range(1, min_idx+1)
            train_loss = loss_hist['train'][i][:min_idx]  # Replace with actual loading logic
            test_loss = loss_hist['test'][i][:min_idx]
            print(f"Train Loss Length: {len(train_loss)}, Test Loss Length: {len(test_loss)}, epochs: {len(epochs)}, min_idx: {min_idx}")
            
            # Create a new figure for each pair
            plt.figure(figsize=(8, 5))
            
            # Plot training and test loss on the same axes
            plt.plot(epochs, train_loss, label=f'Train Loss (Run {i})', color='blue', linestyle='-')
            plt.plot(epochs, test_loss, label=f'Test Loss (Run {i})', color='red', linestyle='--')
            
            # Add labels, title, and legend
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Training vs Test Loss (Run {i})')
            plt.legend()
            
            # Save the plot
            path = get_output_path(args, 'visualization', file_type = None)
            os.makedirs(path, exist_ok=True)
            plt.savefig(f'{path}/loss_curve_{i}.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close the figure to free memory

            plt.show();

    print("Plots saved successfully!")