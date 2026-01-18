import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_loss_curves(csv_path, state_name, palette_colors=["#4C72B0", "#DD8452"]):
    """
    Load data from specified CSV file and plot training vs validation loss curves
    based on its explicit column structure.

    Args:
        csv_path (str): Path to the CSV file.
        state_name (str): State name (e.g., 'ON' or 'OFF') for chart title.
        palette_colors (list): Color list for training and validation curves.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found -> {csv_path}")
        print("Please check if the file path is correct.")
        return

    required_cols = ['epoch', 'train_loss', 'test_loss']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file missing required columns. Expected {required_cols}, but found {list(df.columns)}")
        return

    df_melted = df.melt(
        id_vars=['epoch'],
        value_vars=['train_loss', 'test_loss'],
        var_name='Loss Type',
        value_name='Loss Value'
    )

    df_melted['Loss Type'] = df_melted['Loss Type'].map({
        'train_loss': 'Training Loss',
        'test_loss': 'Validation Loss'
    })

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    ax = sns.lineplot(
        data=df_melted,
        x='epoch',
        y='Loss Value',
        hue='Loss Type',
        palette=palette_colors,
        lw=2,
        ci='sd'
    )

    plt.title(f'Ernie-Switcher {state_name} State', fontsize=16, weight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Legend', fontsize=10, frameon=True, facecolor='white')

    plt.xlim(left=df['epoch'].min())

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()

    output_filename_png = f'{state_name.lower()}_state_loss_curve.png'
    output_filename_svg = f'{state_name.lower()}_state_loss_curve.svg'
    plt.savefig(output_filename_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_filename_svg, bbox_inches='tight')
    print(f"Images saved as: {output_filename_png} and {output_filename_svg}")
    
    plt.show()


if __name__ == '__main__':
    off_state_csv_path = '/home/lirunting/lrt/sample/Prediction_Translation_Strength/sample/Prediction_Translation_Strength/result/loss_records/adjust1_off_all_folds_loss_record_pearson_mse_ernie.csv'
    print("--- Plotting OFF-State Loss Curves ---")
    plot_loss_curves(off_state_csv_path, 'OFF')

    on_state_csv_path = '/home/lirunting/lrt/sample/Prediction_Translation_Strength/sample/Prediction_Translation_Strength/result/adjust_on2_training_log.csv' 
    
    print("\n--- Plotting ON-State Loss Curves ---")
    plot_loss_curves(on_state_csv_path, 'ON')
