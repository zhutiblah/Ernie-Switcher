import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_total_loss_curves(csv_path, output_name='loss_curve', palette_colors=["#2E86AB", "#A23B72"]):
    """
    Load data from a CSV file and plot train_total_loss and test_total_loss curves over epochs.
    
    Args:
        csv_path (str): Path to the CSV file
        output_name (str): Output file name (without extension)
        palette_colors (list): List of colors for training and test curves [train_color, test_color]
    """
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data: {csv_path}")
        print(f"Data dimensions: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found -> {csv_path}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    
    required_cols = ['epoch', 'train_total_loss', 'test_total_loss']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV file missing required columns")
        print(f"Required: {required_cols}")
        print(f"Actual: {list(df.columns)}")
        return

    
    if df[required_cols].isnull().any().any():
        print("Warning: Missing values found in data, they will be removed")
        df = df.dropna(subset=required_cols)

    
    df_melted = df.melt(
        id_vars=['epoch'],
        value_vars=['train_total_loss', 'test_total_loss'],
        var_name='Loss Type',
        value_name='Loss Value'
    )

    
    df_melted['Loss Type'] = df_melted['Loss Type'].map({
        'train_total_loss': 'Training Loss',
        'test_total_loss': 'Test Loss'
    })

    
    try:
        
        plt.style.use('seaborn-paper')
    except:
        try:
            
            plt.style.use('classic')
        except:
            
            pass
    
    
    plt.rcParams.update({
        'font.size': 16,             
        'axes.labelsize': 18,         
        'axes.titlesize': 20,         
        'xtick.labelsize': 15,        
        'ytick.labelsize': 15,        
        'legend.fontsize': 16,        
        'figure.titlesize': 20,       
        'font.family': 'sans-serif',
        'axes.linewidth': 1.5,
        'grid.linewidth': 0.8,
        'lines.linewidth': 2.5,
        'patch.linewidth': 0.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
    })
    
    sns.set_palette(palette_colors)
    
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    
    for loss_type, color in zip(['Training Loss', 'Test Loss'], palette_colors):
        data_subset = df_melted[df_melted['Loss Type'] == loss_type]
        ax.plot(
            data_subset['epoch'],
            data_subset['Loss Value'],
            label=loss_type,
            color=color,
            linewidth=2.5,
            marker='o',
            markersize=4,
            markevery=max(1, len(data_subset) // 20),
            alpha=0.9
        )

    
    
    ax.set_xlabel('Epoch', fontsize=16, fontweight='bold')
    ax.set_ylabel('Total Loss', fontsize=16, fontweight='bold')
    
    ax.set_xlim(df['epoch'].min() - 0.5, df['epoch'].max() + 0.5)
    
    
    y_min = min(df['train_total_loss'].min(), df['test_total_loss'].min())
    y_max = max(df['train_total_loss'].max(), df['test_total_loss'].max())
    y_margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_margin, y_max + y_margin)

    
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)

    
    legend = ax.legend(
        loc='upper right',
        frameon=True,
        fancybox=True,
        shadow=True,
        fontsize=18,
        edgecolor='black',
        facecolor='white',
        framealpha=0.95
    )

    
    ax.tick_params(axis='both', which='major', labelsize=16, 
                   direction='in', length=6, width=1.5)
    
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    
    plt.tight_layout()

    
    
    output_svg = f'{output_name}.svg'
    plt.savefig(output_svg, format='svg', bbox_inches='tight', 
                transparent=True, dpi=300)
    print(f"✓ SVG image saved: {output_svg}")

    
    output_png = f'{output_name}.png'
    plt.savefig(output_png, format='png', bbox_inches='tight', 
                dpi=300, facecolor='white')
    print(f"✓ PNG image saved: {output_png}")

    
    print("\n" + "="*50)
    print("Data Statistics Summary:")
    print("="*50)
    print(f"Epoch Range: {df['epoch'].min():.0f} - {df['epoch'].max():.0f}")
    print(f"\nTraining Loss:")
    print(f"  Min Value: {df['train_total_loss'].min():.6f} (Epoch {df.loc[df['train_total_loss'].idxmin(), 'epoch']:.0f})")
    print(f"  Max Value: {df['train_total_loss'].max():.6f} (Epoch {df.loc[df['train_total_loss'].idxmax(), 'epoch']:.0f})")
    print(f"  Final Value: {df['train_total_loss'].iloc[-1]:.6f}")
    print(f"\nTest Loss:")
    print(f"  Min Value: {df['test_total_loss'].min():.6f} (Epoch {df.loc[df['test_total_loss'].idxmin(), 'epoch']:.0f})")
    print(f"  Max Value: {df['test_total_loss'].max():.6f} (Epoch {df.loc[df['test_total_loss'].idxmax(), 'epoch']:.0f})")
    print(f"  Final Value: {df['test_total_loss'].iloc[-1]:.6f}")
    print("="*50)

    
    plt.show()
    
    
    plt.close()


if __name__ == '__main__':
    
    csv_file_path = '/home/lirunting/lrt/sample/Prediction_Translation_Strength/Synthesizing_mRNA/result/Switch-ddpm-2025-12-08-22-26-loss_history.csv'
    
    print("Starting to plot loss curves...")
    plot_total_loss_curves(
        csv_path=csv_file_path,
        output_name='weight_0.05total_loss_curve',
        palette_colors=["#2E86AB", "#A23B72"]
    )
    print("\nPlotting complete!")
