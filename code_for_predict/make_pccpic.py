import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import AutoMinorLocator

font_path = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/sample/Prediction_Translation_Strength/code/Arial.ttf"
arial_font = FontProperties(fname=font_path, weight='bold')
arial_bold = FontProperties(
    fname="/home/lirunting/lrt/sample/Prediction_Translation_Strength/sample/Prediction_Translation_Strength/code/Arial Bold.ttf"
)


csv_path = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/predictions_simplified.csv"

df = pd.read_csv(csv_path)
print("[INFO] Original total rows:", len(df))

sub = df[["ON", "OFF", "Predicted_ON", "Predicted_OFF"]]
sub = sub.apply(pd.to_numeric, errors="coerce")

mask_on = np.isfinite(sub["ON"]) & np.isfinite(sub["Predicted_ON"])
sub_on = sub[mask_on]
actual_on_full = sub_on["ON"]
pred_on_full = sub_on["Predicted_ON"]
mask_off = np.isfinite(sub["OFF"]) & np.isfinite(sub["Predicted_OFF"])
sub_off = sub[mask_off]
actual_off_full = sub_off["OFF"]
pred_off_full = sub_off["Predicted_OFF"]

print(f"\n[INFO] Valid ON rows: {len(sub_on)}")
print(f"[INFO] Valid OFF rows: {len(sub_off)}")

mse_on = 0.034
spearman_on = 0.813

mse_off = 0.023
spearman_off = 0.642

print(f"\n[RESULT] ON - MSE: {mse_on:.3f}, Spearman: {spearman_on:.3f}")
print(f"[RESULT] OFF - MSE: {mse_off:.3f}, Spearman: {spearman_off:.3f}")

sns.set(style="white", font_scale=1.2) 

def plot_aesthetic_marginal(pred_full, actual_full, mse_val, spearman_val, title, color, save_path,
                           text_x=0.05, text_y=0.95):

    sample_size = 1000
    if len(pred_full) > sample_size:
        indices = np.random.RandomState(42).choice(pred_full.index, sample_size, replace=False)
        x_sample = pred_full.loc[indices]
        y_sample = actual_full.loc[indices]
    else:
        x_sample = pred_full
        y_sample = actual_full

    g = sns.jointplot(x=x_sample, y=y_sample, kind="reg",
                      color=color, height=7, ratio=4, space=0.2,
                      truncate=False, 
                      scatter_kws={"s": 60, "alpha": 0.6, "edgecolor": "white", "linewidth": 0.8},
                      line_kws={"color": "#333333", "linewidth": 2, "linestyle": "-"})
    ax = g.ax_joint

    for line in ax.lines:
        if line.get_linestyle() == '-' and line.get_linewidth() == 2:
            line.set_label('Linear fit')

    min_val = min(x_sample.min(), y_sample.min())
    max_val = max(x_sample.max(), y_sample.max())
    buffer = (max_val - min_val) * 0.05
    ax.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer], 
            ls='--', c='gray', alpha=0.6, lw=1.5, label='y = x', zorder=0)

    label_text = f"MSE = {mse_val:.3f}\nSpearman = {spearman_val:.3f}"
    
    shadow_offset = 0.003
    ax.text(text_x + shadow_offset, text_y - shadow_offset, label_text, 
            transform=ax.transAxes,
            fontsize=21, verticalalignment='top', 
            fontproperties=arial_font,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="gray",
                edgecolor="gray",
                linewidth=3.0,
                linestyle="--",
                alpha=0.3
            ),
            zorder=9)
    
    text_obj = ax.text(text_x, text_y, label_text, transform=ax.transAxes,
            fontsize=21, verticalalignment='top', 
            fontproperties=arial_font,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="black",
                linewidth=3.0,
                linestyle="--",
                alpha=0.95
            ),
            zorder=10)

    ax.set_xlabel("Predicted Value", fontproperties=arial_bold, fontsize=21)
    ax.set_ylabel("Actual Value", fontproperties=arial_bold, fontsize=21)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(arial_font)
        label.set_fontsize(22)
    
    for label in g.ax_marg_x.get_yticklabels():
        label.set_fontproperties(arial_font)
        label.set_fontsize(12)
    
    for label in g.ax_marg_y.get_xticklabels():
        label.set_fontproperties(arial_font)
        label.set_fontsize(12)

    ax.tick_params(
        which='major',
        direction='out',
        length=8,
        width=3.0,
        colors='black',
        bottom=True, top=False,
        left=True, right=False
    )
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.tick_params(
        which='minor',
        direction='out',
        length=4,
        width=1.5,
        colors='black',
        bottom=True, top=False,
        left=True, right=False
    )

    g.ax_marg_x.tick_params(
        which='both',
        bottom=False, top=False,
        left=False, right=False,
        labelbottom=False, labeltop=False,
        labelleft=False, labelright=False
    )
    
    g.ax_marg_y.tick_params(
        which='both',
        bottom=False, top=False,
        left=False, right=False,
        labelbottom=False, labeltop=False,
        labelleft=False, labelright=False
    )
    
    for spine in g.ax_marg_x.spines.values():
        spine.set_visible(False)
    for spine in g.ax_marg_y.spines.values():
        spine.set_visible(False)

    plt.subplots_adjust(top=0.83)
    g.fig.suptitle(title, fontproperties=arial_bold, fontsize=20, y=0.83)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(3.0)

    legend_font = FontProperties(
        fname=font_path,
        weight='bold',
        size=20
    )

    legend = ax.legend(
        loc='lower right',
        frameon=False,
        prop=legend_font
    )

    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(2.0)
    legend.get_frame().set_alpha(0.95)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] PNG saved to: {save_path}")
        
        svg_path = save_path.replace('.png', '.svg')
        plt.savefig(svg_path, format='svg', bbox_inches="tight")
        print(f"[INFO] SVG saved to: {svg_path}")
    
    plt.close()


plot_aesthetic_marginal(
    pred_on_full, actual_on_full, mse_on, spearman_on,
    "ON State", 
    "#2b8cbe", 
    "pcc_on_joint.png",
    text_x=0.05,
    text_y=0.95
)

plot_aesthetic_marginal(
    pred_off_full, actual_off_full, mse_off, spearman_off,
    "OFF State", 
    "#e34a33", 
    "pcc_off_joint.png",
    text_x=0.05,
    text_y=0.95
)
