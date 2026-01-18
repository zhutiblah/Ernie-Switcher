import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

sns.set_theme(style="white", font_scale=1.6)

font_path = "/home/lirunting/lrt/sample/Prediction_Translation_Strength/sample/Prediction_Translation_Strength/code/Arial.ttf"
arial_font = FontProperties(fname=font_path, weight='bold')
arial_bold = FontProperties(
    fname="/home/lirunting/lrt/sample/Prediction_Translation_Strength/sample/Prediction_Translation_Strength/code/Arial Bold.ttf"
)

metrics = ["MSE", "Spearman"]

mse_means = [0.06626, 0.06901, 0.034191]
mse_std   = [0.00242, 0.000528, 0.000131]

spearman_means = [0.78459, 0.77382, 0.813098]
spearman_std   = [0.00826, 0.00337, 0.000591]

models = ["SANDSTORM", "Valeri et al.", "Ernie-Switcher"]
colors = ["#A8C5E2", "#C7E9C0", "#4eb3d3"]

fig, ax1 = plt.subplots(figsize=(8, 6))

x = np.arange(len(metrics))
width = 0.25

for i in range(3):
    ax1.bar(x[0] + (i - 1) * width,
            mse_means[i],
            width,
            yerr=mse_std[i],
            capsize=6,
            label=models[i],
            color=colors[i],
            edgecolor='black',
            linewidth=1.5)
    x_offset = width * 0.1
    if i == 2:
        ax1.text(x[0] + (i - 1) * width+x_offset,
                 mse_means[i] + mse_std[i] + 0.002,
                 f"{mse_means[i]:.3f}",
                 ha='center', va='bottom', fontsize=18, fontweight='bold')

ax1.set_ylabel("MSE Score", fontproperties=arial_bold, fontsize=21, color='black')
ax1.set_ylim(0, 0.10)
ax1.tick_params(axis='y', labelcolor='black', direction='out', length=8, width=2)

for label in ax1.get_yticklabels():
    label.set_fontproperties(arial_font)
    label.set_fontsize(22)

ax2 = ax1.twinx()

for i in range(3):
    ax2.bar(x[1] + (i - 1) * width,
            spearman_means[i],
            width,
            yerr=spearman_std[i],
            capsize=6,
            color=colors[i],
            edgecolor='black',
            linewidth=1.5)

    if i == 2:
        ax2.text(x[1] + (i - 1) * width,
                 spearman_means[i] + spearman_std[i] + 0.005,
                 f"{spearman_means[i]:.3f}",
                 ha='center', va='bottom', fontsize=18, fontweight='bold')

ax2.set_ylabel("Spearman Score", fontproperties=arial_bold, fontsize=21, color='black')
ax2.set_ylim(0.75, 0.85)
ax2.tick_params(axis='y', labelcolor='black', direction='out', length=8, width=2)

for label in ax2.get_yticklabels():
    label.set_fontproperties(arial_font)
    label.set_fontsize(22)

ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontproperties=arial_font, fontsize=21)
ax1.tick_params(axis='x', direction='out', length=8, width=2, colors='black')

for label in ax1.get_xticklabels():
    label.set_fontproperties(arial_bold)
    label.set_fontsize(21)

for spine in ax1.spines.values():
    spine.set_edgecolor('black')
    spine.set_linewidth(3.0)

ax1.legend(loc='upper left', frameon=True, prop=arial_font, fontsize=12)

fig.tight_layout()
plt.savefig("comparison1.png", dpi=300, bbox_inches='tight')
plt.savefig("comparison1.svg", format='svg', bbox_inches='tight')
print("[INFO] Images saved to comparison1.png and comparison1.svg")
plt.show()
