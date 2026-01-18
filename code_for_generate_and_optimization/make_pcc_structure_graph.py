import re
import pandas as pd
import matplotlib.pyplot as plt

ddpm_raw_data = [
    {'Epoch': 100, 'Agreement': 0.916, 'Correlation': 0.848},
    {'Epoch': 200, 'Agreement': 0.912, 'Correlation': 0.906},
    {'Epoch': 300, 'Agreement': 0.923, 'Correlation': 0.905},
    {'Epoch': 400, 'Agreement': 0.952, 'Correlation': 0.909},
    {'Epoch': 500, 'Agreement': 0.968, 'Correlation': 0.922},
]

llm_data_str = """
[OK] embweight_0.03_Switche_epoch=100_6_mer_fre_cor=0.8847274968132962.csv -> agreement=0.967456
[OK] embweight_0.03_Switche_epoch=150_6_mer_fre_cor=0.8957309219046986.csv -> agreement=0.967953
[OK] embweight_0.03_Switche_epoch=200_6_mer_fre_cor=0.8958061104901068.csv -> agreement=0.967871
[OK] embweight_0.03_Switche_epoch=250_6_mer_fre_cor=0.8959467168928344.csv -> agreement=0.967919
[OK] embweight_0.03_Switche_epoch=300_6_mer_fre_cor=0.8929588302166849.csv -> agreement=0.967857
[OK] embweight_0.03_Switche_epoch=350_6_mer_fre_cor=0.896870954550831.csv -> agreement=0.967513
[OK] embweight_0.03_Switche_epoch=400_6_mer_fre_cor=0.9046369779172927.csv -> agreement=0.967949
[OK] embweight_0.03_Switche_epoch=450_6_mer_fre_cor=0.9025914072205775.csv -> agreement=0.968033
[OK] embweight_0.03_Switche_epoch=500_6_mer_fre_cor=0.904186568286493.csv -> agreement=0.967829
[OK] embweight_0.03_Switche_epoch=50_6_mer_fre_cor=0.8674663477534044.csv -> agreement=0.965722

[OK] embweight_0.05_Switche_epoch=100_6_mer_fre_cor=0.8855557226391741.csv -> agreement=0.967870
[OK] embweight_0.05_Switche_epoch=150_6_mer_fre_cor=0.9009749589757675.csv -> agreement=0.967829
[OK] embweight_0.05_Switche_epoch=200_6_mer_fre_cor=0.9119310442275429.csv -> agreement=0.967444
[OK] embweight_0.05_Switche_epoch=250_6_mer_fre_cor=0.9141237496424136.csv -> agreement=0.967413
[OK] embweight_0.05_Switche_epoch=300_6_mer_fre_cor=0.9140421793989953.csv -> agreement=0.967462
[OK] embweight_0.05_Switche_epoch=350_6_mer_fre_cor=0.9196945203129558.csv -> agreement=0.967642
[OK] embweight_0.05_Switche_epoch=400_6_mer_fre_cor=0.920743445217764.csv -> agreement=0.968153
[OK] embweight_0.05_Switche_epoch=450_6_mer_fre_cor=0.9132682814870817.csv -> agreement=0.968189
[OK] embweight_0.05_Switche_epoch=500_6_mer_fre_cor=0.9173824985337474.csv -> agreement=0.968200
[OK] embweight_0.05_Switche_epoch=50_6_mer_fre_cor=0.8995947679386198.csv -> agreement=0.967449

[OK] embweight_0.1(3)_Switche_epoch=100_6_mer_fre_cor=0.876471787357951.csv -> agreement=0.967962
[OK] embweight_0.1(3)_Switche_epoch=150_6_mer_fre_cor=0.8963284346169391.csv -> agreement=0.968150
[OK] embweight_0.1(3)_Switche_epoch=200_6_mer_fre_cor=0.7173103930840445.csv -> agreement=0.954534
[OK] embweight_0.1(3)_Switche_epoch=250_6_mer_fre_cor=0.7825819951935682.csv -> agreement=0.897423
[OK] embweight_0.1(3)_Switche_epoch=300_6_mer_fre_cor=0.8222449244690777.csv -> agreement=0.865411
[OK] embweight_0.1(3)_Switche_epoch=350_6_mer_fre_cor=0.7840919912327093.csv -> agreement=0.858620
[OK] embweight_0.1(3)_Switche_epoch=400_6_mer_fre_cor=0.8408635671498529.csv -> agreement=0.874116
[OK] embweight_0.1(3)_Switche_epoch=450_6_mer_fre_cor=0.8047800190127936.csv -> agreement=0.865475
[OK] embweight_0.1(3)_Switche_epoch=500_6_mer_fre_cor=0.785757817395356.csv -> agreement=0.882629
[OK] embweight_0.1(3)_Switche_epoch=50_6_mer_fre_cor=0.8855057903675256.csv -> agreement=0.966908
"""


def parse_data(data_str):
    parsed_dict = {}
    lines = data_str.strip().split('\n')
    for line in lines:
        weight_match = re.search(r'embweight_([\d\.\(\)]+)_', line)
        epoch_match = re.search(r'epoch=(\d+)', line)
        cor_match = re.search(r'cor=([\d\.]+)\.csv', line)
        agree_match = re.search(r'agreement=([\d\.]+)', line)

        if weight_match and epoch_match and cor_match and agree_match:
            raw_weight = weight_match.group(1)
            weight_label = raw_weight.replace('(3)', '')

            if weight_label not in parsed_dict:
                parsed_dict[weight_label] = []

            parsed_dict[weight_label].append({
                'Epoch': int(epoch_match.group(1)),
                'Correlation': float(cor_match.group(1)),
                'Agreement': float(agree_match.group(1))
            })

    dfs = {}
    for w, data in parsed_dict.items():
        df = pd.DataFrame(data).sort_values('Epoch')
        dfs[w] = df
    return dfs

llm_dfs = parse_data(llm_data_str)

ddpm_df = pd.DataFrame(ddpm_raw_data).sort_values('Epoch') if ddpm_raw_data else None


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 16  

plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

styles = {
    'DDPM': {
        'color': 'black',
        'marker': 'o',
        'linestyle': '--',
        'label': 'DDPM Baseline'
    },
    '0.03': {
        'color': '#A8C5E2',
        'marker': '^',
        'linestyle': '-',
        'label': 'DDPM + LLM (w=0.03)'
    },
    '0.05': {
        'color': '#C7E9C0',
        'marker': 's',
        'linestyle': '-',
        'label': 'DDPM + LLM (w=0.05)'
    },
    '0.1': {
        'color': '#F28F6B',
        'marker': 'D',
        'linestyle': '-',
        'label': 'DDPM + LLM (w=0.1)'
    },
}


def plot_metric(metric_name, y_label, output_filename, title):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    if ddpm_df is not None and not ddpm_df.empty:
        ax.plot(
            ddpm_df['Epoch'],
            ddpm_df[metric_name],
            color=styles['DDPM']['color'],
            marker=styles['DDPM']['marker'],
            linestyle=styles['DDPM']['linestyle'],
            linewidth=3.0,
            markersize=9,
            label=styles['DDPM']['label']
        )
    else:
        print(f"Note: DDPM Baseline data is empty, the curve will not be included in {title}")

    sorted_weights = sorted(llm_dfs.keys(), key=lambda x: float(x))

    for w in sorted_weights:
        df = llm_dfs[w]
        style = styles.get(w, {'color': 'gray', 'marker': 'x', 'linestyle': ':'})
        ax.plot(
            df['Epoch'],
            df[metric_name],
            color=style.get('color'),
            marker=style.get('marker'),
            linestyle=style.get('linestyle', '-'),
            linewidth=3.0,
            markersize=9,
            label=style.get('label', f'Weight {w}')
        )


    ax.set_xlabel('Epoch', fontsize=20, fontweight='bold', labelpad=12)
    ax.set_ylabel(y_label, fontsize=20, fontweight='bold', labelpad=12)

    ax.tick_params(axis='both', which='major', width=2.0, length=6, labelsize=18)

    ax.grid(True, linestyle='--', alpha=0.5)

    ax.legend(fontsize=14, frameon=True, framealpha=0.9, edgecolor='gray', loc='best')

    plt.title(title, fontsize=22, fontweight='bold', pad=15)
    plt.tight_layout()

    plt.savefig(output_filename + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_filename + '.svg', format='svg', bbox_inches='tight')
    print(f"figure saved in: {output_filename}.png")
    plt.show()


plot_metric(
    metric_name='Agreement',
    y_label='Structural Agreement',
    output_filename='Comparison_Agreement',
    title='Structural Agreement Comparison'
)

plot_metric(
    metric_name='Correlation',
    y_label='PCC (6-mer Elements)',
    output_filename='Comparison_PCC',
    title='PCC Comparison'
)
