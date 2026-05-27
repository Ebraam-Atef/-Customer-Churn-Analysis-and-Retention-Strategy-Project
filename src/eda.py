import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)

C_NO  = '#2196F3'
C_YES = '#F44336'


def _save_fig(fig: plt.Figure, path: str | None) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_churn_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    """Pie + bar showing class imbalance (~26.5% churn)."""
    counts = df['Churn'].map({0: 'No Churn', 1: 'Churn'}).value_counts()
    labels = ['No Churn', 'Churn']
    colors = [C_NO, C_YES]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('Class Imbalance · Insight: 26.5% churn rate — SMOTE required',
                 fontsize=10, color='#555')

    axes[0].pie(
        [counts.get('No Churn', 0), counts.get('Churn', 0)],
        labels=labels, autopct='%1.1f%%', startangle=90,
        colors=colors, wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    )
    axes[0].set_title('Churn Distribution', fontweight='bold')

    vals = [counts.get('No Churn', 0), counts.get('Churn', 0)]
    bars = axes[1].bar(labels, vals, color=colors, width=0.45, edgecolor='white')
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                     f'{v:,}', ha='center', fontweight='bold', fontsize=11)
    axes[1].set_ylabel('Customer Count')
    axes[1].set_title('Count by Churn Status', fontweight='bold')

    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_churn_by_contract(df: pd.DataFrame, save_path: str = None) -> None:
    """Grouped bar: churn % per contract type."""
    ct = (
        df.groupby('Contract')['Churn']
        .value_counts(normalize=True)
        .unstack()
        .rename(columns={0: 'No Churn', 1: 'Churn'})
        * 100
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind='bar', ax=ax, color=[C_NO, C_YES], edgecolor='white',
            width=0.6, rot=0)
    ax.set_title(
        'Churn Rate by Contract Type\n'
        'Insight: Month-to-month customers churn at 42.7% vs 2.8% for Two-year',
        fontweight='bold',
    )
    ax.set_xlabel('')
    ax.set_ylabel('Percentage (%)')
    ax.legend(['No Churn', 'Churn'])
    for patch in ax.patches:
        h = patch.get_height()
        if h > 1:
            ax.annotate(f'{h:.1f}%',
                        xy=(patch.get_x() + patch.get_width() / 2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', fontsize=9)
    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_churn_by_tenure(df: pd.DataFrame, save_path: str = None) -> None:
    """Histogram + tenure-group bar rates."""
    df2 = df.copy()
    df2['Churn_Label'] = df2['Churn'].map({0: 'No Churn', 1: 'Churn'})
    df2['tenure_group'] = pd.cut(df2['tenure'], bins=[0, 12, 24, 48, 72],
                                  labels=['0-12m', '13-24m', '25-48m', '49-72m'],
                                  include_lowest=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for label, color in [('No Churn', C_NO), ('Churn', C_YES)]:
        axes[0].hist(df2[df2['Churn_Label'] == label]['tenure'],
                     bins=30, alpha=0.55, color=color, label=label, density=True)
    axes[0].set_title('Tenure Distribution by Churn', fontweight='bold')
    axes[0].set_xlabel('Tenure (months)')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    rates = (df2.groupby('tenure_group', observed=False)['Churn'].mean() * 100)
    grad_colors = ['#F44336', '#FF7043', '#FFA726', '#66BB6A']
    bars = axes[1].bar(rates.index, rates.values, color=grad_colors,
                       edgecolor='white', width=0.6)
    for bar, v in zip(bars, rates.values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{v:.1f}%', ha='center', fontsize=10, fontweight='bold')
    axes[1].set_title(
        'Churn Rate by Tenure Group\n'
        'Insight: 47.4% churn in first 12 months — improve onboarding',
        fontweight='bold',
    )
    axes[1].set_xlabel('Tenure Group')
    axes[1].set_ylabel('Churn Rate (%)')
    axes[1].set_ylim(0, 60)

    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_churn_by_services(df: pd.DataFrame, save_path: str = None) -> None:
    """Heatmap: churn % when service subscribed vs not."""
    service_cols = [
        'OnlineSecurity', 'TechSupport', 'OnlineBackup',
        'DeviceProtection', 'StreamingTV', 'StreamingMovies',
    ]
    rates = {col: df.groupby(col)['Churn'].mean() * 100 for col in service_cols}
    rate_df = pd.DataFrame(rates).T
    rate_df = rate_df[[c for c in ['Yes', 'No'] if c in rate_df.columns]]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(rate_df, annot=True, fmt='.1f', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax, cbar_kws={'label': 'Churn %'})
    ax.set_title(
        'Churn Rate (%) by Service Subscription\n'
        'Insight: No OnlineSecurity = 41.8% vs 14.6% with it',
        fontweight='bold',
    )
    plt.tight_layout()
    _save_fig(fig, save_path)


def plot_churn_by_charges(df: pd.DataFrame, save_path: str = None) -> None:
    """Box plots: monthly charges by churn status and internet type."""
    df2 = df.copy()
    df2['Churn_Label'] = df2['Churn'].map({0: 'No Churn', 1: 'Churn'})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(data=df2, x='Churn_Label', y='MonthlyCharges',
                palette=[C_NO, C_YES], ax=axes[0], width=0.5,
                hue='Churn_Label', legend=False)
    axes[0].set_title('Monthly Charges vs Churn\n'
                      'Insight: Churned customers pay ~$15/mo more',
                      fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Monthly Charges ($)')

    sns.boxplot(data=df2, x='InternetService', y='MonthlyCharges',
                hue='Churn_Label', palette=[C_NO, C_YES],
                ax=axes[1], width=0.6)
    axes[1].set_title('Charges by Internet Service & Churn\n'
                      'Insight: Fiber optic is the highest-risk, highest-cost segment',
                      fontweight='bold')
    axes[1].set_xlabel('Internet Service')
    axes[1].set_ylabel('Monthly Charges ($)')

    plt.tight_layout()
    _save_fig(fig, save_path)


def print_eda_summary(df: pd.DataFrame) -> None:
    print('\n' + '═' * 58)
    print('  EDA SUMMARY')
    print('═' * 58)
    print(f'  Rows / Columns      : {df.shape[0]:,} / {df.shape[1]}')
    print(f'  Churn rate          : {df["Churn"].mean():.2%}')
    print(f'  Avg tenure          : {df["tenure"].mean():.1f} months')
    print(f'  Avg monthly charges : ${df["MonthlyCharges"].mean():.2f}')
    print(f'  Missing values      : {df.isnull().sum().sum()}')
    print()
    print('  Top univariate churn drivers:')
    for label, mask in [
        ('Month-to-month contract', df['Contract'] == 'Month-to-month'),
        ('Fiber optic internet',    df['InternetService'] == 'Fiber optic'),
        ('No online security',      df['OnlineSecurity'] == 'No'),
        ('Tenure ≤ 12 months',      df['tenure'] <= 12),
        ('Electronic check payer',  df['PaymentMethod'] == 'Electronic check'),
    ]:
        print(f'  • {label:<28}: {df[mask]["Churn"].mean():.2%} churn rate')
    print('═' * 58 + '\n')


def run_full_eda(df: pd.DataFrame, output_dir: str = 'reports/figures') -> None:
    """Run all EDA plots and print console summary."""
    print_eda_summary(df)
    out = Path(output_dir)
    plot_churn_distribution(df, save_path=str(out / '01_churn_distribution.png'))
    plot_churn_by_contract(df,  save_path=str(out / '02_churn_by_contract.png'))
    plot_churn_by_tenure(df,    save_path=str(out / '03_churn_by_tenure.png'))
    plot_churn_by_services(df,  save_path=str(out / '04_churn_by_services.png'))
    plot_churn_by_charges(df,   save_path=str(out / '05_churn_by_charges.png'))
    print(f'[eda] All 5 figures saved to {output_dir}/')
