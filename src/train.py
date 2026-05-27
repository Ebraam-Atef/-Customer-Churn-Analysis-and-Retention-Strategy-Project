import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# make src/ importable regardless of where the script is called from
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from preprocessing import (
    load_data,
    clean_data,
    engineer_features,
    fit_encoders_and_scaler,
    apply_encoders_and_scaler,
    FEATURE_ORDER,
)
from eda import run_full_eda
from model import (
    split_with_smote,
    train_and_compare,
    tune_random_forest,
    _evaluate,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    save_model_bundle,
)


DATA_PATH  = ROOT / 'data'    / 'Telco-Customer-Churn.csv'
MODEL_PATH = ROOT / 'models'  / 'churn_model.pkl'
FIG_DIR    = ROOT / 'reports' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print('\n' + '=' * 62)
    print('  CUSTOMER CHURN — TRAINING PIPELINE')
    print('=' * 62)

    print('\n[1/7] Loading data...')
    df_raw = load_data(str(DATA_PATH))

    print('[2/7] Cleaning data...')
    df_clean = clean_data(df_raw)

    print('[3/7] Running EDA...')
    run_full_eda(df_clean, output_dir=str(FIG_DIR))

    print('[4/7] Engineering features...')
    df_eng = engineer_features(df_clean)
    print(f'       Total columns: {df_eng.shape[1]}  (original 20 + 9 engineered)')

    # fit on the full dataset so encoders see every possible label value
    print('[5/7] Fitting encoders and scaler...')
    encoders, scaler = fit_encoders_and_scaler(df_eng)

    X_all, y_all = apply_encoders_and_scaler(
        df_eng, encoders, scaler, has_target=True
    )
    print(f'       Feature matrix: {X_all.shape}  |  Target balance: '
          f'{y_all.value_counts().to_dict()}')

    print('[6/7] Splitting data and applying SMOTE...')
    X_train, X_test, y_train, y_test = split_with_smote(X_all, y_all)

    print('\n[7/7] Training and comparing models...')
    results_df, trained_models, raw_results = train_and_compare(
        X_train, X_test, y_train, y_test
    )

    print('\n── Baseline Model Comparison ──────────────────────────────')
    print(results_df.to_string())
    print('─' * 60)

    # VotingClassifier can't be tuned as a unit, so we always tune RF
    best_tunable = 'Random Forest'
    print(f'\n[+] Tuning {best_tunable} with RandomizedSearchCV (30 iterations)...')
    tuned_rf = tune_random_forest(X_train, y_train, n_iter=30)

    tuned_metrics = _evaluate(tuned_rf, X_test, y_test, f'{best_tunable} (Tuned)')
    print('\n── Tuned Model on Held-Out Test Set ───────────────────────')
    for k, v in tuned_metrics.items():
        if k != 'Model':
            print(f'   {k:<22}: {v}')
    print('─' * 60)

    tuned_name = f'{best_tunable} (Tuned)'
    raw_results[tuned_name] = {
        'accuracy': tuned_metrics['Accuracy'],
        'roc_auc' : tuned_metrics['ROC-AUC'],
    }
    trained_models[tuned_name] = tuned_rf

    overall_best = max(raw_results, key=lambda k: raw_results[k]['roc_auc'])
    print(f'\n   Best ROC-AUC across all models: {overall_best} '
          f"({raw_results[overall_best]['roc_auc']:.4f})")
    print(f'   Model saved to PKL: {tuned_name}')

    print('\n[plots] Generating diagnostic figures...')
    plot_feature_importance(
        tuned_rf, FEATURE_ORDER, top_n=15,
        save_path=str(FIG_DIR / '06_feature_importance.png'),
    )
    plot_confusion_matrix(
        tuned_rf, X_test, y_test,
        save_path=str(FIG_DIR / '07_confusion_matrix.png'),
    )
    plot_roc_curve(
        tuned_rf, X_test, y_test,
        save_path=str(FIG_DIR / '08_roc_curve.png'),
    )

    save_model_bundle(
        best_model      = tuned_rf,
        all_models      = trained_models,
        encoders        = encoders,
        scaler          = scaler,
        feature_names   = FEATURE_ORDER,
        best_model_name = tuned_name,
        results         = raw_results,
        save_path       = str(MODEL_PATH),
    )

    print('\n' + '=' * 62)
    print('  TRAINING COMPLETE')
    print('=' * 62)
    print(f'  Model  -> {MODEL_PATH}')
    print(f'  Plots  -> {FIG_DIR}/')
    print()


if __name__ == '__main__':
    main()
