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
    select_production_model,
    _evaluate,
    plot_feature_importance,
    plot_confusion_matrix,
    plot_roc_curve,
    save_model_bundle,
)


DATA_PATH      = ROOT / 'data'      / 'Telco-Customer-Churn.csv'
MODEL_PATH     = ROOT / 'models'    / 'churn_model.pkl'
FIG_DIR        = ROOT / 'reports'   / 'figures'
PROCESSED_DIR  = ROOT / 'data'      / 'processed'
BUSINESS_CSV   = PROCESSED_DIR      / 'cleaned_feature_engineered_dataset.csv'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# The 9 engineered features added by engineer_features(), for documentation
# and for the export-step printout. Kept here (not just inline) so this list
# has one source of truth if engineer_features() ever changes.
ENGINEERED_FEATURES = [
    'tenure_group', 'total_services', 'has_any_security', 'has_any_backup',
    'is_long_term', 'avg_monthly_charges', 'charges_per_service',
    'paperless_electronic', 'tenure_monthly_interaction',
]


def export_business_dataset(df_eng, output_path: Path) -> None:
    """
    Export a human-readable snapshot of the pipeline immediately AFTER
    cleaning + feature engineering, but BEFORE any training-only
    transformation (label encoding, scaling, SMOTE, train/test split).

    This is a presentation-layer export only — it does not mutate df_eng,
    and the caller continues the real pipeline on the original df_eng
    completely unaffected by anything this function does.

    The one transformation applied here: `clean_data()` already converted
    Churn to 0/1 int (required downstream by stratified splitting, SMOTE,
    and apply_encoders_and_scaler's `y = df.pop('Churn')` step). For this
    business-facing export we map it back to "Yes"/"No" so the file reads
    naturally for a non-technical audience. Every other categorical column
    (Contract, InternetService, PaymentMethod, OnlineSecurity, etc.) is
    still in its original string form at this point in the pipeline —
    LabelEncoder hasn't run yet — so no other column needs remapping.
    """
    export_df = df_eng.copy()
    export_df['Churn'] = export_df['Churn'].map({1: 'Yes', 0: 'No'})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False)

    print(f'[export] Business-facing dataset written (post-cleaning + '
          f'feature-engineering, pre-encoding/scaling/SMOTE):')
    print(f'           Rows    : {export_df.shape[0]:,}')
    print(f'           Columns : {export_df.shape[1]}')
    print(f'           Path    : {output_path}')


def main() -> None:
    print('\n' + '=' * 62)
    print('  CUSTOMER CHURN — TRAINING PIPELINE')
    print('=' * 62)

    print('\n[1/8] Loading data...')
    df_raw = load_data(str(DATA_PATH))

    print('[2/8] Cleaning data...')
    df_clean = clean_data(df_raw)

    print('[3/8] Running EDA...')
    run_full_eda(df_clean, output_dir=str(FIG_DIR))

    print('[4/8] Engineering features...')
    df_eng = engineer_features(df_clean)
    print(f'       Total columns: {df_eng.shape[1]}  (original 20 + 9 engineered)')

    # ── Business-facing export — happens HERE: after cleaning + feature
    # engineering, before any encoding/scaling/SMOTE/splitting. df_eng is
    # passed through untouched to step [6/8] immediately after.
    print('[5/8] Exporting business-facing dataset...')
    export_business_dataset(df_eng, BUSINESS_CSV)

    # fit on the full dataset so encoders see every possible label value
    print('[6/8] Fitting encoders and scaler...')
    encoders, scaler = fit_encoders_and_scaler(df_eng)

    X_all, y_all = apply_encoders_and_scaler(
        df_eng, encoders, scaler, has_target=True
    )
    print(f'       Feature matrix: {X_all.shape}  |  Target balance: '
          f'{y_all.value_counts().to_dict()}')

    print('[7/8] Splitting data and applying SMOTE...')
    X_train, X_test, y_train, y_test = split_with_smote(X_all, y_all)

    print('\n[8/8] Training and comparing models...')
    results_df, trained_models, raw_results = train_and_compare(
        X_train, X_test, y_train, y_test
    )

    print('\n── Baseline Model Comparison ──────────────────────────────')
    print(results_df.to_string())
    print('─' * 60)

    # VotingClassifier can't be tuned as a unit, so we always tune RF as
    # one additional candidate. Whether it actually GETS deployed is no
    # longer decided here — see select_production_model() below.
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

    # ── Automatic production-model selection ────────────────────────────
    # Replaces the previous hardcoded `best_model=tuned_rf` deployment.
    # Picks the best ROC-AUC among models that expose feature_importances_
    # (a real production requirement — see select_production_model's
    # docstring in model.py for the full rationale).
    print()
    deployed_name, deployed_model = select_production_model(
        trained_models, raw_results, require_feature_importances=True,
    )
    print(f'\n   Deployed model -> {deployed_name} '
          f'(ROC-AUC={raw_results[deployed_name]["roc_auc"]:.4f}, '
          f'Accuracy={raw_results[deployed_name]["accuracy"]:.4f})')

    print('\n[plots] Generating diagnostic figures...')
    plot_feature_importance(
        deployed_model, FEATURE_ORDER, top_n=15,
        save_path=str(FIG_DIR / '06_feature_importance.png'),
    )
    plot_confusion_matrix(
        deployed_model, X_test, y_test,
        save_path=str(FIG_DIR / '07_confusion_matrix.png'),
    )
    plot_roc_curve(
        deployed_model, X_test, y_test,
        save_path=str(FIG_DIR / '08_roc_curve.png'),
    )

    save_model_bundle(
        best_model      = deployed_model,
        all_models      = trained_models,
        encoders        = encoders,
        scaler          = scaler,
        feature_names   = FEATURE_ORDER,
        best_model_name = deployed_name,
        results         = raw_results,
        save_path       = str(MODEL_PATH),
    )

    print('\n' + '=' * 62)
    print('  TRAINING COMPLETE')
    print('=' * 62)
    print(f'  Deployed model -> {deployed_name}')
    print(f'  Model file     -> {MODEL_PATH}')
    print(f'  Business CSV   -> {BUSINESS_CSV}')
    print(f'  Plots          -> {FIG_DIR}/')
    print()


if __name__ == '__main__':
    main()
