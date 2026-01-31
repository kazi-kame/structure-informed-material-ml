"""Materials TaML - Structure-aware Virtual Lab"""

import os
import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pymatgen.core import Structure

from data_loader import load_data, validate_required_columns
from structure_feature_engine import StructureFeatureEngine
from model_manager import TaMLModel
from query_engine import query_materials
from predict_from_structure import predict_from_structure_file
from alloy_heuristic import parse_alloy_formula, build_solid_solution


def load_structures_from_df(df):
    """Load structures from POSCAR or CIF format in dataframe"""
    structures = []
    failed_count = 0
    
    for i, row in df.iterrows():
        parsed = False
        
        if pd.notna(row.get("cif")):
            try:
                structures.append(Structure.from_str(row["cif"], fmt="cif"))
                parsed = True
            except Exception:
                pass
        
        if not parsed and pd.notna(row.get("poscar")):
            try:
                structures.append(Structure.from_str(row["poscar"], fmt="poscar"))
                parsed = True
            except Exception:
                pass
        
        if not parsed:
            failed_count += 1
            if failed_count <= 5:
                print(f"Warning: Failed to parse structure at row {i}")
    
    if failed_count > 0:
        print(f"Warning: {failed_count} structures failed to parse")
    
    return structures


def setup_model(df, target_col, n_components, test_size=0.2):
    """Train structure-aware model with feature engineering and GP regression"""
    print("\n" + "=" * 60)
    print("TRAINING STRUCTURE-AWARE MODEL")
    print("=" * 60)

    validate_required_columns(df, ["cif", target_col], "training")

    y_raw = df[target_col].values
    valid = (y_raw > 0) & np.isfinite(y_raw)
    df = df[valid].reset_index(drop=True)
    y_raw = y_raw[valid]

    y_log = np.log10(y_raw)
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y_log.reshape(-1, 1)).ravel()

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=42)

    structures = load_structures_from_df(df)
    s_train = [structures[i] for i in train_idx]
    s_test = [structures[i] for i in test_idx]

    fe = StructureFeatureEngine(n_components=n_components)
    X_train = fe.fit_transform(s_train)
    X_test = fe.transform(s_test)

    model = TaMLModel(kernel_type="matern", noise_level=0.1)
    model.fit(X_train, y_scaled[train_idx], np.zeros(len(train_idx)))

    r2 = model.score(X_test, y_scaled[test_idx], np.zeros(len(test_idx)))
    
    print(f"\nTest R-squared: {r2:.3f}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")

    return model, fe, scaler_y


def main():
    # Configuration
    CSV_FILE = r"C:\Users\Kazi\Downloads\materials_taml\data\materials_matminer.csv"
    TARGET_COL = "last phdos peak"  # Will be selected interactively

    print("\n" + "=" * 60)
    print("MATERIALS TaML - VIRTUAL LAB")
    print("=" * 60)

    # Check if file exists
    if not os.path.exists(CSV_FILE):
        print(f"\nError: Data file not found: {CSV_FILE}")
        print("Please run: python load_matminer_data.py first")
        return

    # Load data
    df_full = pd.read_csv(CSV_FILE)
    print(f"\nLoaded {len(df_full)} materials from dataset")
    print(f"\nAvailable columns: {', '.join(df_full.columns)}")
    
    # Check property availability
    print(f"\nProperty availability:")
    for col in df_full.columns:
        if col not in ['formula', 'cif', 'poscar', 'structure']:
            non_null = df_full[col].notna().sum()
            if non_null > 0:
                print(f"  {col}: {non_null}/{len(df_full)} materials")
    
    # Select target property
    print(f"\n" + "=" * 60)
    target_options = [c for c in df_full.columns 
                     if c not in ['formula', 'cif', 'poscar', 'structure'] 
                     and df_full[c].notna().sum() > 100]
    
    if not target_options:
        print("Error: No suitable target properties found")
        return
    
    print("Available target properties:")
    for i, opt in enumerate(target_options, 1):
        print(f"  {i}. {opt}")
    
    choice = input(f"\nSelect property to predict (1-{len(target_options)}, default=1): ").strip()
    
    if choice and choice.isdigit() and 1 <= int(choice) <= len(target_options):
        TARGET_COL = target_options[int(choice) - 1]
    else:
        TARGET_COL = target_options[0]
    
    print(f"\nSelected target: {TARGET_COL}")
    
    # Filter to materials with target property
    df = df_full[df_full[TARGET_COL].notna()].reset_index(drop=True)
    print(f"Materials with {TARGET_COL} data: {len(df)}")
    
    # Limit dataset size for faster training
    max_samples = 500
    if len(df) > max_samples:
        print(f"\nUsing {max_samples} materials for faster training")
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    
    # Train model
    model, fe, scaler_y = setup_model(df, TARGET_COL, n_components=10)

    # Determine units
    if "phdos" in TARGET_COL.lower() or "freq" in TARGET_COL.lower():
        unit_name = "THz"
    elif "energy" in TARGET_COL.lower():
        unit_name = "eV"
    else:
        unit_name = "GPa"

    # Interactive loop
    while True:
        print("\n" + "=" * 60)
        print("OPTIONS:")
        print("  3. Virtual Lab (CIF / POSCAR file)")
        print("  4. Alloy Heuristic (formula + assumed structure)")
        print("  q. Quit")
        print("=" * 60)

        choice = input("Your choice: ").strip().lower()

        if choice in ("q", "quit", "exit"):
            print("\nExiting.")
            break

        if choice == "3":
            path = input("Enter path to CIF or POSCAR file: ").strip()
            
            if not os.path.exists(path):
                print(f"Error: File not found: {path}")
                continue
            
            result = predict_from_structure_file(path, model, fe, scaler_y)

            print("\n" + "=" * 60)
            print("PREDICTION RESULTS")
            print("=" * 60)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Property: {TARGET_COL}")
                print(f"Prediction: {result['prediction']:.2f} {unit_name}")
                print(f"Uncertainty: {result['uncertainty']:.2f} {unit_name}")
                print(f"Confidence: {result['confidence']}")
            print("=" * 60)
            continue

        if choice == "4":
            formula = input("Enter alloy formula (e.g. Fe70Ni30): ").strip()
            proto = input("Assumed structure (FCC / BCC): ").strip().upper()

            if proto not in ['FCC', 'BCC']:
                print("Error: Structure must be FCC or BCC")
                continue

            try:
                alloy = parse_alloy_formula(formula)
                structure = build_solid_solution(alloy, prototype=proto)
            except Exception as e:
                print(f"Error: {e}")
                continue

            X = fe.transform([structure])
            z_pred, z_std = model.predict(X, np.zeros(1))

            log_val = scaler_y.inverse_transform(z_pred.reshape(-1, 1))[0, 0]
            value = 10 ** log_val
            uncertainty = 2.0 * z_std[0] * value

            print("\n" + "=" * 60)
            print("ALLOY HEURISTIC RESULTS")
            print("=" * 60)
            print(f"Formula: {formula}")
            print(f"Assumed structure: {proto} solid solution")
            print(f"Property: {TARGET_COL}")
            print(f"Prediction: {value:.2f} {unit_name}")
            print(f"Uncertainty: {uncertainty:.2f} {unit_name}")
            print(f"Confidence: LOW (heuristic approximation)")
            print("=" * 60)
            continue

        print("Invalid option. Please select 3, 4, or q.")


if __name__ == "__main__":
    main()