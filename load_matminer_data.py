import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("MATMINER INSTANT DATA LOADER")
print("=" * 80)

# Try to import
try:
    from matminer.datasets import load_dataset, get_available_datasets
    print("\n Matminer installed")
except ImportError:
    print("\n Matminer not installed!")
    print("\nInstall with: pip install matminer")
    exit(1)

print("\nAvailable datasets with elastic properties:")
print("-" * 80)

datasets = get_available_datasets()
elastic_datasets = [d for d in datasets if 'elastic' in d.lower() or 'phonon' in d.lower()]

for ds in elastic_datasets:
    print(f"  • {ds}")

print("\n" + "=" * 80)
print("LOADING DATASET")
print("=" * 80)

# Load a dataset with elastic properties
# First time: downloads (~100 MB), then cached forever
print("\nLoading 'matbench_phonons' dataset...")
print("(First time may take 1-2 minutes to download, then instant)")

try:
    df = load_dataset("matbench_phonons")
    print(f" Loaded {len(df)} materials!")
    
    print("\n" + "=" * 80)
    print("DATASET INFO")
    print("=" * 80)
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    # Check what properties we have
    print(f"\n" + "=" * 80)
    print("AVAILABLE PROPERTIES")
    print("=" * 80)
    
    for col in df.columns:
        if col != 'structure':
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null}/{len(df)} materials")
    
    # Try to extract structures
    if 'structure' in df.columns:
        print("\n Structures are available!")
        print(f"  Format: pymatgen Structure objects")
        
        # Sample structure
        sample_struct = df['structure'].iloc[0]
        print(f"\nSample structure:")
        print(f"  Formula: {sample_struct.composition.reduced_formula}")
        print(f"  # Atoms: {len(sample_struct)}")
        print(f"  Lattice: {sample_struct.lattice.abc}")
    
    print("\n" + "=" * 80)
    print("SAVING DATA")
    print("=" * 80)
    
    output_dir = Path(r"C:\Users\Kazi\Downloads\materials_taml\data")
    output_file = output_dir / "materials_matminer.csv"
    structures_dir = output_dir / "structures_matminer"
    
    structures_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare dataframe for saving
    df_export = df.copy()
    
    # If structures exist, save them as CIF files
    if 'structure' in df.columns:
        print("\nSaving CIF files...")
        cif_data = []
        
        for i, row in df.iterrows():
            try:
                struct = row['structure']
                formula = struct.composition.reduced_formula
                
                # Save CIF file
                cif_filename = f"{formula}_{i}.cif"
                cif_path = structures_dir / cif_filename
                struct.to(filename=str(cif_path), fmt='cif')
                
                # Read CIF content
                with open(cif_path, 'r') as f:
                    cif_content = f.read()
                
                cif_data.append(cif_content)
                
                if (i + 1) % 100 == 0:
                    print(f"  Saved {i+1}/{len(df)} structures...")
                    
            except Exception as e:
                cif_data.append('')
                print(f"   Error with structure {i}: {str(e)[:50]}")
        
        df_export['cif'] = cif_data
        df_export = df_export.drop('structure', axis=1)
    
    # Save CSV
    df_export.to_csv(output_file, index=False)
    
    print(f"\n Saved to: {output_file}")
    print(f" CIF files in: {structures_dir}")
    
    print("\n" + "=" * 80)
    print("=" * 80)
    print(f"\nYou now have {len(df)} materials instead of 8!")
    print("\nNext steps:")
    print("  1. Update main.py to use new CSV file")
    print("  2. Run: python main.py")
    print("  3. Model will be much more powerful!")
    
except Exception as e:
    print(f"\n Error: {e}")
    print("\nTrying alternative dataset...")
    
    # Try another dataset
    try:
        print("\nLoading 'elastic_tensor_2015' dataset...")
        df = load_dataset("elastic_tensor_2015")
        print(f" Loaded {len(df)} materials with elastic tensors!")
        print(f"\nColumns: {list(df.columns)}")
        
    except Exception as e2:
        print(f" Also failed: {e2}")
        print("\nPlease check internet connection and try again.")

print("\n" + "=" * 80)