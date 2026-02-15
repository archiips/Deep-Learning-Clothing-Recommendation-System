"""
Master Script to Run All Phase 3 Tasks
Executes all data preparation and feature engineering steps in the correct order
"""

import subprocess
import sys
import time

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        elapsed = time.time() - start_time
        print(f"\n✅ {description} completed in {elapsed:.2f}s")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        elapsed = time.time() - start_time
        print(f"Failed after {elapsed:.2f}s")
        return False

def main():
    """Run all Phase 3 scripts in sequence."""
    print("="*80)
    print("PHASE 3: DATA PREPARATION & FEATURE ENGINEERING")
    print("="*80)
    print("\nThis script will run all Phase 3 tasks in the correct order:")
    print("  1. Complete Data Cleaning")
    print("  2. Complete EDA")
    print("  3. Feature Engineering")
    print("  4. Train/Test Split")
    print("  5. PyTorch Preprocessing")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()

    overall_start = time.time()

    # Task 3.1: Complete Data Cleaning
    if not run_script('complete_data_cleaning.py', 'Task 3.1: Complete Data Cleaning'):
        print("\n❌ Phase 3 failed at data cleaning step")
        return False

    # Task 3.3: Complete EDA
    if not run_script('complete_eda.py', 'Task 3.3: Complete EDA'):
        print("\n❌ Phase 3 failed at EDA step")
        return False

    # Task 3.2: Feature Engineering
    if not run_script('feature_engineering.py', 'Task 3.2: Feature Engineering'):
        print("\n❌ Phase 3 failed at feature engineering step")
        return False

    # Task 3.4: Train/Test Split
    if not run_script('train_test_split.py', 'Task 3.4: Train/Test Split'):
        print("\n❌ Phase 3 failed at train/test split step")
        return False

    # Task 3.5: PyTorch Preprocessing
    if not run_script('pytorch_preprocessing.py', 'Task 3.5: PyTorch Preprocessing'):
        print("\n❌ Phase 3 failed at PyTorch preprocessing step")
        return False

    overall_elapsed = time.time() - overall_start

    print("\n" + "="*80)
    print("PHASE 3 COMPLETE!")
    print("="*80)
    print(f"\nTotal time: {overall_elapsed:.2f}s ({overall_elapsed/60:.2f} minutes)")
    print("\n✅ All Phase 3 tasks completed successfully!")
    print("\nGenerated outputs:")
    print("  📁 Cleaned datasets")
    print("  📊 EDA charts (in charts/ folder)")
    print("  🔢 Feature engineering outputs")
    print("  📂 Train/test splits")
    print("  🔥 PyTorch-ready datasets and encoders")
    print("\nReady to proceed to Phase 4: Model Development!")

    return True

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
