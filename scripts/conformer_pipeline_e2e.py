#!/usr/bin/env python3
"""
End-to-end conformer generation pipeline for ShEPhERD training data (v2 with validation).

This script demonstrates the full pipeline from SMILES to training-ready data:
1. RDKit ETKDG embedding
2. MMFF94 optimization
3. Conformer ensemble generation & clustering
4. xTB optimization in water (ALPB implicit solvent)
5. Final clustering and charge extraction
6. Serialization to pickle format

=============================================================================
USAGE EXAMPLES
=============================================================================

1. Generate conformers for a single molecule:
   python conformer_pipeline_e2e_v2.py --smiles "CCO" --profile

2. Generate conformers for a drug-like molecule with output:
   python conformer_pipeline_e2e_v2.py \\
       --smiles "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1" \\
       --num-confs 100 \\
       --output my_conformers.pkl \\
       --profile

3. VALIDATE the pipeline against reference data:
   python conformer_pipeline_e2e_v2.py --validate

4. Validate with more molecules (slower but more thorough):
   python conformer_pipeline_e2e_v2.py --validate --validate-n 10

=============================================================================
VALIDATION GUIDE
=============================================================================

The --validate flag tests the pipeline against pre-computed reference data in:
  data/conformers/moses_aq/example_molblock_charges.pkl

WHAT THE VALIDATION CHECKS:
---------------------------
1. Charge RMSE < 0.01 (default tolerance)
   - Measures average deviation in partial charges per atom
   - Small differences (0.001-0.005) are normal due to different conformers
   - Large differences (>0.01) suggest xTB or solvent settings are wrong

2. Charge Correlation >= 0.999 (default tolerance)
   - Measures whether charge patterns match (which atoms are +/- charged)
   - Should be very close to 1.0 if xTB is working correctly
   - Low correlation suggests fundamental issues with the pipeline

3. Charge Sum ~= 0 for neutral molecules
   - xTB charges should sum to the formal charge (0 for neutral)
   - Non-zero sums indicate charge extraction bugs

WHY SMALL DIFFERENCES ARE EXPECTED:
-----------------------------------
- ETKDG uses random seeds -> different initial conformers
- xTB optimization may converge to different local minima
- Different num_confs or clustering thresholds affect which conformers survive
- The reference data was generated with specific (unknown) random seeds

WHAT TO LOOK FOR IN FAILURES:
-----------------------------
- If charge correlation is low (<0.99): Check xTB installation, solvent flag
- If charge RMSE is high (>0.05): May indicate different xTB version or settings
- If charges don't sum to ~0: Check charge extraction from xTB output files
- If validation fails on ALL molecules: Likely a systematic issue (xTB not found, etc.)
- If validation fails on SOME molecules: May be edge cases or stereochemistry issues

MANUAL VALIDATION:
------------------
You can also manually inspect generated conformers:

    import pickle
    from rdkit import Chem

    with open('my_conformers.pkl', 'rb') as f:
        data = pickle.load(f)

    molblock, charges = data[0]
    mol = Chem.MolFromMolBlock(molblock, removeHs=False)

    # Check charges sum to formal charge
    print(f"Charge sum: {sum(charges):.6f}")

    # Visualize
    from rdkit.Chem import Draw
    Draw.MolToImage(mol)

=============================================================================
"""

import argparse
import pickle
import time
import sys
from pathlib import Path
from functools import wraps
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem import rdMolAlign

# Import shepherd conformer generation utilities
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from shepherd.shepherd_score_utils.conformer_generation import (
    embed_conformer_from_smiles,
    generate_conformer_ensemble,
    cluster_conformers_butina,
    optimize_conformer_with_xtb,
    optimize_conformer_ensemble_with_xtb,
    generate_opt_conformers_xtb,
)


# =============================================================================
# VALIDATION TOLERANCES
# =============================================================================

@dataclass
class ValidationTolerances:
    """Tolerances for validation tests."""
    max_charge_rmse: float = 0.01      # Maximum allowed charge RMSE
    min_charge_corr: float = 0.999     # Minimum allowed charge correlation
    max_charge_sum_deviation: float = 0.01  # Max deviation from expected charge sum

    def __str__(self):
        return (
            f"  max_charge_rmse: {self.max_charge_rmse}\n"
            f"  min_charge_corr: {self.min_charge_corr}\n"
            f"  max_charge_sum_deviation: {self.max_charge_sum_deviation}"
        )


DEFAULT_TOLERANCES = ValidationTolerances()


# =============================================================================
# PROFILING UTILITIES
# =============================================================================

class Profiler:
    """Simple profiler to track time spent in each stage."""

    def __init__(self):
        self.timings = {}
        self.counts = {}

    def record(self, name: str, duration: float):
        if name not in self.timings:
            self.timings[name] = 0.0
            self.counts[name] = 0
        self.timings[name] += duration
        self.counts[name] += 1

    def reset(self):
        self.timings = {}
        self.counts = {}

    def report(self):
        total = sum(self.timings.values())
        print("\n" + "="*70)
        print("PROFILING REPORT")
        print("="*70)
        print(f"{'Stage':<45} {'Time (s)':<12} {'%':<8} {'Count'}")
        print("-"*70)
        for name, t in sorted(self.timings.items(), key=lambda x: -x[1]):
            pct = (t / total * 100) if total > 0 else 0
            print(f"{name:<45} {t:<12.3f} {pct:<8.1f} {self.counts[name]}")
        print("-"*70)
        print(f"{'TOTAL':<45} {total:<12.3f}")
        print("="*70)


profiler = Profiler()


def timed(name: str):
    """Decorator to time a function."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            duration = time.perf_counter() - start
            profiler.record(name, duration)
            return result
        return wrapper
    return decorator


# =============================================================================
# PIPELINE STEPS (with profiling)
# =============================================================================

@timed("1. SMILES -> RDKit mol + ETKDG embed")
def step_embed(smiles: str, mmff_optimize: bool = True):
    """Embed SMILES into 3D conformer with ETKDG (+ optional MMFF)."""
    mol_3d = embed_conformer_from_smiles(smiles, attempts=50, MMFF_optimize=mmff_optimize)
    return mol_3d


@timed("2. Generate conformer ensemble (ETKDG + MMFF)")
def step_generate_ensemble(mol_3d, num_confs: int = 1000, num_opt_steps: int = 50):
    """Generate multiple conformers with ETKDG, optimize each with MMFF94."""
    conformer_ensemble = generate_conformer_ensemble(
        mol_3d,
        num_confs=num_confs,
        num_threads=4,
        threshold=0.05,
        num_opt_steps=num_opt_steps,
    )
    return conformer_ensemble


@timed("3. Cluster conformers (Butina RMSD)")
def step_cluster(conformers, threshold: float = 0.1):
    """Cluster conformers by RMSD using Butina algorithm."""
    clustered_indices = cluster_conformers_butina(
        conformers,
        threshold=threshold,
        num_max_conformers=None,
    )
    return [conformers[i] for i in clustered_indices]


@timed("4. xTB optimization (water solvent)")
def step_xtb_optimize(conformers, solvent: str = "water", num_processes: int = 1, charge: int = 0):
    """Optimize conformers with GFN2-xTB in implicit solvent."""
    opt_conformers, opt_energies, opt_charges = optimize_conformer_ensemble_with_xtb(
        conformers,
        solvent=solvent,
        num_processes=num_processes,
        charge=charge,
        verbose=True,
    )
    return opt_conformers, opt_energies, opt_charges


@timed("5. Final clustering (post-xTB)")
def step_final_cluster(conformers, energies, charges, threshold: float = 0.1):
    """Final clustering after xTB optimization."""
    clustered_indices = cluster_conformers_butina(
        conformers,
        threshold=threshold,
        num_max_conformers=None,
    )
    clustered_conformers = [conformers[i] for i in clustered_indices]
    clustered_energies = [energies[i] for i in clustered_indices]
    clustered_charges = [charges[i] for i in clustered_indices]
    return clustered_conformers, clustered_energies, clustered_charges


@timed("6. Convert to training format (molblock + charges)")
def step_to_training_format(conformers, charges):
    """Convert to pickle-serializable format: list of (molblock_str, charges_array) tuples."""
    molblocks_and_charges = []
    for conf, charge in zip(conformers, charges):
        mol_block = Chem.MolToMolBlock(conf)
        molblocks_and_charges.append((mol_block, np.array(charge)))
    return molblocks_and_charges


# =============================================================================
# PIPELINE RUNNERS
# =============================================================================

def run_pipeline_stepwise(smiles: str, solvent: str = "water", num_confs: int = 100, verbose: bool = True):
    """
    Run the full pipeline step-by-step with individual profiling.

    Args:
        smiles: Input SMILES string
        solvent: Solvent for xTB (default: "water")
        num_confs: Number of initial conformers to generate
        verbose: Print progress info

    Returns:
        molblocks_and_charges: List of (molblock, charges) tuples ready for training
    """
    if verbose:
        print(f"\nProcessing: {smiles}")
        print(f"Solvent: {solvent}")
        print(f"Initial conformers: {num_confs}")
        print("-" * 40)

    # Step 1: Embed
    mol_3d = step_embed(smiles, mmff_optimize=True)
    if mol_3d is None:
        raise ValueError(f"Failed to embed SMILES: {smiles}")
    if verbose:
        print(f"[ok] Embedded molecule: {mol_3d.GetNumAtoms()} atoms (with H)")

    # Step 2: Generate conformer ensemble
    conformer_ensemble = step_generate_ensemble(mol_3d, num_confs=num_confs)
    if verbose:
        print(f"[ok] Generated {len(conformer_ensemble)} conformers")

    # Step 3: Initial clustering
    clustered_conformers = step_cluster(conformer_ensemble, threshold=0.1)
    if verbose:
        print(f"[ok] Clustered to {len(clustered_conformers)} unique conformers")

    # Step 4: xTB optimization
    opt_conformers, opt_energies, opt_charges = step_xtb_optimize(
        clustered_conformers,
        solvent=solvent,
        charge=Chem.GetFormalCharge(mol_3d),
    )
    if verbose:
        print(f"[ok] xTB optimized {len(opt_conformers)} conformers")
        print(f"     Energy range: {min(opt_energies):.4f} to {max(opt_energies):.4f} Ha")

    # Step 5: Final clustering
    final_conformers, final_energies, final_charges = step_final_cluster(
        opt_conformers, opt_energies, opt_charges, threshold=0.1
    )
    if verbose:
        print(f"[ok] Final ensemble: {len(final_conformers)} conformers")

    # Step 6: Convert to training format
    molblocks_and_charges = step_to_training_format(final_conformers, final_charges)
    if verbose:
        print(f"[ok] Converted to training format")

    return molblocks_and_charges


def run_pipeline_integrated(smiles: str, solvent: str = "water", num_confs: int = 100, verbose: bool = True):
    """
    Run the integrated pipeline using generate_opt_conformers_xtb().
    This is the "production" function that does everything in one call.
    """
    start = time.perf_counter()

    conformers, energies, charges = generate_opt_conformers_xtb(
        smiles,
        charge=0,  # Will be computed from SMILES
        solvent=solvent,
        MMFF_optimize=True,
        num_processes=1,
        verbose=verbose,
        num_confs=num_confs,
    )

    duration = time.perf_counter() - start
    profiler.record("generate_opt_conformers_xtb (integrated)", duration)

    if conformers is None:
        raise ValueError(f"Failed to generate conformers for: {smiles}")

    # Convert to training format
    molblocks_and_charges = []
    for conf, charge in zip(conformers, charges):
        mol_block = Chem.MolToMolBlock(conf)
        molblocks_and_charges.append((mol_block, np.array(charge)))

    return molblocks_and_charges


# =============================================================================
# VALIDATION
# =============================================================================

@dataclass
class ValidationResult:
    """Result of validating one molecule."""
    index: int
    smiles: str
    n_atoms: int
    n_conformers_generated: int
    charge_rmse: float
    charge_mae: float
    charge_corr: float
    charge_sum_ref: float
    charge_sum_gen: float
    passed: bool
    failure_reasons: List[str]


def validate_single_molecule(
    ref_molblock: str,
    ref_charges: np.ndarray,
    index: int,
    tolerances: ValidationTolerances,
    num_confs: int = 50,
    verbose: bool = True,
) -> ValidationResult:
    """
    Validate pipeline by regenerating conformers for a reference molecule
    and comparing charges.

    Args:
        ref_molblock: Reference mol block string
        ref_charges: Reference xTB charges
        index: Index in reference dataset (for reporting)
        tolerances: Validation tolerances
        num_confs: Number of conformers to generate
        verbose: Print progress

    Returns:
        ValidationResult with comparison metrics
    """
    # Extract SMILES from reference
    ref_mol = Chem.MolFromMolBlock(ref_molblock, removeHs=False)
    smiles = Chem.MolToSmiles(Chem.RemoveHs(ref_mol))
    n_atoms = ref_mol.GetNumAtoms()
    ref_charges = np.array(ref_charges)

    if verbose:
        smiles_display = smiles[:50] + "..." if len(smiles) > 50 else smiles
        print(f"\n  [{index}] {smiles_display}")
        print(f"      Atoms: {n_atoms}")

    # Regenerate conformers
    try:
        conformers, energies, charges_list = generate_opt_conformers_xtb(
            smiles,
            solvent='water',
            MMFF_optimize=True,
            num_processes=1,
            num_confs=num_confs,
            verbose=False,
        )
    except Exception as e:
        return ValidationResult(
            index=index,
            smiles=smiles,
            n_atoms=n_atoms,
            n_conformers_generated=0,
            charge_rmse=float('inf'),
            charge_mae=float('inf'),
            charge_corr=0.0,
            charge_sum_ref=float(ref_charges.sum()),
            charge_sum_gen=float('nan'),
            passed=False,
            failure_reasons=[f"Generation failed: {e}"],
        )

    if conformers is None or len(conformers) == 0:
        return ValidationResult(
            index=index,
            smiles=smiles,
            n_atoms=n_atoms,
            n_conformers_generated=0,
            charge_rmse=float('inf'),
            charge_mae=float('inf'),
            charge_corr=0.0,
            charge_sum_ref=float(ref_charges.sum()),
            charge_sum_gen=float('nan'),
            passed=False,
            failure_reasons=["No conformers generated"],
        )

    # Find best matching conformer by charge RMSE
    best_rmse = float('inf')
    best_charges = None

    for charges in charges_list:
        charges = np.array(charges)
        if len(charges) != len(ref_charges):
            continue  # Atom count mismatch
        rmse = np.sqrt(np.mean((charges - ref_charges)**2))
        if rmse < best_rmse:
            best_rmse = rmse
            best_charges = charges

    if best_charges is None:
        return ValidationResult(
            index=index,
            smiles=smiles,
            n_atoms=n_atoms,
            n_conformers_generated=len(conformers),
            charge_rmse=float('inf'),
            charge_mae=float('inf'),
            charge_corr=0.0,
            charge_sum_ref=float(ref_charges.sum()),
            charge_sum_gen=float('nan'),
            passed=False,
            failure_reasons=["Atom count mismatch between reference and generated"],
        )

    # Compute metrics
    charge_rmse = best_rmse
    charge_mae = float(np.mean(np.abs(best_charges - ref_charges)))
    charge_corr = float(np.corrcoef(ref_charges, best_charges)[0, 1])
    charge_sum_ref = float(ref_charges.sum())
    charge_sum_gen = float(best_charges.sum())

    # Check tolerances
    failure_reasons = []
    if charge_rmse > tolerances.max_charge_rmse:
        failure_reasons.append(f"charge_rmse={charge_rmse:.6f} > {tolerances.max_charge_rmse}")
    if charge_corr < tolerances.min_charge_corr:
        failure_reasons.append(f"charge_corr={charge_corr:.6f} < {tolerances.min_charge_corr}")
    if abs(charge_sum_gen) > tolerances.max_charge_sum_deviation:
        failure_reasons.append(f"charge_sum={charge_sum_gen:.6f} deviates from 0")

    passed = len(failure_reasons) == 0

    if verbose:
        status = "PASS" if passed else "FAIL"
        print(f"      Generated: {len(conformers)} conformers")
        print(f"      Charge RMSE: {charge_rmse:.6f}")
        print(f"      Charge corr: {charge_corr:.6f}")
        print(f"      Status: [{status}]")
        if not passed:
            for reason in failure_reasons:
                print(f"        - {reason}")

    return ValidationResult(
        index=index,
        smiles=smiles,
        n_atoms=n_atoms,
        n_conformers_generated=len(conformers),
        charge_rmse=charge_rmse,
        charge_mae=charge_mae,
        charge_corr=charge_corr,
        charge_sum_ref=charge_sum_ref,
        charge_sum_gen=charge_sum_gen,
        passed=passed,
        failure_reasons=failure_reasons,
    )


def run_validation(
    reference_pkl: Path,
    n_molecules: int = 5,
    tolerances: ValidationTolerances = DEFAULT_TOLERANCES,
    num_confs: int = 50,
    verbose: bool = True,
) -> Tuple[List[ValidationResult], bool]:
    """
    Run validation against reference data.

    Args:
        reference_pkl: Path to reference pickle file
        n_molecules: Number of molecules to test
        tolerances: Validation tolerances
        num_confs: Number of conformers to generate per molecule
        verbose: Print progress

    Returns:
        Tuple of (list of ValidationResults, overall_passed)
    """
    print("\n" + "="*70)
    print("VALIDATION MODE")
    print("="*70)
    print(f"Reference data: {reference_pkl}")
    print(f"Molecules to test: {n_molecules}")
    print(f"Conformers per molecule: {num_confs}")
    print(f"\nTolerances:")
    print(tolerances)

    # Load reference data
    with open(reference_pkl, 'rb') as f:
        reference_data = pickle.load(f)

    print(f"\nLoaded {len(reference_data)} reference molecules")

    # Select molecules to test (prefer smaller ones for speed, but diverse sizes)
    # Sort by atom count and sample across the range
    mol_sizes = []
    for i, (molblock, charges) in enumerate(reference_data):
        mol = Chem.MolFromMolBlock(molblock, removeHs=False)
        if mol is not None:
            mol_sizes.append((i, mol.GetNumAtoms()))

    mol_sizes.sort(key=lambda x: x[1])

    # Sample molecules across size range, preferring smaller ones
    # Take from small, medium-small, medium ranges
    n_total = len(mol_sizes)
    test_indices = []

    # Prefer molecules with 15-35 atoms (typical drug-like size, reasonable speed)
    preferred = [(i, n) for i, n in mol_sizes if 15 <= n <= 35]
    if len(preferred) >= n_molecules:
        # Sample evenly from preferred range
        step = len(preferred) // n_molecules
        test_indices = [preferred[i * step][0] for i in range(n_molecules)]
    else:
        # Fall back to smallest available
        test_indices = [mol_sizes[i][0] for i in range(min(n_molecules, len(mol_sizes)))]

    print(f"Selected molecule indices: {test_indices}")
    print("\n" + "-"*70)
    print("Running validation...")
    print("-"*70)

    # Run validation
    results = []
    for idx in test_indices:
        molblock, charges = reference_data[idx]
        result = validate_single_molecule(
            ref_molblock=molblock,
            ref_charges=np.array(charges),
            index=idx,
            tolerances=tolerances,
            num_confs=num_confs,
            verbose=verbose,
        )
        results.append(result)

    # Summary
    n_passed = sum(1 for r in results if r.passed)
    n_failed = len(results) - n_passed
    overall_passed = n_failed == 0

    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"\n{'Index':<8} {'Atoms':<8} {'RMSE':<12} {'Corr':<10} {'Status'}")
    print("-"*50)
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"{r.index:<8} {r.n_atoms:<8} {r.charge_rmse:<12.6f} {r.charge_corr:<10.6f} {status}")
    print("-"*50)

    avg_rmse = np.mean([r.charge_rmse for r in results if r.charge_rmse < float('inf')])
    avg_corr = np.mean([r.charge_corr for r in results if r.charge_corr > 0])

    print(f"\nAverage charge RMSE: {avg_rmse:.6f}")
    print(f"Average charge corr: {avg_corr:.6f}")
    print(f"\nPassed: {n_passed}/{len(results)}")

    if overall_passed:
        print("\n[VALIDATION PASSED] Pipeline produces charges consistent with reference data")
    else:
        print(f"\n[VALIDATION FAILED] {n_failed} molecule(s) failed tolerance checks")
        print("\nSee 'VALIDATION GUIDE' in script docstring for troubleshooting tips")

    return results, overall_passed


# =============================================================================
# MAIN
# =============================================================================

# Example molecules for testing
EXAMPLE_SMILES = {
    "ethanol": "CCO",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
}


def main():
    parser = argparse.ArgumentParser(
        description="Conformer generation pipeline for ShEPhERD (v2 with validation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate conformers for a molecule
  python conformer_pipeline_e2e_v2.py --smiles "CCO" --profile

  # Validate pipeline against reference data
  python conformer_pipeline_e2e_v2.py --validate

  # Validate with custom tolerances
  python conformer_pipeline_e2e_v2.py --validate --validate-n 10 --max-charge-rmse 0.02
        """
    )

    # Generation options
    parser.add_argument("--smiles", type=str, default=None,
                        help="SMILES string to process")
    parser.add_argument("--num-confs", type=int, default=100,
                        help="Number of initial conformers (default: 100)")
    parser.add_argument("--solvent", type=str, default="water",
                        help="Solvent for xTB (default: water)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output pickle file path")
    parser.add_argument("--profile", action="store_true",
                        help="Enable profiling report")
    parser.add_argument("--integrated", action="store_true",
                        help="Use integrated pipeline function instead of step-by-step")

    # Validation options
    parser.add_argument("--validate", action="store_true",
                        help="Run validation against reference data")
    parser.add_argument("--validate-n", type=int, default=5,
                        help="Number of molecules to validate (default: 5)")
    parser.add_argument("--validate-confs", type=int, default=50,
                        help="Conformers per molecule in validation (default: 50)")
    parser.add_argument("--reference-pkl", type=str, default=None,
                        help="Path to reference pickle file (default: moses_aq example)")

    # Tolerance options
    parser.add_argument("--max-charge-rmse", type=float, default=0.01,
                        help="Max charge RMSE tolerance (default: 0.01)")
    parser.add_argument("--min-charge-corr", type=float, default=0.999,
                        help="Min charge correlation tolerance (default: 0.999)")

    args = parser.parse_args()

    # Build tolerances
    tolerances = ValidationTolerances(
        max_charge_rmse=args.max_charge_rmse,
        min_charge_corr=args.min_charge_corr,
    )

    # Validation mode
    if args.validate:
        # Find reference data
        if args.reference_pkl:
            reference_pkl = Path(args.reference_pkl)
        else:
            # Default to moses_aq example
            script_dir = Path(__file__).parent
            reference_pkl = script_dir.parent / "data" / "conformers" / "moses_aq" / "example_molblock_charges.pkl"

        if not reference_pkl.exists():
            print(f"ERROR: Reference file not found: {reference_pkl}")
            print("Specify --reference-pkl or ensure moses_aq example data exists")
            sys.exit(1)

        results, passed = run_validation(
            reference_pkl=reference_pkl,
            n_molecules=args.validate_n,
            tolerances=tolerances,
            num_confs=args.validate_confs,
            verbose=True,
        )

        sys.exit(0 if passed else 1)

    # Generation mode
    if args.smiles is None:
        args.smiles = "CCO"  # Default to ethanol

    print("\n" + "="*70)
    print("ShEPhERD Conformer Generation Pipeline (v2)")
    print("="*70)

    # Run pipeline
    if args.integrated:
        print("\nRunning INTEGRATED pipeline...")
        molblocks_and_charges = run_pipeline_integrated(
            args.smiles,
            solvent=args.solvent,
            num_confs=args.num_confs,
            verbose=True,
        )
    else:
        print("\nRunning STEP-BY-STEP pipeline...")
        molblocks_and_charges = run_pipeline_stepwise(
            args.smiles,
            solvent=args.solvent,
            num_confs=args.num_confs,
            verbose=True,
        )

    # Save output
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'wb') as f:
            pickle.dump(molblocks_and_charges, f)
        print(f"\n[ok] Saved {len(molblocks_and_charges)} conformers to {output_path}")

    # Show sample output
    print("\n" + "-"*40)
    print("Sample output (first conformer):")
    print(f"  MolBlock length: {len(molblocks_and_charges[0][0])} chars")
    print(f"  Num charges: {len(molblocks_and_charges[0][1])}")
    print(f"  Charges: {molblocks_and_charges[0][1][:5]}... (first 5)")
    print(f"  Charge sum: {sum(molblocks_and_charges[0][1]):.6f}")

    # Profiling report
    if args.profile:
        profiler.report()

    return molblocks_and_charges


if __name__ == "__main__":
    main()
