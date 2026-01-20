"""
End-to-end conformer generation pipeline for ShEPhERD training data.

This script demonstrates the full pipeline from SMILES to training-ready data:
1. RDKit ETKDG embedding
2. MMFF94 optimization
3. Conformer ensemble generation & clustering
4. xTB optimization in water (ALPB implicit solvent)
5. Final clustering and charge extraction
6. Serialization to pickle format

Usage:
    python conformer_pipeline_e2e.py [--profile] [--smiles SMILES]
"""

import argparse
import pickle
import time
from pathlib import Path
from functools import wraps
from typing import Optional
import numpy as np

import rdkit
from rdkit import Chem

# Import shepherd conformer generation utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from shepherd.shepherd_score_utils.conformer_generation import (
    embed_conformer_from_smiles,
    generate_conformer_ensemble,
    cluster_conformers_butina,
    optimize_conformer_with_xtb,
    optimize_conformer_ensemble_with_xtb,
    generate_opt_conformers_xtb,
)


# Profiling utilities
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

    def report(self):
        total = sum(self.timings.values())
        print("\n" + "="*60)
        print("PROFILING REPORT")
        print("="*60)
        print(f"{'Stage':<40} {'Time (s)':<12} {'%':<8} {'Count'}")
        print("-"*60)
        for name, t in sorted(self.timings.items(), key=lambda x: -x[1]):
            pct = (t / total * 100) if total > 0 else 0
            print(f"{name:<40} {t:<12.3f} {pct:<8.1f} {self.counts[name]}")
        print("-"*60)
        print(f"{'TOTAL':<40} {total:<12.3f}")
        print("="*60)


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


# Step-by-step pipeline with individual timing
@timed("1. SMILES → RDKit mol + ETKDG embed")
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
        print(f"✓ Embedded molecule: {mol_3d.GetNumAtoms()} atoms (with H)")

    # Step 2: Generate conformer ensemble
    conformer_ensemble = step_generate_ensemble(mol_3d, num_confs=num_confs)
    if verbose:
        print(f"✓ Generated {len(conformer_ensemble)} conformers")

    # Step 3: Initial clustering
    clustered_conformers = step_cluster(conformer_ensemble, threshold=0.1)
    if verbose:
        print(f"✓ Clustered to {len(clustered_conformers)} unique conformers")

    # Step 4: xTB optimization
    opt_conformers, opt_energies, opt_charges = step_xtb_optimize(
        clustered_conformers,
        solvent=solvent,
        charge=Chem.GetFormalCharge(mol_3d),
    )
    if verbose:
        print(f"✓ xTB optimized {len(opt_conformers)} conformers")
        print(f"  Energy range: {min(opt_energies):.4f} to {max(opt_energies):.4f} Ha")

    # Step 5: Final clustering
    final_conformers, final_energies, final_charges = step_final_cluster(
        opt_conformers, opt_energies, opt_charges, threshold=0.1
    )
    if verbose:
        print(f"✓ Final ensemble: {len(final_conformers)} conformers")

    # Step 6: Convert to training format
    molblocks_and_charges = step_to_training_format(final_conformers, final_charges)
    if verbose:
        print(f"✓ Converted to training format")

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


def main():
    parser = argparse.ArgumentParser(description="Conformer generation pipeline for ShEPhERD")
    parser.add_argument("--smiles", type=str, default="CCO",
                        help="SMILES string to process (default: ethanol)")
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
    args = parser.parse_args()

    print("\n" + "="*60)
    print("ShEPhERD Conformer Generation Pipeline")
    print("="*60)

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
        print(f"\n✓ Saved {len(molblocks_and_charges)} conformers to {output_path}")

    # Show sample output
    print("\n" + "-"*40)
    print("Sample output (first conformer):")
    print(f"  MolBlock length: {len(molblocks_and_charges[0][0])} chars")
    print(f"  Num charges: {len(molblocks_and_charges[0][1])}")
    print(f"  Charges: {molblocks_and_charges[0][1][:5]}... (first 5)")

    # Profiling report
    if args.profile:
        profiler.report()

    return molblocks_and_charges


# Example molecules for testing
EXAMPLE_SMILES = {
    "ethanol": "CCO",
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
}


if __name__ == "__main__":
    main()
