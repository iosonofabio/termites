"""Embed all proteins from all termite species using ESM.

This must be run inside the Python 3.9 esm conda environment: source ~/miniconda3/bin/activate && conda activate esm
"""
import os
import pathlib
import subprocess as sp



if __name__ == "__main__":

    fasta_root_folder = pathlib.Path("/mnt/data/projects/termites/data/sc_termite_data/saturn_data/clean_fasta/")
    output_folder = pathlib.Path("/mnt/data/projects/termites/data/sc_termite_data/saturn_data/esm_embeddings/")
    fasta_files = os.listdir(fasta_root_folder)

    for fasta_file in fasta_files:
        print(f"Processing {fasta_file}")

        fasta_file_abs_path = fasta_root_folder / fasta_file
        output_folder_abs_path = output_folder / f"{fasta_file}_esm1b"
        if output_folder_abs_path.exists():
            print(f"Skipping {fasta_file}, already processed")
            continue

        script_path = pathlib.Path(__file__).parent.parent.parent.parent / "software" / "esm" / "scripts" / "extract.py"
        call = [
            "python",
            str(script_path),
            "esm1b_t33_650M_UR50S",
            str(fasta_file_abs_path),
            str(output_folder / f"{fasta_file}_esm1b"),
            "--include",
            "mean",
        ]
        print(" ".join(call))
        sp.run(" ".join(call), check=True, shell=True)
