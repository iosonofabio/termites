"""Clean FASTA files for ESM embedding, for all species.

This must be run inside the Python 3.10 saturn conda environment: source ~/miniconda3/bin/activate && conda activate saturn
"""

import os
import pathlib
import subprocess as sp



if __name__ == "__main__":

    genome_root_folder = pathlib.Path("/mnt/data/projects/termites/data/sc_termite_data/genomes_oist/")
    genome_subfdns = os.listdir(genome_root_folder)
    output_folder = pathlib.Path("/mnt/data/projects/termites/data/sc_termite_data/saturn_data/clean_fasta/")

    for genome_subrfdn in genome_subfdns:
        genome_fdn_path = genome_root_folder / genome_subrfdn

        print(f"Processing {genome_subrfdn}")

        script_path = pathlib.Path(__file__).parent.parent.parent.parent / "software" / "SATURN" / "protein_embeddings" / "clean_fasta.py"
        call = [
            "python",
            str(script_path),
            "--data_path",
            str(genome_fdn_path / f"{genome_subrfdn}_proteins_rep.faa"),
            "--save_path",
            str(output_folder / f"{genome_subrfdn}_proteins_rep.faa"),
        ]
        print(" ".join(call))
        sp.run(" ".join(call), check=True, shell=True)
