"""
Here the snippets in bash from Catherine

../.venv/bin/cellranger-8.0.0/cellranger mkref --genome=punk --fasta=/media/catherine/PortableSSD/sc_termite_data/genomes_oist/Punk/Punk_maskedGenome.fna --genes=/media/catherine/PortableSSD/sc_termite_data/genomes_oist/Punk/Punk_genes.gtf

../.venv/bin/cellranger-8.0.0/cellranger count --id=run_count_pnit_sol \
--fastqs=/media/catherine/PortableSSD/sc_termite_data/raw_reads/pnit_sol/ \
--transcriptome=punk \
--create-bam=true
"""


import os
import sys
import pathlib
import glob
import subprocess as sp


data_fdn = pathlib.Path("/mnt/data/projects/termites/data/sc_termite_data/")


if __name__ == "__main__":

    raw_data_fdn = data_fdn / "raw_reads"
    count_data_fdn = data_fdn / "cell_ranger" / "matrices"

    raw_subfdn = os.listdir(raw_data_fdn)
    count_subfdn = os.listdir(count_data_fdn)

    print(raw_subfdn)
    print(count_subfdn)

    for fdn in raw_subfdn:

        csubfdn_exp = "run_count_" + fdn
        cfdn_exp = count_data_fdn / csubfdn_exp

        if csubfdn_exp in count_subfdn:
            print(fdn, "found")
        else:
            print(fdn, "missing")

            if fdn.startswith("znev"):
                print("Zootermopsis nevadensis already processed, ask Catherine")
                continue

            print("Running cell ranger")
            # NOTE: Pnit has no known genome, but Tom has a closely related species
            if fdn.startswith("pnit"):
                ref_genome_subfdn = "punk"
            elif fdn.startswith("rspe"):
                ref_genome_subfdn = "rfla"
            else:
                ref_genome_subfdn = fdn.split("_")[0]

            ref_genome_fdn = data_fdn / "cell_ranger" / "cell_ranger_ref_genomes" / ref_genome_subfdn
            fastq_fdn = data_fdn / "raw_reads" / fdn

            call = [
                "../software/cellranger-9.0.0/cellranger",
                "count",
                f"--id=run_count_{fdn}",
                f"--fastqs={fastq_fdn}",
                f"--transcriptome={ref_genome_fdn}",
                f"--output-dir={cfdn_exp}",
                "--create-bam=true",
                "--disable-ui",
            ]
            call = " ".join(call)
            print(call)

            sp.run(call, shell=True, check=True)
