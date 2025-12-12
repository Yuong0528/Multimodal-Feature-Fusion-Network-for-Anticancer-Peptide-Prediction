# Transfer .fa file to csv
from Bio import SeqIO
import pandas as pd
import re


def fasta_to_labeled_csv(input_fasta, output_csv):
    sequences = []
    labels = []

    for record in SeqIO.parse(input_fasta, "fasta"):
        seq_id = record.id
        sequence = str(record.seq)


        if re.search(r'Pos\d+', seq_id):
            label = 1
        elif re.search(r'Neg\d+', seq_id):
            label = 0
        else:
            continue  #

        sequences.append(sequence)
        labels.append(label)

    #
    df = pd.DataFrame({'sequence': sequences, 'label': labels})
    df.to_csv(output_csv, index=False)
    print(f"Already transfered finished{len(df)}")



fasta_to_labeled_csv('./ACPMain/ACPMain_ts.fasta', './ACPMain/test.csv')