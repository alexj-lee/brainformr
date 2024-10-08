import re

from biotite.sequence.io.fasta import FastaFile

def get_cds(seq: str) -> str:
    cds_pattern = r'ATG(?:[ATCG]{3})*?(?:TAA|TAG|TGA)'

    matches = re.finditer(cds_pattern, seq, re.IGNORECASE)
    #if cds:
    #    return cds.group()
    #return None
    cds = max((match.group() for match in matches), key=len, default=None)
    return cds

def uracilize(seq: str) -> str:
    return seq.replace('T', 'U')

source_file = '/home/ajl/687997_probes_mRNA.fa'
target_file = '/home/ajl/687997_probes_cds.fa'

reader = FastaFile().read_iter(source_file)

open_handle = open(target_file, 'w')

for idx, (header, seq) in enumerate(reader):
    cds = get_cds(seq)
    
    if not cds:
        raise ValueError(f"No CDS found in sequence {header}.")
    
    cds = uracilize(cds)

    open_handle.write(f'>{header}\n{cds}\n')
