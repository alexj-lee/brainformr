import time
import argparse
import requests
import logging

import anndata as ad

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description='Probe FASTA from anndata file')
    parser.add_argument('anndata_file', type=str, help='Path to the anndata file')
    parser.add_argument('output_file', type=str, help='Output file to save the probe FASTA')
    parser.add_argument('--search_url', type=str, help='URL to search for gene symbols', 
                        default='https://eutils.ncbi.nlm.nih.gov/entrez/eutils')
    parser.add_argument('--sleeptime', type=float, default=1.0, help='Time to sleep between requests')

    return parser.parse_args()

def get_search_url(search_url, gene_symbol):
    search_url = f"{search_url}/esearch.fcgi?db=gene&term={gene_symbol}[Gene%20Name]%20AND%20mus%20musculus[Organism]&retmode=json"
    return search_url

def get_link_url(search_url, gene_id):
    link_url = f"{search_url}/elink.fcgi?dbfrom=gene&db=nuccore&id={gene_id}&linkname=gene_nuccore_refseqrna&retmode=json"
    return link_url

def get_fetch_url(search_url, mrna_id):
    fetch_url = f"{search_url}/efetch.fcgi?db=nuccore&id={mrna_id}&rettype=fasta&retmode=text"
    return fetch_url

def main():
    args = parse_args()

    data = ad.read_h5ad(args.anndata_file)
    gene_symbols = data.var.index.to_list()

    logging.info(f"Found {len(gene_symbols)} gene symbols in the anndata file after loading.")

    if len(gene_symbols) == 0:
        raise ValueError('No gene symbols found in the anndata file')
    
    f_handle_out = open(args.output_file, 'w')
    
    for symbol in gene_symbols:
        #if symbol != 'Ighm': 
        #     continue
        search_url = get_search_url(args.search_url, symbol)
        response = requests.get(search_url)
        """ Return will look like:
        {'header': {'type': 'esearch', 'version': '0.3'}, 
            'esearchresult': 
                {'count': '1', 'retmax': '1', 'retstart': '0', 'idlist': ['373864'], 
                'translationset': [{'from': 'mus musculus[Organism]', 'to': '"Mus musculus"[Organism]'}], 
                'translationstack': [{'term': 'Col27a1[Gene Name]', 'field': 'Gene Name', 'count': '447', 'explode': 'N'}, {'term': '"Mus musculus"[Organism]', 'field': 'Organism', 'count': '251813', 'explode': 'Y'}, 'AND'], 'querytranslation': 'Col27a1[Gene Name] AND "Mus musculus"[Organism]'}}
        """
        response_json = response.json()

        if not response_json['esearchresult']['idlist']:
            print(f"Gene symbol {symbol} not found in NCBI.")

        gene_id = response_json['esearchresult']['idlist'][0]

        link_url = get_link_url(args.search_url, gene_id)
        response = requests.get(link_url)
        link_result = response.json()
        print(link_url)
        print(link_result)
        print(symbol)

        #if not link_result["linksets"][0].get("linksetdbs"):
        #    print(f"No mRNA sequence found for gene: {symbol}")

        mrna_id = None
        #mrna_id = link_result["linksets"][0]["linksetdbs"][0]["links"][0]
        
        try:
            mrna_id = link_result["linksets"][0]["linksetdbs"][0]["links"][0]
        except:
            pass
        
        if mrna_id is None:
            try:
                mrna_id = link_result['linksets'][0]['ids'][0]
            except:
                pass

        
        print(link_result['linksets'][0]['ids'][0])
        if mrna_id is None:
            print(f"No mRNA sequence found for gene: {symbol}")
            import sys
            sys.exit(1)

        fetch_url = get_fetch_url(args.search_url, mrna_id)
        response = requests.get(fetch_url)

        split_fasta = response.text.split('\n')
        header, sequence = split_fasta[0], ''.join(split_fasta[1:])
        print(sequence, header)
        f_handle_out.write(f"{header}\n{sequence}\n")

        logging.info(f"Gene symbol {symbol} found. Writing to file. Sequence is {len(sequence)} long.")
    
        time.sleep(args.sleeptime)

    logging.info('Done!')
    #f_handle_out.close()


    


if __name__ == '__main__':
    main()