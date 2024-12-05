import argparse
import os
import warnings
import pathlib
import logging

import anndata as ad
import pandas as pd
import numpy as np
import torch
import tqdm
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.preprocessing import LabelEncoder

from brainformr.data import CenterMaskSampler, collate

genes_1024 = ["mt-Rnr2","Rn18s","Malat1","mt-Rnr1","Plp1","mt-Cytb","mt-Nd1","Fth1","Calm1","mt-Nd4","Calm2","Cmss1","Snap25","Cst3","Meg3","Camk1d","Mbp","Camk2n1","Apoe","Gnas","Cox4i1","Sparcl1","Slc25a4","Aldoc","Atp1b1","mt-Nd2","Pcp4","Ckb","Slc1a2","Cpe","Ppp3ca","Hsp90ab1","Kif5a","Aldoa","Tuba1a","Map1b","Actb","Mdh1","Rtn1","Ndrg2","Stmn3","Glul","Rbfox1","Snhg11","Lsamp","mt-Co1","Dynll2","Tmsb4x","Lars2","Ptgds","Ndrg4","Uchl1","Syt1","Olfm1","Mobp","Ttc3","Scd2","Basp1","Kcnip4","Atp1a2","Nrxn1","Nrgn","Vsnl1","Hpca","Nap1l5","Gabarapl1","Map1a","App","Itm2b","Nrxn3","Calm3","mt-Nd5","Zwint","Rpl4","Camk2a","Ttr","Nefl","Gpm6a","Gap43","Ncdn","Pcdh9","Serinc1","Dlgap1","Atp5b","Ubb","Syp","Sparc","Tspan7","Gpm6b","Aplp1","Stmn2","Ldhb","Hspa8","Ywhae","Rab6b","Ywhab","Atp5a1","Klc1","Hexb","Ptprd","Pkm","Celf2","Rtn3","Septin7","Nsf","Itm2c","Gphn","Chchd2","Tubb2a","Kif5c","Rps3","Slc25a3","Kif1b","Rab3a","Celf4","Thy1","Dlg2","Prdx5","Ywhag","Nefm","Stxbp1","Cadm2","Clu","Atp5g3","Map2","Atp6v1g2","Nptn","Atp6v1e1","Rtn4","Qk","Eef2","Ywhaz","Gria2","Atp2a2","Ndfip1","Kcnd2","Fgf14","Eno2","Atp5pb","Purb","Ntrk2","Prkcb","Snca","Fkbp3","Cox6a1","Selenow","Chn1","Tspan3","Ntm","Fus","Rab6a","Chgb","Mllt11","Hnrnpa2b1","Arpp21","Sptbn1","Vamp2","Atp6v1b2","Syt11","Cox5b","Tsc22d1","Ociad1","Ghitm","Trf","Rora","Sult4a1","Atp6v1a","Bex3","Tagln3","Tubb5","Tspyl4","Gabbr1","Nsg1","Septin4","Dpp10","Atp5j2","Nkain2","Cplx1","Ank2","Mt1","Camk2b","Atpif1","Atp1a3","6330403K07Rik","Arf3","Gria1","Mir6236","Syn2","Anks1b","Qdpr","Nrsn1","Cox6b1","Fam107a","Dynlrb1","Oxct1","Sptan1","Sub1","Thra","Fbxl16","Nme1","Ctnnb1","Mef2c","Psmc5","Pea15a","Pafah1b1","Gnb1","Uqcrq","Napb","Tcf4","Cox7b","Ndufb10","Bex2","Ppp2r2c","Mapt","Prnp","Dbi","Park7","Apod","Rrp1","Dnm1","Ncald","Ndufa4","Dclk1","Phactr1","Gnao1","Rnf187","Csmd1","Tubb3","Hsbp1","Prkar1a","Sod1","Ncam1","Magi2","Lrp1b","Cnp","Peg3","Ttyh1","Ubc","Ncl","Cd81","Cnbp","Pfn2","Tuba4a","Serbp1","Syne1","Clstn1","Mtch1","Nrg3","Nsg2","Zcchc18","Tecr","Snap47","Ppp3r1","Ndufv2","Oip5os1","Sqstm1","Matr3","Cntnap2","Ppp2ca","Myo5a","Cldn11","Fis1","Bc1","Rab3c","Vdac1","Gad1","Rbfox3","Arl6ip1","Opcml","Ank3","Ctsb","Rgs7bp","Pvalb","Skp1","Slc1a3","Fkbp1a","Mal","Chchd10","Ddx5","Dynll1","Pcsk1n","Negr1","Atp6v0b","Gng3","Atp2b1","Cdc42","Ywhah","Cck","Gm48099","Klf9","Rab2a","Tpi1","Kif1a","Car2","Scg5","Map1lc3a","Slc24a2","Spock2","Eef1a2","Sh3gl2","Mdh2","Rps5","Map7d2","Pink1","Psmb1","Clta","Cplx2","Fam155a","Atp6v1d","Hnrnpu","Cox6c","Pja2","Map4","Micos10","Lmo4","Itpr1","Atp2b2","Aco2","Plcb1","Cfl1","Nlgn1","S100b","Snrpn","Tle5","Gdi1","Rpl8","Sdha","Strbp","Gpi1","Pomp","Sec62","Idh3b","Kalrn","Tenm2","Kif5b","Tmod2","Stmn4","Lrrc4c","Ndn","Kifap3","Aplp2","Ptms","Gprasp1","Mapk10","Ndufa5","Psap","Atp6v0e2","Hspa5","Cdk5r1","Ctnnd2","Prkar1b","Ssb","Auts2","Atxn10","Camta1","Bsg","Hsp90b1","Trim2","Ankrd12","Tcf25","Acot7","Kcnma1","Mtpn","Rufy3","Oxr1","Chd3","Pfkp","Ina","Hpcal4","Adgrb3","Ralyl","Ddn","Zmat2","Pclo","Akr1a1","Slc22a17","Atrx","Sncb","Penk","Prdx2","Nat8l","Fkbp2","Dnajc5","Fbxw7","Drap1","Serpine2","Scg3","Tubb4a","Sdhb","Uqcrc2","Rps14","Pde1c","Pnmal2","Arpc2","Atxn7l3b","Smarca2","Ctnna2","Grid2","Nbea","Uqcrh","Vdac2","Cox5a","Cntn1","Ndufc2","Fgf12","G3bp2","Pura","Rps24","Spag9","Slc4a4","Hint1","Ndufa10","Cadps","Rims1","Ndufa13","Slc17a7","Prkce","Dpp6","Selenop","Cox7a2","Canx","Mbnl2","Atp6ap2","Ap2s1","Erc2","Atp6ap1","Nnat","Prkacb","Ypel3","Laptm4a","Prdx6","Zfp365","Ckmt1","Oaz1","Eid1","Bin1","Hspa4l","Gda","Zbtb20","Hsph1","Scg2","Cbx6","Ndufs2","Swi5","Arpp19","Pfkm","Ndufb6","Gabarap","Syngr1","Rnf227","Slc12a5","Jph4","Mapk8ip1","Lrrtm4","Gas7","Rock2","Mdga2","Pcp4l1","Cend1","Septin3","Pitpna","Gria4","Uqcrfs1","Pfdn5","Car10","Atp5c1","Dync1h1","Vamp1","Smdt1","Atp5k","Edf1","Lamp1","Tmbim6","Pde4b","Cfap36","Nedd8","Nisch","Rsrp1","Ndufb9","Mlf2","Cnksr2","TUSC3","Ndufa6","Phactr3","Impact","Ppp1r9b","Macf1","Plekhb1","Nrep","Reep5","Pak1","Atp5o","Ube2k","Gstm1","Atp6v1f","Arhgap5","Rab14","Cdk8","Selenok","Maged1","Enpp2","Zfr","Psmb6","Uqcrc1","Myt1l","Trpm3","Nucks1","Nrxn2","Car8","Gas5","Cadm1","Tcaf1","Tceal5","Eif1b","Ndufa12","Cox8a","Uqcr11","Ndufb5","Zic1","Ube2b","Cds2","Hnrnpab","Eef1b2","Stip1","Pdha1","Tpm1","Fez1","Rcan2","Unc80","Enc1","Ptn","Il1rapl1","Mapk1","Spop","Psmb4","Rgs8","Cadm3","Frrs1l","Synj1","Cdc42bpa","Pgm2l1","Nrg1","Pcsk2","Magee1","Ensa","Cltc","Ppp3cb","Fkbp8","Bag1","Calb1","Septin8","Ahcyl1","Atp6v1c1","Grin2a","Cyfip2","Abr","Scn2a","Eif5","Nptxr","Kirrel3","Rims2","Suclg1","Resp18","Hivep2","Asic2","Pde10a","Ndufb7","Mt2","Osbpl1a","Fam168a","Atp1a1","Ubl3","Dab1","Clstn3","Uqcr10","Atp5d","Gabrb3","Nckap1","Srp14","Ndufb8","Rnd2","Timm8b","Dync1i2","Zmynd11","Ndrg3","Atp5h","Pcmt1","Cacnb4","Kif21a","Hprt","Uqcrb","Arl3","Serpini1","Dzank1","Gsk3b","Ywhaq","Tmem130","Aak1","Epb41l3","Abat","Nars","Sv2a","Macrod2","Chmp4b","BC031181","Gnaq","Grm5","Meis2","Caln1","Tmem30a","Hspa12a","Hspa9","Ncor1","Atp6v0d1","Ndufa8","Rpl14","Nrcam","Grin2b","Idh3g","Oga","Hspa4","Tax1bp1","Pla2g7","Tceal3","Nefh","Csnk1a1","Vti1b","Sf3b2","Rbx1","Mrfap1","Adgrl1","Pja1","Psma7","R3hdm1","Pebp1","Apc","Hp1bp3","Dnm1l","Trim9","Dner","Srpk2","Prxl2b","Grik2","Apbb1","Gabrb1","Ppp1r9a","Kcnc1","Mag","Ttll7","Cltb","Anp32a","Rbm39","Epb41l1","Arhgef9","Gls","Psip1","Ctxn1","Coa3","Calb2","Dst","Rasgrf1","Rpl26","Frmd4a","Rabep1","Hnrnpk","Mlc1","Npm1","Msl1","Kcnj10","Ndufs7","Slc8a1","Gad2","Rabac1","Cyc1","Dbp","Fut9","Tafa5","Hmgcs1","Hnrnpul2","Psma6","Got1","Syt4","Pfdn2","Tuba1b","Psd3","Luc7l3","Hras","Mff","Dgkb","Saraf","Zfand5","Pnisr","Mog","Akt3","Rac1","Gabra1","Sucla2","Kidins220","Mapk8ip2","Asrgl1","Sdf4","Rheb","Tsc22d4","Mycbp2","Mapre2","Cs","Pbx1","Eif4g2","Gria3","Rangap1","Nedd4","Trim37","2210016L21Rik","Uba1","Ndufa2","Fam168b","Acsbg1","Luc7l2","Nudt3","Luzp2","Snrpd3","Rasgrp1","Brinp1","Flywch1","Camk2g","Arf1","Snap91","Ppm1a","Dock3","Hlf","Arpc5l","Camkv","Zfp706","Ik","Rplp1","Cct7","Cops6","Ppig","Dctn3","Dnm3","Map2k4","Nav3","Arl8b","Grm7","Ccdc85b","Csde1","Ermn","Tspan5","Pafah1b2","Ppp2r5c","Cox14","Rapgef4","Tef","Syt7","Enah","Cryab","Paip2","Efhd2","Plpp3","Acsl3","Dpysl2","Serinc3","Psma2","Ubqln2","Myh10","Mpc2","Sorl1","Sdhc","Faim2","Ptprz1","Emc7","Hdgfl3","Srrm2","Ndufb3","Ndufb2","Cdip1","Ddx1","Eps15","Micu3","Cbarp","Add1","Etnk1","Rph3a","Cd47","Rgs4","Cops9","Shank1","Pde4d","Rnf11","Dync1i1","Scn8a","Vmp1","Prpf19","Napa","Uqcc2","Aldh1a1","Lingo2","Ccdc88a","Araf","Dynlt3","Herc1","Cacna1e","Timm17a","Rad23b","Gnai1","Spred1","Cbx5","Bend6","Scamp5","Pmm1","Agtpbp1","Ddx6","Cuedc2","Rap1gds1","Capza2","Rnf14","Rasgef1a","Ahi1","Map1lc3b","Cct8","Pdhb","Tnik","Napg","Scn1b","Zranb2","Eif4h","Prrc2c","Eif3h","Eloc","Ntrk3","Spock1","Smim10l1","Son","Hdac11","Snhg14","Diras2","Tppp","Ddx24","Rundc3a","Map7d1","Clcn4","Ncs1","Cdr1os","Ndufs6","Akap6","Pkp4","Gaa","Gdi2","Trnp1","Rpl32","Marcks","Pcbp2","Eif4g3","Psmc3","Elavl3","Ccni","Tceal9","Selenom","Ndufs4","Snn","Ctsd","Tmed9","Tulp4","6430548M08Rik","Fnbp1","Plcb4","Stub1","Galnt2l","Lancl1","Snrnp70","Wnk1","Trim44","Ryr2","Cabp1","Cacna1a","Timp2","Wsb2","Ankrd11","Bclaf1","Taok1","Actr2","Ndufs8","Srgap3","Copg1","Ndufc1","Fam131a","Dctn2","Baalc","Disp2","Nfib","Scamp1","Actr1b","Selenof","Tpr","Wdr26","Smarcc2","Pde1a","Ppp2r2b","Enpp5","Cdk5r2","Ptprn","Map2k1","Rab7","Fkbp4","Gde1","Timm13","Guk1","Rps20","Trir","Ash1l","Mien1","Hnrnpc","Rhob","Glud1","1500009C09Rik","Chmp2a","Cfdp1","Bmerb1","Atp9a","Tbcb","Habp4","Gabbr2","Cacna2d1","Eif3c","Chn2","Pcp2","Fam171b","Ptpra","Ndufa11","Cox7a2l","Sfxn5","Top1","Cct2","Hnrnpd","Adgrl3","Gtf2i","Ip6k1","Txn1","Brk1","Sgcz","Chga","Atp1b3","Sars","Nudt4","Kmt2e","Npepps","Ap1s1","Arl2bp","Ppp1cb","Psd","Nbr1","Hnrnpdl","Agap1","Rab1a","Ubr3","Pdap1","Ppp2r1a","Scrn1","Snx3","Vapb","Chd4","Acsl6","Rbm25","Mat2b","Elob","Vcp","Eif1","Fabp3","Abhd8","Glrx5","Ddx3x","Pin1","Zcrb1","Pex5l","Dmd","Vps29","Map9","Fads2","Stx1b","Rab15","H3f3b","Adora1","Calr","Ube2r2","Ubqln1","Pcbp1","Pacsin1","Sv2b","Ube3a","Lynx1","Fads1","Lamp5","Psmd12","Ndufa3","Msi2","Adarb1","Ndfip2","Akap11","Tubb4b","Cwc15","Stau2","Akap9","Capns1","Myl12b","Eif5b","Ablim1","Add2","Dnaja1","Sez6l2","Dnajc7","Rabgap1l","Wbp2","Asph","Rack1","Secisbp2l","Dkk3","Ndufa1","Sst","Grina","Rcn2","Sphkap","Ptk2b","Dtna","Ppp1r1b"]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
#import sys
#sys.path.append('/home/ajl/work/d2//code/brainform')
#import brainform

#logger = logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adata_path", type=str, help="Path to anndata file.", required=True)
    parser.add_argument("--metadata_path", type=str, help="Path to metadata file.", required=True)
    parser.add_argument("--config_path", type=str, help="Path to config file.", required=True)
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint file from training.', required=True)
    parser.add_argument('--output_path', type=str, help='Path to save embeddings.', required=True)
    parser.add_argument('--celltype_colname', type=str, help='The column name that contains the cell type assignments', required=False, default='cell_index')
    parser.add_argument('--preprocess_type', type=str, help='The type of preprocessing to apply to the metadata', required=False, default='aibs')
    parser.add_argument('--bs', type=int, help='Batch size for inference.', required=False, default=32)
    parser.add_argument('--ps', type=int, help='Patch size for inference.', required=False, default=25)
    parser.add_argument('--use_bfloat', type=bool, help='Use bfloat16 for inference.', required=False, default=False)
    parser.add_argument('--chunk', type=bool, help='Whether to chunk along sections.', required=False, default=False)
    parser.add_argument('--reverse', help='Whether to reverse the order of sections.', required=False, default=False, action='store_true')

    return parser.parse_args()

def preprocess_langlieb(adata, metadata, cell_type_col: str = 'annotation_index', filter_genes: bool = True):
    metadata.rename(columns={'Raw_Slideseq_X': 'x',
                            'Raw_Slideseq_Y': 'y',
                            'PuckID': 'brain_section_label',
                            }, inplace=True)

    metadata['cell_type'] = metadata[cell_type_col].astype(int)
    #metadata['cell_type'] = metadata['cell_index'].astype(int)
    metadata = metadata.reset_index(drop=True)

    metadata = metadata[["cell_type", 'cell_label', 'x', 'y', 'brain_section_label']]

    if filter_genes:
        adata = adata[:, genes_1024]

    return adata, metadata

def preprocess_aibs(adata, metadata, cell_type_colname: str = 'subclass_name'):

    metadata["cell_label"] = metadata["cell_label"].astype(str)

    metadata['cell_type'] = metadata[cell_type_colname].astype(str)
    metadata["x"] = metadata["x_reconstructed"] * 100
    metadata["y"] = metadata["y_reconstructed"] * 100
    #metadata["x"] = metadata["spatial_x"] / 10.
    #metadata["y"] = metadata["spatial_y"] / 10.

    metadata = metadata[
        [
            "cell_type",
            "cell_label",
            "x",
            "y",
            "brain_section_label",
        ]
    ]

    #metadata = metadata.sort_values(by='brain_section_label')

    le = LabelEncoder()
    le.fit(sorted(metadata['cell_type'].unique()))
    metadata["cell_type"] = le.transform(metadata["cell_type"])
    metadata["cell_type"] = metadata["cell_type"].astype(int)

    metadata = metadata.reset_index(drop=True)

    return adata, metadata

def preprocess_nov(adata, metadata, cell_type_colname: str = 'cell_type'):
    #adata = adata[:, ~adata.var.index.str.contains("Blank")]

    metadata = adata.obs.copy()

    le = LabelEncoder()
    le.fit(sorted(metadata[cell_type_colname].unique()))
    metadata["cell_type"] = le.transform(metadata[cell_type_colname]).astype(int)
    metadata["cell_label"] = metadata['production_cell_id'].astype(str)

    #metadata['brain_section_label'] = metadata['section'].astype(str)
    metadata['brain_section_label'] = metadata['section'].astype(str)

    metadata["x"] = metadata["center_x"] / 10. # center_x
    metadata["y"] = metadata["center_y"] / 10. # center_y

    metadata = metadata[
    [
        'cell_type',
        "cell_label",
        "x",
        "y",
        'brain_section_label',
    ]
    ]

    #metadata = metadata[metadata['brain_section_label']=='1296400272']

    metadata = metadata.reset_index(drop=True)

    adata = adata[metadata['cell_label']]
    adata.obs = adata.obs[[]]

    return adata, metadata


def main():
    args = parse_args()
    logging.info(f"Arguments: {args}")

    if not os.path.exists(args.adata_path):
        raise FileNotFoundError(f"Anndata file not found: {args.adata_path}")

    if not os.path.exists(args.metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {args.metadata_path}")
    
    adata = ad.read_h5ad(args.adata_path)
    # filter control probes
    adata = adata[:, ~adata.var.index.str.contains("Blank")]
    #adata[:, ~adata.var.gene_symbol.str.contains("Blank")]
    logger.info('Finished reading h5ad.')

    with warnings.catch_warnings():
        if args.preprocess_type == 'nov':
            metadata = None
        else:
            warnings.simplefilter("ignore", pd.errors.DtypeWarning)
            metadata = pd.read_csv(args.metadata_path)

            if args.celltype_colname not in metadata.columns:
                raise ValueError(f"Cell type column {args.celltype_colname} not found in metadata columns. Provided arg: {args.celltype_colname}.")

    if args.preprocess_type == 'aibs':
        logger.info("Preprocessing AIBS data.")
        adata, metadata = preprocess_aibs(adata, metadata, cell_type_colname=args.celltype_colname)
    elif args.preprocess_type == 'langlieb':
        logger.info("Preprocessing Langlieb data.")
        adata, metadata = preprocess_langlieb(adata, metadata, cell_type_col=args.celltype_colname, filter_genes=True)
    elif args.preprocess_type == 'nov':
        adata, metadata = preprocess_nov(adata, metadata, cell_type_colname=args.celltype_colname)
    else:
        raise NotImplementedError(f"Preprocessing type {args.preprocess_type} not implemented.\n\
                                  Please pass one of ['aibs', 'langlieb', 'nov']")

    output_path = pathlib.Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = metadata#[metadata.brain_section_label=='C57BL6J-638850.37']
    adata = adata[metadata["cell_label"]]

    ps = int(args.ps)
    patch_size = (ps, ps)
    bs = args.bs

    logging.info(f"Using patch size: {patch_size}")
    logging.info(f"Using batch size: {bs}")
    config_dct = OmegaConf.load(args.config_path)

    model = instantiate(config_dct.model)

    sd = model.state_dict()
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = {}

    model_state_dict = checkpoint['state_dict']
    for key in model_state_dict:
        if args.preprocess_type == 'aibs' or args.preprocess_type == 'nov':
            key_wo_model = key.replace('model.', '').replace('._orig_mod', '').lstrip('.')
        elif args.preprocess_type == 'langlieb':
            key_wo_model = key.replace('model', '').replace('_orig_mod.', '').lstrip('.')
        state_dict[key_wo_model] = model_state_dict[key]
    # if continuing after Lightning checkpoint need to change keys
    # otherwise can instantiate Lightning object and use that for 
    # embedding generation

    model.load_state_dict(state_dict, strict=True)
    model = model.cuda()
    model.put_device = "cuda"


    if args.chunk is False:

        sampler = CenterMaskSampler(
            metadata=metadata,
            adata=adata,
            patch_size=patch_size,
            cell_id_colname="cell_label",
            cell_type_colname="cell_type",
            tissue_section_colname="brain_section_label",
            max_num_cells=None,
            indices=metadata.index.tolist(),
        )

        loader = torch.utils.data.DataLoader(
            sampler,
            batch_size=bs,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            collate_fn=collate,
            prefetch_factor=4,
        )


        embeds = []

        bf16_context = torch.cuda.amp.autocast_mode.autocast(dtype=torch.bfloat16) if args.use_bfloat else torch.cuda.amp.autocast_mode.autocast(dtype=torch.float32)
        with torch.inference_mode(), bf16_context:
            for batch in tqdm.tqdm(loader):
                fwd = model(batch)
                embed = fwd["neighborhood_repr"].detach().cpu()
                #embed = fwd['ref_cell_embed'].detach().cpu()
                expr_pred = fwd['zinb_params']['mu'].detach().cpu()
                target_expr = fwd['hidden_expression'].detach().cpu()
                embeds.append(embed)

        embeds = torch.cat(embeds)
        parent = pathlib.Path(args.checkpoint_path).parent.name

        new_name = output_path/f'model_{parent}_{pathlib.Path(args.adata_path).stem}_embeds_ps{patch_size[0]}.pth'

        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(embeds, new_name)
    else:
        tot_num_cells = len(metadata)
        num_cells_obs = 0
        idx = 0
        n_uniq = len(metadata['brain_section_label'].unique())

        for brain_section_label in sorted(metadata['brain_section_label'].unique(), reverse=args.reverse):
            parent = pathlib.Path(args.checkpoint_path).parent.name

            new_name = output_path/f'model_{parent}_{pathlib.Path(args.adata_path).stem}_embeds-section-{brain_section_label}_ps{patch_size[0]}.pth'
            if new_name.exists():
                logging.info(f'[{idx} / {n_uniq}] File already exists. Skipping {brain_section_label}.\n')
                idx += 1
                continue

            logging.info(f'[{idx} / {n_uniq}] Starting section {brain_section_label}.\n')
            subsection = metadata[metadata['brain_section_label']==brain_section_label]
            subsection = subsection.reset_index(drop=True)
            sub_adata = adata[subsection['cell_label']]

            sampler = CenterMaskSampler(
                metadata=subsection,
                adata=sub_adata,
                patch_size=patch_size,
                cell_id_colname="cell_label",
                cell_type_colname="cell_type",
                tissue_section_colname="brain_section_label",
                max_num_cells=500,
                indices=subsection.index.tolist(),
            )

            loader = torch.utils.data.DataLoader(
                sampler,
                batch_size=bs,
                shuffle=False,
                num_workers=1,
                pin_memory=True,
                collate_fn=collate,
                #prefetch_factor=4,
            )

            embeds = []
            
            bf16_context = torch.cuda.amp.autocast_mode.autocast(dtype=torch.bfloat16) if args.use_bfloat else torch.cuda.amp.autocast_mode.autocast(dtype=torch.float32)
            with torch.inference_mode(), bf16_context:
                pbar = tqdm.tqdm(loader, total=len(subsection), unit='cells')
                #for batch in tqdm.tqdm(loader):
                for batch in pbar:
                    try:
                        fwd = model(batch)
                        embed = fwd["neighborhood_repr"].detach().cpu()
                        #embed = fwd['ref_cell_embed'].detach().cpu()
                        expr_pred = fwd['zinb_params']['mu'].detach().cpu()
                        target_expr = fwd['hidden_expression'].detach().cpu()
                        embeds.append(embed)
                        pbar.update(batch['bs'])
                    except:
                        bs = batch['bs']
                        nans = np.full(shape=(bs, 384), fill_value=np.nan)
                        embeds.append(torch.tensor(nans))
                        logging.info(f'Error in batch with number {bs} elements.\n')
                        if bs == 1:
                            continue
                        else:
                            import pdb
                            pdb.set_trace()

            embeds = torch.cat(embeds)
            #assert torch.isnan(embeds).sum() == 0, 'Embeds contain NaNs.'
    
            assert len(embeds) == len(subsection), f'Embeds length {len(embeds)} does not match metadata length {len(subsection)}'
            num_cells_obs += len(subsection)

            output_path.mkdir(parents=True, exist_ok=True)
            torch.save(embeds, new_name)
            logging.info(f'Finished saving. File saved at {new_name}.\n')

            idx += 1
            
        if tot_num_cells != num_cells_obs:
            logging.warning(f'Number of cells observed {num_cells_obs} does not match total number of cells {tot_num_cells}.\n')

    #import numpy as np
    #np.save('//tmp/preds.npy', expr_pred.numpy())
    #np.save('//tmp/target.npy', target_expr.numpy())    

    logger.info(f'Finished saving. File saved at {new_name}.\n')

if __name__ == "__main__":
    main()
