checkpoint=/home/ajl/work/d1/model-checkpoints/langlieb/2024-10-08-1605/model-epoch2.ckpt
config=/home/ajl/work/d1/model-checkpoints/langlieb/2024-10-08-1605/config.yaml
checkpoint=/home/ajl/work/d1/model-checkpoints/langlieb/2024-10-09-1404/model.ckpt

adata_path=/home/ajl/work/n1/langlieb/h5ad_normM_log1p/Puck_Num_60.h5ad
metadata_path=/home/ajl/work/n1/langlieb/mapping/Puck_Num_60.mapping_metadata.csv

output_path=/home/ajl/work/d1/paper-revisions/langlieb_embeds/2024-10-08-1605/ps17/
preprocess_type=langlieb

python embeds_from_files_langlieb.py \
        --adata_path $adata_path \
        --metadata_path $metadata_path \
        --output_path $output_path \
        --preprocess_type $preprocess_type \
        --checkpoint $checkpoint \
        --config_path $config \
        --bs 128