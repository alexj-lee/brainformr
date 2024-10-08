import pathlib
import pandas as pd

root = pathlib.Path('/home/ajl/work/d1/langlieb2023/mapping')
for dfpath in root.glob("Puck*mapping_metadata.csv"):

    df = pd.read_csv(dfpath)
#    print(df)
    df = df.drop('Unnamed: 0', axis=1)
    df.columns = ['cell_label'] + df.columns[1:].tolist()
    print(df)
    df.to_csv(dfpath, index=False)