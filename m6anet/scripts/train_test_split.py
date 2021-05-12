import pandas as pd
import numpy as np
import os
import argparse
from multiprocessing import Pool
from sklearn.model_selection import GroupShuffleSplit


def train_test_val_split(info_df):
    info_df["set_type"] = np.repeat("NA", len(info_df))
    np.random.seed(0)
    all_sites = np.arange(len(info_df))
    
    # Create validation set
    train_val_index, test_index = next(GroupShuffleSplit(n_splits=10).split(all_sites, groups=info_df["gene_id"]))
    train_val_sites, test_sites = all_sites[train_val_index], all_sites[test_index]
    
    # Further split train set into train and test
    train_val_group = info_df.iloc[train_val_sites]["gene_id"]
    train_index, val_index = next(GroupShuffleSplit(n_splits=10).split(train_val_sites, groups=train_val_group))
    train_sites, val_sites = train_val_sites[train_index], train_val_sites[val_index]
    
    info_df.loc[train_sites, "set_type"] = np.repeat("Train", len(train_sites))
    info_df.loc[val_sites, "set_type"] = np.repeat("Val", len(val_sites))
    info_df.loc[test_sites, "set_type"] = np.repeat("Test", len(test_sites))
    return info_df


def main():
    parser = argparse.ArgumentParser(description="a script to extract raw signals and event align features from tx hdf5 files")
    parser.add_argument('-i', '--input_dirs', dest='input_dir', default=None, nargs='*', 
                        help="Input directories containing the data.readcount file")              
    args = parser.parse_args()
    input_dirs = args.input_dir
    all_data = None
    all_data_info = []
    for input_dir in input_dirs:
        data_info = pd.read_csv(os.path.join(input_dir, "data.readcount.labelled")).set_index(["gene_id", "genomic_position", "transcript_id", "transcript_position"])
        if all_data is None:
            all_data = data_info
        else:
            all_data = all_data.merge(data_info, how='inner', on=["gene_id", "genomic_position", "transcript_id", "transcript_position"])
        all_data_info.append(data_info.reset_index())

    all_data = all_data.reset_index()
    read_columns = [col for col in all_data.columns if "n_reads_" in col]
    all_data["n_reads"] = all_data[read_columns].sum(axis=1)
    all_data = all_data[all_data["n_reads"] >= 20].reset_index(drop=True) #Filtering out positions with low reads
    all_data = train_test_val_split(all_data.reset_index())
    
    train_genes = np.unique(all_data[all_data["set_type"] == 'Train']["gene_id"].values)
    val_genes = np.unique(all_data[all_data["set_type"] == 'Val']["gene_id"].values)
    test_genes = np.unique(all_data[all_data["set_type"] == 'Test']["gene_id"].values)

    for data, input_dir in zip(all_data_info, input_dirs):
        data["set_type"] = np.repeat("NA", len(data))
        train_index = np.argwhere(data["gene_id"].isin(train_genes).values).flatten()
        val_index = np.argwhere(data["gene_id"].isin(val_genes).values).flatten()
        test_index = np.argwhere(data["gene_id"].isin(test_genes).values).flatten()

        data.loc[train_index, "set_type"] = np.repeat("Train", len(train_index))
        data.loc[val_index, "set_type"] = np.repeat("Val", len(val_index))
        data.loc[test_index, "set_type"] = np.repeat("Test", len(test_index))
        data.to_csv(os.path.join(input_dir, "data.readcount.labelled"), index=False)

if __name__ == '__main__':
    main()
