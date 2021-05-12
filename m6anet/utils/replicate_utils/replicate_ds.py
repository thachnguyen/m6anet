import os
import pandas as pd
import numpy as np
import joblib
import torch
import json
from itertools import product
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit
from .normalization import annotate_kmer_information, create_kmer_mapping_df, create_norm_dict


class NanopolishReplicatesDS(Dataset):

    def __init__(self, root_dirs, min_reads, norm_path=None, site_info=None,
                 num_neighboring_features=1, mode='Inference', site_mode=False,
                 norm_dict_save_path=None,
                 n_processes=1):
        allowed_mode = ('Train', 'Test', 'Val', 'Inference')
        
        if mode not in allowed_mode:
            raise ValueError("Invalid mode passed to dataset, must be one of {}".format(allowed_mode))
        
        self.mode = mode
        self.n_processes = n_processes
        self.site_info = site_info
        self.data_info = self.initialize_data_info(root_dirs, min_reads)
        self.data_fpaths = {os.path.basename(os.path.normpath(root_dir)): os.path.join(root_dir, "data.json")
                            for root_dir in root_dirs}
        self.min_reads = min_reads
        self.site_mode = site_mode
        
        if norm_path is not None:
            self.norm_dict = joblib.load(norm_path)
        else:
            self.norm_dict = self.compute_norm_factors(n_processes, norm_dict_save_path)

        if num_neighboring_features > 5:
            raise ValueError("Invalid neighboring features number {}".format(num_neighboring_features))

        self.num_neighboring_features = num_neighboring_features
        
        # Creating list of kmers and its corresponding mapping
        
        center_motifs = [['A', 'G', 'T'], ['G', 'A'], ['A'], ['C'], ['A', 'C', 'T']]
        flanking_motifs = [['G', 'A', 'C', 'T'] for i in range(self.num_neighboring_features)]
        all_kmers = list(["".join(x) for x in product(*(flanking_motifs + center_motifs + flanking_motifs))])
        
        self.all_kmers = np.unique(np.array(list(map(lambda x: [x[i:i+5] for i in range(len(x) -4)], 
                                            all_kmers))).flatten())
        self.kmer_to_int = {self.all_kmers[i]: i for i in range(len(self.all_kmers))}
        self.int_to_kmer =  {i: self.all_kmers[i] for i in range(len(self.all_kmers))}

        # Inferring total number of neighboring features extracted during dataprep step

        kmer, _ = self._load_data(0)
        self.total_neighboring_features = (len(kmer) - 5) // 2
        left_idx = [(self.total_neighboring_features - num_neighboring_features + j) * 3 + i 
                    for j in range(num_neighboring_features) for i in range(3)]
        center_idx = [self.total_neighboring_features * 3 + i for i in range(3)]
        right_idx = [(self.total_neighboring_features + j) * 3 + i for j in range(1, num_neighboring_features + 1) 
                        for i in range(3)]

        self.indices = np.concatenate([left_idx, center_idx, right_idx]).astype('int')

        if self.mode != 'Inference':
            self.labels = self.data_info["modification_status"].values

    def __len__(self):
        return len(self.data_info)

    def _load_data(self, idx):
        tx_id, tx_pos, kmer = None, None, None
        all_features = []
        for rep_name, fpath in self.data_fpaths.items():
            with open(fpath, 'r') as f:
                row = self.data_info.iloc[idx]
                if (tx_id is None) and (tx_pos is None):
                    tx_id, tx_pos = row[["transcript_id", "transcript_position"]]
                    
                start_pos, end_pos = row[["start_{}".format(rep_name), "end_{}".format(rep_name)]]
                f.seek(start_pos, 0)
                json_str = f.read(end_pos - start_pos)
                pos_info = json.loads(json_str)[tx_id][str(tx_pos)]

                assert(len(pos_info.keys()) == 1)
                
                if kmer is None:
                    kmer, features = list(pos_info.items())[0]
                else:
                    kmer_rep, features = list(pos_info.items())[0]
                    if kmer != kmer_rep:
                        raise ValueError("Found different kmer at the same position of replicates {}, {}, {}, {}"\
                                             .format(tx_id, tx_pos, kmer, kmer_rep))
                all_features.append(np.array(features))
        return kmer, np.concatenate(all_features, axis=0)

    def __getitem__(self, idx):
        kmer, features = self._load_data(idx)
        # Repeating kmer to the number of reads sampled
        kmer = self._retrieve_full_sequence(kmer, self.num_neighboring_features)
        kmer = [kmer[i:i+5] for i in range(2 * self.num_neighboring_features + 1)]

        features = features[np.random.choice(len(features), self.min_reads, replace=False), :]
        features = features[:, self.indices]
        
        if self.norm_dict is not None:
            mean, std = self.get_norm_factor(kmer)
            features = torch.Tensor((features - mean) / std)
        else:
            features = torch.Tensor((features))

        if not self.site_mode:
            kmer = np.repeat(np.array([self.kmer_to_int[kmer] for kmer in kmer])\
                        .reshape(-1, 2 * self.num_neighboring_features + 1), self.min_reads, axis=0)
            kmer = torch.Tensor(kmer)
        else:
            kmer = torch.LongTensor([self.kmer_to_int[kmer] for kmer in kmer])
        if self.mode == 'Inference':
            return features, kmer
        else:
            return features, kmer, self.data_info.iloc[idx]["modification_status"]

    def get_norm_factor(self, list_of_kmers):
        norm_mean, norm_std = [], []
        for kmer in list_of_kmers:
            mean, std = self.norm_dict[kmer]
            norm_mean.append(mean)
            norm_std.append(std)
        return np.concatenate(norm_mean), np.concatenate(norm_std)

    def compute_norm_factors(self, n_processes, norm_dict_save_path):
        if "kmer" not in self.data_info.columns:
            print("k-mer information is not present in column, annotating k-mer information in data info")
            self.data_info = annotate_kmer_information(self.data_fpaths, self.data_info, n_processes)
        kmer_mapping_df = create_kmer_mapping_df(self.data_info, self.data_fpaths)
        norm_dict = create_norm_dict(kmer_mapping_df, self.data_fpaths, n_processes)

        if norm_dict_save_path is not None:
            joblib.dump(norm_dict, os.path.join(norm_dict_save_path, "norm_dict.joblib"))
        return norm_dict

    def _retrieve_full_sequence(self, kmer, n_neighboring_features=0):
        if n_neighboring_features < self.total_neighboring_features:
            return kmer[self.total_neighboring_features - n_neighboring_features:2 * self.total_neighboring_features + n_neighboring_features]
        else:
            return kmer

    def _retrieve_sequence(self, sequence):
        return [sequence[i : i+5] for i in range(len(sequence) - 4)]
    
    def initialize_data_info(self, root_dirs, min_reads):
        all_data = None
        rep_names = [os.path.basename(os.path.normpath(root_dir)) for root_dir in root_dirs]
        for root_dir, rep_name in zip(root_dirs, rep_names):
            data_index = pd.read_csv(os.path.join(root_dir ,"data.index"))
            data_fpath = os.path.join(root_dir, "data.json")

            if self.mode == 'Inference':
                if "kmer" not in data_index.columns:
                    data_index = annotate_kmer_information(data_fpath, data_index, self.n_processes)
                read_count = pd.read_csv(os.path.join(root_dir, "data.readcount"))
            else:
                if self.site_info is None:
                    read_count = pd.read_csv(os.path.join(root_dir, "data.readcount.labelled"))
                else:
                    read_count = pd.read_csv(os.path.join(self.site_info, "data.readcount.labelled"))
                
                read_count = read_count[read_count["set_type"] == self.mode].reset_index(drop=True)

            data_info = data_index.merge(read_count, on=["transcript_id", "transcript_position"])\
                .set_index(["transcript_id", "transcript_position"])
            # Rename will not throw an error if the column specified is not present
            data_info = data_info.rename(columns={col: "{}_{}".format(col, rep_name) for col in 
                                                  ["start", "end", "n_reads", "kmer" ,"modification_status", "set_type"]})
            if all_data is None:
                all_data = data_info
                all_data = all_data.rename({'kmer_{}'.format(rep_name): "kmer"}, axis=1)

                if self.mode != 'Inference':
                    all_data = all_data.rename({'modification_status_{}'.format(rep_name): "modification_status"}, axis=1)
                    all_data = all_data.rename({'set_type_{}'.format(rep_name): "set_type"}, axis=1)
                
            else:
                new_kmer_col = "kmer_{}".format(rep_name)
                all_data = all_data.merge(data_info, how='inner', on=["transcript_id", "transcript_position"])
                assert(np.all(all_data["kmer"] == all_data[new_kmer_col]))
                all_data = all_data.drop(new_kmer_col, axis=1)

                # Checking if the label is consistent
                if self.mode != 'Inference':
                    new_mod_column = "modification_status_{}".format(rep_name)
                    new_set_column = "set_type_{}".format(rep_name)
                    assert(np.all(all_data["modification_status"] == all_data[new_mod_column]))
                    assert(np.all(all_data["set_type"] == all_data[new_set_column]))
                    all_data = all_data.drop(new_mod_column, axis=1)
                    all_data = all_data.drop(new_set_column, axis=1)

        all_data["n_reads"] = all_data[["n_reads_{}".format(rep_name) for rep_name in rep_names]].sum(axis=1) # Perform check for kmer annotations
        all_data = all_data.reset_index() # Restoring transcript_id and position to the columns 
        return all_data[all_data["n_reads"] >= min_reads].reset_index(drop=True)
