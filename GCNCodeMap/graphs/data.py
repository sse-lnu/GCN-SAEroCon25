import pandas as pd
import numpy as np
import random
import torch
from torch_geometric.data import HeteroData, Data
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

class DataNB():
    def __init__(self, df,df_dep):
        self.df = df
        self.df_dep =df_dep
        self.x = None
        self.Y = None
        self.n_classes = None
        self.label_encoder = None
        self.process_data()

    def process_data(self):
        """
        Processes the input data to create node features, labels, edge indices, and attributes.
        """
        self._preprocess_dataframes()
        self._generate_embeddings()

    def _preprocess_dataframes(self):
        """
        Preprocesses df and df_dep by mapping Source_File and Target_File from df.
        """
        self.df = self.df[~self.df.Module.isna()]
        self.df_dep = self.df_dep[self.df_dep['Source'].isin(self.df['Entity']) & self.df_dep['Target'].isin(self.df['Entity'])]
        self.df.fillna('', inplace=True)
        valid_ents = set(self.df['Entity'])
        self.df_dep = self.df_dep[self.df_dep['Source'].isin(valid_ents) & self.df_dep['Target'].isin(valid_ents)]
        entity_to_file = self.df.set_index('Entity')['File'].to_dict()
        self.df_dep['Source_File'] = self.df_dep['Source'].map(entity_to_file)
        self.df_dep['Target_File'] = self.df_dep['Target'].map(entity_to_file)

    def _generate_embeddings(self):
        """
        Generates embeddings for CDA and Code columns, ensuring a consistent feature space for all rows.
        """
        self.df['CDA'] = self.df.apply(lambda row: self._generate_cda_text(row), axis=1).str.lower().str.split().str.join(' ')
        self.df['Code'] = self.df['Code'].fillna('').str.replace('/', ' ')
        self.df['Code'] = self.df['Code'].str.lower()
        corpus = self.df['CDA'].fillna('') + ' ' + self.df['Code']
        count_vectorizer = CountVectorizer()
        self.X = count_vectorizer.fit_transform(corpus).toarray()

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.Y = self.label_encoder.fit_transform(self.df['Module'])
        self.num_classes = len(np.unique(self.Y))

    def _generate_cda_text(self, row):
        """
        Creates CDA text for a given entity by combining dependency text from df_dep
        where the entity appears as Source or Target.
        """
        entity = row['Entity']

        # Find matches where Entity is the Source
        source_matches = self.df_dep[self.df_dep['Source'] == entity]
        source_texts = [
            f"{match['Source_File']} {match['Dependency_Type']} {match['Target_File']}"
            for _, match in source_matches.iterrows()
            if pd.notna(match['Source_File']) and pd.notna(match['Target_File'])
        ]
        # Find matches where Entity is the Target
        target_matches = self.df_dep[self.df_dep['Target'] == entity]
        target_texts = [
            f"{match['Source_File']} {match['Dependency_Type']} {match['Target_File']}"
            for _, match in target_matches.iterrows()
            if pd.notna(match['Source_File']) and pd.notna(match['Target_File'])
        ]

        combined_texts = source_texts + target_texts
        return ' '.join(combined_texts)

    def generate_split(self, q_threshold=0.3, split_ratio=0.05):
        centrality_values = self.df["Closeness_Centrality"]
        labels = self.Y
        all_indices = np.arange(len(labels))

        threshold_value = np.quantile(centrality_values, q_threshold)
        high_centrality_indices = all_indices[centrality_values >= threshold_value]

        # Filter high centrality entities that follow the intended architecture
        allowed_entities = set(self.df_dep.loc[self.df_dep["Allowed"] != 0, ["Source", "Target"]].values.flatten())
        allowed_indices = [i for i in high_centrality_indices if self.df.iloc[i]["Entity"] in allowed_entities]

        num_train_samples = max(1, int(split_ratio * len(all_indices)))
        train_indices = np.random.choice(high_centrality_indices, min(len(high_centrality_indices), num_train_samples), replace=False).tolist()
        for cls in np.unique(labels):
            class_indices = all_indices[labels == cls]
            if not any(labels[train_indices] == cls):  
                random_entity = np.random.choice(class_indices)
                train_indices.append(random_entity)

        if len(train_indices) < num_train_samples:
            remaining_candidates = list(set(allowed_indices) - set(train_indices))
            additional_indices = np.random.choice(remaining_candidates, num_train_samples - len(train_indices), replace=False).tolist()
            train_indices.extend(additional_indices)
        elif len(train_indices) > num_train_samples:
            train_indices = train_indices[:num_train_samples]

        test_indices = np.array(list(set(all_indices) - set(train_indices)))
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        return train_indices, test_indices
    

########### Homogeneous Data #############
class HomogeneousData(Data):
    def __init__(self, df, df_dep):
        super().__init__()
        self.df = df.copy()
        self.df_dep = df_dep.copy()
        self.label_encoder = None
        self.num_classes = None
        self.process_data()

    def process_data(self):
        self._preprocess_dataframes()
        self._create_node_features_and_labels()
        self._create_edges()

    def _preprocess_dataframes(self):
        self.df = self.df[~self.df.Module.isna()]
        self.df_dep = self.df_dep[self.df_dep['Source'].isin(self.df['Entity']) & self.df_dep['Target'].isin(self.df['Entity'])]

        self.df.fillna('', inplace=True)
        valid_ids = set(self.df['File_ID'])
        self.df_dep = self.df_dep[self.df_dep['Source_ID'].isin(valid_ids) & self.df_dep['Target_ID'].isin(valid_ids)]
        self.df = self.df.reset_index(drop=True)
        file_to_new_id = {old_id: new_id for new_id, old_id in enumerate(self.df['File_ID'])}
        self.df_dep['Source_ID'] = self.df_dep['Source_ID'].map(file_to_new_id)
        self.df_dep['Target_ID'] = self.df_dep['Target_ID'].map(file_to_new_id)
        self.df_dep.dropna(subset=['Source_ID', 'Target_ID'], inplace=True)

    def _create_node_features_and_labels(self):
        self.df['Code'] = self.df['Code'].fillna('') +' ' + self.df['File'].str.replace('/', ' ')
        code_vectorizer = CountVectorizer(binary=True)
        code_matrix = code_vectorizer.fit_transform(self.df['Code'])
        code_vectors = code_matrix.toarray()

        self.x = torch.tensor(code_vectors, dtype=torch.float)
        self.label_encoder = LabelEncoder()
        self.df['Label'] = self.label_encoder.fit_transform(self.df['Module'])
        self.y = torch.tensor(self.df['Label'].values, dtype=torch.long)
        self.num_classes = len(torch.unique(self.y))

    def _create_edges(self):
        dependency_encoded = pd.get_dummies(self.df_dep['Dependency_Type'], prefix='DepType')
        self.df_dep = pd.concat([self.df_dep, dependency_encoded], axis=1)
        self.edge_index = torch.tensor(
            np.array([self.df_dep['Source_ID'].values, self.df_dep['Target_ID'].values]),
            dtype=torch.long
        )
        self.edge_attr = torch.stack([
            torch.tensor(row[dependency_encoded.columns].tolist(), dtype=torch.float)
            for _, row in self.df_dep.iterrows()
        ])

    def generate_split(self, q_threshold=0.3, split_ratio=0.05):
        device = self.y.device
        centrality_values = torch.tensor(self.df["Closeness_Centrality"].values, dtype=torch.float, device=device)
        labels = self.y.to(device)
        all_nodes = torch.arange(len(labels), device=device)
        threshold_value = torch.quantile(centrality_values, q_threshold)
        high_centrality_indices = all_nodes[centrality_values >= threshold_value]

        allowed_entities = set(self.df_dep.loc[self.df_dep["Allowed"] != 0, ["Source", "Target"]].values.flatten())
        allowed_indices = torch.tensor(
            [i for i in high_centrality_indices.tolist() if self.df.iloc[i]["Entity"] in allowed_entities],
            dtype=torch.long, device=device
        )
        num_train_samples = max(5, int(split_ratio * len(allowed_indices)))
        train_indices = torch.tensor(random.sample(allowed_indices.tolist(), num_train_samples), dtype=torch.long, device=device)
        test_indices = torch.tensor(list(set(all_nodes.tolist()) - set(train_indices.tolist())), dtype=torch.long, device=device)

        return train_indices[torch.randperm(len(train_indices))], test_indices[torch.randperm(len(test_indices))]

###### HeteroGeneus Data ############

class HeterogeneousData(HeteroData):
    def __init__(self, df, df_dep):
        super().__init__()
        self.df = df.copy()
        self.df_dep = df_dep.copy()
        self.label_encoder = None
        self.num_classes = None
        self.relations = None
        self.process_data()

    def process_data(self):
        self._preprocess_dataframes()
        self._create_node_features_and_labels()
        self._create_edges()

    def _preprocess_dataframes(self):
        self.df = self.df[~self.df.Module.isna()]
        self.df_dep = self.df_dep[self.df_dep['Source'].isin(self.df['Entity']) & self.df_dep['Target'].isin(self.df['Entity'])]
        self.df.fillna('', inplace=True)
        valid_ids = set(self.df['File_ID'])
        self.df_dep = self.df_dep[self.df_dep['Source_ID'].isin(valid_ids) & self.df_dep['Target_ID'].isin(valid_ids)]
        self.df = self.df.reset_index(drop=True)

        file_to_new_id = {old_id: new_id for new_id, old_id in enumerate(self.df['File_ID'])}
        self.df_dep['Source_ID'] = self.df_dep['Source_ID'].map(file_to_new_id)
        self.df_dep['Target_ID'] = self.df_dep['Target_ID'].map(file_to_new_id)
        self.df_dep.dropna(subset=['Source_ID', 'Target_ID'], inplace=True)

    def _create_node_features_and_labels(self):
        self.df['Code'] = self.df['Code'].fillna('').str.replace('/', ' ')
        self.df['Code'] = self.df['Code'].str.lower()
        code_vectorizer = CountVectorizer(binary=True)
        code_matrix = code_vectorizer.fit_transform(self.df['Code'])
        code_vectors = code_matrix.toarray()
        self['entity'].x = torch.tensor(code_vectors, dtype=torch.float)
        self.label_encoder = LabelEncoder()
        self.df['Label'] = self.label_encoder.fit_transform(self.df['Module'])
        self['entity'].y = torch.tensor(self.df['Label'].values, dtype=torch.long)
        self.num_classes = len(torch.unique(self['entity'].y))

    def _create_edges(self):
        self.relations = list(self.df_dep.Dependency_Type.unique())
        edge_dict = defaultdict(list)
        for _, row in self.df_dep.iterrows():
            edge_dict[row['Dependency_Type']].append((row['Source_ID'], row['Target_ID']))
        for dep_type, edges in edge_dict.items():
            source_ids, target_ids = zip(*edges)
            edge_index = np.array([source_ids, target_ids])
            self['entity', dep_type, 'entity'].edge_index = torch.tensor(edge_index, dtype=torch.long)

    def generate_split(self, q_threshold=0.3, split_ratio=0.05):
        device = self['entity'].x.device
        centrality_values = torch.tensor(self.df["Closeness_Centrality"].values, dtype=torch.float, device=device)
        labels = self['entity'].y.to(device)
        all_nodes = torch.arange(len(labels), device=device)
        threshold_value = torch.quantile(centrality_values, q_threshold)
        high_centrality_indices = all_nodes[centrality_values >= threshold_value]

        allowed_entities = set(self.df_dep.loc[self.df_dep["Allowed"] != 0, ["Source", "Target"]].values.flatten())
        allowed_indices = torch.tensor(
            [i for i in high_centrality_indices.tolist() if self.df.iloc[i]["Entity"] in allowed_entities],
            dtype=torch.long, device=device
        )
        num_train_samples = max(5, int(split_ratio * len(allowed_indices)))
        train_indices = torch.tensor(random.sample(allowed_indices.tolist(), num_train_samples), dtype=torch.long, device=device)
        test_indices = torch.tensor(list(set(all_nodes.tolist()) - set(train_indices.tolist())), dtype=torch.long, device=device)

        return train_indices[torch.randperm(len(train_indices))], test_indices[torch.randperm(len(test_indices))]