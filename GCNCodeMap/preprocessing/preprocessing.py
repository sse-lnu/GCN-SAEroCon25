import pandas as pd
import json
import sys
import networkx as nx


def load_labels(labels_path):
    """Loads and parses the labels file into a structured dictionary."""
    with open(labels_path, 'r') as file:
        content = file.read()

    lines = content.split('\n')
    data = {'mapping': [], 'relations': [], 'roots': [], 'modules': []}
    current_section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            if line.startswith('# root-packages'):
                current_section = 'roots'
            elif line.startswith('# mapping'):
                current_section = 'mapping'
            elif line.startswith('# relations'):
                current_section = 'relations'
            elif line.startswith('# modules'):
                current_section = 'modules'
            continue

        if current_section == 'mapping':
            parts = line.split()
            if len(parts) == 2:
                data['mapping'].append({'Module': parts[0], 'Entity': parts[1]})
        elif current_section == 'relations':
            parts = line.split()
            if len(parts) == 2:
                data['relations'].append({'Source': parts[0], 'Target': parts[1]})
        elif current_section == 'roots' and '/' in line:
            data['roots'].append(line.strip())
        elif current_section == 'modules':
            data['modules'].append(line.strip())

    labels = pd.DataFrame(data['mapping'])
    labels['Entity'] = labels['Entity'].str.replace('..', '.').str.replace('\\.', '/').str.replace('*', '.*')
    labels['Entity'] = labels['Entity'].str.replace(r"\(\?:\?!", r"(?!", regex=True)

    architecture = pd.DataFrame(data['relations'])
    root = data['roots'][0].strip('/') if len(data['roots']) == 1 else ''

    return architecture, labels, root


def read_and_process_json(file_path, root):
    """Reads and cleans the JSON file, extracting relevant dependencies and entities."""
    flattened_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    flattened_data.append(pd.json_normalize(data))
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {line} -> {e}")
    except UnicodeDecodeError:
        with open(file_path, encoding='latin-1') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    flattened_data.append(pd.json_normalize(data))
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {line} -> {e}")

    df = pd.concat(flattened_data, ignore_index=True)
    df = df.rename(columns={'name': 'Entity'}).copy()
    df['Entity'] = df['Entity'].str.replace('.', '/')
    df = df[df['Entity'].str.contains(root)]
    df['Entity'] = df['Entity'].str.split('$').str[0]
    df = df[~df['Entity'].str.contains('package-info')]
    df['File'] = df['Entity'].str.split(root).str[1]
    return df


def clean_and_merge(df, labels):
    """Cleans and merges entities, then assigns modules."""
    def clean_tokens(texts):
        return [token for token in texts if len(token) > 1 and token.isalnum()]

    df['texts'] = df['texts'].apply(lambda x: clean_tokens(x) if isinstance(x, list) else [])
    df['Module'] = None
    for _, row in labels.iterrows():
        df.loc[df['Entity'].str.contains(row['Entity'], regex=True, na=False), 'Module'] = row['Module']

    df['File_ID'] = range(1, len(df) + 1)
    df = df[~df['Module'].isna()]
    return df


def extract_dependencies(df, df_dep, architecture):
    """Extracts and filters dependencies based on modules and architecture."""
    entity_set = set(df['Entity'])
    df_dep = pd.DataFrame(df_dep)
    df_dep['source'] = df_dep['source'].str.replace('.', '/')
    df_dep['target'] = df_dep['target'].str.replace('.', '/')
    df_dep = df_dep[df_dep['source'].isin(entity_set) & df_dep['target'].isin(entity_set)]
    df_dep = pd.merge(df_dep, df[['Entity', 'Module']], left_on='source', right_on='Entity', how='left').rename(columns={'Module': 'Source_Module'})
    df_dep = pd.merge(df_dep, df[['Entity', 'Module']], left_on='target', right_on='Entity', how='left').rename(columns={'Module': 'Target_Module'})
    allowed_set = set(zip(architecture['Source'], architecture['Target']))
    df_dep['Allowed'] = df_dep.apply(lambda row: 1 if (row['Source_Module'], row['Target_Module']) in allowed_set or row['Source_Module'] == row['Target_Module'] else 0, axis=1)
    return df_dep

def generate_Graph(df, dep):
    modules = {file: df['Module'].iloc[i] for i, file in enumerate(df['File_ID'])}
    nodes = {file: df['Entity'].iloc[i] for i, file in enumerate(df['File_ID'])}
    G = nx.MultiDiGraph()
    for file in nodes.keys():
        G.add_node(file, code=nodes[file], label=modules[file])
    for _, row in dep.iterrows():
        if row['Source_ID'] in G.nodes and row['Target_ID'] in G.nodes:
            G.add_edge(row['Source_ID'], row['Target_ID'], type=row['Dependency_Type'], weight=row['Dependency_Count'])
            
    df['Closeness_Centrality'] = df['File_ID'].map(nx.closeness_centrality(G))
    return df


def main(labels_path, json_path, dataset_name):
    architecture, labels, root = load_labels(labels_path)
    df = read_and_process_json(json_path, root)
    df = clean_and_merge(df, labels)
    df_dep = extract_dependencies(df, df['deps'], architecture)
    df = generate_Graph(df,df_dep)
    df.to_csv(f'{dataset_name}_entities.csv', index=False)
    df_dep.to_csv(f'{dataset_name}_dependencies.csv', index=False)
    print(f"Processing complete. Results saved as '{dataset_name}_entities.csv' and '{dataset_name}_dependencies.csv'.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        labels_path = sys.argv[1]
        json_path = sys.argv[2]
        dataset_name = sys.argv[3]
    main(labels_path, json_path, dataset_name)
