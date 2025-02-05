import pandas as pd
import json
import argparse
import re
import os
import networkx as nx

def load_labels(labels_path):
    """Loads and parses the labels file into a structured dictionary."""
    with open(labels_path, 'r') as file:
        content = file.read()

    lines = content.split('\n')
    data = {
        'mapping': [],
        'relations': [],
        'roots': [],
        'modules': []
    }

    current_section = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            if line.strip().startswith('# root-packages'):
                current_section = 'roots'
            elif line.strip().startswith('# mapping'):
                current_section = 'mapping'
            elif line.strip().startswith('# relations'):
                current_section = 'relations'
            elif line.strip().startswith('# modules'):
                current_section = 'modules'
            continue

        if current_section == 'mapping':
            map_file = line.split()
            if len(map_file) == 2:
                data['mapping'].append({'Module': map_file[0], 'Entity': map_file[1]})
        elif current_section == 'relations':
            source_target = line.split()
            if len(source_target) == 2:
                data['relations'].append({'Source': source_target[0], 'Target': source_target[1]})
        elif current_section == 'roots':
            if '/' in line:
                data['roots'].append(line.strip())
        elif current_section == 'modules':
            data['modules'].append(line.strip())

    labels = pd.DataFrame(data['mapping'])
    labels.Entity = labels.Entity.str.replace('..', '.')
    labels.Entity = labels.Entity.str.replace('\\.', '/')
    labels.Entity = labels.Entity.str.replace('*', '.*')
    labels['Entity'] = labels['Entity'].str.replace(r"\(\?:\?!", r"(?!", regex=True)

    architecture = pd.DataFrame(data['relations'])

    roots = data['roots']
    if roots:
        root = '|'.join([r.strip('/') for r in roots])
        if '|' in root:
            terms = root.split('|')
            split_terms = [term.split('/') for term in terms]
            common_term = set(split_terms[0]).intersection(*split_terms[1:])
            if common_term:
                root = common_term.pop()
    else:
        root = ''
    return architecture, labels, root

##############################################
 # extracting data and dependencies

def read_json(file_path, encoding='utf-8'):
    flattened_data_list = []
    try:
        with open(file_path, encoding=encoding) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    flattened_data_list.append(pd.json_normalize(data))
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {line} -> {e}")
        flattened_data = pd.concat(flattened_data_list, ignore_index=True)
    except UnicodeDecodeError:
        with open(file_path, encoding='latin-1') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    flattened_data_list.append(pd.json_normalize(data))
                except json.JSONDecodeError as e:
                    print(f"Error decoding line: {line} -> {e}")
        flattened_data = pd.concat(flattened_data_list, ignore_index=True)
    return flattened_data

def cleaning(df, root):
    df = df.copy()
    df.rename(columns={'name': 'Entity'}, inplace=True)
    df['Entity'] = df['Entity'].apply(lambda x: x.replace('.', '/'))
    df = df[df['Entity'].str.contains(root)]
    df['Entity'] = df['Entity'].apply(lambda x: x.split('$')[0])
    df = df[~df['Entity'].str.contains('package-info')]
    df['File'] = df['Entity'].apply(lambda x: x.split(root)[1])
    df['File'] = df['File'].str.lstrip('/')
    return df

def extract_dependencies(df):
    deps_list = []
    entity_set = set(df['Entity'])
    for deps in df['deps']:
        if isinstance(deps, list):
            deps_list.extend(deps)
    
    deps_df = pd.DataFrame(deps_list)
    deps_df['source'] = deps_df['source'].apply(lambda x: x.replace('.', '/'))
    deps_df['target'] = deps_df['target'].apply(lambda x: x.replace('.', '/'))
    
    deps_df = deps_df[deps_df['source'].isin(entity_set) & deps_df['target'].isin(entity_set)]
    deps_df.drop_duplicates(inplace=True)
    deps_df.rename(columns={'source': 'Source', 'target': 'Target', 'type': 'Dependency_Type', 'count': 'Dependency_Count'}, inplace=True)
    
    return deps_df

def clean_code(df):

    def clean_text_tokens(texts):
        cleaned_texts = []
        for token in texts:
            if re.fullmatch(r'[\n\t]{1,}', token):  # Skip newlines/tabs
                continue
            if len(token) == 1 or token.isdigit() or not any(char.isalpha() for char in token):  # Skip single chars/numbers
                continue
            if token in ['<init>', '<clinit>', '<p>']:  
                continue
            token = token.replace(':', '').replace(',', '').replace('$', '')  
            cleaned_texts.append(token.strip())

        return cleaned_texts

    df['texts'] = df['texts'].apply(lambda x: clean_text_tokens(x) if isinstance(x, list) else [])
    df['Code'] = df['texts'].apply(lambda x: ' '.join(x))  
    df['Code'] = df['Code'].apply(lambda x: re.sub(r'/\*.*?\*/|//.*', '', x, flags=re.DOTALL))  # Remove comments
    df['Code'] = df['Code'].apply(lambda x: x.replace('\n', ' ').replace('\r', ' ').replace('\t', ' '))  # Normalize whitespace
    df['Code'] = df['Code'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s.]', '', x))  # Remove non-alphanumeric characters except periods
    dff = df.groupby('Entity', as_index=False).agg({
        'Code': 'first',  
        **{col: 'first' for col in df.columns if col not in ['Entity', 'texts', 'Code']}
    })

    return dff

def get_module(df, labels):
    df = df.copy()
    df['Module'] = None

    for _, row in labels.iterrows():
        file_pattern = row['Entity']
        module = row['Module']
        matched_rows = df['Entity'].str.contains(file_pattern, regex=True, na=False)
        num_matches = matched_rows.sum()
        if num_matches > 0:
            df.loc[matched_rows, 'Module'] = module
    df = df[~df.Module.isna()]
    df['File_ID'] = range(1, len(df) + 1)
    df.rename(columns={'texts': 'Code'}, inplace=True)
    file_id_map = dict(zip(df['Entity'], df['File_ID']))
    
    return df[['File_ID', 'File', 'Entity', 'Code', 'Module']], file_id_map



def get_dependencies(file_id_map, df, df_dep, architecture):
    df_dep = df_dep.copy()
    df = df.copy()
    
    df_merged_source = pd.merge(df_dep, df[['Entity', 'Module']], left_on='Source', right_on='Entity', how='left')
    df_merged_source = df_merged_source.rename(columns={'Module': 'Source_Module'}).drop(columns=['Entity'])
    
    df_merged_target = pd.merge(df_merged_source, df[['Entity', 'Module']], left_on='Target', right_on='Entity', how='left')
    df_merged_target = df_merged_target.rename(columns={'Module': 'Target_Module'}).drop(columns=['Entity'])
    
    df_dep = df_merged_target.copy()
    df_dep = df_dep[(~df_dep.Source_Module.isna()) & (~df_dep.Target_Module.isna())]
    df_dep['Source_ID'] = df_dep['Source'].map(file_id_map)
    df_dep['Target_ID'] = df_dep['Target'].map(file_id_map)
    
    df_dep = df_dep[['Source_ID', 'Source', 'Source_Module', 'Target_ID', 'Target', 'Target_Module', 'Dependency_Type', 'Dependency_Count']]
    allowed_set = set(zip(architecture['Source'], architecture['Target']))
    df_dep['Allowed'] = df_dep.apply(
        lambda row: 1 if (row['Source_Module'], row['Target_Module']) in allowed_set or row['Source_Module'] == row['Target_Module']
        else 0,
        axis=1
    )
    df_dep = df_dep[df_dep.Source != df_dep.Target]
    df_dep.drop_duplicates(inplace=True)
    
    return df_dep

# extracting graphical centralities

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
    labels_path = os.path.abspath(labels_path)
    json_path = os.path.abspath(json_path)

    print(f"Processing {dataset_name}...")
    architecture, labels, root = load_labels(labels_path)

    df = read_json(json_path)
    df = cleaning(df, root)
    df_dep = extract_dependencies(df)
    df = clean_code(df)
    df, file_id_map = get_module(df, labels)
    df_dep = get_dependencies(file_id_map, df, df_dep, architecture)
    df = generate_Graph(df, df_dep)

    processed_dir = os.path.join("data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    df.to_csv(os.path.join(processed_dir, f'{dataset_name}_data.csv'), index=False)
    df_dep.to_csv(os.path.join(processed_dir, f'dep_{dataset_name}.csv'), index=False)

    print(f"Processing complete: {dataset_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON and labels files.")
    parser.add_argument('--labels', required=True, help="Path to the labels file")
    parser.add_argument('--json', required=True, help="Path to the JSON file")
    parser.add_argument('--name', required=True, help="Name of the dataset (used for output files)")
    args = parser.parse_args()
    
    main(args.labels, args.json, args.name)
