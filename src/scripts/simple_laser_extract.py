import os
import json
import glob

def extract_laser_data():
    """
    Simple extraction of LASER database records for training data
    """
    print("=== Simple LASER Data Extraction ===")
    
    # Find all LASER record files
    laser_dir = "../../laser_release/database_store"
    record_files = []
    
    for year in ['2014', '2015']:
        year_dir = os.path.join(laser_dir, year)
        if os.path.exists(year_dir):
            record_files.extend(glob.glob(os.path.join(year_dir, "*.txt")))
    
    print(f"Found {len(record_files)} LASER record files")
    
    # Extract basic information from each record
    extracted_data = []
    
    # Process ALL files, not just first 10
    for record_file in record_files:
        try:
            with open(record_file, 'r') as f:
                content = f.read()
            
            # Parse basic key-value pairs
            data = {}
            for line in content.split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    data[key.strip()] = value.strip().strip('"')
            
            # Extract relevant information
            record = {
                'file': os.path.basename(record_file),
                'title': data.get('Title', ''),
                'doi': data.get('DOI', ''),
                'year': data.get('Year', ''),
                'tags': data.get('Tags', ''),
                'target_product': data.get('Mutant1.TargetMolecule', ''),
                'fold_improvement': data.get('Mutant1.FoldImprovement', ''),
                'final_yield': data.get('Mutant1.FinalYield', ''),
                'mutations': []
            }
            
            # Extract mutations (simplified)
            mutation_count = 0
            for i in range(1, 10):  # Check up to 10 mutations
                gene_key = f'Mutant1.Mutation{i}.GeneName'
                if gene_key in data:
                    mutation = {
                        'gene': data[gene_key],
                        'changes': data.get(f'Mutant1.Mutation{i}.Changes', ''),
                        'effects': data.get(f'Mutant1.Mutation{i}.Effects', '')
                    }
                    record['mutations'].append(mutation)
                    mutation_count += 1
                else:
                    break
            
            # Include ALL records with mutations, not just those with target products
            if record['mutations']:  # Only include records with mutations
                extracted_data.append(record)
                
        except Exception as e:
            print(f"Error processing {record_file}: {e}")
            continue
    
    print(f"Successfully extracted {len(extracted_data)} records")
    
    # Less restrictive filtering - include more metabolic engineering examples
    metabolic_records = []
    for record in extracted_data:
        # Include records with any metabolic engineering relevance
        if (record['mutations'] and  # Has mutations
            any(mut['gene'] for mut in record['mutations']) and  # Has gene names
            record.get('fold_improvement', '')):  # Has performance data
            metabolic_records.append(record)
    
    print(f"Found {len(metabolic_records)} metabolic engineering records")
    
    # Print sample of target products to see what's available
    print("\nSample target products found:")
    for record in extracted_data[:20]:  # Show first 20
        if record['target_product']:
            print(f"  {record['target_product']} (from {record['file']})")
    
    # Create training data
    training_data = []
    for record in metabolic_records:
        features = {
            'num_mutations': len(record['mutations']),
            'has_knockout': any('del' in mut['changes'].lower() for mut in record['mutations']),
            'has_overexpression': any('oe' in mut['changes'].lower() for mut in record['mutations']),
            'has_plasmid': any('plasmid' in mut['changes'].lower() for mut in record['mutations']),
            'fold_improvement': float(record['fold_improvement']) if record['fold_improvement'] and record['fold_improvement'] != 'nan' else 0,
            'final_yield': float(record['final_yield']) if record['final_yield'] and record['final_yield'] != 'nan' else 0
        }
        
        target_genes = [mut['gene'] for mut in record['mutations'] if mut['gene']]
        
        training_data.append({
            'features': features,
            'target_genes': target_genes,
            'record': record
        })
    
    # Save the data
    os.makedirs('../../data', exist_ok=True)
    with open('../../data/laser_training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2, default=str)
    
    print(f"Created {len(training_data)} training examples")
    print("Training data saved to data/laser_training_data.json")
    
    # Print some examples
    if training_data:
        print("\nExample training data:")
        for i, example in enumerate(training_data[:3]):
            print(f"Example {i+1}:")
            print(f"  Features: {example['features']}")
            print(f"  Target genes: {example['target_genes']}")
            print(f"  Paper: {example['record']['title']}")
            print()
    
    return training_data

if __name__ == "__main__":
    extract_laser_data() 