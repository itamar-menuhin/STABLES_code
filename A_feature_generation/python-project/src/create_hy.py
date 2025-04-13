def create_hybrid_data_structure(data):
    # Function to create a hybrid data structure from the given data
    hybrid_structure = {}
    for item in data:
        # Example logic to create a hybrid structure
        hybrid_structure[item['id']] = {
            'name': item['name'],
            'value': item['value'],
            'metadata': item.get('metadata', {})
        }
    return hybrid_structure

def handle_hybrid_process(data):
    # Function to handle processes related to the hybrid data structure
    hybrid_data = create_hybrid_data_structure(data)
    # Additional processing logic can be added here
    return hybrid_data

# Exporting functions for use in other scripts
if __name__ == "__main__":
    # Example usage
    sample_data = [
        {'id': 1, 'name': 'Item 1', 'value': 10},
        {'id': 2, 'name': 'Item 2', 'value': 20}
    ]
    hybrid_data = create_hybrid_data_structure(sample_data)
    print(hybrid_data)