import json
import sys
from pathlib import Path

def flatten_student_problem_json(input_file, output_file):
    """
    Flatten nested student-problem JSON structure.
    
    Input structure: [{seq: [{interaction1}, {interaction2}, ...]}, ...]
    Output structure: [{interaction1}, {interaction2}, ...]
    """
    print(f"Reading from: {input_file}")
    print(f"Output will be written to: {output_file}")
    
    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} student sequences")
        
        # Flatten the structure
        flattened_interactions = []
        total_interactions = 0
        
        for i, student_seq in enumerate(data):
            if 'seq' in student_seq:
                interactions = student_seq['seq']
                flattened_interactions.extend(interactions)
                total_interactions += len(interactions)
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1}/{len(data)} sequences, {total_interactions} interactions so far...")
        
        print(f"\nTotal interactions extracted: {total_interactions}")
        
        # Write the flattened data to output file
        print(f"Writing to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(flattened_interactions, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully flattened JSON file!")
        print(f"  Input: {len(data)} objects (nested)")
        print(f"  Output: {total_interactions} interactions (flat)")
        
        # Show a sample of the first interaction
        if flattened_interactions:
            print(f"\nSample of first interaction:")
            print(json.dumps(flattened_interactions[0], indent=2, ensure_ascii=False))
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Default files
    input_file = "dataset/MOOCRadar/student-problem-coarse.json"
    output_file = "dataset/MOOCRadar/student-problem-coarse-flattened.json"
    
    # Allow command-line arguments
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    flatten_student_problem_json(input_file, output_file)
