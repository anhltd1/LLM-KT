import json
import sys
from pathlib import Path

def peek_jsonl(file_path, num_lines=2):
    """Peek at JSONL file (multiple JSON objects, one per line)"""
    print(f"\n{'='*60}")
    print(f"Peeking at: {file_path}")
    print(f"{'='*60}\n")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                
                try:
                    obj = json.loads(line.strip())
                    print(f"Object {i + 1}:")
                    print("-" * 50)
                    print(f"Fields ({len(obj)} total):")
                    
                    for key, value in obj.items():
                        value_type = type(value).__name__
                        
                        if isinstance(value, (list, dict)):
                            if isinstance(value, list):
                                preview = f"[{len(value)} items]"
                            else:
                                preview = f"{{{len(value)} keys}}"
                        else:
                            value_str = str(value)
                            preview = value_str if len(value_str) <= 80 else value_str[:77] + "..."
                        
                        print(f"  - {key}: {value_type} = {preview}")
                    
                    print(f"\nFull structure:")
                    print(json.dumps(obj, indent=2, ensure_ascii=False))
                    print()
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {i + 1}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error: {e}")

def flatten_problem_json(input_file, output_file, fields_to_keep):
    """
    Flatten problem.json from JSONL format to JSON array with selected fields.
    
    Args:
        input_file: Input JSONL file (one JSON object per line)
        output_file: Output JSON file (array format)
        fields_to_keep: List of field names to keep
    """
    print(f"\nProcessing: {input_file}")
    print(f"Output: {output_file}")
    print(f"Fields to keep: {', '.join(fields_to_keep)}\n")
    
    try:
        problems = []
        total_lines = 0
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines += 1
                line = line.strip()
                
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                    
                    # Extract only the fields we need
                    filtered_obj = {}
                    
                    # Parse the 'detail' field if it exists (it's a string representation of a dict)
                    detail_data = {}
                    if 'detail' in obj and obj['detail']:
                        try:
                            # The detail field is a string representation of a Python dict
                            # We need to safely evaluate it
                            import ast
                            detail_data = ast.literal_eval(obj['detail'])
                        except:
                            detail_data = {}
                    
                    for field in fields_to_keep:
                        if field == 'problem_id':
                            # problem_id is at top level
                            filtered_obj[field] = obj.get(field, None)
                        elif field == 'concepts':
                            # concepts is at top level
                            filtered_obj[field] = obj.get(field, None)
                        elif field == 'options':
                            # options is 'option' in detail
                            filtered_obj[field] = detail_data.get('option', None)
                        else:
                            # Other fields are in detail
                            filtered_obj[field] = detail_data.get(field, None)
                    
                    problems.append(filtered_obj)
                    
                    # Progress indicator
                    if (i + 1) % 1000 == 0:
                        print(f"Processed {i + 1} lines...")
                
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {i + 1} due to JSON error: {e}")
                    continue
        
        print(f"\nTotal lines processed: {total_lines}")
        print(f"Total problems extracted: {len(problems)}")
        
        # Write to output file
        print(f"Writing to {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(problems, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Successfully flattened problem JSON file!")
        print(f"  Output: {len(problems)} problems")
        
        # Show a sample
        if problems:
            print(f"\nSample of first problem:")
            print(json.dumps(problems[0], indent=2, ensure_ascii=False))
        
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    input_file = "dataset/MOOCRadar/problem.json"
    output_file = "dataset/MOOCRadar/problem-flattened.json"
    
    # Fields to keep as specified by user
    fields_to_keep = [
        "problem_id",
        "title",
        "content",
        "options",
        "answer",
        "language",
        "concepts"
    ]
    
    # First, peek at the file
    print("=" * 60)
    print("STEP 1: PEEKING AT FILE STRUCTURE")
    print("=" * 60)
    peek_jsonl(input_file, num_lines=1)
    
    # Then flatten it
    print("\n" + "=" * 60)
    print("STEP 2: FLATTENING FILE")
    print("=" * 60)
    flatten_problem_json(input_file, output_file, fields_to_keep)
