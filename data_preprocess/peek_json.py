import json
import sys
from pathlib import Path

def peek_json(file_path, num_objects=1):
    """
    Load and display the first N objects from a JSON file to inspect their structure.
    
    Args:
        file_path: Path to the JSON file
        num_objects: Number of objects to peek at (default: 1)
    """
    print(f"\n{'='*60}")
    print(f"Peeking at: {file_path}")
    print(f"{'='*60}\n")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            print(f"JSON Type: List with {len(data)} items\n")
            objects_to_show = data[:num_objects]
        elif isinstance(data, dict):
            print(f"JSON Type: Dictionary with {len(data)} keys\n")
            # If it's a dict, show the first few items
            objects_to_show = [dict(list(data.items())[:num_objects])]
        else:
            print(f"JSON Type: {type(data)}\n")
            objects_to_show = [data]
        
        # Display each object
        for i, obj in enumerate(objects_to_show, 1):
            print(f"Object {i}:")
            print("-" * 50)
            
            if isinstance(obj, dict):
                print(f"Fields ({len(obj)} total):")
                for key, value in obj.items():
                    value_type = type(value).__name__
                    
                    # Show a preview of the value
                    if isinstance(value, (list, dict)):
                        if isinstance(value, list):
                            preview = f"[{len(value)} items]"
                        else:
                            preview = f"{{{len(value)} keys}}"
                    else:
                        value_str = str(value)
                        preview = value_str if len(value_str) <= 50 else value_str[:47] + "..."
                    
                    print(f"  - {key}: {value_type} = {preview}")
            else:
                print(f"  Type: {type(obj).__name__}")
                print(f"  Value: {obj}")
            
            print()
        
        # Show full structure of first object for detailed inspection
        if objects_to_show:
            print("\nFull JSON structure of first object:")
            print("-" * 50)
            print(json.dumps(objects_to_show[0], indent=2, ensure_ascii=False))
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Default JSON files to check
    json_files = [
        "dataset/MOOCRadar/student-problem-middle.json",
        "dataset/MOOCRadar/student-problem-fine.json",
        "dataset/MOOCRadar/student-problem-coarse.json",
        "dataset/MOOCRadar/problem.json"
    ]
    
    # If a file path is provided as argument, use that
    if len(sys.argv) > 1:
        json_files = [sys.argv[1]]
    
    # Peek at each file
    for json_file in json_files:
        file_path = Path(json_file)
        if file_path.exists():
            peek_json(json_file)
        else:
            print(f"Skipping {json_file} - file not found\n")
