import os
import json
import re
import argparse
from pathlib import Path

def calculate_switch_rate(data):
    """
    Calculate how often the LLM policy strategy switches between LRU (0) and LFU (1).
    
    Args:
        data: A list of dictionaries containing LLM intervention data
        
    Returns:
        float: The switch rate (percentage of transitions where the policy changed)
        list: The sequence of policies chosen
    """
    # Extract policy decisions using regex
    policies = []
    
    for item in data:
        response = item.get('response', '')
        
        # Try to find policy in the standard format
        policy_match = re.search(r"<policy>policy: (\d)</policy>", response.lower())
        if policy_match:
            policy_number = int(policy_match.group(1))
            if policy_number in [0, 1]:
                policies.append(policy_number)
                continue
                
        # Fallback parsing methods
        if "lru" in response.lower() and "policy: 0" in response.lower():
            policies.append(0)  # Choose LRU
        elif "lfu" in response.lower() and "policy: 1" in response.lower():
            policies.append(1)  # Choose LFU
        elif "policy: 0" in response.lower() or "policy 0" in response.lower():
            policies.append(0)
        elif "policy: 1" in response.lower() or "policy 1" in response.lower():
            policies.append(1)
        else:
            # If no policy can be determined, skip this item
            continue
    
    # Calculate switch rate
    if len(policies) <= 1:
        return 0.0, policies  # No switches possible with 0 or 1 decisions
    
    # Count transitions where the policy changed
    switches = sum(1 for i in range(1, len(policies)) if policies[i] != policies[i-1])
    
    # Calculate switch rate as percentage of transitions
    switch_rate = (switches / (len(policies) - 1)) * 100
    
    return switch_rate, policies

def is_valid_json_file(file_path):
    """Check if a file contains valid JSON data."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Check if the data looks like our expected format
            if isinstance(data, list) and len(data) > 0:
                # Check at least one item has the expected structure
                for item in data:
                    if isinstance(item, dict) and 'response' in item:
                        return True
        return False
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    except Exception:
        return False

def main():
    parser = argparse.ArgumentParser(description='Analyze LLM intervention strategy switch rate across JSON files.')
    parser.add_argument('directory', type=str, help='Directory containing data files to analyze')
    parser.add_argument('--output', type=str, help='Output file to save results (optional)', default=None)
    parser.add_argument('--recursive', action='store_true', help='Recursively search subdirectories')
    args = parser.parse_args()
    
    directory = Path(args.directory)
    
    if not directory.exists() or not directory.is_dir():
        print(f"Error: {directory} is not a valid directory")
        return
    
    results = {}
    overall_policies = []
    
    # Function to process a file
    def process_file(file_path):
        try:
            if not is_valid_json_file(file_path):
                return
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            switch_rate, policies = calculate_switch_rate(data)
            
            # Store results for this file
            results[file_path.name] = {
                'switch_rate': switch_rate,
                'policy_count': len(policies),
                'lru_count': policies.count(0),
                'lfu_count': policies.count(1),
                'policies': policies
            }
            
            # Add to overall policies for aggregate statistics
            overall_policies.extend(policies)
            
            print(f"Processed {file_path.name}: Switch Rate = {switch_rate:.2f}%, "
                  f"Policies = {len(policies)} ({policies.count(0)} LRU, {policies.count(1)} LFU)")
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    # Find and process all files
    processed_count = 0
    skipped_count = 0
    
    if args.recursive:
        # Recursively process all files in all subdirectories
        for root, _, files in os.walk(directory):
            for filename in files:
                file_path = Path(root) / filename
                if is_valid_json_file(file_path):
                    process_file(file_path)
                    processed_count += 1
                else:
                    skipped_count += 1
    else:
        # Process only files in the main directory
        for file_path in directory.iterdir():
            if file_path.is_file():
                if is_valid_json_file(file_path):
                    process_file(file_path)
                    processed_count += 1
                else:
                    skipped_count += 1
    
    print(f"\nProcessed {processed_count} files, skipped {skipped_count} non-valid files.")
    
    if processed_count == 0:
        print("No valid data files found.")
        return
    
    # Calculate aggregate statistics across all files
    if overall_policies:
        # Count overall transitions where policy changed
        overall_switches = sum(1 for i in range(1, len(overall_policies)) 
                              if overall_policies[i] != overall_policies[i-1])
        
        # Calculate overall switch rate
        overall_switch_rate = (overall_switches / (len(overall_policies) - 1)) * 100 if len(overall_policies) > 1 else 0
        
        results['aggregate'] = {
            'total_decisions': len(overall_policies),
            'lru_count': overall_policies.count(0),
            'lfu_count': overall_policies.count(1),
            'overall_switch_rate': overall_switch_rate
        }
        
        print("\nAggregate Statistics:")
        print(f"Total Decisions: {len(overall_policies)}")
        print(f"LRU Decisions: {overall_policies.count(0)} ({(overall_policies.count(0)/len(overall_policies))*100:.2f}%)")
        print(f"LFU Decisions: {overall_policies.count(1)} ({(overall_policies.count(1)/len(overall_policies))*100:.2f}%)")
        print(f"Overall Switch Rate: {overall_switch_rate:.2f}%")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()