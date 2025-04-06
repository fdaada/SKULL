import os
import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

def calculate_global_hit_rate(hits, window_size=100):
    """Calculate cumulative global hit rate over time"""
    cumulative_hits = np.cumsum(hits)
    indices = np.arange(1, len(hits) + 1)
    global_hit_rate = cumulative_hits / indices
    return global_hit_rate

def process_json_files(file_paths, labels, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not file_paths:
        print("No files provided")
        return
    
    print(f"Processing {len(file_paths)} files")
    
    # First plot: Compare global hit rates across all algorithms
    plt.figure(figsize=(12, 3))
    fontSize = 30
    legendFontSize = 30
    all_data = []  # Store data for each file to use later
    
    for i, (json_file, label) in enumerate(zip(file_paths, labels)):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if 'hit_history' not in data or not data['hit_history']:
                print(f"No hit_history in {json_file}, skipping...")
                continue
                
            hit_history = data['hit_history']
            
            # Extract time, hit, and policy info
            times = []
            hits = []
            policies = []
            
            for entry in hit_history:
                times.append(entry['time'])
                hits.append(1 if entry['hit'] else 0)
                policies.append(entry.get('policy', None))
            
            # Convert to numpy arrays for easier manipulation
            times = np.array(times)
            hits = np.array(hits)
            
            # Calculate global hit rate
            global_hit_rate = calculate_global_hit_rate(hits)
            
            # Store data for later use in individual plots
            policy_changes = []
            if any(p is not None for p in policies):
                # Find the first non-None policy
                first_policy = next((p for p in policies if p is not None), None)
                prev_policy = first_policy
                
                for j, policy in enumerate(policies):
                    if policy is not None and policy != prev_policy:
                        policy_changes.append(times[j])
                        prev_policy = policy
            
            all_data.append({
                'file': json_file,
                'label': label,
                'times': times,
                'hits': hits,
                'global_hit_rate': global_hit_rate,
                'policy_changes': policy_changes
            })
            
            # Plot global hit rate for comparison
            plt.plot(times, global_hit_rate, label=label)
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Finalize combined plot
    plt.title('Global Hit Rate Comparison', fontsize=fontSize)
    plt.xlabel('Requests', fontsize=fontSize)
    plt.ylabel('Global Hit Rate', fontsize=fontSize)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=legendFontSize)
    
    # Save the comparison plot
    comparison_plot_path = os.path.join(output_dir, 'global_hit_rate_comparison.pdf')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated comparison plot at {comparison_plot_path}")
    
    # Second set of plots: Individual algorithms with policy changes
    for data in all_data:
        if data['policy_changes']:  # Only create this plot if there are policy changes
            plt.figure(figsize=(12, 3))
            
            # Plot global hit rate
            plt.plot(data['times'], data['global_hit_rate'], label=data['label'], color='blue')
            len_requests = len(data['times']) 
            # Plot vertical lines for policy changes
            cur_plot_policy_change_time = 0
            for change_time in data['policy_changes']:
            
                if change_time - cur_plot_policy_change_time > 0.0001*len_requests:
                    cur_plot_policy_change_time = change_time
                    plt.axvline(x=change_time, color='red', linestyle='--', alpha=0.7)
            
            plt.title(f'Global Hit Rate with Policy Changes - {data["label"]}', fontsize=fontSize)
            plt.xlabel('Requests', fontsize=fontSize)
            plt.ylabel('Global Hit Rate', fontsize=fontSize)
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=legendFontSize)
            
            # Create a sanitized filename from the label
            safe_label = ''.join(c if c.isalnum() else '_' for c in data['label'])
            
            # Save the individual plot
            individual_plot_path = os.path.join(output_dir, f'global_hit_rate_policy_{safe_label}.pdf')
            plt.savefig(individual_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Generated individual plot for {data['label']}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process JSON files and create hit rate plots')
    parser.add_argument('--files', nargs='+', required=True, 
                        help='List of JSON file paths to process')
    parser.add_argument('--labels', nargs='+', required=True, 
                        help='Labels for each file (algorithm names)')
    parser.add_argument('--output-dir', type=str, default='./plots', 
                        help='Output directory for plots (default: ./plots)')
    
    args = parser.parse_args()
    
    # Validate input
    if len(args.files) != len(args.labels):
        print("Error: Number of files must match number of labels")
        exit(1)
    
    for file_path in args.files:
        if not os.path.isfile(file_path):
            print(f"Error: {file_path} is not a valid file")
            exit(1)
    
    process_json_files(args.files, args.labels, args.output_dir)
    print(f"All plots saved to {args.output_dir}")