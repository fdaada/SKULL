from alg.get_algorithm import get_algorithm
import os
import sys
import math
import json
from code.alg.skull import SKULL

if __name__ == '__main__':
    
    hits = 0
    requests = 0
    save_folder = '../log/'+sys.argv[3]
    os.makedirs(save_folder, exist_ok=True)
    save_file_path = os.path.join(save_folder, '$'.join(sys. argv[1:])+'.json')
    os.makedirs(os.path.dirname(save_file_path), exist_ok=True)
    assert not os.path.exists(save_file_path), f"Path {save_file_path} already exists!"
    cache_ratio = float(sys.argv[1])
    assert cache_ratio < 100, "The ratio should be less than 100%!"
    
    algorithm = sys.argv[2]
    trace_file = sys.argv[3]
    with open(trace_file, 'r') as file:   
        line_count = sum(1 for line in file)
    cache_size = int(cache_ratio*line_count/100)
    try:
        llm_intervention_interval_cache_ration = float(sys.argv[4])
        llm_intervention_interval = int(llm_intervention_interval_cache_ration * cache_size)
        model = sys.argv[5]
        reasoning_mode = sys.argv[6]
        assert reasoning_mode in ['low', 'high','medium']
    except:
        llm_intervention_interval = None
        model =None
        reasoning_mode = None
    if cache_size <= 0:
        print("Cache_size should be greater than 0")
        exit(1)
    if llm_intervention_interval and model:
        alg = SKULL(cache_size, llm_frequency=llm_intervention_interval,model=model, save_file_path=save_file_path, reasoning_mode = reasoning_mode)
    else:
        alg = get_algorithm(algorithm)(cache_size)
     
    with open(trace_file, 'r') as f:
        for line in f:
            lba = int(line)
            if lba < 0:
                continue
            requests += 1

            miss, evicted = alg.request(lba)

            if not miss:
                hits += 1

        misses = requests - hits
        print("Results: {:<10} size={:<8} hits={}, misses={}, ios={}, hitrate={:4}%, {}"
                .format(algorithm, cache_size, hits, misses, requests,
                round(100 * hits / requests, 2), trace_file))
    results_dict = {
            "algorithm": algorithm,
            "size": cache_size,
            "hits": hits,
            "misses": misses,
            "ios": requests,
            "hitrate": round(100 * hits / requests, 2),
            "trace_file": trace_file
            }
    log = {
            "result": results_dict, 
            "hit_history": alg.get_hit_history(),
        }
    if llm_intervention_interval and model:
        log["llm_call_log"] = alg.llm_call_log
    with open(save_file_path, 'w') as f:
        json.dump(log, f, indent=4)
