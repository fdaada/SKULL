from .lib.dequedict import DequeDict
from .lib.heapdict import HeapDict
import numpy as np
import re
from openai import OpenAI
client = OpenAI()
import time
is_print = False


class SKULL:
    ######################
    ## INTERNAL CLASSES ##
    ######################

    # Entry to track the page information
    class LeCaR_Entry:
        def __init__(self, oblock, freq=1, time=0):
            self.oblock = oblock
            self.freq = freq
            self.time = time
            self.evicted_time = None

        # Minimal comparitors needed for HeapDict
        def __lt__(self, other):
            if self.freq == other.freq:
                return self.oblock < other.oblock
            return self.freq < other.freq

        # Useful for debugging
        def __repr__(self):
            return "(o={}, f={}, t={})".format(self.oblock, self.freq,
                                               self.time)

    # kwargs: We're using keyword arguments so that they can be passed down as
    #         needed. Please note that cache_size is a required argument and not
    #         optional like all the kwargs are.
    def __init__(self, cache_size, **kwargs):
        # Randomness and Time
        np.random.seed(123)
        self.time = 0

        # Cache
        self.cache_size = cache_size
        self.lru = DequeDict()
        self.lfu = HeapDict()

        # Histories
        self.history_size = cache_size
        self.lru_hist = DequeDict()
        self.lfu_hist = DequeDict()

        # Decision Weights Initilized 
        self.initial_weight = 0.5
        self.W = np.array([self.initial_weight, 1 - self.initial_weight],
                          dtype=np.float32)

        # LLM Decision Parameters
        self.llm_decision_interval = kwargs.get('llm_frequency', 100)  # Call LLM every N accesses
        self.access_count = 0
        
        # Memory of past events for LLM
        self.recent_decisions = []  # Track recent policy decisions
        self.recent_hits = []  # Track recent hits/misses
        self.recent_history_hits = []  # Track if misses were in history
        self.recent_objects = []  # Track recently accessed objects
        
        # Default policy until first LLM decision
        self.current_policy = 0  # Start with LRU (0) by default
        
        # Statistics for LLM
        self.lru_hits = 0
        self.lfu_hits = 0
        self.total_hits = 0
        self.total_misses = 0
        self.history_hit_counts = {
            "lru": 0,  # Hits in LRU history
            "lfu": 0   # Hits in LFU history
        }
        
        # Max items to track for LLM context
        self.max_history_items = 20
        
        # # OpenAI Configuration
        # self.api_key = kwargs.get('api_key', None)
        # if self.api_key:
        #     openai.api_key = self.api_key
        self.model = kwargs.get('model', 'gpt-4o')
        
        # Logs and debugging
        self.log_llm_calls = True
        self.llm_call_log = []
        self.reasoning_mode = kwargs.get('reasoning_mode', None)
        # Retry parameters for API calls
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        print("Using model:", self.model)
        print("LLM decision interval:", self.llm_decision_interval)
        if self.reasoning_mode:
            print("LLM reasoning mode: ", self.reasoning_mode)
        # Complete hit history tracking
        self.complete_hit_history = []  # Will store the complete hit history

    # True if oblock is in cache (which LRU can represent)
    def __contains__(self, oblock):
        return oblock in self.lru

    def cacheFull(self):
        return len(self.lru) == self.cache_size

    # Add Entry to cache with given frequency
    def addToCache(self, oblock, freq):
        x = self.LeCaR_Entry(oblock, freq, self.time)
        self.lru[oblock] = x
        self.lfu[oblock] = x

    # Add Entry to history dictated by policy
    # policy: 0, Add Entry to LRU History
    #         1, Add Entry to LFU History
    def addToHistory(self, x, policy):
        # Use reference to policy_history to reduce redundant code
        policy_history = None
        if policy == 0:
            policy_history = self.lru_hist
        elif policy == 1:
            policy_history = self.lfu_hist
        elif policy == -1:
            return

        # Evict from history is it is full
        if len(policy_history) == self.history_size:
            evicted = self.getLRU(policy_history)
            del policy_history[evicted.oblock]
        policy_history[x.oblock] = x

    # Get the LRU item in the given DequeDict
    def getLRU(self, dequeDict):
        return dequeDict.first()

    def getHeapMin(self):
        return self.lfu.min()

    # Get the choice based on LLM decision instead of weights
    def getChoice(self):
        return self.current_policy
    
    # Make decision using LLM via OpenAI API
    def makeLLMDecision(self):
        # assert self.api_key, "No api key is given."

        
        # Prepare summary information for LLM
        lru_hist_unique = len(self.lru_hist)
        lfu_hist_unique = len(self.lfu_hist)
        
        # Calculate working set size (unique objects accessed recently)
        working_set = set(self.recent_objects[-min(100, len(self.recent_objects)):])
        working_set_size = len(working_set)
        
        # Calculate recency bias (if recent objects tend to be accessed again)
        recency_pattern = []
        for i in range(min(20, len(self.recent_objects)-1)):
            recency_pattern.append(1 if self.recent_objects[-(i+1)] in self.recent_objects[-min(i, len(self.recent_objects)):] else 0)
        
        recency_bias = sum(recency_pattern) / len(recency_pattern) if recency_pattern else 0
            
        context = {
            "cache_size": self.cache_size,
            "total_accesses": self.time,
            "hit_rate": round((self.total_hits / self.time) * 100, 2) if self.time > 0 else 0,
            "lru_history_size": lru_hist_unique,
            "lfu_history_size": lfu_hist_unique,
            "lru_history_hits": self.history_hit_counts["lru"],
            "lfu_history_hits": self.history_hit_counts["lfu"],
            "working_set_size": working_set_size,
            "working_set_to_cache_ratio": working_set_size / self.cache_size if self.cache_size > 0 else 0,
            "recency_bias": round(recency_bias * 100, 2),
            "recent_policy_choices": self.recent_decisions[-10:] if self.recent_decisions else [],
            "recent_hit_rate": round((sum(self.recent_hits[-50:]) / len(self.recent_hits[-50:])) * 100, 2) if self.recent_hits else 0
        }
        
        # Create a formatted prompt for the LLM
        prompt = self._create_llm_prompt(context)
        
        # Call OpenAI API with exponential backoff for retries
        for attempt in range(self.max_retries):
            try:
                if self.reasoning_mode:
                    print("Reasoning effort is ", self.reasoning_mode)
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a cache replacement policy expert. Your task is to analyze access patterns and decide between LRU (Least Recently Used) and LFU (Least Frequently Used) policies."},
                            {"role": "user", "content": prompt}
                        ],
                         reasoning_effort=self.reasoning_mode
                    )
                else:    
                    response = client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a cache replacement policy expert. Your task is to analyze access patterns and decide between LRU (Least Recently Used) and LFU (Least Frequently Used) policies."},
                            {"role": "user", "content": prompt}
                        ],
                    )
                
                # Extract the decision from the response
                decision_text = response.choices[0].message.content.strip().lower()
                
                # Log the interaction if enabled
                if self.log_llm_calls:
                    self.llm_call_log.append({
                        "time": self.time,
                        "context": context,
                        "prompt": prompt,
                        "response": decision_text
                    })
                    if is_print:
                        print(decision_text)
                # Parse the decision
                policy_match = re.search(r"<policy>Policy: (\d)</policy>", decision_text)
                if policy_match:
                    policy_number = int(policy_match.group(1))
                    if policy_number in [0, 1]:
                        return policy_number
                    
                # Fallback parsing if the tag format is not followed
                if "lru" in decision_text.lower() and "policy: 0" in decision_text.lower():
                    return 0  # Choose LRU
                elif "lfu" in decision_text.lower() and "policy: 1" in decision_text.lower():
                    return 1  # Choose LFU
                else:
                    # If response is unclear, try to extract the policy number
                    if "policy: 0" in decision_text.lower() or "policy 0" in decision_text.lower():
                        return 0
                    elif "policy: 1" in decision_text.lower() or "policy 1" in decision_text.lower():
                        return 1
                    else:
                        # Default to balanced approach if parsing fails
                        return 0 if np.random.random() < 0.5 else 1
                        
            except Exception as e:
                print(e)
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    sleep_time = self.retry_delay * (2 ** attempt)
                    time.sleep(sleep_time)
                else:
                    # If all retries fail, fall back to simple heuristic
                    return 0 if self.history_hit_counts["lru"] >= self.history_hit_counts["lfu"] else 1
        
        # Default fallback if API call fails
        return 0 if np.random.random() < 0.5 else 1  # Random choice as last resort
    
    # Create a well-structured prompt for the LLM
    def _create_llm_prompt(self, context):
        prompt = f"""
I need to decide between two cache replacement policies for a memory system.

CURRENT STATISTICS:
- Cache size: {context['cache_size']} objects
- Total accesses: {context['total_accesses']}
- Current hit rate: {context['hit_rate']}%
- Recent hit rate (last 50 accesses): {context['recent_hit_rate']}%

ACCESS PATTERN INSIGHTS:
- Working set size: {context['working_set_size']} unique objects
- Working set to cache ratio: {context['working_set_to_cache_ratio']:.2f}
- Recency bias (likelihood of recently accessed objects being accessed again): {context['recency_bias']}%

POLICY PERFORMANCE:
- LRU history hits: {context['lru_history_hits']} (items that were evicted by LRU but would be useful)
- LFU history hits: {context['lfu_history_hits']} (items that were evicted by LFU but would be useful)
- Recent policy choices: {context['recent_policy_choices']}

Based on this data, which policy should I choose for the next batch of accesses?
- LRU (policy 0): Evicts the least recently used item
- LFU (policy 1): Evicts the least frequently used item

Provide your decision as "<policy>Policy: X</policy>" where X is either 0 (for LRU) or 1 (for LFU), after a brief explanation.
"""
        return prompt

    # Evict an entry
    def evict(self):
        lru = self.getLRU(self.lru)
        lfu = self.getHeapMin()

        evicted = lru
        policy = self.getChoice()

        # Since we're using Entry references, we use is to check
        # that the LRU and LFU Entries are the same Entry
        if lru is lfu:
            evicted, policy = lru, -1
        elif policy == 0:
            evicted = lru
        else:
            evicted = lfu

        del self.lru[evicted.oblock]
        del self.lfu[evicted.oblock]

        evicted.evicted_time = self.time

        self.addToHistory(evicted, policy)
        
        # Track decisions for LLM context
        self.recent_decisions.append(policy)
        if len(self.recent_decisions) > self.max_history_items:
            self.recent_decisions.pop(0)

        return evicted.oblock, policy

    # Cache Hit
    def hit(self, oblock):
        x = self.lru[oblock]
        x.time = self.time

        self.lru[oblock] = x

        x.freq += 1
        self.lfu[oblock] = x
        
        # Track hits for analysis
        self.total_hits += 1
        
        # Track which policy would have kept this item
        # if x is self.getLRU(self.lru):
        #     self.lfu_hits += 1  # This would be evicted by LRU but kept by LFU
        # elif x is self.getHeapMin():
        #     self.lru_hits += 1  # This would be evicted by LFU but kept by LRU
        
        # Track hit info for LLM context
        self.recent_hits.append(1)  # 1 for hit
        if len(self.recent_hits) > self.max_history_items * 2:  # Keep more hit/miss history
            self.recent_hits.pop(0)
            
        # Track accessed object
        self.recent_objects.append(oblock)
        if len(self.recent_objects) > self.max_history_items * 5:  # Keep more object history
            self.recent_objects.pop(0)
             
        # Add to complete hit history
        self.complete_hit_history.append({
            "time": self.time,
            "oblock": oblock,
            "hit": True,
            "policy": self.current_policy,
            "evicted": None,
            "freq": x.freq
        })

    # Cache Miss
    def miss(self, oblock):
        evicted = None

        freq = 1
        history_hit = -1  # -1 means no history hit
        
        # Check if the missed block is in either history
        if oblock in self.lru_hist:
            entry = self.lru_hist[oblock]
            freq = entry.freq + 1
            del self.lru_hist[oblock]
            history_hit = 0  # In LRU history
            self.history_hit_counts["lru"] += 1
        elif oblock in self.lfu_hist:
            entry = self.lfu_hist[oblock]
            freq = entry.freq + 1
            del self.lfu_hist[oblock]
            history_hit = 1  # In LFU history
            self.history_hit_counts["lfu"] += 1
            
        # Track history hit info for LLM
        self.recent_history_hits.append(history_hit)
        if len(self.recent_history_hits) > self.cache_size:
            self.recent_history_hits.pop(0)
        
        # Track miss for statistics
        self.total_misses += 1
        self.recent_hits.append(0)  # 0 for miss
        if len(self.recent_hits) > self.max_history_items * 2:
            self.recent_hits.pop(0)
            
        # Track accessed object
        self.recent_objects.append(oblock)
        if len(self.recent_objects) > self.max_history_items * 5:
            self.recent_objects.pop(0)

        # If the cache is full, evict
        policy_used = None
        if len(self.lru) == self.cache_size:
            evicted, policy_used = self.evict()

        self.addToCache(oblock, freq)
        
        # Add to complete hit history
        self.complete_hit_history.append({
            "time": self.time,
            "oblock": oblock,
            "hit": False,
            "policy": policy_used,
            "evicted": evicted,
            "freq": freq,
            "history_hit": history_hit
        })

        return evicted

    # Process and access request for the given oblock
    def request(self, oblock):
        miss = True
        evicted = None

        self.time += 1
        self.access_count += 1
        
        # Check if it's time to use LLM for a decision
        if self.access_count >= self.llm_decision_interval:
            self.current_policy = self.makeLLMDecision()
            if is_print:
                print("\nLLM is making a decision: ", self.current_policy, "\n")
            self.access_count = 0

        if oblock in self:
            miss = False
            self.hit(oblock)
        else:
            evicted = self.miss(oblock)

        return miss, evicted
        
    # Utility method to get current statistics (useful for debugging or monitoring)
    def get_statistics(self):
        return {
            "time": self.time,
            "hit_rate": (self.total_hits / self.time) * 100 if self.time > 0 else 0,
            "lru_history_hits": self.history_hit_counts["lru"],
            "lfu_history_hits": self.history_hit_counts["lfu"],
            "current_policy": "LRU" if self.current_policy == 0 else "LFU",
            "cache_size": self.cache_size,
            "cache_utilization": len(self.lru) / self.cache_size * 100 if self.cache_size > 0 else 0,
            "llm_calls": len(self.llm_call_log) if hasattr(self, 'llm_call_log') else 0
        }
        
    # Method to get complete hit history or a subset of it
    def get_hit_history(self, last_n=None):
        if last_n is None:
            return self.complete_hit_history
        else:
            return self.complete_hit_history[-last_n:]