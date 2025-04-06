#### SKULL Source Code

The source code for the HotStorage Submission for SKULL. The implementation is built on top of LeCaR.

#### Code source

The LRU, LFU, LeCaR, and SKULL implementations can be found in the code/alg folder.
Additional code for data structure implementations are in the code/alg/lib folder. 

To run experiments, the arguments can be modified appropriately with the specific parameters such as cache size, input trace and algorithm name.
Executing the following command inside the code directory will produce the results: 

> To use SKULL, you need to provide OpenAI API Key. This can be done with: `export OPENAI_API_KEY=your_key_here`

The code can be run simply with:

```python3 run.py <cache_size> <algorithm> <trace_name>```

For instance, running the commands 

```python3 run.py 4 lru data.txt```

will produce results in the following output format

```Results: lru        size=4        hits=181, misses=319, ios=500, hitrate=36.2%, data.txt```\

#### Traces

The synthetic trace with 4 phase changes and the source code to generate phase changes in the working set can be found in the /synth-traces folder.

The FIU workloads (day 3) are publicly available on the [SNIA Website](http://iotta.snia.org/tracetypes/3).
The MSR workloads are publicly available at this [CMU URL](https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/msr/)
