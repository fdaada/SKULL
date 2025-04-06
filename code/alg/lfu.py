from .lib.heapdict import HeapDict


class LFU:
    class LFU_Entry:
        def __init__(self, oblock, freq=1, time=0):
            self.oblock = oblock
            self.freq = freq
            self.time = time

        def __lt__(self, other):
            if self.freq == other.freq:
                return self.time > other.time
            return self.freq < other.freq

        def __repr__(self):
            return "(o={}, f={}, t={})".format(self.oblock, self.freq,
                                               self.time)

    def __init__(self, cache_size, **kwargs):
        self.cache_size = cache_size
        self.lfu = HeapDict()
        self.time = 0
        # Complete hit history tracking
        self.complete_hit_history = []  # Will store the complete hit history

    def __contains__(self, oblock):
        return oblock in self.lfu

    def cacheFull(self):
        return len(self.lfu) == self.cache_size

    def addToCache(self, oblock):
        x = self.LFU_Entry(oblock, freq=1, time=self.time)
        self.lfu[oblock] = x

    def hit(self, oblock):
        x = self.lfu[oblock]
        x.freq += 1
        x.time = self.time
        self.lfu[oblock] = x
        # Add to complete hit history
        self.complete_hit_history.append({
            "time": self.time,
            "oblock": oblock,
            "hit": True,
            "evicted": None,
            "freq": x.freq
        })

    def evict(self):
        lfu_min = self.lfu.popMin()
        return lfu_min.oblock

    def miss(self, oblock):
        evicted = None

        if len(self.lfu) == self.cache_size:
            evicted = self.evict()
        self.addToCache(oblock)

 
        # Add to complete hit history
        self.complete_hit_history.append({
            "time": self.time,
            "oblock": oblock,
            "hit": False,
            "evicted": evicted,
            "freq": 1
        })

        return evicted

    def request(self, oblock):
        miss = True
        evicted = None

        self.time += 1

        if oblock in self:
            miss = False
            self.hit(oblock)
        else:
            evicted = self.miss(oblock)


        return miss, evicted

    # Method to get complete hit history or a subset of it
    def get_hit_history(self, last_n=None):
        if last_n is None:
            return self.complete_hit_history
        else:
            return self.complete_hit_history[-last_n:]