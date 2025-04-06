from .lib.dequedict import DequeDict


class LRU:
    class LRU_Entry:
        def __init__(self, oblock, time=0):
            self.oblock = oblock
            self.time = time

        def __repr__(self):
            return "(o={}, t={})".format(self.oblock, self.time)

    def __init__(self, cache_size, **kwargs):
        self.cache_size = cache_size
        self.lru = DequeDict()

        self.time = 0
        # Complete hit history tracking
        self.complete_hit_history = []  # Will store the complete hit history

    def __contains__(self, oblock):
        return oblock in self.lru

    def cacheFull(self):
        return len(self.lru) == self.cache_size

    def addToCache(self, oblock):
        x = self.LRU_Entry(oblock, self.time)
        self.lru[oblock] = x

    def hit(self, oblock):
        x = self.lru[oblock]
        x.time = self.time
        self.lru[oblock] = x
        # Add to complete hit history
        self.complete_hit_history.append({
            "time": self.time,
            "oblock": oblock,
            "hit": True,
            "evicted": None
        })

    def evict(self):
        lru = self.lru.popFirst()
        return lru.oblock

    def miss(self, oblock):
        evicted = None

        if len(self.lru) == self.cache_size:
            evicted = self.evict()
        self.addToCache(oblock)
        
        # Add to complete hit history
        self.complete_hit_history.append({
            "time": self.time,
            "oblock": oblock,
            "hit": False,
            "evicted": evicted
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