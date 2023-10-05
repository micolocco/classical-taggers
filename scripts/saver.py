
class Saver():
    
    def __init__(self, eventType, tagger, repoPath, KaonCombiner, grid_n, optimized):
        
        self.eventType = eventType
        self.tagger = tagger
        self.repoPath = repoPath
        self.KaonCombiner = KaonCombiner
        self.grid_n = grid_n
        self.optimized = optimized

    # Define name format according to conditions set in configuration file
    def assign_name(self, folder, name):
        
        prePath = f'{self.repoPath}{folder}/{self.eventType}/{self.tagger}/'
        if self.grid_n != None: 
            if self.KaonCombiner:
                name = name + f"_combiner_{self.grid_n}"
            else:
                name = name + f"_{self.grid_n}"
        else:
            if self.optimized:
                name = name + "_optimized"
        return f"{prePath}{name}" 