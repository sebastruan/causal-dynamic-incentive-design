import collections

class CircularBufferMemory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = collections.deque(memory_size )

    def add(self, element):
        self.memory.append(element)

    def __len__(self):
        return len(self.memory)

        