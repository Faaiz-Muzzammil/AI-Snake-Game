class Deque:
    def __init__(self, maxlen=None):
        self.items = []
        self.maxlen = maxlen

    def append(self, item):
        if self.maxlen is not None and len(self.items) >= self.maxlen:
            self.popleft()
        self.items.append(item)

    def appendleft(self, item):
        if self.maxlen is not None and len(self.items) >= self.maxlen:
            self.pop()
        self.items.insert(0, item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def popleft(self):
        if not self.is_empty():
            return self.items.pop(0)

    def __getitem__(self, index):
        return self.items[index]

    def is_empty(self):
        return len(self.items) == 0

    def __len__(self):
        return len(self.items)
