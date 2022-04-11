import numpy as np

class DataLoader: 

    def __init__(self, dataset, labels) -> None:
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, index): 
        return self.dataset[index], self.labels[index]

    def __len__(self): 
        return len(self.dataset)

    def get_batches(self, batch_size, batch_norm=True): 
        indicies = np.arange(0, len(self.dataset))
        np.random.shuffle(indicies)

        for i in range(0, len(indicies), batch_size+1):
            yield (self.__getitem__(indicies[i:i+batch_size]))

        return None, None