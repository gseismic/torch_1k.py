import numpy as np

class DataLoader:
    
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False) -> None:
        '''
        drop_last: 如果设置为 True，在数据集大小不能整除 batch_size 时，丢弃最后一个不足一个批次的数据。
        '''
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.data_size= len(dataset)
        self.max_iter = (self.data_size + self.batch_size -1)//self.batch_size
        
        self.reset()
    
    def reset(self):
        self.iteration = 0
        if self.shuffle:
            self.index = np.random.permutation(self.data_size)
        else:
            self.index = np.arange(self.data_size)
    
    def mark_end(self):
        self.reset()
        raise StopIteration
            
    def __next__(self):
        if self.iteration >= self.max_iter:
            self.reset()
            raise StopIteration
            
        i, batch_size = self.iteration, self.batch_size
        batch_index = self.index[i*batch_size: (i+1)*batch_size]
        if self.drop_last and len(batch_index) != batch_size:
            self.reset()
            raise StopIteration
        
        batch_data = [self.dataset[i] for i in batch_index]
        features = [tu[0] for tu in batch_data]
        labels = [tu[1] for tu in batch_data]

        self.iteration += 1
        return features, labels
    
    def next(self):
        return self.__next__()