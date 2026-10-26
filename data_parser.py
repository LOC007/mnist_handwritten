import struct
import numpy as np
import torch

def load_images(path):
    with open(path,'rb')as f:
        header = f.read(16)
        raw = f.read()
        
    magic,n,rows,cols = struct.unpack('>IIII',header)
    arr = np.frombuffer(raw,dtype=np.uint8).reshape(n,rows,cols)
    arr = arr.astype(np.float32)/255.0
    x = torch.from_numpy(arr).unsqueeze(1)
    return x

def load_labels(path):
    with open(path,'rb')as f:
        header = f.read(8)
        raw = f.read()
    magic,n = struct.unpack('>II',header)
    arr = np.frombuffer(raw,dtype=np.uint8)
    x = torch.tensor(arr,dtype=torch.long)
    return x

if __name__=='__main__':
    x = load_images(r'D:\MNIST_Task\data\train-images-idx3-ubyte')
    y = load_labels(r'D:\MNIST_Task\data\train-labels-idx1-ubyte')
    print(x.shape,y.shape,y[0],y.min(),y.max())
