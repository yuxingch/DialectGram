from scipy.io import loadmat

def load_data(path = "./data/data.mat"):
    return loadmat(path)
