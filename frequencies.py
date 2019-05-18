import util
from model import FrequencyModel

data = util.load_data()
vocab = [i[0] for i in data["vocab"][:, 0]]
inverted_vocab = {k:v for v,k in enumerate(vocab)}

docs = [doc[0, :] for doc in data["w_data"][:, 0]]

model = FrequencyModel(inverted_vocab)
model.fit(docs)
print(model.transform("fuck"))