import pickle

with open('/home/ubuntu/ai-model/models/light_gcn/data/id_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)
    print(mappings.keys())
    print(type(mappings))