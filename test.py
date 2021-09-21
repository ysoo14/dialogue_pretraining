import pickle5 as pickle

path1 = r'./dataset/data.pkl'
dataset1= pickle.load(open(path1, 'rb'), encoding='latin1') 

path2 = r'./dataset/data2.pkl'
dataset2= pickle.load(open(path2, 'rb'), encoding='latin1') 

print(dataset1['encoded_utterance'][0][0].shape)
print(dataset2['contexts'][0][0].shape)