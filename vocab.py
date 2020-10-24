
import os, sys
import json
from collections import defaultdict

class mkvocab:
    """
    data path: vocab path to create or load

    """
    def __init__(self, data_path): 
        if os.path.exists(data_path)==False:
            os.mkdir(data_path)
        Vocab = defaultdict(lambda:0) #unk -> 0
        word_freq = defaultdict(lambda:0)
        if os.path.isfile(data_path+'/vocab.json'):
            sys.stdout.write("load vocab from {}\n".format(data_path))
            with open(data_path+'/vocab.json','r') as f:
                Vocab.update(json.load(f))
            with open(data_path+'/vocab_word_freq.json','r') as f:
                word_freq.update(json.load(f))
            self.Vocab = Vocab
            self.word_freq = word_freq
            self.n_Vocab = len(Vocab)
            id2vocab = [(word, index) for i, (word, index) in enumerate(Vocab.items()) if i==index]
            assert len(id2vocab) == len(list(Vocab.keys()))
            id2vocab.sort(key=lambda x:x[-1])
            self.id2vocab = [word for word, id in id2vocab]
            sys.stdout.write("[vocab size]:{}\n".format(self.n_Vocab))
        else:
            sys.stdout.write("vocab doesn't exist on {}!\n".format(os.path.abspath(data_path)))
            self.create_new() #if vocab doesn't exist, automatically create new vocab

    def add_vocab(self, sent):
        words = sent #.split(" ") #assuming already splitted
        for word in words:
            self.word_freq[word] += 1
            if word not in self.Vocab:
                self.Vocab[word] = self.n_Vocab
                self.id2vocab.append(word)
                self.n_Vocab += 1

    def create_new(self):
        sys.stdout.write("create vocab [[[new!!]]]...\n")
        Vocab = defaultdict(lambda:0) #unk -> 0
        word_freq = defaultdict(lambda:0)
        ini_words = ["<unk>", "<s>", "</s>"]
        for i, word in enumerate(ini_words):
            Vocab[word] = i
        self.Vocab = Vocab
        self.word_freq = word_freq
        self.n_Vocab = len(Vocab)
        self.id2vocab = ini_words

    def save(self, save_path):
        sys.stdout.write("[vocab size]: {}\n".format(self.n_Vocab))
        sys.stdout.write("save vocab to {}\n".format(os.path.abspath(save_path)))
        with open(save_path+'/vocab.json','w') as f:
            json.dump(self.Vocab,f)
        with open(save_path+'/vocab_word_freq.json','w') as f:
            json.dump(self.word_freq,f)

    def cut(self, vocab_threshold): #len(Vocab) -> n_vocab
        new_Vocab = defaultdict(lambda:0)
        new_Vocab["<unk>"] = 0
        new_Vocab["<s>"] = 1
        new_Vocab["</s>"] = 2
        n_new_Vocab = len(new_Vocab)
        for key,value in sorted(self.word_freq.items(), key=lambda x:x[1], reverse=True):
            if key not in ["<unk>","<s>","</s>"]:
                new_Vocab[key] = n_new_Vocab+1
                n_new_Vocab += 1
            if n_new_Vocab == vocab_threshold:
                break
        sys.stdout.write("[vocab size]:{} ---> {} \n".format(self.n_Vocab, n_new_Vocab))
        self.Vocab = new_Vocab
        self.n_Vocab = n_new_Vocab


if  __name__ == "__main__":

    # create vocab
	from scipy import io
	from dataset import load_data	

	data_dir = "coco"
	img_feats, data_list = load_data(data_dir)	

	train_indexes = []
	val_indexes = []
	test_indexes = []

	for index, img_with_sentences in enumerate(data_list):
		split = img_with_sentences["split"] # {'restval': 30504, 'test': 5000, 'train': 82783, 'val': 5000}
		assert index == img_with_sentences["imgid"]

		if split=="train":    
			train_indexes.append(index)
		elif split=="val":
			val_indexes.append(index)
		elif split=="test":
			test_indexes.append(index)
		else:
			pass

	print(len(train_indexes), len(val_indexes), len(test_indexes))

	vocab = mkvocab("./vocab")
	renew_vocab = True

	if renew_vocab:
		vocab.create_new()
		for index in train_indexes:
			sentences = data_list[index]["sentences"]
			for sent in sentences:
				tokens = sent["tokens"]
				vocab.add_vocab(tokens)
		vocab.save("./vocab")
