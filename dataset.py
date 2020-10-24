
from scipy import io
import json

import torch
from torch.utils.data import Dataset, DataLoader

def load_data(data_dir):
    #load visual features
    matdata = io.loadmat(data_dir+"/vgg_feats.mat", squeeze_me=True)
    img_feats = matdata["feats"].transpose(1,0)

    #load coco dict
    with open(data_dir+"/dataset.json", "r") as f:
        dic = json.load(f)
    data_list = dic["images"]
    data_list.sort(key=lambda x:x["imgid"]) # to align to features

    return img_feats, data_list


def get_batch(data):

    img_feat, word_ids, sent_len = data
    sent_batch = torch.stack(word_ids, dim=1).long()
    sent_lens = sent_len.long()
    sorted_sent_lens, indices = torch.sort(sent_lens, descending=True)
    sorted_sent_batch = sent_batch[indices]
    sorted_img_feat = img_feat[indices]

    src = sorted_sent_batch[:,:-1]
    tgt = sorted_sent_batch[:,1:]
    length = sorted_sent_lens - 1
    img_feat = sorted_img_feat

    return src, tgt, length, img_feat


class COCO(Dataset):
    """
    data_dir: data directory which contains "vgg_feats.mat" and "dataset.json"
    mode: train or val or test
    vocab: vocab instance using mkvocab
    """
    def __init__(self, img_feats, data_list, vocab, mode="train", sent_len_max=20):
        self.img_feats = img_feats
        self.data_list = data_list
        self.indexes = self.load_keys(self.data_list, mode) 
        self.vocab = vocab
        self.sent_len_max = sent_len_max

    def load_keys(self, data_list, mode):
        assert mode in ["train", "val", "test"]
        indexes = []

        for index, img_with_sentences in enumerate(data_list):
            split = img_with_sentences["split"] # {'restval': 30504, 'test': 5000, 'train': 82783, 'val': 5000}
            assert index == img_with_sentences["imgid"]
            if split==mode:    
                for i, sent in enumerate(img_with_sentences["sentences"]):
                    indexes.append((index, i))
        return indexes

    def __getitem__(self, index):
        pair_index, sent_index = self.indexes[index] # pair index: img and captions pair      
        img_feat = self.img_feats[pair_index]
        
        sent_list = self.data_list[pair_index]["sentences"]
        sent = sent_list[sent_index]
        tokens = sent["tokens"]

        sent_len = len(tokens)
        if len(tokens) < self.sent_len_max:
            tokens = tokens + ["</s>"]*(self.sent_len_max - len(tokens))
        elif len(tokens) > self.sent_len_max: 
            tokens = tokens[:self.sent_len_max]
            sent_len = self.sent_len_max

        word_ids = [self.vocab.Vocab[token] for token in tokens]

        return img_feat, word_ids, sent_len

    def __len__(self):
        return len(self.indexes)

if  __name__ == "__main__":
	# HOW TO USE (data_dir is "coco")
	img_feats, data_list = load_data("coco")
	coco_train = COCO(img_feats, data_list, vocab, "train", sent_len_max=21)

	batch_size = 2	
	train_loader = DataLoader(coco_train, batch_size=batch_size, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)



