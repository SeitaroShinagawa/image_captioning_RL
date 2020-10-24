#!/usr/bin/env python

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os, sys
import json
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm, trange
from pprint import pprint

from dataset import load_data, COCO, get_batch
from vocab import mkvocab
from model import RNN_DECODER

def date():
    return datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

def argparser():
    parser = argparse.ArgumentParser(description="Train Colorization Model")
    parser.add_argument("-sn", "--save_name", default="result", type=str, help="save name for result dir (make dir under '../results/')")
    parser.add_argument("--data_dir", default="coco", type=str, help="dataset path")
    parser.add_argument("--resume_model_path", default=None, type=str, help="model path to restart training")
    parser.add_argument("--save_interval", default=100, type=int, help="epoch interval for model saving")
    parser.add_argument("-e", "--epochs", default=3000, type=int, help="number of epochs of training")
    parser.add_argument("-bs", "--batch_size", default=16, type=int, help="size of the mini-batch")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--gpu", default=0, type=int, help="ID of gpu")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":

    args = argparser()

    data_dir = "coco"
    sent_len_max = 21
    
    vocab = mkvocab("./vocab") # firstly, run vocab.py to create vocab

    # path settings
    save_name = args.save_name + "_" + date()
    save_p = "./results/" + save_name
    output_p = save_p + "/output/"
    model_p = save_p + "/model/"
    log_p = save_p + "/log/"

    # for tensorboard logging
    board_log_p = "./logs/"+save_name
    writer = SummaryWriter(log_dir=board_log_p)

    for path in [output_p, model_p, log_p, board_log_p]:
        if not os.path.exists(path):
            os.makedirs(path)

    # save args config
    with open(save_p + "/args_config.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    pprint(vars(args))    
    
    # load vocab (firstly, run vocab.py to create vocab)
    vocab = mkvocab("./vocab") 

    # load data
    img_feats, data_list = load_data(data_dir)
    coco_train = COCO(img_feats, data_list, vocab, "train", sent_len_max=sent_len_max)
    coco_val = COCO(img_feats, data_list, vocab, "val", sent_len_max=sent_len_max)

    # set data loader
    train_loader = DataLoader(coco_train, batch_size=args.batch_size, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=1, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)
    
    val_loader = DataLoader(coco_val, batch_size=args.batch_size, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=1, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

    # set model
    n_token = vocab.n_Vocab 
    text_dec = RNN_DECODER(
            n_token,
            ninput = 300, 
            drop_prob = 0.2,
            nhidden = 128, 
            nlayers = 2, 
            n_steps = sent_len_max - 1, 
            bidirectional = True
            )  
    opt = text_dec.make_optimizer(lr=args.lr)

    criterion = nn.NLLLoss()

    # Resume model
    if args.resume_model_path is not None:
        print("Loading a model from:", args.resume_model_path)
        state_dict = torch.load(args.resume_model_path, map_location=lambda storage, loc: storage)
        if "model" in state_dict:
            text_dec.load_state_dict(state_dict["model"])
            opt.load_state_dict(state_dict["optimizer"])
            start_epoch = state_dict["epoch"] + 1
        else:
            net.load_state_dict(state_dict)
            start_epoch = 0

        print("Loading success. Epoch starts at ", start_epoch)
    else:
        start_epoch = 0    

    # gpu mode
    if args.gpu >= 0:
        device = torch.device("cuda:" + str(args.gpu)) 
        text_dec = text_dec.to(device)
        criterion = criterion.to(device)

    # use single precision
    use_single_prec = False
    if use_single_prec:
        from torch.cuda import amp  # pytorch version >= v1.6
        scaler = amp.GradScaler()
    
    save_threshold = args.save_interval
    champ_val_loss = 200
    epochs = args.epochs
    # start train    
    for epoch in trange(start_epoch, epochs, desc="Epoch"):    
        train_loss = 0.0
        text_dec.train()

        for iteration, samples in enumerate(tqdm(train_loader, desc="Train iteration")):        
        
            # sort data
            src, tgt, length, img_feat = get_batch(samples)
            B = img_feat.shape[0]
            if args.gpu >= 0:
                src = src.to(device)
                tgt = tgt.to(device)
                length = length.to(device)
                img_feat = img_feat.to(device)

            # encode image
            hidden = text_dec.embed_img(img_feat)
            
            if not use_single_prec:
                
                # decode caption
                y, word_emb, text_emb = text_dec(src, length, hidden)  # y:(B, ntoken, L)
                
                # loss      
                text_dec.zero_grad()
                loss = criterion(y, tgt[:,:length.max()]) # (B, ntoken, L) 
                loss.backward()
                opt.step()
            
            else: # loss with single precision
                
                # decode caption   
                with amp.autocast():
                    y, word_emb, text_emb = text_dec(src, length, hidden)  # y:(B, ntoken, L)
                    loss = criterion(y, tgt[:,:length.max()]) # (B, ntoken, L) 
                scaler.scale(loss).backward() # to create scaled gradient
                scaler.step(opt)          # unscales gradients
                scaler.update()           # update the scale for next iteration

            torch.nn.utils.clip_grad_norm_(text_dec.parameters(), 0.25)
            train_loss += loss.item() * B
            
        train_loss =  train_loss/len(train_loader)
        writer.add_scalar("train/loss", train_loss, epoch)

        val_loss = 0.0
        text_dec.eval()
        with torch.no_grad():
            for iteration, samples in enumerate(tqdm(val_loader, desc="Validation iteration")):

                # sort data
                src, tgt, length, img_feat = get_batch(samples)
                B = img_feat.shape[0]
                if args.gpu >= 0:
                    src = src.to(device)
                    tgt = tgt.to(device)
                    length = length.to(device)
                    img_feat = img_feat.to(device)

                # encode image
                hidden = text_dec.embed_img(img_feat)
                
                # decode caption
                y, word_emb, text_emb = text_dec(src, length, hidden)  # y:(B, ntoken, L)
                loss = criterion(y, tgt[:,:length.max()]) # (B, ntoken, L) 
                    
                # stop decoding
                #word = vocab.Vocab["<s>"] # initial word    
                #for i in range(sent_len_max):        
                #    next_word, hidden = text_dec.step(word, hidden)  # y:(B, ntoken, L)
        
                val_loss += loss.item() * B
            
            val_loss =  val_loss/len(val_loader)
            writer.add_scalar("val/loss", val_loss, epoch)


        # Save weight
        if val_loss <= champ_val_loss:
            if epoch > save_threshold:
                save_threshold = epoch + save_interval # keep the previous champion
            else:
                if os.path.isfile(champ_weight_name):
                    os.remove(champ_weight_name) # remove the previous champion

            save_weight_name = model_p + "epoch_%d_valloss_%1.4f_weight.pth" %(epoch, val_loss)
            save_states = {"epoch": epoch, "model": text_dec.state_dict(), "optimizer": opt.state_dict()}
            torch.save(save_states, save_weight_name)
            champ_weight_name = save_weight_name
            champ_val_loss = val_loss
            champ_epoch = epoch
