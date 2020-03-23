#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:53:07 2020

@author: blakehillier
"""

from pandas import read_csv, to_datetime
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification, AdamW
from tqdm import tqdm, trange
import io
import numpy as np

"""
Calculates the sentiment of statements in the dataframe text by aligning them to the appropriate 
stock values in stock and calculating the percent change. The text gets labeled as 1 for 
positive and 0 for anything else. It returns a dataframe of the dates, the text, and the sentiment 
label.
"""
def CalcSentiment(text, stock):
    text['Date'] = to_datetime(text['Date'])
    
    stock['Date'] = to_datetime(stock['Date'])
    stock['Date'] = stock['Date'].dt.normalize()
    stock.drop(labels=['LN Close', 'Volume', 'LN Volume'], axis=1, inplace=True)
    
    sentimentData = text.merge(stock, on='Date', how='left')
    sentimentData.describe(include='all')

    # Calculate the percent change between close price
    pctChange = sentimentData['Close'].pct_change()
    # Orient it so we have the future percent change
    pctChange.drop(0, inplace=True)
    sentimentData.drop(sentimentData.index[-1], inplace=True)
    sentimentData['Futur_Pct_Change'] = pctChange.values
    
    sentimentData['Econ_Perf'] = sentimentData['Futur_Pct_Change'].apply(lambda x: 1 if x > 0 else 0)
    return sentimentData.drop(labels=['Close'], axis=1)

def TextPrep(sentimentData):
    sentences = sentimentData.Text.values
    # You can use up to two sentences. [SEP] ends a sentence, [CLS] ends the input text.
    sentences = [sentence + ' [SEP] [CLS]' for sentence in sentences]
    
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    # Turns the string into a sequence of words
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    # We set the maximum input length by finding the largest sample, and then using the extra memory space in the memory block
    # This takes too much memory so for now we set to largest sample size
    largestTextLength = sentimentData.Text.map(lambda x: len(x)).max()
    MAX_LEN = int(pow(2, np.ceil(np.log(largestTextLength)/np.log(3))))
    # Create token IDS
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    # Attention masks mark how many tokens are in a sentence
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return input_ids, attention_masks

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def Train(inputIds, attention_masks, labels):
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(inputIds, 
                                                                                        labels, 
                                                                                        random_state=2020, 
                                                                                        test_size=0.2)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, inputIds, random_state=2020, 
                                                           test_size=0.2)
    # Turn data into torch tensors
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    
    batch_size = 24
    # Create Iterators of the datasets
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
    
    model = XLNetForSequenceClassification.from_pretrained('xlnet-base-cased', num_labels=2)
    # Loads model into GPU memory
    model.cuda()
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','gamma','beta']
    optimizer_grouped_parameters = [
        {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate':0.01},
        {'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate':0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
    
    train_loss_set = []
    epochs = 10
    
    # Find GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for _ in trange(epochs, desc='Epoch'):
        # Train
        model.train()
    
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
    
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            # Forward pass and loss calculation
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
            train_loss_set.append(loss.item())
            # Calculate gradients
            loss.backward()
            # Update weights using gradients
            optimizer.step()
    
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
    
        print('\nTrain loss: {}'.format(tr_loss/nb_tr_steps))
    
        # Valuation
        model.eval()
    
        eval_loss, eval_accuracy = 0,0
        nb_eval_steps, nb_eval_examples = 0,0
        
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # Don't calculate gradients since we are evaluating the model
            with torch.no_grad():
                output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = output[0]
            # Grab logistic values from GPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
    
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
    
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
    
        print('\nValidation Accuracy: {}\n'.format(eval_accuracy/nb_eval_steps))
        
    return model
        
