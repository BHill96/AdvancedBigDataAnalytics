#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 14:53:07 2020

@author: blakehillier
"""

from pandas import to_datetime, DataFrame, concat
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
# MUST INSTALL PYTORCH-TRANSFORMERS
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW
from tqdm import trange
from numpy import argmax, sum 
import nltk
nltk.download('punkt')

"""
Calculates the sentiment of statements in the dataframe text by aligning them to the appropriate 
stock values in stock and calculating the percent change. The text gets labeled as 1 for 
positive and 0 for anything else. It returns a dataframe of the dates, the text, and the sentiment 
label.
"""
def CalcSentiment(text, metric, metricType='Stock'):
    text['Date'] = to_datetime(text['Date'])
    text['Date'] = text['Date'].dt.normalize()
    text.sort_values(['Date'], inplace=True, axis=0, ascending=True)
    text.reset_index(inplace=True)
    
    metric['Date'] = to_datetime(metric['Date'])
    metric['Date'] = metric['Date'].dt.normalize()
    # metric.drop(labels=['LN Close', 'Volume', 'LN Volume'], axis=1, inplace=True)
    
    if metricType == 'Stock':
        sentimentData = text.merge(metric, on='Date', how='left')
    elif metricType == 'Macro':
        sentimentData = text.merge(DataFrame(turnDaily(text, metric)).rename({0:'Date', 1:'GDP'}, axis=1), on='Date', 
                                          how='left')
    else:
        print('ERROR: info of type {0}'.format(type))
    
    sentimentData.describe(include='all')
    
    # Calculate the percent change between close price
    """ pctChange = sentimentData['Close'].pct_change() """
    pctChange = sentimentData['GDP'].pct_change()
    # Orient it so we have the future percent change
    pctChange.drop(0, inplace=True)
    sentimentData.drop(sentimentData.index[-1], inplace=True)
    sentimentData['Futur_Pct_Change'] = pctChange.values
    
    sentimentData['Econ_Perf'] = sentimentData['Futur_Pct_Change'].apply(lambda x: 1 if x > 0 else 0)
    return sentimentData.drop(labels=['Futur_Pct_Change'], axis=1)
    
def turnDaily(stock, info):
    daily = []
    colLabel = info.columns[1]
    i=len(info)-1
    j=len(stock)-1
    while j > -1 and i > -1:
        if info['Date'][i] < stock['Date'][j]:
            #print('{0} < {1}'.format(info['Date'][i], stock['Date'][j]))
            daily.append([stock['Date'][j], info[colLabel][i]])
            j = j-1
        else:
            #print('{0} > {1}'.format(info['Date'][i], stock['Date'][j]))
            i = i-1
    return daily[::-1]

def xlnetPrep(sentenceList):
    par = ''
    for sentence in sentenceList:
        par += sentence +' [SEP] '
    return par

"""
Turns the text into tokens with [SEP] and [CLS] tags for XLNetTokenizer and then creates the input IDs and attention masks.
"""
def TextPrep(sentimentData):
    sentimentData.Text = sentimentData.Text.apply(nltk.tokenize.sent_tokenize)
    sentimentData.Text = sentimentData.Text.apply(xlnetPrep)
    # Turns the string into a sequence of words
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)
    tokenizedText = sentimentData.Text.apply(tokenizer.tokenize)
    MAX_LEN = 128
    SEPToken = tokenizer.tokenize(' [SEP]')
    CLSToken = tokenizer.tokenize(' [CLS]')
    tokenizedText = tokenizedText.apply(lambda x: x[:MAX_LEN-8]+SEPToken+CLSToken if len(x)>MAX_LEN-4 else x+CLSToken)
    # Create token IDS
    input_ids = input_ids = tokenizedText.apply(tokenizer.convert_tokens_to_ids)
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
    # Attention masks mark how many tokens are in a sentence
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    return input_ids, attention_masks

"""
Calculates accuracy for model.
"""
def flat_accuracy(preds, labels):
    pred_flat = argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return sum(pred_flat == labels_flat) / len(labels_flat)

"""
Trains the model on batches and epochs, validating each epoch to judge convergence. SHOULD NOT BE USED FOR FINAL TRAINING.
"""
def Train(inputIds, attention_masks, labels, batch_size=24, epochs=10):
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
    
    # train_loss_set = []
    
    # Find GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainLoss = []
    valAcc = []
    for _ in trange(epochs, desc='Epoch'):
        # Train
        model.train()
    
        trainLoss.append(0)
        nb_tr_examples, nb_tr_steps = 0, 0
    
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            # Forward pass and loss calculation
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
            # Calculate gradients
            loss.backward()
            # Update weights using gradients
            optimizer.step()
    
            trainLoss[-1] += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
    
        print('\nTrain loss: {}'.format(trainLoss[-1]/nb_tr_steps))
    
        # Valuation
        model.eval()
    
        nb_eval_steps = 0
        valAcc.append(0)
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
    
            valAcc[-1] += tmp_eval_accuracy
            nb_eval_steps += 1
    
        print('\nValidation Accuracy: {}\n'.format(valAcc[-1]/nb_eval_steps))
        
    return model, trainLoss, valAcc
        
