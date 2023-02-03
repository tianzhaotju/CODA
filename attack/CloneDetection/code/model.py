import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class CodeT5RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features.reshape(-1, features.size(-1)  * 2)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class CodeBERT(nn.Module):
    def __init__(self, encoder,config,tokenizer,args):
        super(CodeBERT, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
        self.query = 0
    
        
    def forward(self, input_ids=None,labels=None): 
        input_ids=input_ids.view(-1,self.args.block_size)
        outputs = self.encoder(input_ids=input_ids,attention_mask=input_ids.ne(1))[0]
        logits=self.classifier(outputs)
        prob=F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob

    def get_results(self, dataset, batch_size, threshold=0.5):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size,num_workers=0,pin_memory=False)

        ## Evaluate Model
        eval_loss = 0.0
        self.eval()
        logits=[] 
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")       
            label=batch[1].to("cuda") 
            with torch.no_grad():
                lm_loss,logit = self.forward(inputs,label)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
        logits=np.concatenate(logits,0)
        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:,0]]

        return probs, pred_labels


class GraphCodeBERT(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(GraphCodeBERT, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2, attn_mask_2, labels=None):
        bs, l = inputs_ids_1.size()
        inputs_ids = torch.cat((inputs_ids_1.unsqueeze(1), inputs_ids_2.unsqueeze(1)), 1).view(bs * 2, l)
        position_idx = torch.cat((position_idx_1.unsqueeze(1), position_idx_2.unsqueeze(1)), 1).view(bs * 2, l)
        attn_mask = torch.cat((attn_mask_1.unsqueeze(1), attn_mask_2.unsqueeze(1)), 1).view(bs * 2, l, l)

        # embedding
        nodes_mask = position_idx.eq(0)
        token_mask = position_idx.ge(2)
        inputs_embeddings = self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask = nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        nodes_to_token_mask = nodes_to_token_mask / (nodes_to_token_mask.sum(-1) + 1e-10)[:, :, None]
        avg_embeddings = torch.einsum("abc,acd->abd", nodes_to_token_mask, inputs_embeddings)
        inputs_embeddings = inputs_embeddings * (~nodes_mask)[:, :, None] + avg_embeddings * nodes_mask[:, :, None]

        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings, attention_mask=attn_mask, position_ids=position_idx)[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

    def get_results(self, dataset, batch_size, threshold=0.5):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0,
                                     pin_memory=False)

        self.eval()
        logits = []
        for batch in eval_dataloader:
            (inputs_ids_1, position_idx_1, attn_mask_1,
             inputs_ids_2, position_idx_2, attn_mask_2,
             label) = [x.to("cuda") for x in batch]
            with torch.no_grad():
                logit = self.forward(inputs_ids_1, position_idx_1, attn_mask_1, inputs_ids_2, position_idx_2,
                                     attn_mask_2)
                logits.append(logit.cpu().numpy())

        logits = np.concatenate(logits, 0)
        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]

        return probs, pred_labels


class CodeT5(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeT5, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.classifier = CodeT5RobertaClassificationHead(config)
        self.args = args
        self.query = 0

    def forward(self, input_ids=None, labels=None):
        input_ids = input_ids.view(-1, self.args.block_size)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                               labels=input_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        outputs = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                                  hidden_states.size(-1))[:, -1, :]

        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob

    def get_results(self, dataset, batch_size, threshold=0.5):
        '''Given a dataset, return probabilities and labels.'''
        self.query += len(dataset)
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size, num_workers=0,
                                     pin_memory=False)

        ## Evaluate Model
        eval_loss = 0.0
        self.eval()
        logits = []
        for batch in eval_dataloader:
            inputs = batch[0].to("cuda")
            label = batch[1].to("cuda")
            with torch.no_grad():
                lm_loss, logit = self.forward(inputs, label)
                eval_loss += lm_loss.mean().item()
                logits.append(logit.cpu().numpy())
        logits = np.concatenate(logits, 0)

        probs = logits
        pred_labels = [0 if first_softmax > threshold else 1 for first_softmax in logits[:, 0]]

        return probs, pred_labels