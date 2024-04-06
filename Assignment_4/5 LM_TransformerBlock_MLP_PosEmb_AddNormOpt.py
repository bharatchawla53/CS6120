import numpy as np
import torch
import yaml

from torch import nn
from torch.nn import functional as F

''' Look at all previous tokens to generate next
    @Author: Uzair Ahmad
    2022
    +TransformerBlock 
'''


class TransformerBlockLM(nn.Module):
    class TransformerBlock(nn.Module):
        def __init__(self, head_count, in_size, out_size):
            super().__init__()
            self.comm = TransformerBlockLM.MultiHeadAttention(head_count=head_count,
                                                              in_size=in_size,
                                                              out_size=out_size)
            self.think = TransformerBlockLM.MLP(embed_size=out_size)

        def forward(self, x):
            return x + self.think(x + self.comm(x))

    class MLP(nn.Module):
        # FFNN (embed_size, embed_size*4, embed_size)
        def __init__(self, embed_size):
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(embed_size, embed_size * 4),
                                     nn.ReLU(),
                                     nn.Linear(embed_size * 4, embed_size))
            self.layerNorm = nn.LayerNorm(embed_size)

        def forward(self, x):  # think
            return self.layerNorm(self.mlp(x))  # paper - after
            # return self.mlp(self.layerNorm(x)) # alternate - before

    class MultiHeadAttention(nn.Module):
        """
        multiple parallel SA heads (communication among words)
        """

        def __init__(self, head_count, in_size, out_size):
            super().__init__()
            self.heads = nn.ModuleList(
                TransformerBlockLM.SelfAttentionHead(in_size, out_size // head_count)
                for _ in range(head_count)
            )
            self.layerNorm = nn.LayerNorm(out_size)
            # self.proj = nn.Linear(out_size, out_size)

        def forward(self, x):
            # concat over channel/embeddings_size dimension
            return self.layerNorm(torch.cat([head(x) for head in self.heads], dim=-1))  # paper - after
            # return torch.cat([head(self.layerNorm(x)) for head in self.heads], dim=-1) # alternate - before
            # return self.proj(torch.cat([head(x) for head in self.heads], dim=-1))

    class SelfAttentionHead(nn.Module):
        def __init__(self, in_size, out_size):
            """
            in_size is embed_size
            out_size is head_size
            """
            super().__init__()
            self.head_size = out_size
            self.K = nn.Linear(in_size, self.head_size, bias=False)
            self.Q = nn.Linear(in_size, self.head_size, bias=False)
            self.V = nn.Linear(in_size, self.head_size, bias=False)

        def forward(self, x):
            keys = self.K(x)
            queries = self.Q(x)
            # affinities :
            # all the queries will dot-product with all the keys
            # transpose (swap) second dimension (input_length) with third (head_size)
            keys_t = keys.transpose(1, 2)
            autocorrs = (queries @ keys_t) * (self.head_size ** -0.5)  # (batch_size x input_length x input_length)
            '''
            (batch_size x input_length x embed_size) @ (batch_size x embed_size x input_length) ----> (batch_size x input_length x input_length)
            '''
            autocorrs = torch.tril(autocorrs)
            autocorrs = autocorrs.masked_fill(autocorrs == 0, float('-inf'))
            autocorrs = torch.softmax(autocorrs, dim=-1)
            values = self.V(x)  # (batch_size x input_length x head_size)
            out = autocorrs @ values
            return out

    def __init__(self, batch_size,
                 input_length,
                 embed_size,
                 sa_multihead_count,
                 sa_head_size,
                 pos_embed,
                 include_mlp, 
                 text):
        super().__init__()
        self.blocks = None
        self.ffn = None
        self.sa_heads = None
        # sa_head_size head_size of self-attention module
        self.sa_head_size = sa_head_size
        self.sa_multihead_count = sa_multihead_count

        self.val_data = None
        self.train_data = None
        self.val_text = None
        self.train_text = None
        self.K = None
        self.linear_sahead_to_vocab = None
        self.vocab = None
        self.token_embeddings_table = None
        self.vocab_size = None
        self.encoder = None
        self.decoder = None
        self.vocab_size: int
        self.is_pos_emb = pos_embed
        self.include_mlp = include_mlp
        self.text = text
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # input_length = how many consecutive tokens/chars in one input
        self.input_length = input_length
        # batch_size = how many inputs are going to be processed in-parallel (on GPU)
        self.batch_size = batch_size
        # embed_size = embedding size
        self.embed_size = embed_size

        self.lm_head = None
        self.position_embeddings_table = None

    def forward(self, in_ids, target=None):
        
        # if in in_ids have fewer tokens than self.input_length, the slicing operation could result in empty tensor,
        # to address this issue a check can be added in place to ensure the input tensor has enough tokens before performing 
        # slicing function. 
        #if in_ids.size(1) < self.input_length: 
        #    raise ValueError("Input tensor has fewer tokens than input_length.")
        in_ids_emb = self.token_embeddings_table(in_ids[:, -self.input_length:])
        
        if self.is_pos_emb:
            in_ids_pos_emb = self.position_embeddings_table(
                torch.arange(in_ids[:, -self.input_length:].shape[1], device=self.device)
            )
            in_ids_emb = in_ids_emb + in_ids_pos_emb

        block_outputs = self.blocks(in_ids_emb)
        logits = self.linear_sahead_to_vocab(block_outputs)  # compute

        if target is None:
            ce_loss = None
        else:
            batch_size, input_length, vocab_size = logits.shape
            logits_ = logits.view(batch_size * input_length, vocab_size)
            targets = target.view(batch_size * input_length)
            ce_loss = F.cross_entropy(logits_, targets)
        return logits, ce_loss

    def fit(self, train_iters, eval_iters, lr):
        """
        train_iters = how many training iterations
        eval_iters = how many batches to evaluate to get average performance
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for iteration in range(train_iters):
            if iteration % eval_iters == 0:
                avg_loss = self.eval_loss(eval_iters)
                print(f"Loss iter {iteration}: train {avg_loss['train'][0]} val {avg_loss['eval'][0]}")
                print(f"Perplexity iter {iteration}: train {avg_loss['train'][1]} val {avg_loss['eval'][1]}")
            inputs, targets = self.get_batch(split='train')
            _, ce_loss = self(inputs, targets)
            optimizer.zero_grad(set_to_none=True)  # clear gradients of previous step
            ce_loss.backward()  # propagate loss back to each unit in the network
            optimizer.step()  # update network parameters w.r.t the loss
        # torch.save(self, 'sa_pos_')

    def generate(self, context_token_ids, max_new_tokens):
        for _ in range(max_new_tokens):
            token_rep, _ = self(context_token_ids)
            last_token_rep = token_rep[:, -1, :]
            probs = F.softmax(last_token_rep, dim=1)
            next_token = torch.multinomial(probs, num_samples=1)
            context_token_ids = torch.cat((context_token_ids, next_token), dim=1)
        output_text = self.decoder(context_token_ids[0].tolist())
        return output_text

    @torch.no_grad()  # tell torch not to prepare for back-propagation (context manager)
    def eval_loss(self, eval_iters):
        perf = {}
        # set dropout and batch normalization layers to evaluation mode before running inference.
        self.eval()
        for split in ['train', 'eval']:
            losses = torch.zeros(eval_iters)
            perplexities = torch.zeros(eval_iters)
            
            for k in range(eval_iters):
                tokens, targets = self.get_batch(split)  # get random batch of inputs and targete
                _, ce_loss = self(tokens, targets)  # forward pass
                losses[k] = ce_loss.item()  # the value of loss tensor as a standard Python number
                perplexities[k] = torch.exp(ce_loss)
            perf[split] = (losses.mean(), perplexities.mean())
        self.train()  # turn-on training mode-
        return perf

    def prep(self, corpus):
        self.vocab = sorted(list(set(corpus)))
        self.vocab_size = len(self.vocab)
        c2i = {c: i for i, c in
               enumerate(self.vocab)}  # char c to integer i map. assign value i for every word in vocab
        i2c = {i: c for c, i in c2i.items()}  # integer i to char c map

        self.encoder = lambda doc: [c2i[c] for c in doc]
        self.decoder = lambda nums: ''.join([i2c[i] for i in nums])

        n = len(self.text)
        self.train_text = self.text[:int(n * 0.9)]
        self.val_text = self.text[int(n * 0.9):]

        self.train_data = torch.tensor(self.encoder(self.train_text), dtype=torch.long)
        self.val_data = torch.tensor(self.encoder(self.val_text), dtype=torch.long)

        # look-up table for embeddings (vocab_size x embed_size)
        # it will be mapping each token id to a vector of embed_size
        # a wrapper to store vector representations of each token
        self.token_embeddings_table = \
            nn.Embedding(self.vocab_size, self.embed_size)

        if self.is_pos_emb:
            self.position_embeddings_table = nn.Embedding(self.input_length, self.embed_size)

        self.blocks = nn.Sequential(
            TransformerBlockLM.TransformerBlock(head_count=self.sa_multihead_count,
                                                in_size=self.embed_size,
                                                out_size=self.sa_head_size),
            TransformerBlockLM.TransformerBlock(head_count=self.sa_multihead_count,
                                                in_size=self.embed_size,
                                                out_size=self.sa_head_size),
            TransformerBlockLM.TransformerBlock(head_count=self.sa_multihead_count,
                                                in_size=self.embed_size,
                                                out_size=self.sa_head_size),
            TransformerBlockLM.TransformerBlock(head_count=self.sa_multihead_count,
                                                in_size=self.embed_size,
                                                out_size=self.sa_head_size),
            TransformerBlockLM.TransformerBlock(head_count=self.sa_multihead_count,
                                                in_size=self.embed_size,
                                                out_size=self.sa_head_size),
            TransformerBlockLM.TransformerBlock(head_count=self.sa_multihead_count,
                                                in_size=self.embed_size,
                                                out_size=self.sa_head_size),
        )
        # linear projection of sa_head output to vocabulary
        self.linear_sahead_to_vocab = nn.Linear(self.sa_head_size, self.vocab_size)

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        # get random chunks of length batch_size from data
        ix = torch.randint(len(data) - self.input_length,
                           (self.batch_size,))
        inputs_batch = torch.stack([data[i:i + self.input_length] for i in ix])
        targets_batch = torch.stack([data[i + 1:i + self.input_length + 1] for i in ix])
        inputs_batch = inputs_batch.to(self.device)
        targets_batch = targets_batch.to(self.device)
        # inputs_batch is
        return inputs_batch, targets_batch

def main():
    
    # read the configuration file
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)

    # access parent parameters
    model_config = config['ModelConfig']
    training_config = config['TrainingConfig']
    input_config = config['InputConfig']

    # access child parameters of ModelConfig
    batch_size = model_config['batch_size']
    input_length = model_config['input_length']
    embed_size = model_config['embed_size']
    sa_multihead_count = model_config['sa_multihead_count']
    sa_head_size = model_config['sa_head_size']
    pos_embed = model_config['pos_embed']
    include_mlp = model_config['include_mlp']
    split = model_config['split']
    max_new_tokens = model_config['max_new_tokens']

    # access child parameters of TrainingConfig
    train_iters = training_config['train_iters']
    eval_iters = training_config['eval_iters']
    lr = training_config['lr']
    
    # access child parameters of InputConfig
    filepath = input_config['filepath']
    mode = input_config['mode']
    
    # read input file for training
    with open(filepath, mode) as f:
        text = f.read()
     
    # initialize TransformerBlockLM
    model = TransformerBlockLM(batch_size, input_length, embed_size, sa_multihead_count, 
                               sa_head_size, pos_embed, include_mlp, text)

    # move the model to the specified device (GPU if available, otherwise CPU)
    model = model.to(model.device)
    
    # prepare the input of tokens to train the model on
    model.prep(text)
    
    # filter out the model parameters that require gradient
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    
    # print total number of model parameters
    print(f'params {sum([np.prod(p.size()) for p in model_parameters])}')
    
    # generate random batches  
    input_batch, output_batch = model.get_batch(split)
    
    # runs forward pass
    _, _ = model(input_batch, output_batch)
    
    # fit the model 
    model.fit(train_iters, eval_iters, lr)
    
    # generate text sequences
    outputs = model.generate(context_token_ids = torch.zeros((1, 1),
                             dtype = torch.long,
                             device = model.device),
                             max_new_tokens = max_new_tokens)
    
    # print the generated sequence
    print(outputs)

if __name__=="__main__": 
    main() 
