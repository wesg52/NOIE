import torch
from torchtext import data, datasets
from utils import get_arg_tokenizer, get_arg_tokenizer_eval, whitespace_tokenizer, get_pretrained_embeddings
from transformer.my_iterator import MyIterator, rebatch
from transformer.flow import make_model, batch_size_fn, run_epoch
from transformer.greedy import greedy_decode_eval
from torch.autograd import Variable
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel
import time
import numpy as np

BATCH_SIZE = 2048
MAX_OUTPUTS = 20

# Relevant Files
SRC_VOCABULARY = './models/noie_full_6heads_2048ffsrc_vocab.pt'
TGT_VOCABULARY = './models/noie_full_6heads_2048fftrg_vocab.pt'
TRAINED_MODEL = './models/noie_full_1200ff_6heads_r5_epoch0.pt'
TESTING_FILE = './all.txt'
OUTPUT_FILE = './output_2.txt'
MODE = 'mean' # 'mean' 'min' or a number between 0 and 100 representing the percentile (5-10 probably a good value)

def get_testset(train_path):

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    PAD_WORD = '<pad>'
    
    SRC = data.Field(tokenize=get_arg_tokenizer_eval, pad_token=PAD_WORD)


    data_fields = [('src', SRC)]

    full = data.TabularDataset(format='csv',
                               path= train_path,
                               fields=data_fields,
                               skip_header=True)
    #SRC.build_vocab(full, min_freq=10)
    return full, SRC, EOS_WORD, BOS_WORD, PAD_WORD

def batch_size_fn2(new, count, size_so_far):
    """
    Keep augmenting batch and calculate total number of tokens + padding.
    """
    global max_src_in_batch
    global max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    src_elements = count * max_src_in_batch
    return src_elements


if __name__ == '__main__':
    # Load the model and vocabulary. 
    # Put these files in the current directory first
    vocab_src = torch.load(SRC_VOCABULARY)
    vocab_tgt = torch.load(TGT_VOCABULARY)

    model = make_model(len(vocab_src), len(vocab_tgt),
                        n=5, d_model=768,
                        d_ff=1200, h=6,
                        dropout=.1)
    device = torch.device('cpu') # Ziwei uses CPU
    model.load_state_dict(torch.load(TRAINED_MODEL, map_location=device))
    # model = torch.load(TRAINED_MODEL, map_location=device)

    model.eval()

    # Load testing data
    # test, SRC, EOS_WORD, BOS_WORD, PAD_WORD = get_testset('./all.txt')
    # SRC.vocab = vocab_src

    # test_iter = MyIterator(test, batch_size=BATCH_SIZE, device=0, repeat=False, sort_key=lambda x: (len(x.src)), batch_size_fn=batch_size_fn2, train=False)
    with open(TESTING_FILE, encoding='UTF-8') as f:
        raw_data = f.read()
    test_sents = raw_data.split('\n')

    tok = get_arg_tokenizer()

    out_file = open(OUTPUT_FILE, 'w', encoding="utf-8")
    print('Extraction starts')
    start = time.time()
    # for i, batch in enumerate(test_iter):
    #     if i > MAX_OUTPUTS:
    #         break
    #     src = batch.src.transpose(0, 1)[:1]
    #     src_mask = (src != vocab_src.stoi[PAD_WORD]).unsqueeze(-2)
    #     out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=vocab_tgt.stoi[BOS_WORD])
    #     sent = ''
    #     for i in range(1, out.size(1)):
    #         sym = vocab_tgt.itos[out[0, i]]
    #         if sym == EOS_WORD:
    #             break
    #         sent += sym + ' '
    #     test_file.write(sent + "\n")
    for i in range(len(test_sents)):
        # if i == MAX_OUTPUTS:
        #     break
        sent = tok.tokenize(test_sents[i])
        src = torch.LongTensor([[vocab_src.stoi[w] for w in sent]])
        src = Variable(src)
        src_mask = (src != vocab_src.stoi['<blank>']).unsqueeze(-2)
        out, probs = greedy_decode_eval(model, src, src_mask, max_len=60, start_symbol=vocab_tgt.stoi["<s>"], end_symbol=vocab_tgt.stoi["</s>"])
        if MODE == 'mean':
            prob = np.mean(np.exp(probs.flatten()[1:].detach().numpy()))
        elif MODE == 'min':
            prob = np.min(np.exp(probs.flatten()[1:].detach().numpy()))
        else:
            prob = np.percentile(np.exp(probs.flatten()[1:].detach().numpy()), MODE)

        result = '<s> '
        for i in range(1, out.size(1)):
            sym = vocab_tgt.itos[out[0, i]]
            if sym == "</s>": 
                break
            result += sym + " "
        out_file.write(result + '\t' + str(prob) + '\n')
    out_file.close()
    print('Extraction complete')
    end = time.time()
    print('Time taken: ')
    print(end - start)


