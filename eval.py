import torch
from torchtext import data, datasets
from utils import get_arg_tokenizer, whitespace_tokenizer, get_pretrained_embeddings
from transformer.my_iterator import MyIterator, rebatch
from transformer.flow import make_model, batch_size_fn, run_epoch
from transformer.greedy import greedy_decode
from torch.autograd import Variable

BATCH_SIZE = 2048
MAX_OUTPUTS = 20

def get_testset(train_path):
    tokenizer = get_arg_tokenizer()

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    PAD_WORD = '<pad>'
    
    SRC = data.Field(tokenize=whitespace_tokenizer, pad_token=PAD_WORD)


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
    vocab_src = torch.load("./noie_1mil_6heads_gptweightssrc_vocab.pt")
    vocab_tgt = torch.load("./noie_1mil_6heads_gptweightstrg_vocab.pt")
    device = torch.device('cpu')
    model = torch.load('./noie_1mil_6heads_gptweights_epoch5.pt', map_location=device)

    model.eval()

    # Load testing data
    test, SRC, EOS_WORD, BOS_WORD, PAD_WORD = get_testset('./all.txt')
    SRC.vocab = vocab_src

    test_iter = MyIterator(test, batch_size=BATCH_SIZE, device=0, repeat=False, sort_key=lambda x: (len(x.src)), batch_size_fn=batch_size_fn2, train=False)

    testFile = open('./results.txt', 'w', encoding="utf-8")
    print('Extraction starts\n')
    for i, batch in enumerate(test_iter):
        if i > MAX_OUTPUTS:
            break
        src = batch.src.transpose(0, 1)[:1]
        src_mask = (src != vocab_src.stoi[PAD_WORD]).unsqueeze(-2)
        out = greedy_decode(model, src, src_mask, max_len=60, start_symbol=vocab_tgt.stoi[BOS_WORD])
        sent = ''
        for i in range(1, out.size(1)):
            sym = vocab_tgt.itos[out[0, i]]
            if sym == EOS_WORD:
                break
            sent += sym + ' '
        testFile.write(sent + "\n")
    testFile.close()
    print('Extraction complete')


