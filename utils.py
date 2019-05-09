import numpy as np
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTModel


def get_arg_tokenizer():
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    max_key = max(tokenizer.encoder.values())
    add_toks = ['argonestart', 'argoneend', 'relstart', 'relend', 'argtwostart', 'argtwoend']
    word_piece = '</w>'
    for ix, tok in enumerate(add_toks):
        tokenizer.encoder[tok + word_piece] = max_key + 1 + ix
        tokenizer.decoder[max_key + 1 + ix] = tok + word_piece
        tokenizer.cache[tok] = tok + word_piece
    return tokenizer

def get_arg_tokenizer_eval(text):
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    max_key = max(tokenizer.encoder.values())
    add_toks = ['argonestart', 'argoneend', 'relstart', 'relend', 'argtwostart', 'argtwoend']
    word_piece = '</w>'
    for ix, tok in enumerate(add_toks):
        tokenizer.encoder[tok + word_piece] = max_key + 1 + ix
        tokenizer.decoder[max_key + 1 + ix] = tok + word_piece
        tokenizer.cache[tok] = tok + word_piece
    return tokenizer.tokenize(text)

def whitespace_tokenizer(text):
    return text.strip().split(" ")

def get_pretrained_embeddings(vocab):
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    model = OpenAIGPTModel.from_pretrained('openai-gpt')

    gpt_embed = model.tokens_embed._parameters['weight']
    gpt_embed_array = gpt_embed.detach().numpy()
    vec_dim = gpt_embed_array.shape[1]
    embed_array = np.zeros((len(vocab.itos), vec_dim))

    ix_map = {i:vocab.itos[i] for i in range(len(vocab.itos))}
    ix_map = {i:tokenizer.encoder.get(v, 0) for i, v in ix_map.items()}

    embed_array = gpt_embed_array[list(ix_map.values()),:]
    embed_array[vocab.stoi['<pad>']] = np.zeros(vec_dim)

    return embed_array


def swap_tokens(text, encode=True):
    if encode:
        text = text.replace('<arg1>', 'argonestart')
        text = text.replace('</arg1>', 'argoneend')
        text = text.replace('<rel>', 'relstart')
        text = text.replace('</rel>', 'relend')
        text = text.replace('<arg2>', 'argtwostart')
        text = text.replace('</arg2>', 'argtwoend')
        return text
    else:
        text = text.replace('argonestart', '<arg1>')
        text = text.replace('argoneend', '</arg1>')
        text = text.replace('relstart', '<rel>')
        text = text.replace('relend', '</rel>')
        text = text.replace('argtwostart', '<arg2>')
        text = text.replace('argtwoend', '</arg2>')
        return text
