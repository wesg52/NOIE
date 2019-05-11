from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction

class StanfordReader(OieReader):
    
    def __init__(self):
        self.name = 'Stanford'
    
    def read(self, fn):
        with open('./all.txt') as f:
            contents = f.read()
            text_list = contents.split('\n')
        d = {}
        with open(fn) as fin:
            i = 0
            for line in fin:
                data = line.strip().split('\t')
                if len(data) == 3:
                    arg1, rel, arg2 = data[0:3]
                elif len(data) == 2:
                    arg1, rel = data[0:2]
                    arg2 = '<UNK>'
                else:
                    arg1 = data[0]
                    arg2 = '<UNK>'
                    rel = '<UNK>'
                confidence = 0.95
                text = text_list[i]
                i += 1
                curExtraction = Extraction(pred = rel, sent = text, confidence = float(confidence))
                curExtraction.addArg(arg1)
                curExtraction.addArg(arg2)
                d[text] = d.get(text, []) + [curExtraction]
        self.oie = d
