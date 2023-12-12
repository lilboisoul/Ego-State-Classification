from config.globals import num_labels, id2label, label2id, DEVICE
from transformers import BertTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, \
    RobertaTokenizer, DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig

base_model = 'bert-base-uncased'

bert_model = BertForSequenceClassification.from_pretrained(base_model, num_labels=num_labels, id2label=id2label,
                                                           label2id=label2id)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

if DEVICE == 'cuda':
    if bert_model:
        bert_model = bert_model.to('cuda')

