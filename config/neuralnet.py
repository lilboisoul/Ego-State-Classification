from config.globals import NUM_LABELS, ID2LABEL, LABEL2ID, DEVICE
from transformers import BertTokenizer, BertForSequenceClassification

base_model = 'bert-base-uncased'
bert_model = BertForSequenceClassification.from_pretrained(base_model, num_labels=NUM_LABELS, id2label=ID2LABEL,
                                                           label2id=LABEL2ID)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
if DEVICE == 'cuda':
    if bert_model:
        bert_model = bert_model.to('cuda')

