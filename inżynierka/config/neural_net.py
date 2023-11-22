from config.globals import num_labels, id2label, label2id, device
from transformers import BertTokenizer, BertForSequenceClassification, RobertaForSequenceClassification, \
    RobertaTokenizer, DistilBertTokenizer, DistilBertForSequenceClassification, DistilBertConfig

base_model = 'bert-base-uncased'
base_model2 = 'roberta-base'
#base_model3 = 'distilbert-base-uncased'
bert_model = BertForSequenceClassification.from_pretrained(base_model, num_labels=num_labels, id2label=id2label,
                                                      label2id=label2id)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#roberta_tokenizer = RobertaTokenizer.from_pretrained(base_model2)
#roberta_model = RobertaForSequenceClassification.from_pretrained(base_model2, num_labels=num_labels, id2label=id2label,label2id=label2id)

if device == 'cuda':
    if bert_model:
        bert_model = bert_model.to('cuda')
    #roberta_model = roberta_model.to('cuda')


metrics = ""  # [
# smp.utils.metrics.IoU(threshold=0.5),
# smp.utils.metrics.Fscore(threshold=0.5),
# smp.utils.metrics.Accuracy(threshold=0.5),
# smp.utils.metrics.Recall(threshold=0.5),
# smp.utils.metrics.Precision(threshold=0.5),]

optimizer = ""  # torch.optim.Adam([dict(params=model.parameters(), lr=lr),)]
