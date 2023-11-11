from config.globals import num_labels, id2label, label2id, device
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

base_model = 'bert-base-uncased'
finetuned_model = 'output/checkpoint-6000'
model = BertForSequenceClassification.from_pretrained(finetuned_model, num_labels=num_labels, id2label=id2label,
                                                      label2id=label2id)
if device == 'cuda':
    model = model.to('cuda')


metrics = ""  # [
# smp.utils.metrics.IoU(threshold=0.5),
# smp.utils.metrics.Fscore(threshold=0.5),
# smp.utils.metrics.Accuracy(threshold=0.5),
# smp.utils.metrics.Recall(threshold=0.5),
# smp.utils.metrics.Precision(threshold=0.5),]

optimizer = ""  # torch.optim.Adam([dict(params=model.parameters(), lr=lr),)]
