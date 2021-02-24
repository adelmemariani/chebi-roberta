from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

VOCAB_SIZE = 1400
MAX_LEN = 1800

config = RobertaConfig(
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=MAX_LEN,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

model = RobertaForMaskedLM(config=config)
tokenizer = RobertaTokenizerFast.from_pretrained('./tokenizers/BERT_Tokenizer_BPE', max_len=MAX_LEN)

dataset_train = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='./datasets/line_by_line_text_tataset_train',
    block_size=MAX_LEN,
)

dataset_validation = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='./datasets/line_by_line_text_dataset_evaluation',
    block_size=MAX_LEN,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./outputs/BPE_MLM",
    overwrite_output_dir=True,
    num_train_epochs=100,
    per_device_train_batch_size=4,
    save_steps=10000,
    do_eval=True,
    evaluation_strategy='epoch'
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset_train,
    eval_dataset=dataset_validation,
)

trainer.train()

trainer.save_model("./saved_models/pretrained_BPE_15MLM_100Epochs")
print('Model saved!')
