import pickle
import pandas as pd
import numpy as np
import pickle
import transformers
from simpletransformers.classification import MultiLabelClassificationModel, MultiLabelClassificationArgs
import logging
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score
from sklearn import preprocessing

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")

NUM_LABELS = 500
USE_CUDA = True

def f1_wrapper(y_true, predictions):
    min_max_scaler = preprocessing.MinMaxScaler()
    predictions_normalized = min_max_scaler.fit_transform(predictions)

    return f1_score(
         y_true,
         round_raw_values(predictions_normalized, 0.3),
         average='samples'
    )

def recall_score_wrapper(y_true, predictions):
    min_max_scaler = preprocessing.MinMaxScaler()
    predictions_normalized = min_max_scaler.fit_transform(predictions)

    return recall_score(
         y_true,
         round_raw_values(predictions_normalized, 0.3),
         average='samples'
    )

def precision_score_wrapper(y_true, predictions):
    min_max_scaler = preprocessing.MinMaxScaler()
    predictions_normalized = min_max_scaler.fit_transform(predictions)

    return precision_score(
        y_true,
        round_raw_values(predictions_normalized, 0.3),
        average='samples'
    )

def round_raw_values(dataset, theshold):
  rounded_values = []
  for i in dataset:
    i_th_values = []
    for j in i:
      if j > theshold:
        i_th_values.append(1)
      else:
        i_th_values.append(0)
    rounded_values.append(i_th_values)
  return(rounded_values)

def prepare_data(infile):
  data = pickle.load(infile)
  infile.close()

  data_frame = pd.DataFrame.from_dict(data)
  data_frame.reset_index(drop=True, inplace=True)

  data_classes = list(data_frame.columns)
  data_classes.remove('MOLECULEID')
  data_classes.remove('SMILES')

  for col in data_classes:
    data_frame[col] = data_frame[col].astype(int)

  prepared_data = []
  for index, row in data_frame.iterrows():
    prepared_data.append([
                      data_frame.iloc[index].values[1],
                      data_frame.iloc[index].values[2:502].tolist()
                      ])

  prepared_df = pd.DataFrame(prepared_data, columns=['text', 'labels'])
  return prepared_df


train_infile = open('./datasets/train.pkl','rb')
validation_infile = open('./datasets/validation.pkl','rb')
test_infile = open('./datasets/test.pkl','rb')

train_data = prepare_data(train_infile)
validation_data = prepare_data(validation_infile)
test_data = prepare_data(test_infile)

MODEL_ARGs = MultiLabelClassificationArgs(
    reprocess_input_data=True,
    overwrite_output_dir=True,
    num_train_epochs=100,
    #no_save=True,
    #save_model_every_epoch=False,
    #save_eval_checkpoints=False,
    train_batch_size=4,
    evaluate_during_training=True,
    evaluate_during_training_verbose=True,
    eval_batch_size=4,
    threshold = 0.3
    )

model = MultiLabelClassificationModel(model_type='bert',
                                        model_name='./saved_models/SMILES_MLM_Pretrained/pretrained_SMILES_15MLM_100Epochs',
                                        num_labels=NUM_LABELS,
                                        use_cuda=USE_CUDA,
                                        args=MODEL_ARGs,
                                        )

model.train_model(train_df=train_data,
                    eval_df=validation_data,
                    f1=f1_wrapper,
                    recall=recall_score_wrapper,
                    precision=precision_score_wrapper
                    )

predicted, raw_values = model.predict(test_data['text'])

with open('./results/SMILES/predicted_fine_tuned_SMILES_100Epochs.pkl', 'wb') as f:
  pickle.dump(predicted, f)

with open('./results/SMILES/raw_values_fine_tuned_SMILES_100Epochs.pkl', 'wb') as f:
  pickle.dump(raw_values, f)

print('Model outputs saved!')
