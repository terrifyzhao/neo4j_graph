import torch
from torch.utils.data import TensorDataset, DataLoader
from kbqa.torch_utils import TrainHandler
from kbqa.data_process import *
from kbqa.text_cnn import TextCNN

# df = pd.read_csv('question_classification.csv')
# print(df['label'].value_counts())


def load_data(batch_size=32):
    df = pd.read_csv('question_classification.csv')
    train_df, eval_df = split_data(df)
    train_x = df['question']
    train_y = df['label']
    valid_x = eval_df['question']
    valid_y = eval_df['label']

    train_x = padding_seq(train_x.apply(seq2index))
    train_y = np.array(train_y)
    valid_x = padding_seq(valid_x.apply(seq2index))
    valid_y = np.array(valid_y)

    train_data_set = TensorDataset(torch.from_numpy(train_x),
                                   torch.from_numpy(train_y))
    valid_data_set = TensorDataset(torch.from_numpy(valid_x),
                                   torch.from_numpy(valid_y))
    train_data_loader = DataLoader(dataset=train_data_set, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(dataset=valid_data_set, batch_size=batch_size, shuffle=True)

    return train_data_loader, valid_data_loader


train_loader, valid_loader = load_data(batch_size=64)

model = TextCNN(1141, 256, 3)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
model_path = 'text_cnn.p'
handler = TrainHandler(train_loader,
                       valid_loader,
                       model,
                       criterion,
                       optimizer,
                       model_path,
                       batch_size=32,
                       epochs=5,
                       scheduler=None,
                       gpu_num=0)
handler.train()
