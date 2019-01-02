import pandas as pd
from sklearn import preprocessing
from collections import deque
import numpy as np
import random
import time
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM,CuDNNLSTM,BatchNormalization
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.optimizers import Adam

decay=1e-8
lr = 1e-3
SEQ_LEN=60
FUTURE_PERIOD_PREDICT=3
epoch = 10
batch_size = 64
name = str(decay)+'-DY-'+str(lr)+'-LR-'+str(SEQ_LEN)+'-SEQ-'+str(FUTURE_PERIOD_PREDICT)+'-PRED-'+str(int(time.time()))


def classify(current,future):
    if float(future)>float(current):
        return 1
    else:
        return 0
def preprocess_df(df):
    df = df.drop('future',1)
    df = df.drop('target',1)
    for col in df.columns:
        df[col] = df[col].pct_change()
        df.dropna(inplace=True)
        df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace=True)
    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days)==SEQ_LEN:
            sequential_data.append([np.array(prev_days),i[-1]])
    random.shuffle(sequential_data)
    buys = []
    sells = []
    for seq,target in sequential_data:
        if target==0:
            sells.append([seq,target])
        else:
            buys.append([seq,target])
    random.shuffle(buys)
    random.shuffle(sells)
    lower = min(len(buys),len(sells))
    buys = buys[:lower]
    sells = sells[:lower]
    sequential_data = buys+sells
    random.shuffle(sequential_data)
    x=[]
    y=[]
    for seq,target in sequential_data:
        x.append(seq)
        y.append(target)
    return np.array(x),y
df = pd.read_csv('ltc.csv',names=['open','high','low','close'])
df['future']=df['close'].shift(-FUTURE_PERIOD_PREDICT)
df['target']= list(map(classify,df['close'],df['future']))
temp = sorted(df.index.values)
last_5pct = temp[-int(0.05*len(temp))]
validation = df[(df.index >= last_5pct)]
df = df[(df.index < last_5pct)]
trainx,trainy = preprocess_df(df)
validationx,validationy = preprocess_df(validation)
model = Sequential()
model.add(LSTM(16,input_shape=(trainx.shape[1:]),return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(16,input_shape=(trainx.shape[1:]),return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(LSTM(16,input_shape=(trainx.shape[1:])))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Dense(16,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2,activation="softmax"))
opt = Adam(lr=lr,decay=decay)
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
tesorboard = TensorBoard(log_dir='logs/'+str(name))
history = model.fit(x=trainx,y=np.array(trainy),batch_size=batch_size,epochs=epoch,validation_data=(validationx,np.array(validationy)),callbacks=[tesorboard])
