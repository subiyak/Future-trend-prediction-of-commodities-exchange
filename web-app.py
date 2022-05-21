import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas_datareader as data
from keras.models import load_model#model is loaded here so that everytime epoch is not required
import streamlit as st

start ='2010-01-01'
end='2020-12-31'

st.title('Future trend prediction of commodities exchange')

user_input=st.text_input('Enter Stock Ticker','AAPL')#input from user
df=data.DataReader(user_input,'yahoo',start,end)

#Describing Data
st.subheader('Data from 2010-2020')
st.write(df.describe())

#visualization
st.subheader('Closing Price Vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.plot(df.Close)
st.pyplot(fig)



#moving average
st.subheader('Closing Price Vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.plot(ma100,'r')
plt.plot(df.Close,'b')
st.pyplot(fig)


st.subheader('Closing Price Vs Time Chart with 100MA and 200 MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean() #moving average
fig=plt.figure(figsize=(12,6))
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close,'b')
st.pyplot(fig)

#model training
#spliting data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)]) #70 % data as trainig data
data_testing=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])#30 % as testing and if we add the splitting values we get the above number 2517

print(data_training.shape)
print(data_testing.shape)

#for stack lstm model we have to scale down the data
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)




#load the already trained  model
model=load_model('keras_model.h5')

#predictions/testing part
past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test,y_test = np.array(x_test),np.array(y_test)

#making predictions

y_predicted=model.predict(x_test)

scaler=scaler.scale_
scale_factor=1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#Final Graph

st.subheader("Prediction Vs Original")
fig2= plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original Price')
plt.plot(y_predicted,'r',label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
