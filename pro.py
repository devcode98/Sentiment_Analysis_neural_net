

import sys
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


dataset=pd.read_csv('bbc-text.csv')
text_data=dataset['text']

x=open('dev.txt','w')
i=0
while i<2225:
    str1=str(text_data[i])
    x.write(str1)
    i=i+1
x.close()
x=open('dev.txt','r')
para=x.read()
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
list_words=tokenizer.tokenize(para)

from nltk.corpus import stopwords 
stop_words = set(stopwords.words('english')) 
filtered_words = [w for w in list_words if not w in stop_words]

filtered_words=pd.DataFrame(filtered_words)
list_words=filtered_words.drop_duplicates(keep='first')
list_words=np.array(list_words)
list_words=np.transpose(list_words)
''' now we have final list of words with us '''


i=0
j=0
k=0
x=np.zeros((2225,29346))
while i<2225:
    str1=dataset['text'][i]
    bag=tokenizer.tokenize(str1)
    bag= [w for w in bag if not w in stop_words]
    bag.sort()
    ''' now we have shortened the string too '''
    j=0
    while j < len(bag):
         k=0
         
         while k<29346:
            
             if str(list_words[0][k])== str(bag[j]) :
                  x[i][k]=1
                  break
             
             k=k+1
               
         j=j+1       
    i=i+1
    print(i)



''' now we will be dividing our data(bag of words) '''
from sklearn.model_selection import train_test_split
y=dataset['category']
y=np.array(y)
y=y.reshape(-1,1)
encoder=OneHotEncoder(sparse=False)
y=encoder.fit_transform(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33)

input_size=(x_train.shape[1])
import tensorflow as tf



X=tf.placeholder(dtype=tf.float32,shape=(None,29346))
Y=tf.placeholder(dtype=tf.int32,shape=(None,5))

epoch=10
batch_size=150


classes=y_train.shape[1]

hidden_layer1=30
hidden_layer2=30
hidden_layer3=2

output_layer=5
sess=tf.InteractiveSession()
wa=tf.get_variable('wa',[input_size,hidden_layer1])
b1=tf.get_variable('b1',[hidden_layer1])
output_layer1=tf.nn.tanh((tf.add(tf.matmul(X,wa),b1)))

''' now we will move towards the second layer'''
w2=tf.get_variable('w2',[hidden_layer1,hidden_layer2])
b2=tf.get_variable('b2',hidden_layer2)
output_layer2=tf.nn.relu((tf.add(tf.matmul(output_layer1,w2),b2)))

''' now we will move towards the final layer'''
w3=tf.get_variable('w3',[hidden_layer2,output_layer])
b3=tf.get_variable('b3',output_layer)
output=(tf.add(tf.matmul(output_layer2,w3),b3))

'''
w4a=tf.get_variable('w4a',[hidden_layer3,output_layer])
b4=tf.get_variable('b4',output_layer)
output=(tf.add(tf.matmul(final_layer,w4a),b4))
'''
''' now we will be defining our cost function '''

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output,labels=Y))
opt=tf.train.AdamOptimizer().minimize(cost)   

''' now we will move towards training of model'''

init=tf.global_variables_initializer()
sess.run(init) 
list1=[]
sum=0       
for e in range(epoch):
    sum=0
    for i in range(0,len(y_train)):
        start=i*batch_size
        batch_x=x_train[start:start +batch_size]
        batch_y=y_train[start:start +batch_size]
        a,res= sess.run([opt,cost],feed_dict={X:batch_x,Y:batch_y})
        
          
    
pred=tf.equal(tf.math.argmax(output,1),tf.math.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(pred,tf.float32))
sess.run(accuracy,feed_dict={X:x_test,Y:y_test})
''' now we will be looking towards the accuracy '''

sess.close()