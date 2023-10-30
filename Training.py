import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Activation,Dropout
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix
from plot_cm import plot_confusion_matrix
from tensorflow.keras.utils import to_categorical

df=pd.read_pickle("final_audio_data_csv/audio_data1.csv")
X=df["feature"].values
X=np.concatenate(X,axis=0).reshape(len(X),40)

Y=np.array(df["class_label"].tolist())
Y= to_categorical(Y)

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, random_state=42)

model=Sequential([
    Dense(256, input_shape=X_train[0].shape),
    Activation("relu"),
    Dropout(0.5),
    Dense(256),
    Activation("relu"),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

print(model.summary())

model.compile(
    loss="categorical_crossentropy",
    optimizer='adam',
    metrics=['accuracy']
)

print("Model score: \n")
history=model.fit(X_train, Y_train, epochs=1000)
model.save("saved_model/WWD1.h5")
score=model.evaluate(X_test,Y_test)
print(score)

print("Model Classification Report: \n")
Y_pred = np.argmax(model.predict(X_test),axis=1)
cm= confusion_matrix(np.argmax(Y_test,axis=1), Y_pred)
print(classification_report(np.argmax(Y_test,axis=1),Y_pred))
plot_confusion_matrix(cm, classes=["Does not have HELP Word", "Has HELP Word"])




