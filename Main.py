
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import pickle
from PIL import Image, ImageTk 
import cv2
from skimage import color
from skimage.feature import greycomatrix, greycoprops
import scipy.stats as stats
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split 
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from keras.applications.resnet50 import ResNet50
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json

from lime import lime_image
from skimage.segmentation import mark_boundaries


global filename
global X,Y
accuracy = []
precision = []
recall = []
fscore = []
global X_train, X_test, y_train, y_test
global resnet50
global labels

labels = ['No_Oral_Cancer','Oral_Cancer']

with open('model/model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    resnet50_classifier = model_from_json(loaded_model_json)
json_file.close()    
resnet50_classifier.load_weights("model/model_weights.h5")
resnet50_classifier._make_predict_function() 


main = tkinter.Tk()
main.title("Oral Cancer detection using Deep Learning Algorithm") #designing main screen
main.geometry("1300x1200")

def remove_green_pixels(image):
  # Transform from (256,256,3) to (3,256,256)
  channels_first = channels_first_transform(image)

  r_channel = channels_first[0]
  g_channel = channels_first[1]
  b_channel = channels_first[2]

  # Set those pixels where green value is larger than both blue and red to 0
  mask = False == np.multiply(g_channel > r_channel, g_channel > b_channel)
  channels_first = np.multiply(channels_first, mask)

  # Transfrom from (3,256,256) back to (256,256,3)
  image = channels_first.transpose(1, 2, 0)
  return image

def rgb2lab(image):
  return color.rgb2lab(image)

def rgb2gray(image):
  return np.array(color.rgb2gray(image) * 255, dtype=np.uint8)

def glcm(image, offsets=[1], angles=[0], squeeze=False): #extract glcm features
  single_channel_image = image if len(image.shape) == 2 else rgb2gray(image)
  gclm = greycomatrix(single_channel_image, offsets, angles)
  return np.squeeze(gclm) if squeeze else gclm

def histogram_features_bucket_count(image): #texture features will be extracted using histogram
  image = channels_first_transform(image).reshape(3,-1)

  r_channel = image[0]
  g_channel = image[1]
  b_channel = image[2]

  r_hist = np.histogram(r_channel, bins = 26, range=(0,255))[0]
  g_hist = np.histogram(g_channel, bins = 26, range=(0,255))[0]
  b_hist = np.histogram(b_channel, bins = 26, range=(0,255))[0]

  return np.concatenate((r_hist, g_hist, b_hist))

def histogram_features(image):
  color_histogram = np.histogram(image.flatten(), bins = 255, range=(0,255))[0]
  return np.array([
    np.mean(color_histogram),
    np.std(color_histogram),
    stats.entropy(color_histogram),
    stats.kurtosis(color_histogram),
    stats.skew(color_histogram),
    np.sqrt(np.mean(np.square(color_histogram)))
  ])

def texture_features(full_image, offsets=[1], angles=[0], remove_green = True):
  image = remove_green_pixels(full_image) if remove_green else full_image
  gray_image = rgb2gray(image)
  glcmatrix = glcm(gray_image, offsets=offsets, angles=angles)
  return glcm_features(glcmatrix)

def glcm_features(glcm):
  return np.array([
    greycoprops(glcm, 'correlation'),
    greycoprops(glcm, 'contrast'),
    greycoprops(glcm, 'energy'),
    greycoprops(glcm, 'homogeneity'),
    greycoprops(glcm, 'dissimilarity'),
  ]).flatten()

def channels_first_transform(image):
  return image.transpose((2,0,1))

def extract_features(image):
  offsets=[1,3,10,20]
  angles=[0, np.pi/4, np.pi/2]
  channels_first = channels_first_transform(image)
  return np.concatenate((
      texture_features(image, offsets=offsets, angles=angles),
      texture_features(image, offsets=offsets, angles=angles, remove_green=False),
      histogram_features_bucket_count(image),
      histogram_features(channels_first[0]),
      histogram_features(channels_first[1]),
      histogram_features(channels_first[2]),
      ))

def getID(name):
    index = 0
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index 

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir = ".")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')    
    text.insert(END,"Different Types Found in Dataset : "+str(labels)+"\n\n") 
    


    
def featuresExtraction():
    global filename
    global X,Y
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists("model/X.npy"):
        X = np.load('model/X.npy')
        Y = np.load('model/Y.npy')
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (64,64))
                    class_label = getID(name)
                    features = extract_features(img)
                    Y.append(class_label)
                    X.append(features)
                    print(name+" "+root+"/"+directory[j]+" "+str(features.shape)+" "+str(class_label))
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save("model/X",X)
        np.save("model/Y",Y)
    X = X.astype('float32')
    X = X/255 #features normalization
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    #text.insert(END,"Extracted Features : "+str(X[0])+"\n\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset train & test split. 80% dataset images used for training and 20% for testing\n\n")
    text.insert(END,"80% training images : "+str(X_train.shape[0])+"\n\n")
    text.insert(END,"20% testing images : "+str(X_test.shape[0])+"\n\n")

def calculateMetrics(algorithm, predict, y_test):
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)
    text.insert(END,algorithm+" Accuracy  :  "+str(a)+"\n")
    text.insert(END,algorithm+" Precision : "+str(p)+"\n")
    text.insert(END,algorithm+" Recall    : "+str(r)+"\n")
    text.insert(END,algorithm+" FScore    : "+str(f)+"\n\n")
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 3)) 
    ax = sns.heatmap(conf_matrix, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,len(labels)])
    plt.title(algorithm+" Confusion matrix") 
    plt.xticks(rotation=90)
    plt.ylabel('True class') 
    plt.xlabel('Predicted class')
    plt.tight_layout()
    plt.show() 

def runSVM():
    global X_train, X_test, y_train, y_test, X, Y
    global accuracy, precision, recall, fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    text.delete('1.0', END)

    # Retrain the SVM model regardless of previous pickle version to avoid _probA issue
    svm_cls = svm.SVC()
    svm_cls.fit(X, Y)

    # Save updated model in compatible format
    with open('model/svm.txt', 'wb') as file:
        pickle.dump(svm_cls, file)

    # Predict and evaluate
    predict = svm_cls.predict(X_test)
    calculateMetrics("SVM", predict, y_test)



def runresnet50():
    global X_train, X_test, y_train, y_test, X, Y, resnet50
    global accuracy, precision,recall, fscore
    Y1 = to_categorical(Y)
    XX = np.reshape(X, (X.shape[0], X.shape[1], 1, 1))
    X_train, X_test, y_train, y_test = train_test_split(XX, Y1, test_size=0.2,random_state=40)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            resnet50 = model_from_json(loaded_model_json)
        json_file.close()    
        resnet50.load_weights("model/model_weights.h5")
        resnet50._make_predict_function()   
    else:
        resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
        resnet50 = Sequential()
        resnet50.add(Convolution2D(32, 1, 1, input_shape = (XX.shape[1], XX.shape[2], XX.shape[3]), activation = 'relu'))
        resnet50.add(MaxPooling2D(pool_size = (1, 1)))
        resnet50.add(Convolution2D(32, 1, 1, activation = 'relu'))
        resnet50.add(MaxPooling2D(pool_size = (1, 1)))
        resnet50.add(Flatten())
        resnet50.add(Dense(output_dim = 256, activation = 'relu'))
        resnet50.add(Dense(output_dim = Y1.shape[1], activation = 'softmax'))
        resnet50.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = resnet50.fit(XX, Y1, batch_size=12, epochs=50, shuffle=True, verbose=2)
        resnet50.save_weights('model/model_weights.h5')            
        model_json = resnet50.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    #print(resnet50.summary())
    predict = resnet50.predict(X_test)
    predict = np.argmax(predict, axis=1)
    y_test = np.argmax(y_test, axis=1)
    calculateMetrics("Propose resnet50", predict, y_test)


def graph():
    df = pd.DataFrame([['SVM','Accuracy',accuracy[0]],['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','FScore',fscore[0]],['Propose resnet50','Accuracy',accuracy[1]],['Propose resnet50','Precision',precision[1]],['Propose resnet50','Recall',recall[1]],['Propose resnet50','FScore',fscore[1]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

def show_precaution_popup():
    popup = Toplevel(main)
    popup.title("Oral Cancer Care & Precautions")
    popup.geometry("520x420")
    popup.config(bg='misty rose')

    title = Label(popup, text="Oral Cancer Care & Precautions", font=('Helvetica', 14, 'bold'), bg='misty rose', fg='maroon')
    title.pack(pady=12)

    precautions = [
        "1. Brush twice daily and floss regularly.",
        "2. Quit tobacco and limit alcohol consumption.",
        "3. Eat a balanced diet rich in vitamins and antioxidants.",
        "4. Avoid prolonged sun exposure to your lips.",
        "5. Schedule regular oral screenings and dental visits.",
        "6. Consult a doctor if you notice ulcers, lumps, or bleeding.",
        "7. Manage stress effectively and stay well hydrated."
    ]

    for tip in precautions:
        tip_label = Label(popup, text=tip, font=('Helvetica', 11), bg='misty rose', anchor='w', justify=LEFT, wraplength=480)
        tip_label.pack(padx=15, anchor='w')

    close_btn = Button(popup, text="Close", command=popup.destroy, bg='lightcoral', font=('Helvetica', 11, 'bold'))
    close_btn.pack(pady=20)

def predict1():
    global resnet50
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    test = []
    img = cv2.resize(img, (64,64))
    features = extract_features(img)
    test.append(features)
    test = np.asarray(test)
    test = test.astype('float32')
    test = test/255
    test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
    predict = resnet50.predict(test)
    predict = np.argmax(predict)

    animal_label = labels[predict]

    img = cv2.imread(filename)
    img = cv2.resize(img, (800,400))
    cv2.putText(img, f' Predicted as: {animal_label}', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow(f'Predicted as: {animal_label}', img)
    cv2.waitKey(0)
    
    if animal_label == 'Oral_Cancer':
        show_precaution_popup()


def predict():
    global resnet50
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename)
    test = []
    img_resized = cv2.resize(img, (64,64))
    features = extract_features(img_resized)
    test.append(features)
    test = np.asarray(test)
    test = test.astype('float32')
    test = test/255
    test = np.reshape(test, (test.shape[0], test.shape[1], 1, 1))
    predict_result = resnet50.predict(test)
    predict = np.argmax(predict_result)

    animal_label = labels[predict]

    # Display original image with label
    img_display = cv2.imread(filename)
    img_display = cv2.resize(img_display, (800,400))
    cv2.putText(img_display, f' Predicted as: {animal_label}', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # LIME image explanation
    # Reload image and preprocess for lime
    img_lime = cv2.imread(filename)
    img_lime = cv2.resize(img_lime, (64,64))
    img_arr = np.array(img_lime).astype('float32') / 255.0
    img_arr = img_arr.reshape(1, 64, 64, 3)

    # Define prediction function for LIME
    def predict_fn(images):
        images = images.astype('float32')
        features_batch = []
        for img in images:
            f = extract_features(cv2.resize(img, (64,64)))
            features_batch.append(f)
        features_batch = np.asarray(features_batch).astype('float32') / 255.0
        features_batch = features_batch.reshape(features_batch.shape[0], features_batch.shape[1], 1, 1)
        preds = resnet50.predict(features_batch)
        return preds

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_arr[0], predict_fn, top_labels=1, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    lime_image_out = mark_boundaries(temp / 2 + 0.5, mask)

    # Show results
    f, axarr = plt.subplots(1, 2, figsize=(12, 6))
    axarr[0].imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
    axarr[0].set_title(f'Predicted as: {animal_label}')
    axarr[0].axis('off')
    axarr[1].imshow(lime_image_out)
    axarr[1].set_title('LIME Explanation')
    axarr[1].axis('off')
    plt.tight_layout()
    plt.show()

    if animal_label == 'Oral_Cancer':
        show_precaution_popup()







def plot_training_metrics():
    if not os.path.exists("model/history.pckl"):
        messagebox.showerror("Error", "Training history not found.")
        return
    with open("model/history.pckl", "rb") as file:
        history = pickle.load(file)

    plt.figure(figsize=(8, 5))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['loss'], label='Train Loss')
    plt.legend()
    plt.title("Training Accuracy and Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Value")
    plt.show()


   


font = ('times', 16, 'bold')
title = Label(main, text='Oral Cancer detection using Deep Learning Algorithm')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


bg_img = Image.open("image.jpg")  
bg_img = bg_img.resize((1300, 1200), Image.ANTIALIAS)  

bg_img_tk = ImageTk.PhotoImage(bg_img)

bg_label = Label(main, image=bg_img_tk)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)


font = ('times', 16, 'bold')
#title = Label(main, text='Oral Cancer Detection using Deep Learning and Explainable AI')
#title.config(bg='greenyellow', fg='dodger blue')  
#title.config(font=font)           
#title.config(height=3, width=120)       
#title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=25,width=90)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=300)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Kaggle Dataset", command=uploadDataset)
uploadButton.place(x=50,y=300)
uploadButton.config(font=font1)  

featuresButton = Button(main, text="Extract Features", command=featuresExtraction)
featuresButton.place(x=50,y=350)
featuresButton.config(font=font1) 

svmButton = Button(main, text="Run Existing Algorithms", command=runSVM)
svmButton.place(x=50,y=400)
svmButton.config(font=font1)

resnet50Button = Button(main, text="Run Proposed RESNET50 Algorithm", command=runresnet50)
resnet50Button.place(x=50,y=450)
resnet50Button.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=50,y=500)
graphButton.config(font=font1)


graphButton1 = Button(main, text="Predict from test images", command=predict1)
graphButton1.place(x=50,y=550)
graphButton1.config(font=font1)

graphButton1 = Button(main, text="XAI for test images", command=predict)
graphButton1.place(x=50,y=600)
graphButton1.config(font=font1)

metricsButton = Button(main, text="Show Training Metrics Graph", command=plot_training_metrics)
metricsButton.place(x=50,y=650)
metricsButton.config(font=font1)

#main.config(bg='LightSkyBlue')
main.mainloop()
