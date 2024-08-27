import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import imageio
import numpy as np
from datetime import datetime
from tensorflow.keras.layers import Lambda
import matplotlib.pyplot as plt
import math
import xlsxwriter
from pyquaternion import Quaternion

#from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau 

from sklearn.model_selection import train_test_split

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

"""
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
"""
# "QUATERNION"  or "AXIS-ANGLE" 
mode = "QUATERNION"

#Get the current time
def get_time() : 
    now = datetime.now()
    current_time = now.strftime("%d-%m--%H-%M")
    return current_time

def quat_to_axis(q):
      
    quat = Quaternion(q)
    axis = quat.get_axis()
    aa = quat.degrees
    ax = axis[0]
    ay = axis[1]
    az = axis[2]
    axis_numpy = np.array([aa,ax,ay,az])
    
    if axis_numpy[0] < 0:
        axis_numpy = axis_numpy *-1
    
    return axis_numpy

def excel_write(prediction_diff):
    
    #current_time = get_time()
    file_name = os.path.join("graphs/", get_time() + '.xlsx')
    
    workbook = xlsxwriter.Workbook(file_name)
    worksheet = workbook.add_worksheet()
    
    worksheet.set_column('A:C', 10)
    bold = workbook.add_format({'bold': True, 'center_across': True })
    worksheet.write('A1', 'Distance', bold)
    worksheet.write('B1', 'Tilt', bold)
    worksheet.write('C1', 'Rotation', bold)
    for idx, x in np.ndenumerate(prediction_diff):
        worksheet.write(idx[0]+1, idx[1], x)
   
    workbook.close()   
    return True    

def write_to_file(prediction, data):
    f = open("data\prediction.txt", "w")
    for x in range(len(prediction)):
        string = str(prediction[x,0]) + ", " + str(prediction[x,1]) + ", " + str(prediction[x,2]) + ", " + str(prediction[x,3]) + ", " + str(prediction[x,4]) + ", " + str(prediction[x,5]) + ", " + str(prediction[x,6])
        f.write(string + '\n')
    f.close()
    
    f = open("data\labels.txt", "w")
    for x in range(len(data)):
        string = str(data[x,0]) + ", " + str(data[x,1]) + ", " + str(data[x,2]) + ", " + str(data[x,3]) + ", " + str(data[x,4]) + ", " + str(data[x,5]) + ", " + str(data[x,6])
        f.write(string + '\n')
    f.close()
    
    
    return True

def load_data(filePath) :
    
    labels = []
    images = []
    
    with open(filePath, "r") as File :
        for line in File:                   
            elements = line.split(",")
            image_name = elements[0]
            #Euler XYZ
            a = float(elements[1])
            b = float(elements[2])
            c = float(elements[3])
            #Translation XYZ
            x = float(elements[4])
            y = float(elements[5])
            z = float(elements[6])
            #Quaternion WXYZ
            qw = float(elements[7])
            qx = float(elements[8])
            qy = float(elements[9])
            qz = float(elements[10])
            #Axis-Angle WXYZ
            aw = float(elements[11])
            ax = float(elements[12])
            ay = float(elements[13])
            az = float(elements[14])
                      
            if mode == "QUATERNION":
                labels.append([x,y,z,qw,qx,qy,qz])
            elif mode == "AXIS-ANGLE":
                labels.append([x,y,z,aw,ax,ay,az])
            
                
            data_dir,_ = os.path.split(filePath)
            image_path = os.path.join(data_dir, image_name)
            images.append(imageio.imread(image_path))
            
    return np.array(images, dtype=np.uint8), np.array(labels)


def create_network() :         
    
    inputs = layers.Input(shape=(250,250,3))
    
    x = Lambda(lambda x: x/255.0-0.5)(inputs)
    
    x = layers.Conv2D(16, (5,5),activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(32, (5,5),activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(64, (5,5),activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x) 
       
    x = layers.Conv2D(128, (3,3),activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    
    x = layers.Conv2D(128, (3,3),activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
    
    x = layers.Conv2D(128, (3,3),activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)
       

    x = layers.Flatten()(x)

    x = layers.Dense(10, activation='relu')(x)
    
    outputs = layers.Dense(7, activation='linear')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer='adam', loss='mse')
       
    return model

def create_graph(prediction_diff):
    
    #Find length of data
    x = list(range(1, len(prediction_diff) + 1))
    
    #A
    x = prediction_diff[:,[0]]   
    plt.hist(x, bins=20)
    plt.ylabel('Number of results in Bracket') 
    plt.xlabel('Distance Error (Cm)') 
    plt.title('Distance Error') 
    plt.show() 
    
    #B
    x = prediction_diff[:,[1]]   
    plt.hist(x, bins=20)
    plt.ylabel('Number of results in Bracket') 
    plt.xlabel('Tilt Error (degrees)') 
    plt.title('Tilt Error') 
    plt.show() 
    
    #C
    x = prediction_diff[:,[2]]   
    plt.hist(x, bins=20)
    plt.ylabel('Number of results in Bracket') 
    plt.xlabel('Rotational Error (degrees)') 
    plt.title('Rotational Error') 
    plt.show() 
    
    return True

def predict_models(model):
    labels_predict = model.predict(images_test)
    
    if mode == "QUATERNION":
        #Convert labels_predict to axis angles
        for x in range(len(labels_predict)):
            q = labels_predict[x,3:7] 
                           
            axis = quat_to_axis(q)
            
            labels_predict[x,3] = axis[0]
            labels_predict[x,4] = axis[1]
            labels_predict[x,5] = axis[2]
            labels_predict[x,6] = axis[3]
        
        #Convert labels_test to axis angles labels_test
        for x in range(len(labels_test)):
            q = labels_test[x,3:7]
                           
            axis = quat_to_axis(q)
            
            labels_test[x,3] = axis[0]
            labels_test[x,4] = axis[1]
            labels_test[x,5] = axis[2]
            labels_test[x,6] = axis[3]
    """           
    if mode == "AXIS-ANGLE":
        for idx, x in np.ndenumerate(labels_predict):
            if idx[1] == 3:
                labels_predict[idx[0],3] = labels_predict[idx[0],3] *180
    """
                
    prediction_diff = np.zeros((len(labels_predict),3))

    #Determine differences in distance, angle and rotation
    for x in range(len(labels_predict)):
        
        #Distance between unit vectors
        prediction_diff[x,0] = (math.sqrt(((labels_predict[x,0]-labels_test[x,0])**2)+((labels_predict[x,1]-labels_test[x,1])**2)+((labels_predict[x,2]-labels_test[x,2])**2))) * 100 
        #----------------------------------------- 
        #Angle between unit vectors
        a1 = labels_predict[x,4]
        a2 = labels_predict[x,5]
        a3 = labels_predict[x,6]
        b1 = labels_test[x,4]
        b2 = labels_test[x,5]
        b3 = labels_test[x,6]
        
        cdot = a1*b1 + a2*b2 + a3*b3
        a_mag = math.sqrt((a1**2)+(a2**2)+(a3**2))
        b_mag = math.sqrt((b1**2)+(b2**2)+(b3**2))
        
        angle = math.acos( cdot/(a_mag*b_mag) )
        angle = angle*180/math.pi
        
        prediction_diff[x,1] = angle
        #----------------------------------------- 
        a1 = labels_predict[x,3]
        a2 = labels_test[x,3]
        
        if a1 >= 0:
            if a2 >= 0:
                same_sign = True
            else:
                same_sign = False
        elif a1 < 0:
            if a2 < 0:
                same_sign = True
            else:
                same_sign = False
        if same_sign == True:
            prediction_diff[x,2] = math.sqrt((a1)**2) - math.sqrt((a2)**2)
        elif same_sign == False:
            prediction_diff[x,2] = math.sqrt((a1)**2) + math.sqrt((a2)**2)
         
        if prediction_diff[x,2] > 180:
            prediction_diff[x,2] = prediction_diff[x,2] - 360
        if prediction_diff[x,2] < -180:
            prediction_diff[x,2] = prediction_diff[x,2] + 360
            
        prediction_diff[x,2] = math.sqrt((prediction_diff[x,2])**2)  
        
        
        if mode == "AXIS-ANGLE":
            prediction_diff[x,2] = prediction_diff[x,2] * 180
    
    #prediction_diff = labels_test - labels_predict                 #Find the prediction difference    
    #prediction_diff[:,[0,1,2]] = prediction_diff[:,[0,1,2]] * 100  #Rescale translation to -100 & 100    
    #column_diff = prediction_diff.mean(axis=0)                     #Find the average of each column
    
    return labels_predict, labels_test, prediction_diff



if __name__ == "__main__" :
    
    now = datetime.now()
    current_time = now.strftime("%d-%m--%H-%M")
    file_time = os.path.join('models', current_time + "-model.h5")
    
    
    #earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min' )  
    #reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.90, patience=7, verbose=1, epsilon=1e-4, mode='min')
    
    mcp_save = tf.keras.callbacks.ModelCheckpoint(file_time, save_best_only=True, monitor='val_loss', mode='min')
    
    
    images, labels = load_data("data\data.txt")
    
    #labels = labels[:,[3,4,5]]                      #Remove rotational part of label
    #images = images[:200]                          #Take first 1000 elements -> for testing purposes
    #labels = labels[:200]                          #Take first 1000 elements -> for testing purposes
    

    if mode == "AXIS-ANGLE":  
        labels[:,[3]] = labels[:,[3]] / 180             #Scale w to 0 to 2
        labels[:,[3]] = labels[:,[3]] - 1               #Scale w to -1 to 1

    


    
    images_train, images_test, labels_train, labels_test = train_test_split(images, 
                                                                        labels, 
                                                                        test_size=0.20, 
                                                                        random_state=42)
    
    
       
    model = create_network()
    model.load_weights("models/quat-best-0.1449.h5")   
        
    """
    model.fit(images_train, labels_train, batch_size=20, epochs=100000, 
              validation_data=(images_test,labels_test), 
              shuffle=True, 
              callbacks=[mcp_save],
              #verbose=1
              ) 
    """
    
       
    labels_predict, labels_test, prediction_diff = predict_models(model)
    create_graph(prediction_diff)
    excel_write(prediction_diff)
    
    write_to_file(labels_predict,labels_test)    
    #https://biokamikazi.wordpress.com/2016/07/07/export-numpy-array-to-excel-in-python/
    #https://pypi.org/project/XlsxWriter/