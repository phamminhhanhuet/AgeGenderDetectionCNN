import tkinter as tk
from PIL import  Image,ImageTk
import PIL
from tkinter import ttk
from tkinter import filedialog
from tkinter import *
import cv2
import os
from matplotlib import pyplot
import random
import threading
import time

OPTION1 = "Auto Detection With Adience Dataset"
OPTION2 = "Pick Up Your Own Image"


dir = r'D:\Python\Adience Data\faces'
dir_aligned = r'D:\Python\Adience Data\aligned'
dir_fold = r'D:\Python\Adience Data'

age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

categories_fold = ['fold_frontal_0_data.txt', 'fold_frontal_1_data.txt', 'fold_frontal_2_data.txt', 'fold_frontal_3_data.txt', 'fold_frontal_4_data.txt']

MODEL_MEAN_VALUES = [102.4263377603, 102.7689143744, 102.895847746]

# len of MODEL_MEAN_VALUAS is 196608 = 384 x 512 for full-connected layers



class Window(tk.Tk):
    def __init__(self):
        super(Window, self).__init__()
 
        self.title("Age And Gender Classification With CNN & Viola-Jones")
        self.minsize(400,300)
        self.iconphoto(False, tk.PhotoImage(file='logouet.png'))
 
        
        #self.label_frame = ttk.LabelFrame(self, text = "Age and Gender Classification")
        #self.label_frame.grid(column =0, row = 0, padx = 20, pady = 20)
        #self.create_radio()
        
        global age_net
        global gender_net
        age_net, gender_net = self.caffe_models()
        
        global font 
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.button = ttk.Button(self, text = OPTION1, command = self.auto_classification)
        self.button.grid(column=0, row=0)
        
        self.button = ttk.Button(self, text = OPTION2, command = self.file_classification)
        self.button.grid(column=0, row=1) 
    
        self.create_canvas()
        #self.pack()
    
    
    def choose(self):
        ifile = filedialog.askopenfile(parent=self,mode='rb',title='Choose a file')
        path = Image.open(ifile)
    
        self.image2 = ImageTk.PhotoImage(path)
        self.label.configure(image=self.image2)
        self.label.image=self.image2
        
        
    def auto_classification(self):
        self.image_classification(age_net, gender_net)
 
    def rad_event(self):
        radSelected = self.radValues.get()
        if radSelected == 1:
            self.image_classification(age_net, gender_net)

        elif radSelected == 2:
            self.file_classification()


    def create_radio(self):
        self.radValues = tk.IntVar()
        self.rad1 = ttk.Radiobutton(self, text = OPTION1, value = 1, variable = self.radValues, command = self.rad_event)
        self.rad1.grid(column = 0, row = 0, sticky = tk.W, columnspan = 3)

        self.rad2 = ttk.Radiobutton(self, text=OPTION2, value=2, variable=self.radValues,command = self.rad_event)
        self.rad2.grid(column=0, row=1, sticky=tk.W, columnspan=3)
        
        
    def file_classification(self):
        filename = filedialog.askopenfilename(initialdir = "/", title='Please select one facial image.',
                           filetypes=[('Image Files', ['.jpeg', '.jpg', '.png', '.gif',
                                                       '.tiff', '.tif', '.bmp'])])
        image = cv2.imread(filename)
        self.predict_image(filename, "", mode = "manual")
        
        
 
    def create_canvas(self):
        global canvas
        canvas = tk.Canvas(self, height = 1000, width = 1000)
        coord = 10,50, 240, 210
 
        #canvas.pack(expand = YES, fill=BOTH)
        #arc = canvas.create_arc(coord, start = 0, extent = 150, fill = "yellow")
        #canvas.pack()
        #img = Image.open("8.png");
        
        img = PIL.Image.open("8.png", "r", formats=None);
        canvas.image = ImageTk.PhotoImage(img)
        canvas.create_image(0, 0, image=canvas.image, anchor='nw')
        canvas.grid(column = 0, row = 2)
        #canvas.pack()

    def caffe_models(self):
        age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
        gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
        
        return(age_net, gender_net)

    def image_classification(self, age_net, gender_net):
        count = 0
        folder = r'D:\Python\Adience Data\aligned'
        subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
        random.shuffle(subfolders)
        is_destroy = False
        for path in subfolders:    
            for img in os.listdir(path):
                self.predict_image(img, path, mode = "auto")
                k = cv2.waitKey(0) & 0xff
                if k == 27:
                    is_destroy = True
                    cv2.destroyAllWindows()
                    break
            if (is_destroy == True):
                break
                
    def predict_image(self, img, path, mode):
        imgpath = os.path.join(path, img)
        image = cv2.imread(imgpath)
        
        imgcv = PIL.Image.open(imgpath, "r", formats=None);
        canvas.image = ImageTk.PhotoImage(imgcv)
        canvas.create_image(0, 0, image=canvas.image, anchor='nw')
        canvas.grid(column = 0, row = 2)
        
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        pyplot.imshow(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image, 1.1, 5)
        print("=======================================")
        if(len(faces)>0):
            print("Found {} faces".format(str(len(faces))))
        
        for (x, y, w, h )in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
            
            #Get Face 
            face_img = image[y:y+h, h:h+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            #Predict Gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            
            #Predict Age
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            
            overlay_text = "%s %s" % (gender, age)
            cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            im = PIL.Image.fromarray(image)
            imgtk = PIL.ImageTk.PhotoImage(image=im) 
            
            
            print("Predicted: " + age + "," + gender)
        cv2.imshow('frame', image)
            
        if (mode == "auto"):
            is_find = False   
            for fold in categories_fold:
                    foldpath = os.path.join(dir_fold, fold)
                    token = open(foldpath,'r')
                    linestoken=token.readlines()
                    for x in linestoken:
                        if(x.split()[1] == img.split(".")[2] + ".jpg") :
                            print("Actual Group: " + x.split()[3] + x.split()[4] + ", " + x.split()[5])
                            is_find = True
                            break
                    token.close()
                    if is_find == True:
                        break    
        
class MyTestThread(threading.Thread):
    def run(self):
        for i in range(10):
            time.sleep(1)
            a = i+100     

if __name__ == "__main__":
    window = Window()
    t = MyTestThread()
    t.start()
    window.mainloop()
