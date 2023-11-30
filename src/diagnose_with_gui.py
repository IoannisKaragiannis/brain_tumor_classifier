import tkinter
from PIL import Image
from tkinter import filedialog
import cv2 as cv
from frames import *
from display_tumor import *
import utils

import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = str(FILE.parents[0].parents[0])  # object_detection root directory
HOME = os.path.expanduser( '~' )
MODELS_PATH = ROOT + "/models/"
HISTORY_PATH = ROOT + "/history/"

class Gui:
    MainWindow = 0
    listOfWinFrame = list()
    FirstFrame = object()
    val = 0
    fileName = 0
    DT = object()

    def __init__(self, args):
        global MainWindow
        MainWindow = tkinter.Tk()
        self.wHeight = 600
        self.wWidth = 800
        MainWindow.geometry(f'{self.wWidth}x{self.wHeight}')
        MainWindow.resizable(width=False, height=False)
        
        self.image_path = None

        self.args = args

        self.DT = DisplayTumor()

        target_cnn_shape = (args.input_size, args.input_size, 3)

        self.model = utils.Classifier(args)

        print(f"loading model: {args.model_name}.h5")
        self.model.load(MODELS_PATH + args.model_name+".h5", HISTORY_PATH + args.model_name + ".npy")
        print(f"model_summary: {self.model.summary}")

        self.fileName = tkinter.StringVar()

        self.FirstFrame = Frames(self, MainWindow, self.wWidth, self.wHeight, 0, 0)
        # self.FirstFrame.btnView['state'] = 'disable'

        self.listOfWinFrame.append(self.FirstFrame)

        WindowLabel = tkinter.Label(self.FirstFrame.getFrames(), text="Brain Tumor Classification", height=1, width=40)
        WindowLabel.place(x=int(0.05*self.wHeight), y=int(0.01*self.wHeight))
        WindowLabel.configure(background="White", font=("Calibri", 16, "bold"))

        self.val = tkinter.IntVar()

        self.RB1 = tkinter.Radiobutton(self.FirstFrame.getFrames(), text="Diagnose MRI", variable=self.val,
                                  value=1, command=self.check)
        self.RB1.place(x=int(0.8*self.wWidth), y=int(0.3*self.wHeight))

        browseBtn = tkinter.Button(self.FirstFrame.getFrames(), text="Browse", width=8, command=self.browseWindow)
        browseBtn.place(x=int(0.8*self.wWidth), y=int(0.2*self.wHeight))

        MainWindow.mainloop()

    def getListOfWinFrame(self):
        return self.listOfWinFrame

    def browseWindow(self):
        global mriImage
        FILEOPENOPTIONS = dict(defaultextension='*.*',
                               filetypes=[('jpg', '*.jpg'), ('png', '*.png'), ('jpeg', '*.jpeg'), ('All Files', '*.*')])
        self.fileName = filedialog.askopenfilename(**FILEOPENOPTIONS)
        image = Image.open(self.fileName)
        imageName = str(self.fileName)
        self.image_path = imageName
        mriImage = cv.imread(imageName, 1)
        self.listOfWinFrame[0].readImage(image)
        self.listOfWinFrame[0].displayImage()
        self.DT.readImage(image)
        # clean diagnosis
        resLabel = tkinter.Label(self.FirstFrame.getFrames(), text="", height=1, width=25)
        resLabel.configure(background="White", font=("Calibri", 14, "bold"), fg=self.model.class_colors[0])
        resLabel.place(x=int(0.1*self.wWidth), y=int(0.85*self.wHeight))

    def check(self):
        global mriImage
        print(f"self.val.get(): {self.val.get()}")
        if (self.val.get() == 1):
            self.listOfWinFrame = 0
            self.listOfWinFrame = list()
            self.listOfWinFrame.append(self.FirstFrame)

            self.listOfWinFrame[0].setCallObject(self.DT)

            # res = predictTumor(mriImage)

            if self.image_path is not None:

                res = self.model.make_single_prediction(self.image_path)
                
                if res == self.model.class_names[0]:
                    resLabel = tkinter.Label(self.FirstFrame.getFrames(), text="Tumor Detected (Glioblastoma)", height=1, width=25)
                    resLabel.configure(background="White", font=("Calibri", 14, "bold"), fg=self.model.class_colors[0])
                elif res == self.model.class_names[1]:
                    resLabel = tkinter.Label(self.FirstFrame.getFrames(), text="Tumor Detected (Meningioma)", height=1, width=25)
                    resLabel.configure(background="White", font=("Calibri", 14, "bold"), fg=self.model.class_colors[1])
                elif res == self.model.class_names[3]:
                    resLabel = tkinter.Label(self.FirstFrame.getFrames(), text="Tumor Detected (Pituitary)", height=1, width=25)
                    resLabel.configure(background="White", font=("Calibri", 14, "bold"), fg=self.model.class_colors[3])
                else:
                    resLabel = tkinter.Label(self.FirstFrame.getFrames(), text="No Tumor Detected (Healthy)", height=1, width=25)
                    resLabel.configure(background="White", font=("Calibri", 14, "bold"), fg=self.model.class_colors[2])

            else:
                resLabel = tkinter.Label(self.FirstFrame.getFrames(), text="Please provide MRI image", height=1, width=25)
                resLabel.configure(background="White", font=("Calibri", 14, "bold"), fg="red")

            resLabel.place(x=int(0.1*self.wWidth), y=int(0.85*self.wHeight))
            self.val.set(0)

        elif (self.val.get() == 2):
            self.listOfWinFrame = 0
            self.listOfWinFrame = list()
            self.listOfWinFrame.append(self.FirstFrame)

            self.listOfWinFrame[0].setCallObject(self.DT)
            self.listOfWinFrame[0].setMethod(self.DT.removeNoise)
            secFrame = Frames(self, MainWindow, self.wWidth, self.wHeight, self.DT.displayTumor, self.DT)

            self.listOfWinFrame.append(secFrame)

            for i in range(len(self.listOfWinFrame)):
                if (i != 0):
                    self.listOfWinFrame[i].hide()
            self.listOfWinFrame[0].unhide()

            if (len(self.listOfWinFrame) > 1):
                self.listOfWinFrame[0].btnView['state'] = 'active'

        else:
            print("Not Working")

mainObj = Gui(utils.parse_opt())