from threading import Thread
from keras.models import load_model
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
import tkinter as tk
import tkinter.messagebox
import win32gui
from PIL import ImageGrab
import numpy as np
from keras.optimizers import SGD
import matplotlib.pyplot as plt

num_classes = 0
model = load_model('mnist.h5')
img = None
incorrect_imgs = []
incorrect_answ = []
state = False
stateChoice = 0


def predict_digit(image):
    global img
    img = image.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape((1, 28, 28, 1))
    img = img / 255.0
    res = model.predict([img])[0]
    return np.argmax(res), max(res), res


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def save_model():
    global num_classes, model
    (X_train, y_train), (X_test, y_test) = mnist.load_data('mnist.h5')
    # reshape to be [samples][width][height][channels]
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
    X_train = X_train / 255
    X_test = X_test / 255
    if len(incorrect_imgs) > 0:
        for i in range(len(incorrect_imgs)):
            X_train = np.append(X_train, incorrect_imgs[i], axis=0)
            y_train = np.append(y_train, [incorrect_answ[i]], axis=0)

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    mdl = define_model()
    mdl.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
    mdl.save('mnist.h5')


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.res = None
        self.canvas = tk.Canvas(self, width=300, height=300, bg="black", cursor="cross")
        self.label = tk.Label(self, text='<= Draw a number', font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Try again", command=self.clear_all)
        self.button_report = tk.Button(self, text="Report", command=lambda: self.report(self.res))
        self.button_incorrect = tk.Button(self, text="Mark as incorrect", command=self.mark_as_incorrect, bg='red',
                                          foreground='white')
        self.button_correct = tk.Button(self, text="Mark as correct", command=self.mark_as_correct, bg='green',
                                        foreground='white')
        self.label_tries = tk.Label(self, text='Number of tries: ')
        self.label_correct = tk.Label(self, text='Correct: ')
        self.label_incorrect = tk.Label(self, text='Incorrect: ')

        self.label_model = tk.Label(self, borderwidth=2, relief="groove")
        self.correct_number = tk.Entry(self.label_model, font=("Helvetica", 12))
        self.correct_number_label = tk.Label(self.label_model, text='Write the correct number: ')
        self.correct_number_button = tk.Button(self.label_model, text="Save this instance", command=self.save_image)

        self.canvas.grid(row=1, column=0, pady=2, sticky="W")
        self.label.grid(row=1, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=2, column=1, pady=2, padx=2)
        self.button_clear.grid(row=2, column=0, pady=2)
        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.statistics()

    def on_closing(self):
        if len(incorrect_imgs)>0 and tk.messagebox.askyesno("Quit", "Do you want to save model?"):
            t=Thread(target=save_model)
            t.start()
        self.destroy()


    def save_image(self):
        incorrect_imgs.append(img)
        incorrect_answ.append(int(self.correct_number.get()))
        tkinter.messagebox.showinfo(title="Uspesno!", message="Uspesno ste sacuvali!")
        self.clear_all()

    def clear_all(self):
        global stateChoice, state
        state = False
        stateChoice = 0
        self.canvas.delete("all")
        self.label.configure(text='<= Draw a number')
        self.canvas.config(bg="black")
        self.button_incorrect.grid_forget()
        self.button_correct.grid_forget()
        self.button_report.grid_forget()
        self.label_model.grid_forget()
        self.correct_number.grid_forget()
        self.correct_number_label.grid_forget()
        self.correct_number_button.grid_forget()

    def mark_as_correct(self):
        global state, stateChoice
        self.label_model.grid_forget()
        self.correct_number.grid_forget()
        self.correct_number_label.grid_forget()
        self.correct_number_button.grid_forget()
        self.canvas.config(bg="green")
        file = open("dat.txt", "r+")
        correct = int(file.readline());
        incorrect = int(file.readline())
        if state & (stateChoice == 2):
            incorrect = incorrect - 1
            correct = correct + 1
        elif state & (stateChoice == 0):
            correct = correct + 1
        file.seek(0, 0)
        file.write(str(correct) + "\n" + str(incorrect))
        file.close()
        stateChoice = 1
        self.statistics()

    def mark_as_incorrect(self):
        global state, stateChoice
        self.canvas.config(bg="red")
        file = open("dat.txt", "r+")
        correct = int(file.readline());
        incorrect = int(file.readline())
        if state & (stateChoice == 1):
            correct = correct - 1
            incorrect = incorrect + 1
        elif state & (stateChoice == 0):
            incorrect = incorrect + 1

        file.seek(0, 0)
        file.write(str(correct) + "\n" + str(incorrect))
        file.close()
        stateChoice = 2
        self.statistics()
        self.label_model.grid(row=4, column=0, pady=2)
        self.correct_number_label.grid(row=4, column=0, pady=2)
        self.correct_number.grid(row=4, column=1, pady=2)
        self.correct_number_button.grid(row=4, column=2, pady=2)

        # x=img; y=unetBroj;
        # da se ubacuje u listu ako je incorrect

    def classify_handwriting(self):
        global state, stateChoice
        state = True
        stateChoice = 0
        HWND = self.canvas.winfo_id()  # get the handle of the canvas
        rect = win32gui.GetWindowRect(HWND)  # get the coordinate of the canvas
        a, b, c, d = rect
        rect = (a + 4, b + 4, c - 4, d - 4)
        im = ImageGrab.grab(rect)
        digit, acc, self.res = predict_digit(im)
        self.label.configure(text='Number is: ' + str(digit) + '\n Percentage: ' + str(int(acc * 100)) + '%')
        self.canvas.config(bg="blue")
        self.button_incorrect.grid(row=3, column=0, pady=5)
        self.button_correct.grid(row=3, column=1, pady=5)
        self.button_report.grid(row=5, column=2, pady=5)


    def report(self, res):
        labels = range(10)
        values = [i * 100 for i in res]
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        axs[0].bar(labels, values)
        axs[1].scatter(labels, values)
        axs[2].plot(labels, values, color='blue', linestyle='dashdot', linewidth=1, marker='o', markerfacecolor='red',
                    markeredgecolor='black', markersize=4)
        for i in range(0, 3):
            axs[i].set_ylabel('Percentage')
            axs[i].set_xticks(labels)
            for a, b in zip(labels, values):
                b = round(float(b), 2)
                if b > 0:
                    axs[i].text(a, b, '~' + str(b) + '%')

        fig.suptitle('Report')
        plt.show()

    def draw_lines(self, event):
        global stateChoice, state
        state = False
        stateChoice = 0
        self.canvas.config(bg="black")
        self.button_incorrect.grid_forget()
        self.button_correct.grid_forget()
        self.button_report.grid_forget()

        self.label_model.grid_forget()
        self.correct_number.grid_forget()
        self.correct_number_label.grid_forget()
        self.correct_number_button.grid_forget()
        self.label.configure(text='<= Draw a number')
        r = 5
        self.x = event.x
        self.y = event.y
        self.canvas.create_line(self.x + r, self.y - r, self.x - r, self.y + r, fill='white', width=30)

    def statistics(self):
        file = open("dat.txt", "r+")
        correct = int(file.readline());
        incorrect = int(file.readline())
        self.label_tries.grid(row=0, column=0, pady=2, padx=2)
        self.label_correct.grid(row=0, column=1, pady=2, padx=2)
        self.label_incorrect.grid(row=0, column=2, pady=2, padx=2)
        self.label_tries.config(text="Number of tries: " + str(correct + incorrect))
        self.label_correct.config(text="Correct: " + str(correct))
        self.label_incorrect.config(text="Incorrect: " + str(incorrect))
        file.close()


app = App()
tk.mainloop()


