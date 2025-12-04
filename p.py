import os
from tkinter import *
from tkinter import filedialog, ttk, font as tkFont, messagebox
import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import arabic_reshaper
import numpy as np
from PIL import Image
from keras.saving.save import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from ctypes import windll
from bidi.algorithm import get_display
windll.shcore.SetProcessDpiAwareness(1)

model = Sequential()


def train_locally():


    main_dir = "dataset/Data"

    # SETTING TRAIN AND TEST DIRECTORY
    train_dir = os.path.join(main_dir, "train")
    test_dir = os.path.join(main_dir, "test")

    # SETTING DIRECTORY FOR fungal AND NORMAL IMAGES DIRECTORY
    train_first_dir = os.path.join(train_dir, "fugal")
    train_second_dir = os.path.join(train_dir, "healthy")

    test_first_dir = os.path.join(test_dir, "fugal")
    test_second_dir = os.path.join(test_dir, "healthy")

    # MAKING SEPARATE FILES :
    train_first_names = os.listdir(train_first_dir)
    train_second_names = os.listdir(train_second_dir)

    test_first_names = os.listdir(test_first_dir)
    test_second_names = os.listdir(test_second_dir)

    """# 3) PERFORMING DATA VISUALIZATION """

    rows = 4
    columns = 4

    fig = plt.gcf()
    fig.set_size_inches(12, 12)
    fig.canvas.manager.set_window_title('Data Visualization')

    fungal_img = [os.path.join(train_first_dir, filename) for filename in train_first_names[0:8]]
    normal_img = [os.path.join(train_second_dir, filename) for filename in train_second_names[0:8]]

    print(fungal_img)
    print(normal_img)

    merged_img = fungal_img + normal_img



    """# 4) DATA PREPROCESSING AND AUGMENTATION"""

    messagebox.showinfo('info','Press OK if you are sure you want to train a new model''\n'' with the selected Epochs.')
    # CREATING TRAINING, TESTING AND VALIDATION BATCHES
    dgen_train = ImageDataGenerator(rescale=1. / 255, validation_split=0.2, zoom_range=0.2, horizontal_flip=True)

    dgen_validation = ImageDataGenerator(rescale=1. / 255, )

    dgen_test = ImageDataGenerator(rescale=1. / 255, )

    train_generator = dgen_train.flow_from_directory(train_dir, target_size=(80, 80), subset='training',
                                                     batch_size=32,
                                                     class_mode='binary')
    validation_generator = dgen_train.flow_from_directory(train_dir, target_size=(80, 80), subset="validation",
                                                          batch_size=32,
                                                          class_mode="binary")
    test_generator = dgen_test.flow_from_directory(test_dir, target_size=(80, 80), batch_size=32, class_mode="binary")

    print("Class Labels are: ", train_generator.class_indices)
    print("Image shape is : ", train_generator.image_shape)

    """# 5) BUILDING CONVOLUTIONAL NEURAL NETWORK MODEL"""

    # 1) CONVOLUTIONAL LAYER - 1
    model.add(Conv2D(32, (5, 5), padding="same", activation="relu", input_shape=train_generator.image_shape))

    # 2) POOLING LAYER - 1
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 3) DROPOUT LAYER -2
    model.add(Dropout(0.5))

    # 4) CONVOLUTIONAL LAYER - 2
    model.add(Conv2D(64, (5, 5), padding="same", activation="relu"))



    # 7) FLATTENING LAYER TO 2D SHAPE
    model.add(Flatten())

    # 8) ADDING A DENSE LAYER
    model.add(Dense(256, activation='relu'))

    # 9 DROPOUT LAYER - 3
    model.add(Dropout(0.5))

    # 10) FINAL OUTPUT LAYER
    model.add(Dense(1, activation='sigmoid'))

    # PRINTING MODEL SUMMARY
    model.summary()

    """# 6) COMPILING AND TRAINING THE NEURAL NETWORK MODEL"""
    # COMPILING THE MODEL
    model.compile(Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # TRAINING THE MODEL
    history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

    messagebox.showinfo("info",'Model training finished successfully ''\n' ' Press OK to continue.')

    """# 7) PERFORMING EVALUATION """
    # KEYS OF HISTORY OBJECT
    history.history.keys()

    # PLOT GRAPH BETWEEN TRAINING AND VALIDATION LOSS
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training', 'Validation'])
    plt.title("Training and validation losses")
    plt.xlabel('epoch')
    plt.show()

    # PLOT GRAPH BETWEEN TRAINING AND VALIDATION ACCURACY
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['Training', 'Validation'])
    plt.title("Training and validation accuracy")
    plt.xlabel('epoch')
    plt.show()

    # GETTING TEST ACCURACY AND LOSS
    print('\n---------------------------------------------------')
    print('Getting Test accuracy & loss')
    test_loss, test_acc = model.evaluate(test_generator)
    print("Test Set Loss : ", test_loss)
    print("Test Set Accuracy : ", test_acc)

    model.save("model.h5")


"""# 9) PREDICTION ON NEW DATA (UPLOAD FILES)"""


def upload_image():
    messagebox.showinfo('info','Pleas select an image to detect the fungal leaf spot')
    model = load_model('model.h5')
    covid_results = open('Results.csv', 'a')
    #print('Checking the presence of COVID-19 in selected image using the trained model...')
    img_path = filedialog.askopenfilename(initialdir="/",
                                          title="Select an Image",
                                          filetypes=(("Image files", "*.jpg*"),
                                                     ("all files",

                                                    "*.*")))

    img_show = Image.open(img_path)
    img_show.show()
    img = image.load_img(img_path, target_size=(80, 80))
    images = image.img_to_array(img)
    images = np.expand_dims(images, axis=0)
    prediction = model.predict(images)
    if prediction == 0:
        import csv
        with open('Results.csv', 'a') as fd:
            csv_writer = csv.writer(fd)
            csv_writer.writerow([" 1" ])
        messagebox.showinfo('info', "النبات مصاب بمرض بقع الاوراق الفطرية والعلاج المبكر هو استخدام المبيدات التي تحوي على الكلوروثالونيل أو البوسيكال على هيئة رش ورقي ")
    else:
        import csv
        with open('Results.csv', 'a') as fd:
            csv_writer = csv.writer(fd)
            csv_writer.writerow([" 0" ])
            messagebox.showinfo('info', "التقرير يشير الى أن النبات سليم ربما البقع تمثل آثار شمسية ")


    covid_results.close()
    messagebox.showinfo('Success', 'Image categorized and report created successfully.''\n''You can check it in the ')


def model_structure():
    def traverse_datasets(hdf_file):
        def h5py_dataset_iterator(g, prefix=''):
            for key in g.keys():
                item = g[key]
                the_path = '{}/{}'.format(prefix, key)
                if isinstance(item, h5py.Dataset):
                    yield the_path, item
                elif isinstance(item, h5py.Group):
                    yield from h5py_dataset_iterator(item, the_path)

        structure_file = open('Model_Structure.txt', 'w')

        with h5py.File(hdf_file, 'r') as f:
            for (path, dset) in h5py_dataset_iterator(f):
                structure_file.write(path + ' ' + str(dset) +  '\n')

        messagebox.showinfo('Success', 'Model structure report created successfully.')
        structure_file.close()
        return None

    traverse_datasets('model.h5')

# User Interface
def swap():
    f2.pack()
    f1.pack_forget()


def swap1():
    f2.pack_forget()
    f1.pack()


def swap2():
    f3.pack()
    f1.pack_forget()


def swap3():
    f3.pack_forget()
    f1.pack()


root = Tk()
root.title('PLANET-CARE')
root.config(bg='white')
root.geometry('550x650')  # زيادة الارتفاع لاستيعاب التصميم الجديد

# تنسيقات الخطوط والألوان الجديدة
title_font = ('Arial', 22, 'bold')
button_font = ('Arial', 18, 'bold')
content_font = ('Arial', 14)
button_style = {
    'relief': RAISED,
    'bd': 3,
    'padx': 10,
    'pady': 5
}

# إطارات الواجهة
f1 = Frame(root, bg="white", borderwidth=10)
f2 = Frame(root, bg="white", borderwidth=10)
f3 = Frame(root, bg="white", borderwidth=10)

# عنوان التطبيق
label1 = ttk.Label(f1, text=" دعم الزراعة السورية",
                   font=title_font,
                   background="white",
                   foreground="green")
label1.pack(pady=20)

# صورة التطبيق
height = 300
width = 450
cover_image = PhotoImage(file='AI-Brain.PNG').zoom(30).subsample(15) if os.path.exists('AI-Brain.PNG') else PhotoImage()
canvas = Canvas(f1, width=width + 150, bg='white', highlightthickness=0)
canvas.create_image(width / 2, height / 2, image=cover_image)
canvas.pack(pady=20)

# أزرار الإطار الرئيسي (f1)
b1 = Button(f1, text="درب النموذج",
            bg="#FFD700", fg="black",  # أصفر
            font=button_font,
            command=train_locally,
            **button_style)
b1.pack(pady=10, fill=X, padx=30)

b2 = Button(f1, text="اختبار صورة",
            bg="#32CD32", fg="black",  # أخضر
            font=button_font,
            command=upload_image,
            **button_style)
b2.pack(pady=10, fill=X, padx=30)

b3 = Button(f1, text="لمحة عن التطبيق",
            bg="#1E90FF", fg="black",  # أزرق
            font=button_font,
            command=swap2,
            **button_style)
b3.pack(pady=10, fill=X, padx=30)

b4 = Button(f1, text="اتصل بالفريق التقني",
            bg="#FFD700", fg="black",  # أصفر
            font=button_font,
            command=swap,
            **button_style)
b4.pack(pady=10, fill=X, padx=30)

b5 = Button(f1, text="إغلاق التطبيق",
            bg="#FF6347", fg="white",  # أحمر (للتمييز)
            font=button_font,
            command=root.destroy,
            **button_style)
b5.pack(pady=10, fill=X, padx=30)

# محتوى الإطار الثاني (f2) - اتصل بالفريق
text1 = ("Eng. lama khaddour"
         " \nlamakhaddour10@gmail.com ")
r_t = arabic_reshaper.reshape(text1)
t11 = get_display(r_t)
ttk.Label(f2, text=t11, font=content_font,
          background="white", foreground="green",
          justify=CENTER).pack(pady=20, padx=20)

b6 = Button(f2, text="رجوع",
            bg="#1E90FF", fg="black",  # أزرق
            font=button_font,
            command=swap1,
            **button_style)
b6.pack(pady=20, fill=X, padx=30)

# محتوى الإطار الثالث (f3) - لمحة عن التطبيق
app_info = (" لمحة عن ميزات التطبيق  \n"
            "1- التطبيق قادر على تمييز 8 من أخطر الأمراض التي تصيب أوراق النبات \n"
            "2- التطبيق قادر على كشف البقع من خلال تصوير الكاميرا العادية أو كاميرا الموبايل\n"
            "3- دقة التصنيف تصل الى تسع وتسعون بالمئة\n"
            "4- تدريب النموذج يتم مرة وحدة ثم يصبح قابل للاستخدام\n")
ttk.Label(f3, text=app_info, font=content_font,
          background="white", foreground="green",
          justify=CENTER).pack(pady=20, padx=20)

b7 = Button(f3, text="رجوع",
            bg="#1E90FF", fg="black",  # أزرق
            font=button_font,
            command=swap3,
            **button_style)
b7.pack(pady=20, fill=X, padx=30)

# بدء التطبيق
f1.pack()
root.mainloop()
