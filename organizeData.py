from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pandas as pd
from torchvision.io import read_image

LABELS = {
  'akiec': 0,
  'bcc': 1 ,
  'bkl': 2,
  'df': 3,
  'mel': 4,
  'nv': 5,
  'vasc': 6
}

def readData():
  df = pd.read_csv('./data/HAM10000_metadata.csv')
  y = df['dx']
  x = df['image_id']

  test_size = 0.30
  train, test = train_test_split(df, test_size = test_size)

  validation_proportion= 0.30
  train, validation = train_test_split(train, test_size= validation_proportion)
  loadData(train)

def loadData(img_labels):
    img_dir = '../data/HAM10000_images_part_1/'
    idx = 1

    img_path = os.path.join(img_dir, img_labels.iloc[idx]["image_id"]) + '.jpg' #get images to path
    image = read_image(img_path) # converts image to tensor
    print(image)

readData()