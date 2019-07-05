import csv
from shutil import copy2
import os

cwd = os.getcwd()
train_path = cwd + "/train"
num = 0

# 開啟 CSV 檔案
with open('train.csv', newline='') as csvfile:

  # 讀取 CSV 檔案內容
  rows = csv.reader(csvfile)

  # 以迴圈輸出每一列
  for row in rows:
      if num >0:
          folderpath = train_path + "/" + str(row[1])
          if not os.path.isdir(folderpath):
              os.makedirs(folderpath)
          copy2(cwd + "/train_images/" + row[0], folderpath)
      num +=1
      
