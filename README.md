# Face_recognition
```text
pip install -r requirements.txt
conda create -n arcface python=3.8.18
```

## FID(Fréchet Inception Distance) 
計算生成圖片與真實圖片在特定空間內分布的相似程度，FID分數越低，表示生成圖像與真實圖像的分布越接近
```text
pip install pytorch-fid
python -m pytorch_fid "path/to/dataset1" "path/to/dataset2"
```

## Arcface
mean_arcface_csim.py # 計算兩個folder的人臉相似程度  
arcface_1forall.py # 計算一個folder內圖片間的相似度  