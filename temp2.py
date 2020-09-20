from datasets import cityscape

dataset_train = cityscape.CitySegmentation(root='/mnt/mars-fast/segment_datasets/cityscape', 
                    split='train', mode='train') 

dataset_val = cityscape.CitySegmentation(root='/mnt/mars-fast/segment_datasets/cityscape', 
                split='test', mode='val')

print(len(dataset_val))

img, lab = dataset_val[10]
print(img, lab)
