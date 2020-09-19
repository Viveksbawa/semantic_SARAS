from datasets import cityscape




dataset_train= cityscape.CitySegmentation(root= '/mnt/mars-fast/segment_datasets/cityscape',
                    split= 'train')


print(len(dataset_train))


