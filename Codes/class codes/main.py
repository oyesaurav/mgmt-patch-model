from data_preparation.data_sorting import DataPreparation
from data_preparation.image_generation import GenerateImages
from train_model.training import Trainer

patients = 150
modalities = ['T1']
classifications = ['MGMT_positive', 'MGMT_negative']
block_size = (64,64)
stride = 2
inter_dim = (64,64)
min_size = (30,30)
dataset_path = "D:/MGMT research project/NIfTI-files/"
main_dir_path = "D:/mgmt-patch-model/"

DataPreparation_obj = DataPreparation(patients, dataset_path, main_dir_path)
GenerateImages_obj = GenerateImages(modalities,classifications, block_size, stride, inter_dim, dataset_path, main_dir_path)
Trainer_obj = Trainer(modalities, classifications, main_dir_path, min_size, inter_dim)

DataPreparation_obj.prepare_data()
GenerateImages_obj.main()
Trainer_obj.main()