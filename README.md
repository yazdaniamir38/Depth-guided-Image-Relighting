# Depth-guided-Image-Relighting
A deep Learning model for one-to-one and any-to-any relighting. Our work ranked 2nd in NTIRE 2021 one-to-one depth guided relighting and 5th in any-to-any relighting challenge held in conjuction with CVPR 2021. You can find the challenge results and the coresponding paper to our work here:  
NTIRE 2021 Depth Guided Image Relighting Challenge. Helou et al. [pdf](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Helou_NTIRE_2021_Depth_Guided_Image_Relighting_Challenge_CVPRW_2021_paper.pdf)  
Physically Inspired Dense Fusion Networks for Relighting. Yazdani et al. [pdf](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/papers/Yazdani_Physically_Inspired_Dense_Fusion_Networks_for_Relighting_CVPRW_2021_paper.pdf)
# Citations
Please cite this paper in your publications if it is helpful for your tasks:    

Bibtex:
```
@inproceedings{yang2021S3Net,
    title     = {{S3N}et: A Single Stream Structure for Depth Guided Image Relighting},
    author    = {Yang, Hao-Hsiang and Chen, Wei-Ting and Kuo, Sy-Yen},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
    year      = {2021}
}
```
# Requirements
To correctly train and test the models you need the following packages and libraries:
1. python 3.6.11 or higher.
2. PyTorch 1.6.0
3. OpenCV
4. Kornia
5. Pillow
6. scipy
# Testing (Evaluation)
# One-to-one:
First download the checkpoint from [here](https://drive.google.com/file/d/1-azD3U8c4ag24ecVagB74K7BkMBpKh4s/view?usp=sharing) and save it in 'One_to_one/model/'. Make sure the test set is stored in 'One_to_one/test_set/'. Also generate the normals for the test set using 'One_to_one/dataset_generation/create_normal_mat_file.py' and store them in 'One_to_one/dataset_generation/pre_reqs/'.  
Navigate to the One_to_one folder, in terminal, type:  
python test.py --cuda --model dense121  
where --cuda option is for using GPU if available.
# Any-to-any:
First download the checkpoints from [here](https://drive.google.com/file/d/1FxafveD9QMXFmEvPw3MSMW7xufteN-Jr/view?usp=sharing) (You need to download two sets of files:1) Files in 'model/' and 2)'checkpoint_epochcorrected_100.pth') and save the first ones in 'Any_to_any/model/' and the second one in 'Any_to_any/'.
Navigate to the Any_to_any folder, in terminal, type:  
python test.py --cuda --model dense121  
where --cuda option is for using GPU if available.

# Training
# One-to-one (OIDDR-Net):
# 1)Data Generation:
First download and save training data for one-to-one relighting from [here](https://github.com/majedelhelou/VIDIT) into 'One_to_one/'. Using 'One_to_one/dataset_generation/savemat_depth.py' generate '.mat' files for depth inputs.  
Second you need to generate psuedo ground truth for albedo and shading of the training data. You can use the GoogleColab notebook (psuedo_GT.ipynb. Refer [here](https://github.com/tnarihi/direct-intrinsics) for the 'direct intrinsic decomposition model'). Store the albedo and shading files in 'One_to_one/content/'. (The name format for albedo and shading should be :'a_original_name.png' and 's_original_name.png' repectively. 'original_name' means the original name of the image in training set.)  
Now you can use  'One_to_one/dataset_generation/createPatches.m' to generate patches. 
Subsequently use 'One_to_one/dataset_generation/createH5.py' to generate h5 files.  
Finally, use 'One_to_one/dataset_generation/modify.py' to generate h5 files for normals.

# 2) Training the network 
After generating the '.h5' files, you can start training the network. The encoder in OIDDR-Net is adpoted from densenet feature extraction layers. Furthermore, the lighting estimation network is a pretrained dense-net 121 classification network trained separately on the training inputs and their lighting parameters.  
Download the checkpoint for lighting estimation network from [here](https://drive.google.com/file/d/1FxafveD9QMXFmEvPw3MSMW7xufteN-Jr/view?usp=sharing) ('checkpoint_epochcorrected_100.pth') into 'One_to_one/
To train a model, run main.py. There are parse arguments that you can set. The code tests the model at the end of each epoch on the validation set (You need to download the validation data from [here](https://github.com/majedelhelou/VIDIT) into 'One_to_one/validation/' and 'One_to_one/validation_gt/'. Make sure to generatre normals for validation data and store them in the former folder as well.)  and if the perfromance has been imporoved it saves the checkpoint otherwise it saves the model every 5 epochs.
# Any-to-any (AMIDR-Net):
# 1) Training the lighting-estimation network
TBU
# 2) Training the main model
TBU

The factsheets for our proposed methods in NTIRE 2021 depth guided image relighting challenge as well as the training and testing code can be found here:

https://drive.google.com/drive/folders/18dnrzTJ9zJoo_4-v_BXN0K6Nkhs97j_l
