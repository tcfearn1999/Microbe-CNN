# Microbial Community Modeling with Convolutional Neural Networks

Opensource code for a poster presented at the mycological society of America annual meeting 2024. The poster for this can be found [HERE](https://www.canva.com/design/DAGGJtZGAv0/yRN20vUckEQqaPc46VEvJQ/edit?utm_content=DAGGJtZGAv0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).To run the program do the following:

1. Prequistes of:
*  Python 3.10> 
*  R 4.0>
*  Linux or Unix OS

2. Install dependecies with 

`pip install requirements.txt`

3. Modify the file paths in ~/data_gen/build_communities_stochasicity_antagonism.R under compose file names section of code. Then runthe following

`rscript ~/data_gen/build_communities_stochasicity_antagonism.R`

4. Download pyDeepInsight

`pip install git+https://github.com/alok-ai-lab/pyDeepInsight.git#egg=pyDeepInsight`

5. Generate PCA analysis with DeepInsight. Change the file paths to the appropraite files. Run the following code:

`python ~/preprocessing/DeepInsight_Image_gen.py`

NOTE:must run this for both antagonism and stochasticity files seperately, changing the file pathes each time

6. Train and Test Neural network. Change the file paths and parameters in the approriate files(ResNet or CNN). Run the follwoing code:

`python ~/CNN/CNN_Resnet.py`

OR

`~/CNN/CNN_VGG.py'  
