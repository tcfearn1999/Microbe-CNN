# Microbial Community Modeling with Convolutional Neural Networks

This repository contains open-source code presented in a poster at the Mycological Society of America Annual Meeting 2024. You can find the poster [here](https://www.canva.com/design/DAGGJtZGAv0/yRN20vUckEQqaPc46VEvJQ/edit?utm_content=DAGGJtZGAv0&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton). To run the program, follow these steps:

1. **Prerequisites**:
   - Python 3.10 or later
   - R 4.0 or later
   - Linux or Unix OS

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Modify File Paths**:
   Edit the file paths in `~/data_gen/build_communities_stochasicity_antagonism.R` under the "Compose File Names" section of the code. Then execute the following:
   ```
   rscript ~/data_gen/build_communities_stochasicity_antagonism.R
   ```

4. **Download pyDeepInsight**:
   ```
   pip install git+https://github.com/alok-ai-lab/pyDeepInsight.git#egg=pyDeepInsight
   ```

5. **Generate PCA Analysis with DeepInsight**:
   Change the file paths to the appropriate files and run the following code:
   ```
   python ~/preprocessing/DeepInsight_Image_gen.py
   ```
   *Note: Run this process separately for both antagonism and stochasticity files, updating the file paths each time.*

6. **Train and Test Neural Network**:
   Update the file paths and parameters in the appropriate files (`ResNet` or `CNN`). Execute the following code:
   ```
   python ~/CNN/CNN_Resnet.py
   ```
   or
   ```
   python ~/CNN/CNN_VGG.py
   ```