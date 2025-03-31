# ICU Admission Prediction for COVID-19 Patients using Chest X-ray Imaging

## Abstract
This project focuses on predicting intensive care unit (ICU) admissions for COVID-19 patients using chest X-ray imaging, particularly in scenarios where large datasets may not be available. We employed deep learning for feature extraction and explored transfer learning techniques, comparing ResNet50 models pre-trained on natural image datasets and chest X-ray images. Additionally, we implemented an innovative dataset expansion approach, combining a primary dataset with another containing clinically relevant but different labels through label fusion. Our best-performing model, **TorchX-SBU-RSNA**, achieved an AUC of 0.756, showcasing the value of medical image-specific pre-training and strategic dataset expansion.

Grad-CAM analysis demonstrated that models pre-trained on chest X-ray images focus more accurately on the lung regions, enhancing the prediction of ICU needs. Our approach underscores the potential of using limited but diverse data sources to improve model development and offers a rapid, deployable solution for patient management during healthcare crises.

## Usage
To use this project, you will need to:

1. **Clone the repository**
    ```bash
    git clone https://github.com/yunchibellalin/COVID-ICU.git
    ```

2. **Install the necessary packages using the `requirements.txt` file**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the public datasets**
    - **COVID-19-NY-SBU Dataset**: Available from The Cancer Imaging Archive [here](https://www.cancerimagingarchive.net/collection/covid-19-ny-sbu/). This dataset contains chest X-ray images from COVID-19 patients and is used as the primary dataset for training the model.
    
    - **MIDRC-RICORD-1c Dataset**: Also available from The Cancer Imaging Archive [here](https://www.cancerimagingarchive.net/collection/midrc-ricord-1c/). This dataset serves as the secondary dataset in this study, featuring images with clinically relevant but different labels, which we incorporate through label fusion to enhance model training.

    Both datasets are publicly available and can be freely accessed for research purposes.

4. **Prepare your dataset**
   - **X-ray Imaging**: 
   For lung segmentation, we used the model developed by Illia Ovcharenko [available here](https://github.com/IlliaOvcharenko/lung-segmentation/tree/master?tab=readme-ov-file). After segmentation, each scan was saved with dimensions of 512x512x2, where the first channel is the original X-ray image and the second is the binary lung mask.

   - **Label Fusion**: 
    **COVID-19 NY-SBU Dataset**: We extracted *ICU admission label* [is_icu] identifying patients who were admited to ICU.
    **MIDRC-RICORD-1c Dataset**: *Airspace Disease Grading* was taken from this dataset contains four categories (negative, mild, moderate, and severe) for pneumonia severity. For label fusion, we grouped patients with negative, mild, or moderate disease into the *pseudo-non-ICU* group, and those with severe pneumonia into the *pseudo-ICU* group.

5. **Run the Preprocessing & Training Pipeline**
You can run the `covid_icu.py` script with the following parameters:
    - **sharpen** - Apply image sharpening.  *Options*: `'True'`, `'False'`
    - **histeq** - Apply histogram equalization with CLAHE. *Options*: `'False'`, `'True_{cl}_{gs}'`  
    for example in `'True_.04_8'`:  
        `.04` is the clip limit (cl) multiplied by the grid size (gs*gs)
        `8` is the grid size (gs) for tileGridSize in CLAHE: `cv2.createCLAHE(clipLimit=cl*gs*gs, tileGridSize=(gs, gs))`
    - **pretrained** - Select the pre-trained model to use. *Options*: `'TorchX'`, `'ImageNet'`
    - **opt** - Choose the optimization algorithm. *Options*: `'Adam'`, `'SGD'`
    - **lr** - Set the initial learning rate  
    - **bs** - Define the batch size  
    - **ep** - Set the number of training epochs  
    - **historyDir** - Specify the directory to save the training history and results
    - **jobName** - Provide a name to identify the specific training job  

    ```bash
    python covid_icu.py $sharpen $histeq $pretrained $opt $lr $bs $ep $historyDir $jobName
    ```

    Alternatively, you can run the provided `run.sh` script:

    ```bash
    bash run.sh
    ```
    
6. **To visualize Grad-CAM outputs for model interpretability**
    - Install the Grad-CAM module from [DeepSurv-CNN](https://github.com/deepsurv-cnn/main/tree/main):
    - To visualize Grad-CAM outputs, refer to the example provided in `example_gradcam.ipynb`

    For more details on Grad-CAM, you can also check out the original paper and codebase: [Grad-CAM GitHub](https://github.com/ramprs/grad-cam/tree/master?tab=readme-ov-file).


## Tools and Libraries

The following third-party tools and models were instrumental in developing this project:

- **Lung Segmentation Model**: [Illia Ovcharenko's Lung Segmentation](https://github.com/IlliaOvcharenko/lung-segmentation/tree/master)
- **TorchXRayVision**: Chest X-ray pre-trained model used for transfer learning. Available [here](https://github.com/mlmed/torchxrayvision/tree/master).
- **Grad-CAM Module**: Used for model interpretability and visualizing which regions of the X-rays contribute to the model’s decisions. Original Grad-CAM paper and implementation [here](https://github.com/ramprs/grad-cam/tree/master).

## Citation


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>



    ARCHIVED

    If you use this project or find it helpful in your research, please cite the following tools:
    - Ovcharenko, I., et al. Lung Segmentation Model. GitHub, [https://github.com/IlliaOvcharenko/lung-segmentation/tree/master](https://github.com/IlliaOvcharenko/lung-segmentation/tree/master).
    - TorchXRayVision: Pre-trained X-ray Models. GitHub, [https://github.com/mlmed/torchxrayvision/tree/master](https://github.com/mlmed/torchxrayvision/tree/master).
    - Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. GitHub, [https://github.com/ramprs/grad-cam/tree/master](https://github.com/ramprs/grad-cam/tree/master).


    [![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

    [![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)



    > formatting syntax is to make it as readable
    > as possible. The idea is that a




    ## Tech

    Dillinger uses a number of open source projects to work properly:

    - [AngularJS] - HTML enhanced for web apps!
    - [Ace Editor] - awesome web-based text editor
    - [markdown-it] - Markdown parser done right. Fast and easy to extend.
    - [Twitter Bootstrap] - great UI boilerplate for modern web apps
    - [node.js] - evented I/O for the backend
    - [Express] - fast node.js network app framework [@tjholowaychuk]
    - [Gulp] - the streaming build system
    - [Breakdance](https://breakdance.github.io/breakdance/) - HTML
    to Markdown converter
    - [jQuery] - duh

    And of course Dillinger itself is open source with a [public repository][dill]
    on GitHub.


    ## Plugins

    | Plugin | README |
    | ------ | ------ |
    | Dropbox | [plugins/dropbox/README.md][PlDb] |
    | GitHub | [plugins/github/README.md][PlGh] |
    | Google Drive | [plugins/googledrive/README.md][PlGd] |
    | OneDrive | [plugins/onedrive/README.md][PlOd] |
    | Medium | [plugins/medium/README.md][PlMe] |
    | Google Analytics | [plugins/googleanalytics/README.md][PlGa] |