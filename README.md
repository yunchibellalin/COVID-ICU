# Classification of the ICU Admission for COVID-19 Patients with Transfer Learning Models Using Chest X-Ray Images

## Abstract
**Objectives**: Predicting intensive care unit (ICU) admissions during pandemic outbreaks such as COVID-19 can assist clinicians in early intervention and the better allocation of medical resources. Artificial intelligence (AI) tools are promising for this task, but their development can be hindered by the limited availability of training data. This study aims to explore model development strategies in data-limited scenarios, specifically in detecting the need for ICU admission using chest X-rays of COVID-19 patients by leveraging transfer learning and data extension to improve model performance. **Methods**: We explored convolutional neural networks (CNNs) pre-trained on either natural images or chest X-rays, fine-tuning them on a relatively limited dataset (COVID-19-NY-SBU, n = 899) of lung-segmented X-ray images for ICU admission classification. To further address data scarcity, we introduced a dataset extension strategy that integrates an additional dataset (MIDRC-RICORD-1c, n = 417) with different but clinically relevant labels. **Results**: The TorchX-SBU-RSNA and ELIXR-SBU-RSNA models, leveraging X-ray-pre-trained models with our training data extension approach, enhanced ICU admission classification performance from a baseline AUC of 0.66 (56% sensitivity and 68% specificity) to AUCs of 0.77–0.78 (58–62% sensitivity and 78–80% specificity). The gradient-weighted class activation mapping (Grad-CAM) analysis demonstrated that the TorchX-SBU-RSNA model focused more precisely on the relevant lung regions and reduced the distractions from non-relevant areas compared to the natural image-pre-trained model without data expansion. **Conclusions**: This study demonstrates the benefits of medical image-specific pre-training and strategic dataset expansion in enhancing the model performance of imaging AI models. Moreover, this approach demonstrates the potential of using diverse but limited data sources to alleviate the limitations of model development for medical imaging AI. The developed AI models and training strategies may facilitate more effective and efficient patient management and resource allocation in future outbreaks of infectious respiratory diseases.

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
   For lung segmentation, we used the model developed by Illia Ovcharenko [available here](https://github.com/IlliaOvcharenko/lung-segmentation/tree/master?tab=readme-ov-file). After segmentation, each scan was saved with dimensions of 224x224x2, where the first channel is the original X-ray image and the second is the binary lung mask.

   - **Label Fusion**: 
    **COVID-19 NY-SBU Dataset**: We extracted *ICU admission label* [is_icu] identifying patients who were admited to ICU.
    **MIDRC-RICORD-1c Dataset**: *Airspace Disease Grading* was taken from this dataset contains four categories (negative, mild, moderate, and severe) for pneumonia severity. For label fusion, we grouped patients with negative, mild, or moderate disease into the *non-ICU* group, and those with severe pneumonia into the *ICU* group.

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

    Our model taining weights are available on [Box](https://uab.box.com/s/o3e8z6h5vgjjj8xkgnvwt8lq1n963wk0) 
    
6. **To visualize Grad-CAM outputs for model interpretability**
    - Install the Grad-CAM module from [DeepSurv-CNN](https://github.com/deepsurv-cnn/main/tree/main):
    - To visualize Grad-CAM outputs, refer to the example provided in `example_gradcam.ipynb`

    For more details on Grad-CAM, you can also check out the original paper and codebase: [Grad-CAM GitHub](https://github.com/ramprs/grad-cam/tree/master?tab=readme-ov-file).

7. **ELIXR implementation**
    - To prepare for image embeddings, please refer to ELIXR example code `quick_start_with_hugging_face.ipynb` [Available on GitHub](https://github.com/Google-Health/cxr-foundation/tree/master/notebooks) with trained image encoder [Available on Hugging Face](https://huggingface.co/google/cxr-foundation)
    - To train a classifier with embeddings, please refer to example code `train_data_efficient_classifier.ipynb`[Available on GitHub](https://github.com/Google-Health/cxr-foundation/tree/master/notebooks)

    For more details on ELIXR model, you can also check out the original paper and document: [paper](
https://doi.org/10.48550/arXiv.2308.01317) and [doc](https://developers.google.com/health-ai-developer-foundations/cxr-foundation/get-started).


## Tools and Libraries

The following third-party tools and models were instrumental in developing this project:

- **Lung Segmentation Model**: [Illia Ovcharenko's Lung Segmentation](https://github.com/IlliaOvcharenko/lung-segmentation/tree/master)
- **TorchXRayVision**: Chest X-ray pre-trained model used for transfer learning. Available [here](https://github.com/mlmed/torchxrayvision/tree/master).
- **Grad-CAM Module**: Used for model interpretability and visualizing which regions of the X-rays contribute to the model’s decisions. Original Grad-CAM paper and implementation [here](https://github.com/ramprs/grad-cam/tree/master).
- **ELIXR**: Chest X-ray pre-trained image encoder. Available [here](https://huggingface.co/google/cxr-foundation).

## Citation

If you use this project or find it helpful in your research, please cite:

    Lin, Y. C., & Fang, Y. H. D. (2025). Classification of the ICU Admission for COVID-19 Patients with Transfer Learning Models Using Chest X-Ray Images. Diagnostics, 15(7), 845.