# Kvasir-seg-binary-segmentation

## Page 1

# Experimental Analysis on Kvasir-SEG Dataset with Deep Learning

Student ID 30  
Student ID 25  
Student ID 4  (Hansim) Resolution experiment
Student ID 36

Bachelor in Data Science, Kristiania University College, Oslo, Norway  
PGR207 Deep Learning - Polyp Segmentation

**Abstract**—This study investigates polyp segmentation on the Kvasir-SEG dataset (1,000 colonoscopy images, predefined 880/120 train-validation split) using DeepLabV3+ trained with a composite binary cross-entropy-with-logits and soft Dice loss to jointly optimize pixel-wise accuracy and region overlap. We systematically analyze how four design factors affect segmentation performance: (1) input resolution, (2) training data size, (3) data augmentation strategy, and (4) optimizer-learning-rate configuration, under a fixed data split, data-loading pipeline, early stopping, and learning-rate scheduler. Experiments show that a resolution of $352 \times 352$ maintains validation performance across nearby scales while reducing epoch time by approximately 25% compared with $416 \times 416$. Light geometric augmentations combined with grid distortion provide the largest Dice improvement among the tested augmentation schemes. Varying data size indicates robustness down to 50% of the training set, peak recall at 75%, and pronounced degradation below 25%. Optimizer behaviour depends strongly on learning rate: at 0.01, momentum-based NAG and SGD achieve the highest Dice scores (0.868 and 0.859), whereas at 0.001, adaptive methods perform best, with AdaGrad reaching a Dice of 0.861. Overall, the results suggest that a practical recipe of $352 \times 352$ resolution, light/grid augmentation, and a momentum-based optimizer with learning rate 0.01 offers a strong trade-off between accuracy, stability, and computational cost for polyp segmentation on Kvasir-SEG.

## I. INTRODUCTION

Image segmentation refers to the partitioning of a digital image into multiple meaningful regions based on pixel characteristics such as color and intensity. The primary objective is to simplify or alter the representation of an image to facilitate specific tasks. This technique is extensively utilized in computer vision, a subfield of artificial intelligence (AI) that enables computers to interpret and analyze visual inputs from sources such as images and video. Image segmentation operates at the pixel level. This report focuses on the subcategory of binary segmentation, with particular emphasis on polyp segmentation. [11], [12]

Segmentation is one of the key techniques applied in modern medicine, particularly valuable during diagnosis and early detection. The goal of polyp segmentation is to autonomously detect and segment polyp regions, serving as a procedure in colorectal cancer (CRC) screening. The primary aim of this technology is not to replace medical professionals, but rather to assist and enhance the efficiency of clinical tasks. [4], [13], [14]

Early detection of disease is critical, as prompt diagnosis enables timely intervention and can significantly improve patient outcomes. Recent research indicates that, despite current standards in colorectal cancer screening, a substantial number of precancerous lesions remain undetected. For example, a meta-analysis involving 15,000 colonoscopies performed in 2019 found that approximately 25 to 30 percent of adenomas (polyps that can progress to colorectal cancer if left untreated) were missed in clinical practice. [13]–[15] This gap highlights the need for advanced computer-assisted methods to enhance the visual precision and reduce the rate of overlooked lesions. To address this challenge, we use the Kvasir-SEG dataset, developed by Simula Research Laboratory in collaboration with Vestre Viken Health Trust, Norway. Part of the broader Kvasir dataset series, Kvasir-SEG supports computer-aided analysis in this field. It includes 1,000 gastrointestinal images, each with an expert-annotated segmentation mask highlighting polyp regions. The images are real samples from a Norwegian patient cohort. [10]

## II. METHODOLOGY

### A. Dataset

&lt;img&gt;COLON POLYP SIZES
Diminutive (<5 mm)
Small (6-9 mm)
Large (>10 mm)
Precancerous
Cancerous
MEDICALNEWSTODAY
&lt;/img&gt;

Fig. 1. Polyp sizes

---


## Page 2

The dataset includes 1,000 RGB images with corresponding greyscale annotations, varying in size and pixel dimensions. [10] Although Simula has not published detailed documentation on polyp representation as of this report, it is reasonable to assume that the dataset should contain a broad range of polyp variations in both training and testing images. We also monitored this aspect during model evaluation. The data distribution provides insight into the model’s exposure to the polyp variations and may indicate whether the current split is appropriate or if repartitioning is necessary. Maintaining balance is essential to ensure that the model learns about both common and subtle variations without developing a bias towards more frequently occurring or apparent variations. The objective is to achieve accurate predictions across all possible variants, especially those resembling early or precancerous lesions.

&lt;img&gt;
A grid of five images showing colonoscopy views (top row) and their corresponding masks (bottom row). The images depict various polyps of different shapes and sizes.
&lt;/img&gt;

Fig. 2. Samples

A predefined split is provided through the official Kvasir-SEG GitHub repository, maintained by the dataset’s first author, Debesh Jha. [10] The “Data-split” directory contains two text files (train.txt and val.txt) listing the image names. This predefined split structure facilitates comparison and reproducibility of Kvasir-SEG implementations and experiments. The training file contains 880 names, and the validation file contains 120, resulting in an 88:12 split. The validation set also serves as the testing set during evaluation. The JPG images and their mask counterparts include all 1,000 images, which can be traced by their names, allowing the sets to be split using the text files.

B. Transformations

For all tests, images and masks were resized to 320x320 pixels and subjected to moderate geometric and color augmentation, including shifts, scales, rotations, RGB shifts, and brightness/contrast adjustments. All images were then normalized using standard ImageNet channel statistics to match the pretrained encoder used in DeepLabV3+. Each dimension is normalized using averages of 0.485, 0.456, and 0.406 for each dimension, and standard deviations of 0.229, 0.224, and 0.225, respectively.

C. Loading

Data is loaded through PyTorch’s DataLoader with the shuffling parameter switched on. This function shuffles the ordering of the training data, making it harder for the model to memorize its training data and forcing it to generalize better. Batching and parallel looping are some core features available in this tool that greatly speed up training.

D. Checkpointing & earlystopping

The last things implemented in the training loop were an early stopper and a learning rate scheduler. The early stopper is one of the tools employed in machine learning to fight overfitting or underfitting, depending on the scenario. Overfitting is a common problem in deep learning and is a byproduct of our model memorizing the training data, leading to poor generalization on new, unseen data. The early stopper monitors the progress of both training and validation runs within a set of improvement thresholds, to which it can stop the process if the expected convergence is not achieved. [2] Because model training can change a lot over time, using a fixed learning rate is usually not the best choice. Adjusting the learning rate as training goes on helps the model find better solutions quicker. add set parameters.

E. Loss Functions and Optimizer

To ensure comparability across different experimental factors (image resolution, augmentation, training data size, and optimization), we kept the optimizer and the base loss function fixed unless directly tested Adam served as the baseline optimizer due to its stable convergence for medical image segmentation models.

The loss function combined Binary Cross-Entropy with Logits (PyTorch nn.BCEWithLogitsLoss) and a custom soft Dice loss.

a) Combined loss.

The loss function utilized in all experiments is a weighted sum of a pixel-wise term (Binary Cross-Entropy with Logits) and an overlap term (soft Dice loss):

\[
\mathcal{L} = \lambda_{\text{BCE}} \mathcal{L}_{\text{BCE}}(x, y) + \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}}(x, y),
\tag{1}
\]

with fixed weights $\lambda_{\text{BCE}} = 0.4$ and $\lambda_{\text{Dice}} = 0.7$. The sum of the two weights exceeds 1 because they serve as *relative* importance factors between the two terms, rather than as probabilities.

- **Binary Cross-Entropy with Logits** is utilized which applies the sigmoid non-linearity internally to the model logits $x_i$ and then computes the binary cross-entropy against the target mask $y_i \in \{0, 1\}$. This approach simplifies training by eliminating the need to manually apply sigmoid in the model or training loop.

- **Soft Dice loss** is variant of dice utilized

\[
\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i \hat{y}_i y_i + \varepsilon}{\sum_i \hat{y}_i + \sum_i y_i + \varepsilon},
\tag{2}
\]

where $\hat{y}_i = \sigma(x_i)$ are the sigmoid probabilities, $y_i$ are the target pixels in $[0, 1]$, and $\varepsilon = 10^{-7}$ is a small constant to prevent division by zero.

---


## Page 3

This Dice term helps compensate for class imbalance (large background, small polyp) by directly maximizing region overlap. [5]

**F. Training, Validation, and Evaluation Loops**

Building on the fixed split described earlier (880 training samples and 120 validation samples from `train.txt` and `val.txt`), all experiments use the same image–mask pairing logic: each RGB image is loaded together with its corresponding segmentation mask. Masks are stored as 0/255 images, so they are normalized to the `[0, 1]` range before loss computation. Only the training dataloader is shuffled each epoch in order to reduce bias in gradient updates, while the validation dataloader keeps a fixed order so that validation results remain comparable across runs.

To keep the experimental objective identical across all optimization, augmentation, and resolution studies, we reuse the same composite loss introduced in Section II-E

*   Binary Cross-Entropy with Logits for pixel-wise correctness, and
*   soft Dice loss for region overlap.

Dice is given slightly higher weight because polyp regions are relatively small and we care more about overlap quality than about raw per-pixel accuracy.

**a) Training epoch.**

Each training epoch performs the following steps:

1) Set the model to train mode.
2) Load a mini-batch and run a forward pass to obtain logits.
3) Compute BCEWithLogitsLoss; apply sigmoid to the logits and compute the soft Dice loss.
4) Combine the two terms into the weighted total loss.
5) Backpropagate the loss and perform an optimizer step.

During training we also log additional segmentation metrics (Dice coefficient, IoU score, pixel-wise precision and recall) and average them over the full epoch. This gives a single, consistent place to compare runs without changing the rest of the pipeline.

**b) Validation step.**

After each training epoch, the model is evaluated on the validation split using the *same* loss definition (same BCE+Dice weights) but wrapped in `torch.no_grad()` to disable gradient tracking. The validation Dice coefficient is then used to:

1) drive the ReduceLROnPlateau scheduler (the learning rate is reduced when validation Dice stops improving), and
2) trigger early stopping when the Dice score has not improved for the configured patience.

Whenever the validation Dice reaches a new best value, the model checkpoint is saved to disk. Later experiments can therefore load the same “best” checkpoint, which improves reproducibility and makes cross-experiment comparisons fair.

**G. Learning Rate Scheduler**

After every epoch, the model is evaluated on the validation split and we compute the validation Dice coefficient (i.e. `1 - DiceLoss`). This validation Dice is then used to drive a plateau-based learning rate scheduler. We use PyTorch’s ReduceLROnPlateau:

```python
scheduler =
(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2,
    threshold=1e-3
)
```

We set `mode='max'` because higher Dice scores are better. If the validation Dice does not improve for two consecutive epochs, the scheduler reduces the current learning rate by a factor of 0.5. This makes later epochs more stable and allows the model to converge more smoothly, without changing the rest of the training loop or the optimizer configuration. [2], [7]

**H. Metrics**

We evaluated segmentation quality on the validation set using overlap-based metrics together with basic pixel-wise classification metrics. Ground-truth masks are binary but stored as 0/255 images, so they are first normalized to the `[0, 1]` range. Model predictions are logits; during evaluation we pass them through a sigmoid and threshold at 0.5 to obtain binary masks. When necessary, predicted masks are also resized back to the original mask resolution before metric computation.

**a) Dice coefficient.**

The main metric is the (hard) Dice coefficient, defined as

$$\text{Dice} = \frac{2 |\hat{Y} \cap Y|}{|\hat{Y}| + |Y|}, \quad (3)$$

where $\hat{Y}$ is the predicted binary mask and $Y$ is the ground-truth binary mask. Dice ranges from 0 (no overlap) to 1 (perfect overlap). During training we used a soft Dice term as part of the loss, but for reporting we compute the hard Dice on thresholded predictions. [5], [6]

**b) Accuracy, precision, and recall.**

From the per-pixel counts of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN), we compute:

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}, \quad (4)$$

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad (5)$$

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}. \quad (6)$$

Because polyp pixels are much rarer than background pixels, a high Accuracy can be misleading (the model can predict

---


## Page 4

mostly background and still look good). For this reason, overlap-based scores such as Dice are treated as the primary metric in our experiments.

I. Model Architecture

Before model selection, a comparative trial was conducted between UNet, UNet++ and Deeplabv3+. The evaluation focused on essential metrics such as accuracy and efficiency. Training deep neural networks is time-consuming and resource-intensive. Therefore, an optimal model should demonstrate both efficiency and accuracy. Results sourced on RTX 3060 laptop platform .

<table>
<thead>
<tr>
<th>Model</th>
<th>Epochs</th>
<th>Train/Val Time</th>
<th>Final Val Loss</th>
<th>Max Dice</th>
</tr>
</thead>
<tbody>
<tr>
<td>UNet++</td>
<td>40</td>
<td>1h 10min</td>
<td>0.1677</td>
<td>0.9874</td>
</tr>
<tr>
<td>DeepLabV3+</td>
<td>40</td>
<td>21 min</td>
<td>0.1735</td>
<td>0.9894</td>
</tr>
<tr>
<td>UNet</td>
<td>40</td>
<td>30.5 min</td>
<td>0.1760</td>
<td>0.9888</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>Model</th>
<th>Mean Dice ± SD</th>
<th>Precision</th>
<th>Accuracy</th>
<th>Recall</th>
</tr>
</thead>
<tbody>
<tr>
<td>UNet++</td>
<td>0.8601 ± 0.1706</td>
<td>0.8793</td>
<td>0.9572</td>
<td>0.8866</td>
</tr>
<tr>
<td>DeepLabV3+</td>
<td>0.8604 ± 0.1844</td>
<td>0.8843</td>
<td>0.9610</td>
<td>0.8902</td>
</tr>
<tr>
<td>UNet</td>
<td>0.8592 ± 0.1866</td>
<td>0.8806</td>
<td>0.9562</td>
<td>0.8905</td>
</tr>
</tbody>
</table>

&lt;img&gt;Fig. 3. Polyp sizes&lt;/img&gt;

DeepLabv3+ is an encoder-decoder convolutional neural network (CNN) developed by researchers affiliated with Google. The model is intended for semantic image segmentation and builds upon DeepLabv3. DeepLabv3+ improves upon its predecessor by incorporating a decoder module that refines segmentation results along object edges, thereby addressing a key limitation. Additionally, DeepLabv3+ leverages Atrous Spatial Pyramid Pooling (ASPP) to capture rich contextual information. The model utilizes depthwise separable convolutions in the ASPP, which significantly reduces computational cost while maintaining accuracy. DeepLabv3+ has achieved state-of-the-art performance on industry benchmark datasets such as PASCAL VOC and Cityscapes.

&lt;img&gt;Fig. 4. PASCAL VOC 2012&lt;/img&gt;

&lt;img&gt;Fig. 5. Cityscapes&lt;/img&gt;

III. EXPERIMENTS

A. Image Resolution

Resolution refers to the level of detail in an image, specifically the number of pixels it contains. Higher pixel counts facilitate the identification of finer details, enabling more information to be processed and classified from visual input. [21] Similarly, segmentation models process this information, albeit as numerical tensors. Unlike human perception, which is limited, these models can access every pixel in an image. [23] However, their primary constraint is computational power. Increasing pixel density directly raises the computational resources required to train an image segmentation model. [22] This raises the question of whether the increased computational cost is justified and how the relationship between input size and computational power influences the final prediction. The goal of this experiment is to determine how input resolution affects segmentation accuracy, generalization across scales, and training efficiency on the Kvasir-SEG dataset. An assumption prior to the experiment was that high resolution-optimized models would capture more details like bounderies and small objects while low resolution-optimized models would struggle with small objects but not take as long to train.

&lt;img&gt;Fig. 6. Resolution effects on images&lt;/img&gt;

To evaluate the influence of image resolution on segmentation performance, a series of experiments was conducted in which the model was trained and validated at various input resolutions. These experiments were designed to reveal how scaling the input from the Kvasir-SEG dataset affects

---


## Page 5

model accuracy, generalization, and computation time. For this experiment, all augmentations were removed to maintain a minimal representation of the data. The only exception was the scaling and normalization of both splits. The use of a plateau learning rate scheduler provided additional insight into when the model reached its limit with the selected learning rate. The experiment fully utilized a random seed which can be observed in the con-fig parameter to support reproducibility. The early stopper was utilized in the first part of the experiment which covered direct model performance. Some of the key metrics monitored were epoch time, total loss, dice loss, precision, and IoU.

The first part of the experiment involved training the model on the set range of 64x64, 128x128, 256x256, 320x320, and lastly 512x512. These resolutions were then validated on 128x128, 512x512, and 768x768 pixel images. The setup allowed for the assessment of two key aspects. whether higher input detail improves segmentation accuracy and whether a model trained on one resolution generalizes effectively to others.

To identify a more optimal training and validation range, the second part of the experiment tested several resolutions between 288x288 and 416x416 pixels. This interval was selected based on earlier observations indicating that models trained in this region offered the best balance between accuracy and computational efficiency. Each configuration was trained on one resolution and validated across others to examine self-consistency and cross-scale generalization. Tables VIII-XII summarize the results across validation resolutions, demonstrating that all models within this range perform similarly, with only minor deviations in mean and IoU values.

**B. Augmentation**

To study how geometric data augmentation affects segmen-tation performance, we used the Albumentations library. [8] Albumentations was chosen for its speed and for ensuring that identical transformations are applied to both images and masks, maintaining pixel-level correspondence that is essential for semantic segmentation.

Only geometric and structural augmentations were used in this study. Photometric transformations (e.g., brightness, contrast, hue, gamma) were intentionally excluded, since color in colonoscopy images provides diagnostic cues related to tissue texture and vascular structure. According to Shorten and Khoshgoftaar, data augmentation improves generalization when used to simulate realistic variations, but inappropriate transformations can distort meaningful image features. [9] The focus here was therefore on geometric variations that reflect real conditions in endoscopic imaging, such as small camera shifts, scale changes, and local tissue deformation.

Six augmentation configurations were compared: (1) no augmentation (baseline); (2) *Light* augmentation: horizon-tal flips and mild shift–scale–rotate; (3) *Heavy* augmen-tation: stronger rotations (< 25°), perspective warp, and ElasticTransform; (4) *Elastic Warp*: local, non-grid de-formation using ElasticTransform; (5) *Grid Distortion*: structured geometric warp using a control grid; (6) *Blur/Noise*: GaussianBlur and GaussNoise. Elastic and grid-based distortions were applied with conservative parameters to main-tain anatomical realism.

All input images were resized to 320 × 320 pixels and nor-malized using ImageNet statistics to match the pretrained en-coder of DeepLabV3+. Experiments were conducted using the Kvasir-SEG dataset, [10] a public benchmark for colonoscopy polyp segmentation.

<table>
<thead>
<tr>
<th>ID</th>
<th>Technique</th>
<th>Transformations applied</th>
<th>Purpose</th>
</tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>Baseline</td>
<td>Resize and normalize</td>
<td>Control setup without augmentation.</td>
</tr>
<tr>
<td>2</td>
<td>Light Augmentation</td>
<td>Horizontal flip ($p = 0.5$), ShiftScaleRotate ($\pm 10^\circ$)</td>
<td>Adds small geometric variation similar to camera motion.</td>
</tr>
<tr>
<td>3</td>
<td>Heavy Augmentation</td>
<td>Horizontal and vertical flips, ShiftScaleRotate ($\pm 25^\circ$), Perspective warp, Gaussian blur, ElasticTransform</td>
<td>Tests robustness to larger geometric changes.</td>
</tr>
</tbody>
</table>

<table>
<thead>
<tr>
<th>ID</th>
<th>Technique</th>
<th>Transformations applied</th>
<th>Purpose</th>
</tr>
</thead>
<tbody>
<tr>
<td>4</td>
<td>Elastic Warp</td>
<td>ElasticTransform ($\alpha = 30$, $\sigma = 4$, $\alpha_{affine} = 12$)</td>
<td>Simulates soft-tissue deformation.</td>
</tr>
<tr>
<td>5</td>
<td>Grid Distortion</td>
<td>GridDistortion (num_steps=5, distort_limit=0.15)</td>
<td>Introduces mild geometric warping resembling lens distortion.</td>
</tr>
<tr>
<td>6</td>
<td>Blur / Noise</td>
<td>GaussianBlur (3-5 px), GaussNoise (variance 5-20)</td>
<td>Adds small artifacts and sensor noise for robust-ness.</td>
</tr>
</tbody>
</table>

**C. Training data size**

In this experiment, we evaluated the impact of training data size on the segmentation performance of the DeepLabV3+ model using the Adam optimizer. All hyperparameters were held constant; only the training set size varied across 100%, 75%, 50%, 25%, 10%, and 5%. We reported validation loss, BCE loss, Dice loss, Dice coefficient, IoU loss, IoU score, precision, and recall.

To ensure a fair comparison across data sizes, early stopping was deliberately disabled. This avoids bias where smaller datasets might trigger early stopping sooner (due to noisier validation loss), while larger datasets could train longer and thus appear artificially better. Instead, each model was trained for the same fixed number of epochs, ensuring that differences in performance stem solely from data availability rather than training duration.

**D. Optimization methods**

To investigate how the choice of optimizer affects polyp image segmentation, we keep everything else fixed (model = DeepLabV3+/ same train/val split, same augmentations, same loss = BCEWithLogits + soft Dice) and only change the optimizer. This follows the idea that different optimizers converge to different global and local minima with different speed, and that adaptive methods do not always generalize better than plain SGD with momentum on vision tasks, as reported in the medical segmentation literature. [1]

---


## Page 6

In our experiments we therefore limit ourselves to standard and well documented optimizer that are implemented in PyTorch and are described in the optimization overview by Ruder and in the original Adam Paper.

1) *SGD with momentum*

Plain Stochastic Gradient Decent on its own can be very unstable: if the learning rate is not tuned well, the updates can bounce back and forth ("zig-zag"), especially in narrow valleys of the loss surface. Momentum reduces this by carrying over part of the previous step into the current one. Here, $\gamma$ denotes the momentum term (often set to 0.9) and $\eta$ is the learning rate. [2]

$$\theta_{t+1} = \theta_t - v_t \tag{7}$$

Why is it relevant for us? Mortazi et al. report that momentum-type optimizers still generalize very well in medical segmentation, but they require careful learning-rate tuning. [1] Our experiments will therefore include SGD + momentum as the baseline.

2) *Nesterov Accelerated Gradient (NAG)*

Nesterov accelerated gradient (NAG) is a "look-ahead" version of momentum. Instead of computing the gradient at the current point, it first moves in the direction of the previous momentum and then measures the gradient there. Ruder writes it as:

$$v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta_t - \gamma v_{t-1}), \quad \theta_{t+1} = \theta_t - v_t$$

The main change from standard momentum is that the gradient is evaluated at the projected future position $\theta_t - \gamma v_{t-1}$. This "peek ahead" often leads to faster and more stable convergence. [2] We include it because Mortazi et al. also discuss momentum-rate (MR) together with learning rate (LR) and show that changing both can improve generalization. [1]

3) *AdaGrad*

AdaGrad adjusts the step size for each parameter separately by keeping track of how large its past gradients have been. For a parameter $\theta_i$, it accumulates the squared gradients:

$$G_{t,ii} = \sum_{\tau=1}^{t} g_{\tau,i}^2$$

and updates with:

$$\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii} + \epsilon}} g_{t,i}$$

Ruder shows this exact form and points out that it is useful when different parameters should move at different speeds, because frequently updated parameters get a smaller effective learning rate. The downside is that the denominator keeps growing, so the learning rate keeps shrinking over time. [2] We include it as it is one of the older adaptive methods in the comparisons.

4) *RMSPprop*

RMSPprop addresses AdaGrad's problem of the learning rate shrinking toward zero by keeping an exponential moving average of the squared gradients:

$$E[g_t^2] = \alpha E[g_{t-1}^2] + (1-\alpha)g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g_t^2] + \epsilon}} g_t$$

This is the version Ruder shows, originally from Hinton's lecture, with typical values $\alpha \approx 0.9$ and $\eta \approx 10^{-3}$. [2] We include RMSProp because it is still commonly used in vision models as a simple and effective adaptive optimizer.

5) *Adam*

Adam is an adaptive gradient optimizer, that maintains exponential moving avrages of both the gradients (first moment) and the squared gradient (second moment), and used that to choose a god step size for each weight. For Adam we use the original Kingma & Ba (2014) formulation:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Adam converges fast and needs little tuning, which we already used as the baseline optimizer in our experiments. Mortazi et al. note that adaptive optimizers like Adam make training faster but sometimes converge to a different minimum than SGD, so it is worth reporting Dice/IoU side-by-side. [1], [2]

6) *AdamW*

Loshchilov & Hutter (2019) show that using L2 in Adam is not the same as weight decay, and that you should decouple the decay from the adaptive update. Their final update is:

$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta \lambda \theta_t$$

Concretely, AdamW first performs the Adam parameter update, and only afterwards applies weight decay as a separate step. This separation is what makes AdamW closer to true weight decay and improves its generalization. This also shows that this makes Adam competitive with SGD + momentum on vision tasks [3]. Since we are doing medical segmentation, we include AdamW as the "improved Adam" in the experiments.

IV. RESULTS AND DISCUSSION

A. *Resolution*

1) *part1*

Tables IV-VI show results across 3 validation resolutions (128,512, 768), summarized: Not all tables are present. They can be found in the jupyter-notebook cells.

---


## Page 7

• Models perform best when training and validation resolutions are close.
• Matching resolutions yield the highest Dice and IoU with minimal variation.
• High-resolution models generalize better downward than low ones do upward.
• Small resolution differences still maintain stable results across scales.
• Beyond 512 px, higher detail brings little generalization gain relative to cost.

**TABLE V**
VALIDATED ON 128×128 IMAGES.

<table>
<thead>
<tr>
<th>Train<sub>res</sub></th>
<th>Dice<sub>max</sub></th>
<th>Dice<sub>min</sub></th>
<th>Dice<sub>mean</sub></th>
<th>IoU<sub>max</sub></th>
<th>IoU<sub>min</sub></th>
<th>IoU<sub>mean</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td>64</td>
<td>0.55</td>
<td>0.15</td>
<td>0.47</td>
<td>0.43</td>
<td>0.07</td>
<td>0.36</td>
</tr>
<tr>
<td>128</td>
<td><strong>0.79</strong></td>
<td><strong>0.58</strong></td>
<td><strong>0.75</strong></td>
<td><strong>0.70</strong></td>
<td><strong>0.51</strong></td>
<td><strong>0.66</strong></td>
</tr>
<tr>
<td>256</td>
<td>0.64</td>
<td>0.48</td>
<td>0.58</td>
<td>0.55</td>
<td>0.39</td>
<td>0.50</td>
</tr>
<tr>
<td>320</td>
<td>0.58</td>
<td>0.33</td>
<td>0.49</td>
<td>0.49</td>
<td>0.24</td>
<td>0.40</td>
</tr>
<tr>
<td>512</td>
<td>0.44</td>
<td>0.26</td>
<td>0.39</td>
<td>0.36</td>
<td>0.17</td>
<td>0.31</td>
</tr>
</tbody>
</table>

**TABLE VI**
VALIDATED ON 512×512 IMAGES.

<table>
<thead>
<tr>
<th>Train<sub>res</sub></th>
<th>Dice<sub>max</sub></th>
<th>Dice<sub>min</sub></th>
<th>Dice<sub>mean</sub></th>
<th>IoU<sub>max</sub></th>
<th>IoU<sub>min</sub></th>
<th>IoU<sub>mean</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td>128</td>
<td>0.34</td>
<td>0.11</td>
<td>0.26</td>
<td>0.23</td>
<td>0.04</td>
<td>0.17</td>
</tr>
<tr>
<td>256</td>
<td>0.66</td>
<td>0.28</td>
<td>0.55</td>
<td>0.56</td>
<td>0.21</td>
<td>0.45</td>
</tr>
<tr>
<td>320</td>
<td>0.54</td>
<td>0.05</td>
<td>0.26</td>
<td>0.44</td>
<td>0.00</td>
<td>0.17</td>
</tr>
<tr>
<td>512</td>
<td><strong>0.83</strong></td>
<td><strong>0.31</strong></td>
<td><strong>0.75</strong></td>
<td><strong>0.75</strong></td>
<td><strong>0.24</strong></td>
<td><strong>0.67</strong></td>
</tr>
<tr>
<td>64</td>
<td>0.10</td>
<td>0.02</td>
<td>0.06</td>
<td>0.05</td>
<td>0.00</td>
<td>0.02</td>
</tr>
</tbody>
</table>

**TABLE VII**
VALIDATED ON 768×768 IMAGES.

<table>
<thead>
<tr>
<th>Train<sub>res</sub></th>
<th>Dice<sub>max</sub></th>
<th>Dice<sub>min</sub></th>
<th>Dice<sub>mean</sub></th>
<th>IoU<sub>max</sub></th>
<th>IoU<sub>min</sub></th>
<th>IoU<sub>mean</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td>128</td>
<td>0.15</td>
<td>0.06</td>
<td>0.11</td>
<td>0.08</td>
<td>0.02</td>
<td>0.06</td>
</tr>
<tr>
<td>256</td>
<td>0.20</td>
<td>0.01</td>
<td>0.07</td>
<td>0.13</td>
<td>0.00</td>
<td>0.04</td>
</tr>
<tr>
<td>320</td>
<td>0.27</td>
<td>0.01</td>
<td>0.06</td>
<td>0.19</td>
<td>0.00</td>
<td>0.03</td>
</tr>
<tr>
<td><strong>512</strong></td>
<td><strong>0.77</strong></td>
<td><strong>0.05</strong></td>
<td><strong>0.60</strong></td>
<td><strong>0.68</strong></td>
<td><strong>0.00</strong></td>
<td><strong>0.51</strong></td>
</tr>
<tr>
<td>64</td>
<td>0.09</td>
<td>0.03</td>
<td>0.07</td>
<td>0.04</td>
<td>0.01</td>
<td>0.03</td>
</tr>
</tbody>
</table>

&lt;img&gt;Fig. 7. Models results on 128x128&lt;/img&gt;

&lt;img&gt;Fig. 8. Models results on 512x512&lt;/img&gt;

&lt;img&gt;Fig. 9. Models results on 768x768&lt;/img&gt;

Analysis of validation tables IV through VI revealed several relationships between input resolution and segmentation quality. The model achieved optimal performance when training and validation resolutions were similar, with accuracy declining as the difference between these resolutions increased. This proximity effect was observed at both high and low resolutions. For example, a model trained at 64x64 resolution performed better when validated on 128x128 images than on 512x512 images. Conversely, models trained on 512x512 and 768x768 images could not generalize to 64x64 images, with performance approximating random guessing.

Training and validating on the same resolution yields the most stable results. Across all tables, mean Dice and mean IoU scores are highest when training and validation sizes match. This indicates that the model learns feature scales specific to that resolution, resulting in high accuracy.

Performance degradation is less pronounced when high-resolution models are validated on lower resolutions than when the opposite occurs. Notably, when validation is performed at resolutions slightly above or below the training scale, performance can remain stable or even improve marginally. This suggests that the model generalizes effectively within a narrow range of its training resolution, with mild resolution shifts functioning as a form of implicit augmentation.

Even when high-resolution models lose accuracy at mismatched validation sizes, their mean Dice and mean IoU values decline more gradually. However, increasing resolution beyond a certain threshold does not yield proportionally greater improvements in performance. Instead, performance plateaus, offering only marginal gains relative to the substantial increase in training time and memory consumption.

---


## Page 8

TABLE VIII
MEAN PRECISION AND RECALL FOR MODELS VALIDATED ON 512x512 IMAGES.

<table>
<thead>
<tr>
<th>Train Res.</th>
<th>Precision<sub>mean</sub></th>
<th>Recall<sub>mean</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td>64</td>
<td>0.23</td>
<td>0.03</td>
</tr>
<tr>
<td>128</td>
<td>0.45</td>
<td>0.28</td>
</tr>
<tr>
<td>256</td>
<td>0.82</td>
<td>0.50</td>
</tr>
<tr>
<td>320</td>
<td>0.82</td>
<td>0.19</td>
</tr>
<tr>
<td>512</td>
<td>0.79</td>
<td>0.82</td>
</tr>
</tbody>
</table>

Recall values reflect the model’s ability to identify relevant regions, such as polyp surfaces. The table demonstrates how recall varies depending on the difference between the validation resolution and the training resolution. When validated on 128x128 images, models trained at lower resolutions (64x64 and 128x128) maintain high recall (approximately 0.9 and 0.83), capturing most target regions, though with reduced precision. In contrast, high-resolution models (512x512) exhibit a recall of about 0.6, indicating difficulty in recognizing objects when the input is smaller than the training resolution. At 512x512, models trained within the middle resolution range (320 to 512) perform far better than those in the low end, with recall values showing immense potential. Low-resolution models are unable to match this performance, with recall collapsing to the low 10 percent range, indicating insufficient generalization for larger inputs.
At 768x768, recall values plateau, with even the closest high-resolution models showing no further improvement (mean recall of 0.57). Models trained at resolutions below 512x512 collapse entirely at this scale. Increasing resolution beyond this point results in higher computational cost with minimal performance benefit.

Although higher resolutions generally yield improved segmentation accuracy, they also result in increased computational cost. The table below summarizes the average training time per epoch for each training resolution when validated on 512x512 images.

2) part 2

Building on the broad resolution analysis from Part 1, the second part focuses on the narrower interval between 288x288 and 416x416. The earlier experiments showed that low resolutions collapse sharply at higher validation scales, while very high resolutions add computational cost with only small accuracy gains. The mid-range consistently produced stable Dice and IoU values, suggesting it is the most practical region to search for an optimal training and validation configuration. Part 2 therefore examines this interval in more detail to identify whether any specific resolution offers a stronger balance between accuracy, stability, and efficiency.

TABLE X
DEEPLABV3+ VALIDATED ON 288x288.

<table>
<thead>
<tr>
<th>Train</th>
<th>Dice<sub>max</sub></th>
<th>Dice<sub>min</sub></th>
<th>Dice<sub>mean</sub></th>
<th>IoU<sub>max</sub></th>
<th>IoU<sub>min</sub></th>
<th>IoU<sub>mean</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td>288</td>
<td>0.84</td>
<td>0.63</td>
<td>0.81</td>
<td>0.77</td>
<td>0.56</td>
<td>0.74</td>
</tr>
<tr>
<td>320</td>
<td>0.85</td>
<td>0.60</td>
<td>0.81</td>
<td>0.78</td>
<td>0.52</td>
<td>0.74</td>
</tr>
<tr>
<td>352</td>
<td>0.83</td>
<td>0.56</td>
<td>0.79</td>
<td>0.75</td>
<td>0.48</td>
<td>0.71</td>
</tr>
<tr>
<td>384</td>
<td>0.82</td>
<td>0.64</td>
<td>0.78</td>
<td>0.74</td>
<td>0.56</td>
<td>0.71</td>
</tr>
<tr>
<td>416</td>
<td>0.79</td>
<td>0.59</td>
<td>0.75</td>
<td>0.72</td>
<td>0.54</td>
<td>0.69</td>
</tr>
</tbody>
</table>

TABLE XI
DEEPLABV3+ VALIDATED ON 416x416.

<table>
<thead>
<tr>
<th>Train</th>
<th>Dice<sub>max</sub></th>
<th>Dice<sub>min</sub></th>
<th>Dice<sub>mean</sub></th>
<th>IoU<sub>max</sub></th>
<th>IoU<sub>min</sub></th>
<th>IoU<sub>mean</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td>288</td>
<td>0.83</td>
<td>0.63</td>
<td>0.78</td>
<td>0.74</td>
<td>0.54</td>
<td>0.69</td>
</tr>
<tr>
<td>320</td>
<td>0.83</td>
<td>0.62</td>
<td>0.80</td>
<td>0.75</td>
<td>0.53</td>
<td>0.72</td>
</tr>
<tr>
<td>352</td>
<td>0.83</td>
<td>0.53</td>
<td>0.80</td>
<td>0.76</td>
<td>0.44</td>
<td>0.72</td>
</tr>
<tr>
<td>384</td>
<td>0.85</td>
<td>0.58</td>
<td>0.82</td>
<td>0.78</td>
<td>0.52</td>
<td>0.75</td>
</tr>
<tr>
<td>416</td>
<td>0.84</td>
<td>0.64</td>
<td>0.81</td>
<td>0.76</td>
<td>0.54</td>
<td>0.74</td>
</tr>
</tbody>
</table>

The training times revealed a linear relationship between the chosen resolution and the computer’s processing time. Across all runs, the training times were nearly identical, regardless of the validation set size, indicating an effective pipeline with minimal bottlenecks. The 352x352 configuration provided the best efficiency and accuracy trade-off, achieving strong Dice and IoU scores at higher resolutions while requiring approximately 25% less computation per epoch.

TABLE XII
PERFORMANCE SUMMARY OF DEEPLABV3+ TRAINED ON 352x352 RESOLUTION ACROSS DIFFERENT VALIDATION SCALES.

<table>
<thead>
<tr>
<th>Validation Res.</th>
<th>Dice<sub>max</sub></th>
<th>Dice<sub>min</sub></th>
<th>Dice<sub>mean</sub></th>
<th>IoU<sub>max</sub></th>
<th>IoU<sub>mean</sub></th>
</tr>
</thead>
<tbody>
<tr>
<td>288</td>
<td>0.83</td>
<td>0.56</td>
<td>0.79</td>
<td>0.75</td>
<td>0.71</td>
</tr>
<tr>
<td>320</td>
<td>0.85</td>
<td>0.65</td>
<td>0.81</td>
<td>0.78</td>
<td>0.74</td>
</tr>
<tr>
<td>352</td>
<td>0.85</td>
<td>0.64</td>
<td>0.81</td>
<td>0.77</td>
<td>0.74</td>
</tr>
<tr>
<td>384</td>
<td>0.85</td>
<td>0.63</td>
<td>0.81</td>
<td>0.78</td>
<td>0.74</td>
</tr>
<tr>
<td>416</td>
<td>0.83</td>
<td>0.53</td>
<td>0.80</td>
<td>0.76</td>
<td>0.72</td>
</tr>
</tbody>
</table>

The results indicate a nearly exponential increase in training time as image resolution increases. The 320x320 configuration requires only half the time needed for the 512x512 training setup to complete an epoch (a full iteration over the dataset), while maintaining 94% of its mean dice score. This balance between efficiency and accuracy makes the range between 128- sub 512 configuration an attractive compromise for further experimentation.

---


## Page 9

TABLE XIII
MEAN TRAINING TIME PER RESOLUTION (SECONDS PER EPOCH).

<table>
  <thead>
    <tr>
      <th>Train Resolution</th>
      <th>Mean Time (s/epoch)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>288</td>
      <td>13.6</td>
    </tr>
    <tr>
      <td>320</td>
      <td>15.8</td>
    </tr>
    <tr>
      <td>352</td>
      <td>18.3</td>
    </tr>
    <tr>
      <td>384</td>
      <td>19.7</td>
    </tr>
    <tr>
      <td>416</td>
      <td>24.1</td>
    </tr>
  </tbody>
</table>

Table XII summarizes the mean precision and recall values across all tested and validated resolutions. The close uniformity of results across the 288-416 range confirms that model stability remains largely unaffected by variations in the validation scale. Notably, the 352x352 configuration stands out, showing a well-balanced level of precision and recall across all validation resolutions. The validations further cement the credibility of the assumption that training within this mid-range (288-416) yields the most optimal combination of performance and scalability, at least with normal computational resources. This would serve as the foundation for the final training loop.

B. Training Data Size Results

TABLE XV
PERFORMANCE METRICS (LOSS-BASED) OF DEEPLABV3+ TRAINED ON VARYING DATA SIZES AT 352x352 RESOLUTION.

<table>
  <thead>
    <tr>
      <th>Percentage</th>
      <th>Val Loss</th>
      <th>BCE Loss</th>
      <th>Dice Loss</th>
      <th>Dice Coef</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>100</td>
      <td>0.2151</td>
      <td>0.2526</td>
      <td>0.1629</td>
      <td>0.8371</td>
    </tr>
    <tr>
      <td>75</td>
      <td><b>0.2068</b></td>
      <td><b>0.2404</b></td>
      <td><b>0.158</b></td>
      <td><b>0.842</b></td>
    </tr>
    <tr>
      <td>50</td>
      <td>0.2334</td>
      <td>0.2554</td>
      <td>0.1876</td>
      <td>0.8124</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.2943</td>
      <td>0.3399</td>
      <td>0.2262</td>
      <td>0.7738</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.3501</td>
      <td>0.4261</td>
      <td>0.2566</td>
      <td>0.7434</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.3918</td>
      <td>0.3829</td>
      <td>0.3410</td>
      <td>0.6590</td>
    </tr>
  </tbody>
</table>

<table>
  <thead>
    <tr>
      <th>Percentage</th>
      <th>IOU Loss</th>
      <th>IOU Score</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>100</td>
      <td>0.2398</td>
      <td>0.7602</td>
      <td>0.8489</td>
      <td>0.8489</td>
    </tr>
    <tr>
      <td>75</td>
      <td><b>0.2326</b></td>
      <td><b>0.7674</b></td>
      <td>0.8547</td>
      <td><b>0.8912</b></td>
    </tr>
    <tr>
      <td>50</td>
      <td>0.2645</td>
      <td>0.7355</td>
      <td><b>0.8687</b></td>
      <td>0.8423</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.3168</td>
      <td>0.6832</td>
      <td>0.8191</td>
      <td>0.8194</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.3513</td>
      <td>0.6847</td>
      <td>0.8617</td>
      <td>0.7492</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.4281</td>
      <td>0.5719</td>
      <td>0.8308</td>
      <td>0.6751</td>
    </tr>
  </tbody>
</table>

As shown in Table 8, performance improved consistently with larger training sets up to 75–100%. Training on 75% of the data slightly outperformed the 100% run across most metrics (Dice coefficient 0.8420 vs. 0.8371, IoU 0.7674 vs. 0.7602, and validation loss 0.2068 vs. 0.2151). Interestingly, the 50% model achieved the highest precision (0.8687), suggesting it produced fewer false positives, whereas the 75% model achieved the highest recall (0.8912), indicating it detected more true positives. This trade-off reflects how varying data size can subtly shift the model’s balance between precision and recall.

The model remained robust with 50% of the training data (Dice 0.8124, IoU 0.7355), though both validation and component losses increased. At 25 %, segmentation quality declined further (Dice 0.7738, IoU 0.6832), which falls just below the typical 0.8 Dice threshold often used to gauge acceptable segmentation quality. [1], [4] Below this point, performance degraded sharply: Dice 0.7434 at 10 % and 0.6590 at 5 %, suggesting clear underfitting due to insufficient training data. These findings show that data size influences not only overall segmentation quality but also the trade-off between precision and recall. The model trained on 50% of the data achieved the highest precision (fewer false positives), while the 75% model achieved the highest recall (fewer missed positives). Depending on the clinical priority—minimizing false positives (e.g., avoiding unnecessary interventions) or minimizing false negatives (e.g., detecting all pathologies)—the optimal training data size may differ.

C. Augmentation Results and Discussion

TABLE XIV
RAW AVERAGE PRECISION AND RECALL MEANS FOR EACH TRAINING AND VALIDATION RESOLUTION.

<table>
  <thead>
    <tr>
      <th>Val ↓ / Trn →</th>
      <th>288</th>
      <th>320</th>
      <th>352</th>
      <th>384</th>
      <th>416</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>288x288</td>
      <td>0.78 / 0.86</td>
      <td>0.79 / 0.87</td>
      <td>0.80 / 0.88</td>
      <td>0.85 / 0.86</td>
      <td>0.83 / 0.87</td>
    </tr>
    <tr>
      <td>320x320</td>
      <td>0.85 / 0.83</td>
      <td>0.82 / 0.87</td>
      <td>0.85 / 0.87</td>
      <td>0.84 / 0.84</td>
      <td>0.85 / 0.86</td>
    </tr>
    <tr>
      <td>352x352</td>
      <td>0.84 / 0.85</td>
      <td>0.84 / 0.88</td>
      <td>0.84 / 0.86</td>
      <td>0.84 / 0.85</td>
      <td>0.86 / 0.84</td>
    </tr>
    <tr>
      <td>384x384</td>
      <td>0.85 / 0.85</td>
      <td>0.83 / 0.87</td>
      <td>0.86 / 0.85</td>
      <td>0.85 / 0.82</td>
      <td>0.86 / 0.84</td>
    </tr>
    <tr>
      <td>416x416</td>
      <td>0.86 / 0.84</td>
      <td>0.86 / 0.86</td>
      <td>0.85 / 0.83</td>
      <td>0.83 / 0.82</td>
      <td>0.84 / 0.84</td>
    </tr>
  </tbody>
</table>

3) conclusion on resolution

The resolution experiment demonstrates a consistent trend: segmentation model quality improves as resolution increases, but only to a certain threshold. Models collapse when the gap between training and validation resolutions becomes too large. Low-resolution models cannot scale upward, and high-resolution models lose stability when applied to very small inputs. Matching training and validation resolutions yielded the highest Dice and IoU scores, confirming that feature scale alignment is crucial for this dataset when training DeepLabV3+. The experiment further concludes that, within the 288 to 416 resolution range, sensitivity to resolution is greatly reduced. Across this interval, Dice, IoU, precision, and recall remain stable, regardless of the combination. Within these parameters, a resolution of 352x352 consistently emerges as the optimal trade-off, offering high accuracy across all scales while maintaining efficient training times. Training times increase linearly with resolution, indicating that excessively high resolutions result in unnecessary computational expenditure. Overall, the findings indicate that image segmentation does not benefit from extreme resolutions at either end of the tested range.

Six augmentation configurations were tested. The applied transformations and their intended effects are summarized in Table XVI. As shown in Figure 10, flips and rotations are clearly visible, whereas ElasticTransform and GridDistortion create subtle local deformations that are

---


## Page 10

harder to see but add useful spatial variability. For visualization, deformation strength was temporarily increased to make these effects visible, while training used conservative parameters to preserve anatomical realism. Color-based augmentations (e.g., brightness, contrast, hue) were deliberately excluded because color in colonoscopy frames carries clinically meaningful cues.

&lt;img&gt;Fig. 10. Visual examples of the augmentation configurations.&lt;/img&gt;

used a fixed seed (=42) for Python, NumPy, and PyTorch (including CUDA), with cuDNN set to deterministic and num_workers=0 to ensure identical shuffling and batching. No random augmentations were applied, so results results are reproducible.

Neural networks do not optimize over a simple bowl-shaped loss as in the convex case. Instead, the loss surface has many bumps and valleys (see Figure [11]), so an optimizer can get stuck in a “good” but not “best” region, often referred to as a local minimum versus a global minimum. With the higher learning rate (0.01), the momentum-based optimizers (SGD, NAG) could take larger steps and move past small valleys, which is also why they reached the best Dice in Table [XVII]. With the lower learning rate (0.001), the steps were smaller and the adaptive optimizers (AdaGrad, Adam, AdamW) could refine the solution within one region and match the results in Table [XVIII]. A similar idea of shaping updates so that training converges faster and more reliably is described in the LearnOpenCV article on YOLO loss design [20].

<table>
<thead>
<tr>
<th>Experiment</th>
<th>Best Epoch</th>
<th>Best Val Dice ↑</th>
<th>Best Val IoU ↑</th>
<th>Best Val Loss ↓</th>
<th>Mean Val Dice</th>
<th>Mean Val IoU</th>
<th>Mean Val Loss</th>
</tr>
</thead>
<tbody>
<tr>
<td>light_aug</td>
<td>16</td>
<td>0.8453</td>
<td>0.7826</td>
<td>0.1741</td>
<td>0.7852</td>
<td>0.7129</td>
<td>0.2376</td>
</tr>
<tr>
<td>grid_distortion</td>
<td>17</td>
<td>0.8411</td>
<td>0.7742</td>
<td>0.1879</td>
<td>0.7878</td>
<td>0.7144</td>
<td>0.2415</td>
</tr>
<tr>
<td>heavy_aug</td>
<td>19</td>
<td>0.8265</td>
<td>0.7600</td>
<td>0.1984</td>
<td>0.7678</td>
<td>0.6976</td>
<td>0.2498</td>
</tr>
<tr>
<td>blur_noise</td>
<td>18</td>
<td>0.8047</td>
<td>0.7249</td>
<td>0.2400</td>
<td>0.7569</td>
<td>0.6799</td>
<td>0.2756</td>
</tr>
<tr>
<td>baseline</td>
<td>11</td>
<td>0.7891</td>
<td>0.7087</td>
<td>0.2440</td>
<td>0.7378</td>
<td>0.6565</td>
<td>0.3015</td>
</tr>
<tr>
<td>elastic_warp</td>
<td>5</td>
<td>0.7692</td>
<td>0.6936</td>
<td>0.2732</td>
<td>0.7477</td>
<td>0.6695</td>
<td>0.2815</td>
</tr>
</tbody>
</table>

**TABLE XVI**

VALIDATION PERFORMANCE ACROSS AUGMENTATION CONFIGURATIONS (ARROWS INDICATE DIRECTION OF IMPROVEMENT).

**Findings.** The *Light Augmentation* setup achieved the best overall performance with a Dice of 0.845 and the lowest validation loss. *Grid Distortion* also performed strongly, suggesting that mild non-rigid spatial changes can improve generalization. *Heavy Augmentation* and *Blur/Noise* gave moderate gains over the baseline but slightly reduced precision, likely because strong transformations distorted fine polyp boundaries. *Elastic Warp* produced the lowest Dice, probably due to over-deformation of small regions.

**Takeaway.** Moderate geometric augmentation improves generalization and stability, while overly aggressive distortions tend to reduce accuracy. In small medical datasets, geometric augmentation is an effective lever to boost performance. [8], [9] Realistic transformations such as flips and mild rotations deliver the largest improvements while preserving clinical interpretability; elastic and grid-based methods offer smaller but meaningful gains in spatial robustness. Overall, the results confirm that carefully balanced geometric augmentation improves model reliability on the Kvasir-SEG dataset. [8], [10]

**D. Optimizer experimental setup and results**

In all experiments, the same network architecture and loss function were used; only the optimizer and the learning rate (LR) were varied. Six optimizers were compared: AdaGrad, Adam, AdamW, NAG, RMSProp, and SGD, each at two learning rates (0.01 and 0.001). All other hyperparameters (batch size, augmentation, combined Dice+BCE loss, and the learning-rate scheduler) were kept constant. All optimizer runs

&lt;img&gt;Fig. 11. Convex vs. non-convex optimization landscape. Convex objectives have a single minimizer; non-convex objectives may contain local minima, saddle points, and global minima.&lt;/img&gt;

After each epoch, the model was evaluated on the validation split, and the Dice coefficient (DSC) was used as the primary optimization signal. A Reduce-on-Plateau learning-rate scheduler was used to decrease the step size when the validation DSC did not improve for two consecutive epochs; in that case, the learning rate was reduced by a factor of 0.5. To avoid unnecessary training once the model had converged, early stopping was applied. The early-stopping criterion was configured through the experiment configuration (patience = 9, minimum improvement = 0.003): training was stopped if the validation DSC did not improve by at least 0.003 over nine epochs. This ensured that each experiment ran until either convergence or a maximum of 100 epochs, making the results comparable across optimizers and learning-rate settings.

---


## Page 11

Learning Rate = 0.01

&lt;img&gt;A line graph showing Validation Dice over training epochs for each optimizer at a learning rate of 0.01. The x-axis represents Epochs (0 to 40), and the y-axis represents Dice Score (0 to 1). Six lines represent different optimizers:
- AdaGrad (blue circles)
- Adam (orange triangles)
- AdamW (green squares)
- NAG (red diamonds)
- RMSProp (purple pentagons)
- SGD (brown squares)

The SGD and NAG lines rise quickly and reach the highest Dice values, converging early and stabilizing at strong performance. In contrast, the adaptive optimizers (AdaGrad, Adam, AdamW, RMSProp) improve more slowly and plateau at significantly lower Dice values.&lt;/img&gt;

Fig. 12. Validation Dice over training epochs for each optimizer at a learning rate of 0.01. The momentum-based optimizers (SGD and NAG) rise quickly and reach the highest Dice values, converging early and stabilizing at strong performance. In contrast, the adaptive optimizers (AdaGrad, Adam, AdamW, RMSProp) improve more slowly and plateau at significantly lower Dice values.

fast convergence to a high-performing plateau. All optimizers eventually stabilized by the end of training at this learning rate, and no serious oscillations were observed in their validation Dice curves.

As summarized in Table **XVII** below, Nesterov Accelerated Gradient (NAG) obtained the highest Dice (0.868) at LR = 0.01, closely followed by SGD (0.859). These two optimizers had almost no gap between their peak Dice and their mean Dice over the last epochs, which reflects stable performance once converged. Among the adaptive optimizers, AdaGrad achieved the best Dice (0.776), but it peaked mid-training (epoch 19) and then declined slightly to 0.766 by the final epoch. The other adaptive methods (Adam, AdamW, RMSProp) improved more gradually and peaked near the last epochs, but they never caught up to SGD or NAG at this higher learning rate. At higher learning rate, the adaptive optimizers likely shrink their step sizes to quickly and follow noisy, parameter wise updates, so they settle in shaper, worse minima, while SGD and NAG keeps more consistent large steps and reach wider valleys that generalize better on Kvasir-SEG data.

<table>
<thead>
<tr>
<th>Optimizer</th>
<th>Mean Val Dice</th>
<th>Best Val Dice</th>
<th>Epoch of Best</th>
<th>Final Val Dice</th>
</tr>
</thead>
<tbody>
<tr>
<td>NAG</td>
<td><strong>0.847</strong></td>
<td><strong>0.868</strong></td>
<td>25</td>
<td><strong>0.868</strong></td>
</tr>
<tr>
<td>SGD</td>
<td>0.838</td>
<td>0.859</td>
<td>19</td>
<td>0.859</td>
</tr>
<tr>
<td>AdaGrad</td>
<td>0.662</td>
<td>0.776</td>
<td>19</td>
<td>0.766</td>
</tr>
<tr>
<td>RMSProp</td>
<td>0.632</td>
<td>0.764</td>
<td>34</td>
<td>0.761</td>
</tr>
<tr>
<td>AdamW</td>
<td>0.628</td>
<td>0.751</td>
<td>39</td>
<td>0.748</td>
</tr>
<tr>
<td>Adam</td>
<td>0.607</td>
<td>0.728</td>
<td>34</td>
<td>0.719</td>
</tr>
</tbody>
</table>

TABLE XVII
VALIDATION DICE FOR DIFFERENT OPTIMIZERS (STARTING LR = 0.01).

Learning Rate = 0.001

&lt;img&gt;A line graph showing Validation Dice over training epochs for each optimizer at a learning rate of 0.001. The x-axis represents Epochs (0 to 40), and the y-axis represents Dice Score (0 to 1). Six lines represent different optimizers:
- AdaGrad (blue circles)
- Adam (orange triangles)
- AdamW (green squares)
- NAG (red diamonds)
- RMSProp (purple pentagons)
- SGD (brown squares)

Here the adaptive optimizers (AdaGrad, Adam, AdamW, RMSProp) benefit from the smaller step size, improving steadily throughout training and ultimately achieving higher final Dice values than the momentum-based methods. SGD and NAG converge more slowly and reach lower peaks at this learning rate.&lt;/img&gt;

Fig. 13. Validation Dice over training epochs for each optimizer at a learning rate of 0.001. Here the adaptive optimizers (AdaGrad, Adam, AdamW, RMSProp) benefit from the smaller step size, improving steadily throughout training and ultimately achieving higher final Dice values than the momentum-based methods. SGD and NAG converge more slowly and reach lower peaks at this learning rate.

1) Optimizer comparison at LR = 0.01

At the higher learning rate of 0.01 we observe clear differences among the optimizers. The momentum-based optimizers achieved the highest validation Dice scores (≈ 0.86–0.87) and converged faster, whereas the adaptive optimizers reached lower peaks (≈ 0.73–0.78). In Figure 12, SGD and NAG show rapid early improvements and then flatten out, indicating

&lt;img&gt;Five pairs of images showing original images, ground truth masks, and predicted masks for validation predictions with DeepLabV3+ and NAG at learning rate 10⁻³ (best checkpoint exp4_NAG_final.pth).&lt;/img&gt;

Fig. 14. Representative validation predictions for DeepLabV3+ with NAG at learning rate 10⁻³ (best checkpoint exp4_NAG_final.pth).

---


## Page 12

&lt;img&gt;Fig. 15. Representative validation predictions for DeepLabV3+ with SGD at learning rate $10^{-2}$ (best checkpoint exp4_SGD_final.pth).&lt;/img&gt;

**2) Optimizer comparison at $LR = 0.001$**

Lowering the learning rate to 0.001 had a different impact on optimizer performance, especially benefiting the adaptive optimizers. Figure [13] shows that the adaptive optimizers achieved higher final Dice scores at the lower learning rate than they did at 0.01. Their learning curves are more gradual but steadily improving. In contrast, the momentum-based optimizers converged more slowly under $LR = 0.001$ and reached lower peak Dice values compared to their high-LR results.

The performance ranking at $LR = 0.001$ therefore shifted in favor of the adaptive optimizers, as shown in Table [XVIII]. AdaGrad obtained the highest Dice (0.861), which is a substantial improvement over its performance at $LR = 0.01$ (0.776). AdamW and Adam also improved, reaching peak Dice values of 0.841 and 0.836, respectively. By contrast, SGD and NAG peaked around 0.79 at the lower LR, which underlines the importance of tuning the learning rate per optimizer. Overall, the adaptive methods exhibit stable convergence at 0.001 and achieve higher performance when the step size is reduced.

<table>
<thead>
<tr>
<th>Optimizer</th>
<th>Mean Val Dice</th>
<th>Best Val Dice</th>
<th>Epoch of Best</th>
<th>Final Val Dice</th>
</tr>
</thead>
<tbody>
<tr>
<td>AdaGrad</td>
<td><strong>0.828</strong></td>
<td><strong>0.861</strong></td>
<td>30</td>
<td><strong>0.861</strong></td>
</tr>
<tr>
<td>AdamW</td>
<td>0.800</td>
<td>0.841</td>
<td>18</td>
<td>0.833</td>
</tr>
<tr>
<td>Adam</td>
<td>0.795</td>
<td>0.836</td>
<td>14</td>
<td>0.828</td>
</tr>
<tr>
<td>RMSProp</td>
<td>0.725</td>
<td>0.813</td>
<td>37</td>
<td>0.807</td>
</tr>
<tr>
<td>NAG</td>
<td>0.737</td>
<td>0.790</td>
<td>36</td>
<td>0.790</td>
</tr>
<tr>
<td>SGD</td>
<td>0.745</td>
<td>0.789</td>
<td>37</td>
<td>0.787</td>
</tr>
</tbody>
</table>

**TABLE XVIII**
VALIDATION DICE FOR DIFFERENT OPTIMIZERS (STARTING LR = 0.001).

&lt;img&gt;Fig. 16. Representative validation predictions for DeepLabV3+ with AdaGrad at learning rate $10^{-3}$ (best checkpoint exp4_AdaGrad_LR(0.001)_final.pth).&lt;/img&gt;

**3) Summary of learning-rate findings**

The optimizer experiments show three clear patterns.

First, optimizer performance is tightly coupled to the learning rate. At the higher learning rate (0.01), the momentum-based optimizers (SGD, NAG) were clearly superior. At the lower learning rate (0.001), the ranking flipped and the adaptive optimizers (AdaGrad, AdamW, Adam) performed better. This shows that it is not sufficient to “just pick an optimizer”, each optimizer has a learning-rate where it works best.

Second, the mean Dice confirms the stability aspect. At $LR = 0.01$, SGD and NAG achieved mean Dice around 0.84–0.85, meaning they were not only good at their peaks but also consistently good throughout training. The adaptive methods at 0.01 had means around 0.60–0.66. At $LR = 0.001$ this changed: AdaGrad reached a mean of 0.828, AdamW 0.800, and Adam 0.795 (Table [XIX]), showing that these adaptive methods benefit from the smaller step size.

Third, the number of epochs to reach the peak also shifted with the learning rate. With the higher $LR = 0.01$, the best runs for

---


## Page 13

the momentum-based optimizers peaked early (epochs 19 and 25). When the LR was lowered to 0.001, several optimizers needed more epochs to reach their best value: AdaGrad peaked at epoch 30, RMSProp at epoch 37, and even the momentum optimizers took 36–37 epochs at the lower LR. Only Adam and AdamW peaked earlier at 0.001 (epochs 14 and 18) than at 0.01. This suggests that lower learning rates often lead to longer training times, unless an adaptive optimizer is used that “likes” the smaller step size.

Overall, our experiments support the following practical recipe for Kvasir-SEG polyp segmentation: train at 352×352 with light plus grid-based augmentation, use a momentum-based optimizer at a learning rate of 0.01, and, where possible, exploit at least 75% of the training set to maximise performance. This configuration offers a favourable balance between accuracy, stability, and computational cost, and provides a strong baseline for future extensions to other architectures, datasets, or clinical settings.

VI. INDIVIDUAL CONTRIBUTIONS

Student ID: 30
Responsible for the *data size experiment*, developing the main code structure, as well as performing code cleaning, commenting, and final refinements on the main code template.

Student ID: 25
Responsible for the *data augmentation experiments*, development of the main code pipeline, and overall code cleaning and optimization.

Student ID: 4
Jointly responsible for *core code development*, code cleaning, commenting, and maintaining the main project structure. Contributed extensively to the *methodology section*, documentation, and final report preparation in Overleaf. Specifically led the *resolution experiments*, evaluating how image size impacted segmentation accuracy, and helped with integration of visual figures and tables in the final paper.

Student ID: 36
Jointly responsible for *core code development*, code cleaning, commenting, and maintaining the main project structure. Contributed extensively to the *methodology section*, documentation, and final report preparation in Overleaf. Specifically led the *optimizer experiments*, performing training runs, comparative analysis, and interpretation of results. Also assisted in result visualization, proofreading, and ensuring consistency across sections.

TABLE XIX
BEST-PERFORMING OPTIMIZER–LEARNING-RATE COMBINATIONS.

<table>
<thead>
<tr>
<th>Optimizer</th>
<th>Best LR</th>
<th>Best Val Dice</th>
<th>Mean Val Dice</th>
<th>Epoch of Best</th>
</tr>
</thead>
<tbody>
<tr>
<td>NAG</td>
<td>0.01</td>
<td><strong>0.868</strong></td>
<td><strong>0.847</strong></td>
<td>25</td>
</tr>
<tr>
<td>SGD</td>
<td>0.01</td>
<td>0.859</td>
<td>0.838</td>
<td>19</td>
</tr>
<tr>
<td>AdaGrad</td>
<td>0.001</td>
<td>0.861</td>
<td>0.828</td>
<td>30</td>
</tr>
<tr>
<td>AdamW</td>
<td>0.001</td>
<td>0.841</td>
<td>0.800</td>
<td>18</td>
</tr>
<tr>
<td>Adam</td>
<td>0.001</td>
<td>0.836</td>
<td>0.795</td>
<td><strong>14</strong></td>
</tr>
<tr>
<td>RMSProp</td>
<td>0.001</td>
<td>0.813</td>
<td>0.725</td>
<td>37</td>
</tr>
</tbody>
</table>

V. CONCLUSION

Using a controlled DeepLabV3+ pipeline with BCE-with-logits plus soft Dice loss and a fixed train–validation split, we systematically mapped how input resolution, data augmentation, training data volume, and optimization choices shape polyp segmentation performance on Kvasir-SEG. Resolution experiments consistently favoured the 288–416 range, with 352×352 emerging as a sweet spot that maintains strong Dice scores while reducing computational cost compared with higher resolutions, making it a practical choice for iterative experimentation and deployment on heterogeneous hardware. Data augmentation had a clear impact on performance. Light geometric transforms (horizontal flips and mild shift-scale-rotate) combined with grid distortion improved both best and mean Dice over the non-augmented baseline, whereas heavier, elastic, or blur-dominated policies underperformed. This suggests that geometry-preserving variability is more beneficial than aggressive deformations, which may over-warp clinically relevant structures.

Varying the training data size showed useful resilience at 50% of the training set (Dice around 0.81), peak recall at 75%, and sharp degradation below 25%. In practice, this highlights a trade-off: if clinical priorities emphasize sensitivity, training with at least 75% of the available data is preferable, while 50% can be acceptable when computational or annotation budgets are tight.

Optimization behaviour was tightly coupled to learning rate. Momentum-based optimizers (NAG and SGD) excelled at a learning rate of 0.01 (with NAG reaching a best Dice of 0.868), whereas adaptive optimizers dominated at 0.001 (with AdaGrad achieving a best Dice of 0.861). Mean-Dice trajectories confirmed the stability of these regimes: high-learning-rate momentum runs tended to peak earlier, while low-learning-rate adaptive runs improved more gradually.

---


## Page 14

# REFERENCES

[1] A. Mortazi, V. Cicek, E. Keles, and U. Bagci, “Selecting the Best Optimizers for Deep Learning-Based Medical Image Segmentation,” *Frontiers in Radiology*, vol. 3, 2023. Available: https://doi.org/10.3389/fradi.2023.1175473

[2] S. Ruder, “An Overview of Gradient Descent Optimization Algorithms,” *ruder.io*, Jan. 2016. Available: https://www.ruder.io/optimizing-gradient-descent/

[3] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” *arXiv preprint arXiv:1711.05101*, 2017. Available: https://arxiv.org/abs/1711.05101

[4] J. J. Lucido et al., “Validation of clinical acceptability of deep-learning-based automated segmentation of organs-at-risk for head-and-neck radiotherapy treatment planning,” *Frontiers in Oncology*, vol. 13, Apr. 2023. Available: https://doi.org/10.3389/fonc.2023.1137803

[5] Z. Wang, T. Popordanoska, J. Bertels, R. Lemmens, and M. B. Blaschko, “Dice Semimetric Losses: Optimizing the Dice Score with Soft Labels,” *Medical Image Computing and Computer Assisted Intervention – MICCAI 2023*, 2024. Available: https://arxiv.org/abs/2303.16296

[6] K. H. Zou, S. K. Warfield, A. Bharatha, C. M. Tempany, M. R. Kaus, S. J. Haker, W. M. Wells III, F. A. Jolesz, and R. Kikinis, “Statistical validation of image segmentation quality based on a spatial overlap index,” *Academic Radiology*, vol. 11, no. 2, pp. 178–189, Feb. 2004. Available: https://pubmed.ncbi.nlm.nih.gov/14974593

[7] A. Zhang, Z. C. Lipton, M. Li, and A. J. Smola, “Dive into Deep Learning,” *d2l.ai*, 2021. Available: https://d2l.ai/chapter_optimization/lr-scheduler.html

[8] A. Buslaev, V. I. Iglovikov, E. Khvedchenya, A. Parinov, M. Druzhinin, and A. A. Kalinin, “Albumentations: Fast and Flexible Image Augmentations,” *Information*, vol. 11, no. 2, p. 125, 2020. Available: https://doi.org/10.3390/info11020125

[9] C. Shorten and T. M. Khoshgoftaar, “A survey on image data augmentation for deep learning,” *Journal of Big Data*, vol. 6, no. 1, p. 60, 2019. Available: https://doi.org/10.1186/s40537-019-0197-0

[10] D. Jha, P. H. Smedsrud, M. A. Riegler, P. Halvorsen, T. de Lange, D. Johansen, and H. D. Johansen, “Kvasir-SEG: A segmented polyp dataset,” in *MMM 2020*, pp. 451–462, Springer, 2020. Available: https://doi.org/10.1007/978-3-030-37734-2_37

[11] Wikipedia contributors, “Image segmentation,” *Wikipedia, The Free Encyclopedia*, Jul. 2019. Available: https://en.wikipedia.org/wiki/Image_segmentation

[12] IBM, “Image segmentation,” *IBM Think*, Sep. 2023. Available: https://www.ibm.com/think/topics/image-segmentation

[13] K. E. van Keulen, I. S. Papanikolaou, T. W. C. Mak, P. Apostolopoulos, H. Neumann, G. Delconte, M. Furnari, *et al.*, “Comparison of adenoma miss rate and adenoma detection rate between conventional colonoscopy and colonoscopy with second-generation distal attachment cuff: A multicenter, randomized, back-to-back trial,” *Gastrointestinal Endoscopy*, vol. 99, no. 5, pp. 798–808.e3, 2024. Available: https://doi.org/10.1016/j.gie.2023.11.017

[14] S. Zhao, S. Wang, P. Pan, T. Xia, X. Chang, X. Yang, L. Guo, *et al.*, “Magnitude, risk factors, and factors associated with adenoma miss rate of tandem colonoscopy: A systematic review and meta-analysis,” *Gastroenterology*, vol. 156, no. 6, pp. 1661–1674.e11, 2019. Available: https://doi.org/10.1053/j.gastro.2019.01.260

[15] M. Than, J. Witherspoon, J. Shami, P. Patil, and A. Saklani, “Diagnostic miss rate for colorectal cancer: An audit,” *Annals of Gastroenterology*, vol. 28, no. 1, p. 94, 2015. Available: https://pmc.ncbi.nlm.nih.gov/articles/PMC4290010/

[16] “Colon polyp size chart: How doctors classify polyps,” *Medical News Today*, Jul. 2023. Available: https://www.medicalnewstoday.com/articles/colon-polyp-size-chart

[17] L.-C. Chen, Y. Zhu, G. Papandreou, F. Schroff, and H. Adam, “Encoder-decoder with atrous separable convolution for semantic image segmentation,” *arXiv preprint arXiv:1802.02611*, 2018. Available: https://arxiv.org/pdf/1802.02611v3

[18] “PASCAL VOC 2012 dataset,” Kaggle. Available: https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset

[19] “Cityscapes dataset – semantic understanding of urban street scenes.” Available: https://www.cityscapes-dataset.com/

[20] “YOLO Loss Function Part 1: SIoU and Focal Loss,” *LearnOpenCV*, Jan. 2024. Available: https://learnopencv.com/yolo-loss-function-siou-focal-loss

[21] “Understanding Image Resolution,” *Adobe Lightroom Classic – Key Concepts*, 2024. Available: https://helpx.adobe.com/lightroom-classic/lightroom-key-concepts/resolution.html

[22] R. Thompson, “Resolution, Bandwidth, and Noise: Technical Foundations for Digital Imaging,” *MIT Initiative on the Digital Economy*, Sep. 2020. Available: https://ide.mit.edu/wp-content/uploads/2020/09/RBN.Thompson.pdf

[23] “Introduction to PyTorch Tensors,” *PyTorch Tutorials*, Updated Sep. 22 2025. Available: https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html

[24] “High-resolution imaging advances for deep learning systems,” *Nature Communications*, vol. XX, 2025. Available: https://www.nature.com/articles/s41467-025-64679-2
