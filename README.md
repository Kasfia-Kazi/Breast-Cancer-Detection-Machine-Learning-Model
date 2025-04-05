# ML4641-Project-Team10
`/main/`: Main directory for breast cancer detection project

`/main/README.md`: Description of project, machine learning models used, and results

`/main/project_final.py`: Source code for machine learning models used

`/main/Breast_cancer_data.csv`: Source data for breast cancer detection

## Introduction
Breast cancer (BRC) is the most commonly diagnosed cancer worldwide and has a mortality rate of 2.5% within 10 years of diagnosis. 4% of women with BRC will have metastatic BRC (MBRC) where tumors spread to vital organs, significantly worsening the prognosis and making treatments more challenging [1]. Breast cancer can also be prevented through regular mammography screenings to identify risk of breast cancer since early detection has shown to reduce breast cancer mortality [1, 2]. Machine learning techniques can be used to analyze and identify tumor detection for breast cancer patients. Various differentiation techniques have been used to identify and classify breast cancer tumor behavior like Bayes Network, Pruned Tree, kNN algorithm [3]. Thus, there is a significant need for machine learning algorithms that can identify and detect the presence of cancerous lumps.

## Dataset Description
The breast cancer dataset quantifies the size of cancerous regions. It provides a total of 5 measurements: radius, texture, perimeter, area, smoothness. The diagnosis (cancerous or not) for each data point is also given, so we are able to use supervised learning methods. A total of 569 suspicious masses are recorded in the dataset. The dataset was provided by Dr. William H. Wolberg from University of Wisconsin Hospitals, Madison.

[Dataset link](https://www.kaggle.com/datasets/merishnasuwal/breast-cancer-prediction-dataset)
https://www.kaggle.com/datasets/merishnasuwal/breast-cancer-prediction-dataset

## Problem Definition
Breast cancer is diagnosed by observing abnormal lumps found in the breast, and then deciding if the lump is cancerous. However, it is not always clear whether a lump is cancerous or not, and doctors may misdiagnose patients, either leaving cancer untreated or having a patient undergo unnecessary treatment.  As such, if we can identify whether or not a lump is cancerous based by measuring its features, this may lead to simplification in the process of diagnosing breast cancer.

## Data Preprocessing Methods
Prior to implementing our models, we first sought to preprocess our data. For ridge regression, since we wanted to ensure that the regularization penalty for ridge regression (taking the square) was applied uniformly so that no feature dominated the others, we standardized our training data (X values). We then ran ridge regression on our standardized data using alpha values from 0.5 to 1.0 (which affects the degree of regularization penalty applied). 

Prior to running logistic regression and support vector machine, we wanted to implement dimension reduction in case there were any features that were correlated with each other. To do this, we performed Principal Component Analysis on our data, reducing our five features to 3 principal components. However, since there is a trade-off between information loss and dimensionality reduction, as the principal components are not easy to interpret (if the results are good, we won’t know which features specifically were of the most use), we decided to run Logistic Regression and SVM on both our standardized data without applying PCA and on our standardized data with PCA. This would also allow us to directly see what impact PCA had (improved accuracy, etc).

## ML Model 1: Ridge Regression
Our purpose is to predict whether or not a breast lump/mass is cancerous or noncancerous. Since our data gives us the diagnosis for each patient (Y values/labels), we are able to use supervised learning to predict the outcome for each patient. While there is a wide array of algorithms for supervised learning, we choose to use Ridge Regression over Linear Regression due to the fact that linear regression is prone to overfitting when coefficients are large. We also decided to use ridge regression over lasso regression as ridge regression finds a closed-form solution. 
### Model Performance

|||
:-------------------------:|:-------------------------:
![image](https://github.com/user-attachments/assets/d6d83482-cbe8-44e0-b11e-ec832c78c9ca) | ![image](https://github.com/user-attachments/assets/56e93a9e-85b6-4f56-93be-7ba5c447ea32) |
![image](https://github.com/user-attachments/assets/8cb0e80c-708e-4c43-8c72-5d7903b0d8fb)
|![image](https://github.com/user-attachments/assets/4653eb2a-6c56-4f4e-b7fe-ba013ebcab9c)

### Results/Discussion
The labeling for cancerous vs noncancerous regions are binary as shown by the residual plots and prediction visualization, thus indicating the use of a classification model to further determine pairings would be more optimal. The heatmap of the feature correlations to the diagnosis indicates the possible feature reduction of redundant features: mean_texture and mean_smoothness due to their correlation values lying closer to zero than the other features. Their respective ridge regression coefficient values are also of small magnitude, indicating the low impact of these features on the diagnosis. Similarly, according to the ridge regression model mean_radius may also be a feature with low impact on the diagnosis of the lump. The actual vs predicted graph displays the model has a somewhat accurate prediction of diagnoses, with the predicted values centering around the true value. However, the model has a wide range of predicted values that requires further tuning of the model and classification to improve the predicted diagnosis.

To evaluate our regression model, we calculated the mean squared error and the root mean squared error. The mean squared error is the expected value of the squared error loss, and we want to minimize this value which would minimize error in our model. We also used cross validation to determine the best alpha value, which is the value that minimizes our MSE, by checking 100 potential alpha values within the range (0, 5). However, cross-validation found the best alpha value to be 1.0, where the difference in the MSE was very small (compared to alpha=0.5). The MSE at alpha of 1.0 was 0.0827 and the RMSE was 0.2877.

## ML Model 2: Logistic Regression
The previous model indicated a binary classification would better suit our dataset to predict whether or not a breast lump/mass is cancerous or noncancerous. While there is a wide array of algorithms for supervised learning, we choose to use Logistic Regression due to its suitability for binary classification problems, which aligns perfectly with our goal of predicting a binary outcome (cancerous or noncancerous). Logistic Regression models the probability of the default class (cancerous) and provides clear probabilities for each class, making it easier to interpret. Additionally, it assumes a linear relationship between the features and the log odds of the outcome, which often fits well with medical data. This makes Logistic Regression a robust and interpretable choice for our cancer prediction model.

### Model Performance

Without PCA |  With PCA (3 components)
:-------------------------:|:-------------------------:
![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/76d8ac23-2b44-4921-8fa1-9eb4d538345c) | ![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/11ee98b8-0b6b-430b-b5bf-0c1095fe4fa8)

Visualization With PCA (3 components)|Visualization With PCA (2 components)|
:-------------------------:|--|
![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/c9085322-0a39-4a17-8198-029fdab93ce3)|![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/991f11cf-47e2-4144-b81b-ad79c5e02640)

The logistic model with PCA has a higher accuracy (0.95) compared to the logistic model without PCA feature reduction (0.91). This indicates PCA has potentially improved the model's ability to correctly classify the instances. For Class 0, the precision is slightly lower with PCA (0.89 vs. 0.98) but recall is higher (0.94 vs. 0.83). This suggests PCA might have made the model more sensitive to detecting Class 0 instances, at the expense of precision. For Class 1, both precision and recall are improved with PCA (0.97 vs. 0.87 and 0.95 vs. 0.98, respectively). This indicates PCA has improved the model's ability to correctly identify Class 1 instances.

The ROC (Receiver Operating Characteristic) curve evaluates the performance of a logistic regression model by plotting the trade-off between the True Positive Rate (sensitivity) and the False Positive Rate across different thresholds. The AUC (Area Under the Curve) provides a single value that summarizes the model's overall ability to distinguish between classes, with a higher AUC indicating better performance. The ROC AUC score of 0.99 for the logistic regression with PCA indicates an excellent model performance in distinguishing between classes in the test dataset. The closer the ROC AUC is to 1, the better the model is at classifying the positive and negative classes. The PR AUC score of 0.99 was achieved on the test set, so it indicates that the model is not only fitting the training data well but also generalizing effectively to new data. In this case, the high PR AUC is a positive indicator of model performance and not a sign of overfitting.

## ML Model 3: Support Vector Machine
We choose to use Support Vector Machine (SVM) due to its effectiveness in high-dimensional spaces and its robustness in handling binary classification tasks. SVM works well when there is a clear margin of separation between classes and is particularly effective in cases where the number of features exceeds the number of samples. By finding the optimal hyperplane that best separates the classes, SVM maximizes the margin between the data points of different classes, reducing the risk of overfitting. Additionally, the use of kernel functions allows SVM to handle non-linear relationships, providing flexibility and power in our cancer prediction model.

We tried using 4 different kernels for our SVM model: Radial Basis Function (RBF), Linear, Polynomial, and Sigmoid. The RBF kernel is effective for capturing complex, nonlinear relationships; the Linear kernel is simple and efficient for linearly separable data; the Polynomial kernel can model interactions of varying degrees between features; and the Sigmoid kernel is akin to a neural network's activation function, making it versatile for certain nonlinear problems. By evaluating all these kernels, we aimed to identify the one that best captures the underlying patterns in our data, thereby enhancing the model's accuracy and robustness.

|Results|  Without PCA |  With PCA (3 components) 
:-------------------------:|:-------------------------:|:-------------------------:
Radial Basis Function |![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/f513d8ce-ffad-4753-9513-0c7e1f2fa814) | ![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/d3f4029a-3cc2-4a42-8304-d293c5ca5ea7) 
Linear | ![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/2f2b3985-1e83-4c95-96d4-06f90a5f4052) | ![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/a51ea987-dc85-4739-ad0d-d892ac6a32b5)
Polynomial |![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/465fe8a5-84af-4c31-b4c3-182f258a9f61)|![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/ffcbcfc4-bed7-4382-b31d-c1c35e372374)
Sigmoid|![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/43efd1d0-d2e8-4616-ba36-7834bd636998)|  ![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/ae9333cd-96e5-493e-9324-dd250df03573)

|| Visualization Without PCA|
:-------------------------:|:-------------------------:|
Radial Basis Function |![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/9f0bfa33-0190-4657-8fae-90f023a9ccc1)
Linear | ![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/94d7a41b-b43c-4c9c-bf90-5bb36bac8fe3)
Polynomial |![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/90a4ac8b-c487-4dbc-9986-610b7b4d9a8c)
Sigmoid|![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/66c4cbc8-fc6d-4dc7-af72-40e7804d347a)

||  Visualization With PCA (2 components) | Results
:-------------------------:|:-------------------------:|:-------------------------:
Radial Basis Function |![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/12c2402d-1e90-4c1b-8299-bd1724e335d1) |![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/ea81eaaa-7822-4d0a-924f-c84ef3adfdd9)
Linear | ![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/5dd8cb1a-f506-48ca-a34f-ef9a899cd09c) | ![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/0b575c90-dc8b-4969-9301-140c4df1352f)
Polynomial |![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/eb588489-0ec2-45c3-8566-e7cc99a335c9) | ![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/8c81b808-fb84-4d13-872d-922caf284399)
Sigmoid|![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/aa522471-202d-41fa-b1ba-7716c48c6b78) | ![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/82155cb3-9925-485e-985d-9a827af06204)

|||
|--|---|
![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/9e44e701-7b6a-4dbb-9f53-a059db3cc301)|![image](https://github.gatech.edu/jhou73/ML4641-Project-Team10/assets/54690/865043f1-9245-4213-81d8-e57328f35dbf)


After optimizing gamma and the regularization parameters, RBF performed the best alongside Sigmoid (accuracy of 0.9649), with Polynomial being the second best (accuracy of 0.9561) and Linear being the worst (accuracy of 0.9473). Generally, these models performed better prediction with a lower gamma and higher regularization.
We also ran SVM on our data after it had been preprocessed using PCA. Our SVM models actually performed worse with the reduced features. Accuracy was still above 90%, but dropped around 3-5% from without PCA for the varying kernels. This decline suggests that the reduction in the number of features resulted in a loss of important information, making the SVM models less effective.  This analysis underscores the importance of selecting appropriate kernels for SVM and considering the trade-offs involved in dimensionality reduction techniques like PCA.

## ML Model Comparison
The SVM models using the Radial Basis Function (RBF) and Sigmoid kernels without PCA achieved the highest accuracy, both reaching 0.9649, indicating their superior performance compared to the other models. The Polynomial kernel without PCA followed closely with an accuracy of 0.9561, while the Linear kernel without PCA had a slightly lower accuracy of 0.9474. Logistic Regression, without PCA achieved an accuracy of 0.9122 and with PCA, achieved an accuracy of 0.9474, demonstrating robust performance but trailing behind SVM. Ridge Regression, with an MSE of 0.0828 and RMSE of 0.2877, showed relatively modest performance in comparison. Applying PCA improved the accuracy of the logistic regression model but generally reduced the accuracy of SVM models, suggesting that dimensionality reduction led to a loss of critical information. Overall, SVM models with RBF and Sigmoid kernels performed best without PCA, while logistic regression's accuracy improved with PCA, highlighting the trade-offs between feature reduction and model performance.

## Next Steps
Our immediate next step would be to utilize another classification method, likely Random Forest, on our data and compare the results. We want to use a classification method as our Y-labels are discrete, represented by 1 or 0 (cancerous or not) in our data, which would suit any classification method over regression, which is better for continuous labels. Ultimately, we would like to improve our accuracy as much as possible, while still avoiding overfitting.

In the future we hope to expound upon our current investigation into breast cancer by generalizing to other unidentified masses and cancers. In particular, we would like to aid the diagnosis process of other female-oriented cancers, such as ovarian cancer, which often take longer to recognize and diagnose.


### References 
[1] A. N. Giaquinto et al., “Breast Cancer Statistics, 2022,” CA: A Cancer Journal for Clinicians, vol. 72, no. 6, Oct. 2022, doi: https://doi.org/10.3322/caac.21754.
‌

[2] S. A. Mohammed, S. Darrab, S. A. Noaman, and G. Saake, “Analysis of Breast Cancer Detection Using Different Machine Learning Techniques,” Data Mining and Big Data, pp. 108–117, 2020, doi: https://doi.org/10.1007/978-981-15-7205-0_10.


[3] S. Sharma, A. Aggarwal, and T. Choudhury, “Breast Cancer Detection Using Machine Learning Algorithms,” 2018 International Conference on Computational Techniques, Electronics and Mechanical Systems (CTEMS), Dec. 2018, doi: https://doi.org/10.1109/ctems.2018.8769187.


## Gantt Chart
[Gantt Chart](https://docs.google.com/spreadsheets/d/1AL4n2p8k2Mc5V8-by4T80Ta2D8M54iaX/edit?usp=sharing&ouid=103889397454760625638&rtpof=true&sd=true)
https://docs.google.com/spreadsheets/d/1AL4n2p8k2Mc5V8-by4T80Ta2D8M54iaX/edit?usp=sharing&ouid=103889397454760625638&rtpof=true&sd=true

## Video
[Final Report](https://www.youtube.com/watch?v=CKlv4HODdf4)
https://www.youtube.com/watch?v=CKlv4HODdf4

[Proposal](https://youtu.be/P_NKG6ySoxM)
https://youtu.be/P_NKG6ySoxM

## Contribution Table (Final Report)
| Team Member | Contributions |
|-------------|---------------|
|Rebecca Sun   | Implemented PCA, Logistic Regression with and without PCA, described Data Preprocessing Methods, quantitative metrics for SVM w/o PCA, Next Steps, Video, Midterm Corrections (Dataset Description, Problem definition, Ridge Regression)|
|Jenny Hou | Implemented SVM (with and without PCA), SVM Quantitative Metrics w/ PCA, Discussion of Results, Midterm Corrections (Dataset Description)|
|Kasfia Kazi | Logistic Regression: Implemented model, Quantitative Metrics w/ and w/o PCA, Visualizations, and Discussion of Results. SVM: Visualisations, and Discussion of Results. Other: ML Model Comparison, Midterm Corrections (Visualizations)|

## Contribution Table (Midterm Checkpoint)
| Team Member | Contributions |
|-------------|---------------|
| Rebecca Sun | Method description, Preprocessing (standardization), Running Ridge Regression, Calculating MSE, RMSE, Next Steps |
| Jenny Hou | Running Ridge Regression, Potential Results and Discussion |
| Kasfia Kazi | Running Ridge Regression, Visualization, Quantitative Metrics, Potential Results and Discussion |

## Contribution Table (Proposal)
| Team Member | Contributions |
|-------------|---------------|
| Rebecca Sun  | Methods, Dataset, Gantt Chart, Contribution Table, Slides, Video Creation and Youtube Upload | 
| Jenny Hou  | Github Management, Contribution Table, Slides, Problem Definition, Potential Results and Discussion  | 
| Kasfia Kazi  | Introduction, Potential Results and Discussion, Methods, Slides, References  |

