

PROJECT TEAL





Jenny Fish

HELLO!

We are Project Teal

2





PROJECT TEAL

Jenny Fish

MEET THE PROJECT TEAL TEAM

Jenny

Fish

Isha

Adam

Claudy

Samiha

Khan

Sandeep

Pvn

Mehar

Chaturvedi

(7) Results

Angadi

(1) Big Picture

(2) Background

(3) Research

(4) Model

(5) Experiments

(6) Code

(8) Future Work

3





PROJECT TEAL

Jenny Fish

1

WHAT’S THE BIG PICTURE?

Letʼs start with the high level overview

4





PROJECT TEAL

Jenny Fish

PROJECT OVERVIEW

◉ The goal is to build a model that accurately predicts

malignant or benign tumors.

◉ Our Base Model is a research paper, which analyzed data to

find out if someone is at serious risk of Ovarian Cancer

based on their 49 biomarkers and non-biomarkers.

◉ We sought to improve the accuracy of the base model by

tuning various hyperparameters.

5





PROJECT TEAL

Jenny Fish

2

BACKGROUND: OVARIAN CANCER

Social Impact

6





PROJECT TEAL

Jenny Fish

21,000Diagnoses

14,000 Deaths!

Source: https://www.cdc.gov/cancer/ovarian/statistics/index.htm

7





PROJECT TEAL

Jenny Fish

OVARIAN CANCER IMPACTS

◉ Oꢀen asymptomatic until later stages (25% detected at Stage I)

◉ Diagnosed early - 90% survival rate

◉ Later stages, very low survival rate

◉ CA125, HE4, CEA are common biomarkers associated with Ovarian

Cancer

◉ CA125 considered a gold standard biomarker

◉ Current diagnosis algorithm — ROMA test (based on CA125 and HE4)

8





PROJECT TEAL

Isha Angadi

3

RESEARCH

Ovarian Cancer Scientific Information

9





PROJECT TEAL

Isha Angadi

OVARIAN CANCER STUDY (PAPER)

“Using Machine Learning to Predict Ovarian Cancer” by Lu, Fan, et al.

Published: International Journal of Medical Informatics

Aim:

◉ To improve the accuracy of early diagnosis and detection of

ovarian cancer using machine learning feature selection

method — MRMR to build decision tree.

Data:

◉ 171 OC patients and 178 BOT patients, 49 features

◉ Train/Test split — 235/114 values

Source: https://www.sciencedirect.com/science/article/pii/S1386505620302781

10





Isha Angadi

“Using Machine Learning to Predict Ovarian Cancer” Process

11





PROJECT TEAL

Isha Angadi

OVARIAN CANCER STUDY (PAPER)

“Using Machine Learning to Predict Ovarian Cancer” by Lu, Fan, et al.

Published: International Journal of Medical Informatics

Procedure:

◉ Handling missing data

◉ Using MRMR feature reduction,

◉ Building a decision tree model.

◉ Performing cross validation.

◉ Produce confusion matrix and accuracies.

Results:

◉

CEA and HE4 have the most significant prediction power when it comes

to the classification of ovarian cancer vs the benign ovarian tumors.

Source: https://www.sciencedirect.com/science/article/pii/S1386505620302781

12





PROJECT TEAL

Adam Claudy

4

BUILDING OUR MODEL

Comparing Research Model with Ourʼs

13





PROJECT TEAL

Adam Claudy

PROJECT PIPELINE

●

●

Handling object data type

Handling columns with a given

missing rate tolerance.

01

02

03

04

DATA PREPROCESSING

●

Handling missing data

●

●

Impurity measure

Depth

DECISION TREE MODEL

EXPERIMENTS

RESULTS

●

3 Model Versions

●

●

●

Visualizations

Confusion Matrices

Metrics

14





PROJECT TEAL

Adam Claudy

DATA PREPROCESSING

◉ Convert all feature columns into numeric form.

◉ Data is missing at random (MAR)

◉ Remove columns which exceed the specified missing rate

tolerance. (25%, 50%)

◉ 2 biomarkers removed (CA72-4, NEU)

◉ Impute NAs with mean, median or mode.

15





PROJECT TEAL

Adam Claudy

SHOWING DATA SKEWNESS - MEAN VS MEDIAN

16





PROJECT TEAL

Adam Claudy

4000

3000

2000

1000

0

ADAMʼS DATA VISUALIZATIONS

[Features](https://drive.google.com/file/d/1bYCHav6RJBGW8n-l5Y21hcK8ZAyyZoDZ/view?usp=sharing)[ ](https://drive.google.com/file/d/1bYCHav6RJBGW8n-l5Y21hcK8ZAyyZoDZ/view?usp=sharing)[Histogram](https://drive.google.com/file/d/1bYCHav6RJBGW8n-l5Y21hcK8ZAyyZoDZ/view?usp=sharing)

17





PROJECT TEAL

Adam Claudy

FEATURE SELECTION

Why do we need feature selection?

◉ Base Model reduced features using Minimum Redundancy -

Maximum Relevance (MRMR) (from 48 to 8).

◉ Experiment using all features to test if feature selection is

required.

18





PROJECT TEAL

Adam Claudy

DECISION TREE MODEL

Hyperparameters

◉ Impurity Measure

◉ Gini

◉ Entropy

◉ Depth of tree

19





PROJECT TEAL

Samiha Khan

5

EXPERIMENTS

20





PROJECT TEAL

Samiha Khan

EXPERIMENT VARIATIONS

Stratified k-cross

Validation :

1

3

True or False

Feature Selection :

2

4

8 (selected by MRMR) or

all features

Shuﬀle Data :

True or False

Imputation of Missing Data :

Mean, Mode, or Median

DT Impurity :

gini or entropy

5

21





PROJECT TEAL

Samiha Khan

EXPERIMENT OUTPUTS

● Confusion Matrix

● Specificity

Sensitivity

● PPV

● NPV

● Overall Accuracy

● F1 Score

● Mean Stratified Cross Validation Accuracy

● Teal Score

22





PROJECT TEAL

Sandeep Pvn

6

CODE

Metrics Insight and Code

23





PROJECT TEAL

Sandeep Pvn

CODE

Jupyter Notebook:

[https://colab.research.google.com/drive/12dhDfeTJQj8N](https://colab.research.google.com/drive/12dhDfeTJQj8NSfUnsfsw06HlpoqOjQcy#scrollTo=00rR7B5NwI2J)

[SfUnsfsw06HlpoqOjQcy#scrollTo=00rR7B5NwI2J](https://colab.research.google.com/drive/12dhDfeTJQj8NSfUnsfsw06HlpoqOjQcy#scrollTo=00rR7B5NwI2J)

24





PROJECT TEAL

Sandeep Pvn

METRICS

Confusion Matrix

Objective : To reduce FP

Actual

BOT

OC

◉

We take 2 metrics,

specificity and precision

into account.

We combine the metrics

into one score, the Teal

score.

Predict

BOT

TP

FN

FP

TN

◉

OC

25





PROJECT TEAL

Mehar Chaturvedi

7

RESULTS

Model Results

26





PROJECT TEAL

Mehar Chaturvedi

RESULTS

◉ Why Stratified k-cross validation is required?

◉ Feature selection

◉ Why Shuﬀling is required?

◉ What do we mean by shuﬀling?

◉ Which impute method is better and why?

CLASS IMBALANCE!!

Overfitting for OC class

Data

Training -

Testing -

37.8% BOT class

62.2 % OC class

78.2 % BOT class

21.8 % OC class

27





PROJECT TEAL

RESULTS : Confusion Matrix (Shuﬄe)

Mehar Chaturvedi

Before Shuﬀling : Paper

Before Shuﬀling : Teal

Actual

Predicted

Actual

Predicted

BOT

OC

BOT

OC

BOT

OC

80

9

0

BOT

OC

76

13

0

25

25

Aꢀer Shuﬀling : Paper

Aꢀer Shuﬀling : Teal

Actual

Predicted

Actual BOT

OC

BOT

OC

Predicted

BOT

43

8

13

BOT

47

4

13

OC

50

OC

50

28





PROJECT TEAL

RESULTS : Confusion Matrix (Impute Methods)

Mehar Chaturvedi

\*Aꢀer Stratified-K-Cross Validation and Shuﬀling

Mode

Mean

BOT

Median

Actual

Actual

Predicted

BOT

45

OC

16

Actual

Predicted

OC

13

BOT

43

OC

13

Predicted

BOT

OC

BOT

OC

47

4

BOT

OC

6

47

50

8

50

Actual

Mean

Median

Mode

Predicted

Teal Score

Precision

Specificity

0.9798

0.783

0.794

0.9788

0.768

0.794

0.9787

0.738

0.746

29





PROJECT TEAL

Mehar Chaturvedi

RESULTS

30





PROJECT TEAL

Mehar Chaturvedi

RESULTS METRICS

PAPER

TEAL

31





PROJECT TEAL

Mehar Chaturvedi

DECISION TREE COMPARISON

PAPER

TEAL

●

The tree achieves the best mean

cross-validation accuracy 87.65957 +/-4.73852

% on training dataset

32





PROJECT TEAL

Mehar Chaturvedi

CONFUSION MATRIX COMPARISON

PAPER

TEAL

33





PROJECT TEAL

Sandeep Pvn

8

FUTURE WORK

Neural Networks

34





PROJECT TEAL

Sandeep Pvn

FUTURE WORK AND SUGGESTIONS

● Customization: run the model on any generalized data set

○ implement customizing imputing techniques for each column

○ Try to obtain and use genetic data

● Gini vs. Entropy

● Grid search: Increase code eﬀiciency and compute the optimum

values of hyperparameters.

● Neural Network: Running the model through a neural network to

improve the accuracies.

● Analyse and predict if and when BOT converts to OC

○ Change is system. Need Time Series Data

35





PROJECT TEAL

Sandeep Pvn

FUTURE WORK AND SUGGESTIONS

●

●

Run the model on any generalized data

set

Implement customizing imputing

techniques for each column

01

02

03

04

05

Customization

Genetic Data

●

●

Try to obtain and use genetic data

Increase code eﬃciency and compute

the optimum values of

hyperparameters.

Grid search &

Pipelining

●

●

Use pipelining to speed up

Running the model through a neural

network to improve the accuracies.

Neural Network

BOT to OC

●

●

Analyse and predict if and when BOT

converts to OC

Change is system. Need Time Series

Data

36





PROJECT TEAL

9

QUESTIONS FOR PROF. PREM

37





PROJECT TEAL

QUESTIONS FOR REFLECTION

◉ When we shuﬀle our data, it is making a very big diﬀerence — Why

does shuﬀling makes such a big diﬀerence with our results?

◉ Why is mean giving a better result than median and mode?

38





PROJECT TEAL

THANKS!

Any questions?

39





SOURCES

◉ Lu, M., Fan, Z., Xu, B., Chen, L., Zheng, X., Li, J., Znati, T., Mi,

Q. and Jiang, J., 2021. Using machine learning to predict

ovarian cancer.

[https://www.sciencedirect.com/science/article/pii/S138650](https://www.sciencedirect.com/science/article/pii/S1386505620302781)

[5620302781](https://www.sciencedirect.com/science/article/pii/S1386505620302781)

40

