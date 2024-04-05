# AI-Human Generated Classifier

### By Itamar Kraitman & Haim Goldfisher

## Introduction
This project is aimed at developing an AI-powered classifier capable of distinguishing between human-generated and AI-generated text.

## Prerequisites
- Python 3.x installed on your system.

## Setup Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/haimgoldfisher/Ai-Human-Generated.git
    ```

2. Create and activate a virtual environment (recommended):
    ```bash
    # On macOS/Linux
    python3 -m venv venv
    source venv/bin/activate

    # On Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3. Install dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application
To run the main application, execute the following command:
```bash
python main.py
```

After running the program, you will be prompted to enter text. Please enter your text when prompted and press Enter to classify whether the text was written by a human or AI.


## Background

### Preprocessing

The preprocessing module enhances textual data by extracting various features essential for NLP. These features include calculating punctuation percentage, evaluating tone, conducting sentiment analysis, measuring the richness of text, determining the percentage of specific punctuation, POS tagging, entity recognition, computing readability indices, assessing syntactic complexity, and averaging sentence length. This process returns the dataset with additional columns containing the extracted features.
We utilize NLTK (Natural Language Toolkit) for tokenization, POS tagging, and stopwords handling. Additionally, SpaCy is employed for entity recognition tasks. These libraries, along with pandas for data manipulation, enable us to extract various features from textual data efficiently.
The preprocessing module extracts the following features from textual data:
1. Punctuation Percentage
2. Tone Evaluation
3. Sentiment Analysis (Positive, Negative, Neutral)
4. Richness of Text
5. Character Percentage (Comma, Period, Question Mark)
6. POS Tagging (Noun, Pronoun, Verb, Adjective, Adverb, Preposition, Conjunction, Interjection)
7. Entity Recognition
8. Entity Ratio
9. Stopwords Percentage
10. Uppercase Percentage
11. Text Length
12. Word Count
13. Readability Indices (Flesch Reading Ease, Flesch-Kincaid Grade Level, Gunning Fog Index)
14. Syntactic Complexity
15. Average Sentence Length

### Train-Validation-Test Split

We partition our dataset into three subsets: training, validation, and test sets. Initially, we divide the entire dataset into training and the rest using an 80-20 split. Then, we further split the remaining data equally into validation and test sets, each comprising 50% of the remaining dataset. During this process, we ensure to remove the "generated" column from all subsets to separate the input features from the labels. This systematic splitting strategy ensures that we have distinct datasets for training, validation, and testing, enabling robust evaluation of our models' performance.

### Standarization

After extracting the features from the textual data, we apply standardization using the StandardScaler from Scikit-Learn. Standardization rescales the features to have a mean of 0 and a standard deviation of 1, ensuring that each feature contributes equally to the analysis. This process is essential for models that rely on distance-based algorithms or regularization techniques, as it prevents features with larger scales from dominating the model (e.g., "Average Sentence Length" values are always greater than 1). The StandardScaler is used to transform each feature independently, preserving the shape of the distribution while centering it around the mean. By standardizing the features, we ensure better model performance and stability across different datasets.
Furthermore, it's important to note that the standardization process is performed solely on the training set. Once the scaler is fit to the training data and the features are transformed, the same scaler is then applied to the validation and test sets. This ensures consistency in scaling across all datasets, maintaining the integrity of the model evaluation process.

### Regularization

We employ L1 regularization with linear regression solely to eliminate unnecessary columns from the dataset. L1 regularization, also known as Lasso, introduces a penalty term proportional to the absolute value of the coefficients. By doing so, it encourages sparse solutions, effectively driving some coefficients to zero. In our case, applying L1 regularization helps to automatically select relevant features while discarding irrelevant ones. This process aids in reducing overfitting and enhancing model interpretability by simplifying the feature set to only include those with significant predictive power.

### Vectorization

We utilize the "Doc2Vec" algorithm from the Gensim library to translate our text data into fixed-size vectors of dimension 50. Initially, we train the "Doc2Vec" model exclusively on the training dataset. Subsequently, we apply the trained model to generate vectors for the text in the training, validation, and test datasets using the infer_vector method. This allows us to remove the original "text" column from our datasets, thus reducing dimensionality. Additionally, we expand the single vector into 50 separate columns, enabling us to incorporate the vectorized text features into our model training process.

### Modeling

In this section, we explore various classification algorithms to train predictive models on our dataset. Here's an overview of the process:

1. Model Selection:
   - We consider a range of classification models, including Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), AdaBoost, Decision Tree, and Random Forest.

2. Hyperparameter Tuning:
   - Each model undergoes hyperparameter tuning using grid search cross-validation (GridSearchCV).
   - Hyperparameters and their corresponding parameter grids are predefined for optimization.

3. Classification Pipeline:
   - We define a `classification_pipeline` function to streamline model evaluation.
   - This function performs grid search cross-validation to find the best hyperparameters and evaluates model performance on the validation set.

4. Evaluation Metrics:
   - Standard classification metrics such as precision, recall, F1-score, and area under the ROC curve (AUC) are utilized.
   - These metrics provide insights into predictive performance and class differentiation ability.

5. Results Visualization:
   - Classification reports, confusion matrices, and ROC curves are visualized for each model to interpret performance.
   - The classification report offers detailed class metrics, while the confusion matrix illustrates prediction outcomes.
   - The ROC curve depicts the true positive rate vs. false positive rate trade-off across thresholds.

6. Model Selection and Training:
   - Based on validation results, the best-performing model (e.g., SVM with optimized hyperparameters) is selected.
   - The chosen model is trained on the entire training dataset for deployment.
   - In order to address concerns regarding AI-generated text, we establish a threshold of 60%. If the model predicts with a probability greater than 60% that a text was generated by AI, it will classify it as AI-generated; otherwise, it will classify it as human-generated. This thresholding approach aims to minimize the risk of falsely attributing AI-generated texts to humans, thereby ensuring fair and responsible use of the model.












