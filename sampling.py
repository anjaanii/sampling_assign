import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
df = pd.read_csv(url)

# Separate the majority and minority classes
majority_class = df[df['Class'] == 0]
minority_class = df[df['Class'] == 1]

# Upsample the minority class to match the majority class
minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class), random_state=42)

# Combine the majority class with the upsampled minority class
balanced_df = pd.concat([majority_class, minority_upsampled])

# Display the class distribution in the balanced dataset
print(balanced_df['Class'].value_counts())

# Now, 'balanced_df' contains the balanced dataset



# Specify parameters for sample size calculation
confidence_level = 0.95
z_score = 1.96
estimated_proportion = 0.5
margin_of_error = 0.05

# Calculate sample size
sample_size = int((z_score**2 * estimated_proportion * (1 - estimated_proportion)) / margin_of_error**2)

# Create five samples
for i in range(1, 6):
    # Randomly select 'sample_size' instances for each sample
    sample = resample(df, replace=False, n_samples=sample_size, random_state=i)
    
    # Display the class distribution in the sample
    print(f"Sample {i} class distribution:\n{sample['Class'].value_counts()}\n")



# Separate features and target variable
X = df.drop('Class', axis=1)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'M1': RandomForestClassifier(),
    'M2': GradientBoostingClassifier(),
    'M3': SVC(),
    'M4': RandomForestClassifier(),
    'M5': GradientBoostingClassifier(),
}

# Define sampling techniques
samplings = {
    'Sampling1': RandomOverSampler(),
    'Sampling2': RandomUnderSampler(),
    'Sampling3': SMOTE(),
    'Sampling4': ADASYN(),
    'Sampling5': SMOTETomek(),
}

# Apply sampling techniques on models
for sampling_name, sampler in samplings.items():
    print(f"\nApplying {sampling_name}:\n")
    
    # Apply sampling technique to the training data
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    # Train and evaluate each model on the resampled data
    for model_name, model in models.items():
        # Train the model on the resampled data
        model.fit(X_resampled, y_resampled)
        
        # Make predictions on the test set
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Display results
        print(f"Model {model_name} with {sampling_name}:\n")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Classification Report:\n{report}\n")

