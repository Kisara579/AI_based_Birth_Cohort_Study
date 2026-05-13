import glob
import re
from pathlib import Path

files = Path('training/premature_delivery').glob('*.py')

for f in files:
    if 'prepare' in f.name: continue
    
    content = f.read_text(encoding='utf-8')
    
    # 1. Remove initial train_test_split and fix numeric/categorical extraction
    content = re.sub(r'# Split data into train and test.*?numeric_features = X_train\.select_dtypes', 'numeric_features = X.select_dtypes', content, flags=re.DOTALL)
    content = content.replace('categorical_features = X_train.select_dtypes', 'categorical_features = X.select_dtypes')
    
    # 2. Find model definition
    model_match = re.search(r'(# .*? model\nmodel = .*?\))\n', content, flags=re.DOTALL)
    if not model_match:
        print(f"Model match failed in {f.name}")
        continue
    model_def = model_match.group(1)
    
    # 3. Find the pipeline definition and everything down to pipeline.fit
    pipeline_match = re.search(r'pipeline = Pipeline\(steps=\[.*?pipeline\.fit\(X_train, y_train\)\n', content, flags=re.DOTALL)
    
    if not pipeline_match:
        print(f"Pipeline match failed in {f.name}")
        continue
        
    replacement = f'''{model_def}

# 1. Preprocess ALL data
X_preprocessed = preprocessor.fit_transform(X)

# 2. Apply SMOTE to the entire preprocessed dataset
print("="*80)
print("STEP 1: CLASS BALANCE (BEFORE SMOTE)")
print("="*80)
class_counts_before = y.value_counts().sort_index()
print(f"  Class 0: {{class_counts_before[0]}}")
print(f"  Class 1: {{class_counts_before[1]}}")

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)
X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

import pandas as pd
class_counts_after = pd.Series(y_resampled).value_counts().sort_index()
print(f"\\nAFTER SMOTE: Class 0={{class_counts_after[0]}}, Class 1={{class_counts_after[1]}} (1:1)\\n")

# 3. Split the RESAMPLED data into Train and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 4. Construct final pipeline for artifact saving
from sklearn.pipeline import Pipeline as SklearnPipeline
pipeline = SklearnPipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

# Train model on balanced training split
model.fit(X_train, y_train)
'''
    # Replace the old model definition and pipeline logic with the new one
    content = content.replace(model_def, '')
    content = content.replace(pipeline_match.group(0), replacement)
    
    # Update cross_val_predict to use 'model' instead of 'pipeline'
    content = content.replace("cross_val_predict(\n    pipeline, X_train", "cross_val_predict(\n    model, X_train")
    content = content.replace("cross_val_predict(pipeline, X_train", "cross_val_predict(model, X_train")
    
    # Update test prediction to use 'model' instead of 'pipeline'
    content = content.replace("y_prob = pipeline.predict_proba(X_test)", "y_prob = model.predict_proba(X_test)")
    
    f.write_text(content, encoding='utf-8')
    print(f'Successfully refactored {f.name}')
