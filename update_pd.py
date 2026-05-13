import glob, re
from pathlib import Path

# 1. Decision Tree
f = Path('training/premature_delivery/training_premature_delivery_decision_tree.py')
content = f.read_text(encoding='utf-8')
content = content.replace('from sklearn.ensemble import GradientBoostingClassifier', 'from sklearn.tree import DecisionTreeClassifier')
content = re.sub(r'model = GradientBoostingClassifier\(.*?\)', 'model = DecisionTreeClassifier(random_state=42)', content, flags=re.DOTALL)
content = content.replace('algorithm="gradientboosting"', 'algorithm="decisiontree"')
content = content.replace('label="Gradient Boosting"', 'label="Decision Tree"')
content = content.replace('description="Gradient Boosting classifier', 'description="Decision Tree classifier')
f.write_text(content, encoding='utf-8')

# 2. SVM
f = Path('training/premature_delivery/training_premature_delivery_svm.py')
content = f.read_text(encoding='utf-8')
content = content.replace('LinearSVC', 'SVC') # Fix imports
content = re.sub(r'model = SVC\(.*?\)', "model = SVC(C=10.0, gamma='scale', random_state=42, max_iter=2000, probability=True)", content, flags=re.DOTALL)
content = content.replace('algorithm="linearsvm"', 'algorithm="svm"')
content = content.replace('label="Linear SVM"', 'label="SVM"')
content = content.replace('description="Linear SVM classifier', 'description="SVM classifier')
f.write_text(content, encoding='utf-8')

# 3. Logistic Regression
f = Path('training/premature_delivery/training_premature_delivery_logistic_regression.py')
content = f.read_text(encoding='utf-8')
content = re.sub(r'model = LogisticRegression\(.*?\)', 'model = LogisticRegression(C=10.0, random_state=42, max_iter=1000)', content, flags=re.DOTALL)
f.write_text(content, encoding='utf-8')

# 4. Random Forest
f = Path('training/premature_delivery/training_premature_delivery_randomforest.py')
content = f.read_text(encoding='utf-8')
content = re.sub(r'model = RandomForestClassifier\(.*?\)', 'model = RandomForestClassifier(n_estimators=100, random_state=42)', content, flags=re.DOTALL)
f.write_text(content, encoding='utf-8')

# 5. XGBoost
f = Path('training/premature_delivery/training_premature_delivery_xgboost.py')
content = f.read_text(encoding='utf-8')
content = re.sub(r'model = XGBClassifier\(.*?\)', "model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, min_child_weight=1, random_state=42, eval_metric='logloss')", content, flags=re.DOTALL)
f.write_text(content, encoding='utf-8')
print('Hyperparameters updated.')
