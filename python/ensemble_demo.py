#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ensemble Learning Demonstration

This script demonstrates the new homogeneous and heterogeneous ensemble methods
added to the Quantra ensemble learning system.
"""

import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from ensemble_learning import EnsembleFactory, HomogeneousEnsemble, HeterogeneousEnsemble
except ImportError:
    print("Please run this script from the python directory of the Quantra project")
    exit(1)


def demonstrate_homogeneous_ensembles():
    """Demonstrate homogeneous ensemble methods"""
    print("=" * 60)
    print("HOMOGENEOUS ENSEMBLE DEMONSTRATION")
    print("=" * 60)
    
    # Create regression dataset
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Base model for comparison
    base_model = DecisionTreeRegressor(random_state=42)
    base_model.fit(X_train, y_train)
    base_preds = base_model.predict(X_test)
    base_rmse = np.sqrt(mean_squared_error(y_test, base_preds))
    
    print(f"\nBase Decision Tree RMSE: {base_rmse:.4f}")
    print("\nHomogeneous Ensemble Results:")
    print("-" * 40)
    
    # 1. Bagging Ensemble
    bagging = EnsembleFactory.create_bagging_ensemble(
        base_estimator=DecisionTreeRegressor(random_state=42),
        n_estimators=10,
        max_samples=0.8,
        random_state=42
    )
    bagging.fit(X_train, y_train)
    bagging_preds = bagging.predict(X_test)
    bagging_rmse = np.sqrt(mean_squared_error(y_test, bagging_preds))
    print(f"Bagging (10 trees, 80% samples): {bagging_rmse:.4f}")
    
    # 2. Random Subspace Method
    random_subspace = EnsembleFactory.create_random_subspace_ensemble(
        base_estimator=DecisionTreeRegressor(random_state=42),
        n_estimators=10,
        max_features=0.6,
        random_state=42
    )
    random_subspace.fit(X_train, y_train)
    rs_preds = random_subspace.predict(X_test)
    rs_rmse = np.sqrt(mean_squared_error(y_test, rs_preds))
    print(f"Random Subspace (60% features): {rs_rmse:.4f}")
    
    # 3. Pasting (without replacement)
    pasting = HomogeneousEnsemble(
        base_estimator=DecisionTreeRegressor(random_state=42),
        n_estimators=10,
        ensemble_method='pasting',
        max_samples=0.7,
        bootstrap=False,
        random_state=42
    )
    pasting.fit(X_train, y_train)
    pasting_preds = pasting.predict(X_test)
    pasting_rmse = np.sqrt(mean_squared_error(y_test, pasting_preds))
    print(f"Pasting (70% samples, no replacement): {pasting_rmse:.4f}")
    
    # 4. Extra Trees
    extra_trees = HomogeneousEnsemble(
        base_estimator=DecisionTreeRegressor(random_state=42),
        n_estimators=10,
        ensemble_method='extra_trees',
        max_features=0.8,
        random_state=42
    )
    extra_trees.fit(X_train, y_train)
    et_preds = extra_trees.predict(X_test)
    et_rmse = np.sqrt(mean_squared_error(y_test, et_preds))
    print(f"Extra Trees (80% features): {et_rmse:.4f}")
    
    print(f"\nImprovement Summary:")
    print(f"Best ensemble improvement: {((base_rmse - min(bagging_rmse, rs_rmse, pasting_rmse, et_rmse)) / base_rmse * 100):.1f}%")


def demonstrate_heterogeneous_ensembles():
    """Demonstrate heterogeneous ensemble methods"""
    print("\n" + "=" * 60)
    print("HETEROGENEOUS ENSEMBLE DEMONSTRATION")
    print("=" * 60)
    
    # Create regression dataset
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train diverse base models
    models = []
    model_names = []
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    models.append(rf)
    model_names.append("Random Forest")
    
    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=50, random_state=42)
    gb.fit(X_train, y_train)
    models.append(gb)
    model_names.append("Gradient Boosting")
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    models.append(lr)
    model_names.append("Linear Regression")
    
    # Elastic Net
    en = ElasticNet(alpha=0.1, random_state=42)
    en.fit(X_train, y_train)
    models.append(en)
    model_names.append("Elastic Net")
    
    # Individual model performance
    print("\nIndividual Model Performance:")
    print("-" * 40)
    individual_rmses = []
    for model, name in zip(models, model_names):
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        individual_rmses.append(rmse)
        print(f"{name:16}: {rmse:.4f}")
    
    print(f"\nBest individual RMSE: {min(individual_rmses):.4f}")
    
    # Heterogeneous ensemble methods
    print("\nHeterogeneous Ensemble Results:")
    print("-" * 40)
    
    # 1. Simple Average
    simple_avg = HeterogeneousEnsemble(
        models=models,
        ensemble_method='simple_average',
        task_type='regression'
    )
    simple_preds = simple_avg.predict(X_test)
    simple_rmse = np.sqrt(mean_squared_error(y_test, simple_preds))
    print(f"Simple Average: {simple_rmse:.4f}")
    
    # 2. Weighted Average
    weighted_avg = HeterogeneousEnsemble(
        models=models,
        ensemble_method='weighted_average',
        task_type='regression'
    )
    weighted_preds = weighted_avg.predict(X_test)
    weighted_rmse = np.sqrt(mean_squared_error(y_test, weighted_preds))
    print(f"Weighted Average: {weighted_rmse:.4f}")
    
    # 3. Stacking
    stacking = EnsembleFactory.create_stacking_ensemble(
        models=models,
        cv_folds=5,
        task_type='regression'
    )
    stacking.fit(X_train, y_train)
    stacking_preds = stacking.predict(X_test)
    stacking_rmse = np.sqrt(mean_squared_error(y_test, stacking_preds))
    print(f"CV Stacking: {stacking_rmse:.4f}")
    
    # 4. Blending
    blending = EnsembleFactory.create_blending_ensemble(
        models=models,
        holdout_size=0.2,
        task_type='regression'
    )
    blending.fit(X_train, y_train)
    blending_preds = blending.predict(X_test)
    blending_rmse = np.sqrt(mean_squared_error(y_test, blending_preds))
    print(f"Blending: {blending_rmse:.4f}")
    
    # 5. Dynamic Selection
    dynamic = HeterogeneousEnsemble(
        models=models,
        ensemble_method='dynamic_selection',
        task_type='regression'
    )
    # Add some performance history for dynamic selection
    for model_wrapper in dynamic.models:
        model_wrapper.record_performance(np.random.uniform(0.5, 2.0), 'rmse')
    
    dynamic_preds = dynamic.predict(X_test)
    dynamic_rmse = np.sqrt(mean_squared_error(y_test, dynamic_preds))
    print(f"Dynamic Selection: {dynamic_rmse:.4f}")
    
    ensemble_rmses = [simple_rmse, weighted_rmse, stacking_rmse, blending_rmse, dynamic_rmse]
    best_ensemble_rmse = min(ensemble_rmses)
    print(f"\nBest ensemble RMSE: {best_ensemble_rmse:.4f}")
    print(f"Improvement over best individual: {((min(individual_rmses) - best_ensemble_rmse) / min(individual_rmses) * 100):.1f}%")


def demonstrate_classification_ensembles():
    """Demonstrate ensemble methods for classification"""
    print("\n" + "=" * 60)
    print("CLASSIFICATION ENSEMBLE DEMONSTRATION")
    print("=" * 60)
    
    # Create classification dataset
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                              n_redundant=2, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Base model for homogeneous ensemble
    base_classifier = LogisticRegression(max_iter=1000, random_state=42)
    
    # Homogeneous classification ensemble
    print("\nHomogeneous Classification Ensemble:")
    print("-" * 40)
    
    cls_bagging = EnsembleFactory.create_bagging_ensemble(
        base_estimator=base_classifier,
        n_estimators=10,
        max_samples=0.8,
        random_state=42
    )
    cls_bagging.fit(X_train, y_train)
    cls_preds = cls_bagging.predict(X_test)
    cls_acc = accuracy_score(y_test, cls_preds)
    cls_proba = cls_bagging.predict_proba(X_test)
    
    print(f"Bagging Classification Accuracy: {cls_acc:.4f}")
    print(f"Probability predictions shape: {cls_proba.shape}")
    
    # Individual classifier for comparison
    base_classifier.fit(X_train, y_train)
    base_preds = base_classifier.predict(X_test)
    base_acc = accuracy_score(y_test, base_preds)
    print(f"Base Classifier Accuracy: {base_acc:.4f}")
    
    improvement = ((cls_acc - base_acc) / base_acc * 100)
    print(f"Accuracy improvement: {improvement:.1f}%")


def main():
    """Run all ensemble demonstrations"""
    print("QUANTRA ENSEMBLE LEARNING DEMONSTRATION")
    print("New Homogeneous and Heterogeneous Ensemble Methods")
    
    try:
        demonstrate_homogeneous_ensembles()
        demonstrate_heterogeneous_ensembles()
        demonstrate_classification_ensembles()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("\nKey Takeaways:")
        print("1. Homogeneous ensembles improve stability by training multiple")
        print("   instances of the same model on different data subsets")
        print("2. Heterogeneous ensembles combine different model types")
        print("   to leverage diverse learning approaches")
        print("3. Advanced methods like stacking and blending often")
        print("   provide superior performance over simple averaging")
        print("4. Factory methods make ensemble creation simple and consistent")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()