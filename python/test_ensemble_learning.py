#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    from python.ensemble_learning import (
        ModelWrapper, EnsembleModel, create_ensemble_from_models, train_model_ensemble,
        HomogeneousEnsemble, HeterogeneousEnsemble, EnsembleFactory
    )
    ENSEMBLE_MODULE_AVAILABLE = True
except ImportError:
    try:
        from ensemble_learning import (
            ModelWrapper, EnsembleModel, create_ensemble_from_models, train_model_ensemble,
            HomogeneousEnsemble, HeterogeneousEnsemble, EnsembleFactory
        )
        ENSEMBLE_MODULE_AVAILABLE = True
    except ImportError:
        ENSEMBLE_MODULE_AVAILABLE = False

@unittest.skipIf(not ENSEMBLE_MODULE_AVAILABLE, "Ensemble learning module not available")
class TestEnsembleLearning(unittest.TestCase):
    
    def setUp(self):
        """Set up test data and models"""
        # Regression data
        X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
        self.X_reg_train, self.X_reg_test, self.y_reg_train, self.y_reg_test = train_test_split(
            X_reg, y_reg, test_size=0.2, random_state=42
        )
        
        # Classification data
        X_cls, y_cls = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                                          n_redundant=2, n_classes=3, random_state=42)
        self.X_cls_train, self.X_cls_test, self.y_cls_train, self.y_cls_test = train_test_split(
            X_cls, y_cls, test_size=0.2, random_state=42
        )
        
        # Regression models
        self.rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_reg.fit(self.X_reg_train, self.y_reg_train)
        
        self.lr_reg = LinearRegression()
        self.lr_reg.fit(self.X_reg_train, self.y_reg_train)
        
        # Classification models
        self.rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_cls.fit(self.X_cls_train, self.y_cls_train)
        
        self.log_cls = LogisticRegression(max_iter=1000, random_state=42)
        self.log_cls.fit(self.X_cls_train, self.y_cls_train)
    
    def test_model_wrapper(self):
        """Test ModelWrapper functionality"""
        # Test wrapping sklearn model
        rf_wrapper = ModelWrapper(self.rf_reg, model_type='sklearn', name='RandomForest')
        self.assertEqual(rf_wrapper.name, 'RandomForest')
        self.assertEqual(rf_wrapper.model_type, 'sklearn')
        
        # Test prediction
        preds = rf_wrapper.predict(self.X_reg_test)
        self.assertEqual(preds.shape, (self.X_reg_test.shape[0],))
        
        # Test feature importance
        importance = rf_wrapper.get_feature_importance()
        self.assertEqual(len(importance), self.X_reg_test.shape[1])
        
        # Test performance tracking
        rf_wrapper.record_performance(0.5, 'rmse')
        self.assertEqual(len(rf_wrapper.performance_history), 1)
        self.assertEqual(rf_wrapper.get_recent_performance('rmse'), 0.5)
    
    def test_ensemble_regression(self):
        """Test ensemble for regression task"""
        # Create ensemble
        ensemble = EnsembleModel(
            models=[self.rf_reg, self.lr_reg],
            model_types=['sklearn', 'sklearn'],
            ensemble_method='weighted_average',
            task_type='regression'
        )
        
        # Test prediction
        preds = ensemble.predict(self.X_reg_test)
        self.assertEqual(preds.shape, (self.X_reg_test.shape[0],))
        
        # Calculate RMSE
        mse = mean_squared_error(self.y_reg_test, preds)
        rmse = np.sqrt(mse)
        
        # Compare to individual models
        rf_preds = self.rf_reg.predict(self.X_reg_test)
        lr_preds = self.lr_reg.predict(self.X_reg_test)
        
        rf_mse = mean_squared_error(self.y_reg_test, rf_preds)
        lr_mse = mean_squared_error(self.y_reg_test, lr_preds)
        rf_rmse = np.sqrt(rf_mse)
        lr_rmse = np.sqrt(lr_mse)
        
        # Ensemble should be at least as good as the average of individual models
        avg_rmse = (rf_rmse + lr_rmse) / 2
        self.assertLessEqual(rmse, avg_rmse * 1.05)  # Allow 5% tolerance
        
        # Test ensemble evaluation
        eval_results = ensemble.evaluate(self.X_reg_test, self.y_reg_test)
        self.assertIn('ensemble', eval_results)
        self.assertIn('rmse', eval_results['ensemble'])
        
        # Test feature importance
        importance = ensemble.get_feature_importance()
        self.assertGreaterEqual(len(importance), 1)
    
    def test_ensemble_classification(self):
        """Test ensemble for classification task"""
        # Create ensemble
        ensemble = EnsembleModel(
            models=[self.rf_cls, self.log_cls],
            model_types=['sklearn', 'sklearn'],
            ensemble_method='weighted_average',
            voting='soft',
            task_type='classification'
        )
        
        # Test prediction
        preds = ensemble.predict(self.X_cls_test)
        self.assertEqual(preds.shape, (self.X_cls_test.shape[0],))
        
        # Calculate accuracy
        acc = accuracy_score(self.y_cls_test, preds)
        
        # Compare to individual models
        rf_preds = self.rf_cls.predict(self.X_cls_test)
        log_preds = self.log_cls.predict(self.X_cls_test)
        rf_acc = accuracy_score(self.y_cls_test, rf_preds)
        log_acc = accuracy_score(self.y_cls_test, log_preds)
        
        # Ensemble should be at least as good as the worst individual model
        min_acc = min(rf_acc, log_acc)
        self.assertGreaterEqual(acc, min_acc * 0.95)  # Allow 5% tolerance
        
        # Test predict_proba
        if hasattr(ensemble, 'predict_proba'):
            proba = ensemble.predict_proba(self.X_cls_test)
            self.assertEqual(proba.shape, (self.X_cls_test.shape[0], 3))  # 3 classes
            # Probabilities should sum to 1
            self.assertTrue(np.allclose(np.sum(proba, axis=1), np.ones(self.X_cls_test.shape[0])))
    
    def test_train_model_ensemble(self):
        """Test automatic ensemble training"""
        # Regression ensemble
        reg_ensemble, reg_results = train_model_ensemble(
            self.X_reg_train, self.y_reg_train,
            models_to_train=[
                {'model_class': RandomForestRegressor, 'n_estimators': 50, 'random_state': 42},
                {'model_class': LinearRegression}
            ],
            task_type='regression',
            ensemble_method='weighted_average',
            dynamic_weighting=True,
            random_state=42
        )
        
        # Test the ensemble
        reg_preds = reg_ensemble.predict(self.X_reg_test)
        reg_mse = mean_squared_error(self.y_reg_test, reg_preds)
        reg_rmse = np.sqrt(reg_mse)
        
        # Results should include metrics
        self.assertIn('ensemble_metrics', reg_results)
        self.assertIn('model_metrics', reg_results)
        
        # Classification ensemble
        cls_ensemble, cls_results = train_model_ensemble(
            self.X_cls_train, self.y_cls_train,
            models_to_train=[
                {'model_class': RandomForestClassifier, 'n_estimators': 50},
                {'model_class': LogisticRegression, 'max_iter': 1000}
            ],
            task_type='classification',
            ensemble_method='weighted_average',
            voting='soft',
            dynamic_weighting=True,
            random_state=42
        )
        
        # Test the ensemble
        cls_preds = cls_ensemble.predict(self.X_cls_test)
        cls_acc = accuracy_score(self.y_cls_test, cls_preds)
        
        # Results should include metrics
        self.assertIn('ensemble_metrics', cls_results)
        self.assertIn('model_metrics', cls_results)
    
    def test_stacking_ensemble(self):
        """Test stacking ensemble method"""
        # Create stacking ensemble for regression
        stacking_ensemble = EnsembleModel(
            models=[self.rf_reg, self.lr_reg],
            model_types=['sklearn', 'sklearn'],
            ensemble_method='stacking',
            task_type='regression'
        )
        
        # Fit the meta-model
        stacking_ensemble.fit(self.X_reg_train, self.y_reg_train)
        
        # Test prediction
        preds = stacking_ensemble.predict(self.X_reg_test)
        self.assertEqual(preds.shape, (self.X_reg_test.shape[0],))
        
        # Classification stacking
        stacking_cls = EnsembleModel(
            models=[self.rf_cls, self.log_cls],
            model_types=['sklearn', 'sklearn'],
            ensemble_method='stacking',
            task_type='classification'
        )
        
        # Fit the meta-model
        stacking_cls.fit(self.X_cls_train, self.y_cls_train)
        
        # Test prediction
        cls_preds = stacking_cls.predict(self.X_cls_test)
        self.assertEqual(cls_preds.shape, (self.X_cls_test.shape[0],))

    def test_homogeneous_ensemble_bagging(self):
        """Test homogeneous ensemble with bagging"""
        # Test bagging for regression
        bagging_reg = HomogeneousEnsemble(
            base_estimator=LinearRegression(),
            n_estimators=5,
            ensemble_method='bagging',
            max_samples=0.8,
            random_state=42
        )
        
        bagging_reg.fit(self.X_reg_train, self.y_reg_train)
        preds = bagging_reg.predict(self.X_reg_test)
        self.assertEqual(preds.shape, (self.X_reg_test.shape[0],))
        
        # Check that multiple estimators were created
        self.assertEqual(len(bagging_reg.estimators_), 5)
        self.assertEqual(len(bagging_reg.feature_indices_), 5)
        self.assertEqual(len(bagging_reg.sample_indices_), 5)
        
        # Test bagging for classification
        from sklearn.linear_model import LogisticRegression
        bagging_cls = HomogeneousEnsemble(
            base_estimator=LogisticRegression(max_iter=1000, random_state=42),
            n_estimators=3,
            ensemble_method='bagging',
            random_state=42
        )
        
        bagging_cls.fit(self.X_cls_train, self.y_cls_train)
        cls_preds = bagging_cls.predict(self.X_cls_test)
        self.assertEqual(cls_preds.shape, (self.X_cls_test.shape[0],))
        
        # Test predict_proba
        cls_proba = bagging_cls.predict_proba(self.X_cls_test)
        self.assertEqual(cls_proba.shape, (self.X_cls_test.shape[0], 3))  # 3 classes
        self.assertTrue(np.allclose(np.sum(cls_proba, axis=1), np.ones(self.X_cls_test.shape[0])))
    
    def test_homogeneous_ensemble_random_subspace(self):
        """Test homogeneous ensemble with random subspace method"""
        rs_ensemble = HomogeneousEnsemble(
            base_estimator=RandomForestRegressor(n_estimators=10, random_state=42),
            n_estimators=3,
            ensemble_method='random_subspace',
            max_features=0.5,
            random_state=42
        )
        
        rs_ensemble.fit(self.X_reg_train, self.y_reg_train)
        preds = rs_ensemble.predict(self.X_reg_test)
        self.assertEqual(preds.shape, (self.X_reg_test.shape[0],))
        
        # Check that feature indices are different for each estimator
        feature_indices = rs_ensemble.feature_indices_
        self.assertEqual(len(feature_indices), 3)
        
        # Each should use about half the features
        for indices in feature_indices:
            self.assertLessEqual(len(indices), self.X_reg_train.shape[1])
            self.assertGreater(len(indices), 0)
    
    def test_homogeneous_ensemble_pasting(self):
        """Test homogeneous ensemble with pasting (no replacement)"""
        pasting_ensemble = HomogeneousEnsemble(
            base_estimator=LinearRegression(),
            n_estimators=3,
            ensemble_method='pasting',
            max_samples=0.6,
            bootstrap=False,
            random_state=42
        )
        
        pasting_ensemble.fit(self.X_reg_train, self.y_reg_train)
        preds = pasting_ensemble.predict(self.X_reg_test)
        self.assertEqual(preds.shape, (self.X_reg_test.shape[0],))
        
        # Check bootstrap is disabled
        self.assertFalse(pasting_ensemble.bootstrap)
    
    def test_heterogeneous_ensemble_blending(self):
        """Test heterogeneous ensemble with blending"""
        blending_ensemble = HeterogeneousEnsemble(
            models=[self.rf_reg, self.lr_reg],
            model_types=['sklearn', 'sklearn'],
            ensemble_method='blending',
            blending_holdout=0.3,
            task_type='regression'
        )
        
        # Fit with blending
        blending_ensemble.fit(self.X_reg_train, self.y_reg_train)
        
        # Test prediction
        preds = blending_ensemble.predict(self.X_reg_test)
        self.assertEqual(preds.shape, (self.X_reg_test.shape[0],))
        
        # Check that meta-model was created
        self.assertIsNotNone(blending_ensemble.meta_model)
    
    def test_heterogeneous_ensemble_cv_stacking(self):
        """Test heterogeneous ensemble with cross-validation stacking"""
        cv_stacking_ensemble = HeterogeneousEnsemble(
            models=[self.rf_reg, self.lr_reg],
            model_types=['sklearn', 'sklearn'],
            ensemble_method='cv_stacking',
            cv_folds=3,
            task_type='regression'
        )
        
        # Fit with CV stacking
        cv_stacking_ensemble.fit(self.X_reg_train, self.y_reg_train)
        
        # Test prediction
        preds = cv_stacking_ensemble.predict(self.X_reg_test)
        self.assertEqual(preds.shape, (self.X_reg_test.shape[0],))
        
        # Check that meta-model was created
        self.assertIsNotNone(cv_stacking_ensemble.meta_model)
    
    def test_ensemble_factory(self):
        """Test ensemble factory methods"""
        # Test bagging ensemble creation
        bagging_ensemble = EnsembleFactory.create_bagging_ensemble(
            base_estimator=LinearRegression(),
            n_estimators=3,
            max_samples=0.8,
            random_state=42
        )
        
        self.assertIsInstance(bagging_ensemble, HomogeneousEnsemble)
        self.assertEqual(bagging_ensemble.ensemble_method, 'bagging')
        self.assertEqual(bagging_ensemble.n_estimators, 3)
        
        # Test random subspace ensemble creation
        rs_ensemble = EnsembleFactory.create_random_subspace_ensemble(
            base_estimator=RandomForestRegressor(n_estimators=10, random_state=42),
            n_estimators=3,
            max_features=0.5,
            random_state=42
        )
        
        self.assertIsInstance(rs_ensemble, HomogeneousEnsemble)
        self.assertEqual(rs_ensemble.ensemble_method, 'random_subspace')
        
        # Test stacking ensemble creation
        stacking_ensemble = EnsembleFactory.create_stacking_ensemble(
            models=[self.rf_reg, self.lr_reg],
            cv_folds=3
        )
        
        self.assertIsInstance(stacking_ensemble, HeterogeneousEnsemble)
        self.assertEqual(stacking_ensemble.ensemble_method, 'cv_stacking')
        
        # Test blending ensemble creation
        blending_ensemble = EnsembleFactory.create_blending_ensemble(
            models=[self.rf_reg, self.lr_reg],
            holdout_size=0.3
        )
        
        self.assertIsInstance(blending_ensemble, HeterogeneousEnsemble)
        self.assertEqual(blending_ensemble.ensemble_method, 'blending')
    
    def test_ensemble_factory_fitted_performance(self):
        """Test that factory-created ensembles can be fitted and perform reasonably"""
        # Test bagging ensemble
        bagging_ensemble = EnsembleFactory.create_bagging_ensemble(
            base_estimator=RandomForestRegressor(n_estimators=10, random_state=42),
            n_estimators=3,
            random_state=42
        )
        
        bagging_ensemble.fit(self.X_reg_train, self.y_reg_train)
        bagging_preds = bagging_ensemble.predict(self.X_reg_test)
        bagging_rmse = np.sqrt(mean_squared_error(self.y_reg_test, bagging_preds))
        
        # Test individual model for comparison
        individual_rf = RandomForestRegressor(n_estimators=10, random_state=42)
        individual_rf.fit(self.X_reg_train, self.y_reg_train)
        individual_preds = individual_rf.predict(self.X_reg_test)
        individual_rmse = np.sqrt(mean_squared_error(self.y_reg_test, individual_preds))
        
        # Ensemble should be competitive (within 50% of individual model)
        self.assertLess(bagging_rmse, individual_rmse * 1.5)
        
        # Test heterogeneous stacking
        stacking_ensemble = EnsembleFactory.create_stacking_ensemble(
            models=[self.rf_reg, self.lr_reg],
            cv_folds=3,
            task_type='regression'
        )
        
        stacking_ensemble.fit(self.X_reg_train, self.y_reg_train)
        stacking_preds = stacking_ensemble.predict(self.X_reg_test)
        stacking_rmse = np.sqrt(mean_squared_error(self.y_reg_test, stacking_preds))
        
        # Stacking should be reasonable
        self.assertLess(stacking_rmse, individual_rmse * 2.0)
    
    def test_parameter_validation(self):
        """Test parameter validation in homogeneous ensembles"""
        # Test invalid max_samples
        with self.assertRaises(ValueError):
            ensemble = HomogeneousEnsemble(
                base_estimator=LinearRegression(),
                max_samples=2.0  # Invalid: > 1.0 for float
            )
            ensemble.fit(self.X_reg_train, self.y_reg_train)
        
        # Test invalid max_features  
        with self.assertRaises(ValueError):
            ensemble = HomogeneousEnsemble(
                base_estimator=LinearRegression(),
                max_features=0.0  # Invalid: <= 0
            )
            ensemble.fit(self.X_reg_train, self.y_reg_train)


if __name__ == '__main__':
    unittest.main()