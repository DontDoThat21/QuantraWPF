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
    from python.ensemble_learning import ModelWrapper, EnsembleModel, create_ensemble_from_models, train_model_ensemble
    ENSEMBLE_MODULE_AVAILABLE = True
except ImportError:
    try:
        from ensemble_learning import ModelWrapper, EnsembleModel, create_ensemble_from_models, train_model_ensemble
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


if __name__ == '__main__':
    unittest.main()