#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temporal Fusion Transformer implementation for multi-horizon stock prediction.
Based on: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
Reference: https://arxiv.org/abs/1912.09363
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger('temporal_fusion_transformer')


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) with GLU activation.
    Core building block of TFT.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 dropout: float = 0.1, context_dim: Optional[int] = None):
        super(GatedResidualNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        
        # Context processing if provided
        if context_dim is not None:
            self.context_projection = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_projection = None
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # GLU gate
        self.gate = nn.Linear(hidden_dim, output_dim * 2)
        
        # Skip connection
        if input_dim != output_dim:
            self.skip_layer = nn.Linear(input_dim, output_dim)
        else:
            self.skip_layer = None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through GRN.
        
        Args:
            x: Input tensor of shape (batch, ..., input_dim)
            context: Optional context tensor of shape (batch, ..., context_dim)
            
        Returns:
            Output tensor of shape (batch, ..., output_dim)
        """
        # First linear layer
        hidden = self.fc1(x)
        
        # Add context if provided
        if self.context_projection is not None and context is not None:
            # Expand context to match hidden dimensions if needed
            context_proj = self.context_projection(context)
            hidden = hidden + context_proj
        
        hidden = self.elu(hidden)
        
        # Second linear layer with dropout
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        
        # GLU gating mechanism
        gated = self.gate(hidden)
        gated, gate = gated.chunk(2, dim=-1)
        gated = gated * torch.sigmoid(gate)
        
        # Skip connection
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x
        
        # Add and normalize
        output = self.layer_norm(gated + skip)
        
        return output


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) for automatic feature selection.
    Uses Gated Residual Network (GRN) for each variable.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_features: int,
                 dropout: float = 0.1, context_dim: Optional[int] = None):
        super(VariableSelectionNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_features = num_features
        
        # Flatten input for variable-wise processing
        self.flatten_dim = input_dim
        
        # GRN for variable selection weights
        self.weight_grn = GatedResidualNetwork(
            input_dim=input_dim * num_features if num_features > 1 else input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            dropout=dropout,
            context_dim=context_dim
        )
        
        # GRN for each variable's transformation
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout
            ) for _ in range(num_features)
        ])
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through VSN.
        
        Args:
            x: Input tensor of shape (batch, time, features) or (batch, features)
            context: Optional context tensor of shape (batch, hidden_dim)
            
        Returns:
            Tuple of:
                - Selected output of shape (batch, time, hidden_dim) or (batch, hidden_dim)
                - Variable importance weights of shape (batch, num_features)
        """
        has_time_dim = len(x.shape) == 3
        
        if has_time_dim:
            batch_size, seq_len, _ = x.shape
            # Flatten for weight computation
            x_flat = x.reshape(batch_size * seq_len, -1)
            
            # Expand context for time dimension
            if context is not None:
                context_expanded = context.unsqueeze(1).expand(-1, seq_len, -1)
                context_flat = context_expanded.reshape(batch_size * seq_len, -1)
            else:
                context_flat = None
        else:
            batch_size = x.shape[0]
            x_flat = x
            context_flat = context
        
        # Compute variable selection weights
        weights_input = x_flat
            
        weights = self.weight_grn(weights_input, context_flat)
        weights = self.softmax(weights)  # (batch*time, num_features)
        
        # Transform each variable through its GRN
        variable_outputs = []
        feature_per_var = x_flat.shape[-1] // self.num_features if self.num_features > 1 else x_flat.shape[-1]
        
        for i, grn in enumerate(self.variable_grns):
            if self.num_features > 1:
                var_input = x_flat[:, i * feature_per_var:(i + 1) * feature_per_var]
            else:
                var_input = x_flat
            var_output = grn(var_input)  # (batch*time, hidden_dim)
            variable_outputs.append(var_output)
        
        # Stack variable outputs: (batch*time, num_features, hidden_dim)
        variable_outputs = torch.stack(variable_outputs, dim=1)
        
        # Apply variable selection weights
        weights_expanded = weights.unsqueeze(-1)  # (batch*time, num_features, 1)
        selected = (variable_outputs * weights_expanded).sum(dim=1)  # (batch*time, hidden_dim)
        
        if has_time_dim:
            selected = selected.reshape(batch_size, seq_len, -1)
            weights = weights.reshape(batch_size, seq_len, -1).mean(dim=1)  # Average over time
        
        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    """
    Multi-head attention with interpretable attention weights.
    Stores attention patterns for visualization.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attention_weights = None  # Store for interpretation
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                need_weights: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weight storage.
        
        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            need_weights: Whether to return attention weights
            
        Returns:
            Tuple of output and attention weights
        """
        output, weights = self.attention(query, key, value, need_weights=need_weights)
        self.attention_weights = weights  # Store for later access
        return output, weights


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon stock prediction.
    
    Architecture:
    1. Variable Selection Networks (static & temporal)
    2. LSTM encoders (past & future)
    3. Multi-head attention
    4. Gated residual networks
    5. Quantile output layers
    """
    
    def __init__(self,
                 input_dim: int,
                 static_dim: int,
                 hidden_dim: int = 128,
                 num_heads: int = 4,
                 num_lstm_layers: int = 2,
                 dropout: float = 0.1,
                 num_quantiles: int = 5,
                 forecast_horizons: Optional[List[int]] = None,
                 num_attention_layers: int = 2):
        super(TemporalFusionTransformer, self).__init__()
        
        if forecast_horizons is None:
            forecast_horizons = [5, 10, 20, 30]
        
        self.input_dim = input_dim
        self.static_dim = static_dim
        self.hidden_dim = hidden_dim
        self.num_quantiles = num_quantiles
        self.forecast_horizons = forecast_horizons
        self.quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]  # 10th, 25th, median, 75th, 90th
        
        # Ensure hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            hidden_dim = num_heads * (hidden_dim // num_heads + 1)
            self.hidden_dim = hidden_dim
            logger.info(f"Adjusted hidden_dim to {hidden_dim} for divisibility by num_heads")
        
        # 1. Static covariate encoder
        self.static_embedding = nn.Linear(static_dim, hidden_dim)
        self.static_context_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # 2. Temporal variable selection (past)
        # Each time step has input_dim features
        self.temporal_embedding = nn.Linear(input_dim, hidden_dim)
        self.past_vsn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout, context_dim=hidden_dim)
        
        # 3. LSTM encoder for past observations
        self.past_lstm = nn.LSTM(hidden_dim, hidden_dim, num_lstm_layers,
                                 batch_first=True, dropout=dropout if num_lstm_layers > 1 else 0)
        
        # 4. Gated skip connection
        self.gated_skip = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
        
        # 5. Static enrichment
        self.static_enrichment = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout, context_dim=hidden_dim)
        
        # 6. Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            InterpretableMultiHeadAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_attention_layers)
        ])
        
        # Post-attention GRNs
        self.post_attention_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout)
            for _ in range(num_attention_layers)
        ])
        
        # 7. Position-wise feed-forward
        self.position_wise_grn = GatedResidualNetwork(hidden_dim, hidden_dim * 4, hidden_dim, dropout)
        
        # 8. Output layers for each horizon and quantile
        self.output_layers = nn.ModuleDict()
        for h in forecast_horizons:
            self.output_layers[f"horizon_{h}"] = nn.ModuleList([
                nn.Linear(hidden_dim, 1) for _ in range(num_quantiles)
            ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Variable importance tracking
        self.variable_importance = None
        
    def forward(self,
                past_features: torch.Tensor,        # (batch, past_seq_len, input_dim)
                static_features: torch.Tensor,      # (batch, static_dim)
                future_features: Optional[torch.Tensor] = None  # (batch, future_seq_len, calendar_dim)
               ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of TFT.
        
        STEP 6 ENHANCEMENT: Now properly processes future calendar features as known-future-inputs.
        
        Args:
            past_features: Historical temporal features (batch, past_seq_len, input_dim)
            static_features: Static/time-invariant features (batch, static_dim)
            future_features: Optional known future calendar features (batch, future_seq_len, calendar_dim)
                           Examples: day_of_week, month, quarter, is_holiday, is_month_end, etc.
                           These are deterministic and known in advance for the forecast horizon.
            
        Returns:
            dict with keys:
                - 'predictions': dict of horizon predictions, each (batch, num_quantiles)
                - 'attention_weights': List of attention weight tensors
                - 'variable_importance': (batch, input_dim)
                - 'static_context': (batch, hidden_dim)
        """
        batch_size = past_features.size(0)
        past_seq_len = past_features.size(1)
        
        # 1. Static covariate processing
        static_embedded = self.static_embedding(static_features)  # (batch, hidden_dim)
        static_context = self.static_context_grn(static_embedded)  # (batch, hidden_dim)
        
        # 2. Temporal embedding and variable selection for past
        past_embedded = self.temporal_embedding(past_features)  # (batch, past_seq_len, hidden_dim)
        
        # Apply variable selection with static context
        static_context_expanded = static_context.unsqueeze(1).expand(-1, past_seq_len, -1)
        selected_past = self.past_vsn(past_embedded, static_context_expanded)  # (batch, past_seq_len, hidden_dim)
        
        # 3. LSTM encoding of past observations
        past_lstm_out, (h_past, c_past) = self.past_lstm(selected_past)  # (batch, past_seq_len, hidden_dim)
        
        # 4. Process future calendar features if provided (STEP 6 ENHANCEMENT)
        if future_features is not None and future_features.size(1) > 0:
            future_seq_len = future_features.size(1)
            
            # Project future calendar features to hidden_dim
            # Note: calendar features are typically low-dimensional (e.g., 12 dims)
            if not hasattr(self, 'future_embedding'):
                # Dynamically create future embedding layer if it doesn't exist
                calendar_dim = future_features.size(-1)
                self.future_embedding = nn.Linear(calendar_dim, self.hidden_dim).to(future_features.device)
                logger.info(f"Created future_embedding layer: {calendar_dim} -> {self.hidden_dim}")
            
            future_embedded = self.future_embedding(future_features)  # (batch, future_seq_len, hidden_dim)
            
            # Process future features through LSTM decoder (continuing from past state)
            future_lstm_out, _ = self.past_lstm(future_embedded, (h_past, c_past))
            
            # Concatenate past and future sequences
            combined_sequence = torch.cat([past_lstm_out, future_lstm_out], dim=1)  # (batch, past+future, hidden_dim)
            combined_seq_len = past_seq_len + future_seq_len
            
            # Expand static context for combined sequence
            static_context_combined = static_context.unsqueeze(1).expand(-1, combined_seq_len, -1)
            
            # Apply gated skip connection to combined sequence
            lstm_out = self.gated_skip(combined_sequence)
        else:
            # No future features - use only past (backward compatible)
            lstm_out = self.gated_skip(past_lstm_out)
            static_context_combined = static_context_expanded
            combined_seq_len = past_seq_len
        
        # 5. Static enrichment
        enriched = self.static_enrichment(lstm_out, static_context_combined)  # (batch, combined_seq_len, hidden_dim)
        
        # 6. Multi-head attention with residual connections
        # Attention now has access to both past and future calendar information
        attention_out = enriched
        attention_weights_list = []
        
        for i, (attention_layer, grn) in enumerate(zip(self.attention_layers, self.post_attention_grns)):
            attn_out, weights = attention_layer(attention_out, attention_out, attention_out)
            attention_weights_list.append(weights)
            
            # Apply post-attention GRN and residual
            attn_out = grn(attn_out)
            attention_out = self.layer_norm(attention_out + attn_out)
        
        # 7. Position-wise feed-forward
        output = self.position_wise_grn(attention_out)  # (batch, combined_seq_len, hidden_dim)
        
        # 8. Take the last time step for prediction
        # This now includes information from future calendar features if provided
        final_output = output[:, -1, :]  # (batch, hidden_dim)
        
        # 9. Generate predictions for each horizon and quantile
        predictions = {}
        for horizon in self.forecast_horizons:
            horizon_key = f"horizon_{horizon}"
            quantile_predictions = []
            for quantile_layer in self.output_layers[horizon_key]:
                pred = quantile_layer(final_output)  # (batch, 1)
                quantile_predictions.append(pred)
            predictions[horizon_key] = torch.cat(quantile_predictions, dim=1)  # (batch, num_quantiles)
        
        # Estimate variable importance from gradients (simplified)
        # In practice, you would compute this during training with gradient hooks
        variable_importance = torch.zeros(batch_size, self.input_dim, device=past_features.device)
        
        return {
            'predictions': predictions,
            'attention_weights': attention_weights_list,
            'variable_importance': variable_importance,
            'static_context': static_context
        }
    
    def interpret_prediction(self,
                            outputs: Dict[str, torch.Tensor],
                            feature_names: List[str]) -> Dict[str, any]:
        """
        Generate interpretation information for predictions.
        
        Args:
            outputs: Output dict from forward pass
            feature_names: Names of input features
            
        Returns:
            - Top contributing features
            - Attention patterns
            - Uncertainty estimates
        """
        interpretation = {
            'top_features': [],
            'attention_patterns': [],
            'uncertainty_estimates': {}
        }
        
        # Feature importance
        if 'variable_importance' in outputs:
            importance = outputs['variable_importance'].mean(dim=0).cpu().numpy()
            if len(importance) > 0 and len(feature_names) > 0:
                top_indices = np.argsort(importance)[-min(10, len(importance)):][::-1]
                interpretation['top_features'] = [
                    {'feature': feature_names[i] if i < len(feature_names) else f'feature_{i}', 
                     'importance': float(importance[i])}
                    for i in top_indices
                ]
        
        # Uncertainty (from quantile spread)
        for horizon_key, pred_quantiles in outputs['predictions'].items():
            # pred_quantiles: (batch, num_quantiles)
            if pred_quantiles.size(1) >= 5:
                q10 = pred_quantiles[:, 0]  # 10th percentile
                q90 = pred_quantiles[:, 4]  # 90th percentile
                uncertainty = (q90 - q10).mean().item()
                interpretation['uncertainty_estimates'][horizon_key] = uncertainty
        
        return interpretation


class QuantileLoss(nn.Module):
    """
    Quantile loss for training TFT.
    Ensures proper calibration of prediction intervals.
    """
    
    def __init__(self, quantiles: Optional[List[float]] = None):
        super(QuantileLoss, self).__init__()
        if quantiles is None:
            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        self.quantiles = quantiles
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.
        
        Args:
            predictions: (batch, num_quantiles)
            targets: (batch, 1) or (batch,)
            
        Returns:
            Scalar loss value
        """
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)
        
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[:, i:i+1]
            losses.append(torch.max((q - 1) * errors, q * errors))
        
        return torch.mean(torch.cat(losses, dim=1))


def train_tft_model(model: TemporalFusionTransformer,
                    train_loader,
                    val_loader,
                    epochs: int = 50,
                    lr: float = 0.001,
                    device: str = 'cuda') -> Dict[str, List[float]]:
    """
    Train TFT model with quantile loss.
    
    Args:
        model: TFT model instance
        train_loader: Training data loader (list of dicts or DataLoader)
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Training history dict with train_loss and val_loss
    """
    # Move model to device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    loss_fn = QuantileLoss([0.1, 0.25, 0.5, 0.75, 0.9])
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        train_losses = []
        
        for batch in train_loader:
            # Handle both dict format and tuple format
            if isinstance(batch, dict):
                past_features = batch['past_features'].to(device)
                static_features = batch['static_features'].to(device)
                targets = batch['targets'].to(device)
            else:
                past_features, static_features, targets = batch
                past_features = past_features.to(device)
                static_features = static_features.to(device)
                targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(past_features, static_features)
            
            # Calculate loss for each horizon
            total_loss = 0
            for horizon_idx, horizon in enumerate(model.forecast_horizons):
                horizon_key = f"horizon_{horizon}"
                pred_quantiles = outputs['predictions'][horizon_key]
                
                if targets.dim() == 2 and targets.size(1) > 1:
                    target = targets[:, horizon_idx:horizon_idx+1]
                else:
                    target = targets
                    
                loss = loss_fn(pred_quantiles, target)
                total_loss += loss
            
            total_loss = total_loss / len(model.forecast_horizons)
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_losses.append(total_loss.item())
        
        # Validation loop
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    past_features = batch['past_features'].to(device)
                    static_features = batch['static_features'].to(device)
                    targets = batch['targets'].to(device)
                else:
                    past_features, static_features, targets = batch
                    past_features = past_features.to(device)
                    static_features = static_features.to(device)
                    targets = targets.to(device)
                
                outputs = model(past_features, static_features)
                
                total_loss = 0
                for horizon_idx, horizon in enumerate(model.forecast_horizons):
                    horizon_key = f"horizon_{horizon}"
                    pred_quantiles = outputs['predictions'][horizon_key]
                    
                    if targets.dim() == 2 and targets.size(1) > 1:
                        target = targets[:, horizon_idx:horizon_idx+1]
                    else:
                        target = targets
                        
                    loss = loss_fn(pred_quantiles, target)
                    total_loss += loss
                
                total_loss = total_loss / len(model.forecast_horizons)
                val_losses.append(total_loss.item())
        
        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    return history