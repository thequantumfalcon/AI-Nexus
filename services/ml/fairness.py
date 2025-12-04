"""
Fairness and Bias Detection Module
===================================

Detect and mitigate bias in ML models across protected attributes.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FairnessMetrics:
    """Fairness evaluation metrics"""
    demographic_parity_difference: float
    equalized_odds_difference: float
    disparate_impact_ratio: float
    equal_opportunity_difference: float
    average_odds_difference: float
    protected_group_stats: Dict[str, Dict[str, float]]


class FairnessAnalyzer:
    """
    Analyze ML model fairness across demographic groups
    """
    
    def __init__(self):
        self.epsilon = 1e-10  # Avoid division by zero
    
    def evaluate_fairness(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attribute: np.ndarray,
        favorable_outcome: int = 1
    ) -> FairnessMetrics:
        """
        Comprehensive fairness evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels  
            protected_attribute: Group membership (e.g., 0=majority, 1=minority)
            favorable_outcome: Which outcome is favorable (default=1)
        """
        logger.info("Evaluating model fairness")
        
        # Get unique groups
        groups = np.unique(protected_attribute)
        
        if len(groups) != 2:
            logger.warning(f"Expected 2 groups, got {len(groups)}. Using binary split.")
            groups = groups[:2]
        
        group_0_mask = protected_attribute == groups[0]
        group_1_mask = protected_attribute == groups[1]
        
        # Calculate metrics for each group
        metrics_0 = self._calculate_group_metrics(
            y_true[group_0_mask],
            y_pred[group_0_mask],
            favorable_outcome
        )
        metrics_1 = self._calculate_group_metrics(
            y_true[group_1_mask],
            y_pred[group_1_mask],
            favorable_outcome
        )
        
        # Fairness metrics
        dp_diff = self._demographic_parity_difference(metrics_0, metrics_1)
        eo_diff = self._equalized_odds_difference(metrics_0, metrics_1)
        di_ratio = self._disparate_impact_ratio(metrics_0, metrics_1)
        eop_diff = self._equal_opportunity_difference(metrics_0, metrics_1)
        ao_diff = self._average_odds_difference(metrics_0, metrics_1)
        
        return FairnessMetrics(
            demographic_parity_difference=dp_diff,
            equalized_odds_difference=eo_diff,
            disparate_impact_ratio=di_ratio,
            equal_opportunity_difference=eop_diff,
            average_odds_difference=ao_diff,
            protected_group_stats={
                f'group_{groups[0]}': metrics_0,
                f'group_{groups[1]}': metrics_1
            }
        )
    
    def _calculate_group_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        favorable_outcome: int
    ) -> Dict[str, float]:
        """Calculate performance metrics for a group"""
        if len(y_true) == 0:
            return {
                'selection_rate': 0.0,
                'tpr': 0.0,
                'fpr': 0.0,
                'tnr': 0.0,
                'fnr': 0.0,
                'precision': 0.0,
                'accuracy': 0.0
            }
        
        # Confusion matrix elements
        tp = np.sum((y_true == favorable_outcome) & (y_pred == favorable_outcome))
        fp = np.sum((y_true != favorable_outcome) & (y_pred == favorable_outcome))
        tn = np.sum((y_true != favorable_outcome) & (y_pred != favorable_outcome))
        fn = np.sum((y_true == favorable_outcome) & (y_pred != favorable_outcome))
        
        # Rates
        selection_rate = np.mean(y_pred == favorable_outcome)
        tpr = tp / (tp + fn + self.epsilon)  # True Positive Rate (Recall)
        fpr = fp / (fp + tn + self.epsilon)  # False Positive Rate
        tnr = tn / (tn + fp + self.epsilon)  # True Negative Rate (Specificity)
        fnr = fn / (fn + tp + self.epsilon)  # False Negative Rate
        precision = tp / (tp + fp + self.epsilon)
        accuracy = (tp + tn) / (tp + tn + fp + fn + self.epsilon)
        
        return {
            'selection_rate': float(selection_rate),
            'tpr': float(tpr),
            'fpr': float(fpr),
            'tnr': float(tnr),
            'fnr': float(fnr),
            'precision': float(precision),
            'accuracy': float(accuracy),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    
    def _demographic_parity_difference(
        self,
        metrics_0: Dict,
        metrics_1: Dict
    ) -> float:
        """
        Demographic Parity Difference
        
        Difference in selection rates between groups.
        Perfect fairness: 0.0
        Range: [-1, 1]
        """
        return abs(metrics_1['selection_rate'] - metrics_0['selection_rate'])
    
    def _disparate_impact_ratio(
        self,
        metrics_0: Dict,
        metrics_1: Dict
    ) -> float:
        """
        Disparate Impact Ratio
        
        Ratio of selection rates (minority / majority).
        Perfect fairness: 1.0
        "80% rule": >= 0.8 is considered fair
        Range: [0, inf]
        """
        if metrics_0['selection_rate'] < self.epsilon:
            return 0.0
        return metrics_1['selection_rate'] / (metrics_0['selection_rate'] + self.epsilon)
    
    def _equalized_odds_difference(
        self,
        metrics_0: Dict,
        metrics_1: Dict
    ) -> float:
        """
        Equalized Odds Difference
        
        Maximum difference in TPR and FPR between groups.
        Perfect fairness: 0.0
        Range: [0, 1]
        """
        tpr_diff = abs(metrics_1['tpr'] - metrics_0['tpr'])
        fpr_diff = abs(metrics_1['fpr'] - metrics_0['fpr'])
        return max(tpr_diff, fpr_diff)
    
    def _equal_opportunity_difference(
        self,
        metrics_0: Dict,
        metrics_1: Dict
    ) -> float:
        """
        Equal Opportunity Difference
        
        Difference in True Positive Rates (recall) between groups.
        Perfect fairness: 0.0
        Range: [0, 1]
        """
        return abs(metrics_1['tpr'] - metrics_0['tpr'])
    
    def _average_odds_difference(
        self,
        metrics_0: Dict,
        metrics_1: Dict
    ) -> float:
        """
        Average Odds Difference
        
        Average of TPR and FPR differences.
        Perfect fairness: 0.0
        Range: [0, 1]
        """
        tpr_diff = abs(metrics_1['tpr'] - metrics_0['tpr'])
        fpr_diff = abs(metrics_1['fpr'] - metrics_0['fpr'])
        return (tpr_diff + fpr_diff) / 2.0
    
    def generate_fairness_report(
        self,
        fairness_metrics: FairnessMetrics,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Generate human-readable fairness report
        
        Args:
            fairness_metrics: FairnessMetrics object
            threshold: Threshold for acceptable difference (default 0.1 = 10%)
        """
        report = {
            'overall_assessment': 'PASS',
            'violations': [],
            'metrics': {},
            'recommendations': []
        }
        
        # Check each metric
        if fairness_metrics.demographic_parity_difference > threshold:
            report['overall_assessment'] = 'FAIL'
            report['violations'].append({
                'metric': 'Demographic Parity',
                'value': fairness_metrics.demographic_parity_difference,
                'threshold': threshold,
                'severity': 'HIGH' if fairness_metrics.demographic_parity_difference > 0.2 else 'MEDIUM'
            })
            report['recommendations'].append(
                "Consider reweighting training samples or using fairness constraints"
            )
        
        if fairness_metrics.disparate_impact_ratio < 0.8 or fairness_metrics.disparate_impact_ratio > 1.25:
            report['overall_assessment'] = 'FAIL'
            report['violations'].append({
                'metric': 'Disparate Impact',
                'value': fairness_metrics.disparate_impact_ratio,
                'threshold': '0.8-1.25',
                'severity': 'HIGH'
            })
            report['recommendations'].append(
                "Model violates the '80% rule' - significant disparate impact detected"
            )
        
        if fairness_metrics.equalized_odds_difference > threshold:
            report['violations'].append({
                'metric': 'Equalized Odds',
                'value': fairness_metrics.equalized_odds_difference,
                'threshold': threshold,
                'severity': 'MEDIUM'
            })
        
        if fairness_metrics.equal_opportunity_difference > threshold:
            report['violations'].append({
                'metric': 'Equal Opportunity',
                'value': fairness_metrics.equal_opportunity_difference,
                'threshold': threshold,
                'severity': 'MEDIUM'
            })
            report['recommendations'].append(
                "Different true positive rates across groups - model may underserve protected group"
            )
        
        # Add all metrics
        report['metrics'] = {
            'demographic_parity_difference': fairness_metrics.demographic_parity_difference,
            'disparate_impact_ratio': fairness_metrics.disparate_impact_ratio,
            'equalized_odds_difference': fairness_metrics.equalized_odds_difference,
            'equal_opportunity_difference': fairness_metrics.equal_opportunity_difference,
            'average_odds_difference': fairness_metrics.average_odds_difference
        }
        
        # Group statistics
        report['group_statistics'] = fairness_metrics.protected_group_stats
        
        return report
    
    def mitigate_bias_reweighting(
        self,
        y: np.ndarray,
        protected_attribute: np.ndarray,
        favorable_outcome: int = 1
    ) -> np.ndarray:
        """
        Generate sample weights to mitigate bias
        
        Returns weights that balance representation across groups
        """
        groups = np.unique(protected_attribute)
        weights = np.ones(len(y))
        
        for group in groups:
            group_mask = protected_attribute == group
            favorable_mask = y == favorable_outcome
            
            # Weight to balance favorable outcomes across groups
            group_favorable_rate = np.mean(y[group_mask] == favorable_outcome)
            overall_favorable_rate = np.mean(y == favorable_outcome)
            
            if group_favorable_rate > 0:
                weight_factor = overall_favorable_rate / (group_favorable_rate + self.epsilon)
                weights[group_mask & favorable_mask] *= weight_factor
        
        # Normalize weights
        weights = weights / np.mean(weights)
        
        logger.info(f"Generated bias mitigation weights (min={weights.min():.3f}, max={weights.max():.3f})")
        return weights
