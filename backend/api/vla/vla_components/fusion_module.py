"""
Enhanced fusion module for Vision-Language-Action (VLA) systems
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging


class FusionMethod(Enum):
    """Types of multimodal fusion methods"""
    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    CROSS_ATTENTION = "cross_attention"
    CONCATENATION = "concatenation"
    DYNAMIC_FUSION = "dynamic_fusion"


class FusionModule:
    """Enhanced multimodal fusion module for combining vision and language features"""

    def __init__(self, fusion_method: FusionMethod = FusionMethod.CROSS_ATTENTION):
        self.fusion_method = fusion_method
        self.logger = logging.getLogger(__name__)

        # Initialize fusion components
        self._initialize_fusion_components()

    def _initialize_fusion_components(self):
        """Initialize fusion-specific components"""
        self.feature_dim = 512  # Base dimension for fusion
        self.attention_heads = 8
        self.fusion_weights = np.ones(2)  # Equal weights for vision and language initially

    def fuse_features(self, vision_features: Dict[str, Any],
                     language_features: Dict[str, Any],
                     method: Optional[FusionMethod] = None) -> Dict[str, Any]:
        """Fuse vision and language features using specified method"""
        try:
            fusion_method = method or self.fusion_method

            if fusion_method == FusionMethod.EARLY_FUSION:
                fused_features, attention_weights = self._early_fusion(vision_features, language_features)
            elif fusion_method == FusionMethod.LATE_FUSION:
                fused_features, attention_weights = self._late_fusion(vision_features, language_features)
            elif fusion_method == FusionMethod.CROSS_ATTENTION:
                fused_features, attention_weights = self._cross_attention_fusion(vision_features, language_features)
            elif fusion_method == FusionMethod.CONCATENATION:
                fused_features, attention_weights = self._concatenation_fusion(vision_features, language_features)
            elif fusion_method == FusionMethod.DYNAMIC_FUSION:
                fused_features, attention_weights = self._dynamic_fusion(vision_features, language_features)
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")

            return {
                "fused_features": fused_features,
                "attention_weights": attention_weights,
                "fusion_method_used": fusion_method.value,
                "confidence": self._calculate_fusion_confidence(vision_features, language_features)
            }
        except Exception as e:
            self.logger.error(f"Error in feature fusion: {str(e)}")
            raise

    def _early_fusion(self, vision_features: Dict[str, Any],
                     language_features: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Concatenate features early in the pipeline"""
        # Extract visual features
        vis_features = self._extract_visual_features(vision_features)
        lang_features = self._extract_language_features(language_features)

        # Normalize to same length
        min_len = min(len(vis_features), len(lang_features))
        vis_features = vis_features[:min_len]
        lang_features = lang_features[:min_len]

        # Concatenate features
        fused_features = np.concatenate([vis_features, lang_features])

        # No attention weights for early fusion
        return fused_features, None

    def _late_fusion(self, vision_features: Dict[str, Any],
                    language_features: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Combine decisions from separate pathways"""
        # Extract and process features separately
        vis_features = self._extract_visual_features(vision_features)
        lang_features = self._extract_language_features(language_features)

        # Normalize to same length
        min_len = min(len(vis_features), len(lang_features))
        vis_part = vis_features[:min_len//2] if len(vis_features) >= min_len//2 else vis_features
        lang_part = lang_features[:min_len//2] if len(lang_features) >= min_len//2 else lang_features

        # Pad if needed
        target_len = min_len // 2
        if len(vis_part) < target_len:
            vis_part = np.pad(vis_part, (0, target_len - len(vis_part)))
        if len(lang_part) < target_len:
            lang_part = np.pad(lang_part, (0, target_len - len(lang_part)))

        # Weighted combination
        weights = self.fusion_weights / np.sum(self.fusion_weights)
        combined = weights[0] * vis_part + weights[1] * lang_part

        return combined, self.fusion_weights

    def _cross_attention_fusion(self, vision_features: Dict[str, Any],
                               language_features: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Use attention mechanism to combine features"""
        # Extract features
        vis_features = self._extract_visual_features(vision_features)
        lang_features = self._extract_language_features(language_features)

        # Normalize to same length
        min_len = min(len(vis_features), len(lang_features))
        vis_features = vis_features[:min_len]
        lang_features = lang_features[:min_len]

        # Normalize features
        vis_norm = vis_features / (np.linalg.norm(vis_features) + 1e-8)
        lang_norm = lang_features / (np.linalg.norm(lang_features) + 1e-8)

        # Compute attention weights using dot product (similarity)
        attention_weights = vis_norm * lang_norm  # Element-wise multiplication for attention
        attention_weights = np.clip(attention_weights, 0, 1)  # Normalize to [0,1]

        # Apply attention-weighted combination
        weighted_vis = vis_features * np.mean(attention_weights)
        weighted_lang = lang_features * (1 - np.mean(attention_weights))
        fused_features = 0.6 * weighted_vis + 0.4 * weighted_lang

        return fused_features, attention_weights

    def _concatenation_fusion(self, vision_features: Dict[str, Any],
                             language_features: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Simple concatenation of features"""
        vis_features = self._extract_visual_features(vision_features)
        lang_features = self._extract_language_features(language_features)

        # Concatenate features
        fused_features = np.concatenate([vis_features, lang_features])

        return fused_features, None

    def _dynamic_fusion(self, vision_features: Dict[str, Any],
                       language_features: Dict[str, Any]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Dynamic fusion based on task and context"""
        # Extract features
        vis_features = self._extract_visual_features(vision_features)
        lang_features = self._extract_language_features(language_features)

        # Normalize to same length
        min_len = min(len(vis_features), len(lang_features))
        vis_features = vis_features[:min_len]
        lang_features = lang_features[:min_len]

        # Calculate dynamic weights based on feature quality/confidence
        vis_quality = self._assess_feature_quality(vis_features)
        lang_quality = self._assess_feature_quality(lang_features)

        # Normalize qualities
        total_quality = vis_quality + lang_quality
        if total_quality > 0:
            vis_weight = vis_quality / total_quality
            lang_weight = lang_quality / total_quality
        else:
            vis_weight, lang_weight = 0.5, 0.5

        # Weighted combination
        fused_features = vis_weight * vis_features + lang_weight * lang_features
        attention_weights = np.array([vis_weight, lang_weight])

        return fused_features, attention_weights

    def _extract_visual_features(self, vision_features: Dict[str, Any]) -> np.ndarray:
        """Extract visual features for fusion"""
        # Try different feature sources in order of preference
        if "image_features" in vision_features:
            features = vision_features["image_features"]
        elif "spatial_features" in vision_features:
            features = vision_features["spatial_features"]
        elif "semantic_features" in vision_features:
            features = vision_features["semantic_features"]
        else:
            # Default: random features for simulation
            features = np.random.random(self.feature_dim).astype(np.float32)

        # Ensure it's a numpy array
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        return features.astype(np.float32)

    def _extract_language_features(self, language_features: Dict[str, Any]) -> np.ndarray:
        """Extract language features for fusion"""
        # Try different feature sources in order of preference
        if "text_embeddings" in language_features:
            features = language_features["text_embeddings"]
        elif "semantic_meaning" in language_features:
            # Convert semantic meaning to features (simulated)
            features = np.random.random(self.feature_dim).astype(np.float32)
        else:
            # Default: random features for simulation
            features = np.random.random(self.feature_dim).astype(np.float32)

        # Ensure it's a numpy array
        if not isinstance(features, np.ndarray):
            features = np.array(features)

        return features.astype(np.float32)

    def _assess_feature_quality(self, features: np.ndarray) -> float:
        """Assess the quality of features (simulated)"""
        # Calculate quality based on feature statistics
        if len(features) == 0:
            return 0.0

        # Variance-based quality (higher variance indicates more information)
        variance = np.var(features)
        # Normalize to [0, 1]
        quality = min(1.0, variance * 10)  # Scale factor for normalization

        return quality

    def _calculate_fusion_confidence(self, vision_features: Dict[str, Any],
                                   language_features: Dict[str, Any]) -> float:
        """Calculate confidence in the fusion result"""
        # Base confidence from feature qualities
        vis_features = self._extract_visual_features(vision_features)
        lang_features = self._extract_language_features(language_features)

        vis_quality = self._assess_feature_quality(vis_features)
        lang_quality = self._assess_feature_quality(lang_features)

        # Average quality as confidence
        confidence = (vis_quality + lang_quality) / 2.0

        # Boost if both modalities provide good features
        if vis_quality > 0.5 and lang_quality > 0.5:
            confidence *= 1.2  # Boost for good multimodal input

        return min(1.0, confidence)

    def adapt_fusion_weights(self, vision_features: Dict[str, Any],
                           language_features: Dict[str, Any],
                           task_context: Optional[Dict[str, Any]] = None) -> None:
        """Adapt fusion weights based on current context"""
        try:
            # Assess quality of each modality
            vis_quality = self._assess_feature_quality(
                self._extract_visual_features(vision_features)
            )
            lang_quality = self._assess_feature_quality(
                self._extract_language_features(language_features)
            )

            # Adjust weights based on quality
            if vis_quality > lang_quality:
                self.fusion_weights = np.array([0.7, 0.3])  # Favor vision
            elif lang_quality > vis_quality:
                self.fusion_weights = np.array([0.3, 0.7])  # Favor language
            else:
                self.fusion_weights = np.array([0.5, 0.5])  # Equal weights

            # Further adjust based on task context
            if task_context:
                task_type = task_context.get("task_type", "")
                if task_type in ["navigation", "grasping", "manipulation"]:
                    # Vision is more important for spatial tasks
                    self.fusion_weights = np.array([0.6, 0.4])
                elif task_type in ["communication", "instruction_following"]:
                    # Language is more important for instruction tasks
                    self.fusion_weights = np.array([0.4, 0.6])

        except Exception as e:
            self.logger.error(f"Error adapting fusion weights: {str(e)}")
            # Keep original weights if adaptation fails

    def multi_modal_attention(self, features_list: List[np.ndarray],
                            modalities: List[str]) -> Dict[str, Any]:
        """Perform attention across multiple modalities"""
        try:
            # Normalize all feature vectors to same length
            min_len = min(len(feat) for feat in features_list)
            normalized_features = [feat[:min_len] for feat in features_list]

            # Compute attention weights between all modalities
            attention_matrix = np.zeros((len(modalities), len(modalities)))
            for i, feat1 in enumerate(normalized_features):
                for j, feat2 in enumerate(normalized_features):
                    if i != j:
                        # Compute similarity (dot product)
                        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-8)
                        attention_matrix[i, j] = similarity

            # Compute attended features for each modality
            attended_features = []
            for i, feat in enumerate(normalized_features):
                # Weight features by attention from other modalities
                weights = attention_matrix[i, :]  # Attention from this modality to others
                attended = feat.copy()
                for j, other_feat in enumerate(normalized_features):
                    if i != j:
                        attended += weights[j] * other_feat
                attended_features.append(attended)

            # Fuse all attended features
            fused_features = np.mean(attended_features, axis=0)

            return {
                "fused_features": fused_features,
                "attention_matrix": attention_matrix.tolist(),
                "attended_features": [f.tolist() for f in attended_features],
                "modality_weights": np.mean(attention_matrix, axis=1).tolist()
            }
        except Exception as e:
            self.logger.error(f"Error in multi-modal attention: {str(e)}")
            raise

    def uncertainty_aware_fusion(self, vision_features: Dict[str, Any],
                               language_features: Dict[str, Any],
                               vision_uncertainty: Optional[float] = None,
                               language_uncertainty: Optional[float] = None) -> Dict[str, Any]:
        """Perform fusion that accounts for uncertainty in each modality"""
        try:
            # Extract features
            vis_features = self._extract_visual_features(vision_features)
            lang_features = self._extract_language_features(language_features)

            # Get uncertainty values
            if vision_uncertainty is None:
                # Estimate from feature quality
                vis_quality = self._assess_feature_quality(vis_features)
                vision_uncertainty = 1.0 - vis_quality

            if language_uncertainty is None:
                # Estimate from feature quality
                lang_quality = self._assess_feature_quality(lang_features)
                language_uncertainty = 1.0 - lang_quality

            # Compute confidence (inverse of uncertainty)
            vis_confidence = 1.0 - vision_uncertainty
            lang_confidence = 1.0 - language_uncertainty

            # Normalize confidences
            total_confidence = vis_confidence + lang_confidence
            if total_confidence > 0:
                vis_weight = vis_confidence / total_confidence
                lang_weight = lang_confidence / total_confidence
            else:
                vis_weight, lang_weight = 0.5, 0.5

            # Weighted fusion
            min_len = min(len(vis_features), len(lang_features))
            vis_part = vis_features[:min_len]
            lang_part = lang_features[:min_len]

            fused_features = vis_weight * vis_part + lang_weight * lang_part

            return {
                "fused_features": fused_features,
                "fusion_weights": [float(vis_weight), float(lang_weight)],
                "input_uncertainties": [float(vision_uncertainty), float(language_uncertainty)],
                "confidence_scores": [float(vis_confidence), float(lang_confidence)],
                "fusion_method_used": "uncertainty_aware"
            }
        except Exception as e:
            self.logger.error(f"Error in uncertainty-aware fusion: {str(e)}")
            raise

    def temporal_fusion(self, vision_sequence: List[Dict[str, Any]],
                       language_sequence: List[Dict[str, Any]],
                       temporal_weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """Fuse features across time steps"""
        try:
            if temporal_weights is None:
                # Equal weights for all time steps
                temporal_weights = [1.0 / len(vision_sequence)] * len(vision_sequence)

            # Fuse each time step
            fused_timesteps = []
            attention_weights_timesteps = []

            for i, (vis_feat, lang_feat) in enumerate(zip(vision_sequence, language_sequence)):
                fusion_result = self.fuse_features(vis_feat, lang_feat)
                fused_timesteps.append(fusion_result["fused_features"])
                attention_weights_timesteps.append(fusion_result.get("attention_weights"))

            # Apply temporal weights
            fused_timesteps = np.array(fused_timesteps)
            temporal_weights = np.array(temporal_weights)

            # Weighted average across time
            if len(temporal_weights) == len(fused_timesteps):
                weighted_features = np.average(fused_timesteps, axis=0, weights=temporal_weights)
            else:
                # If weights don't match, use equal weighting
                weighted_features = np.mean(fused_timesteps, axis=0)

            return {
                "temporally_fused_features": weighted_features,
                "timestep_features": [f.tolist() for f in fused_timesteps],
                "temporal_weights": temporal_weights.tolist(),
                "fusion_method_used": "temporal"
            }
        except Exception as e:
            self.logger.error(f"Error in temporal fusion: {str(e)}")
            raise


# Singleton instance
fusion_module = FusionModule()