"""
Evolution Analysis Service

Analyzes multiple photos of the same mole over time to detect changes
using ABCDE criteria as reference. Provides detailed change tracking
and risk assessment based on evolution patterns.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ABCDESnapshot:
    """ABCDE metrics at a specific point in time."""
    timestamp: str
    asymmetry_score: float
    border_score: float
    color_score: float
    diameter_mm: float
    total_score: float
    risk_level: str
    ml_probability: Optional[float] = None

    def to_dict(self):
        return {
            'timestamp': self.timestamp,
            'asymmetry_score': self.asymmetry_score,
            'border_score': self.border_score,
            'color_score': self.color_score,
            'diameter_mm': self.diameter_mm,
            'total_score': self.total_score,
            'risk_level': self.risk_level,
            'ml_probability': self.ml_probability
        }


@dataclass
class EvolutionChange:
    """Represents a change in a specific ABCDE criterion."""
    criterion: str  # 'asymmetry', 'border', 'color', 'diameter', 'overall'
    initial_value: float
    current_value: float
    change_amount: float
    change_percent: float
    trend: str  # 'increasing', 'decreasing', 'stable'
    risk_impact: str  # 'positive' (better), 'negative' (worse), 'neutral'
    description: str

    def to_dict(self):
        return {
            'criterion': self.criterion,
            'initial_value': self.initial_value,
            'current_value': self.current_value,
            'change_amount': self.change_amount,
            'change_percent': self.change_percent,
            'trend': self.trend,
            'risk_impact': self.risk_impact,
            'description': self.description
        }


@dataclass
class EvolutionReport:
    """Complete evolution analysis report."""
    success: bool
    mole_id: Optional[str] = None
    analysis_date: str = ""
    total_images_analyzed: int = 0
    time_span_days: int = 0

    # ABCDE snapshots over time
    snapshots: List[ABCDESnapshot] = field(default_factory=list)

    # Detailed changes
    changes: List[EvolutionChange] = field(default_factory=list)

    # Overall assessment
    evolution_detected: bool = False
    evolution_score: float = 0.0  # 0-1, higher = more change
    risk_trajectory: str = "stable"  # 'improving', 'stable', 'worsening'

    # ABCDE-specific evolution
    abcde_evolution: Dict[str, Any] = field(default_factory=dict)

    # Recommendations
    significant_changes: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    requires_attention: bool = False
    urgent_referral: bool = False

    error: Optional[str] = None

    def to_dict(self):
        return {
            'success': self.success,
            'mole_id': self.mole_id,
            'analysis_date': self.analysis_date,
            'total_images_analyzed': self.total_images_analyzed,
            'time_span_days': self.time_span_days,
            'snapshots': [s.to_dict() for s in self.snapshots],
            'changes': [c.to_dict() for c in self.changes],
            'evolution_detected': self.evolution_detected,
            'evolution_score': self.evolution_score,
            'risk_trajectory': self.risk_trajectory,
            'abcde_evolution': self.abcde_evolution,
            'significant_changes': self.significant_changes,
            'recommendations': self.recommendations,
            'requires_attention': self.requires_attention,
            'urgent_referral': self.urgent_referral,
            'error': self.error
        }


class EvolutionAnalysisService:
    """
    Service for analyzing mole evolution over time using ABCDE criteria.

    The 'E' in ABCDE stands for Evolution - changes over time are one of the
    most important indicators for melanoma risk assessment.
    """

    # Thresholds for significant changes
    ASYMMETRY_CHANGE_THRESHOLD = 0.15  # 15% change is significant
    BORDER_CHANGE_THRESHOLD = 0.15
    COLOR_CHANGE_THRESHOLD = 0.15
    DIAMETER_CHANGE_THRESHOLD_MM = 1.0  # 1mm growth is significant
    DIAMETER_CHANGE_THRESHOLD_PERCENT = 0.20  # 20% size change
    OVERALL_CHANGE_THRESHOLD = 0.10

    # Risk weights for evolution
    EVOLUTION_WEIGHTS = {
        'diameter_increase': 0.30,  # Size increase is very concerning
        'asymmetry_increase': 0.25,
        'border_increase': 0.20,
        'color_increase': 0.25
    }

    def __init__(self):
        self.abcde_analyzer = None
        self.mole_analysis_service = None
        self.image_processor = None

    def _get_abcde_analyzer(self):
        """Lazy load ABCDE analyzer."""
        if self.abcde_analyzer is None:
            from .ABCDEAnalyzer import ABCDEAnalyzer
            self.abcde_analyzer = ABCDEAnalyzer()
        return self.abcde_analyzer

    def _get_analysis_service(self):
        """Lazy load mole analysis service."""
        if self.mole_analysis_service is None:
            from .MoleAnalysisService import MoleAnalysisService
            self.mole_analysis_service = MoleAnalysisService()
        return self.mole_analysis_service

    def _get_image_processor(self):
        """Lazy load image processor."""
        if self.image_processor is None:
            from .ImageProcessor import ImageProcessor
            self.image_processor = ImageProcessor()
        return self.image_processor

    def analyze_evolution(self, images: List[Dict[str, Any]],
                          mole_id: Optional[str] = None,
                          reference_mm: Optional[float] = None) -> EvolutionReport:
        """
        Analyze evolution of a mole across multiple images over time.

        Args:
            images: List of dicts with 'image_base64' and 'timestamp' (ISO format)
                   Sorted from oldest to newest.
            mole_id: Optional identifier for the mole
            reference_mm: Reference object size in mm for diameter calculation

        Returns:
            EvolutionReport with detailed analysis
        """
        report = EvolutionReport(
            success=False,
            mole_id=mole_id,
            analysis_date=datetime.now().isoformat()
        )

        if len(images) < 2:
            report.error = "At least 2 images are required for evolution analysis"
            return report

        try:
            # Sort images by timestamp
            sorted_images = sorted(images, key=lambda x: x.get('timestamp', ''))

            # Analyze each image
            snapshots = []
            analysis_service = self._get_analysis_service()

            for img_data in sorted_images:
                image_base64 = img_data.get('image_base64')
                timestamp = img_data.get('timestamp', datetime.now().isoformat())

                if not image_base64:
                    continue

                # Perform comprehensive analysis
                analysis = analysis_service.analyze_mole(
                    image_base64,
                    include_abcde=True,
                    include_ml=True,
                    reference_mm=reference_mm
                )

                if analysis and analysis.abcde_score:
                    snapshot = ABCDESnapshot(
                        timestamp=timestamp,
                        asymmetry_score=analysis.abcde_score.asymmetry_score,
                        border_score=analysis.abcde_score.border_score,
                        color_score=analysis.abcde_score.color_score,
                        diameter_mm=analysis.abcde_score.diameter_mm,
                        total_score=analysis.abcde_score.total_score,
                        risk_level=analysis.abcde_score.risk_level,
                        ml_probability=analysis.melanoma_probability
                    )
                    snapshots.append(snapshot)

            if len(snapshots) < 2:
                report.error = "Could not analyze enough images. Please ensure images are clear."
                return report

            report.snapshots = snapshots
            report.total_images_analyzed = len(snapshots)

            # Calculate time span
            try:
                first_date = datetime.fromisoformat(snapshots[0].timestamp.replace('Z', '+00:00'))
                last_date = datetime.fromisoformat(snapshots[-1].timestamp.replace('Z', '+00:00'))
                report.time_span_days = (last_date - first_date).days
            except:
                report.time_span_days = 0

            # Analyze changes between first and last snapshot
            first = snapshots[0]
            last = snapshots[-1]

            changes = self._analyze_changes(first, last, report.time_span_days)
            report.changes = changes

            # Calculate evolution metrics
            report.evolution_detected = any(
                abs(c.change_percent) >= 10 for c in changes
            )

            report.evolution_score = self._calculate_evolution_score(changes)

            # Determine risk trajectory
            report.risk_trajectory = self._determine_trajectory(changes, snapshots)

            # Build ABCDE evolution summary
            report.abcde_evolution = self._build_abcde_evolution(changes, snapshots)

            # Generate significant changes list
            report.significant_changes = self._identify_significant_changes(changes)

            # Generate recommendations
            report.recommendations = self._generate_recommendations(
                changes, report.evolution_score, report.time_span_days
            )

            # Determine if attention/referral needed
            report.requires_attention = (
                report.evolution_score > 0.3 or
                report.risk_trajectory == 'worsening' or
                last.risk_level in ['high', 'very_high']
            )

            report.urgent_referral = (
                report.evolution_score > 0.5 or
                (report.risk_trajectory == 'worsening' and last.risk_level == 'very_high') or
                any('rapid' in c.description.lower() for c in changes)
            )

            report.success = True

        except Exception as e:
            logger.error(f"Evolution analysis failed: {e}")
            report.error = str(e)

        return report

    def _analyze_changes(self, first: ABCDESnapshot, last: ABCDESnapshot,
                         days: int) -> List[EvolutionChange]:
        """Analyze changes between two snapshots."""
        changes = []

        # Asymmetry change
        asym_change = last.asymmetry_score - first.asymmetry_score
        asym_percent = (asym_change / max(first.asymmetry_score, 0.01)) * 100
        changes.append(EvolutionChange(
            criterion='asymmetry',
            initial_value=first.asymmetry_score,
            current_value=last.asymmetry_score,
            change_amount=asym_change,
            change_percent=asym_percent,
            trend=self._get_trend(asym_change),
            risk_impact='negative' if asym_change > self.ASYMMETRY_CHANGE_THRESHOLD else 'neutral',
            description=self._describe_asymmetry_change(asym_change, days)
        ))

        # Border change
        border_change = last.border_score - first.border_score
        border_percent = (border_change / max(first.border_score, 0.01)) * 100
        changes.append(EvolutionChange(
            criterion='border',
            initial_value=first.border_score,
            current_value=last.border_score,
            change_amount=border_change,
            change_percent=border_percent,
            trend=self._get_trend(border_change),
            risk_impact='negative' if border_change > self.BORDER_CHANGE_THRESHOLD else 'neutral',
            description=self._describe_border_change(border_change, days)
        ))

        # Color change
        color_change = last.color_score - first.color_score
        color_percent = (color_change / max(first.color_score, 0.01)) * 100
        changes.append(EvolutionChange(
            criterion='color',
            initial_value=first.color_score,
            current_value=last.color_score,
            change_amount=color_change,
            change_percent=color_percent,
            trend=self._get_trend(color_change),
            risk_impact='negative' if color_change > self.COLOR_CHANGE_THRESHOLD else 'neutral',
            description=self._describe_color_change(color_change, days)
        ))

        # Diameter change
        diameter_change = last.diameter_mm - first.diameter_mm
        diameter_percent = (diameter_change / max(first.diameter_mm, 0.1)) * 100
        diameter_risk = 'negative' if (
            diameter_change > self.DIAMETER_CHANGE_THRESHOLD_MM or
            diameter_percent > self.DIAMETER_CHANGE_THRESHOLD_PERCENT * 100
        ) else 'neutral'
        changes.append(EvolutionChange(
            criterion='diameter',
            initial_value=first.diameter_mm,
            current_value=last.diameter_mm,
            change_amount=diameter_change,
            change_percent=diameter_percent,
            trend=self._get_trend(diameter_change),
            risk_impact=diameter_risk,
            description=self._describe_diameter_change(diameter_change, diameter_percent, days)
        ))

        # Overall change
        overall_change = last.total_score - first.total_score
        overall_percent = (overall_change / max(first.total_score, 0.01)) * 100
        changes.append(EvolutionChange(
            criterion='overall',
            initial_value=first.total_score,
            current_value=last.total_score,
            change_amount=overall_change,
            change_percent=overall_percent,
            trend=self._get_trend(overall_change),
            risk_impact='negative' if overall_change > self.OVERALL_CHANGE_THRESHOLD else 'neutral',
            description=self._describe_overall_change(overall_change, first.risk_level, last.risk_level)
        ))

        return changes

    def _get_trend(self, change: float) -> str:
        """Determine trend based on change amount."""
        if change > 0.05:
            return 'increasing'
        elif change < -0.05:
            return 'decreasing'
        return 'stable'

    def _describe_asymmetry_change(self, change: float, days: int) -> str:
        """Generate description for asymmetry change."""
        if abs(change) < 0.05:
            return "Asymmetry has remained stable."
        elif change > 0:
            if days > 0 and change / (days / 30) > 0.1:
                return f"Asymmetry has increased rapidly (significant change in {days} days)."
            return "Asymmetry has increased, indicating the mole shape is becoming less uniform."
        else:
            return "Asymmetry has decreased slightly."

    def _describe_border_change(self, change: float, days: int) -> str:
        """Generate description for border change."""
        if abs(change) < 0.05:
            return "Border regularity has remained stable."
        elif change > 0:
            if days > 0 and change / (days / 30) > 0.1:
                return f"Border has become more irregular rapidly ({days} days)."
            return "Border has become more irregular or uneven."
        else:
            return "Border has become slightly more regular."

    def _describe_color_change(self, change: float, days: int) -> str:
        """Generate description for color change."""
        if abs(change) < 0.05:
            return "Color variation has remained stable."
        elif change > 0:
            if days > 0 and change / (days / 30) > 0.1:
                return f"Color variation has increased rapidly ({days} days)."
            return "Color has become more varied or new colors have appeared."
        else:
            return "Color variation has decreased slightly."

    def _describe_diameter_change(self, change_mm: float, change_percent: float, days: int) -> str:
        """Generate description for diameter change."""
        if abs(change_mm) < 0.5:
            return "Size has remained stable."
        elif change_mm > 0:
            growth_desc = f"grown by {change_mm:.1f}mm ({change_percent:.0f}%)"
            if days > 0:
                monthly_growth = change_mm / (days / 30)
                if monthly_growth > 0.5:
                    return f"The mole has {growth_desc}. This is rapid growth that warrants attention."
            return f"The mole has {growth_desc}."
        else:
            return f"The mole has shrunk by {abs(change_mm):.1f}mm."

    def _describe_overall_change(self, change: float, initial_risk: str, current_risk: str) -> str:
        """Generate description for overall change."""
        if initial_risk != current_risk:
            return f"Risk level has changed from {initial_risk} to {current_risk}."
        elif abs(change) < 0.05:
            return f"Overall risk score has remained stable at {current_risk} level."
        elif change > 0:
            return f"Overall risk score has increased while remaining at {current_risk} level."
        else:
            return f"Overall risk score has slightly improved."

    def _calculate_evolution_score(self, changes: List[EvolutionChange]) -> float:
        """Calculate overall evolution score (0-1)."""
        weighted_sum = 0.0

        for change in changes:
            if change.criterion == 'diameter' and change.change_amount > 0:
                weighted_sum += self.EVOLUTION_WEIGHTS['diameter_increase'] * min(1.0, abs(change.change_percent) / 50)
            elif change.criterion == 'asymmetry' and change.change_amount > 0:
                weighted_sum += self.EVOLUTION_WEIGHTS['asymmetry_increase'] * min(1.0, abs(change.change_amount) / 0.3)
            elif change.criterion == 'border' and change.change_amount > 0:
                weighted_sum += self.EVOLUTION_WEIGHTS['border_increase'] * min(1.0, abs(change.change_amount) / 0.3)
            elif change.criterion == 'color' and change.change_amount > 0:
                weighted_sum += self.EVOLUTION_WEIGHTS['color_increase'] * min(1.0, abs(change.change_amount) / 0.3)

        return min(1.0, weighted_sum)

    def _determine_trajectory(self, changes: List[EvolutionChange],
                              snapshots: List[ABCDESnapshot]) -> str:
        """Determine the overall risk trajectory."""
        negative_changes = sum(1 for c in changes if c.risk_impact == 'negative')
        positive_changes = sum(1 for c in changes if c.risk_impact == 'positive')

        # Check if risk level has changed
        if len(snapshots) >= 2:
            risk_levels = ['low', 'moderate', 'high', 'very_high']
            first_risk_idx = risk_levels.index(snapshots[0].risk_level) if snapshots[0].risk_level in risk_levels else 0
            last_risk_idx = risk_levels.index(snapshots[-1].risk_level) if snapshots[-1].risk_level in risk_levels else 0

            if last_risk_idx > first_risk_idx:
                return 'worsening'
            elif last_risk_idx < first_risk_idx:
                return 'improving'

        if negative_changes >= 2:
            return 'worsening'
        elif positive_changes >= 2:
            return 'improving'

        return 'stable'

    def _build_abcde_evolution(self, changes: List[EvolutionChange],
                               snapshots: List[ABCDESnapshot]) -> Dict[str, Any]:
        """Build detailed ABCDE evolution summary."""
        evolution = {
            'A_asymmetry': {
                'description': 'Asymmetry measures how different one half of the mole is from the other.',
                'initial': None,
                'current': None,
                'trend': None,
                'concern_level': 'low'
            },
            'B_border': {
                'description': 'Border irregularity indicates how uneven or ragged the edges are.',
                'initial': None,
                'current': None,
                'trend': None,
                'concern_level': 'low'
            },
            'C_color': {
                'description': 'Color variation shows how many different colors or shades are present.',
                'initial': None,
                'current': None,
                'trend': None,
                'concern_level': 'low'
            },
            'D_diameter': {
                'description': 'Diameter measures the size of the mole. >6mm is a concern.',
                'initial': None,
                'current': None,
                'trend': None,
                'concern_level': 'low'
            },
            'E_evolution': {
                'description': 'Evolution tracks changes over time - one of the most important indicators.',
                'summary': None,
                'concern_level': 'low'
            }
        }

        for change in changes:
            if change.criterion == 'asymmetry':
                evolution['A_asymmetry']['initial'] = change.initial_value
                evolution['A_asymmetry']['current'] = change.current_value
                evolution['A_asymmetry']['trend'] = change.trend
                if change.risk_impact == 'negative':
                    evolution['A_asymmetry']['concern_level'] = 'moderate' if change.change_percent < 30 else 'high'

            elif change.criterion == 'border':
                evolution['B_border']['initial'] = change.initial_value
                evolution['B_border']['current'] = change.current_value
                evolution['B_border']['trend'] = change.trend
                if change.risk_impact == 'negative':
                    evolution['B_border']['concern_level'] = 'moderate' if change.change_percent < 30 else 'high'

            elif change.criterion == 'color':
                evolution['C_color']['initial'] = change.initial_value
                evolution['C_color']['current'] = change.current_value
                evolution['C_color']['trend'] = change.trend
                if change.risk_impact == 'negative':
                    evolution['C_color']['concern_level'] = 'moderate' if change.change_percent < 30 else 'high'

            elif change.criterion == 'diameter':
                evolution['D_diameter']['initial'] = change.initial_value
                evolution['D_diameter']['current'] = change.current_value
                evolution['D_diameter']['trend'] = change.trend
                if change.risk_impact == 'negative':
                    evolution['D_diameter']['concern_level'] = 'moderate' if change.change_percent < 30 else 'high'

            elif change.criterion == 'overall':
                evolution['E_evolution']['summary'] = change.description
                if change.risk_impact == 'negative':
                    evolution['E_evolution']['concern_level'] = 'moderate' if change.change_percent < 20 else 'high'

        return evolution

    def _identify_significant_changes(self, changes: List[EvolutionChange]) -> List[str]:
        """Identify and list significant changes."""
        significant = []

        for change in changes:
            if change.criterion == 'overall':
                continue

            if change.risk_impact == 'negative':
                significant.append(f"{change.criterion.upper()}: {change.description}")
            elif abs(change.change_percent) > 20:
                significant.append(f"{change.criterion.upper()}: Notable change detected - {change.description}")

        return significant

    def _generate_recommendations(self, changes: List[EvolutionChange],
                                   evolution_score: float, days: int) -> List[str]:
        """Generate recommendations based on evolution analysis."""
        recommendations = []

        # Check for rapid changes
        rapid_change = any(
            c.risk_impact == 'negative' and days > 0 and abs(c.change_percent) / (days / 30) > 20
            for c in changes
        )

        if rapid_change:
            recommendations.append(
                "URGENT: Rapid changes detected. Please consult a dermatologist as soon as possible."
            )

        # Check evolution score
        if evolution_score > 0.5:
            recommendations.append(
                "Significant evolution detected. Professional examination is strongly recommended."
            )
        elif evolution_score > 0.3:
            recommendations.append(
                "Moderate changes observed. Consider scheduling a dermatologist appointment."
            )

        # Specific criterion recommendations
        for change in changes:
            if change.criterion == 'diameter' and change.change_amount > self.DIAMETER_CHANGE_THRESHOLD_MM:
                recommendations.append(
                    f"Size has increased by {change.change_amount:.1f}mm. Monitor closely and consult if growth continues."
                )
            elif change.criterion == 'color' and change.risk_impact == 'negative':
                recommendations.append(
                    "New colors or increased color variation detected. This warrants professional evaluation."
                )
            elif change.criterion == 'border' and change.risk_impact == 'negative':
                recommendations.append(
                    "Border has become more irregular. Consider having this checked by a dermatologist."
                )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "Continue regular monitoring. Take photos monthly to track any changes."
            )
        else:
            recommendations.append(
                "Continue taking regular photos to track changes over time."
            )

        # Medical disclaimer
        recommendations.append(
            "DISCLAIMER: This analysis is for informational purposes only and is not a medical diagnosis. "
            "Always consult a qualified healthcare professional for medical advice."
        )

        return recommendations


# Singleton instance
_evolution_service = None


def get_evolution_service() -> EvolutionAnalysisService:
    """Get singleton instance of EvolutionAnalysisService."""
    global _evolution_service
    if _evolution_service is None:
        _evolution_service = EvolutionAnalysisService()
    return _evolution_service
