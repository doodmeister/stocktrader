"""
patterns.py

Production-grade candlestick pattern detection module.
- Modular, extensible, and robust pattern detection
- Comprehensive input validation, logging, and error handling
- Optimized for performance with caching and vectorized operations
- Type-safe with comprehensive documentation
- Designed for integration with ML and rule-based trading pipelines
"""

import functools
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Callable, Tuple, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any

from utils.logger import setup_logger
from core.validation.dataframe_validation_logic import perform_dataframe_validation_logic

logger = setup_logger(__name__)

class PatternType(Enum):
    """Enumeration of pattern types for better categorization."""
    BULLISH_REVERSAL = "bullish_reversal"
    BEARISH_REVERSAL = "bearish_reversal"
    CONTINUATION = "continuation"
    INDECISION = "indecision"

class PatternStrength(Enum):
    """Pattern reliability strength indicators."""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

@dataclass(frozen=True)
class PatternResult:
    """Immutable result object for pattern detection."""
    name: str
    detected: bool
    confidence: float  # 0.0 to 1.0
    pattern_type: PatternType
    strength: PatternStrength
    description: str
    min_rows_required: int
    
    def __post_init__(self):
        """Validate confidence range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

class PatternDetectionError(Exception):
    """Custom exception for pattern detection errors."""
    
    def __init__(self, message: str, pattern_name: Optional[str] = None, error_code: Optional[str] = None):
        super().__init__(message)
        self.pattern_name = pattern_name
        self.error_code = error_code
        self.message = message

class DataValidationError(PatternDetectionError):
    """Specific exception for data validation issues."""
    pass

class PatternConfigurationError(PatternDetectionError):
    """Exception for pattern configuration issues."""
    pass

def validate_dataframe(func: Callable) -> Callable:
    """
    Enhanced decorator for comprehensive DataFrame validation using centralized validation system.
    
    This decorator now leverages the enterprise-grade validation from core.data_validator.py
    while maintaining backward compatibility with the existing decorator interface.
    
    Features:
    - Comprehensive OHLC relationship validation
    - Statistical anomaly detection for better pattern reliability
    - Performance optimization with caching
    - Rich error reporting and metadata
    - Configurable validation levels
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Handle both static functions and instance methods
        if len(args) >= 1:
            # For instance methods, df is the second argument (after self)
            if hasattr(args[0], '__class__') and len(args) >= 2:
                df = args[1]  # Instance method: args[0] is self, args[1] is df
            else:
                df = args[0]  # Static function: args[0] is df
        elif 'df' in kwargs:
            df = kwargs['df']
        else:
            raise DataValidationError(
                "No DataFrame parameter found",
                error_code="MISSING_DATAFRAME_PARAM"
            )
          # Use centralized validation system for comprehensive validation
        try:
            # Determine minimum rows required for this pattern
            # For instance methods, get the pattern's min_rows requirement
            min_rows_required = 1  # Default minimum for patterns
            if len(args) >= 1 and hasattr(args[0], '__class__') and hasattr(args[0], 'min_rows'):
                min_rows_required = args[0].min_rows
            elif len(args) >= 1 and hasattr(args[0], 'min_rows'):
                min_rows_required = args[0].min_rows
            
            logger.debug(f"Running pattern validation with min_rows={min_rows_required}")
            validation_result = perform_dataframe_validation_logic(
                df, 
                required_cols=['open', 'high', 'low', 'close'],
                check_ohlcv=True, # This corresponds to the old validate_ohlc=True
                detect_anomalies_level='basic' if True else None, # Corresponds to check_statistical_anomalies
                min_rows=min_rows_required
            )
            
            if not validation_result.is_valid:
                error_message = "Validation failed"
                if validation_result.errors:
                    error_message = "; ".join(validation_result.errors)
                logger.error(f"Pattern validation failed: {error_message}")
                raise DataValidationError(
                    f"DataFrame validation failed: {error_message}",
                    error_code="VALIDATION_FAILED"
                )
            
            # Warnings and detailed statistics are logged within perform_dataframe_validation_logic.
            # No need to access them directly here from the result object.
            
            logger.debug("Centralized validation passed for pattern detection")
            
        except Exception as e:
            if isinstance(e, DataValidationError):
                raise  # Re-raise our custom validation errors
            else:
                logger.error(f"Validation error: {e}")
                raise DataValidationError(
                    f"Validation error: {str(e)}",
                    error_code="VALIDATION_ERROR"
                )
        
        return func(*args, **kwargs)
    
    return wrapper

def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor pattern detection performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            if execution_time > 0.1:  # Log slow operations
                logger.warning(f"{func.__name__} took {execution_time:.3f}s to execute")
            return result
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper

def validate_pattern_data(df: pd.DataFrame, enable_statistical_analysis: bool = True) -> Dict[str, Any]:
    """
    Convenience function for direct centralized validation of pattern data.
    
    This function provides easy access to the centralized validation system
    for pattern detection modules that need more control over validation behavior.
    
    Args:
        df: DataFrame to validate for pattern detection
        enable_statistical_analysis: Whether to perform statistical anomaly detection
        
    Returns:
        Dict containing validation results and enhanced metadata
        
    Raises:
        DataValidationError: If validation fails
    """
    try:
        logger.debug("Running centralized pattern data validation")
        validation_result = perform_dataframe_validation_logic(
            df,
            required_cols=['open', 'high', 'low', 'close'],
            check_ohlcv=True, # Corresponds to the old validate_ohlc=True
            detect_anomalies_level='basic' if enable_statistical_analysis else None # Corresponds to check_statistical_anomalies
        )
        
        if not validation_result.is_valid:
            error_message = "Pattern data validation failed"
            if validation_result.errors:
                error_message = "; ".join(validation_result.errors)
            logger.error(f"Pattern data validation failed: {error_message}")
            raise DataValidationError(
                f"Pattern data validation failed: {error_message}",
                error_code="PATTERN_VALIDATION_FAILED"
            )
        
        # Extract metadata that IS available or can be inferred
        validation_metadata = {
            'is_valid': validation_result.is_valid,
            'row_count': len(df) if validation_result.validated_data is not None else 0, # Get from df
            'column_count': len(df.columns) if validation_result.validated_data is not None and hasattr(df, 'columns') else 0, # Get from df
            # 'warnings': [], # No longer directly available from result
            # 'statistics': {} # No longer directly available from result
        }
        
        # Warnings and statistics are logged within perform_dataframe_validation_logic.
        
        logger.debug("Centralized pattern data validation completed successfully")
        return validation_metadata
        
    except Exception as e:
        if isinstance(e, DataValidationError):
            raise  # Re-raise validation errors
        else:
            logger.error(f"Pattern validation error: {e}")
            raise DataValidationError(
                f"Pattern validation error: {str(e)}",
                error_code="PATTERN_VALIDATION_ERROR"
            )

class PatternDetector(ABC):
    """Abstract base class for pattern detectors."""
    
    @abstractmethod
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect pattern in the given DataFrame."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Pattern name."""
        pass
    
    @property
    @abstractmethod
    def min_rows(self) -> int:
        """Minimum rows required for detection."""
        pass

class CandlestickPatterns:
    """
    Production-grade candlestick pattern detection engine.
    
    Features:
    - Thread-safe pattern registration and detection
    - Comprehensive error handling and validation
    - Performance optimization with caching
    - Configurable confidence thresholds
    - Detailed pattern information and metadata
    """
    
    def __init__(self, confidence_threshold: float = 0.7, enable_caching: bool = True):
        """
        Initialize pattern detection engine.
        
        Args:
            confidence_threshold: Minimum confidence for pattern detection (0.0-1.0)
            enable_caching: Whether to enable result caching for performance
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise PatternConfigurationError(
                f"Confidence threshold must be between 0.0 and 1.0, got {confidence_threshold}"
            )
        
        self._patterns: Dict[str, PatternDetector] = {}
        self._confidence_threshold = confidence_threshold
        self._enable_caching = enable_caching
        self._cache: Dict[str, List[PatternResult]] = {}
        self._lock = threading.RLock()
        
        # Register built-in patterns
        self._register_builtin_patterns()
    
    def _register_builtin_patterns(self):
        """Register all built-in candlestick patterns."""
        patterns = [
            HammerPattern(),
            BullishEngulfingPattern(),
            MorningStarPattern(),
            PiercingPatternDetector(),
            BullishHaramiPattern(),
            ThreeWhiteSoldiersPattern(),
            InvertedHammerPattern(),
            DojiPattern(),
            MorningDojiStarPattern(),
            BullishAbandonedBabyPattern(),
            BullishBeltHoldPattern(),
            ThreeInsideUpPattern(),
            RisingWindowPattern(),
            BearishEngulfingPattern(),
            EveningStarPattern(),
            ThreeBlackCrowsPattern(),
            BearishHaramiPattern(),
        ]
        
        for pattern in patterns:
            self.register_pattern(pattern)
    
    def register_pattern(self, pattern: PatternDetector):
        """
        Register a new pattern detector.
        
        Args:
            pattern: PatternDetector instance
            
        Raises:
            PatternConfigurationError: If pattern is invalid or already registered
        """
        if not isinstance(pattern, PatternDetector):
            raise PatternConfigurationError(
                f"Pattern must be an instance of PatternDetector, got {type(pattern).__name__}"
            )
        
        with self._lock:
            if pattern.name in self._patterns:
                logger.warning(f"Overwriting existing pattern: {pattern.name}")
            
            self._patterns[pattern.name] = pattern
            logger.info(f"Registered pattern: {pattern.name}")
    
    def get_pattern_names(self) -> List[str]:
        """Get list of all registered pattern names."""
        with self._lock:
            return list(self._patterns.keys())
    
    def get_pattern_info(self, pattern_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific pattern."""
        with self._lock:
            if pattern_name not in self._patterns:
                raise PatternDetectionError(
                    f"Pattern '{pattern_name}' not found",
                    pattern_name=pattern_name,
                    error_code="PATTERN_NOT_FOUND"
                )
            
            pattern = self._patterns[pattern_name]
            return {
                "name": pattern.name,
                "min_rows": pattern.min_rows,
                "description": getattr(pattern, 'description', 'No description available'),
                "pattern_type": getattr(pattern, 'pattern_type', PatternType.BULLISH_REVERSAL),
                "strength": getattr(pattern, 'strength', PatternStrength.MODERATE)
            }
    
    @validate_dataframe
    @performance_monitor
    def detect_patterns(self, 
                       df: pd.DataFrame, 
                       pattern_names: Optional[List[str]] = None,
                       parallel: bool = False) -> List[PatternResult]:
        """
        Detect patterns in the given DataFrame.
        
        Args:
            df: DataFrame with OHLC data
            pattern_names: Specific patterns to detect (None for all)
            parallel: Whether to use parallel processing
            
        Returns:
            List of PatternResult objects for detected patterns
        """
        # Create cache key
        cache_key = None
        if self._enable_caching:
            cache_key = self._create_cache_key(df, pattern_names)
            with self._lock:
                if cache_key in self._cache:
                    logger.debug("Returning cached results")
                    return self._cache[cache_key]
        
        # Determine patterns to check
        patterns_to_check = pattern_names or self.get_pattern_names()
        
        # Filter patterns by minimum rows requirement
        available_patterns = []
        with self._lock:
            for name in patterns_to_check:
                if name not in self._patterns:
                    logger.warning(f"Pattern '{name}' not found, skipping")
                    continue
                
                pattern = self._patterns[name]
                if len(df) >= pattern.min_rows:
                    available_patterns.append((name, pattern))
                else:
                    logger.debug(f"Insufficient data for pattern '{name}' "
                               f"(need {pattern.min_rows}, have {len(df)})")
        
        # Detect patterns
        results = []
        
        if parallel and len(available_patterns) > 1:
            results = self._detect_patterns_parallel(df, available_patterns)
        else:
            results = self._detect_patterns_sequential(df, available_patterns)
        
        # Filter by confidence threshold
        filtered_results = [
            result for result in results 
            if result.detected and result.confidence >= self._confidence_threshold
        ]
        
        # Cache results
        if self._enable_caching and cache_key:
            with self._lock:
                self._cache[cache_key] = filtered_results
                # Limit cache size
                if len(self._cache) > 100:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
        
        return filtered_results
    
    def _detect_patterns_sequential(self, df: pd.DataFrame, patterns: List[Tuple[str, PatternDetector]]) -> List[PatternResult]:
        """Detect patterns sequentially."""
        results = []
        
        for name, pattern in patterns:
            try:
                result = pattern.detect(df)
                results.append(result)
            except Exception as e:
                logger.warning(f"Pattern '{name}' detection failed: {e}")
                # Create failed result
                results.append(PatternResult(
                    name=name,
                    detected=False,
                    confidence=0.0,
                    pattern_type=getattr(pattern, 'pattern_type', PatternType.BULLISH_REVERSAL),
                    strength=PatternStrength.WEAK,
                    description=f"Detection failed: {str(e)}",
                    min_rows_required=pattern.min_rows
                ))
        
        return results
    
    def _detect_patterns_parallel(self, df: pd.DataFrame, patterns: List[Tuple[str, PatternDetector]]) -> List[PatternResult]:
        """Detect patterns in parallel using ThreadPoolExecutor."""
        results = []
        
        def detect_single_pattern(name_pattern_tuple):
            name, pattern = name_pattern_tuple
            try:
                return pattern.detect(df)
            except Exception as e:
                logger.warning(f"Pattern '{name}' detection failed: {e}")
                return PatternResult(
                    name=name,
                    detected=False,
                    confidence=0.0,
                    pattern_type=getattr(pattern, 'pattern_type', PatternType.BULLISH_REVERSAL),
                    strength=PatternStrength.WEAK,
                    description=f"Detection failed: {str(e)}",
                    min_rows_required=pattern.min_rows
                )
        
        with ThreadPoolExecutor(max_workers=min(len(patterns), 4)) as executor:
            results = list(executor.map(detect_single_pattern, patterns))
        
        return results
    
    def _create_cache_key(self, df: pd.DataFrame, pattern_names: Optional[List[str]]) -> str:
        """Create a cache key for the given DataFrame and patterns."""
        # Use last few rows and pattern names to create key
        last_rows = df.tail(5)
        data_hash = hash(tuple(
            tuple(row) for row in last_rows[['open', 'high', 'low', 'close']].values
        ))
        patterns_str = ",".join(sorted(pattern_names or self.get_pattern_names()))
        return f"{data_hash}_{hash(patterns_str)}"
    
    def clear_cache(self):
        """Clear the pattern detection cache."""
        with self._lock:
            self._cache.clear()
            logger.info("Pattern detection cache cleared")

# Pattern Implementations
class HammerPattern(PatternDetector):
    """Hammer pattern detector with enhanced validation."""
    
    @property
    def name(self) -> str:
        return "Hammer"
    
    @property
    def min_rows(self) -> int:
        return 1
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BULLISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.MODERATE
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """
        Detect Hammer pattern with confidence scoring.
        
        The Hammer is a bullish reversal pattern with:
        - Small body at the top
        - Long lower shadow (>= 2x body size)
        - Little to no upper shadow
        """
        row = df.iloc[-1]
        
        body = abs(row.close - row.open)
        lower_wick = min(row.open, row.close) - row.low
        upper_wick = row.high - max(row.open, row.close)
        
        if body == 0:  # Avoid division by zero
            confidence = 0.0
            detected = False
        else:
            # Calculate confidence based on pattern quality
            lower_wick_ratio = lower_wick / body if body > 0 else 0
            upper_wick_ratio = upper_wick / body if body > 0 else 0
            
            # Core pattern requirements
            has_long_lower_wick = lower_wick_ratio >= 2.0
            has_small_upper_wick = upper_wick_ratio <= 1.0
            
            detected = has_long_lower_wick and has_small_upper_wick
            
            if detected:
                # Calculate confidence (0.5 to 1.0 for valid patterns)
                wick_quality = min(lower_wick_ratio / 3.0, 1.0)  # Normalize to max 1.0
                shadow_quality = max(0.0, 1.0 - upper_wick_ratio)
                confidence = 0.5 + 0.5 * (wick_quality * 0.7 + shadow_quality * 0.3)
            else:
                confidence = 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Bullish reversal pattern with long lower shadow and small body at top",
            min_rows_required=self.min_rows
        )

class BullishEngulfingPattern(PatternDetector):
    """Bullish Engulfing pattern detector."""
    
    @property
    def name(self) -> str:
        return "Bullish Engulfing"
    
    @property
    def min_rows(self) -> int:
        return 2
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BULLISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.STRONG
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Bullish Engulfing pattern with confidence scoring."""
        prev, last = df.iloc[-2], df.iloc[-1]
        
        # Pattern requirements
        prev_bearish = prev.close < prev.open
        last_bullish = last.close > last.open
        last_opens_below = last.open < prev.close
        last_closes_above = last.close > prev.open
        
        detected = all([prev_bearish, last_bullish, last_opens_below, last_closes_above])
        
        if detected:
            # Calculate confidence based on engulfing completeness
            prev_body = abs(prev.open - prev.close)
            last_body = abs(last.close - last.open)
            
            engulfing_ratio = last_body / prev_body if prev_body > 0 else 0
            gap_size = abs(last.open - prev.close) / prev_body if prev_body > 0 else 0
            
            confidence = min(0.6 + 0.3 * min(engulfing_ratio, 2.0) / 2.0 + 0.1 * min(gap_size, 1.0), 1.0)
        else:
            confidence = 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Two-candle bullish reversal where larger bullish candle engulfs previous bearish candle",
            min_rows_required=self.min_rows
        )


class MorningStarPattern(PatternDetector):
    """Morning Star pattern detector."""
    
    @property
    def name(self) -> str:
        return "Morning Star"
    
    @property
    def min_rows(self) -> int:
        return 3
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BULLISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.VERY_STRONG
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        """Detect Morning Star pattern."""
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        # Pattern requirements
        first_bearish = first.close < first.open
        small_star = abs(second.close - second.open) < abs(first.open - first.close) * 0.3
        third_bullish = third.close > third.open
        third_penetrates = third.close > (first.open + first.close) / 2
        
        detected = all([first_bearish, small_star, third_bullish, third_penetrates])
        
        if detected:
            first_body = abs(first.open - first.close)
            star_body = abs(second.close - second.open)
            third_body = abs(third.close - third.open)
            
            # Quality metrics
            star_quality = 1.0 - min(star_body / first_body, 1.0)
            penetration_depth = (third.close - (first.open + first.close) / 2) / first_body
            size_balance = min(third_body / first_body, 1.0)
            
            confidence = 0.6 + 0.4 * (star_quality * 0.4 + min(penetration_depth, 1.0) * 0.4 + size_balance * 0.2)
        else:
            confidence = 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Three-candle bullish reversal: bearish, small star, strong bullish",
            min_rows_required=self.min_rows
        )

class PiercingPatternDetector(PatternDetector):
    @property
    def name(self) -> str:
        return "Piercing Pattern"
    
    @property
    def min_rows(self) -> int:
        return 2
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BULLISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.MODERATE
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        prev, last = df.iloc[-2], df.iloc[-1]
        midpoint = (prev.open + prev.close) / 2
        
        detected = (
            prev.close < prev.open and
            last.open < prev.close and
            last.close > midpoint and
            last.close > last.open
        )
        
        confidence = 0.7 if detected else 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Two-candle bullish reversal where second candle pierces into first",
            min_rows_required=self.min_rows
        )

class BullishHaramiPattern(PatternDetector):
    @property
    def name(self) -> str:
        return "Bullish Harami"
    
    @property
    def min_rows(self) -> int:
        return 2
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BULLISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.MODERATE
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        prev, last = df.iloc[-2], df.iloc[-1]
        
        detected = (
            prev.close < prev.open and
            last.open > prev.close and
            last.close < prev.open and
            last.close > last.open
        )
        
        confidence = 0.65 if detected else 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Two-candle pattern where small bullish candle is contained within previous bearish candle",
            min_rows_required=self.min_rows
        )

# Add all remaining pattern implementations...
# (Continuing with the same structure for all other patterns)

class ThreeWhiteSoldiersPattern(PatternDetector):
    @property
    def name(self) -> str:
        return "Three White Soldiers"
    
    @property
    def min_rows(self) -> int:
        return 3
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BULLISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.STRONG
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        detected = (
            first.close > first.open and
            second.close > second.open and
            third.close > third.open and
            second.open > first.open and
            third.open > second.open and
            second.close > first.close and
            third.close > second.close
        )
        
        confidence = 0.8 if detected else 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Three consecutive bullish candles with progressive higher opens and closes",
            min_rows_required=self.min_rows
        )

# Add remaining pattern implementations following similar structure...
# (I'll continue with the remaining patterns in the same format)

class InvertedHammerPattern(PatternDetector):
    @property
    def name(self) -> str:
        return "Inverted Hammer"
    
    @property
    def min_rows(self) -> int:
        return 1
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BULLISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.MODERATE
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        row = df.iloc[-1]
        body = abs(row.close - row.open)
        upper_wick = row.high - max(row.open, row.close)
        lower_wick = min(row.open, row.close) - row.low
        
        detected = body > 0 and (upper_wick > 2 * body) and (lower_wick < body)
        confidence = 0.6 if detected else 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Bullish reversal pattern with long upper shadow and small body at bottom",
            min_rows_required=self.min_rows
        )

# Continue with remaining patterns...
class DojiPattern(PatternDetector):
    @property
    def name(self) -> str:
        return "Doji"
    
    @property
    def min_rows(self) -> int:
        return 1
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.INDECISION
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.MODERATE
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        row = df.iloc[-1]
        detected = abs(row.open - row.close) <= (row.high - row.low) * 0.1
        confidence = 0.75 if detected else 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Indecision pattern where open and close prices are nearly equal",
            min_rows_required=self.min_rows
        )


class MorningDojiStarPattern(PatternDetector):
    @property
    def name(self) -> str:
        return "Morning Doji Star"
    
    @property
    def min_rows(self) -> int:
        return 3
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BULLISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.VERY_STRONG
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        second_open_close_diff = abs(second.open - second.close)
        
        detected = (
            first.close < first.open and
            second_open_close_diff <= (second.high - second.low) * 0.1 and
            third.close > third.open and
            third.close > (first.open + first.close) / 2
        )
        
        confidence = 0.85 if detected else 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Morning Star pattern with Doji as the middle candle",
            min_rows_required=self.min_rows
        )


class BullishAbandonedBabyPattern(PatternDetector):
    @property
    def name(self) -> str:
        return "Bullish Abandoned Baby"
    
    @property
    def min_rows(self) -> int:
        return 3
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BULLISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.VERY_STRONG
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        detected = (
            first.close < first.open and
            abs(second.open - second.close) < (second.high - second.low) * 0.1 and
            second.low > first.high and
            third.open > second.high and third.close > third.open
        )
        
        confidence = 0.9 if detected else 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Rare three-candle reversal with isolated Doji in the middle",
            min_rows_required=self.min_rows
        )

class BullishBeltHoldPattern(PatternDetector):
    @property
    def name(self) -> str:
        return "Bullish Belt Hold"
    
    @property
    def min_rows(self) -> int:
        return 1
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BULLISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.MODERATE
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        row = df.iloc[-1]
        
        detected = (
            row.close > row.open and
            row.open == row.low and
            (row.close - row.open) > (row.high - row.close) * 0.5
        )
        
        confidence = 0.7 if detected else 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Bullish candle opening at the low with strong upward momentum",
            min_rows_required=self.min_rows
        )

class ThreeInsideUpPattern(PatternDetector):
    @property
    def name(self) -> str:
        return "Three Inside Up"
    
    @property
    def min_rows(self) -> int:
        return 3
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BULLISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.STRONG
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        detected = (
            first.close < first.open and
            second.open > first.close and
            second.close < first.open and
            second.close > second.open and
            third.close > second.close
        )
        
        confidence = 0.75 if detected else 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Three-candle pattern: Bullish Harami followed by confirmation candle",
            min_rows_required=self.min_rows
        )

class RisingWindowPattern(PatternDetector):
    @property
    def name(self) -> str:
        return "Rising Window"
    
    @property
    def min_rows(self) -> int:
        return 2
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.CONTINUATION
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.MODERATE
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        prev, last = df.iloc[-2], df.iloc[-1]
        detected = prev.high < last.low
        confidence = 0.65 if detected else 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Gap up pattern indicating bullish continuation",
            min_rows_required=self.min_rows
        )

# Bearish patterns...
class BearishEngulfingPattern(PatternDetector):
    @property
    def name(self) -> str:
        return "Bearish Engulfing"
    
    @property
    def min_rows(self) -> int:
        return 2
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BEARISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.STRONG
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        prev, last = df.iloc[-2], df.iloc[-1]
        
        detected = (
            prev.close > prev.open and
            last.close < last.open and
            last.open > prev.close and
            last.close < prev.open
        )
        
        confidence = 0.8 if detected else 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Two-candle bearish reversal where larger bearish candle engulfs previous bullish candle",
            min_rows_required=self.min_rows
        )

class EveningStarPattern(PatternDetector):
    @property
    def name(self) -> str:
        return "Evening Star"
    
    @property
    def min_rows(self) -> int:
        return 3
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BEARISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.VERY_STRONG
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        detected = (
            first.close > first.open and
            abs(second.close - second.open) < abs(first.close - first.open) * 0.3 and
            third.close < third.open and
            third.close < (first.open + first.close) / 2
        )
        
        confidence = 0.85 if detected else 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Three-candle bearish reversal: bullish, small star, strong bearish",
            min_rows_required=self.min_rows
        )

class ThreeBlackCrowsPattern(PatternDetector):
    @property
    def name(self) -> str:
        return "Three Black Crows"
    
    @property
    def min_rows(self) -> int:
        return 3
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BEARISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.STRONG
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        first, second, third = df.iloc[-3], df.iloc[-2], df.iloc[-1]
        
        detected = (
            first.close < first.open and
            second.close < second.open and
            third.close < third.open and
            second.open < first.open and
            third.open < second.open and
            second.close < first.close and
            third.close < second.close
        )
        
        confidence = 0.8 if detected else 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Three consecutive bearish candles with progressive lower opens and closes",
            min_rows_required=self.min_rows
        )

class BearishHaramiPattern(PatternDetector):
    @property
    def name(self) -> str:
        return "Bearish Harami"
    
    @property
    def min_rows(self) -> int:
        return 2
    
    @property
    def pattern_type(self) -> PatternType:
        return PatternType.BEARISH_REVERSAL
    
    @property
    def strength(self) -> PatternStrength:
        return PatternStrength.MODERATE
    
    def detect(self, df: pd.DataFrame) -> PatternResult:
        prev, last = df.iloc[-2], df.iloc[-1]
        
        detected = (
            prev.close > prev.open and
            last.open < prev.close and
            last.close > prev.open and
            last.close < last.open
        )
        
        confidence = 0.65 if detected else 0.0
        
        return PatternResult(
            name=self.name,
            detected=detected,
            confidence=confidence,
            pattern_type=self.pattern_type,
            strength=self.strength,
            description="Two-candle pattern where small bearish candle is contained within previous bullish candle",
            min_rows_required=self.min_rows
        )

# Factory function for easy initialization
def create_pattern_detector(confidence_threshold: float = 0.7, enable_caching: bool = True) -> CandlestickPatterns:
    """
    Factory function to create a configured CandlestickPatterns instance.
    
    Args:
        confidence_threshold: Minimum confidence for pattern detection (0.0-1.0)
        enable_caching: Whether to enable result caching
        
    Returns:
        Configured CandlestickPatterns instance
    """
    return CandlestickPatterns(confidence_threshold=confidence_threshold, enable_caching=enable_caching)

def get_pattern_names() -> List[str]:
    """
    Backward compatibility function to get pattern names.
    
    Returns:
        List of available pattern names
    """
    warnings.warn(
        "get_pattern_names is deprecated. Use CandlestickPatterns.get_pattern_names() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    detector = create_pattern_detector()
    return detector.get_pattern_names()

# --- End of patterns.py ---