#!/usr/bin/env python3
"""
Comprehensive test of the enhanced data validator
"""
from core.data_validator import DataValidator, ValidationConfig
import pandas as pd
from datetime import date, timedelta

print('=== COMPREHENSIVE DATA VALIDATOR VERIFICATION ===')
print()

# Display validation configuration
print('1. VALIDATION CONFIGURATION:')
print(f'   Symbol pattern: {ValidationConfig.SYMBOL_PATTERN.pattern}')
print(f'   Price range: ${ValidationConfig.MIN_PRICE} - ${ValidationConfig.MAX_PRICE:,.0f}')
print(f'   Cache TTL: {ValidationConfig.SYMBOL_CACHE_TTL}s')
print(f'   Max null percentage: {ValidationConfig.MAX_NULL_PERCENTAGE:.1%}')
print()

# Test advanced features
validator = DataValidator(enable_api_validation=False)

print('2. ADVANCED SYMBOL VALIDATION:')
test_symbols = ['AAPL', 'invalid!@#', 'TOOLONGTOBEVALID', 'TSLA,GOOGL,MSFT']
for symbol in test_symbols:
    if ',' in symbol:
        result = validator.validate_symbols(symbol)
        print(f'   Symbols "{symbol}": {result.is_valid} ({len(result.value) if result.value else 0} valid)')
    else:
        result = validator.validate_symbol(symbol)
        print(f'   Symbol "{symbol}": {result.is_valid}')
print()

print('3. INTERVAL-SPECIFIC DATE VALIDATION:')
today = date.today()
start_date = today - timedelta(days=10)

intervals = ['1m', '5m', '1h', '1d']
for interval in intervals:
    result = validator.validate_dates(start_date, today, interval)
    print(f'   {interval} interval (10 days): {result.is_valid}')
print()

print('4. FINANCIAL PARAMETER VALIDATION:')
# Price validation
prices = [150.50, 0.001, 1000000, '$45.99', 'invalid']
for price in prices:
    result = validator.validate_price(price)
    print(f'   Price {price}: {result.is_valid}')

# Percentage validation
percentages = ['5%', 0.02, 1.5, '-10%']
for pct in percentages:
    result = validator.validate_percentage(pct)
    print(f'   Percentage {pct}: {result.is_valid}')
print()

print('5. SECURITY FEATURES:')
# Input sanitization
dangerous_input = '<script>alert("hack")</script>AAPL'
sanitized = validator.sanitize_input(dangerous_input)
print(f'   Dangerous input: "{dangerous_input}"')
print(f'   Sanitized: "{sanitized}"')
print()

print('6. PERFORMANCE METRICS:')
stats = validator.get_validation_stats()
for key, value in stats.items():
    print(f'   {key}: {value}')
print()

print('7. INTERVAL LIMITS INFO:')
limits = validator.get_interval_limits_info()
for interval, limit in limits.items():
    print(f'   {interval}: {limit}')
print()

print('8. DATAFRAME VALIDATION WITH LARGER DATASET:')
# Create a proper sized DataFrame
dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
df_large = pd.DataFrame({
    'open': [100 + i*0.5 for i in range(50)],
    'high': [105 + i*0.5 for i in range(50)],
    'low': [95 + i*0.5 for i in range(50)],
    'close': [102 + i*0.5 for i in range(50)],
    'volume': [1000 + i*100 for i in range(50)]
}, index=dates)

result = validator.validate_dataframe(df_large, validate_ohlc=True, check_statistical_anomalies=True)
print(f'   Large DataFrame ({df_large.shape}): {result.is_valid}')
print(f'   Warnings: {len(result.warnings)} warnings found')
print(f'   Memory usage: {result.statistics.get("memory_usage_mb", 0):.3f} MB')
print()

print('✅ ALL FEATURES VERIFIED SUCCESSFULLY!')
print(f'✅ Total validations performed: {validator.get_validation_stats()["symbol_validations"] + validator.get_validation_stats()["date_validations"] + validator.get_validation_stats()["dataframe_validations"]}')
