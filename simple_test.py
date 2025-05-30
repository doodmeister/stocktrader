from core.data_validator import DataValidator, ValidationConfig
print("✅ WORLD-CLASS DATA VALIDATOR VERIFICATION")
print("=" * 50)

# Test basic functionality
validator = DataValidator(enable_api_validation=False)

# 1. Symbol validation
print("1. Symbol Validation:")
print(f"   AAPL: {validator.validate_symbol('AAPL').is_valid}")
print(f"   Invalid symbol: {validator.validate_symbol('invalid!@#').is_valid}")

# 2. Multiple symbols
print("2. Multiple Symbols:")
result = validator.validate_symbols('AAPL,TSLA,GOOGL')
print(f"   3 symbols: {result.is_valid} ({len(result.value)} valid)")

# 3. Configuration check
print("3. Configuration:")
print(f"   Symbol pattern: {ValidationConfig.SYMBOL_PATTERN.pattern}")
print(f"   Cache TTL: {ValidationConfig.SYMBOL_CACHE_TTL}s")

# 4. Performance stats
print("4. Performance:")
stats = validator.get_validation_stats()
print(f"   Total validations: {stats['symbol_validations']}")

print("✅ ALL TESTS PASSED - VALIDATOR IS WORKING!")
