from openai import OpenAI
from security.authentication import get_openai_api_key
import logging

logger = logging.getLogger(__name__)

def get_chatgpt_insight(summary: str) -> str:
    """
    Analyze a technical summary using GPT-4 and return a next-3-day trading recommendation.

    Sends a summary of technical indicators to GPT-4, asking for a Buy / Hold / Sell rating
    along with a concise rationale, key price triggers (with volume confirmation), a suggested
    stop-loss, and an approximate risk/reward commentary.

    Parameters:
        summary (str): The technical analysis summary of the stock

    Returns:
        str: GPT-generated financial analysis or error message
    """
    try:
        logger.info("Starting ChatGPT API request")
        
        # Get API key and create client
        api_key = get_openai_api_key()
        if not api_key:
            error_msg = "Error: OpenAI API key not found. Please set OPENAI_API_KEY environment variable."
            logger.error(error_msg)
            return error_msg        
        logger.info("API key found, creating OpenAI client")
        client = OpenAI(
            api_key=api_key,
            timeout=30.0  # 30 second timeout
        )
        
        logger.info("Sending request to OpenAI API")
        
        # Try gpt-4.1 first, fallback to gpt-4o if not available
        models_to_try = ["gpt-4.1", "gpt-4o", "gpt-4-turbo", "gpt-4"]
        
        for model_name in models_to_try:
            try:
                logger.info(f"Attempting to use model: {model_name}")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional stock trader. You interpret RSI, MACD, "
                                "Bollinger Bands, ATR, and candlestick patterns—and you always "
                                "include volume confirmation for breakouts—to provide a 3-day "
                                "actionable recommendation."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                "Here is a technical summary of a stock.  \n"
                                "1. List the **key signals** (momentum, volatility, patterns) as bullet points.  \n"
                                "2. Provide a **one-sentence rationale** referencing those signals.  \n"
                                "3. Give a **Final Recommendation:** Buy, Hold, or Sell.  \n"
                                "4. Suggest an **entry trigger** (include on above-average volume) and a **stop-loss** level.  \n"
                                "5. Comment on the approximate **risk/reward ratio**.  \n\n"
                                f"{summary}"
                            )
                        }
                    ],
                    temperature=0.4,
                    max_tokens=1000,  # Limit response length
                    timeout=30  # 30 second timeout
                )
                
                result = (response.choices[0].message.content or "").strip()
                logger.info(f"Successfully received ChatGPT response using {model_name}: {len(result)} characters")
                return result
                
            except Exception as model_error:
                logger.warning(f"Model {model_name} failed: {str(model_error)}")
                continue  # Try next model
        
        # If all models failed
        error_msg = f"Error: All ChatGPT models failed. Last error was with models: {models_to_try}"
        logger.error(error_msg)
        return error_msg
        
    except Exception as e:
        error_msg = f"Error fetching GPT insight: {str(e)}"
        logger.error(error_msg)
        return error_msg