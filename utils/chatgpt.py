import openai

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
        response = openai.chat.completions.create(
            model="gpt-4.1",
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
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error fetching GPT insight: {str(e)}"