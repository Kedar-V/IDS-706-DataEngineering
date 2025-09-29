import openai
import os

class NL2SQLAssistant:
    def __init__(self, api_key=None, table="bitcoin_daily_price"):
        self.api_key = api_key or os.getenv("MY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not set. Set MY_API_KEY environment variable or pass as argument."
            )
        openai.api_key = self.api_key
        self.table = table

        # Full schema for reference
        self.schema = f"""
        CREATE TABLE IF NOT EXISTS {self.table} (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            open DECIMAL(18,8) NOT NULL,
            high DECIMAL(18,8) NOT NULL,
            low DECIMAL(18,8) NOT NULL,
            close DECIMAL(18,8) NOT NULL,
            volume DECIMAL(18,8) NOT NULL,
            UNIQUE KEY unique_timestamp (timestamp)
        );
        """

    def convert(self, prompt):
        """
        Convert natural language prompt to SQL for MySQL OHLC bitcoin price table.
        Uses full schema for better accuracy.
        """
        system_message = f"""
        You are a helpful assistant that converts natural language into SQL queries
        for the following MySQL table schema:

        {self.schema}

        Only return valid SQL queries for MySQL using this table.
        """

        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        sql = response.choices[0].message.content.strip()
        return sql


if __name__ == "__main__":
    assistant = NL2SQLAssistant()
    sql = assistant.convert("Show me all rows where close is above 50000")
    print(sql)
