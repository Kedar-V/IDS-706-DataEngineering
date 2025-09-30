
import sys
sys.path.append('../assistant')
import pandas as pd
import streamlit as st
from assistant import NL2SQLAssistant
from controller import BitcoinOHLCController
from model import BitcoinOHLCModel


st.title("Personal AI Trading Assistant")

class BitcoinOHLCViewer:
    def __init__(self):
        self.model = BitcoinOHLCModel()
        self.controller = BitcoinOHLCController(self.model)

    def show_ohlc_table(self):
        import datetime
        df, error = self.controller.get_ohlc_df()
        if error:
            st.error(f"Error: {error}")
            return
        if df.empty:
            st.warning("No data found in the database.")
            return

        # Date selection widgets
        min_date = df['timestamp'].min().date() if not df.empty else datetime.date.today()
        max_date = df['timestamp'].max().date() if not df.empty else datetime.date.today()
        start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
        end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

        # Filter dataframe by selected date range
        mask = (df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)
        filtered_df = df.loc[mask]
        st.dataframe(filtered_df)

    def show_assistant(self):
        st.subheader("Enter your query in natural language")

        # Example prompts dropdown
        examples = [
            "Show me all rows where close is above 50000",
            "Get the top 5 days with highest volume",
            "Find all rows where open is below 30000",
            "Calculate the average close price per month",
            "Show all rows for September 2025"
        ]
        
        example_prompt = st.selectbox("Or pick an example query:", examples)
        
        # Text area for user input
        prompt = st.text_area("Or write your own query:", "")

        # Use example if selected
        if example_prompt != "--Select--":
            prompt = example_prompt

        sql = None
        results = None
        if st.button("Fetch Data") and prompt:
            try:
                assistant = NL2SQLAssistant()
                sql = assistant.convert(prompt)
                st.code(sql, language="sql")
                
                # Execute SQL safely
                cursor = self.model.dao.conn.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                results = pd.DataFrame(rows, columns=columns)
                cursor.close()
            except Exception as e:
                st.error(f"Error: {e}")

        if results is not None:
            st.dataframe(results)

    def show_menu(self):
        menu = st.sidebar.radio("Menu", ["Data View", "Ask Assistant"], index=0)
        if menu == "Data View":
            self.show_ohlc_table()
        elif menu == "Ask Assistant":
            self.show_assistant()

    def close(self):
        self.model.close()

viewer = BitcoinOHLCViewer()
viewer.show_menu()
viewer.close()
