# import os
# import openai
# import requests
# import sqlite3
# from tenacity import retry, wait_random_exponential, stop_after_attempt
# from dotenv import load_dotenv
# from conversation import Conversation


# load_dotenv()
# openai.api_key = os.getenv("GOOGLE_API_KEY")
# MODEL = "gpt-4o"
# DATABASE = "db/movies.sqlite"
# USER_MESSAGE = "Hi, what are the directors of top 10 movies with the budget more than 1 mil by user rating"

# @retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
# def chat_completion_request(messages, functions=None, model=MODEL):
#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": "Bearer " + openai.api_key,
#     }
#     json_data = {"model": model, "messages": messages}
#     if functions is not None:
#         json_data.update({"functions": functions})
#     try:
#         response = requests.post(
#             "https://api.openai.com/v1/chat/completions",
#             headers=headers,
#             json=json_data,
#         )
#         return response
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return e


# if __name__ == '__main__':
#     conversation = Conversation()
#     conn = sqlite3.connect(DATABASE)


# database_schema_string = """
# Table: movies
# Columns: id, original_title, budget, popularity, release_date, revenue, title, vote_average, overview, tagline, uid, director_id
# Table: directors
# Columns: name, id, gender, uid, department
# """

# print(f"Database schema string: '{database_schema_string}'")

# functions = [
#     {
#         "name": "ask_database",
#         "description": "Use this function to answer user questions about data. Output should be a fully formed SQL query.",
#         "parameters": {
#             "type": "object",
#             "properties": {
#                 "query": {
#                     "type": "string",
#                     "description": f"""
#                             SQL query extracting info to answer the user's question.
#                             SQL should be written using this database schema:
#                             {database_schema_string}
#                             The query should be returned in plain text, not in JSON.
#                             """,
#                 }
#             },
#             "required": ["query"],
#         },
#     }
# ]


# def ask_database(conn, query):
#     """Function to query SQLite database with provided SQL query."""
#     try:
#         results = conn.execute(query).fetchall()
#         return results
#     except Exception as e:
#         raise Exception(f"SQL error: {e}")


# def chat_completion_with_function_execution(messages, functions=None):
#     """This function makes a ChatCompletion API call and if a function call is requested, executes the function"""
#     try:
#         response = chat_completion_request(messages, functions)
#         full_message = response.json()["choices"][0]
#         if full_message["finish_reason"] == "function_call":
#             print(f"Function generation requested, calling function")
#             return call_function(messages, full_message)
#         else:
#             print(f"Function not required, responding to user")
#             return response.json()
#     except Exception as e:
#         print("Unable to generate ChatCompletion response")
#         print(f"Exception: {e}")
#         return response


# def call_function(messages, full_message):
#     """Executes function calls using model generated function arguments."""

#     # We'll add our one function here - this can be extended with any additional functions
#     if full_message["message"]["function_call"]["name"] == "ask_database":
#         query = eval(full_message["message"]["function_call"]["arguments"])
#         print(f"Prepped query is {query}")
#         try:
#             results = ask_database(conn, query["query"])
#         except Exception as e:
#             print(e)

#             # This following block tries to fix any issues in query generation with a subsequent call
#             messages.append(
#                 {
#                     "role": "system",
#                     "content": f"""Query: {query['query']}
# The previous query received the error {e}. 
# Please return a fixed SQL query in plain text.
# Your response should consist of ONLY the SQL query with the separator sql_start at the beginning and sql_end at the end""",
#                 }
#             )
#             response = chat_completion_request(messages, model=MODEL)

#             # Retrying with the fixed SQL query. If it fails a second time we exit.
#             try:
#                 cleaned_query = response.json()["choices"][0]["message"][
#                     "content"
#                 ].split("sql_start")[1]
#                 cleaned_query = cleaned_query.split("sql_end")[0]
#                 #print(cleaned_query)
#                 results = ask_database(conn, cleaned_query)
#                 #print(results)
#                 print("Got on second try")

#             except Exception as e:
#                 print("Second failure, exiting")

#                 print(f"Function execution failed")
#                 print(f"Error message: {e}")

#         messages.append(
#             {"role": "function", "name": "ask_database", "content": str(results)}
#         )

#         try:
#             response = chat_completion_request(messages)
#             return response.json()
#         except Exception as e:
#             print(type(e))
#             print(e)
#             raise Exception("Function chat request failed")
#     else:
#         raise Exception("Function does not exist and cannot be called")


# agent_system_message = """You are DatabaseGPT, a helpful assistant who gets answers to user questions from the Database
# Provide as many details as possible to your users
# Begin!"""

# sql_conversation = Conversation()
# sql_conversation.add_message("system", agent_system_message)
# sql_conversation.add_message(
#     "user", USER_MESSAGE
# )

# chat_response = chat_completion_with_function_execution(
#     sql_conversation.conversation_history, functions=functions
# )
# try:
#     assistant_message = chat_response["choices"][0]["message"]["content"]
# except Exception as e:
#     print(e)
#     print(chat_response)

# sql_conversation.add_message("assistant", assistant_message)
# sql_conversation.display_conversation(detailed=True)







































import os
import sqlite3
import google.generativeai as genai
from dotenv import load_dotenv
from conversation import Conversation

# Load Gemini API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Use Gemini-Pro model
model = genai.GenerativeModel("gemini-pro")

# Constants
DATABASE = "db/movies.sqlite"
USER_MESSAGE = "Hi, what are the directors of top 10 movies with the budget more than 1 mil by user rating"

# Database schema string
database_schema_string = """
Table: movies
Columns: id, original_title, budget, popularity, release_date, revenue, title, vote_average, overview, tagline, uid, director_id
Table: directors
Columns: name, id, gender, uid, department
"""

# Connect to DB
conn = sqlite3.connect(DATABASE)

# Query function
def ask_database(conn, query):
    try:
        return conn.execute(query).fetchall()
    except Exception as e:
        raise Exception(f"SQL error: {e}")

# Simulate function calling
def get_sql_query_from_model(user_input, schema):
    prompt = f"""
You are SQLGenBot. Given the user question and the schema, generate an SQL query ONLY.
Schema:
{schema}

User: {user_input}
SQL:"""
    response = model.generate_content(prompt)
    return response.text.strip()

# Build conversation
conversation = Conversation()
conversation.add_message("system", "You are DatabaseGPT. Answer questions by querying the database.")
conversation.add_message("user", USER_MESSAGE)

# Step 1: Ask Gemini to generate SQL
sql_query = get_sql_query_from_model(USER_MESSAGE, database_schema_string)
print(f"Generated SQL Query:\n{sql_query}")

# Step 2: Execute SQL
try:
    results = ask_database(conn, sql_query)
except Exception as e:
    print(f"Initial query failed: {e}")
    # Optional: re-ask Gemini to fix it
    retry_prompt = f"""
The following query caused an error:

{sql_query}

Error: {e}
Please fix the SQL query based on this error and schema:
{database_schema_string}
Fixed SQL:"""
    fixed_sql = model.generate_content(retry_prompt).text.strip()
    print(f"Fixed SQL: {fixed_sql}")
    results = ask_database(conn, fixed_sql)

# Step 3: Final Gemini answer
final_prompt = f"""
Here is the result of the SQL query:
{results}

Please answer the user's original question in plain English:
"{USER_MESSAGE}"
"""
response = model.generate_content(final_prompt)
assistant_reply = response.text.strip()

# Store and display conversation
conversation.add_message("assistant", assistant_reply)
conversation.display_conversation(detailed=True)
