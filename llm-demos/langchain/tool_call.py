"""
Advanced LangChain Tool Calling System with Database and JIRA Integration.

This module demonstrates enterprise-level tool calling with proper error handling,
logging, type safety, and configuration management.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urljoin

import requests
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tool_calling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Configuration for database connections."""
    path: str = "db/movies.sqlite"
    timeout: int = 30
    
    def __post_init__(self):
        """Validate database path exists."""
        if not Path(self.path).exists():
            raise FileNotFoundError(f"Database file not found: {self.path}")


@dataclass
class JiraConfig:
    """Configuration for JIRA API integration."""
    base_url: str = "https://your-company.atlassian.net"
    api_endpoint: str = "/rest/api/2/issue/"
    email: str = "your_email"
    api_token: str = "your_api_token"
    project_key: str = "SUP"
    issue_type: str = "Task"
    timeout: int = 30
    max_retries: int = 3
    
    @property
    def api_url(self) -> str:
        """Get the full API URL."""
        return urljoin(self.base_url, self.api_endpoint)
    
    @property
    def auth(self) -> Tuple[str, str]:
        """Get authentication tuple."""
        return (self.email, self.api_token)


@dataclass
class LLMConfig:
    """Configuration for language model."""
    model_name: str = "gpt-4o-mini"
    provider: str = "openai"
    temperature: float = 0.1
    max_tokens: Optional[int] = None


class DatabaseManager:
    """Advanced database manager with connection pooling and error handling."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.schema = """
        Table: movies
        Columns: id, original_title, budget, popularity, release_date, revenue, title, vote_average, overview, tagline, uid, director_id
        Table: directors
        Columns: name, id, gender, uid, department
        """
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = sqlite3.connect(
                self.config.path,
                timeout=self.config.timeout,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results as list of dictionaries."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            return [dict(row) for row in results]


class JiraManager:
    """Advanced JIRA API manager with retry logic and error handling."""
    
    def __init__(self, config: JiraConfig):
        self.config = config
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def create_ticket(self, summary: str, description: str) -> Dict[str, Any]:
        """Create a JIRA ticket and return the response."""
        payload = {
            "fields": {
                "project": {"key": self.config.project_key},
                "summary": summary,
                "description": description,
                "issuetype": {"name": self.config.issue_type}
            }
        }
        
        headers = {"Content-Type": "application/json"}
        
        try:
            response = self.session.post(
                self.config.api_url,
                auth=self.config.auth,
                json=payload,
                headers=headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Successfully created JIRA ticket: {result.get('key', 'Unknown')}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"JIRA API error: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JIRA response: {e}")
            raise


class AdvancedToolManager:
    """Advanced tool manager for LangChain integration."""
    
    def __init__(
        self,
        db_config: DatabaseConfig,
        jira_config: JiraConfig,
        llm_config: LLMConfig
    ):
        self.db_manager = DatabaseManager(db_config)
        self.jira_manager = JiraManager(jira_config)
        self.llm_config = llm_config
        self.llm = self._initialize_llm()
        self.tools = self._create_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
    
    def _initialize_llm(self) -> BaseChatModel:
        """Initialize the language model with configuration."""
        return init_chat_model(
            self.llm_config.model_name,
            model_provider=self.llm_config.provider,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens
        )
    
    def _create_tools(self) -> List:
        """Create and return the list of tools."""
        
        @tool
        def ask_database(sql_query: str) -> str:
            """
            Executes a SQL query against the SQLite movies database and returns results.
            
            Args:
                sql_query: The SQL query to execute
                
            Returns:
                JSON string of query results or error message
            """
            try:
                results = self.db_manager.execute_query(sql_query)
                logger.info(f"Database query executed successfully, returned {len(results)} rows")
                return json.dumps(results, indent=2, default=str)
            except Exception as e:
                error_msg = f"Database query failed: {str(e)}"
                logger.error(error_msg)
                return error_msg
        
        @tool
        def create_support_ticket(summary: str, description: str) -> str:
            """
            Creates a support ticket in JIRA.
            
            Args:
                summary: Brief summary of the ticket
                description: Detailed description of the issue
                
            Returns:
                JSON string with ticket information or error message
            """
            try:
                result = self.jira_manager.create_ticket(summary, description)
                
                # Format the response with ticket URL
                ticket_key = result.get('key', 'Unknown')
                ticket_url = f"{self.jira_manager.config.base_url}/browse/{ticket_key}"
                
                response = {
                    "success": True,
                    "ticket_key": ticket_key,
                    "ticket_url": ticket_url,
                    "message": f"Support ticket created successfully: {ticket_url}"
                }
                
                return json.dumps(response, indent=2)
                
            except Exception as e:
                error_msg = f"Failed to create support ticket: {str(e)}"
                logger.error(error_msg)
                return json.dumps({
                    "success": False,
                    "error": error_msg
                }, indent=2)
        
        return [ask_database, create_support_ticket]
    
    def _create_system_message(self) -> SystemMessage:
        """Create the system message with instructions."""
        system_content = f"""
You are an advanced AI assistant with access to specialized tools:

1. **ask_database**: Query the movies SQLite database
   Database Schema:
   {self.db_manager.schema}

2. **create_support_ticket**: Create JIRA support tickets

Guidelines:
- For movie-related queries, use ask_database tool with proper SQL
- For support requests, use create_support_ticket with clear summary and description
- Always provide helpful context and explanations
- Handle errors gracefully and inform the user
- Return JIRA URLs in format: {self.jira_manager.config.base_url}/browse/TICKET-KEY
"""
        return SystemMessage(system_content.strip())
    
    def process_query(self, query: str) -> str:
        """
        Process a user query using the tool-enabled LLM.
        
        Args:
            query: User's question or request
            
        Returns:
            Final response from the AI
        """
        logger.info(f"Processing query: {query}")
        
        messages: List[BaseMessage] = [
            self._create_system_message(),
            HumanMessage(query)
        ]
        
        try:
            # Get initial AI response
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)
            
            # Process tool calls if any
            if hasattr(ai_msg, 'tool_calls') and ai_msg.tool_calls:
                logger.info(f"Processing {len(ai_msg.tool_calls)} tool calls")
                
                for tool_call in ai_msg.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                    
                    # Find and execute the tool
                    selected_tool = None
                    for tool in self.tools:
                        if tool.name == tool_name:
                            selected_tool = tool
                            break
                    
                    if selected_tool:
                        tool_result = selected_tool.invoke(tool_args)
                        tool_msg = ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"]
                        )
                        messages.append(tool_msg)
                        logger.info(f"Tool {tool_name} executed successfully")
                    else:
                        logger.error(f"Tool {tool_name} not found")
                
                # Get final response after tool execution
                final_response = self.llm_with_tools.invoke(messages)
                return final_response.content
            
            else:
                # No tools needed, return direct response
                return ai_msg.content
                
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            logger.error(error_msg)
            return error_msg


def main():
    """Main execution function demonstrating the advanced tool system."""
    try:
        # Configuration
        db_config = DatabaseConfig()
        jira_config = JiraConfig()
        llm_config = LLMConfig()
        
        # Initialize tool manager
        tool_manager = AdvancedToolManager(db_config, jira_config, llm_config)
        
        # Example queries
        queries = [
            "Can you create a support ticket for me? I want to add movies released in 2024.",
            "What are the top 5 highest-grossing movies in the database?",
            "Show me directors who have made more than 2 movies."
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*60}")
            print(f"Query {i}: {query}")
            print('='*60)
            
            result = tool_manager.process_query(query)
            print(f"\nResult:\n{result}")
            
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == '__main__':
    main()