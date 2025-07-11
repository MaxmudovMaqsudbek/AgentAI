import asyncio
import datetime
import os
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gemini_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration settings for Excel processing."""
    input_file: str = 'data.xlsx'
    output_file: str = 'data_output.xlsx'
    model_name: str = 'gemini-pro'
    timeout_seconds: int = 30
    default_response: str = "❌ Request timed out"
    max_workers: int = 3
    temperature: float = 0.7
    max_output_tokens: int = 1024


class GeminiExcelProcessor:
    """Advanced Excel processor using Google Gemini AI."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize processor with configuration."""
        self.config = config
        self._setup_gemini()
        self.call_count = 0
        self.successful_calls = 0
        self.failed_calls = 0
        
    def _setup_gemini(self) -> None:
        """Configure Gemini AI with API key."""
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        
        self.generation_config = genai.types.GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        
        self.model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config=self.generation_config
        )
        
        logger.info(f"✅ Gemini AI configured with model: {self.config.model_name}")
    
    def _make_gemini_call(self, system_message: str, user_input: str) -> str:
        """
        Make a single call to Gemini AI.
        
        Args:
            system_message: System prompt/instructions
            user_input: User message/query
            
        Returns:
            AI response content
        """
        try:
            # Combine system and user messages for Gemini
            combined_prompt = f"""System Instructions: {system_message}

User Query: {user_input}

Please respond according to the system instructions above."""

            response = self.model.generate_content(combined_prompt)
            
            if response.text:
                return response.text.strip()
            else:
                return "❌ Empty response from Gemini AI"
                
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            raise
    
    def _api_call_with_timeout(self, system_message: str, user_input: str) -> str:
        """
        Execute Gemini API call with timeout protection.
        
        Args:
            system_message: System prompt
            user_input: User message
            
        Returns:
            AI response or timeout message
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                future = executor.submit(self._make_gemini_call, system_message, user_input)
                result = future.result(timeout=self.config.timeout_seconds)
                self.successful_calls += 1
                return result
                
            except TimeoutError:
                logger.warning(f"⏱️ Request timed out after {self.config.timeout_seconds} seconds")
                self.failed_calls += 1
                return self.config.default_response
                
            except Exception as e:
                logger.error(f"❌ API call failed: {str(e)}")
                self.failed_calls += 1
                return f"❌ Error: {str(e)}"
    
    def _validate_input_file(self) -> pd.DataFrame:
        """
        Validate and load the input Excel file.
        
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If required columns are missing
        """
        input_path = Path(self.config.input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"❌ Input file not found: {self.config.input_file}")
        
        try:
            df = pd.read_excel(input_path)
            
            required_columns = ['SYSTEM MESSAGE', 'USER']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"❌ Missing required columns: {missing_columns}")
            
            logger.info(f"✅ Loaded {len(df)} rows from {self.config.input_file}")
            return df
            
        except Exception as e:
            raise ValueError(f"❌ Failed to read Excel file: {str(e)}")
    
    def _process_row(self, index: int, row: pd.Series) -> Tuple[int, str]:
        """
        Process a single row from the DataFrame.
        
        Args:
            index: Row index
            row: Row data
            
        Returns:
            Tuple of (index, ai_response)
        """
        self.call_count += 1
        system_message = str(row['SYSTEM MESSAGE'])
        user_message = str(row['USER'])
        
        logger.info(f"🔄 Processing call #{self.call_count} (Row {index + 1})")
        
        ai_response = self._api_call_with_timeout(system_message, user_message)
        
        logger.info(f"✅ Completed call #{self.call_count}")
        return index, ai_response
    
    def process_excel_file(self) -> None:
        """
        Main processing method to handle the entire Excel file.
        """
        start_time = datetime.datetime.now()
        logger.info("🚀 Starting Excel processing with Gemini AI")
        
        try:
            # Load and validate input
            dataframe = self._validate_input_file()
            
            # Add ASSISTANT column if it doesn't exist
            if 'ASSISTANT' not in dataframe.columns:
                dataframe['ASSISTANT'] = ''
            
            # Process rows with controlled concurrency
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                for index, row in dataframe.iterrows():
                    future = executor.submit(self._process_row, index, row)
                    futures.append(future)
                
                # Collect results
                for future in futures:
                    try:
                        index, ai_response = future.result()
                        dataframe.at[index, 'ASSISTANT'] = ai_response
                    except Exception as e:
                        logger.error(f"❌ Row processing failed: {str(e)}")
            
            # Save results
            output_path = Path(self.config.output_file)
            dataframe.to_excel(output_path, index=False)
            
            # Calculate and log statistics
            time_elapsed = (datetime.datetime.now() - start_time).total_seconds()
            
            logger.info("=" * 60)
            logger.info("📊 PROCESSING SUMMARY")
            logger.info("=" * 60)
            logger.info(f"✅ Total rows processed: {len(dataframe)}")
            logger.info(f"✅ Successful API calls: {self.successful_calls}")
            logger.info(f"❌ Failed API calls: {self.failed_calls}")
            logger.info(f"⏱️ Total processing time: {time_elapsed:.2f} seconds")
            logger.info(f"📈 Average time per call: {time_elapsed/self.call_count:.2f} seconds")
            logger.info(f"💾 Results saved to: {self.config.output_file}")
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {str(e)}")
            raise


def main():
    """Main execution function."""
    try:
        # Initialize configuration
        config = ProcessingConfig()
        
        # Create and run processor
        processor = GeminiExcelProcessor(config)
        processor.process_excel_file()
        
        print("🎉 Excel processing completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Application failed: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == '__main__':
    main()


