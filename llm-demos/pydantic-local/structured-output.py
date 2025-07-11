"""
Advanced Local AI Review Analysis System.

This module provides an enterprise-grade review analysis system using local AI models
with structured output parsing, comprehensive validation, batch processing, analytics,
and export capabilities for customer feedback analysis.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from statistics import mean, median, stdev

import openai
from pydantic import BaseModel, Field, validator, ValidationError
from dotenv import load_dotenv


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('review_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SentimentLevel(Enum):
    """Sentiment levels for reviews."""
    VERY_NEGATIVE = 1
    NEGATIVE = 2
    NEUTRAL = 3
    POSITIVE = 4
    VERY_POSITIVE = 5


class ReviewCategory(Enum):
    """Review analysis categories."""
    PRICE = "price"
    PRODUCT_QUALITY = "product_quality"
    DELIVERY = "delivery"
    CUSTOMER_SERVICE = "customer_service"
    OVERALL_SATISFACTION = "overall_satisfaction"


@dataclass
class AnalysisConfig:
    """Configuration for review analysis."""
    model: str = "phi-4"
    temperature: float = 0.3
    max_tokens: int = 2000
    timeout: int = 60
    batch_size: int = 10
    enable_sentiment_analysis: bool = True
    enable_keyword_extraction: bool = True
    save_intermediate_results: bool = True


class ReviewTagged(BaseModel):
    """Enhanced review analysis model with comprehensive validation."""
    
    id: int = Field(..., ge=0, description="The unique ID of the review")
    review: str = Field(..., min_length=1, max_length=5000, description="The original review text")
    
    # Rating fields (0-5 scale)
    price: int = Field(..., ge=0, le=5, description="Price rating (0=no data, 1=very poor, 5=excellent)")
    product_quality: int = Field(..., ge=0, le=5, description="Product quality rating")
    delivery: int = Field(..., ge=0, le=5, description="Delivery experience rating")
    customer_service: int = Field(..., ge=0, le=5, description="Customer service rating")
    overall_satisfaction: int = Field(..., ge=0, le=5, description="Overall satisfaction rating")
    
    # Analysis metadata
    insufficient_data: bool = Field(False, description="True if insufficient data for analysis")
    confidence_score: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in the analysis")
    sentiment: str = Field("neutral", description="Overall sentiment of the review")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    
    # Timestamps
    analyzed_at: Optional[datetime] = Field(default_factory=datetime.now, description="Analysis timestamp")
    
    @validator('sentiment')
    def validate_sentiment(cls, v):
        """Validate sentiment values."""
        valid_sentiments = ["very_negative", "negative", "neutral", "positive", "very_positive"]
        if v.lower() not in valid_sentiments:
            return "neutral"
        return v.lower()
    
    @validator('keywords')
    def validate_keywords(cls, v):
        """Clean and validate keywords."""
        if not v:
            return []
        # Remove empty strings and duplicates, limit to 10 keywords
        cleaned = list(set([kw.strip().lower() for kw in v if kw.strip()]))
        return cleaned[:10]
    
    @property
    def average_rating(self) -> float:
        """Calculate average rating excluding zeros."""
        ratings = [self.price, self.product_quality, self.delivery, 
                  self.customer_service, self.overall_satisfaction]
        non_zero_ratings = [r for r in ratings if r > 0]
        return mean(non_zero_ratings) if non_zero_ratings else 0.0
    
    @property
    def rating_summary(self) -> Dict[str, Any]:
        """Get a summary of all ratings."""
        return {
            "price": self.price,
            "product_quality": self.product_quality,
            "delivery": self.delivery,
            "customer_service": self.customer_service,
            "overall_satisfaction": self.overall_satisfaction,
            "average": round(self.average_rating, 2),
            "sentiment": self.sentiment,
            "confidence": round(self.confidence_score, 2)
        }


class BatchReviewResult(BaseModel):
    """Results from batch review analysis."""
    
    total_reviews: int = Field(..., description="Total number of reviews processed")
    successful_analyses: int = Field(..., description="Number of successful analyses")
    failed_analyses: int = Field(..., description="Number of failed analyses")
    average_processing_time: float = Field(..., description="Average time per review")
    
    # Aggregate statistics
    overall_sentiment_distribution: Dict[str, int] = Field(default_factory=dict)
    average_ratings: Dict[str, float] = Field(default_factory=dict)
    top_keywords: List[str] = Field(default_factory=list)
    
    # Individual results
    tagged_reviews: List[ReviewTagged] = Field(default_factory=list)
    
    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    processing_duration: float = Field(0.0, description="Total processing time in seconds")


class ReviewAnalysisPrompts:
    """Collection of analysis prompts for different use cases."""
    
    @staticmethod
    def get_comprehensive_prompt() -> str:
        """Get comprehensive review analysis prompt."""
        return """You are an advanced review analysis AI agent specializing in customer feedback analysis.

Your task is to analyze customer reviews and provide structured ratings on a 0-5 scale where:
- 0: No relevant information available
- 1: Very poor/negative experience
- 2: Poor/mostly negative experience  
- 3: Average/neutral experience
- 4: Good/positive experience
- 5: Excellent/very positive experience

Analyze these specific categories:
1. **Price**: Value for money, pricing satisfaction, cost-effectiveness
2. **Product Quality**: Build quality, functionality, durability, design
3. **Delivery**: Shipping speed, packaging, delivery experience
4. **Customer Service**: Support responsiveness, helpfulness, problem resolution
5. **Overall Satisfaction**: General experience and recommendation likelihood

Additionally, provide:
- Overall sentiment classification
- Confidence score (0.0-1.0) for your analysis
- Key relevant keywords (max 10)
- Flag if insufficient data exists for reliable analysis

Be objective, consistent, and base ratings strictly on the review content."""
    
    @staticmethod
    def get_sentiment_focused_prompt() -> str:
        """Get sentiment-focused analysis prompt."""
        return """You are a sentiment analysis specialist. Analyze the customer review for emotional tone and satisfaction levels.

Focus on identifying:
- Emotional indicators (positive/negative language)
- Satisfaction levels across different aspects
- Overall customer sentiment
- Confidence in your sentiment assessment

Provide structured ratings and clear sentiment classification."""
    
    @staticmethod
    def get_keyword_extraction_prompt() -> str:
        """Get keyword extraction prompt."""
        return """You are a keyword extraction specialist. Identify the most important and relevant keywords from customer reviews.

Extract keywords that represent:
- Product features mentioned
- Service aspects discussed
- Emotional indicators
- Specific problems or praises
- Business-relevant terms

Limit to the 10 most relevant keywords."""


class LocalReviewAnalyzer:
    """Advanced local AI review analyzer with comprehensive features."""
    
    def __init__(self, base_url: str = "http://localhost:1234/v1/", api_key: Optional[str] = None):
        """Initialize the review analyzer."""
        self.client = openai.Client(
            base_url=base_url,
            api_key=api_key or "dummy-key"
        )
        self.base_url = base_url
        self.analysis_history: List[ReviewTagged] = []
        
        logger.info(f"Review analyzer initialized with base URL: {base_url}")
    
    def analyze_single_review(
        self,
        review_text: str,
        review_id: int,
        config: AnalysisConfig,
        custom_prompt: Optional[str] = None
    ) -> ReviewTagged:
        """
        Analyze a single review with comprehensive error handling.
        
        Args:
            review_text: The review text to analyze
            review_id: Unique identifier for the review
            config: Analysis configuration
            custom_prompt: Optional custom analysis prompt
            
        Returns:
            Structured review analysis result
        """
        start_time = time.time()
        
        try:
            # Use custom prompt or default comprehensive prompt
            system_prompt = custom_prompt or ReviewAnalysisPrompts.get_comprehensive_prompt()
            
            # Prepare the user message with review context
            user_message = f"""Review ID: {review_id}
Review Text: "{review_text}"

Please analyze this review and provide structured ratings for all categories."""
            
            logger.info(f"Analyzing review {review_id}")
            
            response = self.client.beta.chat.completions.parse(
                model=config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                response_format=ReviewTagged,
            )
            
            parsed_result = response.choices[0].message.parsed
            
            if parsed_result is None:
                raise ValueError("Failed to parse structured output")
            
            # Ensure ID and review text are set correctly
            parsed_result.id = review_id
            parsed_result.review = review_text
            
            processing_time = time.time() - start_time
            logger.info(f"Review {review_id} analyzed successfully in {processing_time:.3f}s")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Failed to analyze review {review_id}: {e}")
            
            # Return a default failed analysis
            return ReviewTagged(
                id=review_id,
                review=review_text,
                price=0,
                product_quality=0,
                delivery=0,
                customer_service=0,
                overall_satisfaction=0,
                insufficient_data=True,
                confidence_score=0.0,
                sentiment="neutral",
                keywords=[]
            )
    
    def analyze_batch_reviews(
        self,
        reviews: List[Union[str, Dict[str, Any]]],
        config: AnalysisConfig,
        custom_prompt: Optional[str] = None
    ) -> BatchReviewResult:
        """
        Analyze multiple reviews in batch with comprehensive analytics.
        
        Args:
            reviews: List of review texts or dicts with 'id' and 'text' keys
            config: Analysis configuration
            custom_prompt: Optional custom analysis prompt
            
        Returns:
            Batch analysis results with statistics
        """
        start_time = time.time()
        
        logger.info(f"Starting batch analysis of {len(reviews)} reviews")
        
        # Normalize review input format
        normalized_reviews = []
        for i, review in enumerate(reviews):
            if isinstance(review, str):
                normalized_reviews.append({"id": i + 1, "text": review})
            elif isinstance(review, dict):
                normalized_reviews.append({
                    "id": review.get("id", i + 1),
                    "text": review.get("text", str(review))
                })
            else:
                normalized_reviews.append({"id": i + 1, "text": str(review)})
        
        # Process reviews in batches
        all_results = []
        successful_count = 0
        failed_count = 0
        processing_times = []
        
        for i in range(0, len(normalized_reviews), config.batch_size):
            batch = normalized_reviews[i:i + config.batch_size]
            
            logger.info(f"Processing batch {i//config.batch_size + 1}/{(len(normalized_reviews)-1)//config.batch_size + 1}")
            
            for review_data in batch:
                batch_start = time.time()
                
                try:
                    result = self.analyze_single_review(
                        review_data["text"],
                        review_data["id"],
                        config,
                        custom_prompt
                    )
                    
                    all_results.append(result)
                    
                    if not result.insufficient_data:
                        successful_count += 1
                    else:
                        failed_count += 1
                    
                    batch_time = time.time() - batch_start
                    processing_times.append(batch_time)
                    
                except Exception as e:
                    logger.error(f"Failed to process review {review_data['id']}: {e}")
                    failed_count += 1
            
            # Save intermediate results if enabled
            if config.save_intermediate_results and i > 0:
                self._save_intermediate_results(all_results, i)
        
        # Calculate analytics
        total_time = time.time() - start_time
        avg_processing_time = mean(processing_times) if processing_times else 0
        
        # Generate comprehensive analytics
        analytics = self._generate_batch_analytics(all_results)
        
        result = BatchReviewResult(
            total_reviews=len(reviews),
            successful_analyses=successful_count,
            failed_analyses=failed_count,
            average_processing_time=avg_processing_time,
            overall_sentiment_distribution=analytics["sentiment_distribution"],
            average_ratings=analytics["average_ratings"],
            top_keywords=analytics["top_keywords"],
            tagged_reviews=all_results,
            processing_duration=total_time
        )
        
        logger.info(f"Batch analysis completed: {successful_count} successful, {failed_count} failed")
        
        return result
    
    def _generate_batch_analytics(self, results: List[ReviewTagged]) -> Dict[str, Any]:
        """Generate comprehensive analytics from batch results."""
        if not results:
            return {
                "sentiment_distribution": {},
                "average_ratings": {},
                "top_keywords": []
            }
        
        # Sentiment distribution
        sentiment_counts = {}
        for result in results:
            sentiment = result.sentiment
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        # Average ratings (excluding zeros)
        rating_categories = ["price", "product_quality", "delivery", "customer_service", "overall_satisfaction"]
        average_ratings = {}
        
        for category in rating_categories:
            ratings = [getattr(result, category) for result in results if getattr(result, category) > 0]
            average_ratings[category] = round(mean(ratings), 2) if ratings else 0.0
        
        # Top keywords
        all_keywords = []
        for result in results:
            all_keywords.extend(result.keywords)
        
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        top_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        top_keywords = [keyword for keyword, count in top_keywords]
        
        return {
            "sentiment_distribution": sentiment_counts,
            "average_ratings": average_ratings,
            "top_keywords": top_keywords
        }
    
    def _save_intermediate_results(self, results: List[ReviewTagged], batch_index: int):
        """Save intermediate results during batch processing."""
        filename = f"intermediate_results_batch_{batch_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            data = [result.dict() for result in results]
            with open(filename, "w") as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Intermediate results saved to {filename}")
        except Exception as e:
            logger.warning(f"Failed to save intermediate results: {e}")
    
    def export_results(
        self,
        results: Union[BatchReviewResult, List[ReviewTagged]],
        filename: Optional[str] = None,
        format_type: str = "json"
    ) -> str:
        """Export analysis results to various formats."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"review_analysis_{timestamp}.{format_type}"
        
        try:
            if format_type.lower() == "json":
                with open(filename, "w") as f:
                    if isinstance(results, BatchReviewResult):
                        json.dump(results.dict(), f, indent=2, default=str)
                    else:
                        json.dump([r.dict() for r in results], f, indent=2, default=str)
            
            elif format_type.lower() == "csv":
                import csv
                
                # Extract data for CSV
                if isinstance(results, BatchReviewResult):
                    data = results.tagged_reviews
                else:
                    data = results
                
                with open(filename, "w", newline='', encoding='utf-8') as f:
                    if data:
                        writer = csv.DictWriter(f, fieldnames=data[0].dict().keys())
                        writer.writeheader()
                        for result in data:
                            writer.writerow(result.dict())
            
            logger.info(f"Results exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            raise
    
    def generate_summary_report(self, results: BatchReviewResult) -> str:
        """Generate a comprehensive summary report."""
        report_lines = [
            "="*60,
            "CUSTOMER REVIEW ANALYSIS SUMMARY REPORT",
            "="*60,
            f"Analysis Date: {results.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Reviews Processed: {results.total_reviews}",
            f"Successful Analyses: {results.successful_analyses}",
            f"Failed Analyses: {results.failed_analyses}",
            f"Success Rate: {(results.successful_analyses/results.total_reviews)*100:.1f}%",
            f"Processing Duration: {results.processing_duration:.2f} seconds",
            f"Average Processing Time: {results.average_processing_time:.3f} seconds per review",
            "",
            "AVERAGE RATINGS:",
            "-"*20
        ]
        
        for category, rating in results.average_ratings.items():
            report_lines.append(f"{category.replace('_', ' ').title()}: {rating}/5.0")
        
        report_lines.extend([
            "",
            "SENTIMENT DISTRIBUTION:",
            "-"*25
        ])
        
        total_with_sentiment = sum(results.overall_sentiment_distribution.values())
        for sentiment, count in results.overall_sentiment_distribution.items():
            percentage = (count / total_with_sentiment) * 100 if total_with_sentiment > 0 else 0
            report_lines.append(f"{sentiment.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "TOP KEYWORDS:",
            "-"*15
        ])
        
        for i, keyword in enumerate(results.top_keywords[:10], 1):
            report_lines.append(f"{i:2d}. {keyword}")
        
        report_lines.append("="*60)
        
        return "\n".join(report_lines)


def demo_comprehensive_analysis():
    """Demonstrate comprehensive review analysis capabilities."""
    try:
        # Initialize analyzer
        analyzer = LocalReviewAnalyzer()
        
        print("🚀 Advanced Local AI Review Analysis System Demo")
        print("=" * 60)
        
        # Sample reviews with varying content
        sample_reviews = [
            {
                "id": 1,
                "text": "The product was great quality and exactly as described. Fast delivery and excellent customer service when I had questions. Worth every penny!"
            },
            {
                "id": 2,
                "text": "Terrible experience. Product broke after one day, delivery took 3 weeks, and customer service hung up on me twice. Overpriced garbage."
            },
            {
                "id": 3,
                "text": "Decent product for the price. Delivery was on time. Had one small issue but customer service resolved it quickly."
            },
            {
                "id": 4,
                "text": "Amazing quality! Super fast shipping and the customer service team went above and beyond. Highly recommend!"
            },
            {
                "id": 5,
                "text": "Can I get a refund on my pizza order? It's cold and has pineapple on it."
            }
        ]
        
        # Configuration
        config = AnalysisConfig(
            model="phi-4",
            temperature=0.1,
            batch_size=3,
            enable_sentiment_analysis=True,
            save_intermediate_results=True
        )
        
        # Perform batch analysis
        print(f"\n📊 Analyzing {len(sample_reviews)} sample reviews...")
        results = analyzer.analyze_batch_reviews(sample_reviews, config)
        
        # Display summary
        print(f"\n📈 Analysis Results:")
        print(f"✅ Successful: {results.successful_analyses}")
        print(f"❌ Failed: {results.failed_analyses}")
        print(f"⏱️ Total time: {results.processing_duration:.2f}s")
        print(f"🎯 Average rating: {results.average_ratings.get('overall_satisfaction', 0)}/5")
        
        # Show individual results
        print(f"\n📝 Individual Review Analysis:")
        for review in results.tagged_reviews[:3]:  # Show first 3
            print(f"\nReview {review.id}:")
            print(f"  Text: {review.review[:100]}...")
            print(f"  Ratings: {review.rating_summary}")
            print(f"  Keywords: {review.keywords[:5]}")
        
        # Generate and save comprehensive report
        report = analyzer.generate_summary_report(results)
        print(f"\n📄 Summary Report:")
        print(report)
        
        # Export results
        json_file = analyzer.export_results(results, format_type="json")
        csv_file = analyzer.export_results(results, format_type="csv")
        
        print(f"\n💾 Results exported:")
        print(f"  📊 JSON: {json_file}")
        print(f"  📈 CSV: {csv_file}")
        
        # Save report
        report_file = f"review_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_file, "w") as f:
            f.write(report)
        print(f"  📄 Report: {report_file}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo error: {e}")


def interactive_analysis():
    """Interactive review analysis mode."""
    analyzer = LocalReviewAnalyzer()
    config = AnalysisConfig()
    
    print("🔍 Interactive Review Analysis System")
    print("📝 Commands: 'analyze', 'batch', 'config', 'export', 'quit'")
    print("=" * 60)
    
    analysis_results = []
    
    while True:
        try:
            command = input("\nCommand: ").strip().lower()
            
            if command == 'quit':
                print("👋 Goodbye!")
                break
            
            elif command == 'analyze':
                review_text = input("Enter review text: ").strip()
                if review_text:
                    result = analyzer.analyze_single_review(review_text, len(analysis_results) + 1, config)
                    analysis_results.append(result)
                    
                    print(f"\n📊 Analysis Result:")
                    print(f"Ratings: {result.rating_summary}")
                    print(f"Sentiment: {result.sentiment}")
                    print(f"Keywords: {result.keywords}")
                    print(f"Confidence: {result.confidence_score:.2f}")
            
            elif command == 'batch':
                print("Enter reviews (one per line, empty line to finish):")
                reviews = []
                while True:
                    review = input("> ").strip()
                    if not review:
                        break
                    reviews.append(review)
                
                if reviews:
                    batch_results = analyzer.analyze_batch_reviews(reviews, config)
                    analysis_results.extend(batch_results.tagged_reviews)
                    
                    print(f"\n📈 Batch Results:")
                    print(f"Processed: {batch_results.total_reviews}")
                    print(f"Successful: {batch_results.successful_analyses}")
                    print(f"Average ratings: {batch_results.average_ratings}")
            
            elif command == 'config':
                print(f"Current config: {config}")
                
            elif command == 'export':
                if analysis_results:
                    filename = analyzer.export_results(analysis_results)
                    print(f"💾 Results exported to: {filename}")
                else:
                    print("❌ No results to export")
            
            else:
                print("❌ Unknown command")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Interactive analysis error: {e}")
            print(f"❌ Error: {e}")


def main():
    """Main execution function."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_analysis()
    else:
        demo_comprehensive_analysis()


if __name__ == '__main__':
    main()