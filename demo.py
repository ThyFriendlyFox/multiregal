#!/usr/bin/env python3
"""
Demo script for the Automatic Multivariable Regression Analysis Agent.
This script demonstrates how to use the agent to analyze house price data.
"""

import asyncio
import sys
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from regression_analyzer.agent import root_agent

# Sample CSV data for demo
SAMPLE_DATA = """sqft,bedrooms,bathrooms,age,garage,lot_size,neighborhood_score,school_rating,crime_rate,distance_to_downtown,price
1200,2,1,15,1,5000,7,8,3.2,12,180000
1800,3,2,8,2,7200,8,9,2.1,8,285000
2400,4,3,5,2,9600,9,9,1.5,15,425000
1500,3,2,12,1,6000,6,7,4.1,18,220000
2800,5,4,3,3,12000,9,10,1.2,10,520000
1100,2,1,20,0,4500,5,6,5.5,25,165000
2200,4,3,7,2,8800,8,8,2.8,12,380000
1600,3,2,10,1,6400,7,8,3.0,16,240000
3200,6,5,2,3,15000,10,10,0.8,8,680000
1400,2,1,18,1,5600,6,7,4.2,20,195000"""

async def main():
    """Run the regression analysis demo."""
    
    print("🏠 AUTOMATIC MULTIVARIABLE REGRESSION ANALYSIS DEMO")
    print("=" * 60)
    print("This demo analyzes house price data to identify the factors")
    print("that have the greatest effect on property values.\n")
    
    # Setup session and runner
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name="regression_demo",
        user_id="demo_user",
        session_id="demo_session"
    )
    
    runner = Runner(
        agent=root_agent,
        app_name="regression_demo",
        session_service=session_service
    )
    
    # Demo analysis request
    analysis_request = f"""Please analyze this house price dataset to identify which factors have the greatest effect on property values.

Here's the data:
{SAMPLE_DATA}

The target variable is 'price' (house price in dollars).

Please perform a complete analysis including:
1. Data preprocessing and exploration
2. Factor identification using multiple methods
3. Regression analysis with the best factors
4. Generate formulas showing the relationships
5. Provide insights and practical recommendations

I want to understand which features most strongly influence house prices and get a formula I can use for price prediction."""
    
    print("📊 ANALYSIS REQUEST:")
    print("-" * 30)
    print("Analyzing house price data to identify key factors...")
    print("Target variable: price")
    print("Available factors: sqft, bedrooms, bathrooms, age, garage, lot_size,")
    print("                  neighborhood_score, school_rating, crime_rate, distance_to_downtown")
    print("\n🤖 AGENT ANALYSIS:")
    print("-" * 30)
    
    # Run the analysis
    content = types.Content(
        role='user',
        parts=[types.Part(text=analysis_request)]
    )
    
    try:
        events = runner.run_async(
            user_id="demo_user",
            session_id="demo_session",
            new_message=content
        )
        
        final_response = ""
        async for event in events:
            if event.is_final_response() and event.content and event.content.parts:
                final_response = event.content.parts[0].text
                print(final_response)
                break
        
        print("\n" + "=" * 60)
        print("✅ Analysis complete! The agent has identified the key factors")
        print("   affecting house prices and generated predictive formulas.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
        sys.exit(130) 