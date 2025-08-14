import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.reasoning import ReasoningTools
from agno.models.openai import OpenAIChat
from textwrap import dedent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


news_agent = Agent(
    name="NewsFetcher",
    role="Fetch news articles about stocks (AAPL, Microsoft, etc.)",
    model=OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY),
    tools=[DuckDuckGoTools()],
    instructions=dedent("""
        Retrieve recent stock news with dates and sources for major companies like:
        - Apple (AAPL)
        - Microsoft (MSFT) 
        - Google/Alphabet (GOOGL)
        - Amazon (AMZN)
        - Tesla (TSLA)
        
        Focus on:
        - Earnings reports
        - Product announcements
        - Market-moving news
        - Executive changes
        - Regulatory developments
        
        Include publication dates, sources, and key details.
        Prioritize news from the last 7 days.
    """).strip(),
    markdown=True,
)

# Analysis agent
analysis_agent = Agent(
    name="AnalysisAgent", 
    role="Analyze and synthesize sentiment from news articles",
    model=OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY),
    tools=[ReasoningTools()],
    instructions=dedent("""
        Analyze the collected news articles for:
        
        Market Sentiment Analysis:
        - Bullish vs bearish sentiment
        - Market confidence indicators
        - Investor reactions
        
        Fundamental Analysis:
        - Financial performance indicators
        - Growth prospects
        - Competitive positioning
        - Risk factors
        
        Technical Insights:
        - Trading volume patterns
        - Price movement catalysts
        - Support/resistance levels (if mentioned)
        
        Provide quantitative sentiment scores where possible (1-10 scale).
    """).strip(),
    markdown=True,
)

# Report writing agent
writer_agent = Agent(
    name="ReportWriter",
    role="Write comprehensive stock market analysis reports", 
    model=OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY),
    instructions=dedent("""
        Create a professional investment research report with these sections:
        
        1. **Executive Summary**
           - Key highlights and investment thesis
           - Overall market sentiment
           - Top recommendations
        
        2. **Market News Digest** 
           - Major developments by company
           - Earnings highlights
           - Product/service updates
        
        3. **Sentiment Analysis**
           - Market mood indicators
           - Bullish vs bearish factors
           - Risk assessment
        
        4. **Strategic Insights**
           - Investment implications
           - Sector trends
           - Timing considerations
        
        5. **Action Items**
           - Watch list recommendations  
           - Key dates to monitor
           - Risk management suggestions
        
        Use clear formatting with headers, bullet points, and tables where appropriate.
        Include disclaimer about investment risks.
    """).strip(),
    markdown=True,
)

# Team coordination
stock_analysis_team = Team(
    name="Stock Market Analysis Team",
    mode="coordinate",
    members=[news_agent, analysis_agent, writer_agent],
    model=OpenAIChat(id="gpt-4o-mini", api_key=OPENAI_API_KEY),
    instructions=dedent("""
        Execute a comprehensive stock market analysis workflow:
        
        Phase 1 - Data Collection:
        - NewsFetcher gathers recent news for major stocks
        - Focus on market-moving events and announcements
        - Ensure data quality and source credibility
        
        Phase 2 - Analysis:
        - AnalysisAgent processes news for sentiment and insights
        - Identify patterns, trends, and investment implications
        - Quantify sentiment where possible
        
        Phase 3 - Reporting:
        - ReportWriter synthesizes findings into actionable report
        - Present clear investment thesis and recommendations
        - Include appropriate risk disclaimers
        
        Collaborate effectively and ensure information flows smoothly between phases.
    """).strip(),
    show_tool_calls=True,
    markdown=True,
)


if __name__ == "__main__":

    stock_analysis_team.print_response(
    "Prepare me a report on the latest Apple product announcements.", 
            stream=True

)
    
   