import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.reasoning import ReasoningTools
from agno.models.openai import OpenAIChat
from textwrap import dedent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Technical Analysis Agent
tech_agent = Agent(
    name="TechAnalyst",
    role="Analyze technology sector developments and innovation trends",
    model=OpenAIChat(id="gpt-4o", api_key=OPENAI_API_KEY),
    tools=[DuckDuckGoTools(), ReasoningTools(add_instructions=True)],
    instructions=dedent("""
        Analyze major tech companies' recent developments focusing on:
        
        Target Companies:
        - Apple (AAPL): Hardware, services, ecosystem
        - Microsoft (MSFT): Cloud, AI, enterprise solutions
        - Google/Alphabet (GOOGL): Search, cloud, AI innovations
        - Amazon (AMZN): AWS, e-commerce, logistics
        - Tesla (TSLA): EVs, energy, autonomous driving
        - Meta (META): VR/AR, social platforms, metaverse
        - NVIDIA (NVDA): AI chips, data centers, gaming
        
        Analysis Focus:
        - Recent product launches and updates
        - Technology breakthrough announcements
        - R&D investments and innovation pipeline
        - Competitive positioning changes
        - Industry disruption potential
        
        Technical Significance:
        - Rate innovation impact per company (1-10 scale)
        - Compare competitive advantages
        - Identify market-moving technologies
        - Assess long-term growth drivers
        
        Provide specific examples and cross-company comparisons.
    """).strip(),
    markdown=True,
)

# Market Sentiment Agent
market_agent = Agent(
    name="MarketSentiment",
    role="Analyze market sentiment and financial performance across major tech stocks",
    model=OpenAIChat(id="gpt-4o", api_key=OPENAI_API_KEY),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True)],
    instructions=dedent("""
        Analyze major tech stocks market performance and sentiment:
        
        Target Stocks Portfolio:
        - AAPL (Apple): Consumer tech, services revenue
        - MSFT (Microsoft): Enterprise software, cloud growth
        - GOOGL (Alphabet): Advertising, cloud, AI investments
        - AMZN (Amazon): E-commerce, AWS, logistics innovation
        - TSLA (Tesla): EV market, energy storage, autonomy
        - META (Meta): Social media, VR/AR investments
        - NVDA (NVIDIA): AI hardware, data center demand
        
        Financial Analysis:
        - Recent stock price movements (1M, 3M, 6M)
        - Trading volumes and volatility patterns
        - P/E ratios and valuation metrics
        - Revenue growth and profit margins
        - Sector performance comparison
        
        Market Sentiment Indicators:
        - Analyst ratings and price targets
        - Institutional vs retail activity
        - Options flow and sentiment
        - News sentiment impact on prices
        - Sector rotation trends
        
        Risk Assessment:
        - Market cap concentration risks
        - Regulatory environment changes
        - Economic sensitivity factors
        - Competition and market share shifts
        
        Provide specific financial metrics and comparative analysis.
    """).strip(),
    markdown=True,
)

# Synthesis Agent
synth_agent = Agent(
    name="SynthesisAgent",
    role="Combine tech sector analysis with market insights for portfolio-level intelligence",
    model=OpenAIChat(id="gpt-4o", api_key=OPENAI_API_KEY),
    instructions=dedent("""
        Synthesize technology and market analyses across major tech stocks:
        
        Portfolio Integration Analysis:
        - Connect innovation trends to market opportunities across companies
        - Identify sector-wide themes and disruption patterns
        - Assess competitive dynamics and market share shifts
        - Evaluate diversification benefits within tech sector
        
        Cross-Company Comparisons:
        - Innovation leadership rankings
        - Financial performance metrics comparison
        - Growth trajectory assessments
        - Risk-adjusted return potentials
        
        Strategic Investment Insights:
        - Sector allocation recommendations
        - Individual stock weighting suggestions
        - Timing considerations for entries/exits
        - Hedge strategies and risk management
        
        Market Dynamics Assessment:
        - Sector rotation implications
        - Economic cycle positioning
        - Regulatory impact across companies
        - Technology adoption curve positioning
        
        Actionable Portfolio Recommendations:
        - Top picks with rationale
        - Overweight/underweight suggestions
        - Key metrics to monitor for each stock
        - Risk management strategies
        - Timeline considerations for investment decisions
        
        Present a unified investment thesis for the tech sector portfolio.
    """).strip(),
    markdown=True,
)

# Collaboration Team
tech_stocks_team = Team(
    name="Tech Stocks Analysis Collaboration Team",
    mode="collaborate", 
    members=[tech_agent, market_agent, synth_agent],
    model=OpenAIChat(id="gpt-4o", api_key=OPENAI_API_KEY),
    instructions=dedent("""
        Collaborative tech sector analysis workflow:
        
        Phase 1 - Independent Sector Analysis:
        - TechAnalyst: Focus on innovation trends across AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA
        - MarketSentiment: Focus on financial metrics and market dynamics for all target stocks
        - Each agent provides comprehensive multi-company perspective
        
        Phase 2 - Portfolio Synthesis:
        - SynthesisAgent: Combines technology and market insights into portfolio strategy
        - Identifies cross-company synergies and competitive dynamics
        - Provides actionable investment recommendations
        
        Collaboration Principles:
        - Cover entire tech sector, not individual companies
        - Maintain specialized expertise per agent
        - Focus on comparative analysis and sector trends
        - Deliver portfolio-level investment intelligence
        - Consider sector rotation and market timing
    """).strip(),
    show_members_responses=True,
    markdown=True,
)

tech_stocks_team.print_response("Evaluate Google latest product impact from both technical and financial perspectives.", stream=True)