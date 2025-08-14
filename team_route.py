import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.models.openai import OpenAIChat
from textwrap import dedent

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Financial Analysis Agent
price_agent = Agent(
    name="PriceAgent",
    role="Answer financial queries on major tech stocks",
    model=OpenAIChat(id="gpt-4o", api_key=OPENAI_API_KEY),
    tools=[YFinanceTools(stock_price=True, company_info=True)],
    instructions=dedent("""
        Provide comprehensive financial analysis for major tech stocks:
        
        Target Stocks:
        - AAPL (Apple): Consumer electronics, services
        - MSFT (Microsoft): Software, cloud services  
        - GOOGL (Alphabet): Search, advertising, cloud
        - AMZN (Amazon): E-commerce, AWS cloud
        - TSLA (Tesla): Electric vehicles, energy
        - META (Meta): Social media, VR/AR
        - NVDA (NVIDIA): Graphics, AI chips
        - NFLX (Netflix): Streaming services
        - AMD (Advanced Micro Devices): Processors, graphics
        - CRM (Salesforce): Enterprise software
        
        Financial Metrics to Cover:
        - Current stock prices and daily changes
        - Trading volumes and market cap
        - P/E ratios and valuation metrics  
        - 52-week highs/lows and trends
        - Revenue growth and profitability
        - Analyst ratings and price targets
        - Dividend information where applicable
        - Sector performance comparisons
        
        Always provide context and explain what the numbers mean for investors.
    """).strip(),
    markdown=True,
)

# Product News Agent
news_agent = Agent(
    name="NewsAgent",
    role="Answer product announcement and tech industry news questions",
    model=OpenAIChat(id="gpt-4o", api_key=OPENAI_API_KEY),
    tools=[DuckDuckGoTools()],
    instructions=dedent("""
        Summarize product announcements and tech industry news with sentiment analysis:
        
        Companies Coverage:
        - Apple: iPhones, iPads, Macs, Apple Watch, Vision Pro, services
        - Microsoft: Windows, Office, Azure, Surface, Xbox, AI tools
        - Google: Search updates, Pixel devices, Chrome, AI/ML, cloud
        - Amazon: Echo devices, AWS services, Prime, logistics, AI
        - Tesla: Model updates, Cybertruck, energy products, FSD
        - Meta: Facebook/Instagram updates, VR headsets, AR, metaverse
        - NVIDIA: Graphics cards, AI chips, data center hardware
        - Netflix: Content releases, platform features, international expansion
        - AMD: CPU/GPU releases, server processors, gaming hardware
        - Salesforce: CRM updates, AI integration, platform features
        
        Analysis Focus:
        - Recent product launches and updates (last 30 days priority)
        - Technology breakthroughs and innovations
        - Market reception and initial reviews
        - Competitive positioning and responses
        - Industry impact and disruption potential
        - Consumer and business sentiment
        - Pricing strategies and market availability
        
        Provide specific dates, sources, and market sentiment where available.
    """).strip(),
    markdown=True,
)

# Market Analysis Agent
market_agent = Agent(
    name="MarketAgent",
    role="Answer broader market trends and industry analysis questions",
    model=OpenAIChat(id="gpt-4o", api_key=OPENAI_API_KEY),
    tools=[DuckDuckGoTools(), YFinanceTools(stock_price=True, company_info=True)],
    instructions=dedent("""
        Provide market trends and industry analysis across the tech sector:
        
        Market Analysis Areas:
        - Technology sector trends and themes
        - Industry disruption patterns
        - Regulatory impacts and policy changes
        - Economic factors affecting tech stocks
        - Competitive landscape shifts
        - Innovation cycles and adoption rates
        - Supply chain and manufacturing trends
        - International market dynamics
        
        Cross-Company Comparisons:
        - Market share analysis
        - Growth trajectory comparisons
        - Innovation leadership assessment
        - Financial performance rankings
        - Risk factor evaluations
        - Strategic positioning analysis
        
        Investment Perspective:
        - Sector rotation implications
        - Economic cycle positioning
        - Valuation comparisons
        - Growth vs value considerations
        - Risk-adjusted return potential
        
        Combine news research with financial data for comprehensive market insights.
    """).strip(),
    markdown=True,
)

# Tech Products Router System
tech_router_team = Team(
    name="Tech Products Info Router",
    mode="route",
    members=[price_agent, news_agent, market_agent],
    model=OpenAIChat(id="gpt-4o", api_key=OPENAI_API_KEY),
    instructions=dedent("""
        Route queries to the most relevant specialist agent based on question type:
        
        Routing Logic:
        
        → PriceAgent for:
        - Stock prices, financial metrics, valuations
        - Trading data, volumes, market caps
        - Earnings, revenue, profitability questions
        - Analyst ratings, price targets
        - Financial comparisons between companies
        - Investment fundamentals
        
        → NewsAgent for:
        - Product launches, announcements, updates
        - Technology features, specifications  
        - Company news, executive changes
        - Product reviews, market reception
        - Innovation announcements
        - Specific product questions
        
        → MarketAgent for:
        - Industry trends, sector analysis
        - Market dynamics, competitive landscape
        - Regulatory impacts, policy changes
        - Broader economic factors
        - Cross-company strategic analysis
        - Investment themes and strategies
        
        Consider the primary focus of each question and route to the agent with the most relevant tools and expertise. If a question spans multiple areas, choose the agent whose specialty is most central to providing a comprehensive answer.
    """).strip(),
    show_members_responses=True,
    markdown=True,
)
tech_router_team.print_response("Has Apple announced a new iPhone recently?", stream=True)
tech_router_team.print_response("What is Apple stock performance today?", stream=True)
