# Guardian AI - Intelligent Voice Assistant with Multi-Agent Research System

## Overview

Guardian AI is an advanced voice-controlled assistant that combines speech recognition, computer vision, and artificial intelligence to provide intelligent responses to user queries. The system has evolved into a sophisticated multi-agent architecture capable of conducting deep research through specialized agents working in parallel coordination.

## Architecture

### Core Components

1. **Speech Recognition Module**: Converts voice input to text using Google Speech Recognition
2. **Intent Analysis Engine**: Intelligently routes questions to appropriate handlers
3. **Vision System**: Captures and analyzes screen content using PIL ImageGrab
4. **AI Processing**: Uses Google Gemini AI (1.5-flash and 2.5-flash-image-preview) for text and vision responses
5. **Text-to-Speech Engine**: Converts responses to natural speech using pyttsx3
6. **Multi-Agent Research System**: Distributed research architecture with specialized agents
7. **Orchestrator**: Coordinates agent execution through defined research phases
8. **Knowledge Base**: Centralized, thread-safe data storage for agent communication

### Key Features

- **Intelligent Routing**: Automatically determines if screen analysis is needed
- **Natural Speech Processing**: Supports natural conversation with pause detection
- **Screen Analysis**: Can analyze and describe anything visible on screen
- **General Knowledge**: Answers factual questions without screen context
- **Deep Research**: Multi-agent system for comprehensive research tasks
- **Parallel Processing**: Agents work concurrently for faster results
- **Web Integration**: Multiple search APIs (Brave Search, Serper) for comprehensive data gathering
- **Graceful Error Handling**: Recovers from failures and continues operation
- **Real-time Updates**: Progress tracking through research phases

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Microphone and speakers/headphones
- Google API key for Gemini

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install google-generativeai
pip install SpeechRecognition
pip install pyttsx3
pip install Pillow
pip install python-dotenv
pip install httpx
pip install beautifulsoup4
pip install lxml
pip install pydantic
```

### Configuration

1. Create a `.env` file in the project directory with the following settings:

```bash
# Google Gemini API Configuration
GOOGLE_API_KEY=your_api_key_here

# Optional: Research API Keys for enhanced functionality
BRAVE_SEARCH_API_KEY=your_brave_search_key
SERPER_API_KEY=your_serper_key

# Optional: Integration Keys
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

# Performance and Feature Flags
PERFORMANCE_MODE=fast
ENABLE_CONVERSATION_MEMORY=true
ENABLE_SMART_CONTEXT=true
SPEECH_TIMEOUT=10
TTS_RATE=200
GEMINI_MODEL=gemini-1.5-flash
```

2. Adjust other settings as needed (see .env file for all available options)

### Running the Application

**Main Voice Assistant:**
```bash
python main_v4_fixed.py
```

**Research System Only:**
```bash
python orchestrator.py
```

## Usage Guide

### Voice Commands

- **General Questions**: "What's the capital of France?"
- **Screen Analysis**: "What's on my screen?", "Read this text", "Describe this image"
- **Deep Research**: "Research the latest developments in quantum computing"
- **Exit Commands**: "exit", "goodbye", "quit", "stop"

### Best Practices

1. **Speak Clearly**: The system works best with clear, natural speech
2. **Wait for Prompts**: Allow the system to finish processing before speaking again
3. **Natural Pauses**: The system detects natural speech boundaries
4. **Screen Context**: Make sure relevant content is visible when asking screen-related questions

## File Structure

```
Guardian_AI/
├── main_v4_fixed.py           # Current main application
├── orchestrator.py            # Multi-agent research coordinator
├── requirements.txt           # Python dependencies
├── .env                       # Configuration file
├── CLAUDE.md                 # This documentation
├── prompt.md                 # System architecture documentation
├── agents/                   # Multi-agent research system
│   ├── __init__.py
│   ├── base_agent.py         # Abstract base class for all agents
│   ├── researcher_agent.py   # Web search and data gathering
│   ├── analyst_agent.py      # Content analysis and insights
│   ├── critic_agent.py       # Validation and quality control
│   └── synthesizer_agent.py  # Final report generation
├── shared/                   # Shared components
│   ├── __init__.py
│   └── knowledge_base.py     # Centralized data storage
├── integration/              # External integrations
│   ├── __init__.py
│   └── telegram_handler.py   # Telegram bot integration (planned)
├── old_scripts/              # Previous versions
│   ├── main.py
│   ├── main_v2.py
│   ├── main_v3.py
│   ├── main_v4.py
│   ├── main_v5.py
│   └── main_v5_optimized.py
└── logs/                     # Generated logs
    └── guardian_ai.log
```

## API Configuration

### Google Gemini Setup

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add the key to your `.env` file
4. Ensure billing is set up for API usage

### Model Selection

The system uses multiple Gemini models:
- `gemini-1.5-flash`: Default for text understanding and general responses
- `gemini-2.5-flash-image-preview`: For image analysis and generation

You can change the main model in the `.env` file:

```
GEMINI_MODEL=gemini-1.5-pro  # For higher quality responses
```

### Research API Setup

For enhanced research capabilities, configure additional APIs:

**Brave Search:**
1. Visit [Brave Search API](https://api.search.brave.com/app/keys)
2. Create an API key and add to `.env`:
```
BRAVE_SEARCH_API_KEY=your_brave_search_key
```

**Serper (Google Search):**
1. Visit [Serper.dev](https://serper.dev/)
2. Create an API key and add to `.env`:
```
SERPER_API_KEY=your_serper_key
```

## Troubleshooting

### Common Issues

**"No module named 'speech_recognition'"**
- Install missing dependencies with pip install commands above

**"Could not understand audio"**
- Check microphone permissions
- Reduce background noise
- Speak more clearly and at moderate pace

**"API key not found"**
- Verify `.env` file exists and contains GOOGLE_API_KEY
- Check API key validity in Google Cloud Console

**TTS not working**
- On Windows: Ensure SAPI voices are installed
- On Linux: Install espeak or festival
- On macOS: Built-in voices should work automatically

### Performance Optimization

1. **Reduce API Calls**: Use general mode for simple questions
2. **Optimize Screenshots**: System captures full screen - consider window-specific capture for better performance
3. **Memory Management**: Restart if running for extended periods

## Multi-Agent Research System

### Agent Architecture

The system employs a sophisticated multi-agent architecture for deep research tasks:

**Orchestrator** (`orchestrator.py`):
- Coordinates agent execution through defined phases
- Manages workflow: Planning → Gathering → Analysis → Validation → Synthesis
- Provides real-time progress updates
- Handles error recovery and resource allocation

**Base Agent** (`agents/base_agent.py`):
- Abstract base class with standardized interfaces
- Common logging, configuration, and communication patterns
- Built-in error handling and retry mechanisms
- Resource cleanup and graceful shutdown

**Researcher Agent** (`agents/researcher_agent.py`):
- Multi-strategy web search (Brave Search, Serper APIs)
- Parallel source discovery with load balancing
- Content deduplication and relevance filtering
- Source metadata collection and credibility scoring

**Analyst Agent** (`agents/analyst_agent.py`):
- Real-time content processing pipeline
- Key insight extraction using NLP techniques
- Content summarization with context preservation
- Pattern recognition and trend analysis

**Critic Agent** (`agents/critic_agent.py`):
- Validation and quality control of research findings
- Fact-checking and source verification
- Bias detection and mitigation
- Research completeness assessment

**Synthesizer Agent** (`agents/synthesizer_agent.py`):
- Final report generation with structured output
- Cross-source synthesis and correlation
- Executive summary creation
- Formatted presentation for different audiences

### Knowledge Base

**Shared Knowledge Base** (`shared/knowledge_base.py`):
- Centralized, thread-safe data storage
- Query-based data organization
- Async-compatible with proper locking
- Automatic cleanup after research completion

## Version History

### v4_fixed (Current)
- Enhanced voice recognition and screen analysis
- Multi-agent research system integration
- Async orchestration with parallel processing
- Multiple search API support (Brave Search, Serper)
- Advanced web scraping with BeautifulSoup
- Improved error handling and recovery
- Environment-based configuration system
- Real-time progress tracking

### v4 (Previous Stable)
- Basic voice recognition and screen analysis
- Intent-based routing
- Hardcoded configuration

### v5 (Legacy Enhanced)
- Environment-based configuration
- Conversation memory and context
- Improved error handling
- Modular architecture
- Enhanced logging system
- Better natural language processing

## Contributing

When modifying the code:

1. **Follow existing patterns**: Maintain the modular structure and agent-based architecture
2. **Add logging**: Use the structured logging system instead of print statements  
3. **Handle errors gracefully**: Implement retry logic and proper exception handling
4. **Update documentation**: Keep this file current with changes
5. **Test thoroughly**: Verify voice, vision, and research functionality
6. **Agent Development**: Follow the BaseAgent pattern for new agents
7. **Async Compatibility**: Use async/await patterns for concurrent operations
8. **Knowledge Base Integration**: Store and retrieve data through the shared knowledge base

## Security Notes

- **API Key Security**: Never commit the `.env` file to version control
- **Screen Privacy**: Be aware that screen capture may include sensitive information  
- **Voice Privacy**: Audio is processed by Google's speech recognition service
- **Web Scraping**: Respect robots.txt and rate limiting when gathering research data
- **Data Privacy**: Research data is temporarily stored in memory and cleaned up after use

## License

This project is for educational and personal use. Respect Google's API terms of service and usage limits.