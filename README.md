# Guardian AI

An advanced voice-controlled AI assistant with multi-agent research capabilities, screen analysis, and AI image generation powered by Google Gemini.

## Features

- **Voice Control**: Natural speech recognition with intelligent pause detection
- **Multiple Interaction Modes**:
  - **Image Mode**: Generate and edit images using Gemini 2.5 Flash
  - **Screen Analysis**: Analyze and describe anything visible on your screen
  - **Research Mode**: Deep web research with multi-agent system
  - **General Knowledge**: Answer questions from AI knowledge base
  - **Auto Mode**: Automatically detect user intent
- **Multi-Agent Research System**: Parallel agent coordination for comprehensive research
- **Interrupt Control**: Press hotkey (Ctrl+I) to interrupt and quickly change modes
- **Text-to-Speech**: Natural voice responses with markdown formatting support

## Architecture

### Core Components

- **Speech Recognition**: Google Speech Recognition with optimized settings
- **AI Processing**: Google Gemini 2.5 Flash & 2.5 Flash Image Preview
- **Vision System**: PIL ImageGrab for screen capture and analysis
- **Multi-Agent System**:
  - Researcher Agent (web search & data gathering)
  - Analyst Agent (content analysis & insights)
  - Critic Agent (validation & quality control)
  - Synthesizer Agent (final report generation)
- **Orchestrator**: Coordinates agent execution through defined research phases

## Prerequisites

- Python 3.8 or higher
- Microphone and speakers/headphones
- Google Gemini API key
- Windows/macOS/Linux

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hellcatxrp/Guardian_AI.git
   cd Guardian_AI
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   # Copy the example environment file
   cp .env.example .env

   # Edit .env and add your API keys
   ```

## Configuration

### Required API Keys

1. **Google Gemini API** (Required):
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add to `.env` as `GOOGLE_API_KEY`

### Optional API Keys

2. **Research Enhancement** (Optional but recommended):
   - **Brave Search API**: [Get key](https://api.search.brave.com/app/keys)
   - **Serper API** (Google Search): [Get key](https://serper.dev/)

3. **Additional Integrations** (Optional):
   - OpenAI API (image generation fallback)
   - Telegram Bot Token
   - Binance API (for crypto features)

### Environment Variables

Edit `.env` file with your configuration. See `.env.example` for all available options.

Key settings:
```env
GOOGLE_API_KEY=your_api_key_here
SPEECH_TIMEOUT=10
SPEECH_PAUSE_THRESHOLD=1.0
GEMINI_MODEL=gemini-2.5-flash
INTERRUPT_HOTKEY=ctrl+i
```

## Usage

### Start the Assistant

```bash
python main_v4_fixed.py
```

### Voice Commands

**Mode Switching**:
- Say "1" or "image mode" - Enter Image Generation Mode
- Say "2" or "screen mode" - Enter Screen Analysis Mode
- Say "3" or "research mode" - Enter Research Mode
- Say "4" or "general mode" - Enter General Knowledge Mode
- Say "5" or "auto mode" - Return to Auto-detect Mode

**Image Generation**:
- "Mountain sunset" - Generate an image
- "Make it purple" - Edit the current image
- "Add birds" - Add elements to the image
- "Brighter" - Adjust image properties

**Screen Analysis**:
- "What's on my screen?"
- "Read this text"
- "Describe this image"

**Research**:
- "Research quantum computing"
- "Find information on climate change"

**General**:
- "What is the capital of France?"
- "Explain machine learning"

**Control**:
- Press **Ctrl+I** to interrupt and quickly change modes
- Say "exit", "goodbye", or "quit" to stop

## Project Structure

```
Guardian_AI/
├── main_v4_fixed.py           # Main application
├── orchestrator.py            # Multi-agent coordinator
├── requirements.txt           # Python dependencies
├── .env.example              # Configuration template
├── README.md                 # This file
├── CLAUDE.md                 # Detailed documentation
├── agents/                   # Multi-agent system
│   ├── base_agent.py         # Base agent class
│   ├── researcher_agent.py   # Web search agent
│   ├── analyst_agent.py      # Analysis agent
│   ├── critic_agent.py       # Validation agent
│   └── synthesizer_agent.py  # Report generation agent
├── shared/                   # Shared components
│   └── knowledge_base.py     # Centralized data storage
├── integration/              # External integrations
│   └── telegram_handler.py   # Telegram bot (planned)
└── old_scripts/              # Previous versions
```

## Troubleshooting

### Common Issues

**"No module named 'speech_recognition'"**
```bash
pip install -r requirements.txt
```

**"Could not understand audio"**
- Check microphone permissions
- Reduce background noise
- Speak clearly at moderate pace
- Adjust `SPEECH_TIMEOUT` and `SPEECH_PAUSE_THRESHOLD` in `.env`

**"API key not found"**
- Verify `.env` file exists in project directory
- Check `GOOGLE_API_KEY` is set correctly
- Ensure no extra spaces around the key

**TTS not working**
- Windows: Ensure SAPI voices are installed
- Linux: Install espeak (`sudo apt install espeak`)
- macOS: Built-in voices should work automatically

### Performance Tips

1. Use `PERFORMANCE_MODE=fast` for quicker responses
2. Adjust `SPEECH_CALIBRATION_DURATION` for faster startup
3. Use specific modes instead of auto-detect for better performance
4. Close unnecessary applications for better microphone quality

## Advanced Features

### Multi-Agent Research System

The research system uses specialized agents working in parallel:

1. **Planning Phase**: Orchestrator creates research strategy
2. **Gathering Phase**: Researcher agent searches multiple sources
3. **Analysis Phase**: Analyst agent processes and extracts insights
4. **Validation Phase**: Critic agent verifies quality and accuracy
5. **Synthesis Phase**: Synthesizer agent creates final report

### Image Generation

Powered by Gemini 2.5 Flash Image Preview:
- Natural language prompts
- Iterative editing
- Context-aware modifications
- Automatic file management

### Screen Analysis

Uses computer vision to:
- Read text from screen
- Describe images and UI elements
- Answer questions about visible content
- OCR and translation support

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Security Notes

- Never commit the `.env` file
- Keep API keys secure
- Be aware screen capture may include sensitive information
- Audio is processed by Google's speech recognition service
- Generated images are saved locally

## License

This project is for educational and personal use. Respect Google's API terms of service and usage limits.

## Acknowledgments

- Google Gemini AI for language models and image generation
- Google Speech Recognition for voice input
- pyttsx3 for text-to-speech
- PIL for screen capture
- All open-source contributors

## Support

For issues, questions, or suggestions:
- Open an issue on [GitHub](https://github.com/hellcatxrp/Guardian_AI/issues)
- Check the [documentation](CLAUDE.md) for detailed information

---

Made with AI assistance
