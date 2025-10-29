import os
import io
import signal
import sys
import google.generativeai as genai
import speech_recognition as sr
import pyttsx3
from PIL import Image, ImageGrab
import threading
import time
from datetime import datetime
from dotenv import load_dotenv
from orchestrator import Orchestrator
import asyncio
import json
import base64
import keyboard

# --- CONFIGURATION ---
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Configure the Gemini models
understanding_model = genai.GenerativeModel('gemini-2.5-flash')
image_model = genai.GenerativeModel('gemini-2.5-flash-image-preview')  # Nano Banana for image generation and editing

brave_search_api_key = os.getenv("BRAVE_SEARCH_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

# Speech Recognition Configuration
SPEECH_TIMEOUT = float(os.getenv("SPEECH_TIMEOUT", "10"))  # Time to wait for speech to start
SPEECH_PAUSE_THRESHOLD = float(os.getenv("SPEECH_PAUSE_THRESHOLD", "1.0"))  # Silence duration to end recording
SPEECH_MAX_RETRIES = int(os.getenv("SPEECH_MAX_RETRIES", "3"))  # Max attempts per command
SPEECH_CALIBRATION_DURATION = float(os.getenv("SPEECH_CALIBRATION_DURATION", "1.0"))  # Noise calibration time
SPEECH_PHRASE_THRESHOLD = float(os.getenv("SPEECH_PHRASE_THRESHOLD", "0.3"))  # Min seconds before phrase starts
SPEECH_NON_SPEAKING_DURATION = float(os.getenv("SPEECH_NON_SPEAKING_DURATION", "0.8"))  # Non-speaking audio to keep

orchestrator_config = {
    "researcher": {
        "brave_search_api_key": brave_search_api_key,
        "serper_api_key": serper_api_key
    },
    "analyst": {
        "llm_model": understanding_model
    },
    "critic": {
        "llm_model": understanding_model
    },
    "synthesizer": {
        "llm_model": understanding_model
    }
}
orchestrator = Orchestrator(orchestrator_config) # Initialize the Orchestrator with config

# Path for saving generated images (configurable via environment variable)
IMAGE_SAVE_PATH = os.getenv("IMAGE_SAVE_PATH", os.path.join(os.getcwd(), "generated_images"))

# Session memory for image generation and editing
image_session_memory = {
    "current_image": None,
    "generated_images": [],
    "conversation_context": []
}

# Global mode state management
current_mode = {
    "mode": "auto",  # auto, image, screen, research, general
    "display_name": "Auto-detect",
    "description": "Automatically detects your intent"
}

# Available modes configuration
available_modes = {
    "1": {"mode": "image", "display_name": "Image Mode", "description": "Generate and edit images with Nano Banana"},
    "2": {"mode": "screen", "display_name": "Screen Analysis", "description": "Analyze what's on your screen"},
    "3": {"mode": "research", "display_name": "Research Mode", "description": "Deep web research with multi-agent system"},
    "4": {"mode": "general", "display_name": "General Knowledge", "description": "Answer questions from my knowledge base"},
    "5": {"mode": "auto", "display_name": "Auto-detect", "description": "Automatically detects your intent (current behavior)"}
}

# Global flag for graceful shutdown
shutdown_flag = threading.Event()

# Global interrupt system
interrupt_flag = threading.Event()
interrupt_hotkey = os.getenv("INTERRUPT_HOTKEY", "f1").lower()
hotkey_listener = None

# Global speech recognition components
recognizer = None
microphone = None

# Global TTS engine reference for interruption
current_tts_engine = None
tts_lock = threading.Lock()  # Ensure only one TTS operation at a time

# --- INTERRUPT SYSTEM ---
def on_hotkey_press():
    """Handle hotkey press for interrupting operations."""
    print(f"\n[!] [INTERRUPT] Hotkey ({interrupt_hotkey.upper()}) pressed - Interrupting current operation!")
    interrupt_flag.set()
    
    # Stop current TTS if running
    if current_tts_engine:
        try:
            print("[INTERRUPT] Stopping TTS...")
            current_tts_engine.stop()
        except:
            pass

def start_hotkey_listener():
    """Start the global hotkey listener using keyboard library."""
    global hotkey_listener
    
    try:
        print(f"[HOTKEY] Setting up global hotkey: {interrupt_hotkey.upper()}")
        print(f"[DEBUG] Using hotkey string: '{interrupt_hotkey}'")
        
        # Use keyboard library - much simpler syntax
        keyboard.add_hotkey(interrupt_hotkey, on_hotkey_press)
        hotkey_listener = True  # Just mark as active
        
        print(f"[HOTKEY] Global hotkey listener started - Press {interrupt_hotkey.upper()} to interrupt operations")
        
    except Exception as e:
        print(f"[ERROR] Failed to setup hotkey listener: {e}")
        print(f"[DEBUG] Error details: {str(e)}")
        print("[WARNING] Hotkey functionality disabled")

def stop_hotkey_listener():
    """Stop the hotkey listener."""
    global hotkey_listener
    if hotkey_listener:
        try:
            keyboard.remove_hotkey(interrupt_hotkey)
            hotkey_listener = False
            print("[HOTKEY] Global hotkey listener stopped")
        except Exception as e:
            print(f"[DEBUG] Error stopping hotkey: {e}")

def display_quick_mode_menu():
    """Display a compact mode selection menu after interrupt."""
    print("\n" + "[!] INTERRUPT - QUICK MODE SELECTION")
    print("="*50)
    
    for key, mode_info in available_modes.items():
        indicator = ">" if current_mode["mode"] == mode_info["mode"] else " "
        print(f"{indicator} [{key}] {mode_info['display_name']}")
    
    print(f"\n* Current: {current_mode['display_name']}")
    print("> Say number (1-5) to switch modes:")

# --- SIGNAL HANDLERS ---
def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print("\n[STOP] Shutting down gracefully...")
    stop_hotkey_listener()
    shutdown_flag.set()
    interrupt_flag.set()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def initialize_speech_system():
    """Pre-initialize speech recognition components to reduce first-use delay."""
    global recognizer, microphone
    
    print("[DEBUG] [INIT] Initializing speech recognition system...")
    init_start = time.time()
    
    try:
        # Initialize recognizer with optimized settings
        recognizer_start = time.time()
        recognizer = sr.Recognizer()
        # Set optimized speech detection parameters
        recognizer.pause_threshold = SPEECH_PAUSE_THRESHOLD
        recognizer.phrase_threshold = SPEECH_PHRASE_THRESHOLD  # Minimum seconds of non-silent audio before phrase starts
        recognizer.non_speaking_duration = SPEECH_NON_SPEAKING_DURATION  # Duration of non-speaking audio to keep after phrase ends
        print(f"[TIME] [INIT] Recognizer created in {time.time() - recognizer_start:.3f}s")
        
        # Initialize and test microphone
        mic_start = time.time()
        microphone = sr.Microphone()
        
        # List available microphones for debugging
        mic_list = sr.Microphone.list_microphone_names()
        print(f"[MIC] [INIT] Found {len(mic_list)} microphone(s)")
        if mic_list:
            device_idx = microphone.device_index
            if device_idx is not None and device_idx < len(mic_list):
                print(f"[MIC] [INIT] Default microphone: {mic_list[device_idx]}")
            else:
                print(f"[MIC] [INIT] Default microphone: System Default")
        
        # Pre-warm the microphone with a quick calibration
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=SPEECH_CALIBRATION_DURATION)
        
        mic_duration = time.time() - mic_start
        print(f"[TIME] [INIT] Microphone initialized and calibrated in {mic_duration:.3f}s")
        
        total_init_time = time.time() - init_start
        print(f"[OK] [INIT] Speech system ready in {total_init_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] [INIT] Failed to initialize speech system: {e}")
        return False

# --- MODE SELECTION SYSTEM ---
def display_mode_selection_menu():
    """Display the mode selection menu to the user."""
    global current_mode
    
    print("\n" + "="*70)
    print("* MODE SELECTION - Choose how you want to interact:")
    print("="*70)
    
    for key, mode_info in available_modes.items():
        indicator = ">" if current_mode["mode"] == mode_info["mode"] else " "
        print(f"{indicator} [{key}] {mode_info['display_name']}")
        print(f"    {mode_info['description']}")
        print()
    
    print(f"* Current mode: {current_mode['display_name']}")
    print("="*70)
    print("> Say the number (1-5) or mode name to switch, or your question/command:")
    print()

def parse_mode_selection(user_input):
    """Parse user input to check if it's a mode selection command."""
    global current_mode
    
    user_input_lower = user_input.lower().strip()
    
    # First, check for direct number selection (highest priority)
    if user_input_lower in available_modes:
        print(f"[DEBUG] Detected number mode selection: {user_input_lower}")
        return available_modes[user_input_lower]
    
    # Check for explicit mode switching phrases ONLY
    explicit_mode_switches = [
        "image mode", "picture mode", "photo mode",
        "screen mode", "screen analysis", 
        "research mode", "deep research",
        "general mode", "general knowledge", "knowledge mode",
        "auto mode", "auto detect", "auto-detect",
        "switch to image", "switch to screen", "switch to research", 
        "switch to general", "switch to auto",
        "change to image", "change to screen", "change to research",
        "change to general", "change to auto"
    ]
    
    for switch_phrase in explicit_mode_switches:
        if switch_phrase in user_input_lower:
            print(f"[DEBUG] Detected explicit mode switch: {switch_phrase}")
            # Determine which mode this maps to
            if "image" in switch_phrase or "picture" in switch_phrase or "photo" in switch_phrase:
                return available_modes["1"]  # Image Mode
            elif "screen" in switch_phrase:
                return available_modes["2"]  # Screen Analysis
            elif "research" in switch_phrase:
                return available_modes["3"]  # Research Mode
            elif "general" in switch_phrase or "knowledge" in switch_phrase:
                return available_modes["4"]  # General Knowledge
            elif "auto" in switch_phrase:
                return available_modes["5"]  # Auto-detect
    
    # IMPORTANT: Exclude content that just mentions mode words
    # These should NOT trigger mode switches
    content_exclusions = [
        "an image of", "create an image", "generate an image", "make an image",
        "draw an image", "create a picture", "generate a picture", "make a picture",
        "show me an image", "I want an image", "can you create an image",
        "please generate an image", "image of a", "picture of a"
    ]
    
    for exclusion in content_exclusions:
        if exclusion in user_input_lower:
            print(f"[DEBUG] Content exclusion detected: '{exclusion}' - NOT a mode switch")
            return None
    
    # Only check for standalone mode words if they're not part of content
    # This is much more restrictive than before
    words = user_input_lower.split()
    
    # Single word mode switches (but be very careful)
    if len(words) == 1:
        single_word_modes = {
            "image": available_modes["1"],
            "picture": available_modes["1"], 
            "photo": available_modes["1"],
            "screen": available_modes["2"],
            "research": available_modes["3"],
            "general": available_modes["4"],
            "knowledge": available_modes["4"],
            "auto": available_modes["5"]
        }
        
        if user_input_lower in single_word_modes:
            print(f"[DEBUG] Single word mode switch: {user_input_lower}")
            return single_word_modes[user_input_lower]
    
    print(f"[DEBUG] No mode switch detected in: '{user_input}'")
    return None

def set_mode(mode_info):
    """Set the current mode and provide confirmation."""
    global current_mode
    
    current_mode = mode_info.copy()
    
    print(f"\n* MODE CHANGED")
    print(f"✅ Now in: {current_mode['display_name']}")
    print(f"📝 {current_mode['description']}")
    
    if current_mode['mode'] == 'image':
        if image_session_memory["current_image"]:
            print(f"🖼️  Current image: {os.path.basename(image_session_memory['current_image']['filepath'])}")
        print("💡 Tip: Say things like 'mountain sunset' to generate, or 'make it purple' to edit")
    elif current_mode['mode'] == 'screen':
        print("💡 Tip: Ask me about what's visible on your screen")
    elif current_mode['mode'] == 'research':
        print("💡 Tip: Ask me to research any topic for comprehensive information")
    elif current_mode['mode'] == 'general':
        print("💡 Tip: Ask me questions from my knowledge base")
    elif current_mode['mode'] == 'auto':
        print("💡 Tip: I'll automatically detect what you want to do")
    
    print()

def get_mode_specific_intent(user_question):
    """Get intent based on current mode, bypassing complex intent detection."""
    global current_mode
    
    mode = current_mode["mode"]
    
    if mode == "image":
        # In image mode, everything is image-related
        if image_session_memory["current_image"] is not None:
            # We have an image - check if this sounds like editing
            
            # Strong editing indicators
            strong_edit_indicators = [
                'make it', 'change it', 'edit it', 'modify it', 'alter it', 'adjust it',
                'make the', 'change the', 'edit the', 'modify the', 'brighter', 'darker',
                'bigger', 'smaller', 'lighter', 'more', 'less', 'add ', 'remove ',
                'replace', 'fix', 'improve', 'enhance'
            ]
            
            # Weak editing indicators (single descriptive words)
            weak_edit_indicators = [
                'purple', 'blue', 'red', 'green', 'yellow', 'orange', 'pink',
                'bright', 'dark', 'light', 'colorful', 'detailed', 'simple',
                'cartoon', 'realistic', 'abstract', 'soft', 'sharp'
            ]
            
            question_lower = user_question.lower()
            
            # Check for strong editing indicators
            if any(indicator in question_lower for indicator in strong_edit_indicators):
                return 'image_edit'
            
            # Check for weak indicators with short phrases (likely incomplete speech)
            words = question_lower.split()
            if len(words) <= 3 and any(indicator in question_lower for indicator in weak_edit_indicators):
                return 'image_edit'
            
            # Check for object + descriptor combinations
            objects = ['sky', 'mountain', 'tree', 'water', 'sun', 'cloud', 'grass', 'building']
            descriptors = weak_edit_indicators + ['big', 'small', 'tall', 'wide']
            
            has_object = any(obj in question_lower for obj in objects)
            has_descriptor = any(desc in question_lower for desc in descriptors)
            
            if has_object and has_descriptor:
                return 'image_edit'
        
        # In image mode, if it's not clearly editing, it's generation
        print(f"[DEBUG] Image Mode: Treating '{user_question}' as generation")
        return 'image_generate'
    
    elif mode == "screen":
        return 'screen'
    elif mode == "research":
        return 'research'
    elif mode == "general":
        return 'general'
    elif mode == "auto":
        return None  # Use the existing intent detection logic
    
    return None

# --- AGENTIC CONTROLLER ---
def analyze_question_intent(question):
    """Intelligent agent that determines if a question needs screen analysis,
    image generation, image editing, or is a general question.
    Returns: 'screen', 'image_generate', 'image_edit', 'research', or 'general'
    """
    global image_session_memory
    print("[AI] Analyzing question intent...")
    question_lower = question.lower()

    # 1. Highest Priority: Image Editing (only if we have a current image)
    if image_session_memory["current_image"] is not None:
        print(f"[DEBUG] Current image exists, checking for edit intent...")
        
        # Standard edit keywords
        edit_keywords = [
            'edit the image', 'modify the image', 'change the image', 'update the image',
            'make it', 'change it', 'edit it', 'modify it', 'alter it',
            'make the image', 'change the image', 'adjust the image',
            'brighter', 'darker', 'bigger', 'smaller', 'different color',
            'add a', 'remove the', 'replace the', 'move the',
            'more detailed', 'less detailed', 'different style',
            'make another', 'create another', 'generate another',
            'similar to', 'like the', 'variation of'
        ]
        
        # Check for explicit edit commands
        if any(keyword in question_lower for keyword in edit_keywords):
            print(f"[DEBUG] Found explicit edit keyword")
            return 'image_edit'
        
        # Check for contextual editing (referencing "it" or "the image")
        contextual_edit_phrases = [
            'it ', 'the image', 'this image', 'that image', 'the picture', 'this picture'
        ]
        action_words = [
            'brighter', 'darker', 'bigger', 'smaller', 'different', 'change', 'modify', 
            'edit', 'alter', 'adjust', 'improve', 'enhance', 'fix'
        ]
        
        has_contextual_reference = any(phrase in question_lower for phrase in contextual_edit_phrases)
        has_action_word = any(word in question_lower for word in action_words)
        
        if has_contextual_reference and has_action_word:
            print(f"[DEBUG] Found contextual edit reference")
            return 'image_edit'
        
        # NEW: Check for partial editing phrases (element + descriptor)
        image_elements = [
            'sky', 'mountain', 'mountains', 'tree', 'trees', 'water', 'ocean', 'sea',
            'sun', 'moon', 'cloud', 'clouds', 'grass', 'flower', 'flowers',
            'building', 'buildings', 'car', 'cars', 'person', 'people', 'animal',
            'background', 'foreground', 'color', 'colors', 'lighting', 'shadow',
            'bird', 'birds', 'dog', 'cat', 'house', 'road', 'path'
        ]
        
        color_descriptors = [
            'purple', 'blue', 'red', 'green', 'yellow', 'orange', 'pink', 'black',
            'white', 'gray', 'grey', 'brown', 'golden', 'silver', 'dark', 'light',
            'bright', 'vibrant', 'pale', 'deep', 'darker', 'brighter', 'lighter'
        ]
        
        size_descriptors = [
            'bigger', 'smaller', 'larger', 'taller', 'shorter', 'wider', 'narrower',
            'huge', 'tiny', 'massive', 'little', 'big', 'small', 'large'
        ]
        
        style_descriptors = [
            'realistic', 'cartoon', 'abstract', 'detailed', 'simple', 'blurry',
            'sharp', 'soft', 'hard', 'smooth', 'rough', 'shiny', 'matte'
        ]
        
        all_descriptors = color_descriptors + size_descriptors + style_descriptors
        
        # Check if we have an element + descriptor combination
        has_element = any(element in question_lower for element in image_elements)
        has_descriptor = any(desc in question_lower for desc in all_descriptors)
        
        if has_element and has_descriptor:
            print(f"[DEBUG] Found element+descriptor combination for editing")
            return 'image_edit'
        
        # Check for single descriptive words that likely mean editing
        single_edit_words = [
            'brighter', 'darker', 'bigger', 'smaller', 'lighter', 'sharper',
            'softer', 'warmer', 'cooler', 'more colorful', 'less colorful'
        ]
        
        if any(word in question_lower for word in single_edit_words):
            print(f"[DEBUG] Found single edit word")
            return 'image_edit'
        
        # FALLBACK: If question is very short (1-3 words) and contains descriptors,
        # assume it's an edit attempt with incomplete speech recognition
        words = question_lower.split()
        if len(words) <= 3 and (has_descriptor or has_element):
            print(f"[DEBUG] Short phrase with descriptor/element - likely incomplete edit command")
            return 'image_edit'

    # 2. Second Priority: New Image Generation
    generation_keywords = [
        'generate an image', 'create an image', 'make a picture', 'draw a picture', 
        'imagine a scene', 'generate a picture', 'create a picture', 'draw an image',
        'make me an image', 'create me a picture', 'generate me a picture'
    ]
    if any(keyword in question_lower for keyword in generation_keywords):
        return 'image_generate'

    # 3. Third Priority: Screen Analysis
    screen_keywords = [
        'screen', 'display', 'see', 'showing', 'visible', 'window', 'tab', 'browser',
        'page', 'website', 'photo', 'text', 'document', 'file',
        'app', 'application', 'program', 'menu', 'button', 'click', 'cursor',
        'what is this', 'what am i looking at', 'describe this', 'explain this',
        'read this', 'what does this say', 'translate this', 'summarize this'
    ]
    # Check for "this image" or "this picture" for screen analysis
    if any(keyword in question_lower for keyword in screen_keywords) or "this image" in question_lower or "this picture" in question_lower:
        return 'screen'

    # 4. Fourth Priority: Research
    research_keywords = [
        'research', 'find information on', 'deep dive into', 'look into', 'investigate'
    ]
    if any(keyword in question_lower for keyword in research_keywords):
        return 'research'

    # 5. Check for ambiguous cases that might be editing attempts
    if image_session_memory["current_image"] is not None:
        # If we have an image and the question is very short or contains descriptive words,
        # it might be an incomplete edit command
        words = question_lower.split()
        descriptive_words = [
            'purple', 'blue', 'red', 'green', 'yellow', 'bright', 'dark', 'light',
            'big', 'small', 'tall', 'short', 'wide', 'narrow', 'colorful', 'detailed'
        ]
        
        if (len(words) <= 2 or 
            any(word in question_lower for word in descriptive_words)):
            print(f"[DEBUG] Potentially ambiguous edit command detected")
            return 'ambiguous_edit'
    
    # 6. Default: General Knowledge
    return 'general'

# --- CORE FUNCTIONS ---
def open_file(filepath):
    """Opens a file using the default application for the current OS."""
    print(f"📂 Opening file: {filepath}")
    try:
        os.startfile(filepath)
    except Exception as e:
        print(f"[ERROR] Error opening file: {e}")

def generate_and_save_image(prompt, is_edit=False, reference_image_path=None):
    """Generates or edits an image using Gemini 2.5 Flash Image Preview (Nano Banana)."""
    global image_session_memory
    
    # Check for interrupt before starting
    if interrupt_flag.is_set():
        print("[INTERRUPT] Image generation cancelled by user")
        return None
    
    print("* Using Gemini 2.5 Flash Image Preview (Nano Banana)...")
    try:
        # Clean and enhance the prompt
        clean_prompt = extract_image_prompt_from_question(prompt)
        print(f"[DEBUG] Using prompt: {clean_prompt}")
        
        # Prepare the content for Gemini
        content_parts = []
        
        if is_edit and reference_image_path:
            # For image editing: include the existing image + edit prompt
            print(f"[DEBUG] Editing image: {reference_image_path}")
            
            # Load and encode the reference image
            with open(reference_image_path, "rb") as image_file:
                image_data = image_file.read()
            
            # Create image part for Gemini
            image_part = {
                'mime_type': 'image/png',
                'data': image_data
            }
            
            # Add image and editing instruction
            content_parts = [
                image_part,
                f"Edit this image: {clean_prompt}. Maintain the overall composition but make the requested changes."
            ]
        else:
            # For new image generation: text prompt only
            content_parts = [clean_prompt]
        
        # Generate/edit image using Gemini
        response = image_model.generate_content(content_parts)
        
        # Check for interrupt after generation
        if interrupt_flag.is_set():
            print("[INTERRUPT] Image generation cancelled after API call")
            return None
        
        # Extract image data from response
        if not response.parts:
            raise ValueError("No image data found in the response.")
        
        # Find the image part in the response
        image_data = None
        for part in response.parts:
            if hasattr(part, 'inline_data') and part.inline_data.mime_type.startswith('image/'):
                image_data = part.inline_data.data
                break
        
        if not image_data:
            raise ValueError("No image data found in response parts.")
        
        # Ensure the save directory exists
        os.makedirs(IMAGE_SAVE_PATH, exist_ok=True)
        
        # Create a descriptive filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in clean_prompt[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_prompt = "_".join(safe_prompt.split())
        
        action_prefix = "edited" if is_edit else "generated"
        filename = f"{action_prefix}_{safe_prompt}_{timestamp}.png"
        filepath = os.path.join(IMAGE_SAVE_PATH, filename)
        
        # Save the image (Gemini returns base64 encoded data)
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        # Update session memory
        image_info = {
            "filepath": filepath,
            "prompt": clean_prompt,
            "timestamp": timestamp,
            "is_edit": is_edit,
            "reference_image": reference_image_path,
            "model": "gemini-2.5-flash-image-preview"
        }
        
        image_session_memory["current_image"] = image_info
        image_session_memory["generated_images"].append(image_info)
        image_session_memory["conversation_context"].append({
            "type": "image_editing" if is_edit else "image_generation",
            "prompt": prompt,
            "result": filepath,
            "timestamp": timestamp
        })
        
        # Save metadata
        save_image_metadata(filepath, image_info)
        
        action_text = "edited" if is_edit else "generated"
        print(f"[OK] Image {action_text} and saved to {filepath}")
        return filepath
        
    except Exception as e:
        print(f"[ERROR] Error with Gemini image generation: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_image_prompt_from_question(question):
    """Extract the actual image generation prompt from the user's natural language question."""
    question_lower = question.lower()
    
    # Remove common trigger phrases to get to the actual prompt
    trigger_phrases = [
        'generate an image of ', 'create an image of ', 'make a picture of ',
        'draw a picture of ', 'imagine a scene of ', 'generate an image ',
        'create an image ', 'make a picture ', 'draw a picture ',
        'imagine a scene ', 'make me an image of ', 'create me an image of '
    ]
    
    clean_prompt = question
    for phrase in trigger_phrases:
        if phrase in question_lower:
            # Find where the phrase ends and extract everything after it
            idx = question_lower.find(phrase) + len(phrase)
            clean_prompt = question[idx:].strip()
            break
    
    # If no trigger phrase found, use the whole question
    if clean_prompt == question:
        # Still try to clean up common prefixes
        prefixes_to_remove = ['can you ', 'please ', 'i want ', 'i would like ']
        for prefix in prefixes_to_remove:
            if clean_prompt.lower().startswith(prefix):
                clean_prompt = clean_prompt[len(prefix):]
                break
    
    return clean_prompt.strip()

def save_image_metadata(filepath, image_info):
    """Save metadata alongside the generated image."""
    try:
        metadata_path = filepath.replace('.png', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(image_info, f, indent=2)
    except Exception as e:
        print(f"[WARNING] Could not save metadata: {e}")

def load_image_metadata(filepath):
    """Load metadata for a generated image."""
    try:
        metadata_path = filepath.replace('.png', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"[WARNING] Could not load metadata: {e}")
    return None

def capture_screen_in_memory():
    """Captures the entire screen and returns it as an in-memory image object."""
    print("📸 Capturing screen...")
    try:
        screenshot = ImageGrab.grab()
        img_byte_arr = io.BytesIO()
        screenshot.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        print("[OK] Screen captured.")
        return {'mime_type': 'image/png', 'data': img_byte_arr.getvalue()}
    except Exception as e:
        print(f"[ERROR] Error capturing screen: {e}")
        return None

def clean_text_for_tts(text):
    """Clean text for natural text-to-speech by removing markdown formatting and simplifying sources."""
    import re
    
    # Remove markdown headers (# ## ###)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    
    # Remove markdown bold/italic markers (* ** _)
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # Remove ** and *
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)    # Remove __ and _
    
    # Remove markdown links - keep just the text, not URLs
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove bullet points and list markers
    text = re.sub(r'^\s*[-*+]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    
    # Simplify the Sources section dramatically
    if "## Sources" in text:
        # Find the sources section
        sources_match = re.search(r'## Sources\s*\n(.*?)(?=\n##|\Z)', text, re.DOTALL)
        if sources_match:
            sources_text = sources_match.group(1)
            
            # Extract just the source names (before the first parenthesis or hyphen)
            source_names = []
            for line in sources_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('-'):
                    # Extract source name (everything before " (" or " -")
                    name_match = re.match(r'^-?\s*([^([\-]+)', line)
                    if name_match:
                        source_name = name_match.group(1).strip()
                        if source_name and len(source_name) > 3:  # Avoid very short matches
                            source_names.append(source_name)
            
            if source_names:
                # Create a simple, spoken version
                if len(source_names) == 1:
                    sources_replacement = f"This research was compiled from {source_names[0]}."
                elif len(source_names) == 2:
                    sources_replacement = f"This research was compiled from {source_names[0]} and {source_names[1]}."
                else:
                    sources_list = ", ".join(source_names[:-1]) + f", and {source_names[-1]}"
                    sources_replacement = f"This research was compiled from a variety of high-quality sources including {sources_list}."
            else:
                sources_replacement = "This research was compiled from multiple high-quality sources."
            
            # Replace the entire sources section
            text = re.sub(r'## Sources\s*\n(.*?)(?=\n##|\Z)', sources_replacement, text, flags=re.DOTALL)
    
    # Clean up extra whitespace and newlines
    text = re.sub(r'\n{3,}', '\n\n', text)  # Replace 3+ newlines with 2
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Remove leading spaces
    
    # Remove any remaining markdown artifacts
    text = re.sub(r'[`~]', '', text)  # Remove backticks and tildes
    text = re.sub(r'\|', '', text)    # Remove table pipes
    
    # Clean up confidence assessment language
    text = re.sub(r'Overall confidence is low \([0-9.]+.*?\)', 'Overall confidence is low', text)
    
    return text.strip()

def speak(text):
    """Converts text to speech and speaks it out loud - INTERRUPTIBLE VERSION."""
    global current_tts_engine, tts_lock
    
    print(f"[SPEAK] AI: {text}")
    
    if shutdown_flag.is_set() or interrupt_flag.is_set():
        print("[DEBUG] TTS: Shutdown/interrupt flag set, skipping speech")
        return
    
    # Clean text for natural speech
    clean_text = clean_text_for_tts(text)
    print(f"[DEBUG] TTS: Cleaned text for speech (removed {len(text) - len(clean_text)} characters)")
    
    # Use lock to ensure only one TTS operation at a time
    with tts_lock:
        try:
            print("[DEBUG] TTS: Creating engine...")
            engine = pyttsx3.init()
            current_tts_engine = engine
            
            print("[DEBUG] TTS: Getting voices...")
            voices = engine.getProperty('voices')
            print(f"[DEBUG] TTS: Found {len(voices)} voices")
            
            # Set voice
            if len(voices) > 2:
                print("[DEBUG] TTS: Setting voice to Zira...")
                engine.setProperty('voice', voices[2].id)
            
            # Set properties
            print("[DEBUG] TTS: Setting rate and volume...")
            engine.setProperty('rate', 200)
            engine.setProperty('volume', 1.0)
            
            print("[DEBUG] TTS: About to speak...")
            engine.say(clean_text)
            
            print("[DEBUG] TTS: Calling runAndWait()...")
            engine.runAndWait()  # Direct call - no threading
            
            print("[DEBUG] TTS: Speech completed normally")
            
            # Cleanup
            print("[DEBUG] TTS: Cleaning up engine...")
            engine.stop()
            current_tts_engine = None
            
            print("[OK] TTS: Speech operation completed successfully")
        
        except Exception as e:
            current_tts_engine = None
            print(f"[ERROR] Error in text-to-speech: {e}")
            print("[FALLBACK] TTS: Continuing without speech - application will proceed normally")
            # Don't print full traceback to avoid cluttering output
            print(f"[DEBUG] TTS Error details: {str(e)}")

def listen_for_command(max_retries=None):
    """Listens for a command with natural pause-based detection."""
    global recognizer, microphone
    
    # Use configured max_retries if not provided
    if max_retries is None:
        max_retries = SPEECH_MAX_RETRIES
    
    start_time = time.time()
    print(f"[TIME] [TIMING] listen_for_command() started at {datetime.now().strftime('%H:%M:%S.%f')}")
    print(f"[DEBUG] [CONFIG] Using timeout={SPEECH_TIMEOUT}s, pause_threshold={SPEECH_PAUSE_THRESHOLD}s, max_retries={max_retries}")
    
    # Use pre-initialized components or fallback to new instances
    if recognizer is None or microphone is None:
        print("[WARNING] [TIMING] Speech system not pre-initialized, creating new instances...")
        init_start = time.time()
        r = sr.Recognizer() if recognizer is None else recognizer
        mic = sr.Microphone() if microphone is None else microphone
        print(f"[TIME] [TIMING] Fallback initialization took {time.time() - init_start:.3f}s")
    else:
        print("[OK] [TIMING] Using pre-initialized speech components")
        r = recognizer
        mic = microphone
    
    for attempt in range(max_retries):
        attempt_start = time.time()
        print(f"[TIME] [TIMING] Attempt {attempt + 1} started at {datetime.now().strftime('%H:%M:%S.%f')}")
        
        if shutdown_flag.is_set():
            return None
            
        try:
            # Microphone initialization timing
            mic_init_start = time.time()
            with mic as source:
                print(f"[TIME] [TIMING] Microphone opened in {time.time() - mic_init_start:.3f}s")
                print(f"\n" + "="*60)
                print(f"[MIC] READY TO LISTEN - Speak when ready!")
                print(f"   I'll wait for you to finish (pause threshold: {SPEECH_PAUSE_THRESHOLD}s)")
                print(f"   Timeout: {SPEECH_TIMEOUT} seconds to start speaking")
                print(f"="*60)
                
                # Adjust recognizer settings for natural speech
                settings_start = time.time()
                r.pause_threshold = SPEECH_PAUSE_THRESHOLD  # Silence duration to stop recording
                r.phrase_threshold = SPEECH_PHRASE_THRESHOLD  # Minimum seconds of non-silent audio before phrase starts
                r.non_speaking_duration = SPEECH_NON_SPEAKING_DURATION  # Duration of non-speaking audio to keep after phrase ends
                
                # Log current recognizer settings for debugging
                print(f"[DEBUG] [CONFIG] Recognizer settings: pause={SPEECH_PAUSE_THRESHOLD}s, phrase={SPEECH_PHRASE_THRESHOLD}s, non_speaking={SPEECH_NON_SPEAKING_DURATION}s")
                
                # Ambient noise calibration with timing
                calibration_start = time.time()
                r.adjust_for_ambient_noise(source, duration=SPEECH_CALIBRATION_DURATION)
                calibration_duration = time.time() - calibration_start
                print(f"[DEBUG] [CALIBRATION] Ambient noise calibration took {calibration_duration:.3f}s")
                
                settings_duration = time.time() - settings_start
                print(f"[TIME] [TIMING] Audio settings adjusted in {settings_duration:.3f}s")
                
                try:
                    # Audio capture with progress monitoring
                    listen_start = time.time()
                    print("[WAIT] Waiting for speech...")
                    print(f"[DEBUG] [LISTEN] Starting r.listen() at {datetime.now().strftime('%H:%M:%S.%f')}")
                    
                    # Create a flag to stop progress monitoring when audio is captured
                    capture_complete = threading.Event()
                    audio_captured = threading.Event()
                    progress_thread = None
                    
                    def progress_monitor():
                        """Monitor and report progress during speech capture"""
                        last_update = time.time()
                        while not capture_complete.is_set():
                            current_time = time.time()
                            elapsed = current_time - listen_start
                            
                            # Show progress every 3 seconds
                            if current_time - last_update >= 3.0:
                                if not audio_captured.is_set():
                                    print(f"[WAIT] Still listening... ({elapsed:.1f}s elapsed)")
                                else:
                                    print(f"[PROCESSING] Processing speech... ({elapsed:.1f}s total)")
                                last_update = current_time
                            
                            time.sleep(0.5)  # Check every 500ms
                    
                    # Start progress monitoring in background
                    progress_thread = threading.Thread(target=progress_monitor, daemon=True)
                    progress_thread.start()
                    
                    try:
                        # The actual speech capture
                        audio = r.listen(source, timeout=SPEECH_TIMEOUT)
                        audio_captured.set()  # Mark that we have audio data
                        
                        listen_duration = time.time() - listen_start
                        capture_complete.set()  # Stop progress monitoring
                        
                        print(f"[OK] Speech captured! Duration: {listen_duration:.3f}s")
                        print(f"[TIME] [TIMING] Audio captured in {listen_duration:.3f}s")
                        print(f"[DEBUG] [LISTEN] r.listen() completed at {datetime.now().strftime('%H:%M:%S.%f')}")
                        
                    except Exception as listen_error:
                        capture_complete.set()  # Stop progress monitoring on error
                        raise listen_error
                except sr.WaitTimeoutError:
                    timeout_duration = time.time() - attempt_start
                    print(f"⏱️ No speech detected after {timeout_duration:.3f}s. Listening again...")
                    continue
                
                if shutdown_flag.is_set():
                    return None
                
                print("🔄 Processing what you said...")
                # Google API timing with timeout and retry
                api_start = time.time()
                
                # Try Google API with timeout handling
                api_success = False
                for api_attempt in range(2):  # Try twice
                    try:
                        print(f"🌐 [API] Attempting Google Speech API (attempt {api_attempt + 1}/2)")
                        command = r.recognize_google(audio, show_all=False)
                        api_duration = time.time() - api_start
                        print(f"[TIME] [TIMING] Google Speech API responded in {api_duration:.3f}s")
                        print(f"👤 You said: {command}")
                        api_success = True
                        break
                    except sr.RequestError as e:
                        api_duration = time.time() - api_start
                        print(f"[ERROR] [API] Google Speech API error (attempt {api_attempt + 1}/2) after {api_duration:.3f}s: {e}")
                        if api_attempt == 0:  # Only retry once
                            print("🔄 [API] Retrying Google Speech API...")
                            time.sleep(0.5)  # Brief delay before retry
                        continue
                    except Exception as e:
                        api_duration = time.time() - api_start
                        print(f"[ERROR] [API] Unexpected API error after {api_duration:.3f}s: {e}")
                        break
                
                if not api_success:
                    print("[ERROR] [API] Failed to get response from Google Speech API after retries")
                    continue
                
                # Add explicit delay to ensure microphone releases
                cleanup_start = time.time()
                print("[DEBUG] Mic: Releasing microphone...")
                time.sleep(0.5)
                print(f"[TIME] [TIMING] Microphone cleanup took {time.time() - cleanup_start:.3f}s")
                
                total_duration = time.time() - start_time
                print(f"[TIME] [TIMING] Total listen_for_command() duration: {total_duration:.3f}s")
                
                return command
                
        except sr.UnknownValueError:
            print("🤔 Sorry, I didn't catch that clearly.")
            if attempt < max_retries - 1:
                print("🔄 Please try again - take your time!")
            continue
        except sr.RequestError as e:
            print(f"[ERROR] Speech service error: {e}")
            if attempt < max_retries - 1:
                print("🔄 Retrying...")
                time.sleep(1)
            continue
        except Exception as e:
            print(f"[ERROR] Unexpected error in speech recognition: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
    
    print("[WARNING] Could not understand command after multiple attempts.")
    return None

def ask_gemini_with_vision(prompt, image):
    """Sends the user's prompt and a screenshot to the Gemini model."""
    if not image:
        return "I couldn't capture the screen, so I can't answer your question."
        
    print("🧠 Analyzing screen and thinking...")
    try:
        response = understanding_model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        print(f"[ERROR] Error with Gemini Vision API: {e}")
        return "Sorry, I encountered an error while analyzing the screen."

def ask_gemini_general(prompt):
    """Sends a text-only prompt to the Gemini model for general questions."""
    print("🧠 Thinking about your question...")
    try:
        response = understanding_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[ERROR] Error with Gemini API: {e}")
        return "Sorry, I encountered an error processing your question."

# --- MAIN LOOP ---
async def main():
    """The main function with intelligent routing."""
    print("Guardian AI Starting...")
    print("[!] * NEW: MODE SELECTION SYSTEM - Choose your interaction mode!")
    print("[!] * Image Mode: Generate and edit images with Nano Banana")
    print("[!] * Screen Mode: Analyze what's on your screen")
    print("[!] * Research Mode: Deep web research with multi-agent system")
    print("[!] * General Mode: Questions from my knowledge base")
    print("[!] * Auto Mode: Automatically detect your intent (classic behavior)")
    print("[!] * DIRECTOR MODE: Press CTRL+I anytime to interrupt and quickly change modes!")
    print("[!] Say 'exit' or 'goodbye' to quit, or press Ctrl+C anytime")
    print()
    
# Pre-initialize speech recognition system
    speech_ready = initialize_speech_system()
    if not speech_ready:
        print("[!] Speech recognition may have degraded performance")
    
    # Start the global hotkey listener
    start_hotkey_listener()
    
    # Start welcome message in background (non-blocking)
    welcome_msg = "Hello! I'm your Guardian AI with a new mode selection system. You can now choose specific modes like Image Mode for Nano Banana, Screen Analysis, Research, or General Knowledge. Just say the number or mode name to switch. Press CTRL+I anytime to interrupt me and quickly change modes!"
    welcome_thread = threading.Thread(target=lambda: speak(welcome_msg), daemon=True)
    welcome_thread.start()
    
    consecutive_failures = 0
    max_consecutive_failures = 3
    
    while not shutdown_flag.is_set():
        try:
            # Check for interrupt at the start of each loop
            if interrupt_flag.is_set():
                print("\n[!] [INTERRUPT] Handling interrupt...")
                display_quick_mode_menu()
                
                # Listen for quick mode selection
                mode_choice = listen_for_command(max_retries=1)  # Single attempt for quick response
                
                if mode_choice:
                    mode_selection = parse_mode_selection(mode_choice)
                    if mode_selection:
                        set_mode(mode_selection)
                        speak(f"Switched to {current_mode['display_name']}. Ready!")
                    else:
                        print("[INTERRUPT] No valid mode selected, continuing...")
                
                # Clear the interrupt flag and continue
                interrupt_flag.clear()
                continue
            
            # 1. Display mode selection menu
            display_mode_selection_menu()
            
            # 2. Listen for user input
            user_question = listen_for_command()
            
            # Check for interrupt after listening
            if interrupt_flag.is_set():
                continue  # Will be handled at the top of the loop
            
            if user_question is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    print("[WARNING] Having trouble hearing you. Let me reset...")
                    speak("I'm having trouble hearing you clearly. Let me try again.")
                    consecutive_failures = 0
                continue
            
            # Reset failure counter on successful recognition
            consecutive_failures = 0
            
            # Check for exit commands
            exit_phrases = ["exit", "goodbye", "quit", "stop"]
            if any(phrase in user_question.lower() for phrase in exit_phrases):
                speak("Goodbye!")
                break
            
            # 3. Check if user is trying to change modes
            mode_selection = parse_mode_selection(user_question)
            if mode_selection:
                set_mode(mode_selection)
                speak(f"Switched to {current_mode['display_name']}. What would you like to do?")
                continue
            
            # 4. Get intent based on current mode or use intelligent detection
            mode_intent = get_mode_specific_intent(user_question)
            if mode_intent:
                intent = mode_intent
                print(f"* Mode-based intent: {intent.upper()} (in {current_mode['display_name']})")
            else:
                intent = analyze_question_intent(user_question)
                print(f"* Auto-detected intent: {intent.upper()}")
            
            # 5. Route to appropriate handler
            answer = None  # Initialize answer
            
            # Check for interrupt before processing
            if interrupt_flag.is_set():
                continue  # Will be handled at the top of the loop
            
            if intent == 'image_generate':
                print("* GENERATING IMAGE...")
                if current_mode["mode"] == "image":
                    print(f"* In Image Mode - Processing: '{user_question}'")
                
                generated_image_path = generate_and_save_image(user_question)
                if generated_image_path:
                    open_file(generated_image_path)
                    if current_mode["mode"] == "image":
                        answer = "✅ Image generated! Since you're in Image Mode, you can now say things like 'make it purple', 'add birds', or 'brighter' to edit it."
                    else:
                        answer = "I've created that image for you using Gemini Nano Banana and am opening it now. You can ask me to edit it with natural language like 'make it brighter' or 'add a sunset'!"
                else:
                    answer = "I'm sorry, I wasn't able to create the image."
            
            elif intent == 'image_edit':
                print("✏️ EDITING IMAGE...")
                if current_mode["mode"] == "image":
                    print(f"* In Image Mode - Editing with: '{user_question}'")
                
                current_image = image_session_memory["current_image"]
                if current_image:
                    print(f"[DEBUG] Editing image: {current_image['filepath']}")
                    edited_image_path = generate_and_save_image(user_question, is_edit=True, reference_image_path=current_image['filepath'])
                    if edited_image_path:
                        open_file(edited_image_path)
                        if current_mode["mode"] == "image":
                            answer = "✅ Image edited! You can continue making changes by saying things like 'darker', 'add trees', or 'make it cartoon style'."
                        else:
                            answer = "I've edited your image using Gemini Nano Banana based on your request. How does it look? Feel free to ask for more changes!"
                    else:
                        answer = "I'm sorry, I wasn't able to edit the image."
                else:
                    if current_mode["mode"] == "image":
                        answer = "No image to edit! Say something like 'mountain sunset' to generate a new image first."
                    else:
                        answer = "I don't see any image to edit. Please generate an image first!"
            
            elif intent == 'ambiguous_edit':
                print("❓ Ambiguous command - asking for clarification...")
                current_image = image_session_memory["current_image"]
                if current_image:
                    # Treat it as an edit attempt with the assumption it's incomplete
                    print(f"[DEBUG] Treating ambiguous command as edit: {user_question}")
                    # Enhance the prompt to make it a proper edit command
                    enhanced_prompt = f"Make the {user_question}"
                    edited_image_path = generate_and_save_image(enhanced_prompt, is_edit=True, reference_image_path=current_image['filepath'])
                    if edited_image_path:
                        open_file(edited_image_path)
                        answer = f"I interpreted '{user_question}' as an edit command and made that change to your image. If this isn't what you wanted, please be more specific next time, like 'make the sky purple' or 'change it to be brighter'."
                    else:
                        answer = "I think you wanted to edit the image, but I couldn't make that change. Try being more specific, like 'make the sky purple' or 'make it brighter'."
                else:
                    answer = "I don't see any image to edit. Please generate an image first!"

            elif intent == 'screen':
                print("* Using screen analysis mode...")
                screenshot = capture_screen_in_memory()
                answer = ask_gemini_with_vision(user_question, screenshot)
            
            elif intent == 'research':
                print("* Using research mode...")
                speak("Starting deep research for you. This might take a moment.")
                
                # Extract the actual research query
                research_keywords = [
                    'research', 'find information on', 'deep dive into', 'look into', 'investigate'
                ]
                extracted_query = user_question
                for keyword in research_keywords:
                    if keyword in user_question.lower():
                        extracted_query = user_question.lower().split(keyword, 1)[1].strip()
                        break
                
                if not extracted_query:
                    speak("I couldn't understand what you want me to research. Please try again.")
                    continue

                # Execute research using the Orchestrator
                research_report = await orchestrator.execute_research(extracted_query)
                answer = f"Here is the research report: {research_report}"

            else:  # general
                print("* Using general knowledge mode...")
                answer = ask_gemini_general(user_question)
            
            # 4. Speak the answer (if not interrupted)
            if interrupt_flag.is_set():
                print("[INTERRUPT] Skipping TTS due to interrupt")
                continue  # Will be handled at the top of the loop
            
            if answer and not shutdown_flag.is_set():
                speak(answer)
            elif not answer:
                speak("I'm sorry, I couldn't generate a response to your question.")
        
        except KeyboardInterrupt:
            print(f"[STOP] KeyboardInterrupt caught in main loop")
            break
        except Exception as e:
            print(f"[ERROR] Unexpected error in main loop: {e}")
            import traceback
            traceback.print_exc()
            speak("I encountered an error, but I'll keep trying to help you.")
            time.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n[STOP] Program interrupted by user.")
    finally:
        print("[CLEANUP] Shutting down hotkey listener...")
        stop_hotkey_listener()
        print("Guardian AI shut down complete.")