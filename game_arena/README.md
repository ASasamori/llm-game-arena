# 🎭 Mafia AI Simulation

A fully automated, AI-driven simulation of the social deduction game **Mafia**.

Powered by Large Language Models (**Google Gemini** and **OpenRouter**), ten distinct AI personalities gather to deceive, deduce, and survive. As a spectator, you have visibility into their secret discussions, inner thoughts, and hidden actions.

## 🚀 Highlights

- **Granular Real-Time Visibility**: Watch the game step-by-step. See Mafia chat turns, individual votes, and role actions in real-time.
- **Inner Monologues**: Each AI speaker has a "Hidden Thought" dropdown, revealing their true strategy and suspicions versus what they say publicly.
- **Spectator Mode**: Special styling for sensitive info. Secret actions are marked `SPECTATOR ONLY` so you can follow the logic without it leaking to the players.
- **Resilient Engine**: Built-in 429 rate-limit handling with automatic retry countdowns and UI feedback.
- **Flexible AI**: Easily swap between Gemini (Pro/Flash) and OpenRouter models for each individual player.

## 🎮 Game Mechanics

### 1. Setup
- **10 Players** are initialized with unique names and avatars from `player_config.json`(Currently all set to gemma-3-27b-it)
- **Roles Assigned**:
    - **3 Mafia**: Wake up each night to coordinate a kill.
    - **1 Doctor**: Protects one player from elimination each night.
    - **1 Sheriff**: Investigates one player each night to determine their alignment.
    - **5 Villagers**: Use social deduction to find and lynch the Mafia.

### 2. Night Phase 🌙 (Granular Turns)
- **Mafia Deliberation**: Mafia members chat sequentially. You see their private strategies one-by-one.
- **Role Actions**: Once chat concludes, the Doctor protects and the Sheriff investigates.
- **Resolution**: The engine calculates if the Mafia's target was saved or eliminated.

### 3. Day Phase ☀️
- **Morning Report**: Night events are announced to the whole town.
- **Discussion**: Players rotate sequentially to speak. They analyze the graveyard, role reveals, and past behavior.
- **Voting**: Every player casts a vote individually. A majority (>50% of alive players) is required to eliminate. Tiles are handled with a "Skip" result.

## 🏃 Getting Started

### Prerequisites
- Python 3.10+
- [Conda](https://docs.conda.io/en/latest/) (Recommended) or `venv`

### Installation
1. **Activate Environment**:
   ```bash
   conda activate arena  # Or your preferred env
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_key
   OPENROUTER_API_KEY=your_openrouter_key
   ```

### Running the App
1. **Start the FastAPI server**:
   ```bash
   uvicorn app.main:app --reload --port 8094
   ```
2. **Open the UI**:
   Navigate to `http://localhost:8094`

## 🕹️ Controls & UI

- **Next Turn**: Manually advance the simulation. The button dynamically displays *who* is currently thinking or acting.
- **Auto-Play**: Automatically advances the game.
- **Speed Slider**: Controls the delay between turns in Auto-Play (1s to 5s).
- **Transcript**: A full log of every event. 
    - **Red Messages**: Mafia secret actions.
    - **Blue/Yellow Messages**: Doctor/Sheriff secret actions.
    - **Dashed Borders**: Spectator-only information.
- **Keyboard Shortcuts**:
    - `Space`: Trigger Next Turn.
    - `A`: Toggle Auto-Play.

## 🛠️ Advanced Configuration

### Player Personas
Edit `player_config.json` to change:
- **Names & Avatars**: Give players unique identities.
- **Model Assignment**: Assign different models (e.g., `gemini-1.5-flash` for villagers, `gemini-1.5-pro` for the boss) to optimize performance or cost.

## 🧪 Testing
Unit tests for the core engine logic (win conditions, phase transitions, role resolution):
```bash
python -m pytest tests/test_engine.py
```

---
