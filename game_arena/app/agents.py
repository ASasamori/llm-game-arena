"""
AI Agent logic - prompts and behavior for different roles.
"""
from app.models import Player, Role, GameState, AIResponse, PlayerStatus
from app.llm_client import get_llm_response, parse_ai_response


def get_system_prompt(player: Player, game_state: GameState, is_private: bool = False) -> str:
    """
    Generate a role-specific system prompt for a player.
    is_private: If True, the prompt is for a secret channel (Mafia chat, Monologue, etc.)
    """
    base_prompt = f"""You are playing a competitive social deduction game called Mafia. Your name is {player.name}.
    
    Current Setting: A tense village where Mafia members are hiding in plain sight.
    Your Goal: Win the game for your team.
    
    CRITICAL INSTRUCTIONS:
    1. This is a GAME. Do not talk about "spending time together" or casual daily activities. Focus on determining who is Mafia or deceiving others.
    2. Your "public_speech" is what other players hear (or your teammates in private channels). Make it count.
    3. Your "inner_monologue" is your private strategy. Use it to analyze the game.
    
    You MUST respond ONLY with valid JSON in this exact format:
    {{
        "inner_monologue": "Your private strategic thoughts. Be analytical.",
        "public_speech": "What you say out loud. Be persuasive."
    }}
    
    Do NOT include any text outside of this JSON structure. Do NOT use markdown code blocks."""

    if player.role == Role.MAFIA:
        # Categorize teammates correctly
        alive_mafia = [p.name for p in game_state.get_alive_mafia() if p.name != player.name]
        dead_mafia = [p.name for p in game_state.get_mafia_players() if p.status == PlayerStatus.DEAD]
        
        teammates_str = ", ".join(alive_mafia) if alive_mafia else "None (you are the last one alive)"
        dead_teammates_str = ", ".join(dead_mafia) if dead_mafia else "None"

        if is_private:
            return base_prompt + f"""
            
YOUR ROLE: MAFIA (Private Strategy)
ALIVE Teammates: {teammates_str}
DEAD/ELIMINATED Teammates: {dead_teammates_str}

This is your SECRET channel. You do NOT need to hide your identity here.
Your goal is to coordinate with your alive teammates to eliminate the Town efficiently.
- Discuss who is the biggest threat to you and should be killed.
- Coordinate your 'innocent' stories for tomorrow.
- Be cold, calculating, and strategic. Do NOT worry about "accidental kills" – every non-mafia death is a step toward victory."""
        else:
            return base_prompt + f"""

YOUR ROLE: MAFIA (Public Deception)
Alive Teammates: {teammates_str}

You are in the public square. You MUST act like an innocent Villager.
- Blend in perfectly.
- Cast suspicion on Town members subtly.
- Defend your mafia teammates without being obvious.
- Never directly reveal you are Mafia."""

    elif player.role == Role.DOCTOR:
        return base_prompt + """

YOUR ROLE: DOCTOR (Town)
You have a special ability: each night you can choose one person to protect from the Mafia's kill.

Your goal is to help Town identify and eliminate all Mafia members while keeping key players alive.
During discussions, try to read people and identify suspicious behavior.

Your inner_monologue should show your analysis and protection strategy.
Your public_speech should contribute to finding the Mafia."""

    elif player.role == Role.SHERIFF:
        return base_prompt + """

YOUR ROLE: SHERIFF (Town)
You have a special ability: each night you can investigate one person to learn if they are Mafia or Town.

Your goal is to use your investigation results to help Town eliminate the Mafia.
Be careful about revealing your role - the Mafia will want to kill you.

Your inner_monologue should show your deductions and strategy.
Your public_speech should guide Town without making yourself an obvious target."""

    else:  # Villager
        return base_prompt + """

YOUR ROLE: VILLAGER (Town)
You are an ordinary villager with no special abilities.

Your goal is to help Town identify and eliminate all Mafia members through discussion and voting.
Pay attention to what others say, look for inconsistencies, and use logic to find the Mafia.

Your inner_monologue should show your suspicions and reasoning.
Your public_speech should contribute to the town discussion."""


def build_context_message(player: Player, game_state: GameState) -> str:
    """
    Build the context message containing game state and transcript for an AI.
    """
    # Build the public transcript (without inner monologues or secrets)
    # Tag dead speakers to prevent hallucinations
    dead_set = set(game_state.graveyard)
    
    transcript_entries = []
    has_player_spoken = False
    
    for entry in game_state.transcript[-20:]:  # Last 20 messages for context
        if entry.is_secret:
            continue
        
        prefix = "[SYSTEM]" if entry.is_system else (entry.speaker + (" [DEAD]" if entry.speaker in dead_set else ""))
        
        if entry.is_system:
            transcript_entries.append(f"{prefix} {entry.content}")
        else:
            transcript_entries.append(f"{prefix}: {entry.content}")
            has_player_spoken = True

    transcript_text = "\n".join(transcript_entries)
    
    if not has_player_spoken:
        transcript_text += "\n\n(CRITICAL: No players have spoken yet in this game. The messages above are only system announcements. Do NOT imagine or reference any prior behavior or messages from other players.)"

    # Add private memory if any
    private_info = ""
    if player.private_memory:
        private_info = "\n\nYOUR PRIVATE KNOWLEDGE (only you know this):\n"
        private_info += "\n".join(f"- {m}" for m in player.private_memory)

    # List alive players
    alive_players = [p.name for p in game_state.get_alive_players()]
    dead_players = game_state.graveyard

    context = f"""=== GAME STATUS ===
Day: {game_state.day_count}
Phase: {game_state.phase.value}
ALIVE Players (You can only interact with these): {', '.join(alive_players)}
DEAD/ELIMINATED Players (Ignore them): {', '.join(dead_players) if dead_players else 'None yet'}
Discussion Turns Remaining (how many more people can speak before it is time to vote): {game_state.discussion_turns_remaining}, CRITICAL: Make sure to wrap up conversations before it is time to vote !!!!
{private_info}

=== RECENT DISCUSSION ===
{transcript_text if transcript_text else '(No discussion yet)'}

=== YOUR TURN ===
It's your turn. Share your thoughts. Respond ONLY with JSON."""

    return context


async def get_player_speech(player: Player, game_state: GameState) -> AIResponse:
    """
    Get a player's speech for the day discussion phase.
    """
    system_prompt = get_system_prompt(player, game_state)
    context = build_context_message(player, game_state)

    response_text = await get_llm_response(
        model_id=player.model_id,
        system_prompt=system_prompt,
        user_message=context,
        temperature=0.8
    )

    parsed = parse_ai_response(response_text)

    return AIResponse(
        inner_monologue=parsed["inner_monologue"],
        public_speech=parsed["public_speech"]
    )


async def get_player_vote(player: Player, game_state: GameState) -> str:
    """
    Get a player's vote during the voting phase.
    Returns the name of the player they want to eliminate, or "skip".
    """
    system_prompt = get_system_prompt(player, game_state)

    alive_names = [p.name for p in game_state.get_alive_players() if p.name != player.name]

    vote_context = build_context_message(player, game_state) + f"""

=== VOTING TIME ===
You must now vote to eliminate someone. Based on the discussion, choose who you think is Mafia.
You can vote for: {', '.join(alive_names)}
Or vote "skip" if you don't want to eliminate anyone.

Respond with JSON:
{{
    "inner_monologue": "Your reasoning for your vote",
    "public_speech": "Just the name of who you're voting for (or 'skip')"
}}"""

    response_text = await get_llm_response(
        model_id=player.model_id,
        system_prompt=system_prompt,
        user_message=vote_context,
        temperature=0.5
    )

    parsed = parse_ai_response(response_text)
    vote = parsed["public_speech"].strip().lower()

    # Try to find a valid name in the response
    for name in alive_names:
        if name.lower() in vote:
            return name

    if "skip" in vote:
        return "skip"

    # Default to skip if we can't parse the vote
    return "skip"


async def get_mafia_discussion_turn(player: Player, game_state: GameState, discussion_history: list[str]) -> str:
    """
    Get a single turn of Mafia private discussion.
    """
    town_players = [p.name for p in game_state.get_alive_town()]
    
    discussion_text = "\n".join(discussion_history) if discussion_history else "(No discussion yet)"
    
    # If it's the very first night, players haven't spoken yet.
    is_first_night = game_state.day_count <= 1
    
    special_instruction = ""
    if is_first_night:
        special_instruction = """
    CRITICAL NIGHT 1 INSTRUCTION:
    This is the FIRST NIGHT. No one has spoken yet. No one has behaved suspiciously.
    Do NOT invent or hallucinate reasons like "X has been quiet" or "Y is aggressive".
    Do NOT use "silence" or "lack of speech" as a reason, as EVERYONE is silent right now.
    Base your choice on random selection, intuition, or general strategy (e.g. "Let's hit a random target").
    Admit that you have no info yet."""

    night_context = f"""=== NIGHT PHASE - MAFIA PRIVATE CHAT ===
It is night. You are plotting with your teammates.
{special_instruction}

Rounds of Mafia Discussion remaining: {game_state.mafia_chat_rounds_remaining}
Alive Town Members: {', '.join(town_players)}

=== DISCUSSION SO FAR ===
{discussion_text}

=== YOUR TURN ===
Discuss who to kill. Propose targets or agree with teammates.
Do NOT make the final decision yet, just discuss.

Respond with JSON:
{{
    "inner_monologue": "Your private thoughts",
    "public_speech": "What you say to your mafia teammates (e.g. 'I think we should take out X')"
}}"""

    response_text = await get_llm_response(
        model_id=player.model_id,
        system_prompt=get_system_prompt(player, game_state, is_private=True),
        user_message=night_context,
        temperature=0.7
    )
    
    parsed = parse_ai_response(response_text)
    return parsed["public_speech"]


async def get_mafia_kill_target(mafia_player: Player, game_state: GameState, discussion_history: list[str] = None) -> str:
    """
    Get the Mafia's kill target during night phase.
    """
    town_players = [p.name for p in game_state.get_alive_town()]
    discussion_text = "\n".join(discussion_history) if discussion_history else "(No prior discussion)"

    night_context = f"""=== NIGHT PHASE - MAFIA FINAL DECISION ===
It is night. The Town is asleep.

Available targets (Town members): {', '.join(town_players)}

=== TEAM DISCUSSION ===
{discussion_text}

=== DECISION TIME ===
Based on the discussion above, you must now choose the final target.

Respond with JSON:
{{
    "inner_monologue": "Reasoning based on team discussion",
    "public_speech": "Just the name of your target"
}}"""

    response_text = await get_llm_response(
        model_id=mafia_player.model_id,
        system_prompt=get_system_prompt(mafia_player, game_state, is_private=True),
        user_message=night_context,
        temperature=0.6
    )

    parsed = parse_ai_response(response_text)
    target = parsed["public_speech"].strip()

    # Validate target
    for name in town_players:
        if name.lower() in target.lower():
            return name

    # Default to first town player if we can't parse
    return town_players[0] if town_players else ""


async def get_doctor_save_target(doctor: Player, game_state: GameState) -> str:
    """
    Get the Doctor's save target during night phase.
    """
    alive_names = [p.name for p in game_state.get_alive_players()]

    night_context = f"""=== NIGHT PHASE - DOCTOR'S CHOICE ===
It is night. You must choose someone to protect from the Mafia tonight.

You can protect: {', '.join(alive_names)}
(You can protect yourself too)

Consider: Who is the Mafia most likely to target?

Respond with JSON:
{{
    "inner_monologue": "Your reasoning for who to save",
    "public_speech": "Just the name of who you want to save"
}}"""

    response_text = await get_llm_response(
        model_id=doctor.model_id,
        system_prompt=get_system_prompt(doctor, game_state),
        user_message=night_context,
        temperature=0.5
    )

    parsed = parse_ai_response(response_text)
    
    # Return tuple of (target_name, inner_monologue) or we can just parse here.
    # But checking engine.py, it expects just a name string. 
    # Valid point. I need to change the return type or handle it in engine.py.
    # Let's return a dict or tuple. engine.py calls via asyncio.gather.
    
    target_name = parsed["public_speech"].strip()
    
    # Validate
    final_target = doctor.name # Default
    if target_name:
        for name in alive_names:
            if name.lower() in target_name.lower():
                final_target = name
                break
                
    return {"target": final_target, "monologue": parsed["inner_monologue"]}


async def get_sheriff_investigate_target(sheriff: Player, game_state: GameState) -> str:
    """
    Get the Sheriff's investigation target during night phase.
    """
    # Can't investigate self or dead players
    targets = [p.name for p in game_state.get_alive_players() if p.name != sheriff.name]

    night_context = f"""=== NIGHT PHASE - SHERIFF'S INVESTIGATION ===
It is night. You can investigate one person to learn if they are Mafia or Town.

You can investigate: {', '.join(targets)}

Consider: Who has been most suspicious in the discussions?

Respond with JSON:
{{
    "inner_monologue": "Your reasoning for who to investigate",
    "public_speech": "Just the name of who you want to investigate"
}}"""

    response_text = await get_llm_response(
        model_id=sheriff.model_id,
        system_prompt=get_system_prompt(sheriff, game_state),
        user_message=night_context,
        temperature=0.5
    )

    parsed = parse_ai_response(response_text)
    target_name = parsed["public_speech"].strip()
    
    final_target = targets[0] if targets else ""
    if target_name:
        for name in targets:
            if name.lower() in target_name.lower():
                final_target = name
                break
                
    return {"target": final_target, "monologue": parsed["inner_monologue"]}
