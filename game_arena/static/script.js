/**
 * Mafia AI Simulation - Frontend Logic
 */

// DOM Elements
const welcomeScreen = document.getElementById('welcome-screen');
const speakerView = document.getElementById('speaker-view');
const nightView = document.getElementById('night-view');
const gameoverView = document.getElementById('gameover-view');

const startGameBtn = document.getElementById('start-game-btn');
const nextTurnBtn = document.getElementById('next-turn-btn');
const restartBtn = document.getElementById('restart-btn');
const autoplayToggle = document.getElementById('autoplay-toggle');
const speedSlider = document.getElementById('speed-slider');

const phaseValue = document.getElementById('phase-value');
const dayValue = document.getElementById('day-value');

const speakerAvatar = document.getElementById('speaker-avatar');
const speakerName = document.getElementById('speaker-name');
const speakerRole = document.getElementById('speaker-role');
const thoughtText = document.getElementById('thought-text');
const speechText = document.getElementById('speech-text');

const nightMessage = document.getElementById('night-message');
const gameoverTitle = document.getElementById('gameover-title');
const gameoverSubtitle = document.getElementById('gameover-subtitle');

const transcriptMessages = document.getElementById('transcript-messages');
const playerRoster = document.getElementById('player-roster');

// State
let isAutoPlaying = false;
let autoPlayInterval = null;
let currentPlayers = [];
let isProcessing = false;
let isRateLimited = false;
let nextSpeaker = null; // Track who is next

// API Functions
async function startGame() {
  try {
    const response = await fetch('/api/game/start', { method: 'POST' });
    const data = await response.json();

    if (data.status === 'started') {
      currentPlayers = data.players;
      nextSpeaker = data.next_speaker; // Store next speaker
      updatePlayerRoster(data.players);
      updatePhaseDisplay(data.phase, data.day);

      // Clear transcript
      transcriptMessages.innerHTML = '';

      // Show night view initially
      hideAllViews();
      nightView.classList.remove('hidden');
      nightMessage.textContent = 'The game begins... Night falls on the village.';

      // Enable next turn button
      nextTurnBtn.disabled = false;
    }
  } catch (error) {
    console.error('Failed to start game:', error);
  }
}

async function nextTurn() {
  if (isProcessing) return;
  isProcessing = true;
  nextTurnBtn.disabled = true;

  const originalText = "Next Turn";

  // Determine thinking message based on current UI phase and potential sub-turn
  const currentPhase = phaseValue.textContent.toLowerCase();
  let thinkingMsg = "Thinking...";

  if (currentPhase.includes('night')) {
    thinkingMsg = nextSpeaker ? `${nextSpeaker} is plotting...` : "Mafia & roles are acting...";
  } else if (currentPhase.includes('voting')) {
    thinkingMsg = nextSpeaker ? `${nextSpeaker} is voting...` : "The village is voting...";
  } else if (nextSpeaker) {
    thinkingMsg = `${nextSpeaker} is thinking...`;
  }

  nextTurnBtn.innerHTML = `<span class="spinner">⏳</span> ${thinkingMsg}`;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 120000); // 120s timeout

  try {
    const response = await fetch('/api/game/next', {
      method: 'POST',
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (response.status === 404) {
      console.warn("Game state not found (404). Stopping auto-play.");
      stopAutoPlay();
      nextTurnBtn.innerHTML = "⚠️ Game Not Started";
      nextTurnBtn.disabled = true;
      return;
    }

    const data = await response.json();

    if (data.status === 'error' && data.error === 'rate_limit') {
      isRateLimited = true;
      let retrySeconds = data.retry_after || 5;

      addTranscriptEntry({
        is_system: true,
        content: `⏳ Gemini is busy. Auto-retry in ${retrySeconds}s...`
      });

      console.warn(`Rate limit hit! Retrying in ${retrySeconds}s...`);
      nextTurnBtn.innerHTML = `⏳ Retrying in ${retrySeconds}s`;

      const countdown = setInterval(() => {
        retrySeconds--;
        if (retrySeconds < 0) {
          clearInterval(countdown);
          isRateLimited = false;
          isProcessing = false;
          nextTurnBtn.disabled = false;
          nextTurn();
        } else {
          nextTurnBtn.innerHTML = `⏳ Retrying in ${retrySeconds}s`;
        }
      }, 1000);
      return;
    }

    // Update the global next speaker for the next turn
    nextSpeaker = data.next_speaker || null;

    handleTurnResult(data);
    nextTurnBtn.innerHTML = originalText;
  } catch (error) {
    clearTimeout(timeoutId);
    console.error('Failed to advance turn:', error);
    nextTurnBtn.innerHTML = originalText;
    // Force a state refresh on error to unblock UI
    refreshState();
  } finally {
    if (!isRateLimited) {
      isProcessing = false;
      nextTurnBtn.disabled = false;
    }
  }
}

async function refreshState() {
  try {
    const response = await fetch('/api/game/state');
    const data = await response.json();

    currentPlayers = data.players;
    nextSpeaker = data.next_speaker; // Keep next speaker in sync
    updatePlayerRoster(data.players);
    updatePhaseDisplay(data.phase, data.day);
    rebuildTranscript(data.transcript);
  } catch (error) {
    console.error('Failed to refresh state:', error);
  }
}

// UI Update Functions
function handleTurnResult(data) {
  updatePhaseDisplay(data.phase, data.day);

  if (data.phase === 'game_over') {
    showGameOver(data.winner);
    stopAutoPlay();
    return;
  }

  // Handle Turn Types
  if (data.turn_type === 'mafia_chat') {
    addTranscriptEntry({
      is_system: false,
      speaker: data.turn.speaker,
      content: data.turn.content,
      is_secret: true // It's secret mafia chat
    });
    return;
  }

  if (data.turn_type === 'vote_turn') {
    refreshState();
    return;
  }

  if (data.turn_type === 'discussion_turn') {
    showSpeakerTurn(data.turn);
    addTranscriptEntry({
      is_system: false,
      speaker: data.turn.speaker,
      content: data.turn.public_speech,
      inner_monologue: data.turn.inner_monologue
    });
    return;
  }

  if (data.turn_type === 'night_resolved') {
    hideAllViews();
    nightView.classList.remove('hidden');
    nightMessage.textContent = data.message || 'Night actions resolved.';
    refreshState();
    return;
  }

  if (data.turn_type === 'voting_resolved' || data.vote_result) {
    hideAllViews();
    if (data.message) {
      addTranscriptEntry({ is_system: true, content: data.message });
    }

    // Show vote summary if available
    const votes = data.vote_result ? data.vote_result.votes : null;
    if (votes) {
      const votesSummary = Object.entries(votes)
        .map(([voter, target]) => `${voter} → ${target}`)
        .join(', ');

      addTranscriptEntry({
        is_system: true,
        content: `Final Votes: ${votesSummary}`
      });
    }

    refreshState();

    // If moving to night, show night view
    if (data.phase === 'night') {
      nightView.classList.remove('hidden');
      nightMessage.textContent = `Night ${data.day} falls...`;
    }
    return;
  }

  // Fallback for phase transitions or generic messages
  if (data.message) {
    addTranscriptEntry({
      is_system: true,
      content: data.message
    });
  }

  // Ensure correct view based on phase if we're not in a specific turn type
  if (data.phase === 'night' && data.turn_type === 'phase_transition') {
    hideAllViews();
    nightView.classList.remove('hidden');
    nightMessage.textContent = `Night ${data.day} falls...`;
  }

  if (data.turn_type === 'phase_transition' || data.message) {
    refreshState();
  }
}

function showSpeakerTurn(turn) {
  hideAllViews();
  speakerView.classList.remove('hidden');

  speakerAvatar.textContent = turn.avatar || '🎭';
  speakerName.textContent = turn.speaker;
  speakerRole.textContent = turn.role;
  speakerRole.setAttribute('data-role', turn.role);

  thoughtText.textContent = turn.inner_monologue || '(Thinking...)';
  speechText.textContent = turn.public_speech || '(No speech)';

  // Animate the bubbles
  document.getElementById('thought-bubble').style.animation = 'none';
  document.getElementById('speech-bubble').style.animation = 'none';
  setTimeout(() => {
    document.getElementById('thought-bubble').style.animation = 'slideIn 0.4s ease';
    document.getElementById('speech-bubble').style.animation = 'slideIn 0.4s ease 0.2s both';
  }, 10);
}

function showGameOver(winner) {
  hideAllViews();
  gameoverView.classList.remove('hidden');

  gameoverTitle.textContent = '🎭 Game Over';

  if (winner === 'mafia') {
    gameoverSubtitle.textContent = '🔴 The Mafia has taken over the village!';
    gameoverSubtitle.style.color = 'var(--mafia-color)';
  } else {
    gameoverSubtitle.textContent = '🟢 The Town has eliminated all Mafia members!';
    gameoverSubtitle.style.color = 'var(--town-color)';
  }

  addTranscriptEntry({
    is_system: true,
    content: `🎭 GAME OVER - ${winner.toUpperCase()} WINS!`
  });
}

function hideAllViews() {
  welcomeScreen.classList.add('hidden');
  speakerView.classList.add('hidden');
  nightView.classList.add('hidden');
  gameoverView.classList.add('hidden');
}

function updatePhaseDisplay(phase, day) {
  phaseValue.textContent = phase ? phase.replace('_', ' ') : '--';
  dayValue.textContent = day || 0;
}

function updatePlayerRoster(players) {
  playerRoster.innerHTML = '';

  players.forEach(player => {
    const chip = document.createElement('div');
    chip.className = `player-chip ${player.status}`;
    chip.innerHTML = `
        <span class="chip-avatar">${player.avatar}</span>
        <div class="chip-info">
            <span class="chip-name">${player.name}</span>
            <span class="chip-role">${player.role}</span>
        </div>
    `;
    chip.title = `${player.name} (${player.role})`;
    playerRoster.appendChild(chip);
  });
}

function addTranscriptEntry(entry) {
  const entryEl = document.createElement('div');
  const isSecret = entry.is_secret || false;

  // Determine secret styling
  let secretType = '';
  if (isSecret) {
    const content = entry.content.toUpperCase();
    if (content.includes('[MAFIA]')) secretType = 'secret-mafia';
    else if (content.includes('[DOCTOR]')) secretType = 'secret-doctor';
    else if (content.includes('[SHERIFF]')) secretType = 'secret-sheriff';
    else if (content.includes('[VOTE]')) secretType = 'secret-generic'; // Use generic for votes
    else secretType = 'secret-generic';
  }

  entryEl.className = `transcript-entry ${entry.is_system ? 'system' : 'player'} ${isSecret ? 'secret' : ''} ${secretType}`;

  // Content Body
  let contentHtml = `<div class="transcript-content">${entry.content}</div>`;

  // Dropdown for Monologue
  if (entry.inner_monologue) {
    contentHtml += `
          <details class="monologue-dropdown">
              <summary>💭 Inner Thoughts</summary>
              <div class="monologue-content">${entry.inner_monologue}</div>
          </details>
        `;
  }

  // Header Layout
  if (entry.is_system) {
    entryEl.innerHTML = contentHtml;
  } else {
    // Look up player role details
    const player = currentPlayers.find(p => p.name === entry.speaker);
    const roleHtml = player ? `<span class="transcript-role" data-role="${player.role}">${player.role}</span>` : '';
    const avatarHtml = player ? `<span class="transcript-avatar">${player.avatar}</span>` : '';

    entryEl.innerHTML = `
              <div class="transcript-header">
                  ${avatarHtml}
                  <span class="transcript-name">${entry.speaker}</span>
                  ${roleHtml}
              </div>
              ${contentHtml}
          `;
  }

  // Remove placeholder if present
  const placeholder = transcriptMessages.querySelector('.transcript-placeholder');
  if (placeholder) {
    placeholder.remove();
  }

  transcriptMessages.appendChild(entryEl);
  // Improved scrolling: scroll to bottom of container
  transcriptMessages.scrollTop = transcriptMessages.scrollHeight;
}

function rebuildTranscript(transcript) {
  transcriptMessages.innerHTML = '';
  transcript.forEach(entry => {
    addTranscriptEntry(entry);
  });
}

// Auto-Play Functions
function toggleAutoPlay() {
  if (autoplayToggle.checked) {
    startAutoPlay();
    nextTurnBtn.disabled = true;
    nextTurnBtn.style.opacity = "0.5";
  } else {
    stopAutoPlay();
    nextTurnBtn.disabled = false;
    nextTurnBtn.style.opacity = "1";
  }
}

function startAutoPlay() {
  if (autoPlayInterval) return;

  isAutoPlaying = true;
  nextTurnBtn.disabled = true;
  nextTurnBtn.style.opacity = "0.5";

  const delay = getAutoPlayDelay();

  autoPlayInterval = setInterval(() => {
    if (!isProcessing) {
      nextTurn();
    }
  }, delay);
}

function stopAutoPlay() {
  isAutoPlaying = false;
  autoplayToggle.checked = false;

  if (autoPlayInterval) {
    clearInterval(autoPlayInterval);
    autoPlayInterval = null;
  }

  // Restore button state
  nextTurnBtn.disabled = isProcessing;
  nextTurnBtn.style.opacity = isProcessing ? "0.5" : "1";
}

function getAutoPlayDelay() {
  // Speed slider: 1 = slow (5000ms), 5 = fast (1000ms)
  const speed = parseInt(speedSlider.value);
  return 6000 - (speed * 1000);
}

function updateAutoPlaySpeed() {
  if (isAutoPlaying) {
    stopAutoPlay();
    startAutoPlay();
  }
}

// Event Listeners
startGameBtn.addEventListener('click', startGame);
nextTurnBtn.addEventListener('click', nextTurn);
restartBtn.addEventListener('click', startGame);
autoplayToggle.addEventListener('change', toggleAutoPlay);
speedSlider.addEventListener('input', updateAutoPlaySpeed);

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !e.repeat) {
    e.preventDefault();
    if (!nextTurnBtn.disabled) {
      nextTurn();
    }
  }

  if (e.code === 'KeyA' && !e.repeat) {
    autoplayToggle.checked = !autoplayToggle.checked;
    toggleAutoPlay();
  }
});

// Initialize
console.log('🎭 Mafia AI Simulation loaded');
