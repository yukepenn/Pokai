#!/usr/bin/env node
/**
 * Self-play battle runner with support for external Python policies via stdin/stdout.
 *
 * Supports two policy types per side:
 * - "node_random_v1": Uses internal random AI (like random_selfplay.js)
 * - "python_external_v1": Delegates decisions to Python via JSON stdin/stdout protocol
 *
 * Protocol for python_external_v1:
 *
 * - Node writes to stdout:
 *   {
 *     "type": "request",
 *     "side": "p1" | "p2",
 *     "request_type": "preview" | "move" | "force-switch" | "wait",
 *     "request": { ... Showdown request ... }
 *   }
 *   or, at the very end of the battle:
 *   {
 *     "type": "result",
 *     "format_id": "...",
 *     "p1_name": "...",
 *     "p2_name": "...",
 *     "winner_side": "p1" | "p2" | "tie" | "unknown",
 *     "winner_name": string | null,
 *     "turns": number | null,
 *     "log": "...",
 *     "p1_team_packed": "...",
 *     "p2_team_packed": "...",
 *     "tier_name": string | null,
 *     "trajectory": { "p1": BattleStep[], "p2": BattleStep[] },
 *     "meta": { ... }
 *   }
 *
 * - Node reads from stdin:
 *   { "type": "action", "choice": "team 1234" } for team preview, or
 *   { "type": "action", "choice": "move 1 1, move 2 1" } for moves / switches.
 *
 * Tera is not a separate request_type: it appears in request.active[i].canTerastallize
 * and, if needed, as a modifier in the choice string (e.g. "move 1 1 tera").
 */

const path = require('path');
const readline = require('readline');
const PS_ROOT = path.resolve(__dirname, '..', '..', 'pokemon-showdown');

// Import Showdown modules from dist/sim (compiled TypeScript)
const {BattleStream, getPlayerStreams, Teams} = require(path.join(PS_ROOT, 'dist', 'sim', 'index.js'));
const {RandomPlayerAI} = require(path.join(PS_ROOT, 'dist', 'sim', 'tools', 'random-player-ai.js'));

// Debug logging (only if PY_POLICY_DEBUG env var is set)
const DEBUG = process.env.PY_POLICY_DEBUG === '1';
function debugLog(msg) {
    if (DEBUG) {
        process.stderr.write('[py-policy-selfplay] ' + msg + '\n');
    }
}

// stdout → Python (protocol)
function sendToPython(obj) {
    const line = JSON.stringify(obj);
    debugLog('sendToPython: ' + line);
    process.stdout.write(line + '\n');
}

// stdin ← Python
const rl = readline.createInterface({
    input: process.stdin,
    output: undefined,
});

function readActionOnce() {
    return new Promise((resolve, reject) => {
        rl.once('line', (line) => {
            debugLog('Received line from Python: ' + line);
            let msg;
            try {
                msg = JSON.parse(line);
            } catch (e) {
                return reject(new Error('Failed to parse JSON from Python: ' + e.message));
            }
            if (!msg || msg.type !== 'action' || typeof msg.choice !== 'string') {
                return reject(new Error('Invalid action message from Python: ' + line));
            }
            debugLog('Parsed action from Python: ' + msg.choice);
            debugLog('readActionOnce: parsed valid action message from Python');
            resolve(msg.choice);
        });
    });
}

// Helper for Python-controlled request handling
async function handlePythonControlledRequest(side, request) {
    const hasForceSwitch =
        Array.isArray(request.forceSwitch) &&
        request.forceSwitch.some(Boolean);

    const requestType = request.teamPreview
        ? 'preview'
        : hasForceSwitch
        ? 'force-switch'
        : request.wait
        ? 'wait'
        : 'move';

    if (request.wait) {
        // Showdown is just telling us to wait; no Python roundtrip needed.
        return 'pass';
    }

    sendToPython({
        type: 'request',
        side,
        request_type: requestType,
        request,
    });

    const choice = await readActionOnce();
    debugLog(`Python choice for side=${side}, type=${requestType}: ${choice}`);
    return choice;
}

function buildFallbackChoiceFromRequest(requestOrStructured) {
    // Handle both direct request objects and structured {request_type, request} objects
    let request;
    if (requestOrStructured && typeof requestOrStructured === 'object') {
        if (requestOrStructured.request) {
            // Structured format: {request_type, request}
            request = requestOrStructured.request;
        } else {
            // Direct request object
            request = requestOrStructured;
        }
    }
    
    // If there is no request info, try to generate a minimal safe choice
    if (!request || typeof request !== 'object') {
        // Last resort: try to use move 1 if we have any context
        return 'move 1';
    }

    // 1) Team preview: choose the first maxChosenTeamSize mons, e.g. "team 1234".
    if (request.teamPreview) {
        const side = request.side || {};
        const mons = Array.isArray(side.pokemon) ? side.pokemon : [];
        const teamSize = mons.length;
        if (!teamSize) {
            return 'team 1';
        }
        const maxChosen = request.maxChosenTeamSize || Math.min(4, teamSize);
        const chosen = [];
        for (let i = 1; i <= teamSize && chosen.length < maxChosen; i++) {
            chosen.push(i);
        }
        return 'team ' + chosen.join('');
    }

    // 2) If this is a "wait" request, Showdown expects us to effectively do nothing.
    if (request.wait) {
        return 'pass';
    }

    // Handle both normal active moves and pure force-switch requests.
    const forceSwitchArr = Array.isArray(request.forceSwitch) ? request.forceSwitch : [];
    const activeArr = Array.isArray(request.active) ? request.active : [];

    function needsExplicitTarget(target) {
        const t = (target || '').toLowerCase();
        // Use denylist approach: only these targets don't need a numeric argument
        const noTargetNeeded = {
            'self': true,
            'allyside': true,
            'foeside': true,
            'all': true,
            'allyteam': true,
            'alladjacent': true,
            'alladjacentfoes': true,
            'randomnormal': true,
        };
        return !noTargetNeeded[t] && t !== '';
    }

    function buildMoveChoice(slot, moveIndex, extraTokens) {
        const tokens = ['move', String(moveIndex)];
        const moves = Array.isArray(slot.moves) ? slot.moves : [];

        if (moveIndex >= 1 && moveIndex <= moves.length) {
            const mv = moves[moveIndex - 1] || {};
            const target = (mv.target || '').toLowerCase();
            if (needsExplicitTarget(target)) {
                // For ally-targeting moves, use -1; for foe-targeting, use 1
                const defaultTarget = target.includes('ally') ? '-1' : '1';
                tokens.push(defaultTarget);
            }
        }

        if (Array.isArray(extraTokens)) {
            for (const tok of extraTokens) {
                if (!tok) continue;
                const lower = tok.toLowerCase();
                // Preserve only flag-like tokens; ignore any numeric targets
                if (lower === 'tera' || lower === 'zmove' || lower === 'z' ||
                    lower === 'mega' || lower === 'dynamax') {
                    tokens.push(tok);
                }
            }
        }

        return tokens.join(' ');
    }

    // Determine required slots: use forceSwitch length if present, otherwise active length
    // If active is empty but forceSwitch exists, we still need to handle all forceSwitch slots
    let numSlots = 0;
    if (forceSwitchArr.length > 0) {
        numSlots = forceSwitchArr.length;
    } else if (activeArr.length > 0) {
        numSlots = activeArr.length;
    } else {
        // No active slots and no force-switch flags: fallback to a single move
        // This should be extremely rare, but prefer move over pass
        return 'move 1';
    }

    const side = request.side || {};
    const team = Array.isArray(side.pokemon) ? side.pokemon : [];

    // Precompute bench indices once: non-fainted and not active
    const switchTargets = [];
    for (let i = 0; i < team.length; i++) {
        const mon = team[i];
        if (!mon) continue;
        const cond = (mon.condition || '').toString();
        const isFainted = cond.includes('fnt');
        const isActive = !!mon.active;
        if (!isFainted && !isActive) {
            switchTargets.push(i + 1); // 1-based indexing
        }
    }

    const usedSwitchTargets = [];

    function takeSwitchTarget() {
        // 1) Prefer true bench
        for (let idx of switchTargets) {
            if (!usedSwitchTargets.includes(idx)) {
                usedSwitchTargets.push(idx);
                return idx;
            }
        }
        // 2) Fallback: ANY non-fainted mon (even if active), to avoid 'pass'
        for (let i = 0; i < team.length; i++) {
            const mon = team[i];
            if (!mon) continue;
            const cond = (mon.condition || '').toString();
            const isFainted = cond.includes('fnt');
            if (isFainted) continue;
            const idx = i + 1;
            if (!usedSwitchTargets.includes(idx)) {
                usedSwitchTargets.push(idx);
                return idx;
            }
        }
        // 3) Truly nothing left
        return null;
    }

    const parts = [];

    // Special case: if active is empty but we have force-switch flags, ALL slots must switch
    // (Python sanitizer behavior: when active is empty in force-switch context, all slots switch)
    const activeIsEmpty = activeArr.length === 0;
    const hasAnyForceSwitch = forceSwitchArr.some(Boolean);
    
    for (let slotIdx = 0; slotIdx < numSlots; slotIdx++) {
        const mustSwitch = !!forceSwitchArr[slotIdx];
        const slotReq = activeArr[slotIdx] || {};
        const moves = Array.isArray(slotReq.moves) ? slotReq.moves : [];

        // 2a) Handle forced switch OR empty active in force-switch context: assign distinct bench mons per slot.
        if (mustSwitch || (activeIsEmpty && hasAnyForceSwitch)) {
            const targetIndex = takeSwitchTarget();
            if (targetIndex != null) {
                parts.push(`switch ${targetIndex}`);
            } else {
                // Truly impossible: all mons fainted, no bench available
                // This is the ONLY case where we allow 'pass' for a forced switch
                parts.push('pass');
            }
            continue;
        }

        // 2b) Normal move turn: pick the first available move and assign targets if needed.
        // INVARIANT: Never use "pass" if there are ANY moves available (even disabled ones)
        if (moves.length > 0) {
            // Prefer enabled moves, but use disabled moves if that's all we have
            let moveIndex = null;
            for (let mi = 0; mi < moves.length; mi++) {
                const mv = moves[mi];
                if (!mv) continue;
                if (!mv.disabled) {
                    moveIndex = mi + 1; // 1-based indexing
                    break;
                }
            }
            // If no enabled moves, use first move anyway (even if disabled)
            if (moveIndex == null && moves.length > 0) {
                moveIndex = 1;
            }
            
            if (moveIndex != null) {
                parts.push(buildMoveChoice(slotReq, moveIndex, []));
            } else {
                // This should never happen if moves.length > 0, but defensive fallback
                parts.push('move 1');
            }
        } else {
            // No moves for this slot - try to find moves from any active entry
            let foundMove = false;
            for (let actIdx = 0; actIdx < activeArr.length; actIdx++) {
                const act = activeArr[actIdx] || {};
                const actMoves = Array.isArray(act.moves) ? act.moves : [];
                if (actMoves.length > 0) {
                    // Use first move from this active entry
                    parts.push(buildMoveChoice(act, 1, []));
                    foundMove = true;
                    break;
                }
            }
            if (!foundMove) {
                // Truly no moves available anywhere - this is extremely rare
                // Prefer a switch if possible, otherwise last resort pass
                const switchTarget = takeSwitchTarget();
                if (switchTarget != null) {
                    parts.push(`switch ${switchTarget}`);
                } else {
                    parts.push('pass');
                }
            }
        }
    }

    // If for some reason we produced fewer parts than slots, duplicate the first part
    // so the overall choice string is syntactically complete for doubles/triples.
    // Prefer duplicating a move/switch over 'pass'
    while (parts.length < numSlots) {
        if (parts.length > 0 && !parts[0].startsWith('pass')) {
            parts.push(parts[0]);
        } else {
            // Last resort: try to generate a move
            parts.push('move 1');
        }
    }

    // Second pass: handle disabled moves and illegal 'pass' on non-forced slots
    // (activeArr and forceSwitchArr are already declared above, reuse them)

    function firstEnabledMoveIdx(slot) {
        const moves = Array.isArray(slot.moves) ? slot.moves : [];
        for (let j = 0; j < moves.length; j++) {
            const m = moves[j] || {};
            if (!m.disabled) return j + 1; // 1-based
        }
        return null;
    }

    for (let slotIdx = 0; slotIdx < parts.length && slotIdx < activeArr.length; slotIdx++) {
        const slot = activeArr[slotIdx] || {};
        const moves = Array.isArray(slot.moves) ? slot.moves : [];
        if (!moves.length) continue;

        const isForceSwitch = !!forceSwitchArr[slotIdx];
        let raw = parts[slotIdx] || '';
        let tokens = raw.split(/\s+/).filter(Boolean);
        if (!tokens.length) continue;

        const action = tokens[0];

        if (action === 'move') {
            let idx = null;
            if (tokens.length > 1) {
                const parsed = parseInt(tokens[1], 10);
                if (Number.isFinite(parsed)) idx = parsed;
            }

            const outOfRange = idx == null || idx < 1 || idx > moves.length;
            const disabled = !outOfRange && moves[idx - 1] && moves[idx - 1].disabled;

            if (outOfRange || disabled) {
                const alt = firstEnabledMoveIdx(slot);
                if (alt !== null && alt !== idx) {
                    // Collect extra, non-numeric tokens from the original choice,
                    // such as 'tera', 'mega', 'zmove', etc.
                    const extraTokens = [];
                    for (let k = 2; k < tokens.length; k++) {
                        const tok = tokens[k];
                        if (!tok) continue;
                        const lower = tok.toLowerCase();
                        // Ignore numeric tokens; let buildMoveChoice decide targets.
                        if (lower === 'tera' || lower === 'zmove' || lower === 'z' ||
                            lower === 'mega' || lower === 'dynamax') {
                            extraTokens.push(tok);
                        }
                    }

                    parts[slotIdx] = buildMoveChoice(slot, alt, extraTokens);
                    raw = parts[slotIdx]; // keep raw in sync in case it's used later
                }
            }
        } else if (action === 'pass' && !isForceSwitch) {
            const alt = firstEnabledMoveIdx(slot);
            if (alt !== null) {
                // Build move choice with correct target handling
                parts[slotIdx] = buildMoveChoice(slot, alt, []);
            }
        }
    }

    const finalChoice = parts.join(', ');

    if (process.env.PY_POLICY_DEBUG === '1') {
        const forceSwitch = Array.isArray(request.forceSwitch) ? request.forceSwitch : [];
        for (let i = 0; i < forceSwitch.length && i < parts.length; i++) {
            if (forceSwitch[i] && parts[i].startsWith('pass')) {
                debugLog(
                    '[py-policy-selfplay] WARN: forced-switch slot sanitized to \'pass\'; ' +
                    JSON.stringify({ slot: i, choice: finalChoice })
                );
                break;
            }
        }
    }

    return finalChoice;
}

/**
 * Wrapper around RandomPlayerAI that logs all request/choice pairs.
 * Uses RandomPlayerAI's existing start/stream handling; we only hook into
 * receiveRequest() and choose().
 */
class LoggingRandomPlayerAI extends RandomPlayerAI {
    constructor(stream, sideId, recorder, options = {}) {
        super(stream, options);
        this.sideId = sideId;
        this.recorder = recorder;
        this._currentRequest = null;
    }

    receiveRequest(request) {
        this._currentRequest = request;
        super.receiveRequest(request);
        this._currentRequest = null;
    }

    chooseTeamPreview(team) {
        const request = this._currentRequest;
        let pokemon;
        let maxChosen;
        if (request && request.side) {
            pokemon = request.side.pokemon || team || [];
            maxChosen = request.maxChosenTeamSize;
        } else {
            pokemon = team || [];
            maxChosen = undefined;
        }
        const total = pokemon.length;
        const clampedMax = maxChosen
            ? Math.max(1, Math.min(maxChosen, total))
            : Math.max(1, Math.min(4, total));

        const availableSlots = [];
        for (let i = 1; i <= total; i++) {
            availableSlots.push(i);
        }

        const chosen = [];
        const slotsCopy = [...availableSlots];
        for (let i = 0; i < clampedMax; i++) {
            const randomIndex = Math.floor(Math.random() * slotsCopy.length);
            chosen.push(slotsCopy.splice(randomIndex, 1)[0]);
        }

        chosen.sort((a, b) => a - b);
        return 'team ' + chosen.join('');
    }

    choose(choiceString) {
        const result = super.choose(choiceString);

        if (this._currentRequest) {
            let snapshotRequest;
            try {
                snapshotRequest = JSON.parse(JSON.stringify(this._currentRequest));
            } catch {
                snapshotRequest = {error: 'failed-to-serialize-request'};
            }

            const requestType = snapshotRequest?.teamPreview
                ? 'preview'
                : snapshotRequest?.forceSwitch
                ? 'force-switch'
                : snapshotRequest?.wait
                ? 'wait'
                : 'move';

            const step = {
                side: this.sideId,
                step_index: this.recorder.length,
                request_type: requestType,
                rqid: snapshotRequest?.rqid ?? null,
                turn: snapshotRequest?.turn ?? null,
                request: snapshotRequest,
                choice: choiceString,
            };

            this.recorder.push(step);
        }

        return result;
    }
}

/**
 * Player AI that delegates all decisions to Python via stdin/stdout.
 * We do NOT override start() or line-level parsing; instead we override
 * receiveRequest(request) so we stay aligned with Showdown's PlayerAI
 * protocol.
 */
class PythonBridgePlayerAI extends RandomPlayerAI {
    constructor(stream, sideId, recorder, options = {}) {
        super(stream, options);
        this.sideId = sideId;
        this.recorder = recorder;
        this._pendingRequest = null;
        this._lastRequest = null;
        this._lastChoice = null;
    }

    receiveRequest(request) {
        // Compute request_type for structured logging
        const hasForceSwitch =
            Array.isArray(request.forceSwitch) &&
            request.forceSwitch.some(Boolean);
        const requestType = request.teamPreview
            ? 'preview'
            : hasForceSwitch
            ? 'force-switch'
            : request.wait
            ? 'wait'
            : 'move';
        
        // Remember the last structured request for potential fallback in receiveError.
        this._lastRequest = {
            request_type: requestType,
            request: request,
        };
        // If you still have _pendingRequest logic, keep it if needed:
        this._pendingRequest = request;

        this._handleRequestAsync(request).catch(e => {
            console.error(`Fatal error in PythonBridgePlayerAI for ${this.sideId}: ${e.message}`);
            try {
                // On hard failure talking to Python, fall back to a local, safe choice.
                const fallbackChoice = buildFallbackChoiceFromRequest(request);
                debugLog(`[${this.sideId}] using local fallback choice after fatal error: ${fallbackChoice}`);
                this.choose(fallbackChoice);
            } catch {
                // Ignore errors in this emergency fallback; let the engine handle it.
            }
        }).finally(() => {
            // You can clear _pendingRequest here if you still use it.
            this._pendingRequest = null;
        });
    }

    async _handleRequestAsync(request) {
        const hasForceSwitch =
            Array.isArray(request.forceSwitch) &&
            request.forceSwitch.some(Boolean);

        const requestType = request.teamPreview
            ? 'preview'
            : hasForceSwitch
            ? 'force-switch'
            : request.wait
            ? 'wait'
            : 'move';

        debugLog(
            `[${this.sideId}] receiveRequest: type=${requestType}, ` +
            `rqid=${request.rqid ?? 'none'}, turn=${request.turn ?? 'none'}`
        );

        let choiceString;
        try {
            debugLog(`[${this.sideId}] sending request to Python via handlePythonControlledRequest`);
            choiceString = await handlePythonControlledRequest(this.sideId, request);
            debugLog(`[${this.sideId}] got Python choice: ${choiceString}`);
        } catch (e) {
            console.error(`Error processing Python-controlled request for ${this.sideId}: ${e.message}`);
            // Fallback to a locally constructed safe choice
            choiceString = buildFallbackChoiceFromRequest(request);
            debugLog(`[${this.sideId}] Using local fallback choice from _handleRequestAsync: ${choiceString}`);
        }

        // Log the step into the trajectory recorder
        let snapshotRequest;
        try {
            snapshotRequest = JSON.parse(JSON.stringify(request));
        } catch {
            snapshotRequest = {error: 'failed-to-serialize-request'};
        }

        const step = {
            side: this.sideId,
            step_index: this.recorder.length,
            request_type: requestType,
            rqid: snapshotRequest?.rqid ?? null,
            turn: snapshotRequest?.turn ?? null,
            request: snapshotRequest,
            choice: choiceString,
        };

        this.recorder.push(step);

        // Remember the last choice for potential error logging
        this._lastChoice = choiceString;

        // Send choice to the simulator
        this.choose(choiceString);
    }

    receiveError(error) {
        const errorMsg = error && error.message ? error.message : String(error);

        // STRICT MODE ONLY: Always treat [Invalid choice] as fatal error
        // Non-strict mode is disabled for now to prevent infinite retry loops
        if (errorMsg.includes('[Invalid choice]')) {
            // Always log the full error context for debugging
            const lastReq = this._lastRequest || {};
            const reqType = lastReq.request_type || 'unknown';
            const request = lastReq.request || {};
            debugLog(
                '[py-policy-selfplay] Invalid choice from Python: ' +
                JSON.stringify({
                    side: this.sideId,
                    error: errorMsg,
                    lastChoice: this._lastChoice,
                    lastRequest: this._lastRequest,
                })
            );
            // Additional diagnostic logging
            debugLog(
                '[strict-invalid-choice] ' +
                JSON.stringify({
                    side: this.sideId,
                    reqType: reqType,
                    lastChoice: this._lastChoice,
                    requestType: reqType,
                    requestForceSwitch: request.forceSwitch,
                    requestActiveLen: request.active ? request.active.length : 0,
                    requestActiveMoves: request.active ? request.active.map(a => a.moves ? a.moves.length : 0) : [],
                })
            );
            // Strict mode: delegate to base class to abort the battle
            // This exposes sanitizer bugs during development/testing
            debugLog(
                '[py-policy-selfplay] Strict mode - treating invalid choice as fatal error'
            );
            super.receiveError(error);
            return;
        }

        // For non-invalid-choice errors, delegate to base handler
        super.receiveError(error);
    }
}

// Parse CLI arguments
const args = process.argv.slice(2);
let formatId = 'gen9vgc2026regf';
let p1Name = 'Bot1';
let p2Name = 'Bot2';
let p1Team = null;
let p2Team = null;
let p1Policy = 'node_random_v1';
let p2Policy = 'node_random_v1';
let maxTurns = 100;
let seed = null;

for (let i = 0; i < args.length; i += 2) {
    const flag = args[i];
    const value = args[i + 1];
    if (flag === '--format-id' && value) {
        formatId = value;
    } else if (flag === '--p1-name' && value) {
        p1Name = value;
    } else if (flag === '--p2-name' && value) {
        p2Name = value;
    } else if (flag === '--p1-packed-team' && value) {
        p1Team = value;
    } else if (flag === '--p2-packed-team' && value) {
        p2Team = value;
    } else if (flag === '--p1-policy' && value) {
        p1Policy = value;
    } else if (flag === '--p2-policy' && value) {
        p2Policy = value;
    } else if (flag === '--max-turns' && value) {
        maxTurns = parseInt(value, 10);
    } else if (flag === '--seed' && value) {
        seed = parseInt(value, 10);
    }
}

// Guard: Currently only one Python-controlled side is supported
if (p1Policy === 'python_external_v1' && p2Policy === 'python_external_v1') {
    throw new Error(
        'Currently only one Python-controlled side is supported. ' +
        'Use python_external_v1 for one side and node_random_v1 for the other.'
    );
}

debugLog('Starting battle with p1Policy=' + p1Policy + ', p2Policy=' + p2Policy);

async function runBattle() {
    try {
        // Create battle stream and player streams
        const battleStream = new BattleStream();
        const streams = getPlayerStreams(battleStream);
        debugLog('BattleStream and player streams created');

        // Arrays to collect trajectory steps
        const p1Steps = [];
        const p2Steps = [];

        // Generate or use provided teams
        if (!p1Team) {
            p1Team = Teams.pack(Teams.generate(formatId));
        }
        if (!p2Team) {
            p2Team = Teams.pack(Teams.generate(formatId));
        }
        debugLog('Teams prepared for p1 and p2');

        // Create player AIs based on policy type
        let p1, p2;

        if (p1Policy === 'node_random_v1') {
            p1 = new LoggingRandomPlayerAI(streams.p1, 'p1', p1Steps);
        } else if (p1Policy === 'python_external_v1') {
            p1 = new PythonBridgePlayerAI(streams.p1, 'p1', p1Steps);
        } else {
            throw new Error(`Unknown p1_policy: ${p1Policy}`);
        }

        if (p2Policy === 'node_random_v1') {
            p2 = new LoggingRandomPlayerAI(streams.p2, 'p2', p2Steps);
        } else if (p2Policy === 'python_external_v1') {
            p2 = new PythonBridgePlayerAI(streams.p2, 'p2', p2Steps);
        } else {
            throw new Error(`Unknown p2_policy: ${p2Policy}`);
        }

        debugLog(`Instantiated AIs: p1Policy=${p1Policy}, p2Policy=${p2Policy}`);

        // Start both AIs
        debugLog('Starting p1 AI');
        const p1Promise = p1.start();
        debugLog('Starting p2 AI');
        const p2Promise = p2.start();

        // Accumulate log lines
        const logLines = [];

        // Read from omniscient stream to collect battle log
        const logPromise = (async () => {
            for await (const chunk of streams.omniscient) {
                logLines.push(chunk);
                if (DEBUG) {
                    debugLog(`[omniscient] received chunk of length ${chunk.length}`);
                }
            }
        })();

        // Start the battle
        const spec = {formatid: formatId};
        const p1spec = {name: p1Name, team: p1Team};
        const p2spec = {name: p2Name, team: p2Team};

        debugLog('Writing >start and >player commands to omniscient stream');
        await streams.omniscient.write(
            `>start ${JSON.stringify(spec)}\n` +
            `>player p1 ${JSON.stringify(p1spec)}\n` +
            `>player p2 ${JSON.stringify(p2spec)}`
        );
        debugLog('Finished writing start and player specs');

        // 等待战斗结束（两个 AI 完成）
        await Promise.all([p1Promise, p2Promise]);
        debugLog('Both AIs completed');

        // 战斗结束后，关闭流以让 logPromise 的 for await 循环结束
        await streams.omniscient.writeEnd();
        debugLog('Omniscient stream closed');

        // 等待 log 循环结束
        await logPromise;
        debugLog('Log promise resolved');

        // Join log lines
        const log = logLines.join('\n');

        // Parse winner and turns from log
        let winnerSide = 'unknown';
        let winnerName = null;
        let turns = null;

        const lines = log.split('\n');
        for (const line of lines) {
            if (line.startsWith('|win|')) {
                winnerName = line.slice(5).trim();
                if (winnerName === p1Name) {
                    winnerSide = 'p1';
                } else if (winnerName === p2Name) {
                    winnerSide = 'p2';
                } else {
                    winnerSide = 'unknown';
                }
                break;
            }
            if (line === '|tie|' || line.startsWith('|tie|')) {
                winnerSide = 'tie';
                winnerName = null;
                break;
            }
            if (line.startsWith('|turn|')) {
                const parts = line.split('|');
                if (parts.length >= 3) {
                    const turnNum = parseInt(parts[2], 10);
                    if (!isNaN(turnNum)) {
                        turns = turnNum;
                    }
                }
            }
        }

        // Extract tier name from log
        let tierName = null;
        for (const line of lines) {
            if (line.startsWith('|tier|')) {
                tierName = line.slice(6).trim();
                break;
            }
        }

        // Output JSON result (same schema as random_selfplay.js)
        debugLog('Constructing final battle result JSON');
        const result = {
            type: 'result',
            format_id: formatId,
            p1_name: p1Name,
            p2_name: p2Name,
            winner_side: winnerSide,
            winner_name: winnerName,
            turns: turns,
            log: log,
            p1_team_packed: p1Team,
            p2_team_packed: p2Team,
            tier_name: tierName,
            trajectory: {
                p1: p1Steps,
                p2: p2Steps,
            },
            meta: {
                battle_policy_id_p1: p1Policy,
                battle_policy_id_p2: p2Policy,
                agent_p1: p1Policy === 'node_random_v1' ? 'RandomPlayerAI' : 'PythonExternal',
                agent_p2: p2Policy === 'node_random_v1' ? 'RandomPlayerAI' : 'PythonExternal',
                env_version: 'pokemon-showdown@local',
            },
        };

        debugLog('Emitting final battle JSON result');
        sendToPython(result);

        // Close readline interface
        rl.close();

        process.exit(0);
    } catch (error) {
        console.error(`Error: ${error.message}`);
        if (error.stack) {
            console.error(error.stack);
        }
        rl.close();
        process.exit(1);
    }
}

// Run the battle
runBattle().catch(error => {
    console.error(`Fatal error: ${error.message}`);
    if (error.stack) {
        console.error(error.stack);
    }
    rl.close();
    process.exit(1);
});
