#!/usr/bin/env node
/**
 * Random self-play battle runner using Pokemon Showdown's BattleStream and RandomPlayerAI.
 * 
 * Outputs a single JSON object to stdout with battle results.
 */

const path = require('path');
const PS_ROOT = path.resolve(__dirname, '..', '..', 'pokemon-showdown');

// Import Showdown modules from dist/sim (compiled TypeScript)
const { BattleStream, getPlayerStreams, Teams } = require(path.join(PS_ROOT, 'dist', 'sim', 'index.js'));
const { RandomPlayerAI } = require(path.join(PS_ROOT, 'dist', 'sim', 'tools', 'random-player-ai.js'));

/**
 * Wrapper around RandomPlayerAI that logs all request/choice pairs.
 */
class LoggingRandomPlayerAI extends RandomPlayerAI {
    /**
     * @param {object} stream - Player stream from getPlayerStreams(...)
     * @param {string} sideId - 'p1' or 'p2'
     * @param {Array<object>} recorder - Array where we will push BattleStep-like dicts
     * @param {object} [options] - Any options to pass to the base RandomPlayerAI
     */
    constructor(stream, sideId, recorder, options = {}) {
        super(stream, options);
        this.sideId = sideId;
        this.recorder = recorder;
    }

    /**
     * Override receiveRequest to intercept requests.
     * The base class processes the request and calls this.choose(choiceString).
     * We store the current request, then let the base class handle it.
     */
    receiveRequest(request) {
        // Store the current request for later matching with the choice
        this._currentRequest = request;
        
        // Call the base implementation which will call this.choose(choiceString)
        super.receiveRequest(request);
        
        // Clear the stored request after processing
        this._currentRequest = null;
    }

    /**
     * Override chooseTeamPreview to return explicit "team XXXX" choices instead of "default".
     * This generates explicit bring-4 selections so that rl_preview can train on non-degenerate data.
     */
    chooseTeamPreview(team) {
        const request = this._currentRequest;
        
        // Use request data if available, otherwise fall back to team parameter
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
        
        // Get maxChosenTeamSize (usually 4 for VGC)
        const clampedMax = maxChosen ? Math.max(1, Math.min(maxChosen, total)) : Math.max(1, Math.min(4, total));

        // Build array of slot indices 1..total (Showdown uses 1-based slots)
        const availableSlots = [];
        for (let i = 1; i <= total; i++) {
            availableSlots.push(i);
        }

        // Randomly sample clampedMax distinct slots
        const chosen = [];
        const slotsCopy = [...availableSlots];
        for (let i = 0; i < clampedMax; i++) {
            const randomIndex = Math.floor(Math.random() * slotsCopy.length);
            chosen.push(slotsCopy.splice(randomIndex, 1)[0]);
        }

        // Sort ascending
        chosen.sort((a, b) => a - b);

        // Return choice string: "team " + slots.join("")
        return "team " + chosen.join("");
    }

    /**
     * Override choose to capture the choice string and match it with the stored request.
     */
    choose(choiceString) {
        // Call the base implementation
        const result = super.choose(choiceString);
        
        // If we have a stored request, log this step
        if (this._currentRequest) {
            // Take a JSON-safe deep copy of the request
            let snapshotRequest;
            try {
                snapshotRequest = JSON.parse(JSON.stringify(this._currentRequest));
            } catch {
                snapshotRequest = { error: 'failed-to-serialize-request' };
            }

            const step = {
                side: this.sideId,
                step_index: this.recorder.length,
                request_type: snapshotRequest?.teamPreview
                    ? 'team-preview'
                    : snapshotRequest?.forceSwitch
                    ? 'force-switch'
                    : snapshotRequest?.wait
                    ? 'wait'
                    : 'move',
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

// Parse CLI arguments
const args = process.argv.slice(2);
let formatId = 'gen9vgc2026regf';
let p1Name = 'Bot1';
let p2Name = 'Bot2';
let p1Team = null;
let p2Team = null;

for (let i = 0; i < args.length; i += 2) {
    const flag = args[i];
    const value = args[i + 1];
    
    if (flag === '--format' && value) {
        formatId = value;
    } else if (flag === '--p1-name' && value) {
        p1Name = value;
    } else if (flag === '--p2-name' && value) {
        p2Name = value;
    } else if (flag === '--p1-team' && value) {
        p1Team = value;
    } else if (flag === '--p2-team' && value) {
        p2Team = value;
    }
}

async function runBattle() {
    try {
        // Create battle stream and player streams
        const battleStream = new BattleStream();
        const streams = getPlayerStreams(battleStream);

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

        // Create logging random player AIs
        const p1 = new LoggingRandomPlayerAI(streams.p1, 'p1', p1Steps);
        const p2 = new LoggingRandomPlayerAI(streams.p2, 'p2', p2Steps);

        // Start both AIs
        const p1Promise = p1.start();
        const p2Promise = p2.start();

        // Accumulate log lines
        const logLines = [];

        // Read from omniscient stream to collect battle log
        const logPromise = (async () => {
            for await (const chunk of streams.omniscient) {
                logLines.push(chunk);
            }
        })();

        // Start the battle
        const spec = { formatid: formatId };
        const p1spec = { name: p1Name, team: p1Team };
        const p2spec = { name: p2Name, team: p2Team };

        await streams.omniscient.write(
            `>start ${JSON.stringify(spec)}\n` +
            `>player p1 ${JSON.stringify(p1spec)}\n` +
            `>player p2 ${JSON.stringify(p2spec)}`
        );

        // Wait for all promises to complete
        await Promise.all([p1Promise, p2Promise, logPromise]);
        
        // Close the stream
        await streams.omniscient.writeEnd();

        // Join log lines (they already contain newlines where needed)
        const log = logLines.join('\n');

        // Parse winner and turns from log
        let winnerSide = 'unknown';
        let winnerName = null;
        let turns = null;

        const lines = log.split('\n');
        for (const line of lines) {
            // Check for win
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
            // Check for tie
            if (line === '|tie|' || line.startsWith('|tie|')) {
                winnerSide = 'tie';
                winnerName = null;
                break;
            }
            // Track turns
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

        // Extract tier name from log (optional)
        let tierName = null;
        for (const line of lines) {
            if (line.startsWith('|tier|')) {
                tierName = line.slice(6).trim();
                break;
            }
        }

        // Output JSON result
        const result = {
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
            // Basic meta so Python can store agent/env information
            agent_p1: 'RandomPlayerAI',
            agent_p2: 'RandomPlayerAI',
            env_version: 'pokemon-showdown@local',
        };

        console.log(JSON.stringify(result));
        process.exit(0);

    } catch (error) {
        // On error, output to stderr and exit with non-zero code
        console.error(`Error: ${error.message}`);
        process.exit(1);
    }
}

// Run the battle
runBattle().catch(error => {
    console.error(`Fatal error: ${error.message}`);
    process.exit(1);
});

