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

        // Generate or use provided teams
        if (!p1Team) {
            p1Team = Teams.pack(Teams.generate(formatId));
        }
        if (!p2Team) {
            p2Team = Teams.pack(Teams.generate(formatId));
        }

        // Create random player AIs
        const p1 = new RandomPlayerAI(streams.p1);
        const p2 = new RandomPlayerAI(streams.p2);

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

        // Output JSON result
        const result = {
            format_id: formatId,
            p1_name: p1Name,
            p2_name: p2Name,
            winner_side: winnerSide,
            winner_name: winnerName,
            turns: turns,
            log: log
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

