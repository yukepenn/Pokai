#!/usr/bin/env node
/**
 * Convert Pokemon Showdown import/export team text to packed team string.
 * 
 * Reads team text from stdin and outputs packed string to stdout.
 */

const path = require('path');
const PS_ROOT = path.resolve(__dirname, '..', '..', 'pokemon-showdown');

// Import Teams module from dist/sim (compiled TypeScript)
const { Teams } = require(path.join(PS_ROOT, 'dist', 'sim', 'index.js'));

// Parse CLI arguments
const args = process.argv.slice(2);
let formatId = 'gen9vgc2026regf';

for (let i = 0; i < args.length; i += 2) {
    const flag = args[i];
    const value = args[i + 1];
    
    if (flag === '--format' && value) {
        formatId = value;
    }
}

// Read stdin
let input = '';
process.stdin.setEncoding('utf8');

process.stdin.on('data', chunk => {
    input += chunk;
});

process.stdin.on('end', () => {
    try {
        // Convert import text to sets array
        // Teams.import(buffer, aggressive) parses export format text
        // aggressive=false means use Dex.getName for name resolution
        const sets = Teams.import(input, false);
        
        if (!sets || !sets.length) {
            console.error('Failed to import team from input text: no sets found');
            process.exit(1);
        }
        
        // Pack the team sets into a single string
        const packed = Teams.pack(sets);
        
        if (!packed) {
            console.error('Failed to pack team');
            process.exit(1);
        }
        
        // Output packed string (trimmed)
        process.stdout.write(String(packed).trim());
        process.exit(0);
        
    } catch (error) {
        console.error(`Error converting team: ${error.message}`);
        process.exit(1);
    }
});

process.stdin.on('error', error => {
    console.error(`Error reading stdin: ${error.message}`);
    process.exit(1);
});

