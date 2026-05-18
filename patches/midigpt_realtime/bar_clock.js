// bar_clock.js — MIDI-GPT bar boundary driver
inlets  = 3;
outlets = 3;

var bar_index      = 0;
var running        = false;
var bar_start_tick = 0;

function list() {
    var args = arrayfromargs(arguments);
    if (args.length < 4) return;
    var bpm      = args[0];
    var ts_num   = args[1];
    var ts_den   = args[2];
    var now_tick = args[3];

    var tpq       = 480;
    var bar_ticks = tpq * 4.0 * (ts_num / ts_den);

    if (inlet == 0) {
        if (!running) return;
        outlet(0, bar_index, ts_num, ts_den);          // bar/end for completed bar
        outlet(1, bar_index + 1, bpm, ts_num, ts_den); // flush: play bar_index+1 notes
        outlet(2, now_tick, bar_ticks, bar_index + 1); // context for next bar
        bar_index++;
        bar_start_tick = now_tick;
    } else if (inlet == 1) {
        bar_index = 0;
        running   = true;
        bar_start_tick = now_tick;
        outlet(2, bar_start_tick, bar_ticks, 0);       // seed bar context for bar 0
    }
}

function stop() {
    running = false;
}
