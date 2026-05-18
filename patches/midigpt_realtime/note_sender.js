// note_sender.js — normalize MIDI to MIDI-GPT /note messages
inlets  = 2;
outlets = 1;

var pending       = {};
var cur_bar_start = 0;
var cur_bar_ticks = 1920;   // 4/4 at 480 tpq default
var cur_bar_idx   = 0;
var track_id      = 0;

function list() {
    var args = arrayfromargs(arguments);

    if (inlet == 1) {
        // bar context update: [bar_start_ticks, bar_ticks, bar_idx]
        cur_bar_start = args[0];
        cur_bar_ticks = args[1];
        cur_bar_idx   = args[2];
        return;
    }

    // inlet 0: [pitch, vel, now_tick]
    if (args.length < 3) return;
    var pitch = args[0];
    var vel   = args[1];
    var now   = args[2];

    if (vel > 0) {
        pending[pitch] = {
            on_tick  : now,
            bar_idx  : cur_bar_idx,
            bar_start: cur_bar_start,
            bar_ticks: cur_bar_ticks
        };
    } else {
        var p = pending[pitch];
        if (!p) return;
        var onset = (p.on_tick - p.bar_start) / p.bar_ticks;
        var dur   = (now       - p.on_tick)   / p.bar_ticks;
        onset = Math.max(0.0,   Math.min(0.9999,       onset));
        dur   = Math.max(0.001, Math.min(1.0 - onset,  dur));
        outlet(0, track_id, pitch, vel, onset, dur, p.bar_idx);
        delete pending[pitch];
    }
}

function settrack(id) {
    track_id = id;
}
