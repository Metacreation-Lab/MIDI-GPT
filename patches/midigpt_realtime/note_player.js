// note_player.js — buffer and schedule AI-generated notes
inlets  = 4;
outlets = 1;

var buffer = {};

function list() {
    var args = arrayfromargs(arguments);

    if (inlet == 0) {
        // /generated/open: allocate buffer for this bar
        buffer[args[1]] = [];

    } else if (inlet == 1) {
        // /generated/note: accumulate
        var bar_idx = args[1];
        if (!buffer[bar_idx]) buffer[bar_idx] = [];
        buffer[bar_idx].push([args[2], args[3], args[4], args[5]]);
        // [pitch, vel, onset, dur]

    } else if (inlet == 2) {
        // /generated/close: batch complete, wait for flush

    } else if (inlet == 3) {
        // flush: play buffered notes for the bar that is starting NOW
        var bar_idx = args[0];
        var bpm     = args[1];
        var ts_num  = args[2];
        var ts_den  = args[3];
        var bar_ms  = (ts_num / ts_den) * 4.0 * (60000.0 / bpm);
        var notes   = buffer[bar_idx];
        if (!notes || notes.length === 0) { delete buffer[bar_idx]; return; }

        for (var i = 0; i < notes.length; i++) {
            var n = notes[i];
            var onset_ms = n[2] * bar_ms;
            var dur_ms   = Math.max(10, n[3] * bar_ms);
            (function(p, v, d, t_ms) {
                var task = new Task(function() { outlet(0, p, v, d); }, this);
                task.schedule(t_ms);
            })(n[0], n[1], dur_ms, onset_ms);
        }
        delete buffer[bar_idx];
    }
}
