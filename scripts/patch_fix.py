#!/usr/bin/env python3
import sys

with open('/Users/paultriana/creative_labs/MIDI-GPT/scripts/gen_midigpt_realtime_patch.py', 'r') as f:
    content = f.read()

# 1. Update BAR_CLOCK_JS
old_bar_clock = """\
function bang() {
    if (!running) return;
    var t         = new Transport();
    var ts_num    = t.timesignature[0];
    var ts_den    = t.timesignature[1];
    var bpm       = t.bpm;
    var now_tick  = t.ticks;
    var tpq       = 480;
    var bar_ticks = tpq * 4.0 * (ts_num / ts_den);

    outlet(0, bar_index, ts_num, ts_den);          // bar/end for completed bar
    outlet(1, bar_index + 1, bpm, ts_num, ts_den); // flush: play bar_index+1 notes
    outlet(2, now_tick, bar_ticks, bar_index + 1); // context for next bar

    bar_index++;
    bar_start_tick = now_tick;
}

function start() {
    bar_index = 0;
    running   = true;
    var t         = new Transport();
    var ts_num    = t.timesignature[0];
    var ts_den    = t.timesignature[1];
    var tpq       = 480;
    var bar_ticks = tpq * 4.0 * (ts_num / ts_den);
    bar_start_tick = t.ticks;
    outlet(2, bar_start_tick, bar_ticks, 0);       // seed bar context for bar 0
}"""

new_bar_clock = """\
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
}"""

# 2. Update NOTE_SENDER_JS
old_note_sender = """\
    // inlet 0: [pitch vel] note event
    var pitch = args[0];
    var vel   = args[1];
    var t     = new Transport();
    var now   = t.ticks;

    if (vel > 0) {"""

new_note_sender = """\
    // inlet 0: [pitch, vel, now_tick]
    if (args.length < 3) return;
    var pitch = args[0];
    var vel   = args[1];
    var now   = args[2];

    if (vel > 0) {"""

content = content.replace(old_bar_clock, new_bar_clock)
content = content.replace(old_note_sender, new_note_sender)

# 3. Update Max patch bar clock generation
old_bar_patch = """\
timepoint_obj = patch.place("timepoint 1 0 0",   starting_pos=[L,       y2])[0]
bar_clock_js  = patch.place("js bar_clock.js",    starting_pos=[L,       y2 + R])[0]
bar_idx_disp  = patch.place("number",              starting_pos=[L + 220, y2 + R])[0]
oscfmt_barend = patch.place("prepend /midigpt/bar/end",
                              starting_pos=[L,     y2 + R*2])[0]

patch.connect(
    (timepoint_obj.outs[0], bar_clock_js.ins[0]),
    (bar_clock_js.outs[0],  oscfmt_barend.ins[0]),
    (oscfmt_barend.outs[0], udpsend.ins[0]),
    (bar_clock_js.outs[2],  bar_idx_disp.ins[0]),  # display current bar index
)

# session start/stop wire to bar_clock
msg_bc_start = patch.place('message "start"', starting_pos=[L + 110, y2 + R - 22])[0]
msg_bc_stop  = patch.place('message "stop"',  starting_pos=[L + 170, y2 + R - 22])[0]
patch.connect(
    (msg_start.outs[0],    msg_bc_start.ins[0]),
    (msg_stop.outs[0],     msg_bc_stop.ins[0]),
    (msg_bc_start.outs[0], bar_clock_js.ins[1]),
    (msg_bc_stop.outs[0],  bar_clock_js.ins[2]),
)"""

new_bar_patch = """\
timepoint_obj = patch.place("timepoint 1 0 0",   starting_pos=[L,       y2])[0]

# Bar Clock Transport Trigger
trans_bc      = patch.place("transport",         starting_pos=[L+100, y2])[0]
unpack_ts     = patch.place("unpack i i",        starting_pos=[L+200, y2])[0]
pack_bc       = patch.place("pack 0. 0 0 0",     starting_pos=[L+100, y2 + R])[0]

# Start Transport Trigger
trans_start   = patch.place("transport",         starting_pos=[L+300, y2])[0]
unpack_tss    = patch.place("unpack i i",        starting_pos=[L+400, y2])[0]
pack_start    = patch.place("pack 0. 0 0 0",     starting_pos=[L+300, y2 + R])[0]

bar_clock_js  = patch.place("js bar_clock.js",    starting_pos=[L,       y2 + R*2])[0]
bar_idx_disp  = patch.place("number",              starting_pos=[L + 160, y2 + R*3])[0]
oscfmt_barend = patch.place("prepend /midigpt/bar/end", starting_pos=[L, y2 + R*3])[0]

patch.connect(
    # Timepoint wiring
    (timepoint_obj.outs[0], trans_bc.ins[0]),      # timepoint hits transport
    (trans_bc.outs[7],      pack_bc.ins[3]),       # ticks
    (trans_bc.outs[5],      unpack_ts.ins[0]),     # timesig list
    (unpack_ts.outs[1],     pack_bc.ins[2]),       # ts_den
    (unpack_ts.outs[0],     pack_bc.ins[1]),       # ts_num
    (trans_bc.outs[4],      pack_bc.ins[0]),       # tempo (triggers pack)
    (pack_bc.outs[0],       bar_clock_js.ins[0]),  # list to bar_clock
    
    (bar_clock_js.outs[0],  oscfmt_barend.ins[0]),
    (oscfmt_barend.outs[0], udpsend.ins[0]),
    (bar_clock_js.outs[2],  bar_idx_disp.ins[0]),  # display current bar index
)

# session start/stop wire to bar_clock
msg_bc_stop  = patch.place('message "stop"',  starting_pos=[L + 300, y2 - 25])[0]

patch.connect(
    # Start wiring
    (msg_start.outs[0],     trans_start.ins[0]),   # start button hits transport
    (trans_start.outs[7],   pack_start.ins[3]),
    (trans_start.outs[5],   unpack_tss.ins[0]),
    (unpack_tss.outs[1],    pack_start.ins[2]),
    (unpack_tss.outs[0],    pack_start.ins[1]),
    (trans_start.outs[4],   pack_start.ins[0]),
    (pack_start.outs[0],    bar_clock_js.ins[1]),  # list to inlet 1
    
    # Stop wiring
    (msg_stop.outs[0],      msg_bc_stop.ins[0]),
    (msg_bc_stop.outs[0],   bar_clock_js.ins[2]),
)"""

content = content.replace(old_bar_patch, new_bar_patch)

# 4. Update Max patch MIDI Input (pack i i -> pack i i i with transport)
old_midi_in = """\
notein_obj  = patch.place("notein 1 1",           starting_pos=[L,      y3])[0]
pack_note   = patch.place("pack i i",              starting_pos=[L,      y3 + R])[0]
note_sender = patch.place("js note_sender.js",     starting_pos=[L,      y3 + R*2])[0]
oscfmt_note = patch.place("prepend /midigpt/note",
                           starting_pos=[L,         y3 + R*3])[0]

patch.connect(
    (notein_obj.outs[0],   pack_note.ins[0]),    # pitch
    (notein_obj.outs[1],   pack_note.ins[1]),    # velocity (includes note-offs as vel=0)
    (pack_note.outs[0],    note_sender.ins[0]),  # [pitch vel] list
    (bar_clock_js.outs[2], note_sender.ins[1]),  # bar context
    (note_sender.outs[0],  oscfmt_note.ins[0]),
    (oscfmt_note.outs[0],  udpsend.ins[0]),
)"""

new_midi_in = """\
notein_obj  = patch.place("notein 1 1",           starting_pos=[L,      y3])[0]
note_t      = patch.place("t i b",                starting_pos=[L,      y3 + R])[0]
trans_note  = patch.place("transport",            starting_pos=[L+100,  y3 + R])[0]
pack_note   = patch.place("pack i i i",           starting_pos=[L,      y3 + R*2])[0]
note_sender = patch.place("js note_sender.js",     starting_pos=[L,      y3 + R*3])[0]
oscfmt_note = patch.place("prepend /midigpt/note", starting_pos=[L, y3 + R*4])[0]

patch.connect(
    (notein_obj.outs[1],   pack_note.ins[1]),    # velocity
    (notein_obj.outs[0],   note_t.ins[0]),       # pitch hits t i b
    
    # right branch of t i b: get ticks
    (note_t.outs[1],       trans_note.ins[0]),
    (trans_note.outs[7],   pack_note.ins[2]),    # ticks
    
    # left branch of t i b: pitch triggers pack
    (note_t.outs[0],       pack_note.ins[0]),
    
    (pack_note.outs[0],    note_sender.ins[0]),  # [pitch, vel, ticks]
    (bar_clock_js.outs[2], note_sender.ins[1]),  # bar context
    (note_sender.outs[0],  oscfmt_note.ins[0]),
    (oscfmt_note.outs[0],  udpsend.ins[0]),
)"""

content = content.replace(old_midi_in, new_midi_in)

with open('/Users/paultriana/creative_labs/MIDI-GPT/scripts/gen_midigpt_realtime_patch.py', 'w') as f:
    f.write(content)

print("Patching logic rewritten!")
