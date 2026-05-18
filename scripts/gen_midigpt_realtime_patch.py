#!/usr/bin/env python3
"""
Generate the MIDI-GPT realtime Max MSP patch.

Output:
  patches/midigpt_realtime/midigpt_realtime.maxpat
  patches/midigpt_realtime/bar_clock.js
  patches/midigpt_realtime/note_sender.js
  patches/midigpt_realtime/note_player.js

Run:
  cd /Users/paultriana/creative_labs/MIDI-GPT
  .venv/bin/python3 scripts/gen_midigpt_realtime_patch.py

Requirements in Max: None (uses native Max objects like 'route').
"""

from pathlib import Path
from maxpylang import MaxPatch

ROOT    = Path(__file__).parent.parent
OUT_DIR = ROOT / "patches" / "midigpt_realtime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# bar_clock.js
#
# Fires at each bar boundary (receives bang from [timepoint 1 0 0]).
#   inlet 0 : bang  — complete current bar, advance counter
#   inlet 1 : "start" message — reset + enable
#   inlet 2 : "stop"  message — disable
#   outlet 0 : [bar_idx ts_num ts_den]           → prepend /midigpt/bar/end
#   outlet 1 : [next_bar_idx bpm ts_num ts_den]  → note_player (flush signal)
#   outlet 2 : [bar_start_ticks bar_ticks bar_idx] → note_sender (bar context)
# ─────────────────────────────────────────────────────────────────────────────
BAR_CLOCK_JS = """\
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
"""

# ─────────────────────────────────────────────────────────────────────────────
# note_sender.js
#
# Converts MIDI note-on/off pairs into normalized /midigpt/note args.
#   inlet 0 : [pitch vel] list — note events from [pack i i] ← [notein]
#   inlet 1 : [bar_start_ticks bar_ticks bar_idx] — bar context from bar_clock
#   outlet 0 : [track_id pitch vel onset dur bar_idx] → prepend /midigpt/note
# ─────────────────────────────────────────────────────────────────────────────
NOTE_SENDER_JS = """\
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
"""

# ─────────────────────────────────────────────────────────────────────────────
# note_player.js
#
# Buffers generated notes per bar_index, schedules MIDI playback via Task.
#   inlet 0 : /generated/open  list [track_id bar_idx note_count]
#   inlet 1 : /generated/note  list [track_id bar_idx pitch vel onset dur]
#   inlet 2 : /generated/close list [track_id bar_idx]
#   inlet 3 : flush list [next_bar_idx bpm ts_num ts_den] from bar_clock
#   outlet 0 : [pitch vel dur_ms] → [unpack i i i] → [makenote] → [noteout]
# ─────────────────────────────────────────────────────────────────────────────
NOTE_PLAYER_JS = """\
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
"""

# Write JS files to the output directory (maxpylang looks in cwd)
import os
os.chdir(OUT_DIR)

for fname, src in [
    ("bar_clock.js",   BAR_CLOCK_JS),
    ("note_sender.js", NOTE_SENDER_JS),
    ("note_player.js", NOTE_PLAYER_JS),
]:
    (OUT_DIR / fname).write_text(src)
    print(f"  wrote {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# Max Patch
# ─────────────────────────────────────────────────────────────────────────────
patch = MaxPatch()

L = 40    # left margin
T = 40    # top margin
R = 65    # row height
C = 240   # column width

# ── Network ───────────────────────────────────────────────────────────────────
udpsend = patch.place("udpsend 127.0.0.1 7400", starting_pos=[L, T])[0]
udprecv = patch.place("udpreceive 7401",         starting_pos=[L + C*3, T])[0]

oscroute = patch.place(
    "route "
    "/midigpt/session/ready /midigpt/session/started /midigpt/session/stopped "
    "/midigpt/generated/open /midigpt/generated/note /midigpt/generated/close "
    "/midigpt/generated/features /midigpt/status /midigpt/error",
    starting_pos=[L + C*3, T + R]
)[0]
patch.connect((udprecv.outs[0], oscroute.ins[0]))

# Status + error (print to Max console)
status_print = patch.place("print midigpt-status",
                            starting_pos=[L + C*3,       T + R*2 + 10])[0]
err_print    = patch.place("print midigpt-error",
                            starting_pos=[L + C*3 + 200, T + R*2 + 10])[0]
patch.connect(
    (oscroute.outs[7], status_print.ins[0]),   # /midigpt/status  (outlet 7)
    (oscroute.outs[8], err_print.ins[0]),      # /midigpt/error   (outlet 8)
)

# ── Session Control ───────────────────────────────────────────────────────────
y0 = T + R*4
patch.place("comment ─── Session ───────────────────────────────────────────",
            starting_pos=[L, y0 - 22])[0]

btn_init  = patch.place("button", starting_pos=[L,       y0])[0]
btn_start = patch.place("button", starting_pos=[L + 70,  y0])[0]
btn_stop  = patch.place("button", starting_pos=[L + 140, y0])[0]
patch.place("comment init",  starting_pos=[L,       y0 + 26])[0]
patch.place("comment start", starting_pos=[L + 70,  y0 + 26])[0]
patch.place("comment stop",  starting_pos=[L + 140, y0 + 26])[0]

msg_init  = patch.place("message /midigpt/session/init live-set",
                        starting_pos=[L,       y0 + R])[0]
msg_start = patch.place("message /midigpt/session/start",
                        starting_pos=[L + 70,  y0 + R])[0]
msg_stop  = patch.place("message /midigpt/session/stop",
                        starting_pos=[L + 140, y0 + R])[0]

patch.connect(
    (btn_init.outs[0],  msg_init.ins[0]),
    (btn_start.outs[0], msg_start.ins[0]),
    (btn_stop.outs[0],  msg_stop.ins[0]),
    (msg_init.outs[0],  udpsend.ins[0]),
    (msg_start.outs[0], udpsend.ins[0]),
    (msg_stop.outs[0],  udpsend.ins[0]),
)

# ── Track Creation ────────────────────────────────────────────────────────────
y1 = y0 + R*3
patch.place("comment ─── Track Setup ──────────────────────────────────────",
            starting_pos=[L, y1 - 22])[0]

btn_t0 = patch.place("button", starting_pos=[L,     y1])[0]
btn_t1 = patch.place("button", starting_pos=[L + C, y1])[0]
msg_t0 = patch.place("message /midigpt/track/create 0 0 10 0",
                     starting_pos=[L,     y1 + 28])[0]
msg_t1 = patch.place("message /midigpt/track/create 1 0 10 1",
                     starting_pos=[L + C, y1 + 28])[0]
patch.place("comment track0: piano conditioning",
            starting_pos=[L,     y1 + R])[0]
patch.place("comment track1: piano AI agent",
            starting_pos=[L + C, y1 + R])[0]

patch.connect(
    (btn_t0.outs[0], msg_t0.ins[0]),
    (btn_t1.outs[0], msg_t1.ins[0]),
    (msg_t0.outs[0], udpsend.ins[0]),
    (msg_t1.outs[0], udpsend.ins[0]),
)

# ── Bar Clock ─────────────────────────────────────────────────────────────────
y2 = y1 + R*3
patch.place("comment ─── Bar Clock ─────────────────────────────────────────",
            starting_pos=[L, y2 - 22])[0]

# Replace timepoint with a loopable, tempo-synced metro + click
metro_obj   = patch.place("metro",                           starting_pos=[L, y2])[0]
metro_ui    = patch.place("button",                          starting_pos=[L-40, y2+R])[0] # visible click 

msg_m_on    = patch.place("message 1",                       starting_pos=[L+30, y2-R])[0]
msg_m_off   = patch.place("message 0",                       starting_pos=[L+80, y2-R])[0]

loadbang_m  = patch.place("loadbang",                        starting_pos=[L+130, y2-R])[0]
msg_metro   = patch.place("message interval 1n, active 1",   starting_pos=[L+130, y2-R//2])[0]

click_obj   = patch.place("click~",                          starting_pos=[L-100, y2+R*2])[0]
res_obj     = patch.place("reson~ 1. 1500 20",               starting_pos=[L-100, y2+R*3])[0]
vol_obj     = patch.place("*~ 0.5",                          starting_pos=[L-100, y2+R*4])[0]
dac_obj     = patch.place("dac~ 1 2",                        starting_pos=[L-100, y2+R*5])[0]

patch.connect(
    (loadbang_m.outs[0], msg_metro.ins[0]),
    (msg_metro.outs[0], metro_obj.ins[0]),
    (btn_start.outs[0], msg_m_on.ins[0]),
    (btn_stop.outs[0], msg_m_off.ins[0]),
    (msg_m_on.outs[0], metro_obj.ins[0]),
    (msg_m_off.outs[0], metro_obj.ins[0]),
    (metro_obj.outs[0], metro_ui.ins[0]),
    (metro_obj.outs[0], click_obj.ins[0]),
    (click_obj.outs[0], res_obj.ins[0]),
    (res_obj.outs[0], vol_obj.ins[0]),
    (vol_obj.outs[0], dac_obj.ins[0]),
    (vol_obj.outs[0], dac_obj.ins[1])
)

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
    (metro_obj.outs[0], trans_bc.ins[0]),          # metro hits transport
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
msg_bc_stop  = patch.place("message stop",  starting_pos=[L + 300, y2 - 25])[0]

patch.connect(
    # Start wiring
    (btn_start.outs[0],     trans_start.ins[0]),   # start button hits transport
    (trans_start.outs[7],   pack_start.ins[3]),
    (trans_start.outs[5],   unpack_tss.ins[0]),
    (unpack_tss.outs[1],    pack_start.ins[2]),
    (unpack_tss.outs[0],    pack_start.ins[1]),
    (trans_start.outs[4],   pack_start.ins[0]),
    (pack_start.outs[0],    bar_clock_js.ins[1]),  # list to inlet 1
    
    # Stop wiring
    (btn_stop.outs[0],      msg_bc_stop.ins[0]),
    (msg_bc_stop.outs[0],   bar_clock_js.ins[2]),
)

# ── MIDI Input ────────────────────────────────────────────────────────────────
y3 = y2 + R*4
patch.place("comment MIDI Input ch1 to Track 0 ─────────────────────",
            starting_pos=[L, y3 - 22])[0]

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
)

# ── Generated Note Playback ───────────────────────────────────────────────────
x_play = L + C*3
y_play = T + R*4
patch.place("comment ─── Generated Note Playback (ch 2) ───────────────────",
            starting_pos=[x_play, y_play - 22])[0]

note_player  = patch.place("js note_player.js",  starting_pos=[x_play, y_play])[0]
unpack_gen   = patch.place("unpack i i i",        starting_pos=[x_play, y_play + R])[0]
makenote_obj = patch.place("makenote",            starting_pos=[x_play, y_play + R*2])[0]
noteout_obj  = patch.place("noteout 2 1",         starting_pos=[x_play, y_play + R*3])[0]

# oscroute outlets: 0=ready 1=started 2=stopped 3=open 4=note 5=close 6=feat 7=status 8=error
patch.connect(
    (oscroute.outs[3],     note_player.ins[0]),   # /generated/open
    (oscroute.outs[4],     note_player.ins[1]),   # /generated/note
    (oscroute.outs[5],     note_player.ins[2]),   # /generated/close
    (bar_clock_js.outs[1], note_player.ins[3]),   # flush [bar_idx bpm ts_num ts_den]
    (note_player.outs[0],  unpack_gen.ins[0]),    # [pitch vel dur_ms]
    (unpack_gen.outs[2],   makenote_obj.ins[2]),  # dur_ms (set before pitch to avoid early trigger)
    (unpack_gen.outs[1],   makenote_obj.ins[1]),  # vel
    (unpack_gen.outs[0],   makenote_obj.ins[0]),  # pitch (triggers note)
    (makenote_obj.outs[0], noteout_obj.ins[0]),   # pitch out
    (makenote_obj.outs[1], noteout_obj.ins[1]),   # velocity out
)

# ── Parameters ────────────────────────────────────────────────────────────────
y_param = y1 + R*3
x_param = L + C*2
patch.place("comment ─── Parameters ────────────────────────────────────────",
            starting_pos=[x_param, y_param - 22])[0]

# Temperature (0.5 – 2.0)
patch.place("comment temperature 0.5-2.0", starting_pos=[x_param, y_param])[0]
temp_sl  = patch.place("slider",               starting_pos=[x_param, y_param + 22])[0]
temp_sc  = patch.place("* 0.015",              starting_pos=[x_param, y_param + 82])[0]
temp_off = patch.place("+ 0.5",                starting_pos=[x_param, y_param + 82 + R])[0]
temp_msg = patch.place("prepend /midigpt/param/set temperature",
                       starting_pos=[x_param, y_param + 82 + R*2])[0]
patch.connect(
    (temp_sl.outs[0],  temp_sc.ins[0]),
    (temp_sc.outs[0],  temp_off.ins[0]),
    (temp_off.outs[0], temp_msg.ins[0]),
    (temp_msg.outs[0], udpsend.ins[0]),
)

# Lookahead bars (1 – 8)
patch.place("comment lookahead 1-8", starting_pos=[x_param + C//2, y_param])[0]
la_num  = patch.place("number",   starting_pos=[x_param + C//2, y_param + 22])[0]
la_clip = patch.place("clip 1 8", starting_pos=[x_param + C//2, y_param + 82])[0]
la_msg  = patch.place("prepend /midigpt/param/set lookahead_bars",
                      starting_pos=[x_param + C//2, y_param + 82 + R])[0]
patch.connect(
    (la_num.outs[0],  la_clip.ins[0]),
    (la_clip.outs[0], la_msg.ins[0]),
    (la_msg.outs[0],  udpsend.ins[0]),
)

# Tempo and TimeSig (Control Transport)
y_trs = y_param
x_trs = x_param + C

patch.place("comment BPM", starting_pos=[x_trs, y_trs])[0]
tempo_num = patch.place("number", starting_pos=[x_trs, y_trs + 22])[0]
tempo_msg = patch.place("message tempo $1", starting_pos=[x_trs, y_trs + 82])[0]

patch.place("comment TimeSig", starting_pos=[x_trs + 80, y_trs])[0]
ts_n_num = patch.place("number", starting_pos=[x_trs + 80, y_trs + 22])[0]
ts_d_num = patch.place("number", starting_pos=[x_trs + 140, y_trs + 22])[0]
ts_pak   = patch.place("pak i i", starting_pos=[x_trs + 80, y_trs + 82])[0]
ts_msg   = patch.place("message timesig $1 $2", starting_pos=[x_trs + 80, y_trs + 82 + R])[0]

patch.connect(
    (tempo_num.outs[0], tempo_msg.ins[0]),
    (tempo_msg.outs[0], trans_bc.ins[0]),
    
    (ts_n_num.outs[0], ts_pak.ins[0]),
    (ts_d_num.outs[0], ts_pak.ins[1]),
    (ts_pak.outs[0], ts_msg.ins[0]),
    (ts_msg.outs[0], trans_bc.ins[0]),
)

# ── Save ──────────────────────────────────────────────────────────────────────
patch_path = str(OUT_DIR / "midigpt_realtime.maxpat")
patch.check()
patch.save(patch_path)
print(f"\nSaved → {patch_path}")
print("Keep all JS files in the same folder as the .maxpat")
print("Note: Patch generation successful. We use native 'route' and 'prepend' objects so NO CNMAT externals are required.")
