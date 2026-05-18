#!/usr/bin/env python3
import sys

with open('/Users/paultriana/creative_labs/MIDI-GPT/scripts/gen_midigpt_realtime_patch.py', 'r') as f:
    content = f.read()

# 1. Update Parameters Setup (change temperature to prepend, add tempo and timesig)
old_params = """\
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
temp_msg = patch.place("message /midigpt/param/set temperature $1",
                       starting_pos=[x_param, y_param + 82 + R*2])[0]
patch.connect(
    (temp_sl.outs[0],  temp_sc.ins[0]),
    (temp_sc.outs[0],  temp_off.ins[0]),
    (temp_off.outs[0], temp_msg.ins[0]),
    (temp_msg.outs[0], udpsend.ins[0]),
)

# Lookahead bars (1 – 8)
patch.place("comment lookahead bars 1-8", starting_pos=[x_param + C//2, y_param])[0]
la_num  = patch.place("number",   starting_pos=[x_param + C//2, y_param + 22])[0]
la_clip = patch.place("clip 1 8", starting_pos=[x_param + C//2, y_param + 82])[0]
la_msg  = patch.place("message /midigpt/param/set lookahead_bars $1",
                      starting_pos=[x_param + C//2, y_param + 82 + R])[0]
patch.connect(
    (la_num.outs[0],  la_clip.ins[0]),
    (la_clip.outs[0], la_msg.ins[0]),
    (la_msg.outs[0],  udpsend.ins[0]),
)"""

new_params = """\
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
)"""

content = content.replace(old_params, new_params)


# 2. Update Bar Clock to use Metronome
old_bar_clock_patch = """\
timepoint_obj = patch.place("timepoint 1 0 0",   starting_pos=[L,       y2])[0]

# Bar Clock Transport Trigger
trans_bc      = patch.place("transport",         starting_pos=[L+100, y2])[0]"""

new_bar_clock_patch = """\
# Replace timepoint with a loopable, tempo-synced metro + click
metro_obj   = patch.place("metro 1n @active 1 @autostart 1", starting_pos=[L, y2])[0]
msg_click   = patch.place("message 37",                      starting_pos=[L-100, y2+R*2])[0]
makenote_m  = patch.place("makenote 100 100",                starting_pos=[L-100, y2+R*3])[0]
noteout_m   = patch.place("noteout 10 1",                    starting_pos=[L-100, y2+R*4])[0]

patch.connect(
    (metro_obj.outs[0], msg_click.ins[0]),
    (msg_click.outs[0], makenote_m.ins[0]),
    (makenote_m.outs[0], noteout_m.ins[0]),
    (makenote_m.outs[1], noteout_m.ins[1])
)

# Bar Clock Transport Trigger
trans_bc      = patch.place("transport",         starting_pos=[L+100, y2])[0]"""

content = content.replace(old_bar_clock_patch, new_bar_clock_patch)

# 3. Replace timepoint_obj connection with metro_obj connection
content = content.replace(
    "(timepoint_obj.outs[0], trans_bc.ins[0]),      # timepoint hits transport",
    "(metro_obj.outs[0], trans_bc.ins[0]),          # metro hits transport"
)

with open('/Users/paultriana/creative_labs/MIDI-GPT/scripts/gen_midigpt_realtime_patch.py', 'w') as f:
    f.write(content)

print("Params, metro, and click track completely refactored!")
