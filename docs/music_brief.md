# Music Generation Brief: EUC '26 Ride Edit

**Reference:** Tyler's Spotify playlist "EUC '26" (135 tracks, ~8.5 hrs)
https://open.spotify.com/playlist/6CswrJGOExV7GBcpJDeIZ7

**Footage:** Electric unicycle (EUC) rides around Colorado, filmed July 2026 on a GoPro Hero 13 and an Insta360 X4 Air. The edit mixes first-person and 360 reframed angles, with speed, carving, trails and group riding. The music needs to feel like motion.

---

## 1. What the playlist actually is

The playlist has four distinct groups of songs. Only the first two are the core sound; the other two are accents.

| # | Group | Share | Reference artists (for your understanding only, see rule 4) | What it contributes |
|---|---|---|---|---|
| A | **Heavy bass: drum & bass / electro house / dubstep** | ~35% (opens the playlist) | Pendulum, Knife Party (7 tracks), Netsky, Virtual Riot, Delta Heavy, Metrik, Dirtyphonics, Zeds Dead, Rezz | Speed, aggression, big drops. This is the "full send" sound. |
| B | **Upbeat swagger: funk-hop / electro-swing / nu-disco / hype** | ~45% (the long middle) | Kali J & LiTTiE (many tracks), ZOI, ittibitti, DARKMINDS, Ivy States, C.U.T., Pigeon John, Norman, Sly & the Family Stallone, Jamiroquai, Earth Wind & Fire | Playful confidence. Brass stabs, funky breakbeats, claps, vocal chops, short punchy tracks around 2 to 3 minutes. This is the "cruising and showing off" sound. |
| C | **Indie / alt electronic, melodic** | ~12% | Empire Of The Sun, ODESZA, Phantogram, Tame Impala, Passion Pit, Parcels, Roosevelt, Crooked Colours, Ghostland Observatory, Bakermat | Open, sunny, scenic. Synth arps, warm pads, a sense of space. Good for landscape and golden-hour shots. |
| D | **Raw blues** | ~3% | Bobby "Blue" Bland, R.L. Burnside, Howlin' Wolf, MHE/Longo "Thrill Is Gone" remix | Gritty, swaggering, a bit dirty. Use this as one accent (an intro, a tension beat, a breakdown), never a whole track. |

**Overall arc of the playlist:** it opens aggressive, settles into a long funky, cocky groove, and closes on liquid drum & bass (Netsky) and a light house track (Bakermat). Mirror that arc in the edit.

**Common threads across all groups:**
- High energy and forward momentum. There's almost nothing slow or sad.
- Tracks are short and punchy, mostly 2 to 4 minutes. Get to the hook fast and skip long intros.
- Plenty of attitude and humor. It's fun and a bit cocky, not epic-cinematic or emotional.
- A heavy low end in group A, and groove in the low end in group B.
- Vocals are mostly chopped or hype-shout style. There aren't many sung ballads.

---

## 2. Target specs for generated music

| Use in edit | Style | BPM | Key ingredients |
|---|---|---|---|
| High-speed runs, downhill, overtakes | Drum & bass / neurofunk | 170–176 | Reese bass, rolling breaks, a big drop on a speed moment, rave stabs |
| Hard-hitting montage, tricks, crashes | Electro house / complextro / dubstep | 126–130 (house) or 140–150 (dubstep/riddim halftime) | Distorted saw leads, wobble bass, build then drop, sirens and FX risers |
| Cruising, group riding, city or trail flow | Funk-hop / electro-swing / nu-disco | 100–124 | Brass stabs, slap or funk bass, breakbeats, claps, chopped vocal "hey!" hits, record scratches |
| Scenic, golden hour, mountain views | Indie electronic / melodic house | 110–124 | Warm analog synths, arps, airy pads, a gentle four-on-the-floor beat |
| Intro, tension, or a breakdown | Gritty blues-electronic hybrid | 85–100 | Dirty slide or electric blues guitar riff over a heavy beat, foot-stomp percussion |
| Outro / credits | Liquid drum & bass | 172–174 | Rolling but uplifting, melodic pads, soulful vocal chops |

---

## 3. Structure and edit fit

- **Cut to the beat.** Make tracks with clear 4- and 8-bar phrases so cuts land on downbeats. Put a clean drop or downbeat where the ride peaks.
- **Hook fast.** The first hit should come within 5 to 10 seconds. Skip ambient intros longer than about 8 bars.
- **Stems or instrumentals.** Give instrumentals at minimum. Stems (drums, bass, music, vocals) are better, so Tyler can duck the music under rider audio and wind in DaVinci Resolve.
- **Leave room for real audio.** The GoPro audio and any Cardo intercom chatter may be used. Keep the 1–4 kHz range from getting too crowded in quieter sections.
- **Tie length to the footage.** Ask Tyler for the final edit length. If he hasn't decided, plan for:
  - a ~70-second short (the Jul 25 17:25 Insta360 + GoPro sync clip): one track in group A or B, a drop at ~0:15, ending clean.
  - a ~10-minute segment (the Jul 25 17:35 sync pair): 3 to 4 tracks that follow the arc from section 1, B → A → C (or blues intro → B → A → C → liquid DnB outro).
- **Loudness:** master to about −14 LUFS integrated for YouTube, with true peak at or below −1 dBTP.

---

## 4. Rules

1. **Make original music, not copies.** The artists above show you the vibe. Don't name them in generation prompts, and don't reproduce their melodies, hooks, vocal lines, signature sounds or lyrics. Describe the style, tempo, instrumentation and mood instead. That's also what most music models respond to best.
2. **No sound-alike vocals.** If vocals are used, keep them short chops, generic hype shouts ("let's go", "hey!") or original lines.
3. **Check licensing.** Confirm that the generation tool's license allows monetized YouTube use before Tyler publishes.
4. **Stay away from:** trap or lo-fi chill (nearly absent from the playlist), orchestral "epic trailer" music, sad piano, country, and anything slow or moody for long stretches.

---

## 5. Example prompt shapes (adjust to the tool you're using)

- "High-energy drum and bass, 174 BPM, rolling breakbeats, gritty reese bass, rave stabs, big drop at 16 bars, instrumental, aggressive and fast, about 3 minutes."
- "Funky electro-swing hip-hop, 110 BPM, punchy brass stabs, slap bass, breakbeat drums, chopped 'hey!' vocal hits, confident and playful, instrumental, 2.5 minutes."
- "Sunny indie electronic, 118 BPM, warm analog synth arpeggios, airy pads, steady four-on-the-floor kick, uplifting, mountain road at golden hour, instrumental."
- "Gritty electric blues riff with slide guitar over a heavy stomp beat, 92 BPM, swaggering and dirty, 30-second intro sting."
- "Liquid drum and bass, 174 BPM, uplifting melodic pads, soulful wordless vocal chops, smooth rolling bass, outro feel, 2 minutes."
