"""
export_fcpxml.py - Write timeline.fcpxml so the edit can be opened in
DaVinci Resolve (File > Import > Timeline).

Plain-language summary:
  * FCPXML is an interchange format Resolve understands. We list every source
    file once (resources) and then every shot in order (spine), each pointing
    at the source and the in/out times.
  * 360 reframing can't be expressed in FCPXML; those clips import as the raw
    equirectangular video. The yaw/pitch/fov we used is written into each
    clip's note so Tyler can recreate the angle with Resolve's 360 tools.
  * Times are written as exact frame counts at the plan's frame rate.

Run:  python tools/export_fcpxml.py <ride-id>
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import pathname2url
from xml.sax.saxutils import escape

from common import output_dir, read_json, run_log, work_dir

FRAME_DUR = {23.976: (1001, 24000), 24: (1, 24), 25: (1, 25), 29.97: (1001, 30000),
             30: (1, 30), 50: (1, 50), 59.94: (1001, 60000), 60: (1, 60)}


def main(ride_id: str) -> Path:
    plan = read_json(work_dir(ride_id) / "edit_plan.json")
    fps = plan["fps"]
    num, den = FRAME_DUR.get(fps, (1, int(round(fps))))

    def rt(seconds: float) -> str:
        frames = int(round(seconds * den / num))
        return f"{frames * num}/{den}s"

    w, h = 1920, 1080
    resources = [f'<format id="r1" name="FFVideoFormat{h}p{str(fps).replace(".", "")}" '
                 f'frameDuration="{num}/{den}s" width="{w}" height="{h}"/>']
    asset_ids = {}
    for i, s in enumerate(plan["sources"], start=2):
        asset_ids[s["path"]] = f"r{i}"
        src = "file:" + pathname2url(str(Path(s["path"]).resolve()))
        resources.append(
            f'<asset id="r{i}" name="{escape(s["name"])}" start="0s" duration="{rt(s["duration"])}" '
            f'hasVideo="1" hasAudio="{1 if s.get("has_audio") else 0}" format="r1">'
            f'<media-rep kind="original-media" src="{escape(src)}"/></asset>')

    clips = []
    for shot in plan["long"]["segments"]:
        note = "; ".join(shot["reasons"])
        if shot.get("v360"):
            v = shot["v360"]
            note += f" | 360 reframe yaw={v['yaw']} pitch={v['pitch']} fov={v['fov']} ({shot['view']})"
        clips.append(
            f'<asset-clip ref="{asset_ids[shot["source"]]}" offset="{rt(shot["out_in"])}" '
            f'name="{escape(shot["source_name"])}" start="{rt(shot["source_in"])}" '
            f'duration="{rt(shot["duration"])}" format="r1"><note>{escape(note)}</note></asset-clip>')

    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.9">
  <resources>
    {chr(10).join('    ' + r for r in resources).strip()}
  </resources>
  <library>
    <event name="{escape(ride_id)}">
      <project name="{escape(ride_id)} long">
        <sequence format="r1" duration="{rt(plan['long']['duration'])}" tcStart="0s" tcFormat="NDF">
          <spine>
            {chr(10).join('            ' + c for c in clips).strip()}
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
"""
    out = output_dir(ride_id) / "timeline.fcpxml"
    out.write_text(xml, encoding="utf-8")
    run_log(ride_id, f"timeline.fcpxml: {len(clips)} clips")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: export_fcpxml.py <ride-id>")
    main(sys.argv[1])
