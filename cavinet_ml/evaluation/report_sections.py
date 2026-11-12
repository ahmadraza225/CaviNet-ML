"""Add or replace parts of docs/EVALUATION_REPORT.md written by later commands
(`heatmap-check`, `fusion`), so that they survive and can be re-run."""


def set_section(text: str, heading: str, section: str) -> str:
    """Insert or replace the section that starts with `heading`, before "## Methods" if there
    is one, otherwise at the end."""
    section = section.rstrip("\n") + "\n"
    if heading in text:
        start = text.index(heading)
        end = text.find("\n## ", start + len(heading))
        return text[:start] + section + ("\n" + text[end + 1 :] if end != -1 else "")
    if "\n## Methods" in text:
        at = text.index("\n## Methods") + 1
        return text[:at] + section + "\n" + text[at:]
    if text.strip():
        return text.rstrip("\n") + "\n\n" + section
    return section


def set_summary_row(text: str, key: str, row: str, *, after: str) -> str:
    """Insert or replace the "## Summary" table row starting with `key`, placed after the row
    starting with `after` (or at the end of the table). No summary table: unchanged."""
    lines = text.split("\n")
    try:
        start = lines.index("## Summary")
    except ValueError:
        return text
    table = [i for i in range(start + 1, len(lines)) if lines[i].startswith("|")]
    if not table:
        return text
    end = table[0]
    while end + 1 < len(lines) and lines[end + 1].startswith("|"):
        end += 1
    rows = range(table[0], end + 1)
    existing = [i for i in rows if lines[i].startswith(key)]
    if existing:
        lines[existing[0]] = row
        return "\n".join(lines)
    anchor = [i for i in rows if lines[i].startswith(after)]
    at = anchor[0] + 1 if anchor else end + 1
    lines.insert(at, row)
    return "\n".join(lines)
