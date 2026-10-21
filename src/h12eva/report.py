"""Plain-text report writer."""

import datetime as _dt
import textwrap

from h12eva import config as cfg

WIDTH = 78


def fmt_years(value):
    """Format a return period; infinite means beyond the fitted endpoint."""
    value = float(value)
    if value == float("inf"):
        return "infinite"
    if value > 1.0e6:
        return "> 1,000,000"
    return f"{value:,.0f}"


class Report:
    """Collect lines and write a formatted plain-text report."""

    def __init__(self, name, title):
        self.name = name
        self.lines = [
            "=" * WIDTH,
            title.upper(),
            "Hoti Cave stalagmite H12 d18O, northern Oman",
            "Sandy Hardian Susanto Herho (NIM 32426003)",
            f"Generated {_dt.date.today().isoformat()}",
            "=" * WIDTH,
            "",
        ]

    def section(self, heading):
        """Start a new section."""
        self.lines += ["", heading, "-" * len(heading)]

    def para(self, text):
        """Add a wrapped paragraph."""
        self.lines += textwrap.wrap(" ".join(text.split()), WIDTH) + [""]

    def kv(self, key, value, width=44):
        """Add an aligned key-value line."""
        self.lines.append(f"{key:<{width}s} {value}")

    def table(self, frame, floatfmt="{:.3f}"):
        """Add a DataFrame as a fixed-width table."""
        shown = frame.copy()
        for col in shown.columns:
            if col == "n":
                shown[col] = shown[col].map(lambda v: f"{v:.0f}")
        text = shown.to_string(
            float_format=lambda v: floatfmt.format(v), index=True)
        self.lines += text.splitlines() + [""]

    def write(self):
        """Write the report to reports/<name>.txt."""
        cfg.REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = cfg.REPORT_DIR / f"{self.name}.txt"
        text = "\n".join(self.lines).rstrip() + "\n"
        text = text.replace("infinite yr", "infinite (event beyond the "
                            "fitted upper endpoint)")
        path.write_text(text, encoding="utf-8")
        return path
