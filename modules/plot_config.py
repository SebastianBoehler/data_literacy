#!/usr/bin/env python3
"""
Central Plot Configuration

Single source of truth for all matplotlib styling across the project.
Import and call `apply_style()` at the start of any plotting script.

Usage:
    from modules.plot_config import apply_style, COLORS, STYLE
    apply_style()
"""

import matplotlib.pyplot as plt

# =============================================================================
# COLOR PALETTE (colorblind-friendly)
# =============================================================================
COLORS = {
    # Primary colors
    "primary": "#4878d0",       # muted blue
    "secondary": "#ee854a",     # muted orange
    "success": "#6acc64",       # muted green
    "danger": "#d65f5f",        # muted red
    
    # Semantic colors for delays
    "on_time": "#00AA00",       # green
    "slight_delay": "#CCCC00",  # yellow
    "moderate_delay": "#FF8800", # orange
    "significant_delay": "#FF0000", # red
    
    # Chart colors
    "steelblue": "steelblue",
    "darkred": "darkred",
    "darkorange": "darkorange",
    "forestgreen": "forestgreen",
    "firebrick": "firebrick",
    
    # Neutral
    "gray": "gray",
    "lightgray": "#cccccc",
}

# Period comparison colors
PERIOD_COLORS = {
    "pre": "#d65f5f",   # red for before
    "post": "#6acc64", # green for after
    "all": "#4878d0",  # blue for all data
}

# =============================================================================
# STYLE SETTINGS
# =============================================================================
STYLE = {
    # Font sizes
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    
    # Figure settings
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    
    # Font family
    "font.family": "serif",
    
    # Grid
    "axes.grid": False,
    "grid.alpha": 0.3,
    
    # Legend
    "legend.framealpha": 0.9,
    "legend.edgecolor": "none",
    
    # Lines
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    
    # Savefig
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
}

# =============================================================================
# TUEPLOTS INTEGRATION (optional academic styling)
# =============================================================================
def _get_tueplots_style():
    """Try to load tueplots styling for academic papers."""
    try:
        from tueplots import bundles, axes
        return {
            **bundles.icml2024(usetex=False, family="serif"),
            **axes.lines(),
        }
    except ImportError:
        return {}

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def apply_style(use_tueplots: bool = True):
    """
    Apply the unified plot style.
    
    Args:
        use_tueplots: If True, try to use tueplots for academic styling.
                      Falls back gracefully if not installed.
    
    Usage:
        from modules.plot_config import apply_style
        apply_style()
    """
    # Start with tueplots if available and requested
    if use_tueplots:
        tueplots_style = _get_tueplots_style()
        if tueplots_style:
            plt.rcParams.update(tueplots_style)
    
    # Apply our custom style (overrides tueplots where specified)
    plt.rcParams.update(STYLE)


def get_period_color(period: str) -> str:
    """Get the color for a specific period (pre/post/all)."""
    return PERIOD_COLORS.get(period, COLORS["primary"])


# =============================================================================
# COMMON FIGURE SIZES
# =============================================================================
FIGSIZE = {
    "single": (8, 5),
    "wide": (10, 5),
    "double": (12, 5),
    "square": (6, 6),
    "combined_2x2": (14, 10),
    "combined_1x2": (14, 5),
}


# =============================================================================
# CONSTANTS
# =============================================================================
LATE_THRESHOLD = 2  # minutes - standard definition of 'late'
SCHEDULE_CHANGE_DATE = "2025-12-14"


if __name__ == "__main__":
    # Demo: show the style settings
    print("Plot Configuration Module")
    print("=" * 50)
    print("\nStyle settings:")
    for key, value in STYLE.items():
        print(f"  {key}: {value}")
    print("\nColors:")
    for key, value in COLORS.items():
        print(f"  {key}: {value}")
