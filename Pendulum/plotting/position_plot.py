"""Continuous angle plots: initial hanging-down position near zero degrees."""

from Pendulum.paths import APP_ROOT

import textwrap

import numpy as np


def continuous_angles(angle_deg):
    """Unwrap rotations and shift upright=0 to hanging-down=0.

    Choose the initial revolution nearest hanging down. Later revolutions
    remain visible, so the plot never cuts or folds a continuous trajectory.
    Logged measurements keep their original upright-zero convention.
    """
    angles = np.asarray(angle_deg, dtype=float)
    shifted = np.rad2deg(np.unwrap(np.deg2rad(angles))) + 180.0
    return shifted - 360.0 * np.floor((shifted[0] + 180.0) / 360.0)


def wrapped_samples(times, angles):
    """Wrap shifted angles to +/-180, restarting at the opposite plot edge."""
    wrapped = (np.asarray(angles) + 180.0) % 360.0 - 180.0
    plot_times, plot_angles = [times[0]], [wrapped[0]]
    for index in range(1, len(wrapped)):
        previous, current = wrapped[index - 1:index + 1]
        difference = current - previous
        if abs(difference) > 180.0:
            edge = 180.0 if difference < 0 else -180.0
            delta = difference + (360.0 if difference < 0 else -360.0)
            fraction = (edge - previous) / delta
            crossing_time = times[index - 1] + fraction * (times[index] - times[index - 1])
            plot_times.extend([crossing_time, np.nan, crossing_time])
            plot_angles.extend([edge, np.nan, -edge])
        plot_times.append(times[index])
        plot_angles.append(current)
    return np.asarray(plot_times), np.asarray(plot_angles)


def position_figure(times, angle_deg, title, *, wrap=False):
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 20, 'axes.titlesize': 24, 'axes.labelsize': 23, 'xtick.labelsize': 20, 'ytick.labelsize': 20, 'legend.fontsize': 18, 'figure.titlesize': 26})

    angles = continuous_angles(angle_deg)
    lower = 180.0 * np.floor(min(0.0, angles.min()) / 180.0)
    upper = 180.0 * np.ceil(max(180.0, angles.max()) / 180.0)
    plot_times = times
    if wrap:
        plot_times, angles = wrapped_samples(times, angles)
        lower, upper = -180.0, 180.0
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(plot_times, angles, linewidth=1.3, label="Pendulum angle")
    for level in np.arange(lower, upper + 1.0, 180.0):
        upright = int(round(level / 180.0)) % 2 != 0
        ax.axhline(level, color="black" if upright else "gray",
                   linestyle="--" if upright else ":", linewidth=0.8,
                   label=(("Upright (±180°)" if upright else "Hanging down (0°)")
                          if wrap else
                          ("Upright (180° + 360° × n)" if upright else
                           "Hanging down (360° × n)"))
                   if level == lower or level == lower + 180.0 else None)
    ax.set(xlabel="Time (s)", ylabel=("Angle from hanging down (degrees)" if wrap else
                                     "Continuous angle from hanging down (degrees)"),
           title=title, xlim=(times[0], times[-1]),
           ylim=(lower - 15.0, upper + 15.0),
           yticks=np.arange(lower, upper + 1.0, 90.0 if wrap else 180.0))
    ax.grid(alpha=0.25)
    ax.set_ylabel(textwrap.fill(ax.get_ylabel(), width=28))
    ax.set_title(textwrap.fill(title, width=48))
    ax.legend()
    fig.tight_layout()
    return fig
