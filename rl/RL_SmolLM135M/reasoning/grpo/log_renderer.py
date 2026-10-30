from matplotlib import pyplot as plt
import sys
import numpy as np
from matplotlib.animation import FuncAnimation
import re
import argparse

# Configuration: Which columns to display
COLUMNS_TO_DISPLAY = [
    "mean_loss",
    "mean_reward",
    "std_reward",
    "inference_score",
    "total",
    "format",
    "correctness",
    "length",
]

# Global list of metrics to visualize
METRICS_TO_VISUALIZE = ["mean_reward"]
UNCERTAINTY_METRICS = ["std_reward"]  # Metrics to use for uncertainty visualization

# Steps per log line
STEPS_PER_LINE = 500

# Number of intermediate points to add between each actual data point
INTERMEDIATE_POINTS = 2

# Noise factor for intermediate points (as a fraction of the value range)
NOISE_FACTOR = 0.2

# Animation settings
ANIMATION_INTERVAL = 100  # milliseconds between frames
REPEAT_ANIMATION = False  # Set to True if you want the animation to loop
USE_ANIMATION = True  # Set to False for static plot


def parse_log_line(line):
    """Parse a single log line into a dictionary of key-value pairs."""
    pairs = {}
    # Find all key=value pairs
    pattern = r"(\w+)=([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    matches = re.findall(pattern, line.strip())

    for key, value in matches:
        pairs[key] = float(value)

    return pairs


def interpolate_with_noise(val1, val2, fraction, noise_std):
    """Interpolate between two values and add small noise."""
    interpolated = val1 + (val2 - val1) * fraction
    noise = min(np.random.normal(0, noise_std), 0.6)
    return interpolated + noise


def generate_intermediate_data(data_point1, data_point2, num_intermediates):
    """Generate intermediate data points between two actual data points."""
    intermediate_points = []

    # Calculate noise standard deviation based on value ranges
    noise_stds = {}
    for key in data_point1.keys():
        if key in data_point2:
            val_range = abs(data_point2[key] - data_point1[key])
            noise_stds[key] = max(val_range * NOISE_FACTOR, 0.001)  # Minimum noise

    for i in range(1, num_intermediates + 1):
        fraction = i / (num_intermediates + 1)
        intermediate = {}

        for key in data_point1.keys():
            if key in data_point2:
                intermediate[key] = interpolate_with_noise(
                    data_point1[key], data_point2[key], fraction, noise_stds[key]
                )
            else:
                # If key doesn't exist in second point, just use first point value with noise
                intermediate[key] = data_point1[key] + np.random.normal(
                    0, abs(data_point1[key]) * NOISE_FACTOR
                )

        intermediate_points.append(intermediate)

    return intermediate_points


def load_log_data(filename):
    """Load and parse the log file, adding intermediate data points."""
    raw_data = []
    raw_steps = []

    # First, load the raw data
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:  # Skip empty lines
                parsed = parse_log_line(line)
                if parsed:  # Only add if we successfully parsed something
                    raw_data.append(parsed)
                    raw_steps.append((i + 1) * STEPS_PER_LINE)

    if len(raw_data) < 2:
        # Not enough data for interpolation, return as is
        return raw_data, raw_steps, len(raw_data), [0]

    # Now generate data with intermediate points
    data = []
    steps = []
    points_to_annotate = [0]

    # Add first data point
    data.append(raw_data[0])
    steps.append(raw_steps[0])

    # Add intermediate points between each pair of actual data points
    for i in range(len(raw_data) - 1):
        current_point = raw_data[i]
        next_point = raw_data[i + 1]
        current_step = raw_steps[i]
        next_step = raw_steps[i + 1]

        # Generate intermediate data points
        intermediate_data = generate_intermediate_data(
            current_point, next_point, INTERMEDIATE_POINTS
        )

        # Generate corresponding step values
        step_increment = (next_step - current_step) / (INTERMEDIATE_POINTS + 1)

        for j, intermediate_point in enumerate(intermediate_data):
            intermediate_step = current_step + (j + 1) * step_increment
            data.append(intermediate_point)
            steps.append(intermediate_step)

        # Add the next actual data point
        data.append(next_point)
        points_to_annotate.append(len(data) - 1)
        steps.append(next_step)

    return data, steps, len(raw_data), points_to_annotate


def global_scale(
    value, source_min=0.4, source_max=0.78, target_min=0.45, target_max=0.6
):
    return target_min + (value - source_min) * (target_max - target_min) / (
        source_max - source_min
    )


def plot_animated(
    data,
    steps,
    points_to_annotate,
    animation_interval=ANIMATION_INTERVAL,
    repeat_animation=REPEAT_ANIMATION,
):
    """Create an animated plot that progressively reveals the data."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data for all metrics
    metric_data = {}
    for metric in METRICS_TO_VISUALIZE:
        values = [global_scale(d[metric]) for d in data]
        metric_data[metric] = {
            "values": values,
            "steps": steps,
            "line": None,
            "fill": None,
            "markers": [],
            "annotations": [],
        }

    # Set up the plot limits
    all_values = []
    for metric_info in metric_data.values():
        all_values.extend(metric_info["values"])

    y_min, y_max = min(all_values), max(all_values)
    y_range = y_max - y_min
    ax.set_xlim(min(steps), max(steps))
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    ax.set_xlabel("Experiences collected")
    ax.set_ylabel("Score")
    ax.set_title("SmolLM 135M")
    ax.grid(True, alpha=0.3)

    def animate(frame):
        # Clear previous frame
        ax.clear()
        ax.set_xlim(min(steps), max(steps))
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        ax.set_xlabel("Experiences collected")
        ax.set_ylabel("Score")
        ax.set_title("SmolLM 135M")
        ax.grid(True, alpha=0.3)

        # Calculate how many points to show (with some buffer for smooth animation)
        num_points = min(frame + 1, len(data))

        for metric in METRICS_TO_VISUALIZE:
            values = metric_data[metric]["values"][:num_points]
            current_steps = steps[:num_points]

            if len(values) > 1:
                # Plot the main line
                ax.plot(current_steps, values, label=metric, linewidth=2)

                # Add uncertainty blanket if std_reward is available
                if "std_reward" in data[0] and metric == "mean_reward":
                    std_values = [d["std_reward"] for d in data[:num_points]]
                    std_scaled = [std * 0.05 for std in std_values]

                    upper_bound = [val + std for val, std in zip(values, std_scaled)]
                    lower_bound = [val - std for val, std in zip(values, std_scaled)]

                    # Fill the uncertainty region
                    ax.fill_between(
                        current_steps,
                        lower_bound,
                        upper_bound,
                        alpha=0.3,
                        label=f"{metric} ± std_reward",
                    )

            # Add circle markers and annotations for points that should be annotated
            for i in points_to_annotate:
                if i < num_points:
                    ax.plot(
                        steps[i],
                        values[i],
                        "o",
                        markersize=8,
                        markerfacecolor="white",
                        markeredgecolor="red",
                        markeredgewidth=2,
                    )
                    ax.annotate(
                        f"{values[i]:.3f}",
                        (steps[i], values[i]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=9,
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7
                        ),
                    )

        ax.legend()

        # Add progress indicator
        # progress = (num_points / len(data)) * 100
        # ax.text(
        #     0.02,
        #     0.98,
        #     f"Progress: {progress:.1f}%",
        #     transform=ax.transAxes,
        #     fontsize=10,
        #     verticalalignment="top",
        #     bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
        # )

    # Create animation
    anim = FuncAnimation(
        fig,
        animate,
        frames=len(data),
        interval=animation_interval,
        repeat=repeat_animation,
        blit=False,
    )

    plt.tight_layout()
    plt.show()

    return anim


def plot_static(data, steps, points_to_annotate):
    """Create a static plot (original functionality)."""
    for metric in METRICS_TO_VISUALIZE:
        values = [global_scale(d[metric]) for d in data]

        # Plot the main line
        plt.plot(steps, values, label=metric, linewidth=2)

        # Add uncertainty blanket if std_reward is available
        if "std_reward" in data[0] and metric == "mean_reward":
            std_values = [d["std_reward"] for d in data]
            # Apply global scaling to std values as well (but treating them as deviations)
            std_scaled = [std * 0.15 for std in std_values]

            upper_bound = [val + std for val, std in zip(values, std_scaled)]
            lower_bound = [val - std for val, std in zip(values, std_scaled)]

            # Fill the uncertainty region
            plt.fill_between(
                steps,
                lower_bound,
                upper_bound,
                alpha=0.3,
                label=f"{metric} ± std_reward",
            )

        # Add circle markers and annotations at INTERMEDIATE_POINTS intervals
        for i in points_to_annotate:
            plt.plot(
                steps[i],
                values[i],
                "o",
                markersize=8,
                markerfacecolor="white",
                markeredgecolor="red",
                markeredgewidth=2,
            )
            plt.annotate(
                f"{values[i]:.3f}",
                (steps[i], values[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            )

        plt.xlabel("Experiences collected")
        plt.ylabel("Score")
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training logs with optional animation"
    )
    parser.add_argument("log_file", help="Path to the log file")
    parser.add_argument(
        "--static", action="store_true", help="Use static plot instead of animation"
    )
    parser.add_argument("--save-gif", type=str, help="Save animation as GIF file")
    parser.add_argument("--save-mp4", type=str, help="Save animation as MP4 file")
    parser.add_argument(
        "--interval",
        type=int,
        default=ANIMATION_INTERVAL,
        help=f"Animation interval in milliseconds (default: {ANIMATION_INTERVAL})",
    )
    parser.add_argument("--repeat", action="store_true", help="Loop the animation")

    args = parser.parse_args()
    log_file = args.log_file

    # Override global settings with command line arguments
    use_animation = not args.static
    animation_interval = args.interval
    repeat_animation = args.repeat

    try:
        # Load data
        print(f"Loading data from {log_file}...")
        data, steps, raw_count, points_to_annotate = load_log_data(log_file)

        if not data:
            print("No data found in log file!")
            sys.exit(1)

        print(f"Loaded {raw_count} original data points")
        print(
            f"Generated {len(data)} total data points (including {INTERMEDIATE_POINTS} intermediate points between each pair)"
        )
        print(f"Available metrics: {list(data[0].keys())}")
        print(f"Visualizing: {METRICS_TO_VISUALIZE}")

        if use_animation:
            print("Creating animated plot...")
            anim = plot_animated(
                data, steps, points_to_annotate, animation_interval, repeat_animation
            )

            # Save animation if requested
            if args.save_gif:
                print(f"Saving animation as GIF: {args.save_gif}")
                anim.save(args.save_gif, writer="pillow", fps=10)

            if args.save_mp4:
                print(f"Saving animation as MP4: {args.save_mp4}")
                anim.save(args.save_mp4, writer="ffmpeg", fps=10)
        else:
            print("Creating static plot...")
            plot_static(data, steps, points_to_annotate)

    except FileNotFoundError:
        print(f"Error: Could not find file {log_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
