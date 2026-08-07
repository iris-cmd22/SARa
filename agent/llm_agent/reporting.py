import matplotlib.pyplot as plt
from sarenv.utils import plot


def save_trajectory_plot(item, trajectory_segments, output_file):
    x_min, y_min, x_max, y_max = item.bounds
    plot.plot_heatmap(
        item=item,
        generated_paths=trajectory_segments,
        name="Traiettoria missione",
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        output_file=output_file,
    )
    print(f"Mappa della traiettoria salvata in: {output_file}")


def save_victims_plot(item, victims_gdf, victim_metrics, trajectory_segments, ipp_point, output_file):
    """Disegnata direttamente qui con matplotlib - non tocchiamo sarenv.utils.plot."""
    x_min, y_min, x_max, y_max = item.bounds
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(item.heatmap, extent=(x_min, x_max, y_min, y_max), origin="lower", cmap="YlOrRd", alpha=0.6)
    victims_gdf.plot(ax=ax, color="blue", markersize=8, alpha=0.6, label="Dispersi campionati")
    found_indices = list(victim_metrics["found_victim_indices"])
    if found_indices:
        victims_gdf.loc[found_indices].plot(ax=ax, color="lime", markersize=60, marker="*", label="Trovato")
    for segment in trajectory_segments:
        xs, ys = segment.xy
        ax.plot(xs, ys, color="black", linewidth=1.5)
    ax.plot(ipp_point.x, ipp_point.y, marker="^", color="red", markersize=12, label="IPP")
    ax.set_title("Dispersi campionati e traiettoria missione")
    ax.legend()
    fig.savefig(output_file)
    plt.close(fig)
    print(f"Mappa dei dispersi salvata in: {output_file}")
