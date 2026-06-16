import plotly.graph_objects as go
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# base colors per model
MODEL_COLORS = {
    "pretrained"    : (0xA5/255, 0xD1/255, 0xD8/255),  # A5D1D8
    "no_interpol"   : (0x18/255, 0x97/255, 0xB2/255),  # 1897B2
    "scaled"        : (0x32/255, 0x6E/255, 0x75/255),  # 326E75
}

def make_color_fn(model: str, vmin: float, vmax: float, invert: bool = False):
    """
    Returns a color function that maps a value to an rgba string,
    varying only the alpha from 0.15 (low) to 1.0 (high).
    invert=True for EE (low value = more opaque = better)
    invert=False for correlation (high value = more opaque = better)
    """
    r, g, b = MODEL_COLORS.get(model, (0.5, 0.5, 0.5))

    def color_fn(val):
        if np.isnan(val):
            return "rgba(200,200,200,0.3)"
        t = (val - vmin) / (vmax - vmin)          # 0 → 1
        t = np.clip(t, 0, 1)
        if invert:
            t = 1 - t                              # low EE → high alpha
        alpha = 0.15 + 0.85 * t                   # range 0.15 – 1.0
        return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{alpha:.2f})"

    return color_fn


def plot_scatter_ecc_sector_plotly(
    df: pd.DataFrame,
    bins: list = [0, 3.375, 6.75, 10.125, 12.75],
    n_sectors: int = 4,
    subject: str = "",
    task: str = "",
    model: str = "",
    save_path: str = None,
) -> go.Figure:

    lim          = bins[-1]
    sector_width = 360 / n_sectors
    sector_angles = {
        "right": 0, "top-right": 45, "top": 90, "top-left": 135,
        "left": 180, "bottom-left": 225, "bottom": 270, "bottom-right": 315,
    }

    vmin, vmax = 0, 10
    to_color = make_color_fn(model, vmin=vmin, vmax=vmax, invert=False)
    ecc_labels = [l for l in df["ecc_bin"].unique() if l != "foveal"]
    r, g, b = MODEL_COLORS.get(model, (0.5, 0.5, 0.5))
    hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    fig = go.Figure()

    # outer rings × sectors
    for _, row in df[df["ecc_bin"] != "foveal"].iterrows():
        color   = to_color(row["mean_ee"])
        b_idx   = ecc_labels.index(row["ecc_bin"]) + 1
        r_inner = bins[b_idx]
        r_outer = bins[b_idx + 1]

        angle_center = sector_angles[row["sector"]]
        arc = np.linspace(np.radians(angle_center - sector_width/2),
                          np.radians(angle_center + sector_width/2), 30)
        outer_x = r_outer * np.cos(arc)
        outer_y = r_outer * np.sin(arc)
        inner_x = r_inner * np.cos(arc[::-1])
        inner_y = r_inner * np.sin(arc[::-1])
        wx = np.concatenate([outer_x, inner_x, [outer_x[0]]])
        wy = np.concatenate([outer_y, inner_y, [outer_y[0]]])

        ee_str = f"{row['mean_ee']:.2f}" if not np.isnan(row["mean_ee"]) else "n/a"
        fig.add_trace(go.Scatter(
            x=wx, y=wy, fill="toself", fillcolor=color,
            line=dict(color="white", width=1), mode="lines",
            hovertemplate=(
                f"<b>{row['ecc_bin']} / {row['sector']}</b><br>"
                f"mean EE: {ee_str} dva<br>"
                f"n = {int(row['n_samples'])}<extra></extra>"
            ),
            showlegend=False,
        ))

    # foveal disc
    foveal  = df[df["ecc_bin"] == "foveal"].iloc[0]
    color   = to_color(foveal["mean_ee"])
    theta   = np.linspace(0, 2*np.pi, 60)
    ee_str  = f"{foveal['mean_ee']:.2f}" if not np.isnan(foveal["mean_ee"]) else "n/a"
    fig.add_trace(go.Scatter(
        x=bins[1]*np.cos(theta), y=bins[1]*np.sin(theta),
        fill="toself", fillcolor=color,
        line=dict(color="white", width=1), mode="lines",
        hovertemplate=(
            f"<b>foveal</b><br>mean EE: {ee_str} dva<br>"
            f"n = {int(foveal['n_samples'])}<extra></extra>"
        ),
        showlegend=False,
    ))

    # ring borders
    for r in bins[1:-1]:
        theta = np.linspace(0, 2*np.pi, 120)
        fig.add_trace(go.Scatter(
            x=r*np.cos(theta), y=r*np.sin(theta),
            mode="lines", line=dict(color="white", width=1, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))

    # sector borders
    for i in range(n_sectors):
        angle = np.radians(i * sector_width - sector_width/2)
        fig.add_trace(go.Scatter(
            x=[bins[1]*np.cos(angle), lim*np.cos(angle)],
            y=[bins[1]*np.sin(angle), lim*np.sin(angle)],
            mode="lines", line=dict(color="white", width=1.2),
            hoverinfo="skip", showlegend=False,
        ))

    # # stimulus coverage square
    # fig.add_shape(
    #     type="rect", x0=-9, y0=-9, x1=9, y1=9,
    #     line=dict(color="white", width=1.5, dash="dash"),
    # )

    # colorbar — single-color alpha ramp
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(
            colorscale=[[0, f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.15)"],
                        [1, f"rgba({int(r*255)},{int(g*255)},{int(b*255)},1.0)"]],
            reversescale=True,           # low EE = full color
            cmin=vmin, cmax=vmax,
            color=[0],
            colorbar=dict(
                title=dict(text="mean EE (dva)", side="right"),
                thickness=15, len=0.75,
                tickvals=[0, 2, 4, 6, 8, 10],
            ),
            showscale=True, size=0,
        ),
        hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(
        template="simple_white",
        title=dict(text=f"{subject}  |  task: {task}  |  model: {model}", font=dict(size=13)),
        xaxis=dict(title="X (dva)", range=[-lim, lim], scaleanchor="y", zeroline=False),
        yaxis=dict(title="Y (dva)", range=[-lim, lim], zeroline=False),
        width=650, height=650,
        margin=dict(l=60, r=60, t=60, b=60),
    )

    if save_path:
        fig.write_image(save_path, scale=2)
    return fig


def plot_scatter_ecc_sector_corr_plotly(
    df: pd.DataFrame,
    bins: list = [0, 3.375, 6.75, 10.125, 12.75],
    n_sectors: int = 4,
    subject: str = "",
    task: str = "",
    model: str = "",
    save_path: str = None,
) -> go.Figure:

    lim          = bins[-1]
    sector_width = 360 / n_sectors
    sector_angles = {
        "right": 0, "top-right": 45, "top": 90, "top-left": 135,
        "left": 180, "bottom-left": 225, "bottom": 270, "bottom-right": 315,
    }

    vmin, vmax = -1, 1
    to_color   = make_color_fn(model, vmin=vmin, vmax=vmax, invert=False)
    ecc_labels = [l for l in df["ecc_bin"].unique() if l != "foveal"]
    r, g, b    = MODEL_COLORS.get(model, (0.5, 0.5, 0.5))

    fig = go.Figure()

    # outer rings × sectors
    for _, row in df[df["ecc_bin"] != "foveal"].iterrows():
        color   = to_color(row["corr"])
        b_idx   = ecc_labels.index(row["ecc_bin"]) + 1
        r_inner = bins[b_idx]
        r_outer = bins[b_idx + 1]

        angle_center = sector_angles[row["sector"]]
        arc     = np.linspace(np.radians(angle_center - sector_width / 2),
                              np.radians(angle_center + sector_width / 2), 30)
        outer_x = r_outer * np.cos(arc)
        outer_y = r_outer * np.sin(arc)
        inner_x = r_inner * np.cos(arc[::-1])
        inner_y = r_inner * np.sin(arc[::-1])
        wx = np.concatenate([outer_x, inner_x, [outer_x[0]]])
        wy = np.concatenate([outer_y, inner_y, [outer_y[0]]])

        r_str = f"{row['corr']:.3f}" if not np.isnan(row["corr"]) else "n/a"
        fig.add_trace(go.Scatter(
            x=wx, y=wy, fill="toself", fillcolor=color,
            line=dict(color="white", width=1), mode="lines",
            hovertemplate=(
                f"<b>{row['ecc_bin']} / {row['sector']}</b><br>"
                f"r = {r_str}<br>"
                f"n = {int(row['n_samples'])}<extra></extra>"
            ),
            showlegend=False,
        ))

    # foveal disc
    foveal = df[df["ecc_bin"] == "foveal"].iloc[0]
    color  = to_color(foveal["corr"])
    theta  = np.linspace(0, 2 * np.pi, 60)
    r_str  = f"{foveal['corr']:.3f}" if not np.isnan(foveal["corr"]) else "n/a"
    fig.add_trace(go.Scatter(
        x=bins[1] * np.cos(theta), y=bins[1] * np.sin(theta),
        fill="toself", fillcolor=color,
        line=dict(color="white", width=1), mode="lines",
        hovertemplate=(
            f"<b>foveal</b><br>r = {r_str}<br>"
            f"n = {int(foveal['n_samples'])}<extra></extra>"
        ),
        showlegend=False,
    ))

    # ring borders
    for r in bins[1:-1]:
        theta = np.linspace(0, 2*np.pi, 120)
        fig.add_trace(go.Scatter(
            x=r*np.cos(theta), y=r*np.sin(theta),
            mode="lines", line=dict(color="white", width=1, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))

    # sector borders
    for i in range(n_sectors):
        angle = np.radians(i * sector_width - sector_width/2)
        fig.add_trace(go.Scatter(
            x=[bins[1]*np.cos(angle), lim*np.cos(angle)],
            y=[bins[1]*np.sin(angle), lim*np.sin(angle)],
            mode="lines", line=dict(color="white", width=1.2),
            hoverinfo="skip", showlegend=False,
        ))


    # # stimulus coverage square
    # fig.add_shape(
    #     type="rect", x0=-9, y0=-9, x1=9, y1=9,
    #     line=dict(color="gray", width=1.5, dash="dash"),
    # )

    # colorbar
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(
            colorscale=[
                [0,   f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.15)"],
                [1,   f"rgba({int(r*255)},{int(g*255)},{int(b*255)},1.0)"],
            ],
            reversescale=False,
            cmin=vmin, cmax=vmax,
            color=[0],
            colorbar=dict(
                title=dict(text="Pearson r", side="right"),
                thickness=15, len=0.75,
                tickvals=[-1, -0.5, 0, 0.5, 1],
            ),
            showscale=True, size=0,
        ),
        hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(
        template="simple_white",
        title=dict(
            text=f"{subject}  |  task: {task}  |  model: {model}",
            font=dict(size=13)
        ),
        xaxis=dict(title="X (dva)", range=[-lim, lim], scaleanchor="y", zeroline=False),
        yaxis=dict(title="Y (dva)", range=[-lim, lim], zeroline=False),
        width=650, height=650,
        margin=dict(l=60, r=60, t=60, b=60),
    )

    if save_path:
        fig.write_image(save_path, scale=2, format='pdf')
    return fig