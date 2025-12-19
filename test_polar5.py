import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
import numpy as np
import plotly.express as px

#st.set_page_config(layout="wide")  # full-width mode
st.set_page_config()  # full-width mode

# ------------------------------
# HELPER FUNCTION
# ------------------------------
def smooth_for_polar(dfin, AngleStep=20, smooth=True, percentile=None):
    df_avg = dfin.copy()
    
    # Round TWA to nearest AngleStep
    df_avg["TWA_int"] = (df_avg["TWA"] / AngleStep).round() * AngleStep
    df_avg["TWA_int"] = df_avg["TWA_int"].astype(int)
    
    # Wind-speed bins
    bins = [0, 5, 10, 15, 20, 25, np.inf]
    labels = ["0-5 knt", "5-10 knt", "10-15 knt", "15-20 knt", "20-25 knt", ">25 knt"]
    df_avg["wind_bin"] = pd.cut(df_avg["wind_speed_knots"], bins=bins, labels=labels, right=False)

    # Speed column
    speed_col = "speed_knots_smooth" if smooth else "speed_knots"

    # Group and aggregate
    if percentile is None:
        df_grouped = df_avg.groupby(["TWA_int", "wind_bin"], as_index=False, observed=False)[speed_col].median()
    else:
        df_grouped = df_avg.groupby(["TWA_int", "wind_bin"], as_index=False, observed=False)[speed_col].apply(
            lambda x: np.percentile(x, percentile)
        ).reset_index()
        df_grouped = df_grouped.rename(columns={0: speed_col})

    # Cleanup
    df_grouped = df_grouped.dropna()
    df_grouped = df_grouped.rename(columns={"TWA_int": "TWA"})
    df_grouped["TWA_rad"] = np.radians(df_grouped["TWA"])

    return df_grouped


def plot_sail_with_datetime(dfin, edition, sail_no,leg):
    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------
    dfin["TWA_round"] = dfin["TWA"].round(1)

    # Ensure datetime column is datetime type
    dfin["datetime"] = pd.to_datetime(dfin["datetime"])

    # Filter sail
    df_sail = dfin.loc[dfin['edition']==edition]
    df_sail = df_sail.loc[df_sail['sail no.']==sail_no]
    df_sail = (
        df_sail[df_sail["leg"] == leg]
        .sort_values("datetime")
    )

    latest_time = df_sail["datetime"].max()

    if df_sail.empty:
        raise ValueError(f"Sail number '{sail_no}' not found.")

    boat_type = df_sail["boat type"].iloc[0]

    # --------------------------------------------------
    # Compute speed range for shaded region
    # --------------------------------------------------
    df_range = (
        dfin[dfin["boat type"] == boat_type]
        .groupby(["wind_bin", "TWA_round"], as_index=False)
        .agg(
            speed_min=("speed_knots_smooth", "min"),
            speed_max=("speed_knots_smooth", "max")
        )
    )

    # Merge onto sail track
    df_sail = df_sail.merge(
        df_range,
        on=["wind_bin", "TWA_round"],
        how="left"
    )

    df_sail = df_sail.loc[df_sail['datetime']<=latest_time]

    # --------------------------------------------------
    # Performance metric (% vs median range speed)
    # --------------------------------------------------
    avg_sail_speed = df_sail["speed_knots_smooth"].mean()

    range_mask = (
        (df["boat type"] == boat_type)
        & df["wind_bin"].isin(df_sail["wind_bin"])
        & df["TWA_round"].isin(df_sail["TWA_round"])
    )

    median_range_speed = df.loc[
        range_mask, "speed_knots_smooth"
    ].median()

    speed_pct = 100 * avg_sail_speed / median_range_speed

    # --------------------------------------------------
    # Plot
    # --------------------------------------------------
    fig = go.Figure()

    # Shaded region (min–max envelope)
    fig.add_trace(go.Scatter(
        x=pd.concat([df_sail["datetime"], df_sail["datetime"][::-1]]),
        y=pd.concat([df_sail["speed_max"], df_sail["speed_min"][::-1]]),
        fill="toself",
        fillcolor="rgba(80, 80, 80, 0.35)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name=f"{boat_type} speed range"
    ))

    # Sail speed line
    fig.add_trace(go.Scatter(
        x=df_sail["datetime"],
        y=df_sail["speed_knots_smooth"],
        mode="lines+markers",
        line=dict(width=3),
        name=f"Sail {sail_no}"
    ))

    fig.update_layout(
        width=300,        
        height=300,        
        title=(
            f"Speed profile<br>"
            f"<sup>Average speed = {speed_pct:.1f}% "
            f"of {boat_type} median polar</sup>"
        ),
        xaxis_title="Datetime",
        yaxis_title="Speed (knots)",
        template="plotly_white",
        xaxis=dict(
            showgrid=True,
            tickformat="%Y-%m-%d %H:%M"  # optional formatting
        )
    )

    return fig

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("processedAll_dfZero45.csv")

df = load_data()
df1=df.copy()
df2=df.copy()
df2['TWA']=df2['TWA']*-1

df = pd.concat([df1, df2], ignore_index=True)
col_logo, col_title, col_polar = st.columns([1, 6,1.5])

with col_logo:
    st.image("logo.png", width=110)
with col_polar:
    st.image("polar.png", width=110)

with col_title:
    st.title("Classe Mini Performance")
st.write("This dashboard visualizes performance data for minitransat boats. These data represent real-world race performance: they are extrapolated from publicly available geolocation data from minitransat races. Boat speed is computed based on location differences between each time-stamped position report while wind speed and direction are fetched from open-meteo archived data (https://open-meteo.com/). Since they are based on real-world data rather than on-the-water sailing tests, for some wind speeds and points of sail data is not available.")

# ------------------------------
# STANDARD PLOT SETTINGS
# ------------------------------
PLOT_STYLE = {
    "rows": 3,
    "cols": 2,
    "horizontal_spacing": 0.01,
    "vertical_spacing": 0.12,
    "subplot_titles_font_size": 12,
    "line_width_boat_type": 2,
    "line_width_sail": 3,
    "color_boat_type": "black",
    "color_sail": "blue",
    "radialaxis_range_pad": 2
}

wind_bins = ["0-5 knt", "5-10 knt", "10-15 knt", "15-20 knt", "20-25 knt", ">25 knt"]

# ------------------------------
# TABS
# ------------------------------
tab1, tab2 = st.tabs(["Compare Boat Types", "Single Boat"])

# ------------------------------
# TAB 1: Compare Boat Types
# ------------------------------
with tab1:
    st.subheader("Compare Boat Types")
    st.write("Select either SERIES or PROTO to compare polars estimated from all the boats available within those classes across all available races and legs.")

    kinds = df["kind"].dropna().unique()
    selected_kind = st.selectbox("Select kind", kinds, key="kind_tab1")
    smooth1 = st.checkbox("Use smoothed speed?", value=True, key="smooth_tab1")

    dfin = df[df["kind"] == selected_kind]

    st.dataframe(
        dfin.groupby(["boat type", "edition"])["boat_id"].nunique().reset_index(name="number of boats"),
        height=350, width=300
    )

    if st.button("Generate polar plots", key="plot_tab1"):
        boat_types = dfin["boat type"].unique()
        speed_col = "speed_knots_smooth" if smooth1 else "speed_knots"

        df_avg_dict = {bt: smooth_for_polar(dfin[dfin["boat type"]==bt], smooth=smooth1) for bt in boat_types}

        #colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        #          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                    "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#393b79"]
        color_map = {bt: colors[i % len(colors)] for i, bt in enumerate(boat_types)}

        fig = sp.make_subplots(
            rows=PLOT_STYLE["rows"], cols=PLOT_STYLE["cols"],
            specs=[[{"type":"polar"}]*PLOT_STYLE["cols"]]*PLOT_STYLE["rows"],
            subplot_titles=wind_bins,
            horizontal_spacing=PLOT_STYLE["horizontal_spacing"],
            vertical_spacing=PLOT_STYLE["vertical_spacing"]
        )

        for ann in fig['layout']['annotations']:
            ann['font'] = dict(size=PLOT_STYLE["subplot_titles_font_size"])
            ann['y'] += 0.04

        max_speed = 0
        for bt in boat_types:
            df_bt = df_avg_dict[bt]
            nboats = len(dfin.loc[dfin['boat type']==bt, 'boat_id'].unique())
            for i, wb in enumerate(wind_bins):
                sub = df_bt[df_bt["wind_bin"]==wb].sort_values("TWA")
                if sub.empty:
                    continue
                r = i//PLOT_STYLE["cols"] + 1
                c = i%PLOT_STYLE["cols"] + 1
                fig.add_trace(
                    go.Scatterpolar(
                        theta=sub["TWA"],
                        r=sub[speed_col],
                        mode="lines",
                        name=f"{bt} ({nboats})",
                        legendgroup=bt,
                        line=dict(color=color_map[bt], width=PLOT_STYLE["line_width_boat_type"]),
                        showlegend=(i==0)
                    ),
                    row=r, col=c
                )
                max_speed = max(max_speed, sub[speed_col].max())

        for i in range(1, len(wind_bins)+1):
            r = (i-1)//PLOT_STYLE["cols"] +1
            c = (i-1)%PLOT_STYLE["cols"] +1
            fig.update_polars(
                radialaxis=dict(range=[0, max_speed + PLOT_STYLE["radialaxis_range_pad"]]),
                angularaxis=dict(direction="clockwise", rotation=90),
                row=r, col=c
            )

        fig.update_layout(
            height=700, width=1300,
            title_text=f"Polar comparison for {selected_kind}. Median performance",
            showlegend=True,
            legend=dict(orientation="v", x=1.05, y=0.6, font=dict(size=10))
        )

        st.plotly_chart(fig, use_container_width=True)


        boat_types = dfin["boat type"].unique()
        speed_col = "speed_knots_smooth" if smooth1 else "speed_knots"

        df_avg_dict = {bt: smooth_for_polar(dfin[dfin["boat type"]==bt], smooth=smooth1,percentile=90) for bt in boat_types}

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        color_map = {bt: colors[i % len(colors)] for i, bt in enumerate(boat_types)}

        fig = sp.make_subplots(
            rows=PLOT_STYLE["rows"], cols=PLOT_STYLE["cols"],
            specs=[[{"type":"polar"}]*PLOT_STYLE["cols"]]*PLOT_STYLE["rows"],
            subplot_titles=wind_bins,
            horizontal_spacing=PLOT_STYLE["horizontal_spacing"],
            vertical_spacing=PLOT_STYLE["vertical_spacing"]
        )

        for ann in fig['layout']['annotations']:
            ann['font'] = dict(size=PLOT_STYLE["subplot_titles_font_size"])
            ann['y'] += 0.04

        max_speed = 0
        for bt in boat_types:
            df_bt = df_avg_dict[bt]
            nboats = len(dfin.loc[dfin['boat type']==bt, 'boat_id'].unique())
            for i, wb in enumerate(wind_bins):
                sub = df_bt[df_bt["wind_bin"]==wb].sort_values("TWA")
                if sub.empty:
                    continue
                r = i//PLOT_STYLE["cols"] + 1
                c = i%PLOT_STYLE["cols"] + 1
                fig.add_trace(
                    go.Scatterpolar(
                        theta=sub["TWA"],
                        r=sub[speed_col],
                        mode="lines",
                        name=f"{bt} ({nboats})",
                        legendgroup=bt,
                        line=dict(color=color_map[bt], width=PLOT_STYLE["line_width_boat_type"]),
                        showlegend=(i==0)
                    ),
                    row=r, col=c
                )
                max_speed = max(max_speed, sub[speed_col].max())

        for i in range(1, len(wind_bins)+1):
            r = (i-1)//PLOT_STYLE["cols"] +1
            c = (i-1)%PLOT_STYLE["cols"] +1
            fig.update_polars(
                radialaxis=dict(range=[0, max_speed + PLOT_STYLE["radialaxis_range_pad"]]),
                angularaxis=dict(direction="clockwise", rotation=90),
                row=r, col=c
            )

        fig.update_layout(
            height=700, width=1300,
            title_text=f"Polar comparison for {selected_kind}. Higher performance (90th percentile)",
            showlegend=True,
            legend=dict(orientation="v", x=1.05, y=0.6, font=dict(size=10))
        )

        st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# TAB 2: Single Sail vs Boat Type
# ------------------------------
with tab2:
    st.subheader("Single Boat")
    st.write("Select a boat to compare its performance against all other boats of its type. ")

    # --------------------------------------------------
    # Session state init
    # --------------------------------------------------
    if "show_polar_tab2" not in st.session_state:
        st.session_state.show_polar_tab2 = False

    # --------------------------------------------------
    # Controls
    # --------------------------------------------------
    sail_no_options = np.sort(df["sail no."].dropna().unique())
    selected_sail = st.selectbox("Select Sail Number", sail_no_options)

    smooth2 = st.checkbox("Use smoothed speed?", value=True)

    # --------------------------------------------------
    # Prepare data
    # --------------------------------------------------
    df_sail = df[df["sail no."] == selected_sail]
    if df_sail.empty:
        st.warning("Selected sail number not found.")
        st.stop()

    boat_type = df_sail["boat type"].iloc[0]
    df_boat_type = df[df["boat type"] == boat_type]

    df_avg_sail = smooth_for_polar(df_sail, smooth=smooth2)
    df_avg_boat_type = smooth_for_polar(df_boat_type, smooth=smooth2)
    df_avg_boat_typeHP = smooth_for_polar(df_boat_type, smooth=smooth2, percentile=90)

    speed_col = "speed_knots_smooth" if smooth2 else "speed_knots"

    # --------------------------------------------------
    # Generate polar plot button
    # --------------------------------------------------
    if st.button("Generate polars"):
        st.session_state.show_polar_tab2 = True

    # --------------------------------------------------
    # Polar plot (persistent)
    # --------------------------------------------------
    if st.session_state.show_polar_tab2:
        fig = sp.make_subplots(
            rows=PLOT_STYLE["rows"],
            cols=PLOT_STYLE["cols"],
            specs=[[{"type": "polar"}] * PLOT_STYLE["cols"]] * PLOT_STYLE["rows"],
            subplot_titles=wind_bins,
            horizontal_spacing=PLOT_STYLE["horizontal_spacing"],
            vertical_spacing=PLOT_STYLE["vertical_spacing"],
        )

        for ann in fig.layout.annotations:
            ann.font.size = PLOT_STYLE["subplot_titles_font_size"]
            ann.y += 0.04

        max_speed = 0

        for i, wb in enumerate(wind_bins):
            r = i // PLOT_STYLE["cols"] + 1
            c = i % PLOT_STYLE["cols"] + 1

            sub_sail = df_avg_sail[df_avg_sail["wind_bin"] == wb].sort_values("TWA")
            sub_type = df_avg_boat_type[df_avg_boat_type["wind_bin"] == wb].sort_values("TWA")
            sub_typeHP = df_avg_boat_typeHP[df_avg_boat_typeHP["wind_bin"] == wb].sort_values("TWA")

            if not sub_type.empty:
                fig.add_trace(
                    go.Scatterpolar(
                        theta=sub_type["TWA"],
                        r=sub_type[speed_col],
                        mode="lines",
                        name=f"{boat_type} (median)",
                        line=dict(
                            color=PLOT_STYLE["color_boat_type"],
                            width=PLOT_STYLE["line_width_boat_type"],
                        ),
                        showlegend=(i == 0),
                    ),
                    row=r,
                    col=c,
                )
                max_speed = max(max_speed, sub_type[speed_col].max())

            if not sub_typeHP.empty:
                fig.add_trace(
                    go.Scatterpolar(
                        theta=sub_typeHP["TWA"],
                        r=sub_typeHP[speed_col],
                        mode="lines",
                        name=f"{boat_type} (90th percentile)",
                        line=dict(
                            color=PLOT_STYLE["color_boat_type"],
                            width=PLOT_STYLE["line_width_boat_type"],
                            dash="dot",
                        ),
                        showlegend=(i == 0),
                    ),
                    row=r,
                    col=c,
                )

            if not sub_sail.empty:
                fig.add_trace(
                    go.Scatterpolar(
                        theta=sub_sail["TWA"],
                        r=sub_sail[speed_col],
                        mode="lines+markers",
                        name=f"Sail {selected_sail}",
                        line=dict(
                            color=PLOT_STYLE["color_sail"],
                            width=PLOT_STYLE["line_width_sail"],
                        ),
                        showlegend=(i == 0),
                    ),
                    row=r,
                    col=c,
                )
                max_speed = max(max_speed, sub_sail[speed_col].max())

        for i in range(len(wind_bins)):
            r = i // PLOT_STYLE["cols"] + 1
            c = i % PLOT_STYLE["cols"] + 1
            fig.update_polars(
                radialaxis=dict(range=[0, max_speed + PLOT_STYLE["radialaxis_range_pad"]]),
                angularaxis=dict(direction="clockwise", rotation=90),
                row=r,
                col=c,
            )

        fig.update_layout(
            height=700,
            width=1300,
            title=f"Polar comparison: Sail {selected_sail} vs {boat_type}",
            legend=dict(x=1.05, y=0.6, font=dict(size=10)),
        )

        st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # Route map (independent of polar plot)
    # --------------------------------------------------
    #st.divider()
    st.subheader("Route map")
    st.write("For the selected boat, visualize its route and its performance along the race compared with all theS other boats of its type.")


    edition_options = np.sort(df_sail["edition"].dropna().unique())
    selected_edition = st.selectbox("Select Edition", edition_options)

    leg_options = np.sort(df_sail["leg"].dropna().unique())
    selected_leg = st.selectbox("Select Leg", leg_options)

    if st.button("Generate route & speed profile"):
        df_route = df_sail[df_sail["edition"] == selected_edition]

        fig_map = px.scatter_geo(
            df_route,
            lat="latitude",
            lon="longitude",
            color="speed_knots",
            hover_data=["datetime", "speed_knots"],
            projection="natural earth",
            color_continuous_scale="Plasma",
        )

        fig_map.update_traces(marker=dict(size=6), showlegend=False)

        fig_map.update_geos(
            center=dict(
                lat=df_route["latitude"].mean(),
                lon=df_route["longitude"].mean(),
            ),
            projection_scale=2.5,
            showland=True,
            landcolor="lightgray",
            showcountries=True,
            lataxis=dict(showgrid=True),
            lonaxis=dict(showgrid=True),
        )

        fig_map.update_layout(
            height=450,
            title=f"Route — Sail {selected_sail}, Edition {selected_edition}",
            margin=dict(l=20, r=20, t=40, b=20),
        )

        st.plotly_chart(fig_map, use_container_width=True)

        fig_speed = plot_sail_with_datetime(df,selected_edition, selected_sail,selected_leg)
        st.plotly_chart(fig_speed, use_container_width=True)
