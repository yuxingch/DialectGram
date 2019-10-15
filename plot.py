import pandas as pd
import numpy as np
import ast
import re
import folium
import os
import matplotlib.colors as mcol
import matplotlib.cm as cm
# from geopy.geocoders import Nominatim
import reverse_geocoder
from collections import defaultdict
from tqdm import tqdm
import plotly.graph_objects as go
from us_state_abbrev import us_state_abbrev

THRESHOLD = 100

class ColorBar:

     def __init__(self, vmin = 0, vmax = 1, colors = ("black", "white")):
         self.vmin = vmin
         self.vmax = vmax
         self.colors = colors

     def to_color(self, num):
         cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", self.colors)

         cnorm = mcol.Normalize(vmin=self.vmin, vmax=self.vmax)

         cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
         cpick.set_array([])
         color_tp = cpick.to_rgba(num, bytes=True)[:3]
         return "#{0:02x}{1:02x}{2:02x}".format(*color_tp)

def get_state(x):
    state = reverse_geocoder.search(x)
    return [i["admin1"] for i in state]

def main():
    input_dir = "output_test"
    output_dir = "map_test"

    filenames = list(map(lambda x : x.replace(".csv", ""), os.listdir(input_dir)))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # geolocator = Nominatim()
    for ind, filename in enumerate(filenames):
        print(f"Processing {filename}, {ind}/{len(filenames)}")

        if not os.path.exists(f"{output_dir}/{filename}"):
            os.makedirs(f"{output_dir}/{filename}")
        try:
            dataset = pd.read_csv(f"{input_dir}/{filename}.csv")
            if len(dataset) <= 1:
                continue

            dataset["LON"], dataset["LAT"] = dataset["xy"].str.split(",").str
            del dataset['xy']
            del dataset["Unnamed: 0"]

            values = np.array([ast.literal_eval(re.sub(" +", ",", list_string)) for list_string in dataset["prob"].values])
            threshold = np.sort(np.mean(values, axis=0))[-2]
            meanings_mask = np.mean(values, axis=0) >= threshold
            masked_values = values[:, meanings_mask]
            masked_values = masked_values / np.sum(masked_values, axis=1, keepdims=True)
            print(f"Max = {np.max(masked_values, axis=0)}")
            print(f"Mean = {np.mean(masked_values, axis=0)}")
            print(f"Min = {np.min(masked_values, axis=0)}")

            world_map = folium.Map(location=[0,0], tiles="cartodbpositron", zoom_start=2)
            cb = ColorBar(vmin=np.min(masked_values, axis=0)[0], vmax=np.max(masked_values, axis=0)[0],
                          colors=("r", "b"))
            for (_, row), prob in zip(dataset.iterrows(), masked_values):
                c = cb.to_color(prob[0])
                folium.Circle(
                    location=(row["LAT"], row["LON"]),
                    popup=row["text"],
                    radius=10000,
                    color=c,
                    fill=True,
                    fill_color=c
                ).add_to(world_map)

            world_map.save(f'{output_dir}/{filename}/worldmap.html')
        except Exception as e:
            print("Fail.")
            print(e)

        usa_dataset = dataset[dataset["country"] == "usa"]
        # usa_dataset["Geo"] = usa_dataset["LAT"] + ", " + usa_dataset["LON"]
        stateValueMap = defaultdict(lambda : np.zeros((2)))
        stateCountMap = defaultdict(int)
        geo_list = [(row["LAT"], row["LON"]) for _, row in usa_dataset.iterrows()]
        for (_, row), state in zip(usa_dataset.iterrows(), get_state(geo_list)):
        # for _, row in tqdm(usa_dataset.iterrows(), total=len(usa_dataset)):
            # state = get_state(row.Geo)
            values = np.array([ast.literal_eval(re.sub(" +", ",", row.prob))])[:, :2]
            normalized_values = values / values.sum()
            stateValueMap[state] += normalized_values.reshape((-1,))
            stateCountMap[state] += 1
        state_keys = list(stateValueMap.keys())
        for k in state_keys:
            if stateCountMap[k] < THRESHOLD:
                del stateValueMap[k]
                continue
            stateValueMap[k] /= stateCountMap[k]

        keys = list(stateValueMap.keys())
        plotDf = pd.DataFrame.from_dict({
            "state" : keys,
            "value1" : [stateValueMap[k][0] for k in keys],
            "value2" : [stateValueMap[k][1] for k in keys],
        })
        plotDf.to_csv(f"{output_dir}/{filename}/heatmap.csv")

        plotDf['state'] = plotDf['state'].map(us_state_abbrev)
        plotDf = plotDf.dropna()
        fig = go.Figure(data=go.Choropleth(
            locations=plotDf['state'],
            z=plotDf['value1'],
            locationmode='USA-states',
            colorscale='Blues',
            autocolorscale=False,
            text=plotDf['state'],  # hover text
            marker_line_color='white',  # line markers between states
            colorbar_title="Usage Proportion",
        ))

        # fig.add_trace(go.Choropleth(
        #     locationmode='USA-states',
        #     locations=plotDf['state'],
        #     z=plotDf['state'],
        #     text=plotDf['state'],
        #     colorscale=[[0, 'rgb(0, 0, 0)'], [1, 'rgb(0, 0, 0)']],
        #     autocolorscale=False,
        #     showscale=False,
        # ))
        # fig.data[0].update(text=plotDf['state'],
        #                    mode='text',
        #                    textposition='bottom center')

        fig = fig.update_layout(
            title_text=f'Sense usage of {filename}',
            geo=dict(
                scope='usa',
                projection=go.layout.geo.Projection(type='albers usa'),
                showlakes=False,) # lakes
                # lakecolor='rgb(255, 255, 255)'),
        )

        fig.write_json(f"{output_dir}/{filename}/heatmap.json")
        fig.write_html(f"{output_dir}/{filename}/heatmap.html")
        # fig.show()

if __name__ == "__main__":
    main()
