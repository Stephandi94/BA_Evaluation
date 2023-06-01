import os
import csv
import numpy as np
import evaluation_visualizer as ev


def plot_performance(files, title, file_title):
    rel_path = os.path.join(r"../Datasets/CARLA/performance")
    cwd = os.getcwd()
    dest = os.path.join(cwd, "Performance")
    data_list = []
    data_label = []
    output = []
    max_len = 0
    for f in files:
        data = np.loadtxt(os.path.join(rel_path, f))
        mean = np.mean(data).round(3)
        if data.size > max_len:
            max_len = data.size
        data_list.append(data)
        name, ext = os.path.splitext(f)
        data_label.append(name)
        print("{}\tmean calc. time:\t{} s [{} FPS]".format(name, mean, (1 / mean).round(3)))
        output.append([name, mean, (1 / mean).round(3)])
    # ev.line_plot(np.arange(0, max_len), data_list, data_label, title, "# simulation steps",
    #              "execution time [s]", os.path.join(dest, "{}.png".format(file_title)))

    vis = ev.VisEval()
    vis.set_x_data([np.arange(0, max_len)])
    vis.set_y_data(data_list)
    vis.set_data_labels(data_label)
    vis.set_x_label("simulation step")
    vis.set_y_label("execution time [s]")
    vis.to_file(dest, file_title)
    vis.set_legend_location('upper left')
    vis.set_xlim([0, max_len])
    vis.set_legend_frame(True)
    if len(data_list) > 3:
        vis.set_legend_col(3)
        vis.set_legend_location('upper right')
    vis.line_plot()

    with open(os.path.join(dest, "{}.csv".format(file_title)), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Sensor", "Mean Ex Time", "Mean FPS"])
        writer.writerows(output)


def main():
    compare_idle_sensors = ["Blickfeld.txt", "CARLA_LiDAR.txt", "Ibeo_LUX.txt", "SCALA.txt", "VLP16.txt", "VLP32.txt"]
    compare_VLP16 = ["VLP16.txt", "VLP16_rain.txt", "VLP16_snow.txt"]
    compare_VLP32 = ["VLP32.txt", "VLP32_road_spray.txt"]

    plot_performance(compare_idle_sensors, "Sensor Performance w/o any Effects", "performance_all_sensors")
    plot_performance(compare_VLP16, "VLP16 Performance with Environmental Effects", "performance_vlp16")
    plot_performance(compare_VLP32, "VLP32 Performance with Road Spray Effects", "performance_vlp32")


if __name__ == '__main__':
    main()
