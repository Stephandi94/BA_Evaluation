import os
import numpy as np
import evaluation_tools as et
import evaluation_visualizer as ev
import evaluation_data as ed

def main():
    cwd = os.getcwd()
    src = os.path.join(r"..\Datasets\CARLA\SensorScenePCLs\overlay")
    des_path = os.path.join(r"ConstructiveSensorEvaluation\SensorPCLComparison")
    files = os.listdir(src)
    files.sort()

    data_labels = ['Blickfeld', 'IbeoLux', 'SCALA', 'VLP16', 'VLP32']
    intensity_list = []
    distance_list = []
    index_considered_intensities = [0, 3, 4]

    for f in files:
        name, ext = os.path.splitext(f)
        pcl = et.pcl_to_numpy(os.path.join(src, f))
        intensities = pcl[:, 3]

        if os.path.exists(os.path.join(des_path, "{}_distances.csv".format(name))):
            distances = ed.read_csv_to_array(os.path.join(des_path, "{}_distances.csv".format(name))).reshape(-1)
        else:
            distances = et.get_distances(pcl)
            ed.nd_array_to_csv(distances.reshape(-1, 1), des_path, "{}_distances".format(name))
        intensity_list.append(intensities)
        distance_list.append(distances)

    vis = ev.VisEval()
    vis.set_figsize((7, 5))
    vis.set_y_data(distance_list)
    vis.set_data_labels(data_labels)
    vis.set_x_label("sensors")
    vis.set_y_label("distance [m]")
    vis.to_file(des_path, "eval_sensor_distances")
    vis.violin_plot()

    new_intensity_list = []
    new_label_list = ['Blickfeld', 'VLP16', 'VLP32']
    for i in index_considered_intensities:
        new_intensity_list.append(intensity_list[i])
    vis = ev.VisEval()
    vis.set_figsize((5, 5))
    vis.set_y_data(new_intensity_list)
    vis.set_data_labels(new_label_list)
    vis.set_x_label("sensors")
    vis.set_y_label("intensity")
    vis.to_file(des_path, "eval_sensor_intensities")
    vis.violin_plot()

if __name__=='__main__':
    main()