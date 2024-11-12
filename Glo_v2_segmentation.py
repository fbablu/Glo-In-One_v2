import subprocess
import argparse
import os
# Define the commands with their arguments

def get_arguments():

    parser = argparse.ArgumentParser(description="Glo_v2")

    parser.add_argument("--dataset_dir", type=str, default='Test_Patch/')
    parser.add_argument("--data_list", type=str, default='Test_Patch/')



    parser.add_argument("--valset_dir", type=str,
                        default='Test_Patch/data_list.csv')

    parser.add_argument("--reload_path", type=str,
                        default='weights/Glo_v2_segmentation_model.pth')
    parser.add_argument("--output_folder", type=str, default='output')

    return parser

if __name__ == '__main__':

    parser = get_arguments()
    print(parser)
    args = parser.parse_args()


    save_csv= [
        "python", "MOTSDataset_2D_Patch_normal_save_csv_Glo_v2.py",
        "--dataset_dir",
        args.dataset_dir,
        "--data_list",
        args.data_list,

    ]

    testing = [
        "python", "testing_2D_patch_v2.py",
        "--valset_dir",
        os.path.join(args.dataset_dir, "data_list.csv"),
        "--reload_path",
        args.reload_path,
        "--output_folder",
        args.output_folder,
    ]

    # Run each command in sequence
    subprocess.run(save_csv)
    subprocess.run(testing)