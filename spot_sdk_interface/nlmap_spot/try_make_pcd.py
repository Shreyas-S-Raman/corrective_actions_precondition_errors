from actions import *
from spot_utils.generate_pointcloud import make_pointcloud

if __name__ == '__main__':
    # EXT_PROMPT = 'red cube on the table'

    # navigation_action_executor = NavigateAction("138.16.161.22")
    # grab_action_executor.nlmap_grab(TEXT_PROMPT)
    # navigation_action_executor = NavigateAction("138.16.161.21")
    # navigation_action_executor.update_images("cup", 7)

    make_pointcloud(data_path='data/default_data/', pose_data_fname='pose_data.pkl', pointcloud_fname='updated_pointcloud.pcd')
