import subprocess
import sys
import getopt
import os

from pathlib import Path

# -- add VirtualHome location to the current path for import:
sys.path.insert(0, "/media/sda/virtualhome/")
from simulation.unity_simulator import comm_unity

from utils import prepare_script

# -- number of frames/sec for rendering simulation execution:
fps = 25

# -- script_file_name :- name of the script with the entire program of agent-executable actions:
script_file_name = None

# -- output_frames_dir :- name of the directory where frames from execution will be outputted to:
output_frames_dir = None

scene_number = 0

class VirtualHomeSimulator():
    
    def __init__(self, version=1, scene=0):
        if version == 1:  
            self.path_to_exec = '/media/sda/virtualhome/simulation/unity_simulator/linux_exec.v2.2.4/linux_exec.v2.2.4.x86_64'
        else:
            self.path_to_exec = '/media/sda/virtualhome/simulation/unity_simulator/linux_exec.v2.3.0/linux_exec.v2.3.0.x86_64'

        # -- run a subprocess which will open the simulator executable for communication:
        self.process = subprocess.Popen([self.path_to_exec, '-screen-fullscreen', '0', '-screen-quality', '2'], stdout=subprocess.PIPE)
        
        # -- this variable corresponds to the communication (pipe?) to the simulator:
        self.comm = None 
        
        # -- there are 7 scenes available in VirtualHome; by default, we will use the first one:
        self.scene_number = scene
    
    def _start_simulator(self):
        # -- establish communication channel for simulator:
        #   NOTE: based on the tutorial from http://virtual-home.org/documentation/master/get_started/get_started.html#installation 
        self.comm = comm_unity.UnityCommunication(file_name=self.path_to_exec)

        # -- always reset the scene before execution (according to docs)
        self.comm.reset(self.scene_number)

        # -- we will randomly select an agent to put in the environment 
        #       (defined here: http://virtual-home.org/documentation/master/kb/agents.html)
        import random, time
        random.seed(time.process_time())

        vh_agents = ['Female1', 'Male1', 'Male2', 'Female2', 'Male6', 'Female4']
        random_agent = random.randint(1, len(vh_agents)) - 1
        
        # -- add the character to the scene:
        self.comm.add_character('Chars/' + vh_agents[random_agent])

    
    def _end_simulator(self):
        self.process.kill()


    def _get_environment_graph(self):
        # -- get the environment graph, which describes the current state of the scene:
        while True:
            # NOTE: taken from here: http://virtual-home.org/documentation/master/get_started/get_started.html#querying-the-scene
            is_success, graph = self.comm.environment_graph()
            if is_success == True:
                return graph

        
    def _run_script(self, script):
        global fps, output_frames_dir
        # -- render the scene and execute the sequence:
        #   NOTE: read more here - http://virtual-home.org/documentation/master/api/interaction.html#unity_simulator.UnityCommunication.render_script
        
        # -- delete the folder of frames if it already exists:
        os.system('rm -rf ' + str(output_frames_dir + '/script/0'))        
        
        while True:
            is_success, _ = self.comm.render_script(
                    script=script, 
                    recording=True, 
                    frame_rate=fps,
                    output_folder=output_frames_dir)
            if is_success == True:
                return
            else:
                self.comm.reset(self.scene_number)

        
#enddef

def frames_to_GIF():
    global output_frames_dir

    # -- list all the names of the images/frames generated from the simulation pipeline:
    filenames = sorted(os.listdir(output_frames_dir + '/script/0/'))

    import imageio.v3 as imageio

    gif_file = str(output_frames_dir + '/output.gif')
    print(gif_file)

    # -- delete previous copy of file if it exists already:
    if os.path.exists(gif_file):
        os.remove(gif_file)

    # -- use image processing library for saving frames as animated GIF:
    frames = []
    for filename in filenames:
        if not filename.endswith('.png'):
            continue
        data = imageio.imread(str(output_frames_dir + '/script/0/' + filename))
        frames.append(data)
    
    imageio.imwrite(gif_file, frames, extension='.gif', fps=5)
    print('\nGenerated GIF image saved as "' + gif_file + '"!')

#enddef


if __name__ == '__main__':
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "env:scr:", ['scene=', 'script='] )
              
    except ImportError:
        print('ERROR in loading "getopt" library!')
        sys.exit()

    else:   
        for opt, arg in opts:
            if opt in ['-env', '--scene']:
                scene_number = int(arg)

            elif opt in ['-scr', '--script']:
                # -- provide the name of the script containing the instructions to be executed:
                script_file_name = str(arg)

    # -- some script file names from the experiments folder were found to have the scene number in their name,
    #       so this is to check if the file names already indicate which scene number to load.
    if script_file_name and 'scene' in script_file_name:
        file_name_parts = script_file_name.split('-scene')
        scene_number = int(file_name_parts[1][0])

    vh_comm = VirtualHomeSimulator(scene=scene_number)
    vh_comm._start_simulator()

    # -- get the initial scene graph:
    initial_scene_graph = vh_comm._get_environment_graph()

    print(script_file_name)

    # -- parse the script file and adjust each object's id numbers:
    script_to_execute, objects_in_script = prepare_script(initial_scene_graph, script_file_name)

    if not script_to_execute:
        # -- exit if there is some issue with parsing:
        print('ERROR in forming script! Please check to see if it is executable.')
        sys.exit()

    output_frames_dir = str('./' + Path(script_file_name).stem + ('_scene' + str(scene_number)) if 'scene' not in script_file_name else '')

    final_scene_graph = vh_comm._get_environment_graph()

    vh_comm._run_script(script_to_execute)
    
    # -- convert the frames from the executed video into a .GIF file:
    frames_to_GIF()

    # -- close the simulator:
    vh_comm._end_simulator()

    
#endif

