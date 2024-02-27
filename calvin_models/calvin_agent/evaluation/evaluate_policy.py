import argparse
from collections import Counter, defaultdict
import logging
import os
from pathlib import Path
import sys
import time

# This is for using the locally installed repo clone when using slurm
from calvin_agent.models.calvin_base_model import CalvinBaseModel

sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from calvin_agent.evaluation.multistep_sequences import get_sequences
from calvin_agent.evaluation.utils import (
    collect_plan,
    count_success,
    create_tsne,
    get_default_model_and_env,
    get_env_state_for_initial_condition,
    get_log_dir,
    join_vis_lang,
    print_and_save,
)
from calvin_agent.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env

import yaml
from torchvision import transforms
from AVL_code.robot_transformer import RobotTransformer
import re
import mediapy as mediapy_lib
import cv2
import textwrap

def init_transform_dict(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225),
                        use_clip_norm=True):
    # Use normalization from: https://github.com/openai/CLIP/blob/a1d071733d7111c9c014f024669f959182114e33/clip/clip.py#L83
    if use_clip_norm:
        norm_mean = (0.48145466, 0.4578275, 0.40821073)
        norm_std = (0.26862954, 0.26130258, 0.27577711)
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(input_res, scale=randcrop_scale),
            transforms.Resize((input_res, input_res)),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            # transforms.Resize(center_crop),
            # transforms.CenterCrop(center_crop),
            # transforms.Resize(input_res),
            transforms.Resize((input_res, input_res)),
            normalize,
        ]),
        'test': transforms.Compose([
            # transforms.Resize(center_crop),
            # transforms.CenterCrop(center_crop),
            # transforms.Resize(input_res),
            transforms.Resize((input_res, input_res)),
            normalize,
        ])
    }
    return tsfm_dict

def get_transforms(split):
    if split in ['train', 'val', 'test']:
        return init_transform_dict()[split]
    else:
        raise ValueError('Split {} not supported.'.format(split))
    
logger = logging.getLogger(__name__)

EP_LEN = 360
NUM_SEQUENCES = 384#1000


def get_epoch(checkpoint):
    if "=" not in checkpoint.stem:
        return "0"
    checkpoint.stem.split("=")[1]


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


class CustomModel(CalvinBaseModel):
    def __init__(self):
        self.model = RobotTransformer()
        checkpoint_path = '/home/nikepupu/Downloads/checkpoint_90.pth'
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint["model"]
        self.model.load_state_dict(state_dict)
        self.model = self.model.cuda()   
        self.model.eval()

        self.transforms = self.get_transforms()
        self.calvin_bin_size = 20
        self.calvin_basedir = '/home/nikepupu/Downloads'
        with open(os.path.join(self.calvin_basedir,  'statistics.yaml'), 'r') as file:
            self.statistics = yaml.load(file, Loader=yaml.FullLoader)
        
        self.robot_obs_mean = self.statistics['robot_obs'][0]['mean']
        self.robot_obs_std = self.statistics['robot_obs'][0]['std']

        self.action_min = self.statistics['act_min_bound']
        self.action_max = self.statistics['act_max_bound']
        self.time_step = 0

        self.current_goal = None
       
        

    def get_transforms(self):
        return get_transforms("train")
    
    def get_action(self, obs):
        return self.model.get_action(obs)
    
    def get_bin_id(self, number, range_min, range_max, num_bins):
        """
        Function to find the bin ID for a given number in a specified range divided into bins.

        :param number: The number for which the bin ID is to be found.
        :param range_min: The minimum value of the range.
        :param range_max: The maximum value of the range.
        :param num_bins: The total number of bins in the range.
        :return: The bin ID in which the given number falls.
        """
        # Check if the number is within the range
        if number < range_min or number > range_max:
           # clip into the range
            number = min(max(number, range_min), range_max)

        # Calculate the width of each bin
        bin_width = (range_max - range_min) / num_bins

        # Calculate the bin ID
        bin_id = int((number - range_min) / bin_width)

        return bin_id
    
    def get_bin_mid(self, bin_id, range_min, range_max, num_bins):
        """
        Function to find the range of values for a given bin ID in a specified range divided into bins.

        :param bin_id: The ID of the bin for which the range is to be found.
        :param range_min: The minimum value of the range.
        :param range_max: The maximum value of the range.
        :param num_bins: The total number of bins in the range.
        :return: The mid point.
        """
        # Calculate the width of each bin
        bin_width = (range_max - range_min) / num_bins

        # Calculate the minimum and maximum values of the range for the given bin ID
        bin_range_min = range_min + bin_id * bin_width
        bin_range_max = bin_range_min + bin_width

        return (bin_range_min+bin_range_max)/2
        

    def reset(self):
        """
        This is called
        """
        self.model.total_embeddings = None
        self.model.length = []

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
       
        if self.time_step >= 4 or goal != self.current_goal:
            self.model.total_embeddings = None
            self.time_step = 0

        # if goal != self.current_goal:
        #     self.model.total_embeddings = None
        #     self.time_step = 0
        #     self.model.length = []
        
        # if self.time_step >= 4:
        #     self.model.remove_first_embeddings()

        self.time_step += 1
  
        rgb = obs['rgb_obs']['rgb_static']
        rgb = torch.tensor(rgb)
      
        rgb = rgb.float() / 255.0

        rgb = rgb.permute(2, 0, 1)

        # plt.imshow(rgb.cpu().numpy())
        # plt.show()
        rgb = self.transforms(rgb)
        rgb = rgb.unsqueeze(0)
        rgb = rgb.unsqueeze(0)
        rgb = rgb.to(torch.float32)


        # def show_image(image, title=''):
        #     import numpy as np
        #     # imagenet_mean = np.array([0.485, 0.456, 0.406])
        #     # imagenet_std = np.array([0.229, 0.224, 0.225])
        #     imagenet_mean = np.array([0.48145466, 0.4578275, 0.40821073 ])
        #     imagenet_std =  np.array([0.26862954, 0.26130258, 0.27577711])
        #     # image is [H, W, 3]
        #     assert image.shape[2] == 3
        #     plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
        #     plt.title(title, fontsize=8)
        #     plt.axis('off')
        #     return

        # show_image(rgb.permute(0, 1, 3, 4, 2)[0][0].cpu(), title='rgb')
        # plt.show()        

        # if self.model.total_embeddings is None:
        self.current_goal = goal
        instruction = goal
        # print('instruction: ', instruction)
        instruction = self.model.tokenizer(instruction, padding='longest', return_tensors="pt", 
                                    truncation=True, max_length=200,  add_special_tokens = False)
        instruction = instruction.input_ids.cuda()

        if self.model.total_embeddings is None:
            self.model.set_instructions(instruction)

        result = '[STARTSTATE]'
        for idx in range(14):
                tmp =  self.get_bin_id(obs['robot_obs'][idx], self.robot_obs_mean[idx] - 3 * self.robot_obs_std[idx], 
                                        self.robot_obs_mean[idx] + 3 * self.robot_obs_std[idx], self.calvin_bin_size)
                result += f'[ROBOTSTATE{idx}_{tmp}]'
        
        if obs['robot_obs'][14] == 1:
            result += '[GRIPPER_OPENED]'
        else:
            result += '[GRIPPER_CLOSED]'

        result += '[ENDOFSTATE]'
        stateobs = result
        rgb = rgb.cuda()
       
        stateobs = self.model.tokenizer(stateobs, padding='longest', return_tensors="pt", 
                                    truncation=True, max_length=200,  add_special_tokens = False).input_ids
        
        # with torch.no_grad():
        #     action = self.model.step2(rgb, instruction, effector_translation, effector_target_translation)
        # exit()
        stateobs = stateobs.cuda()
        
      
        with torch.no_grad():
            action = self.model.step(rgb, stateobs)
        
        action = action[0]
       

        
        # Regular expression to extract the X and Y dimensions
        dim0 = re.search(r'ROBOTACTION0_(\d+)', action)
        dim1 = re.search(r'ROBOTACTION1_(\d+)', action)
        dim2 = re.search(r'ROBOTACTION2_(\d+)', action)
        dim3 = re.search(r'ROBOTACTION3_(\d+)', action)
        dim4 = re.search(r'ROBOTACTION4_(\d+)', action)
        dim5 = re.search(r'ROBOTACTION5_(\d+)', action)
    

        
        # Extracted dimensions
        dimension0 = int(dim0.group(1)) if dim0 else None
        dimension1 = int(dim1.group(1)) if dim1 else None
        dimension2 = int(dim2.group(1)) if dim2 else None
        dimension3 = int(dim3.group(1)) if dim3 else None
        dimension4 = int(dim4.group(1)) if dim4 else None
        dimension5 = int(dim5.group(1)) if dim5 else None
     

        if None in [dimension0, dimension1, dimension2, dimension3, dimension4, dimension5]:
            return np.array([0.0, 0.0, 0., 0., 0., 0., 1])
        
        all_dimensions = [dimension0, dimension1, dimension2, dimension3, dimension4, dimension5]

        output_action = np.array([0., 0., 0., 0., 0., 0., 1])

        for i in range(6):
            output_action[i] = self.get_bin_mid(all_dimensions[i], -1,
                                                  1, self.calvin_bin_size)
        if '[GRIPPER_OPEN]' in action:
            output_action[6] = 1
        else:
            output_action[6] = -1


        

        return output_action


def evaluate_policy(model, env, epoch, eval_log_dir=None, debug=False, create_plan_tsne=False):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results = []
    plans = defaultdict(list)

    if not debug:
        eval_sequences = tqdm(eval_sequences, position=0, leave=True)
    workdir = '/home/nikepupu/Desktop/calvin_demo'
 
    video_dir = os.path.join(workdir, "videos")

    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    for initial_state, eval_sequence in eval_sequences:
        result = evaluate_sequence(env, model, task_oracle, initial_state, eval_sequence, val_annotations, plans, debug)
        results.append(result)
        if not debug:
            eval_sequences.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(count_success(results))]) + "|"
            )

    if create_plan_tsne:
        create_tsne(plans, eval_log_dir, epoch)
    print_and_save(results, eval_sequences, eval_log_dir, epoch)

    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence, val_annotations, plans, debug):
    """
    Evaluates a sequence of language instructions.
    """
    workdir = '/home/nikepupu/Desktop/calvin_demo'
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    
    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    
    for subtask in eval_sequence:
        # frames = []
        success = rollout(env, model, task_checker, subtask, val_annotations, plans, debug)
        if success:
            # if success: 
            #     import time
            #     time_str = time.time()
            #     video_path = os.path.join(workdir,f"{subtask}_{time_str}.mp4")
            #     mediapy_lib.write_video(video_path, frames, fps=10)
                # exit()
            success_counter += 1
        else:
            return success_counter
    return success_counter

def add_debug_info_to_image(image,
                            info_dict,
                            pos=(0, 0),
                            font=cv2.FONT_HERSHEY_DUPLEX,
                            font_scale=1,
                            font_thickness=1,
                            text_color=(0, 0, 0)):
  """Draw debugging text."""
  
  # Increase the image size so that debug text fits.
  image = cv2.resize(image, (640, 360))
  whitespace_per_line = 0.08
  if 'instruction' in info_dict:
    formatted_text = 'instruction: %s' % info_dict['instruction']
  else:
    formatted_text = ''
  wrapped_text = textwrap.wrap(formatted_text, width=35)
  whitespace_height = int(3 * int(image.shape[0] * whitespace_per_line))
  # Add whitespace below image.
  whitespace = np.ones([whitespace_height, image.shape[1], 3],
                       dtype=np.uint8) * 255
  image = np.concatenate([whitespace, image], 0)
  x, y = pos
  wrapped_text = textwrap.wrap(formatted_text, width=35)
  for _, line in enumerate(wrapped_text):
    text_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
    _, text_h = text_size
    cv2.putText(image, line, (x, y + text_h + font_scale - 1), font, font_scale,
                text_color, font_thickness)
    y += int(text_h * 1.2)
  return image

def rollout(env, model, task_oracle, subtask, val_annotations, plans, debug, frames=None):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    obs = env.get_obs()
    # get lang annotation for subtask
    lang_annotation = val_annotations[subtask][0]
    model.reset()
    start_info = env.get_info()

    for step in range(EP_LEN):
        action = model.step(obs, lang_annotation)
        obs, _, _, current_info = env.step(action)
        debug_info = lang_annotation + ' ' + str(np.around(action, decimals=4))
        info_dict = {"instruction": debug_info}
        
        # img = env.render(mode="rgb_array")
        # img = add_debug_info_to_image(img, info_dict)
        # frames.append(img)
        if debug:
            img = env.render(mode="rgb_array")
           
            join_vis_lang(img, lang_annotation)
            # time.sleep(0.1)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(model, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
    return False


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    # arguments for loading custom model or custom language embeddings
    parser.add_argument(
        "--custom_model", action="store_true", help="Use this option to evaluate a custom model architecture."
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    parser.add_argument("--eval_log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    # evaluate a custom model
    if args.custom_model:
        model = CustomModel()
        env = make_env(args.dataset_path)
        evaluate_policy(model, env, epoch=0, debug=args.debug)
    else:
        assert "train_folder" in args

        checkpoints = []
        if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
            print("Evaluating model with last checkpoint.")
            checkpoints = [get_last_checkpoint(Path(args.train_folder))]
        elif args.checkpoints is not None:
            print(f"Evaluating model with checkpoints {args.checkpoints}.")
            checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
        elif args.checkpoints is None and args.last_k_checkpoints is not None:
            print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
            checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints :]
        elif args.checkpoint is not None:
            checkpoints = [Path(args.checkpoint)]

        env = None
        for checkpoint in checkpoints:
            epoch = get_epoch(checkpoint)
            model, env, _ = get_default_model_and_env(
                args.train_folder,
                args.dataset_path,
                checkpoint,
                env=env,
                device_id=args.device,
            )
            evaluate_policy(model, env, epoch, eval_log_dir=args.eval_log_dir, debug=args.debug, create_plan_tsne=True)


if __name__ == "__main__":
    main()
