from AVL_code.robot_transformer import RobotTransformer
import torch
from torchvision import transforms
import re
import matplotlib.pyplot as plt

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
    
class AVL_Agent():
    def __init__(self):
        self.model = RobotTransformer()
        checkpoint_path = '/home/nikepupu/Downloads/checkpoint_178.pth'
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"]
        self.model.load_state_dict(state_dict)
        self.model = self.model.cuda()   
        self.model.eval()

        self.transforms = self.get_transforms()
        self.bin_size = 20


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
        self.model.total_embeddings = None
    
    def step(self, obs):
     
        rgb = torch.tensor(obs['rgb'])
      
        rgb = rgb.float() / 255.0

        rgb = rgb.permute(2, 0, 1)

        # plt.imshow(rgb.cpu().numpy())
        # plt.show()
        rgb = self.transforms(rgb)
        rgb = rgb.unsqueeze(0)
        rgb = rgb.unsqueeze(0)
        rgb = rgb.to(torch.float32)


        def show_image(image, title=''):
            import numpy as np
            # imagenet_mean = np.array([0.485, 0.456, 0.406])
            # imagenet_std = np.array([0.229, 0.224, 0.225])
            imagenet_mean = np.array([0.48145466, 0.4578275, 0.40821073 ])
            imagenet_std =  np.array([0.26862954, 0.26130258, 0.27577711])
            # image is [H, W, 3]
            assert image.shape[2] == 3
            plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
            plt.title(title, fontsize=8)
            plt.axis('off')
            return

        # show_image(rgb.permute(0, 1, 3, 4, 2)[0][0].cpu(), title='rgb')
        # plt.show()        

        # if self.model.total_embeddings is None:
        instruction = obs['instruction']
        instruction = ''.join(chr(id) for id in instruction if id != 0)
        # print('instruction: ', instruction)
        instruction = self.model.tokenizer(instruction, padding='longest', return_tensors="pt", 
                                    truncation=True, max_length=200,  add_special_tokens = False)
        instruction = instruction.input_ids.cuda()

        if self.model.total_embeddings is None:
            self.model.set_instructions(instruction)


        effector_translation = obs['effector_translation']
        effector_target_translation = obs['effector_target_translation']

        ee_t_first_dim =  int(self.get_bin_id(effector_translation[0], 0.15, 0.6, self.bin_size))
        ee_t_second_dim = int(self.get_bin_id(effector_translation[1], -0.3, 0.3, self.bin_size))
        effector_translation= f"[STARTEET][ROBOTEETX{ee_t_first_dim}][ROBOTEETY{ee_t_second_dim}][ENDOFEET]"

        effector_translation = self.model.tokenizer(effector_translation, padding='longest', 
                                                    return_tensors="pt", truncation=True, max_length=200,  add_special_tokens = False)
        effector_translation = effector_translation.input_ids

        ee_tt_first_dim =  int(self.get_bin_id(effector_target_translation[0], 0.15, 0.6, self.bin_size))
        ee_tt_second_dim = int(self.get_bin_id(effector_target_translation[1], -0.3, 0.3, self.bin_size))
        effector_target_translation = f"[STARTEETT][ROBOTEETTX{ee_tt_first_dim}][ROBOTEETTY{ee_tt_second_dim}][ENDOFEETT]"

        effector_target_translation = self.model.tokenizer(effector_target_translation, padding='longest', return_tensors="pt", truncation=True, 
                                                           max_length=200,  add_special_tokens = False)
        effector_target_translation = effector_target_translation.input_ids

        rgb = rgb.cuda()
        effector_translation = effector_translation.cuda()
        effector_target_translation = effector_target_translation.cuda()

        # with torch.no_grad():
        #     action = self.model.step2(rgb, instruction, effector_translation, effector_target_translation)
        # exit()
        with torch.no_grad():
            action = self.model.step(rgb, effector_translation, effector_target_translation)
            
        action = action[0]
        # print('action: ', action)

        if '[TERMINAL]' not in action:
            # Regular expression to extract the X and Y dimensions
            x_dim = re.search(r'ROBOTACTIONX(\d+)', action)
            y_dim = re.search(r'ROBOTACTIONY(\d+)', action)

            
            # Extracted dimensions
            x_dimension = int(x_dim.group(1)) if x_dim else None
            y_dimension = int(y_dim.group(1)) if y_dim else None
            if ( x_dimension is None) or ( y_dimension is None) :
                terminal = True
                x = 0
                y = 0
                return [x, y], terminal
            x = self.get_bin_mid(x_dimension, -0.03, 0.03, self.bin_size)
            y = self.get_bin_mid(y_dimension, -0.03, 0.03, self.bin_size)
            terminal = False
        else:
            terminal = True
            x = 0
            y = 0

        return [x, y], terminal


if __name__ == "__main__":
    agent = AVL_Agent()
    while True:
        continue
       