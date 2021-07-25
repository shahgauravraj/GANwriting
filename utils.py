import wandb
import string
import cv2
import numpy as np
from models import GenModel_FC

# special tokens
START = 0
STOP = 1
PADDING = 2

# + 3 (because of the START, STOP and PADDING)
letter2index = {l : n + 3 for n, l in enumerate(string.ascii_letters)}
index2letter = {n + 3 : l for n, l in enumerate(string.ascii_letters)}


def get_model():
    """Downloads the model artifact from wandb and loads the weights from it into a new generator object.

    Returns:
        (torch.nn.moule): The pretrained generator model.
    """    
    api = wandb.Api()
    artifact = api.artifact('bijin/GANwriting_Reproducibilty_Challenge/GANwriting:v237', type='model')
    model_dir = artifact.download() + '/contran-5000.model'
    
    weights = torch.load(model_dir)
    gen = GenModel_FC(12)
    state_dict = gen.state_dict()

    for key in state_dict.keys():
        state_dict[key] = weights['gen.' + key]
    
    gen.load_state_dict(state_dict)
    return gen

def normalize(img):
    """Normalizes images to the range 0..255.

    Args:
        img (np.array): 3D array of floats.

    Returns:
        img (np.array): 3D array of 8-bit unsigned ints .
    """    
    img = (img - img.min()) / (img.max() - img.min())
    img *= 255
    img = np.uint8(img)
    return img

def convert_and_pad(word):
    """Converts the word to a list of tokens padded to read length 12.

    Args:
        word (string): A string of characters of max length 10.

    Returns:
        List[int]: A list of ints representing the tokens. 
    """    
    new_word = [letter2index[w] for w in word] # Converting each character to its token value 
    new_word = [START] + new_word + [STOP] # START + chars + STOP
    if len(new_word) < 12: # if too short, pad with PADDING token
        new_word.extend([PADDING] * (12 - len(new_word))) 
    return new_word

def preprocess_text(words, max_input_size=10):
    """Converts the each word into a list of tokens, bounded by start and end token. 
    Padding tokens added if necessary to reach max_input_size and splitting if the original word is too long.

    Args:
        words (List[str]): A batch of words as an array of strings.
        max_input_size (int): The max number of tokens in each input

    Returns:
        torch.tensor: A batch of words converted into a tensor.
    """       
    new_words = []
    for w in words:
        w_len = len(w)
        while (w_len > 0):
            new_words.append(convert_and_pad(w[:max_input_size]))
            w = w[max_input_size:]
            w_len -= max_input_size
        
    return torch.from_numpy(np.array(new_words))   

def preprocess_images(imgs):
    """Rescales, resizes and binarizes a batch of images of handwritten words and returns it as a tensor.
    If there are less than 50 images, the original list is shuffled and repeated until 50 is reached.

    Args:
        imgs (np.array[np.uint8]): Original batch of handwritten word image.

    Returns:
        torch.tensor: Preprocessed word image batch.
    """
    new_imgs = []
    for i in imgs:
        i = np.float32(i)
        i = i / 255.0 # Rescaling to 0..1
        i = cv2.resize(i, (216, 64), interpolation=cv2.INTER_CUBIC) # resizing the image for VGG
        i = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) # Grayscaling the image
        _, i = cv2.threshold(i, 0.5, 1, cv2.THRESH_OTSU) # thresholding with Otsu's method for binarization
        new_imgs.append(i)
    new_imgs = np.array(new_imgs)
    return torch.from_numpy(new_imgs)

def postprocess_images(imgs, doc):
    #TODO
    pass