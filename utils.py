import wandb
import string
import cv2
import torch
import random
import docx2txt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models import GenModel_FC
from collections import defaultdict
from PIL import Image

counter = 0

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
        gen (torch.nn.module): The pretrained generator model.
        device (string): The device the model is on (cuda/cpu).
    """    
    api = wandb.Api()
    artifact = api.artifact('bijin/GANwriting_Reproducibilty_Challenge/GANwriting:v237', type='model')
    model_dir = artifact.download() + '/contran-5000.model'
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weights = torch.load(model_dir, map_location=torch.device('cpu'))
    gen = GenModel_FC(12)
    state_dict = gen.state_dict()

    for key in state_dict.keys():
        state_dict[key] = weights['gen.' + key]
    
    gen.load_state_dict(state_dict)
    gen = gen.to(device)
    return gen, device


def strip(s):    
    return s.rstrip()


def update_dicts(w, words, d, imgs_per_line, idx):
    """Updates the given dict as well as imgs_per_line and words list.

    Args:
        w (string): Current word.
        words (List[string]): The list of found words.
        d (dict[set[int]]): The dict to be updated.
        imgs_per_line (dict[int]): A dict of ints to store number of images in each line.
        idx (int): An index to the above dicts and list.
    """    
    if len(w):
        words.append("".join(w))
        imgs_per_line[idx] += (len(w) - 1)//10 + 1

    d[idx].add(imgs_per_line[idx])
    imgs_per_line[idx] += 1


def get_words(text):
    """Converts a long string of text into constituent words, and produces a dict of indices to put the spaces and indents.
    Each word is counted as one or more images depending on its size.
    Each line is considered as an array of images. 
    The spaces and indents dicts have indices to where the spaces and indents would be in the array.
    Each space and indent counts as one image.

    Args:
        text (string): The document to convert in string form.

    Returns:
        words (List[List[string]]): A list of lists of words.
        spaces (dict[set[int]]): A dict of sets where each key in the dict refers to the line number and each item in the sets refer to indices of spaces. 
        indents (dict[set[int]]): A dict of sets where each key in the dict refers to the line number and each item in the sets refer to indices of indents. 
        imgs_per_line (dict[int]): A dict of the number of images in each line.
    """
    lines = list(map(strip, text.split("\n")))[::2]
    
    words = []
    spaces = defaultdict(set)
    indents = defaultdict(set)
    imgs_per_line = defaultdict(int)

    for i, line in enumerate(lines):
        w = []
        for c in line: 
            if c=='\t':
                update_dicts(w, words, indents, imgs_per_line, i)
                w = []
        
            elif c==' ':
                update_dicts(w, words, spaces, imgs_per_line, i)
                w = []

            else:
                w.append(c)
        
        if len(w):
            words.append("".join(w))
            imgs_per_line[i] += (len(w) - 1)//10 + 1
        

    return words, spaces, indents, imgs_per_line


def convert_and_pad(word):
    """Converts the word to a list of tokens padded to read length 12.

    Args:
        word (string): A string of characters of max length 10.

    Returns:
       new_word (List[int]): A list of ints representing the tokens. 
    """    
    new_word = []
    for w in word:
        if w in letter2index:
            new_word.append(letter2index[w]) # Converting each character to its token value, ignoring special non alphabetic characters
    new_word = [START] + new_word + [STOP] # START + chars + STOP
    if len(new_word) < 12: # if too short, pad with PADDING token
        new_word.extend([PADDING] * (12 - len(new_word))) 
    return new_word


def preprocess_text(text, max_input_size=10):
    """Converts the each word into a list of tokens, bounded by start and end token. 
    Padding tokens added if necessary to reach max_input_size and splitting if the original word is too long.

    Args:
        text (file): The document to convert in string form.
        max_input_size (int): The max number of tokens in each input

    Returns:
        (torch.data.utils.DataLoader): A dataloader to the dataset of words converted to tensors with batch size 8.
        spaces (dict[set[int]]): A dict of sets where each key in the dict refers to the line number and each item in the sets refer to indices of spaces. 
        indents (dict[set[int]]): A dict of sets where each key in the dict refers to the line number and each item in the sets refer to indices of indents. 
        imgs_per_line (dict[int]): A dict of the number of images in each line.
    """
    words, spaces, indents, imgs_per_line = get_words(text)
    new_words = []

    for w in words:
        w_len = len(w)
        while (w_len > 0):
            new_words.append(convert_and_pad(w[:max_input_size]))
            w = w[max_input_size:]
            w_len -= max_input_size
        
    new_words = torch.from_numpy(np.array(new_words))
    dataset = TensorDataset(new_words)

    return DataLoader(dataset, batch_size=8, shuffle=False), spaces, indents, imgs_per_line


def shuffle_and_repeat(imgs):
    """Takes the original list or images, shuffles them and if there are less than 50 images, repeats them until we get 50.

    Args:
        imgs (List[np.array]): A list of images as numpy arrays. 

    Returns:
        new_imgs (List[np.array]): A list of images as numpy arrays of size 50.
    """    
    new_imgs = []
    l = len(imgs)
    idx = l
    shuf = list(range(l))
    while len(new_imgs) < 50:
        if idx == l:
            random.shuffle(shuf)
            idx = 0
        new_imgs.append(imgs[shuf[idx]])
        idx += 1
    return new_imgs


def preprocess_images(imgs):
    """Rescales, resizes and binarizes a batch of images of handwritten words and returns it as a tensor.
    If there are less than 50 images, the original list is shuffled and repeated until 50 is reached.

    Args:
        imgs (List[Image]): Original batch of handwritten word image.

    Returns:
        (torch.tensor): Preprocessed word image batch.
    """
    new_imgs = []
    for i in imgs:
        i = np.array(i)
        i = cv2.resize(i, (216, 64), interpolation=cv2.INTER_CUBIC) # resizing the image for VGG
        i = cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) # Grayscaling the image
        _, i = cv2.threshold(i, 0.5, 1, cv2.THRESH_OTSU) # thresholding with Otsu's method for binarization
        i = np.float32(i)
        i = 1 - i
        i = (i - 0.5) / 0.5
        new_imgs.append(i)
    new_imgs = shuffle_and_repeat(new_imgs)
    new_imgs = np.array(new_imgs).reshape((1, 50, 64, 216))
    return torch.from_numpy(new_imgs)


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


def convert_to_images(gen, text_dataloader, preprocessed_imgs, device):
    """Converts the words from the document to handwritten word images in the style of preprocessed_images.

    Args:
        gen (torch.nn.module): The generator model.
        text_dataloader (torch.utils.data.DataLoader): DataLoader object for the words from the document. 
        preprocessed_imgs (torch.tensor): The handwritting images after preprocessing.
        device (string): The device on which to do the conversion(cuda/cpu).

    Returns:
        imgs (List[np.array]): A list of images as numpy arrays.
    """    
    with torch.no_grad():
        style = gen.enc_image(preprocessed_imgs.to(device))
        style = style.expand((8, 50, 64, 216))
        imgs = []
        for idx, word_batch in enumerate(text_dataloader):
            word_batch = word_batch[0].to(device)

            f_xt, f_embed = gen.enc_text(word_batch, style.shape)
            f_mix = gen.mix(style, f_embed)
            xg = gen.decode(f_mix, f_xt).cpu().detach().numpy()

            for x in xg:
                imgs.append(normalize(x.squeeze()))
    
    return imgs


def imgs_to_pdf(imgs):
    """Converts a list of images to pdf format.

    Args:
        imgs (List[np.array]): A list of images as numpy arrays.

    Returns:
        pdf_path (string): Path to the file where the pdf was saved. 
    """    
    new_imgs = []
    for i in imgs:
        new_imgs.append(Image.fromarray(i).convert('RGB')) # converting each array to PIL Image objects

    pdf_path = "need/to/put/something/here.pdf" # maybe use a random word generator?
    new_imgs[0].save(pdf_path, save_all=True, append_images=new_imgs[1:])
    return pdf_path


def postprocess_images(imgs, spaces, indents, imgs_per_line):
    """Converts the list of np.array to a pdf file.

    Args:
        imgs (List[np.array]): An np.array of imgs of words in handwritten form. 
        spaces (dict[set[int]]): A dict of sets of ints containing positions of spaces in each line. 
        indents (dict[set[int]]): A dict of sets of ints containing the positions of indents in each ine.
        imgs_per_line (dict[int]): A dict of ints containing number of images to put in each line. 

    Returns:
        (file): The handwritten document in pdf form.
    """
    # TODO
    return ret