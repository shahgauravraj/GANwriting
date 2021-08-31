import wandb
import string
import cv2
import torch
import random
import docx2txt
import os
import shutil
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from models import GenModel_FC
from collections import defaultdict
from PIL import Image
from config import WAND_API_KEY


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
    os.environ["WANDB_API_KEY"] = WAND_API_KEY

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


def get_run_id():
    """Produces a random string of lowercase ascii characters of size 10 to serve as a run id. Also creates a dir of the same name in ./temp to store run artifacts temporarily. Ensures that the id is not currently in use by checking for existing dir of same name.

    Returns:
        id (string): Identifier string. 
    """    
    size = 10
    while True:
        id = "".join(random.choice(string.ascii_lowercase) for _ in range(size))
        if not os.path.isdir('./temp/' + id):
            os.mkdir('./temp/' + id)
            break
    return id

# TODO: TEST THIS WITH FRONTEND
def convert_files(id, imgs, text):
    """Converts the image files received through request into PIL Images and the text file into a string.

    Args:
        id (string): An identifier for the run, used here as the name for a temp directory.
        imgs (List[file]): A list of image files to convert.
        text (file): The document file to convert.

    Returns:
        new_imgs (List[Image]): A list of PIL Image objects, to be preprocessed.
        new_text (string): The text file converted to string form.
    """    
    new_imgs = []
    for i, img in enumerate(imgs):
        # Current directory to save images in.
        img_path = './temp/' + id + str(i) + '.jpg'
        img.save(img_path)
        new_img = im.open(img_path).convert('RGB')
        new_imgs.append(new_img)

    text_path = './temp/' + id + 'text.docx' 
    text.save(text_path)
    new_text = docx2txt.process(text_path)
    return new_imgs, new_text


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
        text (string): The document to convert in string form.
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


def resize_and_threshold(img, thresh, high):
    """Resizes the image to (216, 64) and does Otsu's thresholding on it.

    Args:
        img (np.array[np.uint8]): Image to be processed.
        mid (int|float): Initial threshold for Otsu's method.
        high (int|float): Max value of the image for Otsu's thresholding.

    Returns:
        img (np.array[np.uint8]): The processed image, pixels will be either 0 or high.
    """    
    if len(img.shape)==3 and img.shape[2]==3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Grayscaling the image
    img = cv2.resize(img, (216, 64), interpolation=cv2.INTER_CUBIC) # resizing the image for VGG
    _, img = cv2.threshold(img, thresh, high, cv2.THRESH_OTSU) # thresholding with Otsu's method for binarization
    return img


def preprocess_images(imgs):
    """Rescales, resizes and binarizes a batch of images of handwritten words and returns it as a tensor.
    If there are less than 50 images, the original list is shuffled and repeated until 50 is reached.

    Args:
        imgs (List[Image]): Original batch of handwritten word image.

    Returns:
        (torch.tensor): Preprocessed word image batch, pixels will be in range -1..1.
    """
    new_imgs = []
    for i in imgs:
        i = np.array(i)
        i = resize_and_threshold(i, 0.5, 1)
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
        img (np.array): 3D array of 8-bit unsigned ints, pixels will be in range 0..255.
    """    
    img = (img - img.min()) / (img.max() - img.min())
    img = 1 - img
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
        imgs (List[np.array]): A list of images as numpy arrays, pixels will be in range 0..255.
    """    
    with torch.no_grad():
        style = gen.enc_image(preprocessed_imgs.to(device))
        imgs = []
        for idx, word_batch in enumerate(text_dataloader):
            word_batch = word_batch[0].to(device)
            
            f_xt, f_embed = gen.enc_text(word_batch, style.shape)

            # the size we need the style tensor to be, 0th index is usually batch size but sometimes smaller, rest is unchanged
            size = [word_batch.shape[0]] + list(style.shape[1:]) 
            f_mix = gen.mix(style.expand(size), f_embed)
            
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

# TODO: TEST THIS TOO, IDK WHAT ABHISHEK DID.
def write2canvas(imgs,spaces,indents,imgs_per_line):
    """Takes in all the images generated and writes them all in blank canvases in their correct positions as in the document. 
    One canvas is equivalent to one page in the document. 
    The function returns a list of canvas whose size is equal to the no. of pages in the document.
    
    Args:
        imgs (np.array[np.uint8]): Images generated by the generator
        spaces (dict[set[int]]): Position of the spaces in each line.
        indents (dict[set[int]]): Position of the indents in each line.
        imgs_per_line (dict[set[int]]): Sum of images, spaces and indents in each line.
        
    Returns:
        np.array[np.uint8]: Array of images equivalent to the pages in the document.
    """
    w, h = 2500, 2700 # Setting page width
    data = np.zeros((h, w), dtype=np.uint8) # Creating np array of zeros of size h*w 
    data[0 : h, 0 : w] = 255 # Setting each value to RGB white value
    
    offset_w, offset_h = 20, 20 #Setting starting position of paste the images
    offset = 0, 0
    pages = math.ceil(len(imgs_per_line) / 30) # Finding no. of pages
    
    out = []
    img = iter(imgs) # Iterator for images of all the generated images of words
    for page in range(pages):
        line = 0
        canvas = im.fromarray(data) #Creating new PIL image canvas to overwrite the generated images on it
        while(offset_h < h):
            no_of_words = imgs_per_line[line]
            sdct = spaces[line] #Extracting the space set for the current line
            idct = indents[line] #Extracting the indent set for the current line

            for count in range(no_of_words): 
                if count in sdct: #Checking if space is required
                    offset_w = offset_w + 20 
                    continue

                if count in idct: #Checking if indent is required
                    offset_w = offset_w + 80
                    continue

                st = im.fromarray(next(img)) # Storing next image in a variable
                st = resize_and_threshold(st, 127, 255)
                st_w, st_h = st.size # Getting the image size
                
                offset = (offset_w, offset_h) # Set the pasting position for the new image
                canvas.paste(st, offset) # Overwrite the generated image over the canvas
                offset_w = offset_w + st_w # Update the offset width

            offset_h = offset_h + 90 # Update the offset width
            offset_w = 20
            line = line + 1 # Update the line no.
        # canvas.save('page'+str(page)+'.png')
        out.append(np.array(canvas)) # Append the canvas in a np array
    return out


def postprocess_images(imgs, spaces, indents, imgs_per_line):
    """Converts the list of np.array to a pdf file.

    Args:
        imgs (List[np.array]): An np.array of imgs of words in handwritten form. 
        spaces (dict[set[int]]): A dict of sets of ints containing positions of spaces in each line. 
        indents (dict[set[int]]): A dict of sets of ints containing the positions of indents in each ine.
        imgs_per_line (dict[int]): A dict of ints containing number of images to put in each line. 
    """
    # TODO
    pass


def cleanup_temp_files(id):
    pass