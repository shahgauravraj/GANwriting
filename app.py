from flask import Flask, jsonify, request, render_template, send_file
import utils


TEMPLATE_PATH = "path/to/html/here"
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
gen, device = utils.get_model()
gen.eval()


@app.route('/', methods = ['GET', 'POST'])
def root():
    """Handles requests from root url.
    On GET request serves index page template.
    On POST does the conversion of sent files.

    Returns:
        [type]: [description]
    """    
    if request.method == 'GET':
        render_template(TEMPLATE_PATH)
    else:
        return handle_post(request)


def handle_post(request):
    """Does the conversion of document to handwritten text.

    Args:
        request (HTTP.POST): The post request from the client containing handwritting samples and document as a string.

    Returns:
        (file) : The conversion output as a pdf.
    """        
    # Preprocessing the handwritting images
    imgs = request.files['imgs']
    preprocessed_imgs = utils.preprocess_images(imgs)

    # Preprocessing the text
    text = request.files['text']
    text, spaces, indents, imgs_per_line = utils.get_words(text)
    text_dataloader = utils.preprocess_text(text)

    # Converting to images
    imgs = utils.convert_to_images(text_dataloader, preprocessed_imgs, device)
    
    # TODO: write postprocess function
    ret = postprocess_images(imgs, spaces, indents, imgs_per_line)
    return ret