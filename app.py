from flask import Flask, jsonify, request, render_template, send_file
import utils


TEMPLATE_PATH = "home.html"
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
        return render_template(TEMPLATE_PATH)
    else:
        # create a random run id      
        id = utils.get_random_run_id()
        try:
            path = handle_post(request, id)
        except:
            pass
        finally:
            # Delete all temp files.
            utils.cleanup_temp_files(id)
        return send_file(path, as_attachment=True)


def handle_post(request, id):
    """Does the conversion of document to handwritten text.

    Args:
        request (HTTP.POST): The post request from the client containing handwritting samples and document as a string.
        id (string): Id of the current run. Used as the directory name where temp files are stored.
        
    Returns:
        id (str) : The path to temp forlder where the pdf file is stored.
    """  

    # Take the received files and convert into required formats
    imgs = request.files['imgs']
    text = request.files['text']
    imgs, text = utils.convert_files(id, imgs, text)

    # Preprocessing the handwritting images
    preprocessed_imgs = utils.preprocess_images(imgs)

    # Preprocessing the text
    text_dataloader, spaces, indents, imgs_per_line = utils.preprocess_text(text)

    # Converting to images
    imgs = utils.convert_to_images(gen, text_dataloader, preprocessed_imgs, device)
    
    # TODO: write postprocess function
    ret = postprocess_images(imgs, spaces, indents, imgs_per_line)

    return id