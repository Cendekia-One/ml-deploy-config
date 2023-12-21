from utils import util_category,util_summary

# For Text Summary with model text_summary.h5 in models file
def text_summary(text):
    return util_summary(model_name="text_summary", text=text)

# For Text category with model text_category.h5 in models file
def text_category(text):
    return util_category(model_name="SavedModel", text=text)
