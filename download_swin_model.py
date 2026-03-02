from tensorflow.keras.applications import ConvNeXtTiny

# Le reste du code est IDENTIQUE à EfficientNetV2
# Remplace juste :
base_model = ConvNeXtTiny(  # au lieu de EfficientNetV2B0
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)