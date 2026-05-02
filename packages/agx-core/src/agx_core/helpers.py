import keras


def _channel_axis():
    return 1 if keras.config.image_data_format() == "channels_first" else -1


def _spatial_slice():
    return (
        slice(-2, None)
        if keras.config.image_data_format() == "channels_first"
        else slice(-3, -1)
    )


def _spatial_axis():
    return (
        [-2, -1] if keras.config.image_data_format() == "channels_first" else [-3, -2]
    )


def _layer_norm_axis():
    return [-2, -1] if keras.config.image_data_format() == "channels_first" else -1


__all__ = [
    "_channel_axis",
    "_spatial_slice",
    "_spatial_axis",
    "_layer_norm_axis",
]
