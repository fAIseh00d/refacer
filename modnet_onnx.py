import cv2
import numpy as np

class MODnet:
    def __init__(self, session):
        self.session = session
        self.model_input = self.session.get_inputs()[0].name
        self.model_output = self.session.get_outputs()[0].name
        self.downscale_en = False

    def _pre_process(self, image_array):
        if image_array.shape[0] > 512: 
            self.downscale_en = True
            image_array = cv2.resize(image_array, (512, 512))
        image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        image_array = image_array.astype('float32') / 255.0
        image_array = (image_array - 0.5) / 0.5
        image_array = np.expand_dims(image_array, axis=0).transpose(0, 3, 1, 2)
        return image_array

    def _post_process(self, result):
        result = np.clip(result, -1, 1)
        result = np.squeeze(result[0]) * 255
        if self.downscale_en:
            result = cv2.resize(np.array(result), self.orig_shape)
        return result.astype(np.uint8)
        
    def get(self, image_array):
        self.orig_shape = (image_array.shape[1],image_array.shape[0])
        image_array = self._pre_process(image_array)
        ort_inputs = {self.model_input: image_array}
        result = self.session.run([self.model_output], ort_inputs)
        result = self._post_process(result)
        return result