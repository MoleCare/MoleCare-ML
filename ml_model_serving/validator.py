class Validator:
    def validate_prediction_id(self, prediction_id):
        if prediction_id is None or prediction_id == "":
            return False
        return True

    def validate_image_base64(self, image_base64):
        if image_base64 is None or image_base64 == "":
            return False
        return True

    def validate(self, prediction_id, image_base64):
        return self.validate_prediction_id(prediction_id) and self.validate_image_base64(image_base64)
