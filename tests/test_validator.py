import unittest

from ml_model_serving.validator import Validator


class ValidatorTestCase(unittest.TestCase):
    def test_validator_validate_prediction_id_when_empty_then_false(self):
        #arrange
        expected_result = False
        validator = Validator()

        #act
        actual_result = validator.validate_prediction_id("")

        #assert
        self.assertEqual(expected_result, actual_result)

    def test_validator_validate_prediction_id_when_none_then_false(self):
        #arrange
        expected_result = False
        validator = Validator()

        #act
        actual_result = validator.validate_prediction_id(None)

        #assert
        self.assertEqual(expected_result, actual_result)

    def test_validator_validate_prediction_id_when_uuid_then_true(self):
        #arrange
        expected_result = True
        validator = Validator()

        #act
        actual_result = validator.validate_prediction_id("8b593281-01c1-4f23-8ffe-7589fdd45e63")

        #assert
        self.assertEqual(expected_result, actual_result)

    def test_validator_validate_image_base64_when_empty_then_false(self):
        #arrange
        expected_result = False
        validator = Validator()

        #act
        actual_result = validator.validate_image_base64("")

        #assert
        self.assertEqual(expected_result, actual_result)

    def test_validator_validate_image_base64_when_none_then_false(self):
        #arrange
        expected_result = False
        validator = Validator()

        #act
        actual_result = validator.validate_image_base64(None)

        #assert
        self.assertEqual(expected_result, actual_result)

    def test_validator_validate_image_base64_when_image_then_true(self):
        #arrange
        expected_result = True
        validator = Validator()

        #act
        actual_result = validator.validate_image_base64("image")

        #assert
        self.assertEqual(expected_result, actual_result)


if __name__ == '__main__':
    unittest.main()
