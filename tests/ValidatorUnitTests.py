import unittest

from ml_model_serving.Validator import Validator


class ValidatorTestCase(unittest.TestCase):
    def test_validator_validate_prediction_id_when_empty_then_false(self):
        #arrange
        expectedResult = False
        validator = Validator()

        #act
        actualResult = validator.validate_prediction_id("")

        #assert
        self.assertEqual(expectedResult, actualResult)

    def test_validator_validate_prediction_id_when_none_then_false(self):
        #arrange
        expectedResult = False
        validator = Validator()

        #act
        actualResult = validator.validate_prediction_id(None)

        #assert
        self.assertEqual(expectedResult, actualResult)

    def test_validator_validate_prediction_id_when_uuid_then_true(self):
        #arrange
        expectedResult = True
        validator = Validator()

        #act
        actualResult = validator.validate_prediction_id("8b593281-01c1-4f23-8ffe-7589fdd45e63")

        #assert
        self.assertEqual(expectedResult, actualResult)

    def test_validator_validate_image_base64_when_empty_then_false(self):
        #arrange
        expectedResult = False
        validator = Validator()

        #act
        actualResult = validator.validate_image_base64("")

        #assert
        self.assertEqual(expectedResult, actualResult)

    def test_validator_validate_image_base64_when_none_then_false(self):
        #arrange
        expectedResult = False
        validator = Validator()

        #act
        actualResult = validator.validate_image_base64(None)

        #assert
        self.assertEqual(expectedResult, actualResult)

    def test_validator_validate_image_base64_when_image_then_true(self):
        #arrange
        expectedResult = True
        validator = Validator()

        #act
        actualResult = validator.validate_image_base64("image")

        #assert
        self.assertEqual(expectedResult, actualResult)


if __name__ == '__main__':
    unittest.main()
